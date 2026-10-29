import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from cricmind import config
from cricmind.features.context import FEATURES, SEQUENCE
from cricmind.models.cric2vec import LITE_NUMERIC, Cric2Vec, Encoder


def _frame(n=64, year=2022, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES if c not in ("phase", "competition")})
    d["competition"] = rng.choice(["IPL", "__other__"], n)
    d["batter_id"] = rng.choice(["b1", "b2", "rare_b"], n, p=[0.5, 0.45, 0.05])
    d["bowler_id"] = rng.choice(["w1", "w2"], n)
    d["venue_key"] = "v1"
    d["year"] = year
    d["y"] = rng.integers(0, len(config.OUTCOME_CLASSES), n)
    d["runs_last12"], d["wkts_last12"] = rng.normal(size=n), rng.normal(size=n)
    d["last_outcome"] = rng.integers(0, 8, n)
    return d


@pytest.fixture(autouse=True)
def _tiny_floor(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_BALLS_EMBEDDING", 5)  # the test frames have 64 rows


def _fit(frame, sequence=False):
    return Encoder(sequence).fit(frame, ["IPL"])


def test_encoder_shapes_and_unk_for_rare_and_unseen_ids():
    f = _frame()
    enc = _fit(f)
    out = enc.transform(f)
    assert out["x"].shape == (len(f), enc.width) and out["x"].dtype == np.float32
    assert (out["bat"][f["batter_id"] == "rare_b"] == 0).all()  # below the floor -> UNK
    new = f.head(3).assign(batter_id="debutant", venue_key="elsewhere")
    t = enc.transform(new)
    assert (t["bat"] == 0).all() and (t["venue"] == 0).all()  # venue floor 1,000 > 64 rows: all UNK


def test_masked_numerics_are_filled_with_the_train_mean_and_phase_is_not_an_input():
    f = _frame()
    f.loc[:9, "target"] = np.nan
    enc = _fit(f)
    assert "phase" not in enc.numeric
    x = enc.transform(f)["x"]
    j = enc.numeric.index("target")
    assert (x[:10, j] == 0).all()  # standardised mean
    assert abs(x[10:, j].mean()) < 1e-5  # the other rows are standardised with the same train mean


def test_encoder_fits_on_train_rows_only():
    with pytest.raises(ValueError, match="train rows only"):
        _fit(_frame(year=config.VAL_YEAR))


def test_sequence_arm_adds_columns_and_base_arm_ignores_them():
    f = _frame()
    base, seq = _fit(f), _fit(f, sequence=True)
    assert seq.width - base.width == len(SEQUENCE) - 1 + 8  # two numerics + 8-way last-outcome one-hot
    assert np.array_equal(base.transform(f.drop(columns=SEQUENCE))["x"].shape, (len(f), base.width))


def test_epoch_zero_equals_the_prior():
    f = _frame()
    enc = _fit(f)
    prior = np.bincount(f["y"], minlength=7) / len(f)
    model = Cric2Vec(enc, dim=4, prior=prior).eval()
    b = {k: torch.as_tensor(v) for k, v in enc.transform(f).items()}
    logits = model(b["bat"], b["bowl"], b["venue"], b["x"])
    assert logits.shape == (len(f), 7)
    # bias = log prior, so the softmax stays near the prior (not exactly: random weights)
    assert torch.softmax(logits, 1).mean(0).sub(torch.as_tensor(prior)).abs().max() < 0.1


def test_overfits_one_batch():
    torch.manual_seed(config.SEED)
    f = _frame(n=32)
    enc = _fit(f)
    b = {k: torch.as_tensor(v) for k, v in enc.transform(f).items()}
    y = torch.as_tensor(f["y"].to_numpy(dtype="int64"))
    model = Cric2Vec(enc, dim=8, prior=np.full(7, 1 / 7)).eval()  # eval: no dropout noise
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(model(b["bat"], b["bowl"], b["venue"], b["x"]), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05  # random labels: only memorisation gets here, so the wiring can learn


def test_embeddings_have_an_unk_row_per_vocab():
    f = _frame()
    enc = _fit(f)
    emb = Cric2Vec(enc, dim=4, prior=np.full(7, 1 / 7)).embeddings()
    assert emb["batter"].shape == (len(enc.bat_vocab) + 1, 4)
    assert emb["bowler"].shape == (len(enc.bowl_vocab) + 1, 4)


def test_lite_encoder_uses_only_the_situation_inputs_and_no_competition_one_hot():
    f = _frame()
    enc = Encoder(numeric=LITE_NUMERIC, competition=False).fit(f, ["IPL"])
    assert enc.numeric == list(LITE_NUMERIC) and enc.width == len(LITE_NUMERIC)
    # the dropped columns (and the competition) are not read at all
    lite_only = f.drop(columns=[c for c in f if c in FEATURES and c not in LITE_NUMERIC])
    x = enc.transform(lite_only)["x"]
    assert x.shape == (len(f), 3) and abs(x.mean(axis=0)).max() < 1e-5
