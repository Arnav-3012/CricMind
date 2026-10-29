import numpy as np
import pandas as pd
import pytest
import torch

from cricmind import config
from cricmind.features.context import NUMERIC
from cricmind.models import train as tr
from cricmind.models.cric2vec import LITE_NUMERIC
from cricmind.models.outcome_compare import slices


def _frame(n=256, year=2022, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    d["competition"] = "IPL"
    d["batter_id"] = rng.choice(["b1", "b2"], n)
    d["bowler_id"] = rng.choice(["w1", "w2"], n)
    d["venue_key"] = "v1"
    d["year"] = year
    d["y"] = rng.integers(0, len(config.OUTCOME_CLASSES), n)
    return d


@pytest.fixture(autouse=True)
def _tiny(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_BALLS_EMBEDDING", 5)
    monkeypatch.setattr(tr, "DEVICE", torch.device("cpu"))  # MPS is not bit-reproducible
    monkeypatch.setattr(tr, "BATCH", 64)


def test_fit_net_returns_probabilities_and_the_same_seed_gives_the_same_model():
    f, h = _frame(), _frame(year=2022, seed=1)
    a, _ = tr.fit_net(f, ["IPL"], dim=4, seed=1, epochs=2)
    b, _ = tr.fit_net(f, ["IPL"], dim=4, seed=1, epochs=2)
    P = a.predict_proba(h)
    assert P.shape == (len(h), len(config.OUTCOME_CLASSES)) and np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(P, b.predict_proba(h), atol=1e-6)


def test_holdout_run_stops_early_with_a_curve_no_longer_than_max_epochs(monkeypatch):
    monkeypatch.setattr(tr, "MAX_EPOCHS", 6)
    monkeypatch.setattr(tr, "PATIENCE", 1)
    f, h = _frame(), _frame(seed=1)
    _, curve = tr.fit_net(f, ["IPL"], dim=4, seed=0, holdout=h)
    assert 2 <= len(curve) <= 6 and np.isfinite(curve).all()
    # stopped the moment the best epoch was `PATIENCE` epochs behind, or ran out of epochs
    assert len(curve) == 6 or len(curve) - 1 - int(np.argmin(curve)) >= 1


def test_extra_columns_widen_the_dense_input():
    f = _frame()
    f["bat_venue_n"] = 1.0
    plain, _ = tr.fit_net(f, ["IPL"], dim=4, seed=0, epochs=1)
    extra, _ = tr.fit_net(f, ["IPL"], dim=4, seed=0, epochs=1, extra=("bat_venue_n",))
    assert extra.encoder.width == plain.encoder.width + 1


def test_lite_net_reads_only_the_situation_inputs():
    f, h = _frame(), _frame(seed=1)
    lite_cols = ["batter_id", "bowler_id", "venue_key", "year", "y", *LITE_NUMERIC]
    net, _ = tr.fit_net(f[lite_cols], ["IPL"], dim=4, seed=0, epochs=1, lite=True)
    assert net.encoder.width == len(LITE_NUMERIC)
    P = net.predict_proba(h[lite_cols])  # no competition and no dropped column is needed
    assert P.shape == (len(h), len(config.OUTCOME_CLASSES)) and np.allclose(P.sum(axis=1), 1.0, atol=1e-5)


def test_tune_lite_picks_a_grid_dim_and_a_bounded_epoch_count(monkeypatch):
    monkeypatch.setattr(tr, "DIMS", (2, 4))
    monkeypatch.setattr(tr, "MAX_EPOCHS", 3)
    t = tr.tune_lite(_frame(), _frame(seed=1), ["IPL"])
    assert t["dim"] in (2, 4) and 1 <= t["epochs"] <= 3
    assert set(t["holdout_log_loss_by_dim"]) == {"2", "4"} and t["dim_at_grid_edge"] == (t["dim"] == 4)


def test_slices_follow_raw_ids_not_the_unk_floor():
    train = _frame()
    val = _frame(year=2022, seed=1)
    val["batter_id"] = ["b1"] * (len(val) - 1) + ["newbie"]
    val["phase"] = "middle"
    val["is_reduced"] = False
    val["match_id"] = "m"
    m = slices(val, train)
    assert m["unseen_batter"].sum() == 1 and not m["unseen_bowler"].any()
    assert m["unseen_pair"][-1] and m["phase_middle"].all() and not m["reduced"].any()
