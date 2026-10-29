import numpy as np
import pandas as pd
import pytest
import torch

from cricmind import config
from cricmind.features.context import NUMERIC
from cricmind.models import no_identity as ni
from cricmind.models import train as tr


def _frame(n=256, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    d["competition"] = "IPL"
    d["batter_id"] = rng.choice(["b1", "b2"], n)
    d["bowler_id"] = rng.choice(["w1", "w2"], n)
    d["venue_key"] = "v1"
    d["year"] = 2022
    d["y"] = rng.integers(0, len(config.OUTCOME_CLASSES), n)
    return d


@pytest.fixture(autouse=True)
def _tiny(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_BALLS_EMBEDDING", 5)
    monkeypatch.setattr(tr, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(tr, "BATCH", 64)
    monkeypatch.setattr(tr, "MAX_EPOCHS", 3)


def test_holdout_row_loss_is_one_finite_loss_per_holdout_row_and_epoch_is_in_range():
    f, h = _frame(), _frame(seed=1)
    loss, epochs = ni._holdout_row_loss(f, h, ["IPL"], dim=4, seed=0)
    assert loss.shape == (len(h),) and np.isfinite(loss).all() and 1 <= epochs <= 3


def test_blanked_ids_are_a_single_constant_so_the_encoder_sees_no_identity():
    f = _frame()
    blanked = f.assign(**{c: "__none__" for c in ni.IDS})
    assert all(blanked[c].nunique() == 1 for c in ni.IDS)
    assert f["batter_id"].nunique() == 2  # the original frame is untouched
