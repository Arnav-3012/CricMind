import numpy as np
import pandas as pd
import pytest
import torch

from cricmind import config
from cricmind.features.context import NUMERIC
from cricmind.models import no_identity_lgbm as nl
from cricmind.models import small_dims as sd
from cricmind.models import train as tr


def test_split_gap_parts_add_up_to_the_holdout_gap():
    s = nl.split_gap(mlp_with=1.4371, mlp_without=1.4389, tree_without=1.4350, tree_stats=1.4320)
    assert s["C_context_gap"] + s["I_identity_extraction_gap"] == pytest.approx(s["gap_on_holdout"])
    assert s["gap_on_holdout"] == pytest.approx(1.4371 - 1.4320)


def test_split_gap_names_the_larger_source_and_calls_a_near_equal_pair_a_tie():
    # C = 0.004, G = 0.003, E = 0.001 -> I = 0.002: context is larger by more than the tie margin
    assert nl.split_gap(1.440, 1.441, 1.437, 1.434)["larger_source"] == "context"
    # C = 0.001, G = 0.006, E = 0.001 -> I = 0.005
    assert nl.split_gap(1.437, 1.438, 1.437, 1.431)["larger_source"] == "identity_extraction"
    # C = 0.002, G = 0.0025, E = 0.0005 -> I = 0.002: equal, inside the 0.001 margin
    assert nl.split_gap(1.4375, 1.4380, 1.436, 1.4335)["larger_source"] == "tie"


def test_decision_requires_the_pre_registered_gain():
    results = {2: {"holdout_log_loss": 1.4360}, 4: {"holdout_log_loss": 1.4340}}
    helps = sd.decision(1.4371, results)
    assert helps["best_small_dim"] == 4 and helps["smaller_dim_helps"] is True
    flat = sd.decision(1.4350, results)  # gain 0.0010 < 0.002
    assert flat["best_small_dim"] == 4 and flat["smaller_dim_helps"] is False


def test_small_dim_losses_returns_one_finite_loss_and_a_valid_epoch_per_size(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_BALLS_EMBEDDING", 5)
    monkeypatch.setattr(tr, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(tr, "BATCH", 64)
    monkeypatch.setattr(tr, "MAX_EPOCHS", 3)

    def frame(seed):
        rng = np.random.default_rng(seed)
        d = pd.DataFrame({c: rng.normal(size=256) for c in NUMERIC})
        d["competition"] = "IPL"
        d["batter_id"] = rng.choice(["b1", "b2"], 256)
        d["bowler_id"] = rng.choice(["w1", "w2"], 256)
        d["venue_key"] = "v1"
        d["year"] = 2022
        d["y"] = rng.integers(0, len(config.OUTCOME_CLASSES), 256)
        return d

    out = sd.small_dim_losses(frame(0), frame(1), ["IPL"], (2, 4), seed=0)
    assert set(out) == {2, 4}
    assert all(np.isfinite(v["holdout_log_loss"]) and 1 <= v["epochs"] <= 3 for v in out.values())
