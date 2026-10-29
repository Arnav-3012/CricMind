import numpy as np
import pandas as pd
import pytest
import torch

from cricmind import config
from cricmind.features.context import NUMERIC
from cricmind.models import sequence_test as st
from cricmind.models import train as tr


def _diff(value, hi):
    return {"value": value, "lo": value - 0.001, "hi": hi, "width": 0.002}


def test_verdict_needs_both_the_threshold_and_a_ci_below_zero():
    ok = {"a": 0.0002, "b": 0.0003}
    assert st.verdict(_diff(-0.003, -0.001), ok)["build_phase_3b"]
    assert not st.verdict(_diff(-0.0015, -0.001), ok)["build_phase_3b"]  # gain under the threshold
    assert not st.verdict(_diff(-0.003, 0.0005), ok)["build_phase_3b"]  # CI includes 0
    assert not st.verdict(_diff(0.001, 0.002), ok)["build_phase_3b"]  # sequence hurt


def test_verdict_flags_a_wide_seed_spread():
    assert not st.verdict(_diff(-0.003, -0.001), {"a": 0.002, "b": 0.0})["seed_spread_ok"]


@pytest.fixture
def _tiny(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_BALLS_EMBEDDING", 5)
    monkeypatch.setattr(tr, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(tr, "BATCH", 64)


def test_sequence_arm_trains_and_predicts_on_a_frame_with_the_recent_ball_columns(_tiny):
    rng = np.random.default_rng(0)
    n = 128
    d = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    d["competition"] = "IPL"
    d["batter_id"], d["bowler_id"], d["venue_key"] = "b1", "w1", "v1"
    d["year"], d["y"] = 2022, rng.integers(0, 7, n)
    d["runs_last12"], d["wkts_last12"] = rng.normal(size=n), rng.normal(size=n)
    d["last_outcome"] = rng.integers(0, 8, n)
    net, _ = tr.fit_net(d, ["IPL"], dim=4, seed=0, sequence=True, epochs=1)
    P = net.predict_proba(d)
    assert P.shape == (n, 7) and np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
