import numpy as np
import pandas as pd
import pytest

from cricmind.models import embedding_probe as ep


def test_player_stats_match_a_hand_count_and_mask_thin_phases(monkeypatch):
    monkeypatch.setattr(ep, "MIN_PHASE_BALLS", 4)
    # class ids: 0 dot, 1 single, 4 four, 5 six, 6 wicket. powerplay: dot, 1, four, W; death: one six
    f = pd.DataFrame(
        {
            "batter_id": ["p1"] * 5,
            "phase": ["powerplay"] * 4 + ["death"],
            "y": [0, 1, 4, 6, 5],
        }
    )
    s = ep.player_stats(f, "batter_id").loc["p1"]
    assert s["strike_rate_powerplay"] == pytest.approx((0 + 1 + 4 + 0) / 4 * 100)
    assert s["dot_pct_powerplay"] == pytest.approx(25.0)
    assert s["boundary_pct_powerplay"] == pytest.approx(25.0)
    assert s["wicket_rate_powerplay"] == pytest.approx(25.0)
    assert np.isnan(s["strike_rate_death"])  # one ball is below the floor


def test_reduced_rows_without_a_phase_are_dropped():
    f = pd.DataFrame({"batter_id": ["p1"] * 3, "phase": [None, None, "middle"], "y": [1, 1, 0]})
    assert list(ep.player_stats(f, "batter_id").columns) == [
        f"{s}_middle" for s in ("strike_rate", "dot_pct", "boundary_pct", "wicket_rate")
    ]


def test_probe_finds_a_planted_signal_and_nothing_in_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    planted = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=300)
    assert ep.probe_r2(X, planted) > 0.8
    assert ep.probe_r2(X, rng.normal(size=300)) < 0.1


def test_reliability_and_verdict_cut_offs():
    a = np.arange(50.0)
    assert ep.reliability(a, a + 0.0) == pytest.approx(1.0)
    assert ep.verdict(0.6) == "encodes_style" and ep.verdict(0.1) == "does_not"
    assert ep.verdict(0.3) == "partial_phase4_decides"
