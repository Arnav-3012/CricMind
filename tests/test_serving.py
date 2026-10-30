"""D-058 serving artifact: the saved tables, loaded by `Serving`, give the numbers of the fitted models."""

import numpy as np
import pandas as pd
import pytest

from cricmind import config
from cricmind.agent.serving import Serving
from cricmind.features.context import MASKED, NUMERIC, PHASES
from cricmind.features.stats import apply_all, fit_all, oof_all
from cricmind.matchups.build_serving import build
from cricmind.matchups.chain import Chain, bat_keys, bowl_keys, cell_keys, pair_keys
from cricmind.models.baselines import GlobalFreq, Lgbm, PhaseFreq
from cricmind.models.p3_data import BASE_COLS

K = len(config.OUTCOME_CLASSES)
PLAYERS = list("abcde")
S = [50.0, 20.0, 30.0, 10.0]
TREES = 5
LABELS = {"batter": pd.Series({"a": 0, "b": 1, "c": 0}), "bowler": pd.Series({"a": 0, "b": 1})}
BASE = {"batter": LABELS["batter"].iloc[:2], "bowler": LABELS["bowler"]}  # 'c' is placed by nearest centre
ROLE_REPORT = {
    "cells": {
        role: {
            "k": 2,
            "stable": role == "batter",
            "chosen": {"boot_ari_mean": 0.6},
            "profiles": [{"cluster": c, "size": 1, "top": {"f": {"z": 0.5, "mean": 1.0}}} for c in (0, 1)],
        }
        for role in ("batter", "bowler")
    }
}


def make_frame(n: int, seed: int) -> pd.DataFrame:
    """A context frame with non-negative counts, real phases, and reduced rows masked as in D-030."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 50, n).astype("float64") for c in NUMERIC})
    df["innings"] = rng.integers(1, 3, n)
    df["over"] = rng.integers(0, 20, n)
    df["ball_in_over"] = rng.integers(1, 7, n).astype("float64")
    df["wickets_before"] = rng.integers(0, 10, n).astype("float64")
    df["phase"] = np.select([df["over"] < 6, df["over"] < 15], PHASES[:2], PHASES[2]).astype(object)
    df["competition"] = rng.choice(["A", "B"], n)
    df["batter_id"], df["bowler_id"] = rng.choice(PLAYERS, n), rng.choice(PLAYERS, n)
    df["venue_key"] = rng.choice(["v1", "v2"], n)
    df["is_reduced"] = rng.random(n) < 0.05
    df.loc[df["is_reduced"], MASKED] = np.nan
    df["y"] = rng.integers(0, K, n).astype("int8")
    df["match_id"] = rng.integers(0, 60, n).astype(str)
    df["year"] = 2020
    return df


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    train, val = make_frame(8000, 0), make_frame(2000, 1)
    out = tmp_path_factory.mktemp("serving")
    build(train, LABELS, BASE, S, TREES, ROLE_REPORT, out)
    return train, val, Serving(out)


def test_chain_proba_matches_the_fitted_chain(built):
    train, val, srv = built
    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    specs = [("cell", cell_keys(LABELS)), ("bat", bat_keys), ("bowl", bowl_keys), ("pair", pair_keys)]
    chain = Chain(specs, phase_freq)
    chain.s = S
    chain.refit(train, phase_freq)
    P, n = srv.chain_proba(val)
    assert np.allclose(P, chain.predict_proba(val), atol=1e-12)
    assert np.allclose(n["pair"], chain.levels[3].lookup(val).sum(axis=1))
    unlabelled = ~val["batter_id"].isin(LABELS["batter"].index).to_numpy()
    assert (n["cell"][unlabelled] == 0).all()  # an unlabelled batter skips the archetype level


def test_next_ball_proba_matches_the_fitted_model(built):
    train, val, srv = built
    features = srv.manifest["next_ball"]["features"]
    model = Lgbm(TREES, features).fit(pd.concat([train, oof_all(train)], axis=1))
    expected = model.predict_proba(pd.concat([val, apply_all(val, fit_all(train))], axis=1))
    assert np.allclose(srv.next_ball_proba(val), expected, atol=1e-9)
    assert srv.manifest["next_ball"]["features"][-len(BASE_COLS) :] == BASE_COLS


def test_state_frame_fills_and_lists_assumed_inputs(built):
    _, _, srv = built
    frame, info = srv.state_frame("a", "b", over=17, innings=1, wickets_before=3)
    assert "wickets_before" not in info["assumed"] and "score_before" in info["assumed"]
    assert "target" not in info["state"] and np.isnan(frame.loc[0, "req_rate"])
    assert frame.loc[0, "phase"] == "death" and frame.loc[0, "balls_bowled"] == 102
    assert frame.loc[0, "competition"] == srv.manifest["next_ball"]["default_competition"]
    assert np.isclose(srv.next_ball_proba(frame).sum(), 1.0)
    with pytest.raises(ValueError):
        srv.state_frame("a", "b", over=20)
    with pytest.raises(ValueError):
        srv.state_frame("a", "b", over=3, wickets_before=10)


def test_archetype_marks_placed_players_and_unlabelled_roles(built):
    _, _, srv = built
    assert srv.archetype("a")["batter"]["placed"] == "clustered"
    c = srv.archetype("c")
    assert c["batter"]["placed"] == "nearest_centre" and c["bowler"] is None
    assert c["batter"]["clustering_stable"] is True and c["batter"]["bin_traits"]["f"]["z"] == 0.5


def test_build_refuses_rows_after_train(tmp_path):
    late = make_frame(500, 2).assign(year=config.TRAIN_END_YEAR + 1)
    with pytest.raises(ValueError):
        build(late, LABELS, BASE, S, TREES, ROLE_REPORT, tmp_path)
