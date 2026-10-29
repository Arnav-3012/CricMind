import json

import numpy as np
import pandas as pd
import pytest

from cricmind.features.context import PHASES
from cricmind.matchups import guard, headline
from cricmind.matchups.table import MatchupTable, extend_labels, fit_strength, nearest_cluster
from cricmind.models.baselines import GlobalFreq, PhaseFreq

LABELS = {
    "batter": pd.Series({"b0": 0, "b1": 1}),
    "bowler": pd.Series({"w0": 0, "w1": 1}),
}


def make_frame(n: int, seed: int, signal: bool) -> pd.DataFrame:
    """b2 and w2 have no label; ~5% of rows have no phase (a reduced match). With `signal`, the batter label
    changes the outcome distribution, so cells carry information beyond the phase."""
    rng = np.random.default_rng(seed)
    bat = rng.choice(["b0", "b1", "b2"], n)
    bowl = rng.choice(["w0", "w1", "w2"], n)
    phase = rng.choice(PHASES, n).astype(object)
    phase[rng.random(n) < 0.05] = None
    p_dot = np.where(signal & (bat == "b0"), 0.7, 0.3)
    y = np.where(rng.random(n) < p_dot, 0, rng.integers(1, 7, n))
    return pd.DataFrame(
        {
            "batter_id": bat,
            "bowler_id": bowl,
            "phase": phase,
            "y": y.astype("int8"),
            "match_id": rng.integers(0, 50, n),
        }
    )


@pytest.fixture
def fitted():
    train = make_frame(6000, 0, signal=True)
    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    return train, MatchupTable(LABELS).fit(train, phase_freq), phase_freq


def test_probabilities_sum_to_one(fitted):
    train, table, _ = fitted
    for s in (0.0, 5.0, 1e6):
        P = table.predict_proba(make_frame(500, 1, signal=True), s)
        assert np.allclose(P.sum(axis=1), 1.0)
        assert (P >= 0).all()


def test_s_zero_is_cell_rates_and_huge_s_is_the_phase_prior(fitted):
    train, table, phase_freq = fitted
    test = make_frame(500, 1, signal=True)
    served = table.served(test)
    P0 = table.predict_proba(test, 0.0)[served]
    cell = table.cells(test)[served]
    counts = table.counts[cell]
    assert np.allclose(P0, counts / counts.sum(axis=1, keepdims=True))
    assert np.allclose(table.predict_proba(test, 1e12), phase_freq.predict_proba(test), atol=1e-6)


def test_rows_without_a_label_or_phase_get_the_phase_baseline(fitted):
    _, table, phase_freq = fitted
    test = make_frame(800, 2, signal=True)
    unserved = ~table.served(test)
    assert unserved.any()
    rows = test[unserved]
    assert (rows["batter_id"].eq("b2") | rows["bowler_id"].eq("w2") | rows["phase"].isna()).all()
    assert np.allclose(table.predict_proba(test, 7.0)[unserved], phase_freq.predict_proba(test)[unserved])


def test_counts_use_only_labelled_phased_train_rows(fitted):
    train, table, _ = fitted
    assert table.counts.sum() == table.served(train).sum()


def test_to_frame_matches_predict_proba(fitted):
    _, table, _ = fitted
    test = make_frame(300, 3, signal=True)
    served = table.served(test)
    frame = table.to_frame(9.0)
    assert len(frame) == table.n_cells
    assert np.allclose(frame.filter(like="p_").sum(axis=1), 1.0)
    cell = table.cells(test)[served]
    assert np.allclose(frame.filter(like="p_").to_numpy()[cell], table.predict_proba(test, 9.0)[served])


def test_fit_strength_trusts_cells_more_when_they_carry_signal():
    flat_train, flat_val = make_frame(6000, 0, signal=False), make_frame(3000, 1, signal=False)
    sig_train, sig_val = make_frame(6000, 0, signal=True), make_frame(3000, 1, signal=True)
    s = {}
    for name, train, val in (("flat", flat_train, flat_val), ("signal", sig_train, sig_val)):
        table = MatchupTable(LABELS).fit(train, PhaseFreq(GlobalFreq()).fit(train))
        s[name] = fit_strength(table, val)["s"]
    assert s["flat"] > s["signal"]


def test_nearest_cluster_gives_each_player_the_closest_centre():
    from sklearn.cluster import KMeans

    centres = np.array([[0.0, 0.0], [10.0, 10.0]])
    X = np.vstack([centres[0] + 0.1, centres[1] - 0.1, [[1.0, 1.0]], [[9.0, 9.0]]])
    model = KMeans(2, init=centres, n_init=1).fit(np.vstack([centres, X]))
    ids = pd.Index(["a", "b", "c", "d"])
    got = nearest_cluster(model, X, ids)
    assert got["a"] == got["c"] != got["b"] == got["d"]


def test_extend_labels_keeps_phase4_labels_and_adds_only_players_over_the_floor(tmp_path, monkeypatch):
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from cricmind.matchups import table as table_mod

    rng = np.random.default_rng(0)
    # 4 players meet the 50-ball floor (a, b heavy, c, d light); "e" has 10 balls and stays unlabelled
    balls = {"a": 400, "b": 400, "c": 60, "d": 60, "e": 10}
    train = pd.DataFrame({"batter_id": np.repeat(list(balls), list(balls.values()))})
    train["bowler_id"] = train["batter_id"]
    (tmp_path / "p4").mkdir()
    for role in ("batter", "bowler"):
        pipe = make_pipeline(StandardScaler(), KMeans(2, n_init=1, random_state=0))
        pipe.fit(rng.normal(size=(40, 3)) + np.repeat([[0, 0, 0], [8, 8, 8]], 20, axis=0))
        joblib.dump(pipe, tmp_path / "p4" / f"kmeans_stats_{role}.joblib")
    monkeypatch.setattr(table_mod.config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        table_mod,
        "stats_vectors",
        lambda frame, role, ids: pd.DataFrame(np.tile([[0.0, 0.0, 0.0]], (len(ids), 1)), index=ids),
    )
    base = {"batter": pd.Series({"a": 1, "b": 1}), "bowler": pd.Series({"a": 1, "b": 1})}
    labels, info = extend_labels("stats", train, base)
    for role in ("batter", "bowler"):
        assert set(labels[role].index) == {"a", "b", "c", "d"}
        assert (labels[role][["a", "b"]] == 1).all()  # the Phase 4 labels win
        assert info[role]["players"] == 4 and info[role]["base_players"] == 2


def test_lock_refuses_a_second_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "lock_path", lambda: tmp_path / "test_lock.json")
    guard.claim_test("tables")
    guard.claim_test("cric2vec")  # a different scorer has its own claim
    with pytest.raises(RuntimeError, match="already read"):
        guard.claim_test("tables")
    assert set(json.loads((tmp_path / "test_lock.json").read_text())) == {"tables", "cric2vec"}
    with pytest.raises(ValueError):
        guard.claim_test("val")


def test_model_probabilities_from_other_test_rows_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(headline, "report_dir", lambda: tmp_path)
    monkeypatch.setattr(headline, "artifact_dir", lambda: tmp_path)
    (tmp_path / "cric2vec_test.json").write_text(json.dumps({"test_key": 1, "test_rows": 4}))
    np.save(tmp_path / "test_P_cric2vec.npy", np.full((4, 7), 1 / 7))
    assert headline.load_model_probs("cric2vec", 1, 4).shape == (4, 7)
    with pytest.raises(ValueError, match="different test rows"):
        headline.load_model_probs("cric2vec", 2, 4)


def test_verdict_needs_the_ci_to_exclude_zero():
    assert headline.verdict({"lo": -0.004, "hi": -0.001}) == "a_better"
    assert headline.verdict({"lo": 0.001, "hi": 0.004}) == "a_worse"
    assert headline.verdict({"lo": -0.001, "hi": 0.002}) == "no_difference"
