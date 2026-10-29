import joblib
import numpy as np
import pandas as pd
import pytest

from cricmind import config
from cricmind.archetypes import cluster, stability
from cricmind.archetypes import vectors as vec
from cricmind.archetypes.profiles import profile
from cricmind.features.context import PHASES

K = len(config.OUTCOME_CLASSES)


def _blobs(per=40, k=3, dim=4, spread=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(loc=8 * i, scale=spread, size=(per, dim)) for i in range(k)])


@pytest.fixture
def small_search(monkeypatch):
    """Fewer ks and seeds so the stability code runs in seconds; the logic is unchanged."""
    for mod in (stability, cluster):
        monkeypatch.setattr(mod, "KS", range(3, 5))
    monkeypatch.setattr(stability, "SEED_ARI_SEEDS", range(3))


def test_resample_draws_whole_matches_with_replacement_and_is_seeded():
    f = pd.DataFrame({"match_id": np.repeat(np.arange(30), 2), "v": np.arange(60)})
    r = vec.resample_matches(f, 0)
    assert r["match_id"].isin(f["match_id"]).all()
    assert (r.groupby("match_id").size() % 2 == 0).all()  # a match is in whole or not at all
    assert r["match_id"].nunique() < 30  # 30 draws of 30 matches repeat some and miss others
    pd.testing.assert_frame_equal(r, vec.resample_matches(f, 0))
    assert not r.equals(vec.resample_matches(f, 1))


def test_eligibility_counts_legal_balls_per_role_and_sorts_by_id():
    f = pd.DataFrame({"batter_id": ["b"] * 3 + ["a"] * 2 + ["c"], "bowler_id": ["x"] * 6})
    assert vec.eligibility(f, "batter", 2).to_dict() == {"a": 2, "b": 3}
    assert list(vec.eligibility(f, "batter", 2).index) == ["a", "b"]
    assert vec.eligibility(f, "bowler", 6).to_dict() == {"x": 6}


def test_stats_vector_matches_a_hand_calculation_and_thin_phases_stay_finite():
    # class ids: 0 dot, 1 single, 4 four, 6 wicket. p1 powerplay: dot, 1, four, W; the phase prior is uniform
    # on {0, 1, 4, 6} because p2 faced the same four balls
    y = [0, 1, 4, 6] * 2
    f = pd.DataFrame(
        {
            "batter_id": ["p1"] * 4 + ["p2"] * 4,
            "phase": ["powerplay"] * 8,
            "y": y,
        }
    )
    ids = pd.Index(["p1", "p3"])  # p3 never batted
    s = vec.stats_vectors(f, "batter", ids, m=4)
    assert list(s.columns) == vec.STAT_COLUMNS and s.shape == (2, 12)
    prior = np.zeros(K)
    prior[[0, 1, 4, 6]] = 0.25
    counts = np.zeros(K)
    counts[[0, 1, 4, 6]] = 1
    p = (counts + 4 * prior) / (4 + 4)
    assert s.loc["p1", "dot_pct_powerplay"] == pytest.approx(p[0] * 100)
    assert s.loc["p1", "dismissal_rate_powerplay"] == pytest.approx(p[6] * 100)
    assert s.loc["p1", "boundary_pct_powerplay"] == pytest.approx(p[4] * 100)
    never_batted = s.loc["p3", "dot_pct_powerplay"]
    assert never_batted == pytest.approx(prior[0] * 100)  # exactly the phase prior


def test_reduced_rows_without_a_phase_drop_out_of_the_stats_vector():
    f = pd.DataFrame({"batter_id": ["p1"] * 4, "phase": ["middle", "middle", None, None], "y": [0, 1, 6, 6]})
    s = vec.stats_vectors(f, "batter", pd.Index(["p1"]), m=1)
    wickets = s.loc["p1", "dismissal_rate_middle"]
    assert wickets == 0.0  # the two wickets had no phase, so they are not counted


def test_embedding_rows_follow_the_index_and_missing_players_fail(tmp_path):
    emb = np.arange(10.0).reshape(5, 2)
    pd.DataFrame(
        {
            "player_id": ["__UNK__", "a", "b", "__UNK__", "a"],
            "role": ["batter", "batter", "batter", "bowler", "bowler"],
            "n_train_balls": 1,
        }
    ).to_parquet(tmp_path / "embedding_index.parquet")
    np.save(tmp_path / "embeddings.npy", emb)
    got = vec.embedding_rows(tmp_path, "bowler", pd.Index(["a"]))
    np.testing.assert_array_equal(got, emb[[4]])
    with pytest.raises(ValueError, match="no embedding"):
        vec.embedding_rows(tmp_path, "bowler", pd.Index(["b"]))
    ids = pd.Index(["a", "b"])  # D-045: a player UNK in a resample gets the shared UNK row
    np.testing.assert_array_equal(vec.embedding_rows(tmp_path, "bowler", ids, allow_unk=True), emb[[4, 3]])
    assert list(vec.has_own_embedding(tmp_path, "bowler", ids)) == [True, False]


def test_choose_k_needs_the_cutoff_then_takes_top_silhouette_and_the_smaller_k_on_ties():
    def row(k, sil, ari):
        return {"k": k, "silhouette": sil, "boot_ari_mean": ari}

    assert stability.choose_k([row(3, 0.5, 0.59), row(4, 0.9, 0.2)]) is None
    table = [row(3, 0.30, 0.9), row(4, 0.40, 0.7), row(5, 0.398, 0.8), row(6, 0.60, 0.5)]
    assert stability.choose_k(table)["k"] == 4  # k=6 fails the cut-off; 5 is within 0.005 of 4 but larger
    assert stability.choose_k([row(3, 0.402, 0.9), row(4, 0.40, 0.9)])["k"] == 3


def test_stable_blobs_have_high_ari_and_the_ladder_stops_at_step_zero(small_search):
    X = _blobs()
    rng = np.random.default_rng(1)
    boots = [X + rng.normal(scale=0.1, size=X.shape) for _ in range(4)]
    cell = stability.Cell(np.full(len(X), 1_000), X, boots)
    r = stability.run_ladder(cell)
    assert (r["ladder_step"], r["k"], r["stable"], r["pca"]) == (0, 3, True, False)
    assert r["chosen"]["boot_ari_mean"] > 0.95 and r["chosen"]["seed_ari_mean"] > 0.95
    assert set(r["sensitivity_choice"]) == {"150", "500"}


def test_noise_walks_the_whole_ladder_and_reports_the_best_k_as_unstable(small_search):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 4))
    boots = [rng.normal(size=X.shape) for _ in range(4)]  # nothing is shared with the reference
    n_balls = np.where(np.arange(len(X)) < 100, 1_000, 400)  # 500 drops 20 players
    r = stability.run_ladder(stability.Cell(n_balls, X, boots))
    assert (r["ladder_step"], r["stable"], r["pca"], r["min_balls"]) == (3, False, True, 500)
    assert r["n_players"] == 100
    best = max(r["tables"]["500_pca"], key=lambda t: t["boot_ari_mean"])
    assert r["k"] == best["k"]
    assert {"300_raw", "500_raw", "500_pca"} <= set(r["tables"])


def test_seeding_experiment_reports_spread_for_both_inits():
    out = stability.seeding_experiment(_blobs(per=30), ks=(3, 4), seeds=range(4))
    assert set(out) == {"random", "k-means++"} and set(out["random"]) == {"3", "4"}
    cell = out["k-means++"]["3"]
    assert cell["inertia_min"] <= cell["inertia_mean"] <= cell["inertia_max"] and cell["n_iter_max"] >= 1


def test_profile_reports_z_against_the_population_and_the_nearest_players():
    stats = pd.DataFrame({"s1": [0.0, 0.2, 10.0, 10.2], "s2": [5.0, 5.0, 5.0, 5.0]}, index=list("abcd"))
    space = stats.to_numpy()
    labels = np.array([0, 0, 1, 1])
    centers = np.array([[0.1, 5.0], [10.1, 5.0]])
    p = profile(stats, space, labels, centers)
    assert [c["size"] for c in p] == [2, 2]
    assert p[1]["top_stats"][0]["stat"] == "s1" and p[1]["top_stats"][0]["z"] > 0 > p[0]["top_stats"][0]["z"]
    assert p[0]["z_stats"]["s2"] == 0.0  # constant stat: std replaced by 1, z = 0
    assert set(p[0]["central_players"]) == {"a", "b"} and p[0]["name"] is None


def _synthetic_train(seed=0, n_players=100, n_matches=250, per_match=400):
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    f = pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(n_matches), per_match),
            "batter_id": rng.choice([f"bat{i:02d}" for i in range(n_players)], n),
            "bowler_id": rng.choice([f"bwl{i:02d}" for i in range(n_players)], n),
            "phase": rng.choice(PHASES, n),
            "y": rng.integers(0, K, n),
        }
    )
    return f


def _fake_embeddings(path, seed, dim=4):
    rng = np.random.default_rng(seed)
    rows = [("__UNK__", "batter")] + [(f"bat{i:02d}", "batter") for i in range(100)]
    rows += [("__UNK__", "bowler")] + [(f"bwl{i:02d}", "bowler") for i in range(100)]
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["player_id", "role"]).assign(n_train_balls=1).to_parquet(
        path / "embedding_index.parquet"
    )
    np.save(path / "embeddings.npy", rng.normal(size=(len(rows), dim)))


def test_cluster_run_writes_every_output_with_the_documented_contract(tmp_path, monkeypatch, small_search):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "art")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "rep")
    monkeypatch.setattr(
        cluster, "seeding_experiment", lambda X: stability.seeding_experiment(X, (3,), range(2))
    )
    n_boot = 3
    _fake_embeddings(config.ARTIFACTS / "p3", 0)
    for b in range(n_boot):
        _fake_embeddings(vec.boot_dir(b), b + 1)

    out = cluster.run(_synthetic_train(), n_boot=n_boot)

    assert set(out["clusters"]) == {"stats_batter", "stats_bowler", "emb_batter", "emb_bowler"}
    for rep in ("stats", "emb"):
        frame = pd.read_parquet(vec.artifact_dir() / f"archetypes_{rep}.parquet")
        assert set(frame["method"]) == {"kmeans", "gmm", "hdbscan"}
        assert set(frame["role"]) == {"batter", "bowler"}
        assert {"player_id", "role", "rep", "method", "k", "label", "prob", "outlier"} == set(frame.columns)
        assert (frame["rep"] == rep).all()
    for role in vec.ROLES:
        model = joblib.load(vec.artifact_dir() / f"kmeans_stats_{role}.joblib")
        c = out["clusters"][f"stats_{role}"]
        assert model[-1].n_clusters == c["k"]
        assert model.predict(np.zeros((1, 12))).shape == (1,)  # Phase 5 assigns raw stats vectors
        assert -1.0 <= out["comparison"][role]["cross_rep_ari"] <= 1.0
    for name in ("clusters", "seeding", "comparison", "profiles"):
        assert (vec.report_dir() / f"{name}.json").exists()


def test_jaccard_ignores_label_names_and_drops_for_a_scrambled_resample():
    from cricmind.archetypes import diagnostics as dg

    ref = np.repeat([0, 1, 2], 20)
    renamed = np.repeat([2, 0, 1], 20)  # same groups, different ids
    np.testing.assert_allclose(dg.jaccard_by_cluster(ref, renamed), 1.0)
    scrambled = np.random.default_rng(0).integers(0, 3, 60)
    assert dg.jaccard_by_cluster(ref, scrambled).max() < 0.6
    merged = np.repeat([0, 0, 1], 20)  # clusters 0 and 1 merge: each overlaps half of the union
    assert dg.jaccard_by_cluster(ref, merged).tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_readings_and_variant_verdict_follow_the_d046_cut_offs():
    from cricmind.archetypes import diagnostics as dg

    assert [dg.reading(j) for j in (0.8, 0.7, 0.55, 0.4)] == ["stable", "pattern", "untrusted", "dissolved"]
    base = [{"k": 3, "boot_ari_mean": 0.47}, {"k": 4, "boot_ari_mean": 0.31}]

    def best(ari):
        return dg.variant_verdict(base, [{"k": 3, "boot_ari_mean": ari}])["verdict"]

    assert (best(0.65), best(0.58), best(0.52)) == ("fixes", "improves", "no_change")
    assert dg.KEEP_DIMS == list(range(9))  # stat-major order: dismissal rate is the last 3 dims


def test_diagnostics_run_after_cluster_run_writes_both_checks(tmp_path, monkeypatch, small_search):
    from cricmind.archetypes import diagnostics as dg

    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "art")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "rep")
    monkeypatch.setattr(
        cluster, "seeding_experiment", lambda X: stability.seeding_experiment(X, (3,), range(2))
    )
    _fake_embeddings(config.ARTIFACTS / "p3", 0)
    for b in range(3):
        _fake_embeddings(vec.boot_dir(b), b + 1)
    train = _synthetic_train()
    cluster.run(train, n_boot=3)

    out = dg.run(train, n_boot=3)

    assert set(out["clusterwise"]) == {"stats_batter", "stats_bowler", "emb_batter", "emb_bowler"}
    for clusters in out["clusterwise"].values():
        assert all(0.0 < c["mean_jaccard"] <= 1.0 and c["size"] > 0 for c in clusters)
    for v in out["dismissal_free_vector"].values():
        assert len(v["dims"]) == 9 and v["verdict"] in {"fixes", "improves", "no_change"}
    assert (vec.report_dir() / "diagnostics.json").exists()
