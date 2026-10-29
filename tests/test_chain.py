import numpy as np
import pandas as pd
import pytest

from cricmind import config
from cricmind.archetypes import roles
from cricmind.eval.metrics import row_log_loss
from cricmind.features.context import PHASES
from cricmind.matchups import table
from cricmind.matchups.chain import Chain, Level, bat_keys, bowl_keys, cell_keys, pair_keys, shrink
from cricmind.models.baselines import GlobalFreq, PhaseFreq

PLAYERS = ["a", "b", "c", "d", "e"]


def make_frame(n: int, seed: int) -> pd.DataFrame:
    """Batter 'a' hits more often than the rest, so a batter level has something to find."""
    rng = np.random.default_rng(seed)
    over = rng.integers(0, 20, n)
    bat = rng.choice(PLAYERS, n)
    y = np.where((bat == "a") & (rng.random(n) < 0.4), 4, rng.integers(0, 7, n))
    phase = np.select([over < 6, over < 15], PHASES[:2], PHASES[2]).astype(object)
    phase[rng.random(n) < 0.04] = None  # reduced-match rows have no phase
    return pd.DataFrame(
        {
            "batter_id": bat,
            "bowler_id": rng.choice(PLAYERS, n),
            "over": over,
            "wickets_before": rng.integers(0, 8, n),
            "batter_balls_faced": rng.integers(0, 3, n),
            "phase": phase,
            "y": y.astype("int8"),
            "match_id": rng.integers(0, 40, n),
        }
    )


LABELS = {
    "batter": pd.Series({"a": 0, "b": 0, "c": 1}),
    "bowler": pd.Series({"a": 0, "b": 1}),
}
SPECS = [("cell", cell_keys(LABELS)), ("bat", bat_keys), ("bowl", bowl_keys), ("pair", pair_keys)]


@pytest.fixture
def fitted():
    train, val = make_frame(8000, 0), make_frame(4000, 1)
    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    chain = Chain(SPECS, phase_freq)
    return train, val, phase_freq, chain, chain.fit(train, val)


def test_level_lookup_is_zero_for_unseen_and_incomplete_keys():
    train = make_frame(2000, 0)
    level = Level("bat", bat_keys).fit(train)
    probe = pd.DataFrame({"batter_id": ["a", "zzz", "a"], "phase": ["death", "death", None]})
    counts = level.lookup(probe)
    assert counts[0].sum() > 0 and counts[1].sum() == 0 and counts[2].sum() == 0


def test_shrink_limits():
    counts, prior = np.array([[3.0, 1.0]]), np.array([[0.5, 0.5]])
    assert np.allclose(shrink(counts, prior, 0.0), [[0.75, 0.25]])
    assert np.allclose(shrink(counts, prior, 1e12), prior, atol=1e-9)


def test_each_level_never_hurts_val_and_rows_sum_to_one(fitted):
    _, val, _, chain, report = fitted
    losses = [s["tune_log_loss"] for s in report["steps"]]
    # s tops out at 1e6, not infinity, so a useless level may cost a hair; 1e-4 is far below any real gain
    assert all(b <= a + 1e-4 for a, b in zip(losses, losses[1:], strict=False))
    assert np.allclose(chain.predict_proba(val).sum(axis=1), 1.0)


def test_predict_reproduces_the_fit_path(fitted):
    _, val, _, chain, report = fitted
    final = row_log_loss(chain.predict_proba(val), val["y"].to_numpy()).mean()
    assert final == pytest.approx(report["steps"][-1]["tune_log_loss"])


def test_refit_keeps_s_and_changes_only_the_counts(fitted):
    train, val, phase_freq, chain, _ = fitted
    s_before, before = list(chain.s), chain.predict_proba(val)
    chain.refit(make_frame(3000, 5), PhaseFreq(GlobalFreq()).fit(make_frame(3000, 5)))
    assert chain.s == s_before
    assert not np.allclose(chain.predict_proba(val), before)
    chain.refit(train, phase_freq)
    assert np.allclose(chain.predict_proba(val), before)


def test_rows_with_no_phase_get_the_phase_baseline(fitted):
    _, val, phase_freq, chain, _ = fitted
    no_phase = val["phase"].isna().to_numpy()
    assert no_phase.any()
    assert np.allclose(chain.predict_proba(val)[no_phase], phase_freq.predict_proba(val)[no_phase])


def test_unlabelled_players_skip_the_cell_level():
    train = make_frame(2000, 0)
    keys = cell_keys(LABELS)(train)
    unlabelled = ~train["batter_id"].isin(LABELS["batter"].index) | ~train["bowler_id"].isin(
        LABELS["bowler"].index
    )
    assert keys[unlabelled].isna().any(axis=1).all()


def test_usage_shares_sum_to_one_and_lean_toward_the_phase_a_player_uses():
    frame = make_frame(6000, 0)
    frame.loc[frame["batter_id"] == "a", "over"] = 18  # 'a' only bats at the death
    ids = pd.Index(PLAYERS)
    shares = roles.usage_shares(frame, "batter", ids)
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert shares.loc["a", "share_death"] > 0.8 > shares.loc["b", "share_death"]


def test_entry_features_shrink_toward_the_population_for_a_batter_with_no_entries():
    frame = make_frame(3000, 0)
    frame.loc[frame["batter_id"] == "e", "batter_balls_faced"] = 1  # 'e' never has a first ball
    entry = roles.entry_features(frame, pd.Index(PLAYERS))
    first = frame[frame["batter_balls_faced"] == 0]
    assert entry.loc["e", "entry_over"] == pytest.approx(first["over"].mean())


def test_refit_labels_use_only_the_part_and_the_given_recipe():
    part = make_frame(6000, 0)
    spec = {role: {"k": 2, "min_balls": 100, "pca": False} for role in ("batter", "bowler")}
    labels, models = table.refit_labels("role", part, spec)
    for role in ("batter", "bowler"):
        assert set(labels[role].index) <= set(PLAYERS)
        assert set(labels[role]) <= {0, 1}
        assert models[role][-1].n_clusters == 2
    extended, info = table.extend_labels("role", part, labels, min_balls=50, models=models)
    assert info["batter"]["agrees_with_phase4"] == 1.0  # re-assigning the fitted players returns their labels
    assert extended["batter"].index.is_unique


def test_role_vector_shapes():
    frame = make_frame(6000, 0)
    ids = pd.Index(PLAYERS)
    assert roles.role_vectors(frame, "batter", ids).shape == (5, 14)
    assert roles.role_vectors(frame, "bowler", ids).shape == (5, 12)
    assert config.PHASE_EDGES == (6, 15)  # make_frame's phase labels assume these edges
