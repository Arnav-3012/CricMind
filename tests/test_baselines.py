import numpy as np
import pandas as pd
import pytest

from cricmind import config
from cricmind.features.context import MASKED, NUMERIC, PHASES
from cricmind.models import baselines as bl

K = len(config.OUTCOME_CLASSES)


def _frame(n, seed, year, informative=False):
    """A synthetic context frame: random numerics, real phase labels, reduced rows masked as in D-030."""
    rng = np.random.default_rng(seed)
    over = rng.integers(0, 20, n)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    df["over"] = over
    df["wickets_before"] = rng.integers(0, 10, n)
    df["phase"] = np.select([over < 6, over < 15], PHASES[:2], PHASES[2])
    df["competition"] = rng.choice(["A", "B"], n)
    df["is_reduced"] = rng.random(n) < 0.2
    y = rng.integers(0, K, n)
    if informative:  # y follows the phase 90% of the time, so a phase model must beat the global one
        code = pd.Categorical(df["phase"], PHASES).codes
        y = np.where(rng.random(n) < 0.9, code * 2, y)
    df["y"] = y.astype("int8")
    df["match_id"] = rng.integers(0, 30, n).astype(str)
    df["year"] = year
    df.loc[df["is_reduced"], MASKED] = np.nan
    return df


def _tiny(phases, ys, wickets=None):
    return pd.DataFrame({"phase": phases, "y": np.array(ys, dtype="int8"),
                         "wickets_before": wickets if wickets is not None else [0] * len(ys)})  # fmt: skip


def test_phase_freq_matches_a_hand_computed_table():
    train = _tiny(["powerplay", "powerplay", "death"], [0, 0, 6])
    model = bl.PhaseFreq(bl.GlobalFreq()).fit(train)
    g = np.array([2, 0, 0, 0, 0, 0, 1]) / 3
    powerplay = (np.array([2, 0, 0, 0, 0, 0, 0]) + g) / (2 + 1)  # additive guard toward the global shares
    np.testing.assert_allclose(model.table[0], powerplay)
    np.testing.assert_allclose(model.table[1], g)  # no middle balls: the cell equals its parent
    np.testing.assert_allclose(model.table.sum(axis=1), 1.0)


def test_masked_rows_fall_back_to_the_previous_baseline():
    train = _frame(1000, 0, 2022)
    val = _frame(400, 1, 2024)
    glob = bl.GlobalFreq().fit(train)
    phase = bl.PhaseFreq(glob).fit(train)
    phase_wk = bl.PhaseWicketsFreq(phase).fit(train)
    masked = val["phase"].isna().to_numpy()
    assert masked.sum() > 0
    np.testing.assert_array_equal(phase.predict_proba(val)[masked], glob.predict_proba(val)[masked])
    np.testing.assert_array_equal(phase_wk.predict_proba(val)[masked], phase.predict_proba(val)[masked])
    assert not np.allclose(phase_wk.predict_proba(val)[~masked], phase.predict_proba(val)[~masked])


def test_wicket_bands_split_the_phase_cells():
    train = _tiny(["middle"] * 4, [0, 0, 6, 6], wickets=[0, 0, 7, 7])
    model = bl.PhaseWicketsFreq(bl.PhaseFreq(bl.GlobalFreq())).fit(train)
    low, high = model.table[1, 0], model.table[1, 2]  # middle, 0-2 wickets vs 6+
    assert low[0] > low[6] and high[6] > high[0]


def test_fitted_models_give_valid_probabilities():
    train, val = _frame(1000, 0, 2022), _frame(300, 1, 2024)
    glob = bl.GlobalFreq()
    phase = bl.PhaseFreq(glob)
    models = [glob, phase, bl.PhaseWicketsFreq(phase), bl.Logistic(C=0.1), bl.Lgbm(n_estimators=5)]
    for m in models:
        P = m.fit(train).predict_proba(val)
        bl._check_proba(P, len(val), m.name)
        assert (P > 0).all(), m.name


def test_predictions_do_not_read_val_labels():
    train, val = _frame(1000, 0, 2022), _frame(300, 1, 2024)
    shuffled = val.assign(y=np.random.default_rng(5).permutation(val["y"].to_numpy()))
    model = bl.PhaseWicketsFreq(bl.PhaseFreq(bl.GlobalFreq())).fit(train)
    np.testing.assert_array_equal(model.predict_proba(val), model.predict_proba(shuffled))


def test_logistic_has_no_missing_indicator_for_reduced_rows():
    train = _frame(1000, 0, 2022)
    model = bl.Logistic(C=0.1).fit(train)
    n_features = len(NUMERIC) + len(PHASES) + len(model.levels["competition"])
    assert model.lr.coef_.shape == (K, n_features)  # no extra column that could encode is_reduced
    reduced = train[train["is_reduced"]].head(3)
    phase_block = model._X(reduced)[:, len(NUMERIC) : len(NUMERIC) + len(PHASES)]
    assert phase_block.sum() == 0  # a missing phase is an all-zero one-hot row


def test_logistic_tune_picks_c_from_the_grid():
    scores = bl.Logistic().tune(_frame(1000, 0, 2022), _frame(400, 1, 2023))
    assert set(scores) == set(bl.Logistic.C_GRID)


def test_lgbm_tune_sets_a_tree_count(monkeypatch):
    monkeypatch.setattr(bl.Lgbm, "MAX_TREES", 20)
    model = bl.Lgbm()
    model.tune(_frame(1000, 0, 2022), _frame(400, 1, 2023))
    assert 1 <= model.n_estimators <= 20


def test_score_models_uses_the_same_rows_and_slices_add_up():
    train, val = _frame(2000, 0, 2022, informative=True), _frame(800, 1, 2024, informative=True)
    glob = bl.GlobalFreq()
    models = [glob, bl.PhaseFreq(glob), bl.PhaseWicketsFreq(bl.PhaseFreq(glob))]
    for m in models:
        m.fit(train)
    results = bl.score_models(models, val)
    assert [r["all"]["n"] for r in results] == [len(val)] * 3
    for r in results:
        assert r["unreduced"]["n"] + r["reduced"]["n"] == len(val)
        assert r["all"]["log_loss_ci"]["lo"] <= r["all"]["log_loss"] <= r["all"]["log_loss_ci"]["hi"]
    assert results[0]["beats_previous"] is None
    assert results[1]["beats_previous"] is True and results[1]["beats_previous_ci_excludes_0"] is True
    assert results[1]["paired_vs_previous"]["previous"] == "global_freq"


def test_check_proba_rejects_bad_matrices():
    with pytest.raises(ValueError):
        bl._check_proba(np.full((2, K), 0.5), 2, "bad")  # rows sum to 3.5
