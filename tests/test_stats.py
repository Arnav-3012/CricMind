import numpy as np
import pandas as pd
import pytest

from cricmind import config
from cricmind.features import stats

K = len(config.OUTCOME_CLASSES)


def _frame(n_matches=20, balls=6, seed=0):
    """Random labels, 3 batters and 2 venues, one ball sequence per match; `solo` appears in match 0 only."""
    rng = np.random.default_rng(seed)
    n = n_matches * balls
    df = pd.DataFrame(
        {
            "match_id": np.repeat([f"m{i}" for i in range(n_matches)], balls),
            "batter_id": rng.choice(["a", "b", "c"], n),
            "venue_key": np.repeat(rng.choice(["v1", "v2"], n_matches), balls),
            "y": rng.integers(0, K, n),
        }
    )
    df.loc[0, "batter_id"] = "solo"
    return df


def test_fit_and_apply_follow_the_shrinkage_formula_and_unseen_keys_get_the_prior():
    train = pd.DataFrame({"batter_id": ["a"] * 4 + ["b"], "y": [0, 0, 0, 6, 1]})
    table = stats.fit_rates(train, ["batter_id"], m=10)
    out = stats.apply_rates(pd.DataFrame({"batter_id": ["a", "zzz"]}), table, ["batter_id"], "bat")
    prior = np.bincount(train["y"], minlength=K) / 5
    a_counts = np.bincount([0, 0, 0, 6], minlength=K)
    assert out.loc[0, "bat_n"] == 4
    np.testing.assert_allclose(out.loc[0, "bat_p0":"bat_p6"], (a_counts + 10 * prior) / 14)
    assert out.loc[1, "bat_n"] == 0  # an unseen key: no evidence, so exactly the prior
    np.testing.assert_allclose(out.loc[1, "bat_p0":"bat_p6"], prior)


def test_oof_rates_match_a_brute_force_loop_over_other_folds():
    df = _frame()
    out = stats.oof_rates(df, ["batter_id"], "bat", m=7)
    fold = stats.fold_of(df["match_id"])
    for i in range(len(df)):
        others = df[fold != fold[i]]
        same = others[others["batter_id"] == df["batter_id"].iloc[i]]
        prior = np.bincount(others["y"], minlength=K) / len(others)
        counts = np.bincount(same["y"], minlength=K)
        assert out["bat_n"].iloc[i] == len(same), i
        np.testing.assert_allclose(out.iloc[i, 1:], (counts + 7 * prior) / (len(same) + 7), err_msg=str(i))


def test_a_rows_own_label_never_reaches_its_own_features():
    df = _frame()
    flipped = df.copy()
    flipped.loc[3, "y"] = (df.loc[3, "y"] + 1) % K
    a = stats.oof_rates(df, ["batter_id"], "bat")
    b = stats.oof_rates(flipped, ["batter_id"], "bat")
    fold = stats.fold_of(df["match_id"])
    same_fold = fold == fold[3]
    pd.testing.assert_frame_equal(a[same_fold], b[same_fold])  # rows sharing the fold never see ball 3
    assert not a[~same_fold].equals(b[~same_fold])  # the other folds do, which is the point of the encoding


def test_a_key_seen_in_one_match_has_no_out_of_fold_evidence():
    df = _frame()
    out = stats.oof_rates(df, ["batter_id"], "bat")
    assert out.loc[0, "bat_n"] == 0  # `solo` plays only match 0, whose fold is held out
    assert out.loc[0, "bat_p0":"bat_p6"].sum() == pytest.approx(1.0)  # the fold prior, not a label


def test_folds_hold_whole_matches_and_are_balanced():
    df = _frame(n_matches=20)
    fold = stats.fold_of(df["match_id"], folds=5)
    assert pd.Series(fold).groupby(df["match_id"]).nunique().eq(1).all()
    assert np.bincount(fold[::6]).tolist() == [4] * 5  # 20 matches over 5 folds
    assert (stats.fold_of(df["match_id"], folds=5) == fold).all()  # deterministic


def test_rates_sum_to_one_for_single_and_composite_keys():
    df = _frame()
    for keys in (["batter_id"], ["batter_id", "venue_key"]):
        out = stats.oof_rates(df, keys, "k")
        np.testing.assert_allclose(out[[f"k_p{i}" for i in range(K)]].sum(axis=1), 1.0)
        table = stats.fit_rates(df, keys)
        applied = stats.apply_rates(df, table, keys, "k")
        np.testing.assert_allclose(applied[[f"k_p{i}" for i in range(K)]].sum(axis=1), 1.0)


def test_all_helpers_produce_the_documented_columns():
    df = _frame().assign(bowler_id="x")
    specs = {"bat": ["batter_id"], "venue": ["venue_key"]}
    oof = stats.oof_all(df, specs)
    applied = stats.apply_all(df, stats.fit_all(df, specs), specs)
    assert list(oof.columns) == list(applied.columns) == stats.stat_columns(specs)
    assert oof.index.equals(df.index)
