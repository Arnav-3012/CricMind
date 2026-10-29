import pandas as pd
import pytest

from cricmind import config
from cricmind.features import players


def _frame(batters, bowlers, year=2022):
    return pd.DataFrame({"batter_id": batters, "bowler_id": bowlers, "year": year})


def test_vocab_keeps_ids_at_the_floor_and_reserves_zero_for_unk():
    f = _frame(["a"] * 3 + ["b"] * 2 + ["c"], ["x"] * 6)
    assert players.build_vocab(f, "batter_id", floor=2) == {"a": 1, "b": 2}  # c has 1 ball: below the floor
    assert players.build_vocab(f, "batter_id", floor=7) == {}
    assert players.UNK == 0


def test_encode_maps_rare_and_unseen_ids_to_unk():
    vocab = {"a": 1, "b": 2}
    codes = players.encode(pd.Series(["b", "zzz", "a", "c"]), vocab)
    assert codes.tolist() == [2, 0, 1, 0]
    assert codes.dtype == "int64"


def test_vocab_is_fitted_on_train_rows_only():
    with pytest.raises(ValueError, match="train rows only"):
        players.build_vocab(_frame(["a"], ["x"], year=config.VAL_YEAR), "batter_id", floor=1)
    with pytest.raises(ValueError, match="train rows only"):
        players.seen_pairs(_frame(["a"], ["x"], year=config.VAL_YEAR))


def test_holdout_vocab_built_from_earlier_years_makes_a_debutant_unk():
    fit = _frame(["old"] * 3, ["x"] * 3, year=2022)  # the 2023 holdout debutant is not in the fit rows
    vocab = players.build_vocab(fit, "batter_id", floor=1)
    assert players.encode(pd.Series(["old", "debutant"]), vocab).tolist() == [1, 0]


def test_unseen_pair_mask_is_about_the_pair_not_the_players():
    train = _frame(["a", "a", "b"], ["x", "y", "x"])
    pairs = players.seen_pairs(train)
    val = _frame(["a", "a", "b", "b", "new"], ["x", "y", "y", "x", "x"], year=config.VAL_YEAR)
    mask = players.unseen_pair_mask(val, pairs)
    # (b, y) is unseen although b and y were each in train; (new, x) is unseen because new is.
    assert mask.tolist() == [False, False, True, False, True]
