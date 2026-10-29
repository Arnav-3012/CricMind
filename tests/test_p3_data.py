import datetime as dt

import pandas as pd
import pytest

from cricmind import config
from cricmind.models.p3_data import check_split_integrity, holdout_split


def _frame(match_ids, dates):
    return pd.DataFrame({"match_id": match_ids, "match_date": dates})


def test_integrity_passes_for_disjoint_matches_in_time_order():
    train = _frame(["a", "b"], [dt.date(2023, 12, 30), dt.date(2023, 12, 31)])
    check_split_integrity(train, _frame(["c"], [dt.date(2024, 1, 1)]))


def test_integrity_rejects_a_match_in_both_splits():
    with pytest.raises(ValueError, match="both train and val"):
        check_split_integrity(
            _frame(["a"], [dt.date(2023, 5, 1)]), _frame(["a", "c"], [dt.date(2024, 5, 1)] * 2)
        )


def test_integrity_rejects_time_overlap():
    with pytest.raises(ValueError, match="overlap in time"):
        check_split_integrity(_frame(["a"], [dt.date(2024, 3, 1)]), _frame(["c"], [dt.date(2024, 1, 1)]))


def test_holdout_split_is_the_tuning_year_and_strictly_earlier_years():
    train = pd.DataFrame({"year": [2021, 2022, config.TUNE_HOLDOUT_YEAR, config.TUNE_HOLDOUT_YEAR]})
    fit, hold = holdout_split(train)
    assert (fit["year"] < config.TUNE_HOLDOUT_YEAR).all() and len(fit) == 2
    assert (hold["year"] == config.TUNE_HOLDOUT_YEAR).all() and len(hold) == 2
