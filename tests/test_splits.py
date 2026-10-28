import datetime as dt

import duckdb
import pandas as pd
import pytest

from cricmind.data import parse, profile
from cricmind.features import splits
from test_parse import _d, _raw

BOUNDARY_DATES = ["2008-04-18", "2022-12-31", "2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01",
                  "2026-12-31", "2027-01-01"]  # fmt: skip


def test_assign_split_boundaries():
    got = splits.assign_split(pd.Series(pd.to_datetime(BOUNDARY_DATES))).tolist()
    assert got == ["train", "train", "train", "val", "val", "test", "test", "other"]


def test_assign_split_agrees_with_the_profile_sql():
    values = ", ".join(f"(date '{d}')" for d in BOUNDARY_DATES)
    sql = f"select {profile.SPLIT_SQL} from (values {values}) t(match_date)"
    in_sql = [r[0] for r in duckdb.connect().execute(sql).fetchall()]
    assert in_sql == splits.assign_split(pd.Series(pd.to_datetime(BOUNDARY_DATES))).tolist()


def _write(tmp_path, dates):
    rows = []
    for i, date in enumerate(dates):
        rows += parse.parse_match(_raw([[_d()]], dates=[date]), f"m{i}")[1]
    parse.derive(pd.DataFrame(rows)).to_parquet(tmp_path / "deliveries.parquet", index=False)


def test_load_deliveries_returns_only_the_requested_splits(tmp_path):
    _write(tmp_path, ["2023-12-31", "2024-01-01", "2025-06-01"])

    def got(*which):
        return sorted(set(splits.load_deliveries(which, tmp_path)["match_id"]))

    assert got("train") == ["m0"]
    assert got("train", "val") == ["m0", "m1"]
    assert got("test") == ["m2"]  # only when asked for by name


def test_load_deliveries_rejects_an_unknown_split(tmp_path):
    _write(tmp_path, ["2024-05-01"])
    with pytest.raises(ValueError, match="unknown split"):
        splits.load_deliveries(("validation",), tmp_path)


def test_bounds_are_dates():
    assert all(isinstance(b, dt.date) for pair in splits.BOUNDS.values() for b in pair)
