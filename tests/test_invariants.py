"""Phase 1 data invariants. Full processed parquet locally; the tests/fixtures/*.json matches in CI."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cricmind import config
from cricmind.data import parse

FIXTURES = Path(__file__).parent / "fixtures"
TARGET_MISMATCH_MAX = 0.005  # plan: Cricsheet stores no innings total, so target.runs - 1 is the cross-check


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = config.DATA_PROCESSED
    if (processed / "deliveries.parquet").exists():
        return (
            pd.read_parquet(processed / "deliveries.parquet"),
            pd.read_parquet(processed / "matches.parquet"),
        )
    parsed = []
    for f in sorted(FIXTURES.glob("*.json")):
        raw = json.loads(f.read_text())
        if parse.in_scope(raw):
            parsed.append(parse.parse_match(raw, f.stem))
    if not parsed:
        pytest.skip("no processed data and no fixtures")
    deliveries = parse.derive(pd.DataFrame([r for _, rows in parsed for r in rows]))
    return deliveries, pd.DataFrame([m for m, _ in parsed], columns=parse.MATCH_COLUMNS)


def test_no_duplicate_delivery_keys(data):
    deliveries, _ = data
    assert not deliveries.duplicated(["match_id", "innings", "ball_seq"]).any()


def test_outcome_and_legal_ball_are_null_exactly_when_not_legal(data):
    deliveries, _ = data
    assert (deliveries["outcome"].isna() == ~deliveries["is_legal"]).all()
    assert (deliveries["legal_ball_in_over"].isna() == ~deliveries["is_legal"]).all()


def test_legal_balls_remaining_never_negative(data):
    deliveries, _ = data  # D-025 regression: miscounted overs once pushed this to -1
    assert deliveries["legal_balls_remaining"].min() >= 0


def test_player_ids_non_null_and_in_registry(data):
    deliveries, _ = data
    assert not deliveries[["batter_id", "bowler_id", "non_striker_id"]].isna().any().any()
    people = config.DATA_RAW / config.DATA_SNAPSHOT / "people.csv"
    if not people.exists():
        pytest.skip("people.csv not downloaded")
    known = set(pd.read_csv(people, usecols=["identifier"])["identifier"])
    used = pd.concat([deliveries[c] for c in ("batter_id", "bowler_id", "non_striker_id", "player_out_id")])
    unknown = set(used.dropna()) - known
    assert not unknown, f"{len(unknown)} ids not in people.csv, e.g. {sorted(unknown)[:5]}"


def test_first_innings_total_plus_one_is_the_target(data):
    deliveries, matches = data
    regular = deliveries[~deliveries["is_super_over"]]
    total = regular[regular["innings"] == 1].groupby("match_id")["runs_total"].sum().rename("total")
    target = regular[regular["innings"] == 2].groupby("match_id")["target"].first()
    check = pd.concat([total, target], axis=1).join(matches.set_index("match_id")["method"])
    check = check[check["method"].isna()].dropna(subset=["total", "target"])  # D/L resets the target
    if check.empty:
        pytest.skip("no chase without a method in this data")
    mismatch = (check["total"] + 1 != check["target"]).mean()
    assert mismatch <= TARGET_MISMATCH_MAX, f"{mismatch:.3%} of {len(check)} matches"
