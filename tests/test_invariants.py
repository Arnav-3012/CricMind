"""Phase 1 data invariants. Full processed parquet locally; the tests/fixtures/*.json matches in CI."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cricmind import config
from cricmind.data import parse

FIXTURES = Path(__file__).parent / "fixtures"
TARGET_MISMATCH_MAX = 0.005  # plan: Cricsheet stores no innings total, so target.runs - 1 is the cross-check
BALL_RUNS_MISMATCH_MAX = 0.0001  # plan: 0.01%; a guess until the first full run shows the real rate


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


def test_a_W_ball_names_the_striker_and_a_bowler_dismissal(data):
    deliveries, _ = data  # D-028 regression: a retirement listed first once labelled a W ball "retired hurt"
    w = deliveries[deliveries["outcome"] == "W"]
    assert not w["wicket_kind"].isin(list(parse.NOT_STRIKER_WICKET)).any()
    assert (w["player_out_id"] == w["batter_id"]).all()


def test_is_reduced_agrees_between_matches_and_deliveries(data):
    deliveries, matches = data  # D-028
    per_match = deliveries.groupby("match_id")["is_reduced"].agg(["min", "max"])
    assert (per_match["min"] == per_match["max"]).all(), "is_reduced varies within a match"
    joined = per_match["max"].rename("d").to_frame().join(matches.set_index("match_id")["is_reduced"])
    assert (joined["d"] == joined["is_reduced"]).all()
    assert matches.loc[matches["method"].notna(), "is_reduced"].all()


def test_result_is_win_only_with_a_winner(data):
    _, matches = data
    assert matches.loc[matches["result"] == "win", "winner"].notna().all()


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


def test_venue_key_is_non_null_and_the_same_on_a_match_and_its_deliveries(data):
    deliveries, matches = data
    assert not deliveries["venue_key"].isna().any()
    assert not matches["venue_key"].isna().any()
    per_match = deliveries[["match_id", "venue_key"]].drop_duplicates()
    joined = per_match.merge(matches[["match_id", "venue_key"]], on="match_id", suffixes=("_d", "_m"))
    assert len(joined) == len(matches) == len(per_match)
    assert (joined["venue_key_d"] == joined["venue_key_m"]).all()


def test_venue_key_and_registry_venue_id_agree(data):
    _, matches = data  # D-027: registry ids check the key (never replace it); both directions must be 1:1
    with_id = matches.dropna(subset=["venue_id"])
    assert (with_id.groupby("venue_key")["venue_id"].nunique() <= 1).all(), "one venue_key, two registry ids"
    assert (with_id.groupby("venue_id")["venue_key"].nunique() == 1).all(), "one registry id, two venue_keys"


def test_first_innings_total_plus_one_is_the_target(data):
    deliveries, matches = data
    regular = deliveries[~deliveries["is_super_over"]]
    total = regular[regular["innings"] == 1].groupby("match_id")["runs_total"].sum().rename("total")
    target = regular[regular["innings"] == 2].groupby("match_id")["target"].first()
    check = pd.concat([total, target], axis=1).join(matches.set_index("match_id")["is_reduced"])
    check = check[~check["is_reduced"]].dropna(subset=["total", "target"])  # a revision resets the target
    if check.empty:
        pytest.skip("no unreduced chase in this data")
    mismatch = (check["total"] + 1 != check["target"]).mean()
    assert mismatch <= TARGET_MISMATCH_MAX, f"{mismatch:.3%} of {len(check)} matches"


def test_ball_runs_add_up(data):
    deliveries, _ = data  # runs_batter feeds the outcome classes; Cricsheet gives runs_total independently
    bad = deliveries["runs_batter"] + deliveries["runs_extras"] != deliveries["runs_total"]
    examples = deliveries.loc[bad, "match_id"].head(3).tolist()
    assert bad.mean() <= BALL_RUNS_MISMATCH_MAX, f"{bad.sum()} of {len(bad)} balls, e.g. {examples}"
