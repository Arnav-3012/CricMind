import json

import pandas as pd

from cricmind.data import parse, profile
from test_parse import _d, _raw


def _write_tiny(processed):
    bowled = [{"player_out": "A", "kind": "bowled"}]
    run_out = [{"player_out": "B", "kind": "run out"}]
    raw = _raw([[_d(), _d(runs=1), _d(extras={"wides": 1}), _d(wickets=bowled), _d(wickets=run_out)]])
    match, rows = parse.parse_match(raw, "m1")
    processed.mkdir()
    parse.derive(pd.DataFrame(rows)).to_parquet(processed / "deliveries.parquet", index=False)
    matches = pd.DataFrame([match], columns=parse.MATCH_COLUMNS)
    matches["match_date"] = pd.to_datetime(matches["match_date"]).dt.date
    matches.to_parquet(processed / "matches.parquet", index=False)


def test_profile_counts_on_a_tiny_match(tmp_path):
    _write_tiny(tmp_path / "processed")
    p = profile.build_profile(tmp_path / "processed", tmp_path / "raw")
    json.dumps(p, default=float)  # must be serialisable

    assert p["totals"]["modelled_legal_balls"] == 4  # the wide is excluded
    assert p["totals"]["deliveries"] == 5
    assert p["per_year"] == [{"year": 2024, "matches": 1, "legal_balls": 4}]
    shares = {r["outcome"]: r["n"] for r in p["outcome_share_overall"]}
    assert shares == {"dot": 2, "1": 1, "W": 1}  # a non-striker run out is not W
    (ro,) = p["run_out_share_by_phase"]
    assert (ro["phase"], ro["dismissals"], ro["run_outs"], ro["share"]) == ("powerplay", 2, 1, 0.5)
    assert p["venue_id_null"]["match_share"] == 1.0
    assert p["venues"] == {
        "venue_keys": 1, "raw_venue_city_pairs": 1, "keys_without_city": 0, "registry_ids": 0,
    }  # fmt: skip
    assert p["venue_names_in_several_cities"] == []
    assert p["checks"]["balls_over_budget"] == 0
    assert p["checks"]["unflagged_overs_over_6_legal_balls"] == 0
