import json

import pandas as pd

from cricmind.data import parse, profile
from test_parse import _d, _raw


def _write_tiny(processed, raw_dir, year):
    bowled = [{"player_out": "A", "kind": "bowled"}]
    run_out = [{"player_out": "B", "kind": "run out"}]
    raw = _raw([[_d(), _d(runs=1), _d(extras={"wides": 1}), _d(wickets=bowled), _d(wickets=run_out)]])
    match, rows = parse.parse_match(raw, "m1")
    match["match_date"] = f"{year}-06-01"  # D-005: the split is by match year
    deliveries = pd.DataFrame(rows)
    deliveries["match_date"] = f"{year}-06-01"
    processed.mkdir()
    parse.derive(deliveries).to_parquet(processed / "deliveries.parquet", index=False)
    matches = pd.DataFrame([match], columns=parse.MATCH_COLUMNS)
    matches["match_date"] = pd.to_datetime(matches["match_date"]).dt.date
    matches.to_parquet(processed / "matches.parquet", index=False)
    raw_dir.mkdir()
    (raw_dir / "MANIFEST.json").write_text(json.dumps({"snapshot": "test"}))  # D-021: profile needs the pin


def _profile(tmp_path, year=2023):
    _write_tiny(tmp_path / "processed", tmp_path / "raw", year)
    return profile.build_profile(tmp_path / "processed", tmp_path / "raw")


def test_profile_counts_on_a_tiny_match(tmp_path):
    p = _profile(tmp_path)  # 2023 is a train season
    json.dumps(p, default=float)  # must be serialisable

    assert p["totals"]["modelled_legal_balls"] == 4  # the wide is excluded
    assert p["totals"]["deliveries"] == 5
    assert (p["totals"]["reduced_matches"], p["totals"]["reduced_modelled_balls"]) == (0, 0)
    assert p["totals"]["matches_with_method"] == 0
    assert p["rows_by_split"] == [{"split": "train", "matches": 1, "legal_balls": 4}]
    assert p["per_year"] == [{"year": 2023, "matches": 1, "legal_balls": 4}]
    shares = {r["outcome"]: r["n"] for r in p["outcome_share_overall"]}
    assert shares == {"dot": 2, "1": 1, "W": 1}  # a non-striker run out is not W
    (ro,) = p["run_out_share_by_phase"]
    assert (ro["phase"], ro["dismissals"], ro["run_outs"], ro["share"]) == ("powerplay", 2, 1, 0.5)
    assert [r["run_outs"] for r in p["run_out_by_over"]] == [1]
    assert sum(r["legal_balls"] for r in p["run_out_by_over"]) == 4
    assert set(p["run_out_by_role"]) == {"striker_run_outs", "non_striker_run_outs"}
    assert p["venue_id_null"]["match_share"] == 1.0
    assert p["venues"] == {
        "venue_keys": 1, "raw_venue_city_pairs": 1, "keys_without_city": 0, "registry_ids": 0,
    }  # fmt: skip
    assert p["venue_names_in_several_cities"] == []
    assert p["checks"]["balls_over_budget"] == 0
    assert p["checks"]["unflagged_overs_over_6_legal_balls"] == 0
    assert p["checks"]["duplicate_keys"] == 0
    assert p["checks"]["outcome_legal_mismatch"] == 0
    assert p["checks"]["ball_runs_mismatch"] == 0
    assert p["checks"]["target_cross_check"]["checked"] == 0  # no second innings in the tiny match


def test_design_figures_skip_non_train_rows(tmp_path):
    p = _profile(tmp_path, year=2025)  # a test season: counted, never used for design figures
    assert p["rows_by_split"] == [{"split": "test", "matches": 1, "legal_balls": 4}]
    assert p["totals"]["modelled_legal_balls"] == 4  # totals still count every split
    assert p["outcome_share_overall"] == []
    assert p["outcome_share_by_phase"] == []
    assert p["run_out_share_by_phase"] == []
    assert p["run_out_by_over"] == []
    assert p["run_out_by_role"] == {"striker_run_outs": 0, "non_striker_run_outs": 0}
