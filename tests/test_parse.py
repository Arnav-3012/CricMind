import pandas as pd
import pytest

from cricmind.data import parse

PEOPLE = {n: f"id_{n}" for n in ("A", "B", "C", "D", "BOWL", "KEEP")}


def _raw(deliveries_by_over, *, overs=20, innings_extra=None, extra_innings=None, **info_over):
    inn = {"team": "X", "overs": [{"over": i, "deliveries": ds} for i, ds in enumerate(deliveries_by_over)]}
    inn.update(innings_extra or {})
    info = {
        "match_type": "T20", "gender": "male", "balls_per_over": 6, "overs": overs, "dates": ["2024-05-01"],
        "season": "2024", "team_type": "club", "teams": ["X", "Y"], "venue": "V", "city": "C",
        "toss": {"winner": "X", "decision": "bat"}, "outcome": {"winner": "X", "by": {"runs": 5}},
        "event": {"name": "Some League"}, "registry": {"people": PEOPLE},
    }  # fmt: skip
    info.update(info_over)
    return {"meta": {"data_version": "1.2.0"}, "info": info, "innings": [inn, *(extra_innings or [])]}


def _d(runs=0, batter="A", non_striker="B", extras=None, wickets=None, non_boundary=False, total=None):
    ex = sum((extras or {}).values())
    r = {"batter": runs, "extras": ex, "total": runs + ex if total is None else total}
    if non_boundary:
        r["non_boundary"] = True
    d = {"batter": batter, "bowler": "BOWL", "non_striker": non_striker, "runs": r}
    if extras:
        d["extras"] = extras
    if wickets:
        d["wickets"] = wickets
    return d


def _rows(deliveries_by_over, **kw):
    return parse.parse_match(_raw(deliveries_by_over, **kw), "m1")[1]


def _derived(deliveries_by_over, **kw):
    return parse.derive(pd.DataFrame(_rows(deliveries_by_over, **kw)))


def test_wide_is_not_legal_and_has_no_outcome():
    (row,) = _rows([[_d(extras={"wides": 1})]])
    assert (row["is_legal"], row["outcome"], row["wides"]) == (False, None, 1)
    assert row["legal_ball_in_over"] is None


def test_noball_with_byes_keeps_both_extras():
    (row,) = _rows([[_d(extras={"noballs": 1, "byes": 2})]])
    assert (row["noballs"], row["byes"], row["is_legal"], row["runs_extras"]) == (1, 2, False, 3)


def test_legal_ball_in_over_skips_extras():
    rows = _rows([[_d(), _d(extras={"wides": 1}), _d(), _d(extras={"noballs": 1}), _d()]])
    assert [r["legal_ball_in_over"] for r in rows] == [1, None, 2, None, 3]
    assert [r["ball_seq"] for r in rows] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("runs", "non_boundary", "expected"),
    [(0, False, "dot"), (1, False, "1"), (2, False, "2"), (3, False, "3+"), (5, False, "3+"),
     (4, False, "4"), (4, True, "3+"), (6, False, "6")],
)  # fmt: skip
def test_outcome_classes(runs, non_boundary, expected):
    (row,) = _rows([[_d(runs=runs, non_boundary=non_boundary)]])
    assert row["outcome"] == expected


def test_bowled_is_W_and_counts_as_a_fall():
    (row,) = _rows([[_d(wickets=[{"player_out": "A", "kind": "bowled"}])]])
    assert (row["outcome"], row["n_wickets"], row["wicket_kind"]) == ("W", 1, "bowled")
    assert row["player_out_id"] == "id_A"


def test_striker_run_out_falls_back_to_runs_class():
    (row,) = _rows([[_d(runs=1, wickets=[{"player_out": "A", "kind": "run out"}])]])
    assert (row["outcome"], row["n_wickets"], row["wicket_kind"]) == ("1", 1, "run out")


def test_non_striker_run_out_is_not_W():
    (row,) = _rows([[_d(runs=0, wickets=[{"player_out": "B", "kind": "run out"}])]])
    assert (row["outcome"], row["player_out_id"]) == ("dot", "id_B")


def test_stumped_off_a_wide_keeps_wicket_but_no_outcome():
    (row,) = _rows([[_d(extras={"wides": 1}, wickets=[{"player_out": "A", "kind": "stumped"}])]])
    assert (row["outcome"], row["is_legal"]) == (None, False)
    assert (row["n_wickets"], row["wicket_kind"]) == (1, "stumped")


def test_retired_hurt_is_not_a_fall_nor_W():
    (row,) = _rows([[_d(wickets=[{"player_out": "A", "kind": "retired hurt"}])]])
    assert (row["n_wickets"], row["outcome"], row["wicket_kind"]) == (0, "dot", "retired hurt")


def test_run_out_plus_retired_out_is_two_falls_and_not_W():
    wk = [{"player_out": "B", "kind": "run out"}, {"player_out": "A", "kind": "retired out"}]
    (row,) = _rows([[_d(wickets=wk)]])
    assert (row["n_wickets"], row["outcome"], row["player_out_id"]) == (2, "dot", "id_B")


def test_penalty_runs_do_not_leak_into_batter_outcome():
    (row,) = _rows([[_d(runs=2, extras={"penalty": 5})]])
    assert (row["outcome"], row["penalty"], row["runs_total"]) == ("2", 5, 7)


def test_registry_ids_are_used_and_unresolved_name_raises():
    (row,) = _rows([[_d(batter="C", non_striker="D")]])
    assert (row["batter_id"], row["bowler_id"], row["non_striker_id"]) == ("id_C", "id_BOWL", "id_D")
    with pytest.raises(ValueError, match="Ghost"):
        _rows([[_d(batter="Ghost")]])
    with pytest.raises(ValueError, match="Ghost"):
        _rows([[_d(wickets=[{"player_out": "Ghost", "kind": "bowled"}])]])


def test_venue_id_only_when_registry_has_venues():
    assert parse.parse_match(_raw([[_d()]]), "m")[0]["venue_id"] is None
    reg = {"people": PEOPLE, "venues": {"V": "ven1"}}
    match, rows = parse.parse_match(_raw([[_d()]], registry=reg), "m")
    assert (match["venue_id"], rows[0]["venue_id"]) == ("ven1", "ven1")


def test_super_over_is_flagged_numbered_after_regular_innings_and_has_one_over_budget():
    so = {"team": "Y", "super_over": True, "overs": [{"over": 0, "deliveries": [_d(runs=1)]}]}
    match, rows = parse.parse_match(_raw([[_d()]], extra_innings=[so]), "m")
    assert match["has_super_over"] is True
    got = [(r["innings"], r["is_super_over"], r["ball_budget"]) for r in rows]
    assert got == [(1, False, 120), (2, True, 6)]


def test_chase_target_and_fractional_target_overs():
    inn = {"target": {"overs": 35.1, "runs": 90}}
    (row,) = _rows([[_d()]], innings_extra=inn)
    assert (row["target"], row["ball_budget"]) == (90, 211)  # D-023: 35 overs + 1 ball
    inn = {"target": {"overs": 20, "runs": 150}}
    assert _rows([[_d()]], innings_extra=inn)[0]["ball_budget"] == 120


def test_miscounted_over_accepts_str_or_int_balls():
    inn = {"miscounted_overs": {"0": {"balls": "7"}}}
    rows = _rows([[_d()] * 7, [_d()]], innings_extra=inn)
    assert [r["miscounted_balls"] for r in rows] == [7] * 7 + [None]
    assert rows[6]["legal_ball_in_over"] == 7


def test_derive_state_columns_are_before_the_ball():
    bowled = [{"player_out": "A", "kind": "bowled"}]
    out = _derived([[_d(runs=4), _d(extras={"wides": 1}), _d(runs=1, wickets=bowled), _d()]])
    assert out["score_before"].tolist() == [0, 4, 5, 6]  # the wide's run counts towards the score
    assert out["wickets_before"].tolist() == [0, 0, 0, 1]
    assert out["legal_balls_remaining"].tolist() == [120, 119, 119, 118]  # the wide does not use a ball
    assert "ball_budget" not in out.columns
    assert list(out.columns) == parse.DELIVERY_COLUMNS


def test_derive_state_resets_per_innings():
    second = {"team": "Y", "target": {"overs": 20, "runs": 8},
              "overs": [{"over": 0, "deliveries": [_d(runs=6), _d(runs=1)]}]}  # fmt: skip
    out = _derived([[_d(runs=4)]], extra_innings=[second])
    assert out["score_before"].tolist() == [0, 0, 6]
    assert out["target"].isna().tolist() == [True, False, False]
    assert out["target"].dropna().tolist() == [8, 8]
    assert out["legal_balls_remaining"].tolist() == [120, 120, 119]


def test_match_row_competition_and_scope():
    raw = _raw([[_d()]], event={"name": "NatWest T20 Blast"})
    assert parse.parse_match(raw, "m")[0]["competition"] == "Vitality Blast"
    raw = _raw([[_d()]], match_type="IT20")
    assert parse.parse_match(raw, "m")[0]["competition"] == "T20I"
    assert parse.in_scope(raw)
    assert not parse.in_scope(_raw([[_d()]], balls_per_over=5))
    assert not parse.in_scope(_raw([[_d()]], overs=50))
    assert not parse.in_scope(_raw([[_d()]], match_type="ODI"))
    assert not parse.in_scope(_raw([[_d()]], gender="female"))


def test_season_year_is_calendar_year_of_first_date():
    match = parse.parse_match(_raw([[_d()]], dates=["2024-01-03", "2024-01-04"], season="2023/24"), "m")[0]
    assert (match["season_year"], match["season"], match["match_date"]) == (2024, "2023/24", "2024-01-03")
