"""Data profile for the Phase 1 gate -> reports/p1/data_profile.json (all numbers come from code).

uv run python -m cricmind.data.profile
"""

import json
from pathlib import Path

import duckdb

from cricmind import config
from cricmind.data.download import MANIFEST

# Phase boundaries (architecture.md context features): powerplay 0-5, middle 6-14, death 15-19.
PHASE_SQL = "case when over < 6 then 'powerplay' when over < 15 then 'middle' else 'death' end"
# D-005: split by match year (config.py). Test rows are only counted; design figures use TRAIN rows.
SPLIT_SQL = (
    f"case when year(match_date) <= {config.TRAIN_END_YEAR} then 'train' "
    f"when year(match_date) = {config.VAL_YEAR} then 'val' "
    f"when year(match_date) between {config.TEST_START_YEAR} and {config.TEST_END_YEAR} then 'test' "
    "else 'other' end"
)
MODELLED = "is_legal and not is_super_over"  # the rows every later phase trains and evaluates on
TRAIN = "split = 'train'"  # the only rows allowed to inform a design decision (D-026, the D-004 trigger)
TOP_COMPETITIONS = 40


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def _snapshot(raw_dir: Path) -> dict:
    return json.loads((raw_dir / MANIFEST).read_text())  # D-021: a profile without its pin is not a report


def build_profile(processed_dir: Path | None = None, raw_dir: Path | None = None) -> dict:
    processed_dir = processed_dir or config.DATA_PROCESSED
    raw_dir = raw_dir or config.DATA_RAW / config.DATA_SNAPSHOT
    con = duckdb.connect()  # in memory; the parquet files are only read
    con.execute(
        f"create view d as select *, {PHASE_SQL} as phase, {SPLIT_SQL} as split "
        f"from read_parquet('{processed_dir / 'deliveries.parquet'}')"
    )
    con.execute(f"create view m as select * from read_parquet('{processed_dir / 'matches.parquet'}')")

    by_competition = _rows(
        con,
        f"""select competition, count(distinct match_id) matches,
                   count(*) filter (where {MODELLED}) legal_balls
            from d group by 1 order by legal_balls desc""",
    )
    rest = by_competition[TOP_COMPETITIONS:]
    top = by_competition[:TOP_COMPETITIONS]
    if rest:
        top.append(
            {
                "competition": f"other ({len(rest)} competitions)",
                "matches": sum(r["matches"] for r in rest),
                "legal_balls": sum(r["legal_balls"] for r in rest),
            }
        )

    share = "cast(count(*) * 1.0 / sum(count(*)) over ({}) as double)"
    return {
        "snapshot": _snapshot(raw_dir),
        "parse_counts": json.loads((config.REPORTS / "p1" / "parse_counts.json").read_text())
        if (config.REPORTS / "p1" / "parse_counts.json").exists()
        else None,
        "totals": _rows(
            con,
            f"""select (select count(*) from m) matches, count(*) deliveries,
                       count(*) filter (where is_legal) legal_balls,
                       count(*) filter (where {MODELLED}) modelled_legal_balls,
                       count(*) filter (where is_super_over) super_over_rows,
                       (select count(*) from m where is_reduced) reduced_matches,
                       count(*) filter (where {MODELLED} and is_reduced) reduced_modelled_balls,
                       (select count(*) from m where method is not null) matches_with_method from d""",
        )[0],
        "rows_by_split": _rows(
            con,
            f"""select split, count(distinct match_id) matches,
                       count(*) filter (where {MODELLED}) legal_balls
                from d group by 1 order by 1""",
        ),
        "per_year": _rows(
            con,
            f"""select year(match_date) as year, count(distinct match_id) matches,
                       count(*) filter (where {MODELLED}) legal_balls from d group by 1 order by 1""",
        ),
        "per_competition": top,
        "outcome_share_overall": _rows(
            con,
            f"""select outcome, count(*) n, {share.format("")} share from d
                where {MODELLED} and {TRAIN} group by 1 order by n desc""",
        ),
        "outcome_share_by_phase": _rows(
            con,
            f"""select phase, outcome, count(*) n, {share.format("partition by phase")} share from d
                where {MODELLED} and {TRAIN} group by 1, 2 order by 1, n desc""",
        ),
        # D-004 revisit trigger: is the run-out share of dismissals large in some phase? Train rows only.
        "run_out_share_by_phase": _rows(
            con,
            f"""select phase, count(*) dismissals, count(*) filter (where wicket_kind = 'run out') run_outs,
                      cast(count(*) filter (where wicket_kind = 'run out') * 1.0 / count(*) as double) share
               from d where n_wickets > 0 and not is_super_over and {TRAIN} group by 1 order by 1""",
        ),
        # D-026: where the run-out hazard sits (over 19 is the death-over spike). Train rows only.
        "run_out_by_over": _rows(
            con,
            f"""select over, legal_balls, run_outs,
                       cast(run_outs * 1000.0 / legal_balls as double) run_outs_per_1000_balls
                from (select over, count(*) filter (where {MODELLED}) legal_balls,
                             count(*) filter (where wicket_kind = 'run out'
                                              and n_wickets > 0 and not is_super_over) run_outs
                      from d where {TRAIN} group by 1)
                order by over""",
        ),
        # D-026: striker vs non-striker run-outs (the pair-level signal D-026 dropped). Train rows only.
        "run_out_by_role": _rows(
            con,
            f"""select count(*) filter (where player_out_id = batter_id) striker_run_outs,
                      count(*) filter (where player_out_id = non_striker_id) non_striker_run_outs
               from d where {TRAIN} and wicket_kind = 'run out' and n_wickets > 0 and not is_super_over""",
        )[0],
        "venue_id_null": _rows(
            con,
            """select cast(count(*) filter (where venue_id is null) * 1.0 / count(*) as double) match_share,
                      count(*) filter (where venue_id is null) matches_null, count(*) matches from m""",
        )[0],
        # D-027: how far the alias file got. `venue_names_in_several_cities` is the review list: a name that
        # is really one ground under two cities needs an alias row, a real clash (County Ground) does not.
        "venues": _rows(
            con,
            """select count(distinct venue_key) venue_keys,
                      (select count(*) from (select distinct venue, city from m)) raw_venue_city_pairs,
                      count(distinct venue_key) filter (where venue_key like '%|') keys_without_city,
                      count(distinct venue_id) registry_ids from m""",
        )[0],
        "venue_names_in_several_cities": _rows(
            con,
            """select split_part(venue_key, '|', 1) venue,
                      list_sort(list(distinct split_part(venue_key, '|', 2))) cities
               from m group by 1 having count(distinct venue_key) > 1 order by 1""",
        ),
        "checks": {
            "balls_over_budget": _rows(con, "select count(*) n from d where legal_balls_remaining < 0")[0][
                "n"
            ],
            # Reported only: the over-length invariant was dropped (logs.md 2026-10-28).
            "unflagged_overs_over_6_legal_balls": _rows(
                con,
                """select count(*) n from (
                       select 1 from d group by match_id, innings, over
                       having count(*) filter (where is_legal) > 6
                          and (max(miscounted_balls) is null
                               or max(miscounted_balls) != count(*) filter (where is_legal)))""",
            )[0]["n"],
            "duplicate_keys": _rows(
                con,
                """select count(*) n from (select match_id, innings, ball_seq from d
                   group by 1, 2, 3 having count(*) > 1)""",
            )[0]["n"],
            # invariant: outcome is null exactly when the ball is not legal (D-023)
            "outcome_legal_mismatch": _rows(
                con, "select count(*) n from d where (outcome is null) = is_legal"
            )[0]["n"],
            # D-030: over and ball in over are kept on is_reduced rows, so their range must be clean in both
            # groups. Ball in over can reach 7 (a miscounted over) and is never null on a modelled row.
            "ball_position_by_reduced": _rows(
                con,
                f"""select is_reduced, count(*) n, min(over) min_over, max(over) max_over,
                          min(legal_ball_in_over) min_ball, max(legal_ball_in_over) max_ball,
                          count(*) filter (where legal_ball_in_over is null) null_ball
                   from d where {MODELLED} group by 1 order by 1""",
            ),
            # invariant: runs_batter + runs_extras = runs_total on every ball (test_invariants caps the rate)
            "ball_runs_mismatch": _rows(
                con, "select count(*) n from d where runs_batter + runs_extras != runs_total"
            )[0]["n"],
            # invariant: 1st innings total + 1 = target on unreduced chases (test_invariants caps the rate)
            "target_cross_check": _rows(
                con,
                """select count(*) checked, count(*) filter (where total + 1 != target) mismatched,
                          cast(count(*) filter (where total + 1 != target) * 1.0 / nullif(count(*), 0)
                               as double) mismatch_rate
                   from (select i1.total, i2.target from
                            (select match_id, sum(runs_total) total from d
                             where innings = 1 and not is_super_over group by 1) i1
                         join (select match_id, any_value(target) target from d
                               where innings = 2 and not is_super_over group by 1) i2 using (match_id)
                         join m using (match_id)
                         where not m.is_reduced and i2.target is not null)""",
            )[0],
            "dismissal_kinds": _rows(
                con,
                """select wicket_kind, count(*) n from d where wicket_kind is not null
                   group by 1 order by n desc""",
            ),
        },
    }


if __name__ == "__main__":
    profile = build_profile()
    out = config.REPORTS / "p1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_profile.json").write_text(json.dumps(profile, indent=2, default=float) + "\n")
    t = profile["totals"]
    print(f"matches={t['matches']} modelled_legal_balls={t['modelled_legal_balls']} -> {out}")
