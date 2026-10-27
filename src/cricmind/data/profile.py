"""Data profile for the Phase 1 gate -> reports/p1/data_profile.json (all numbers come from code).

uv run python -m cricmind.data.profile
"""

import json
from pathlib import Path

import duckdb

from cricmind import config

# Phase boundaries (architecture.md context features): powerplay 0-5, middle 6-14, death 15-19.
PHASE_SQL = "case when over < 6 then 'powerplay' when over < 15 then 'middle' else 'death' end"
MODELLED = "is_legal and not is_super_over"  # the rows every later phase trains and evaluates on
TOP_COMPETITIONS = 40


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def _snapshot(raw_dir: Path) -> dict:
    manifest = raw_dir / "MANIFEST.json"
    return json.loads(manifest.read_text()) if manifest.exists() else {"snapshot": config.DATA_SNAPSHOT}


def build_profile(processed_dir: Path | None = None, raw_dir: Path | None = None) -> dict:
    processed_dir = processed_dir or config.DATA_PROCESSED
    raw_dir = raw_dir or config.DATA_RAW / config.DATA_SNAPSHOT
    con = duckdb.connect()  # in memory; the parquet files are only read
    con.execute(
        f"create view d as select *, {PHASE_SQL} as phase "
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
        top.append({
            "competition": f"other ({len(rest)} competitions)",
            "matches": sum(r["matches"] for r in rest),
            "legal_balls": sum(r["legal_balls"] for r in rest),
        })

    share = "cast(count(*) * 1.0 / sum(count(*)) over ({}) as double)"
    return {
        "snapshot": _snapshot(raw_dir),
        "parse_counts": json.loads((config.REPORTS / "p1" / "parse_counts.json").read_text())
        if (config.REPORTS / "p1" / "parse_counts.json").exists() else None,
        "totals": _rows(
            con,
            f"""select (select count(*) from m) matches, count(*) deliveries,
                       count(*) filter (where is_legal) legal_balls,
                       count(*) filter (where {MODELLED}) modelled_legal_balls,
                       count(*) filter (where is_super_over) super_over_rows from d""",
        )[0],
        "per_year": _rows(
            con,
            f"""select year(match_date) as year, count(distinct match_id) matches,
                       count(*) filter (where {MODELLED}) legal_balls from d group by 1 order by 1""",
        ),
        "per_competition": top,
        "outcome_share_overall": _rows(
            con,
            f"""select outcome, count(*) n, {share.format('')} share from d
                where {MODELLED} group by 1 order by n desc""",
        ),
        "outcome_share_by_phase": _rows(
            con,
            f"""select phase, outcome, count(*) n, {share.format('partition by phase')} share from d
                where {MODELLED} group by 1, 2 order by 1, n desc""",
        ),
        # D-004 revisit trigger: is the run-out share of dismissals large in some phase?
        "run_out_share_by_phase": _rows(
            con,
            """select phase, count(*) dismissals, count(*) filter (where wicket_kind = 'run out') run_outs,
                      cast(count(*) filter (where wicket_kind = 'run out') * 1.0 / count(*) as double) share
               from d where n_wickets > 0 and not is_super_over group by 1 order by 1""",
        ),
        "venue_id_null": _rows(
            con,
            """select cast(count(*) filter (where venue_id is null) * 1.0 / count(*) as double) match_share,
                      count(*) filter (where venue_id is null) matches_null, count(*) matches from m""",
        )[0],
        "checks": {
            "balls_over_budget": _rows(
                con, "select count(*) n from d where legal_balls_remaining < 0"
            )[0]["n"],
            # Reported only: the over-length invariant was dropped (logs.md 2026-10-28).
            "unflagged_overs_over_6_legal_balls": _rows(
                con,
                """select count(*) n from (
                       select 1 from d group by match_id, innings, over
                       having count(*) filter (where is_legal) > 6
                          and (max(miscounted_balls) is null
                               or max(miscounted_balls) != count(*) filter (where is_legal)))""",
            )[0]["n"],
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
