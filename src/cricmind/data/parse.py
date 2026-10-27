"""Cricsheet JSON -> deliveries.parquet + matches.parquet (D-003, D-004, D-020..D-024).

uv run python -m cricmind.data.parse

`parse_match` is pure (one match dict -> rows) and holds every rule that is local to a delivery or an over;
`derive` adds the cumulative state columns with vectorised pandas.
"""

import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from cricmind import config

# D-004/D-023: dismissals that are not the bowler beating the batter; the ball falls back to its runs class.
NOT_STRIKER_WICKET = frozenset({"run out", "retired hurt", "retired out", "retired not out"})
# D-023: these do not count towards the fall of wickets in `wickets_before`.
NOT_A_FALL = frozenset({"retired hurt", "retired not out"})

DELIVERY_COLUMNS = [
    "match_id", "match_date", "competition", "match_type", "venue_id", "innings", "is_super_over", "over",
    "ball_seq", "legal_ball_in_over", "miscounted_balls", "batter_id", "bowler_id", "non_striker_id",
    "runs_batter", "runs_extras", "runs_total", "non_boundary", "wides", "noballs", "byes", "legbyes",
    "penalty", "is_legal", "n_wickets", "wicket_kind", "player_out_id", "outcome", "score_before",
    "wickets_before", "target", "legal_balls_remaining",
]  # fmt: skip
MATCH_COLUMNS = [
    "match_id", "match_date", "season_year", "season", "competition", "event_name", "match_type", "team_type",
    "venue_id", "venue", "city", "team1", "team2", "toss_winner", "toss_decision", "winner", "result",
    "method", "scheduled_overs", "has_super_over", "data_version",
]  # fmt: skip
NULLABLE_INT = ["legal_ball_in_over", "miscounted_balls", "target"]


def in_scope(raw: dict) -> bool:
    """D-020 filter. The 6-ball/20-over check is what removes The Hundred (its match_type is undocumented)."""
    info = raw["info"]
    return (
        info["match_type"] in config.MATCH_TYPES
        and info["gender"] == config.GENDER
        and info["balls_per_over"] == config.BALLS_PER_OVER
        and info.get("overs") == config.SCHEDULED_OVERS
    )


def _balls(overs: float | int, per_over: int) -> int:
    """`target.overs` is cricket notation (35.1 = 35 overs 1 ball), not a decimal; D-023."""
    whole, _, ball = str(overs).partition(".")
    return int(whole) * per_over + int(ball or 0)


def _competition(info: dict) -> tuple[str, str | None]:
    event = info.get("event")
    name = event.get("name") if isinstance(event, dict) else event
    if info["match_type"] == "IT20":
        return "T20I", name
    return config.COMPETITION_RENAMES.get(name, name) or "Unknown", name


def _outcome(is_legal: bool, runs_batter: int, non_boundary: bool, striker_out: bool) -> str | None:
    """D-023. Null for wides/no-balls; W only when the striker is out and it is not a run out/retirement."""
    if not is_legal:
        return None
    if striker_out:
        return "W"
    # D-023 literal: runs_batter == 6 is "6" even if non_boundary (all-run six; ~1 in 50k balls).
    if runs_batter == 6:
        return "6"
    if runs_batter == 4 and not non_boundary:
        return "4"
    if runs_batter >= 3:
        return "3+"
    return ("dot", "1", "2")[runs_batter]


def parse_match(raw: dict, match_id: str) -> tuple[dict, list[dict]]:
    """One in-scope match -> (matches row, delivery rows). Unresolved player name raises (D-003)."""
    info = raw["info"]
    people = info["registry"]["people"]
    venue_id = info.get("registry", {}).get("venues", {}).get(info.get("venue"))  # only v1.3.0 files carry it
    per_over = info["balls_per_over"]
    competition, event_name = _competition(info)
    outcome = info.get("outcome", {})

    def pid(name: str) -> str:
        try:
            return people[name]
        except KeyError:
            raise ValueError(f"{match_id}: no registry id for {name!r}") from None

    rows: list[dict] = []
    has_super_over = False
    # innings = 1-based file position, so super overs are 3+ (D-024)
    for innings_no, inn in enumerate(raw["innings"], start=1):
        is_super = bool(inn.get("super_over"))
        has_super_over |= is_super
        target = inn.get("target")
        budget = _balls(target["overs"], per_over) if target else info["overs"] * per_over
        if is_super:
            budget = per_over
        miscounted = {int(o): int(v["balls"]) for o, v in inn.get("miscounted_overs", {}).items()}
        ball_seq = 0
        for over in inn["overs"]:
            legal_in_over = 0
            for d in over["deliveries"]:
                ex, runs = d.get("extras", {}), d["runs"]
                wickets = d.get("wickets", [])
                wides, noballs = ex.get("wides", 0), ex.get("noballs", 0)
                is_legal = wides == 0 and noballs == 0
                non_boundary = bool(runs.get("non_boundary", False))
                legal_in_over += is_legal
                batter = pid(d["batter"])
                out_ids = [pid(w["player_out"]) for w in wickets]
                striker_out = any(
                    o == batter and w["kind"] not in NOT_STRIKER_WICKET
                    for o, w in zip(out_ids, wickets, strict=True)
                )
                rows.append({
                    "match_id": match_id,
                    "match_date": info["dates"][0],
                    "competition": competition,
                    "match_type": info["match_type"],
                    "venue_id": venue_id,
                    "innings": innings_no,
                    "is_super_over": is_super,
                    "over": over["over"],
                    "ball_seq": ball_seq,
                    "legal_ball_in_over": legal_in_over if is_legal else None,
                    "miscounted_balls": miscounted.get(over["over"]),
                    "batter_id": batter,
                    "bowler_id": pid(d["bowler"]),
                    "non_striker_id": pid(d["non_striker"]),
                    "runs_batter": runs["batter"],
                    "runs_extras": runs["extras"],
                    "runs_total": runs["total"],
                    "non_boundary": non_boundary,
                    "wides": wides,
                    "noballs": noballs,
                    "byes": ex.get("byes", 0),
                    "legbyes": ex.get("legbyes", 0),
                    "penalty": ex.get("penalty", 0),
                    "is_legal": is_legal,
                    "n_wickets": sum(w["kind"] not in NOT_A_FALL for w in wickets),
                    "wicket_kind": wickets[0]["kind"] if wickets else None,
                    "player_out_id": out_ids[0] if wickets else None,
                    "outcome": _outcome(is_legal, runs["batter"], non_boundary, striker_out),
                    "target": None if is_super or not target else target["runs"],
                    "ball_budget": budget,
                })  # fmt: skip
                ball_seq += 1

    teams = info.get("teams", [])
    toss = info.get("toss", {})
    match = {
        "match_id": match_id,
        "match_date": info["dates"][0],
        "season_year": int(info["dates"][0][:4]),  # calendar year of the match, not the season string (D-024)
        "season": str(info["season"]),
        "competition": competition,
        "event_name": event_name,
        "match_type": info["match_type"],
        "team_type": info["team_type"],
        "venue_id": venue_id,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": teams[0] if teams else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner") or outcome.get("eliminator") or outcome.get("bowl_out"),
        "result": outcome.get("result", "win"),
        "method": outcome.get("method"),
        "scheduled_overs": info["overs"],
        "has_super_over": has_super_over,
        "data_version": raw["meta"]["data_version"],
    }  # fmt: skip
    return match, rows


def derive(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Cumulative innings state, vectorised: score/wickets before the ball and legal balls left (D-023)."""
    df = deliveries.sort_values(["match_id", "innings", "ball_seq"], kind="stable").reset_index(drop=True)
    keys = [df["match_id"], df["innings"]]
    df["score_before"] = df["runs_total"].groupby(keys).cumsum() - df["runs_total"]
    df["wickets_before"] = df["n_wickets"].groupby(keys).cumsum() - df["n_wickets"]
    legal = df["is_legal"].astype("int64")
    df["legal_balls_remaining"] = df.pop("ball_budget") - (legal.groupby(keys).cumsum() - legal)
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    df = df[DELIVERY_COLUMNS]
    return df.astype({c: "Int64" for c in NULLABLE_INT})


def build(raw_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    """Read the pinned zip in place (D-020), write both parquet files, return counts for the data profile."""
    raw_dir = raw_dir or config.DATA_RAW / config.DATA_SNAPSHOT
    out_dir = out_dir or config.DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)

    matches, rows, counts = [], [], Counter()
    with zipfile.ZipFile(raw_dir / "all_male_json.zip") as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".json")):
            counts["files"] += 1
            raw = json.loads(zf.read(name))
            if not in_scope(raw):
                counts["dropped_scope"] += 1
                continue
            match, match_rows = parse_match(raw, Path(name).stem)
            if not match_rows:
                counts["dropped_no_deliveries"] += 1
                continue
            matches.append(match)
            rows.extend(match_rows)  # lean: one in-memory list (~2.5M dicts) - chunk if RAM is tight

    deliveries = derive(pd.DataFrame(rows))
    matches_df = pd.DataFrame(matches, columns=MATCH_COLUMNS)
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"]).dt.date
    deliveries.to_parquet(out_dir / "deliveries.parquet", index=False)
    matches_df.to_parquet(out_dir / "matches.parquet", index=False)
    return {**counts, "matches": len(matches_df), "deliveries": len(deliveries)}


if __name__ == "__main__":
    summary = build()
    report = config.REPORTS / "p1"
    report.mkdir(parents=True, exist_ok=True)
    (report / "parse_counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)
