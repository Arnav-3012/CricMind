"""Builds evals/agent_questions.jsonl: 30 agent questions, ground truth computed by SQL (Phase 6, plan.md).

uv run python -m cricmind.eval.agent_questions

Serves the matchup goal: 9 head-to-head lookups (observed batter-vs-bowler numbers), 4 what-if probability
questions (truth from the saved D-058 chain that the `matchup` tool serves: a model output, not a count), 9
single-player lookups the matchup answers build on, and 8 questions the data cannot answer (the agent
must refuse).
All counts are over `legal_deliveries` of every season in the snapshot (descriptive, not a held-out claim).
Refusal convention: an answer that cannot come from the data starts with `CANNOT_ANSWER:` (scored by string).
"""

import json

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.agent.serving import Serving
from cricmind.data.db import connect

OUT = config.ROOT / "evals" / "agent_questions.jsonl"
PROFILE = ["player_profile", "sql"]
MIN_PAIR_BALLS = 30
PAIR_POOL = 150
SLOTS = ["balls", "runs", "dismissals", "sr", "balls", "runs", "dismissals", "sr", "dismissals"]
WHATIF = [("powerplay", "W"), ("middle", "dot"), ("death", "6"), ("death", "W")]
CLASS_WORDS = {"W": "dismissed", "dot": "a dot ball", "6": "a six"}

REFUSE = [
    "Is {bat} a left-handed or right-handed batter?",
    "Is {bat2} left-handed or right-handed?",
    "Does {bowl} bowl pace or spin, and what is his bowling style?",
    "How much did {bat} go for in the last IPL auction?",
    "Will {bat} score a fifty in his next match?",
    "What will the pitch and weather be like at the next T20 World Cup final?",
    "Who has scored the most runs in women's T20 internationals?",
    "How many Test runs has {bat} scored?",
]


def names_unique(con) -> str:
    """Names collide (CLAUDE.md), so questions only use players whose name maps to one registry id."""
    return "p.name in (select name from players group by name having count(*) = 1)"


def top_players(con, role: str, n: int) -> pd.DataFrame:
    col = f"{role}_id"
    return con.execute(
        f"select d.{col} as id, p.name, count(*) as balls from legal_deliveries d "
        f"join players p on p.player_id = d.{col} where {names_unique(con)} "
        f"group by all order by balls desc limit {n}"
    ).df()


def pair_pool(con) -> pd.DataFrame:
    return con.execute(
        f"select d.batter_id, d.bowler_id, pb.name as bat, pw.name as bowl, count(*) as balls, "
        f"sum(d.runs_batter) as runs, sum((d.outcome = 'W')::int) as dismissals "
        f"from legal_deliveries d join players pb on pb.player_id = d.batter_id "
        f"join players pw on pw.player_id = d.bowler_id "
        f"where pb.name in (select name from players group by name having count(*) = 1) "
        f"and pw.name in (select name from players group by name having count(*) = 1) "
        f"group by all having balls >= {MIN_PAIR_BALLS} order by balls desc limit {PAIR_POOL}"
    ).df()


def rec(cat, question, truth, tol, source, tools, ids, sql=None, refuse=False) -> dict:
    return {
        "category": cat,
        "question": question,
        "truth": truth,
        "tolerance": tol,
        "truth_source": source,
        "acceptable_first_tools": tools,
        "player_ids": ids,
        "sql": sql,
        "refuse": refuse,
    }


def one(con, sql: str, *params):
    return con.execute(sql, list(params)).fetchone()


def lookups(con, bats: pd.DataFrame, bowls: pd.DataFrame) -> list[dict]:
    out = []
    for _, b in bats.iloc[[0, 3]].iterrows():
        sql = "select sum(runs_batter) from legal_deliveries where batter_id = ?"
        out.append(
            rec(
                "lookup",
                f"How many runs has {b['name']} scored off legal deliveries in the dataset?",
                int(one(con, sql, b["id"])[0]),
                0,
                "sql",
                ["player_profile", "sql"],
                [b["id"]],
                sql,
            )
        )
    for _, w in bowls.iloc[[1, 4]].iterrows():
        sql = "select count(*) from legal_deliveries where bowler_id = ? and outcome = 'W'"
        out.append(
            rec(
                "lookup",
                f"How many batter dismissals (excluding run outs) has {w['name']} taken?",
                int(one(con, sql, w["id"])[0]),
                0,
                "sql",
                ["player_profile", "sql"],
                [w["id"]],
                sql,
            )
        )
    b = bats.iloc[5]
    sql = "select 100.0 * sum(runs_batter) / count(*) from legal_deliveries where batter_id = ?"
    out.append(
        rec(
            "lookup",
            f"What is {b['name']}'s strike rate off legal deliveries?",
            round(one(con, sql, b["id"])[0], 2),
            0.1,
            "sql",
            PROFILE,
            [b["id"]],
            sql,
        )
    )
    b = bats.iloc[8]
    sql = "select 100.0 * avg((outcome in ('4', '6'))::int) from legal_deliveries where batter_id = ?"
    out.append(
        rec(
            "lookup",
            f"What percentage of the legal balls {b['name']} faces go for four or six?",
            round(one(con, sql, b["id"])[0], 2),
            0.1,
            "sql",
            PROFILE,
            [b["id"]],
            sql,
        )
    )
    w = bowls.iloc[7]
    sql = "select 100.0 * avg((outcome = 'dot')::int) from legal_deliveries where bowler_id = ?"
    out.append(
        rec(
            "lookup",
            f"What percentage of {w['name']}'s legal balls are dot balls?",
            round(one(con, sql, w["id"])[0], 2),
            0.1,
            "sql",
            PROFILE,
            [w["id"]],
            sql,
        )
    )
    sql = (
        "select d.bowler_id, p.name, count(*) as w from legal_deliveries d "
        "join matches m using (match_id) join players p on p.player_id = d.bowler_id "
        "where d.outcome = 'W' and m.season_year = 2024 group by all order by w desc limit 1"
    )
    top = con.execute(sql).fetchone()
    out.append(
        rec(
            "lookup",
            "Which bowler took the most batter dismissals (excluding run outs) in 2024?",
            top[1],
            0,
            "sql",
            ["top_players", "sql"],
            [top[0]],
            sql,
        )
    )
    sql = (
        "select d.batter_id, p.name, sum(d.runs_batter) as r from legal_deliveries d "
        "join matches m using (match_id) join players p on p.player_id = d.batter_id "
        "where m.season_year = 2024 group by all order by r desc limit 1"
    )
    top = con.execute(sql).fetchone()
    out.append(
        rec(
            "lookup",
            "Which batter scored the most runs off legal balls in 2024?",
            top[1],
            0,
            "sql",
            ["top_players", "sql"],
            [top[0]],
            sql,
        )
    )
    return out


def matchup_obs(pairs: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    out, left = [], pairs.copy()
    for i, slot in enumerate(SLOTS):
        pool = left[left["dismissals"] >= 2] if slot == "dismissals" else left
        row = pool.sample(1, random_state=config.SEED + i).iloc[0]
        left = left.drop(row.name)
        who = f"{row['bat']} against {row['bowl']}"
        ids = [row["batter_id"], row["bowler_id"]]
        q, truth, tol = {
            "balls": (f"How many legal balls has {who} faced?", int(row["balls"]), 0),
            "runs": (f"How many runs has {who} scored off legal balls?", int(row["runs"]), 0),
            "dismissals": (
                f"How many times has {row['bat']} been dismissed by {row['bowl']} (excluding run outs)?",
                int(row["dismissals"]),
                0,
            ),
            "sr": (f"What is the strike rate of {who}?", round(100 * row["runs"] / row["balls"], 2), 0.1),
        }[slot]
        sql = (
            "select count(*), sum(runs_batter), sum((outcome = 'W')::int) from legal_deliveries "
            "where batter_id = ? and bowler_id = ?"
        )
        out.append(rec("matchup", q, truth, tol, "sql", ["matchup", "sql"], ids, sql))
    return out, left


def whatif(left: pd.DataFrame, srv: Serving) -> list[dict]:
    """D-058(7): the truth is the chain probability the tool returns; it tests quoting, not model accuracy."""
    out = []
    for i, (phase, cls) in enumerate(WHATIF):
        row = left.sample(1, random_state=config.SEED + 100 + i).iloc[0]
        left = left.drop(row.name)
        q = (
            f"What is the model's probability that {row['bat']} is {CLASS_WORDS[cls]} on a given ball "
            f"against {row['bowl']} in the {phase} overs? Give it as a fraction."
        )
        frame = pd.DataFrame([{"batter_id": row["batter_id"], "bowler_id": row["bowler_id"], "phase": phase}])
        p = srv.chain_proba(frame)[0][0, config.OUTCOME_CLASSES.index(cls)]
        out.append(
            rec(
                "what-if",
                q,
                round(float(p), 4),
                0.01,
                f"tool:matchup:{phase}:{cls}",
                ["matchup"],
                [row["batter_id"], row["bowler_id"]],
            )
        )
    return out


def refusals(bats: pd.DataFrame, bowls: pd.DataFrame) -> list[dict]:
    fills = {"bat": bats.iloc[0]["name"], "bat2": bats.iloc[2]["name"], "bowl": bowls.iloc[0]["name"]}
    return [rec("refuse", t.format(**fills), None, None, None, [], [], refuse=True) for t in REFUSE]


def main() -> None:
    con = connect()
    bats, bowls = top_players(con, "batter", 30), top_players(con, "bowler", 30)
    obs, left = matchup_obs(pair_pool(con))
    rows = lookups(con, bats, bowls) + obs + whatif(left, Serving()) + refusals(bats, bowls)
    for i, r in enumerate(rows):
        r["id"] = f"q{i + 1:02d}"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    counts = pd.Series([r["category"] for r in rows]).value_counts().to_dict()
    print(f"wrote {len(rows)} questions to {OUT}: {counts}; seed {config.SEED}")
    assert np.isfinite([r["truth"] for r in rows if isinstance(r["truth"], float)]).all()


if __name__ == "__main__":
    main()
