"""Builds evals/agent_questions.jsonl: 30 agent questions, ground truth computed by SQL (Phase 6, D-060).

uv run python -m cricmind.eval.agent_questions

Serves the matchup goal. Every question names one tool it must start with, so each tool is tested:
- 12 single-player lookups: 5 `player_profile`, 4 `top_players` (leaderboards), 3 `sql` (no typed tool fits)
- 8 head-to-head lookups (`matchup`, the observed block: counts from the data)
- 4 what-if probabilities (`matchup`, the model block; truth from the saved D-058 chain, a model output)
- 6 questions the data cannot answer (the agent must refuse)
A few questions use a bare surname, as users type; those surnames match exactly one player with legal balls.
`archetype_of`, `predict_next_ball` and mixed two-tool questions are in `tool_questions.py`.
All counts are over `legal_deliveries` of every season in the snapshot (descriptive, not a held-out claim).
Refusal convention: an answer that cannot come from the data starts with `CANNOT_ANSWER:` (scored by string).
"""

import json

import pandas as pd

from cricmind import config
from cricmind.agent.serving import Serving
from cricmind.data.db import connect

OUT = config.ROOT / "evals" / "agent_questions.jsonl"
MIN_PAIR_BALLS = 30
PAIR_POOL = 150
RATE_MIN_BALLS = 300  # the top_players default for rate metrics
ENOUGH_BALLS = f"having count(*) >= {RATE_MIN_BALLS} "
SLOTS = ["balls", "runs", "dismissals", "sr", "balls", "runs", "dismissals", "sr"]
WHATIF = [("powerplay", "W"), ("middle", "dot"), ("death", "6"), ("death", "W")]
CLASS_WORDS = {"W": "dismissed", "dot": "a dot ball", "6": "a six"}

REFUSE = [
    "Is {bat} a left-handed or right-handed batter?",
    "Does {bowl} bowl pace or spin, and what is his bowling style?",
    "How much did {bat} go for in the last IPL auction?",
    "Will {bat} score a fifty in his next match?",
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
        f"group by all order by balls desc, id limit {n}"  # id breaks ties: the order must not vary by run
    ).df()


def pair_pool(con) -> pd.DataFrame:
    return con.execute(
        f"select d.batter_id, d.bowler_id, pb.name as bat, pw.name as bowl, count(*) as balls, "
        f"sum(d.runs_batter) as runs, sum((d.outcome = 'W')::int) as dismissals "
        f"from legal_deliveries d join players pb on pb.player_id = d.batter_id "
        f"join players pw on pw.player_id = d.bowler_id "
        f"where pb.name in (select name from players group by name having count(*) = 1) "
        f"and pw.name in (select name from players group by name having count(*) = 1) "
        f"group by all having balls >= {MIN_PAIR_BALLS} "
        f"order by balls desc, d.batter_id, d.bowler_id limit {PAIR_POOL}"
    ).df()


def rec(cat, question, truth, tol, source, tools, ids, sql=None, refuse=False, required=None) -> dict:
    return {
        "category": cat,
        "question": question,
        "truth": truth,
        "tolerance": tol,
        "truth_source": source,
        "acceptable_first_tools": tools,
        "required_tools": required or [],  # set only for questions that need several tools
        "player_ids": ids,
        "sql": sql,
        "refuse": refuse,
    }


def one(con, sql: str, *params):
    return con.execute(sql, list(params)).fetchone()


def shortener(con):
    """name -> bare surname when exactly one player with legal balls contains it (so a user typing only the
    surname gets one match), else the registry name."""
    low = (
        con.execute(
            "select distinct lower(name) from players where player_id in "
            "(select batter_id from legal_deliveries union select bowler_id from legal_deliveries)"
        )
        .df()
        .iloc[:, 0]
    )

    def short(name: str) -> str:
        surname = name.split()[-1]
        return surname if low.str.contains(surname.lower(), regex=False).sum() == 1 else name

    return short


def lookups(con, bats: pd.DataFrame, bowls: pd.DataFrame, short) -> list[dict]:
    out = []

    def add(question, sql, params, tools, ids, tol=0):
        value = one(con, sql, *params)[0]
        truth = int(value) if tol == 0 else round(float(value), 2)
        out.append(rec("lookup", question, truth, tol, "sql", tools, ids, sql))

    def add_top(question, sql, tools):
        top = one(con, sql)
        out.append(rec("lookup", question, top[1], 0, "sql", tools, [top[0]], sql))

    profile, topn, sqlonly = ["player_profile"], ["top_players"], ["sql"]
    b = bats.iloc[0]
    add(
        f"How many runs has {b['name']} scored off legal deliveries in the dataset?",
        "select sum(runs_batter) from legal_deliveries where batter_id = ?",
        [b["id"]], profile, [b["id"]],
    )  # fmt: skip
    w = bowls.iloc[1]
    add(
        f"How many batter dismissals (excluding run outs) has {w['name']} taken?",
        "select count(*) from legal_deliveries where bowler_id = ? and outcome = 'W'",
        [w["id"]], profile, [w["id"]],
    )  # fmt: skip
    b = bats.iloc[5]
    add(
        f"What is {short(b['name'])}'s strike rate off legal deliveries?",
        "select 100.0 * sum(runs_batter) / count(*) from legal_deliveries where batter_id = ?",
        [b["id"]], profile, [b["id"]], 0.1,
    )  # fmt: skip
    b = bats.iloc[8]
    add(
        f"What percentage of the legal balls {short(b['name'])} faces go for four or six?",
        "select 100.0 * avg((outcome in ('4', '6'))::int) from legal_deliveries where batter_id = ?",
        [b["id"]], profile, [b["id"]], 0.1,
    )  # fmt: skip
    w = bowls.iloc[7]
    add(
        f"What percentage of {short(w['name'])}'s legal balls are dot balls?",
        "select 100.0 * avg((outcome = 'dot')::int) from legal_deliveries where bowler_id = ?",
        [w["id"]], profile, [w["id"]], 0.1,
    )  # fmt: skip

    def leaderboard(select, role, order, year, having=""):
        return (
            f"select d.{role}_id, p.name from legal_deliveries d join matches m using (match_id) "
            f"join players p on p.player_id = d.{role}_id where {select}m.season_year = {year} "
            f"group by all {having}order by {order} desc, d.{role}_id limit 1"
        )

    add_top(
        "Which bowler took the most batter dismissals (excluding run outs) in 2024?",
        leaderboard("d.outcome = 'W' and ", "bowler", "count(*)", 2024), topn,
    )  # fmt: skip
    add_top(
        "Which batter scored the most runs off legal balls in 2024?",
        leaderboard("", "batter", "sum(d.runs_batter)", 2024), topn,
    )  # fmt: skip
    add_top(
        f"Which batter has the highest strike rate in 2024 among those who faced at least {RATE_MIN_BALLS} "
        "legal balls?",
        leaderboard("", "batter", "sum(d.runs_batter) * 1.0 / count(*)", 2024, ENOUGH_BALLS),
        topn,
    )  # fmt: skip
    add_top(
        f"Which bowler has the highest dot-ball percentage in 2023 among those who bowled at least "
        f"{RATE_MIN_BALLS} legal balls?",
        leaderboard("", "bowler", "avg((d.outcome = 'dot')::int)", 2023, ENOUGH_BALLS),
        topn,
    )  # fmt: skip

    six = "select count(*) from legal_deliveries d join matches m using (match_id) where d.batter_id = ? "
    six += "and d.outcome = '6' and m.season_year = 2023"
    active = (r for _, r in bats.iloc[10:20].iterrows() if one(con, six, r["id"])[0] >= 5)
    b = next(active)  # the first top batter who hit sixes in 2023
    add(f"How many sixes did {b['name']} hit in 2023?", six, [b["id"]], sqlonly, [b["id"]])
    w = bowls.iloc[10]
    add(
        f"In how many matches has {w['name']} bowled at least one legal ball?",
        "select count(distinct match_id) from legal_deliveries where bowler_id = ?",
        [w["id"]], sqlonly, [w["id"]],
    )  # fmt: skip
    b = bats.iloc[20]
    comp = one(
        con,
        "select m.competition from legal_deliveries d join matches m using (match_id) where d.batter_id = ? "
        "group by 1 order by count(*) desc, 1 limit 1",
        b["id"],
    )[0]
    add(
        f"How many runs has {b['name']} scored in {comp}?",
        "select sum(d.runs_batter) from legal_deliveries d join matches m using (match_id) "
        "where d.batter_id = ? and m.competition = ?",
        [b["id"], comp], sqlonly, [b["id"]],
    )  # fmt: skip
    return out


def matchup_obs(pairs: pd.DataFrame, short) -> tuple[list[dict], pd.DataFrame]:
    out, left = [], pairs.copy()
    sql = (
        "select count(*), sum(runs_batter), sum((outcome = 'W')::int) from legal_deliveries "
        "where batter_id = ? and bowler_id = ?"
    )
    for i, slot in enumerate(SLOTS):
        pool = left[left["dismissals"] >= 2] if slot == "dismissals" else left
        row = pool.sample(1, random_state=config.SEED + i).iloc[0]
        left = left.drop(row.name)
        bat, bowl = (short(row["bat"]), short(row["bowl"])) if i % 4 == 0 else (row["bat"], row["bowl"])
        who = f"{bat} against {bowl}"
        ids = [row["batter_id"], row["bowler_id"]]
        q, truth, tol = {
            "balls": (f"How many legal balls has {who} faced?", int(row["balls"]), 0),
            "runs": (f"How many runs has {who} scored off legal balls?", int(row["runs"]), 0),
            "dismissals": (
                f"How many times has {bat} been dismissed by {bowl} (excluding run outs)?",
                int(row["dismissals"]),
                0,
            ),
            "sr": (f"What is the strike rate of {who}?", round(100 * row["runs"] / row["balls"], 2), 0.1),
        }[slot]
        out.append(rec("matchup", q, truth, tol, "sql", ["matchup"], ids, sql))
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
    fills = {"bat": bats.iloc[0]["name"], "bowl": bowls.iloc[0]["name"]}
    return [rec("refuse", t.format(**fills), None, None, None, [], [], refuse=True) for t in REFUSE]


def main() -> None:
    con = connect()
    short = shortener(con)
    bats, bowls = top_players(con, "batter", 30), top_players(con, "bowler", 30)
    obs, left = matchup_obs(pair_pool(con), short)
    rows = lookups(con, bats, bowls, short) + obs + whatif(left, Serving()) + refusals(bats, bowls)
    for i, r in enumerate(rows):
        r["id"] = f"q{i + 1:02d}"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    counts = pd.Series([r["category"] for r in rows]).value_counts().to_dict()
    print(f"wrote {len(rows)} questions to {OUT}: {counts}; seed {config.SEED}")


if __name__ == "__main__":
    main()
