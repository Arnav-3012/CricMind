"""Builds evals/agent_questions_tools.jsonl: 20 questions for the rest of the tools (D-059, D-060).

uv run python -m cricmind.eval.tool_questions

The 30 main questions never call `archetype_of` or `predict_next_ball`, so a model could break them without
moving the 80% gate. Truths for the two come from the saved D-058 artifact (the files the tools read), so
they test whether the agent calls the tool and quotes its numbers, not whether the model is accurate.
- 7 archetype: 3 ask how X plays (scored on quoting any of the bin's trait numbers; the description is printed
  for a human), 2 ask for train balls, 2 for the bin size.
- 7 next-ball: a full situation in the question, truth = the model's probability for it.
- 6 mixed: each needs two tools (`required_tools`) and every listed number must appear in the answer.
"""

import json

import pandas as pd

from cricmind import config
from cricmind.agent.serving import Serving
from cricmind.data.db import connect
from cricmind.eval.agent_questions import names_unique, one, pair_pool, rec

OUT = config.ROOT / "evals" / "agent_questions_tools.jsonl"
MIN_BALLS = 300  # clustered players only: they carry a bin description
READ_TOL = 0.05  # trait z-scores and means are quoted to 2 decimals
# (innings, over, score_before, wickets_before, target, class); over is 0-based like the tool's
SCENARIOS = [
    (1, 2, 18, 0, None, "W"), (1, 8, 60, 2, None, "dot"), (1, 17, 140, 4, None, "6"),
    (2, 3, 25, 0, 170, "4"), (2, 9, 70, 2, 165, "dot"), (2, 14, 110, 4, 175, "W"), (2, 19, 165, 6, 185, "W"),
]  # fmt: skip
ARCH = ["archetype_of"]
SR_SQL = "select 100.0 * sum(runs_batter) / count(*) from legal_deliveries where batter_id = ?"
MATCHES_SQL = "select count(distinct match_id) from legal_deliveries where bowler_id = ?"
TOP_2024 = (
    "select d.{role}_id, p.name, {agg} from legal_deliveries d join matches m using (match_id) "
    "join players p on p.player_id = d.{role}_id where {cond}m.season_year = 2024 "
    "group by all order by {agg} desc, d.{role}_id limit 1"
)
EVENTS = {"W": "is dismissed", "dot": "faces a dot ball", "6": "hits a six", "4": "hits a four"}


def labelled(con, srv: Serving, role: str, n: int, seed: int) -> pd.DataFrame:
    """n clustered players of `role` with a unique name, sampled from a pool in a fixed order."""
    names = con.execute(f"select p.player_id, p.name from players p where {names_unique(con)}").df()
    lab = srv.labels
    lab = lab[(lab["role"] == role) & (lab["placed"] == "clustered") & (lab["train_balls"] >= MIN_BALLS)]
    pool = lab.merge(names.drop_duplicates("player_id"), on="player_id").sort_values("player_id")
    return pool.sample(n, random_state=seed)


def archetype_questions(con, srv: Serving) -> list[dict]:
    bats = labelled(con, srv, "batter", 4, config.SEED + 300)
    bowls = labelled(con, srv, "bowler", 3, config.SEED + 301)
    bat_kinds = ["read", "read", "balls", "size"]
    plan = [  # (role, player row, kind)
        *[("batter", p, k) for p, k in zip(bats.itertuples(), bat_kinds, strict=True)],
        *[("bowler", p, k) for p, k in zip(bowls.itertuples(), ["read", "balls", "size"], strict=True)],
    ]
    out = []
    for role, p, kind in plan:
        info = srv.archetype(p.player_id)[role]
        who = f"{p.name} as a {role}"
        if kind == "read":
            # generic on purpose: a yes/no about a chosen trait may not be among the bin's 3 shown traits
            q = f"How does {p.name} typically play as a {role}, going by his archetype? Quote the numbers."
            truth = [round(v[k], 2) for v in info["bin_traits"].values() for k in ("z", "mean")]
            r = rec("archetype", q, truth, READ_TOL, "tool:archetype_of:traits", ARCH, [p.player_id])
            r["rubric"] = (
                "bin traits "
                + ", ".join(f"{f} z={v['z']} mean={v['mean']}" for f, v in info["bin_traits"].items())
                + " (judge whether the description fits them)"
            )
        elif kind == "balls":
            q = f"How many legal balls does the archetype model have for {who} in its training seasons?"
            r = rec("archetype", q, info["train_balls"], 0, "tool:archetype_of:balls", ARCH, [p.player_id])
        else:
            q = f"How many clustered players are in the same {role} archetype bin as {p.name}?"
            size = info["clustered_players_in_bin"]
            r = rec("archetype", q, size, 0, "tool:archetype_of:size", ARCH, [p.player_id])
        out.append(r)
    return out


def situation_prob(srv: Serving, pair, innings, over, score, wickets, target, cls) -> float:
    given = {"score_before": score, "wickets_before": wickets} | ({"target": target} if target else {})
    frame, _ = srv.state_frame(pair["batter_id"], pair["bowler_id"], over, innings, None, **given)
    return round(float(srv.next_ball_proba(frame)[0][config.OUTCOME_CLASSES.index(cls)]), 4)


def situation_text(innings, over, score, wickets, target) -> str:
    chase = f", chasing {target}" if target else ""
    where = f"innings {innings}, over {over} (0 is the first over)"
    return f"{where}, {wickets} wickets down, score {score}{chase}"


def next_ball_questions(con, srv: Serving) -> list[dict]:
    pairs = pair_pool(con).sample(len(SCENARIOS), random_state=config.SEED + 400)
    out = []
    for (_, pair), sc in zip(pairs.iterrows(), SCENARIOS, strict=True):
        *state, cls = sc
        q = (
            f"{situation_text(*state).capitalize()}. What is the model's probability that {pair['bat']} "
            f"{EVENTS[cls]} on the next ball against {pair['bowl']}? Give it as a fraction."
        )
        p = situation_prob(srv, pair, *sc)
        ids = [pair["batter_id"], pair["bowler_id"]]
        out.append(rec("next-ball", q, p, 0.01, f"tool:predict_next_ball:{cls}", ["predict_next_ball"], ids))
    return out


def mixed(tools: list[str], question: str, truth: list, ids: list, source: str) -> dict:
    r = rec("mixed", question, truth, 0.01, source, tools, ids, required=tools)
    r["all_of"] = True  # every listed number must be quoted
    return r


def mixed_questions(con, srv: Serving) -> list[dict]:
    pool = pair_pool(con)
    sample = pool[pool["dismissals"] >= 2].sample(3, random_state=config.SEED + 500)
    p3, p4, p5 = (r for _, r in sample.iterrows())
    out = []

    b = labelled(con, srv, "batter", 1, config.SEED + 302).iloc[0]
    sr = one(con, SR_SQL, b["player_id"])
    size = srv.archetype(b["player_id"])["batter"]["clustered_players_in_bin"]
    out.append(
        mixed(
            ["player_profile", "archetype_of"],
            f"What is {b['name']}'s career strike rate off legal balls, and how many clustered players share "
            "his batter archetype bin? Give both numbers.",
            [round(sr[0], 2), size], [b["player_id"]], "sql+tool:archetype_of",
        )
    )  # fmt: skip

    top = one(con, TOP_2024.format(role="batter", agg="sum(d.runs_batter)", cond=""))
    arch = srv.archetype(top[0])["batter"]
    assert arch, "the 2024 top scorer has no batter bin; pick the question differently"
    out.append(
        mixed(
            ["top_players", "archetype_of"],
            "Who scored the most runs off legal balls in 2024, and how many legal balls does the archetype "
            "model have for him as a batter? Give his run total and that ball count.",
            [int(top[2]), arch["train_balls"]], [top[0]], "sql+tool:archetype_of",
        )
    )  # fmt: skip

    ids = [p3["batter_id"], p3["bowler_id"]]
    prob = situation_prob(srv, p3, 1, 17, 140, 4, None, "W")
    out.append(
        mixed(
            ["matchup", "predict_next_ball"],
            f"How many times has {p3['bat']} been dismissed by {p3['bowl']} (excluding run outs), and what "
            "is the model's probability that he is dismissed by him on the next ball in "
            f"{situation_text(1, 17, 140, 4, None)}? Give the count and the probability as a fraction.",
            [int(p3["dismissals"]), prob], ids, "sql+tool:predict_next_ball",
        )
    )  # fmt: skip

    ids = [p4["batter_id"], p4["bowler_id"]]
    sr = one(con, SR_SQL, ids[0])
    prob = situation_prob(srv, p4, 2, 14, 110, 4, 175, "6")
    out.append(
        mixed(
            ["player_profile", "predict_next_ball"],
            f"What is {p4['bat']}'s career strike rate off legal balls, and what is the model's probability "
            f"that he hits a six on the next ball against {p4['bowl']} in "
            f"{situation_text(2, 14, 110, 4, 175)}? Give the strike rate and the probability as a fraction.",
            [round(sr[0], 2), prob], ids, "sql+tool:predict_next_ball",
        )
    )  # fmt: skip

    matches = one(con, MATCHES_SQL, p5["bowler_id"])
    out.append(
        mixed(
            ["sql", "matchup"],
            f"In how many matches has {p5['bowl']} bowled at least one legal ball, and how many legal balls "
            f"has {p5['bat']} faced against him? Give both numbers.",
            [int(matches[0]), int(p5["balls"])], [p5["batter_id"], p5["bowler_id"]], "sql",
        )
    )  # fmt: skip

    top = one(con, TOP_2024.format(role="bowler", agg="count(*)", cond="d.outcome = 'W' and "))
    career = one(con, "select count(*) from legal_deliveries where bowler_id = ? and outcome = 'W'", top[0])
    out.append(
        mixed(
            ["top_players", "player_profile"],
            "Which bowler took the most batter dismissals (excluding run outs) in 2024? Give his "
            "dismissals in 2024 and his career dismissals.",
            [int(top[2]), int(career[0])], [top[0]], "sql",
        )
    )  # fmt: skip
    return out


def main() -> None:
    con, srv = connect(), Serving()
    rows = archetype_questions(con, srv) + next_ball_questions(con, srv) + mixed_questions(con, srv)
    for i, r in enumerate(rows):
        r["id"] = f"t{i + 1:02d}"
    OUT.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    counts = pd.Series([r["category"] for r in rows]).value_counts().to_dict()
    print(f"wrote {len(rows)} questions to {OUT}: {counts}; seed {config.SEED}")


if __name__ == "__main__":
    main()
