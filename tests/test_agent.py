"""Agent tools, SQL guard, tool loop and scorer. Synthetic parquet and a scripted fake LLM; no network."""

import json

import numpy as np
import pandas as pd
import pytest

from cricmind.agent.agent import run
from cricmind.agent.tools import Toolbox, guard_sql
from cricmind.data.db import connect
from cricmind.eval.agent_eval import answer_ok, numbers, score_one, summarise


@pytest.fixture()
def box(tmp_path):
    rows = [  # match, batter, bowler, over, runs_batter, outcome, legal, super_over
        ("m1", "b1", "w1", 0, 4, "4", True, False),
        ("m1", "b1", "w1", 1, 0, "dot", True, False),
        ("m1", "b1", "w1", 3, 0, "W", True, False),
        ("m1", "b1", "w1", 16, 1, "1", True, False),
        ("m1", "b1", "w1", 2, 4, None, False, False),  # wide/no-ball: not a legal ball
        ("m1", "b1", "w1", 0, 6, "6", True, True),  # super over: excluded
        ("m2", "b2", "w1", 10, 6, "6", True, False),
        ("m2", "b3", "w1", 11, 2, "2", True, False),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "batter_id",
            "bowler_id",
            "over",
            "runs_batter",
            "outcome",
            "is_legal",
            "is_super_over",
        ],
    )
    df["runs_total"] = df["runs_batter"]
    df.to_parquet(tmp_path / "deliveries.parquet")
    pd.DataFrame({"match_id": ["m1", "m2"], "season_year": [2023, 2024]}).to_parquet(
        tmp_path / "matches.parquet"
    )
    (tmp_path / "people.csv").write_text(
        "identifier,name,unique_name\nb1,A Bat,A Bat\nb2,B Bat,B Bat\nb3,A Bat,A Bat 2\nw1,C Bowl,C Bowl\n"
    )
    return Toolbox(connect(tmp_path, tmp_path / "people.csv"))


@pytest.mark.parametrize(
    "sql",
    [
        "drop table deliveries",
        "update deliveries set outcome = 'x'",
        "select 1; select 2",
        "select 1 -- hi",
        "select * from read_parquet('x.parquet')",
        "select * from '/etc/passwd'",
        "select * from deliveries where outcome = 'a/b'",
        "with a as (select 1) copy a to 'x.csv'",
        "",
    ],
)
def test_guard_rejects(sql):
    with pytest.raises(ValueError):
        guard_sql(sql)


def test_guard_accepts_and_limits():
    assert guard_sql("select 1;").endswith("limit 21")
    assert guard_sql("WITH a AS (select 1 as x) select x from a", limit=5).endswith("limit 6")


def test_sql_tool_truncates(box):
    out = json.loads(box.call("sql", {"query": "select * from legal_deliveries"}))
    assert len(out["rows"]) == 6 and out["truncated"] is False
    assert "error" in json.loads(box.call("sql", {"query": "drop view deliveries"}))


def test_player_profile_counts_legal_balls_only(box):
    out = json.loads(box.call("player_profile", {"player": "b1"}))
    bat = out["batting"]
    assert (bat["balls"], bat["runs"], bat["dismissals"]) == (4, 5, 1)
    assert bat["strike_rate"] == 125.0 and bat["boundary_pct"] == 25.0 and bat["dot_pct"] == 25.0
    assert bat["by_phase"]["powerplay"]["balls"] == 3 and bat["by_phase"]["death"]["balls"] == 1
    assert out["bowling"] is None


def test_player_profile_ambiguous_and_missing(box):
    amb = json.loads(box.call("player_profile", {"player": "A Bat"}))
    assert amb["ambiguous"] and {c["id"] for c in amb["candidates"]} == {"b1", "b3"}
    assert json.loads(box.call("player_profile", {"player": "Nobody"})) == {"found": False}
    assert json.loads(box.call("player_profile", {"player": "C Bowl"}))["bowling"]["wickets"] == 1


def test_top_players_and_bad_args(box):
    out = json.loads(box.call("top_players", {"role": "batter", "metric": "runs", "season_year": 2024}))
    assert out["rows"][0]["id"] == "b2" and out["rows"][0]["value"] == 6
    assert "error" in json.loads(box.call("top_players", {"role": "batter", "metric": "wickets"}))
    assert "error" in json.loads(box.call("top_players", {"bogus": 1}))
    assert "error" in json.loads(box.call("nope", {}))


class StubServing:
    """Stands in for the D-058 artifact: uniform probabilities, fixed counts. Tests the tool wiring only."""

    def chain_proba(self, df):
        return np.full((len(df), 7), 1 / 7), {"cell": np.zeros(len(df)), "bat": np.full(len(df), 4.0)}

    def archetype(self, player_id):
        return {"batter": {"bin": 1, "placed": "clustered"}, "bowler": None}

    def state_frame(self, batter_id, bowler_id, over, innings=1, competition=None, **given):
        if not 0 <= over < 20:
            raise ValueError("over must be 0-19")
        return pd.DataFrame([{"over": over}]), {"state": {"over": over, **given}, "assumed": ["score_before"]}

    def next_ball_proba(self, frame):
        return np.full((len(frame), 7), 1 / 7)

    def train_balls(self, prefix, key):
        return 3


@pytest.fixture()
def sbox(box):
    return Toolbox(box.con, StubServing())


def test_serving_tools_are_offered_only_with_an_artifact(box, sbox):
    serving = {"matchup", "archetype_of", "predict_next_ball"}
    assert not serving & set(box.tools) and serving <= set(sbox.tools)


def test_matchup_keeps_observed_counts_and_model_apart(sbox):
    out = json.loads(sbox.call("matchup", {"batter": "b1", "bowler": "C Bowl"}))
    obs = out["observed"]
    assert (obs["balls"], obs["runs"], obs["dismissals"]) == (4, 5, 1)  # legal, no super over
    assert set(out["model"]) == {"powerplay", "middle", "death"}
    assert abs(sum(out["model"]["death"]["p"].values()) - 1) < 1e-3
    assert out["model"]["death"]["train_balls"] == {"cell": 0, "bat": 4}
    death = json.loads(sbox.call("matchup", {"batter": "b1", "bowler": "w1", "phase": "death"}))
    assert death["observed"]["balls"] == 1 and list(death["model"]) == ["death"]
    none = json.loads(sbox.call("matchup", {"batter": "b2", "bowler": "w1", "phase": "death"}))
    assert none["observed"] == {"balls": 0}


def test_matchup_and_archetype_pass_name_problems_back(sbox):
    assert json.loads(sbox.call("matchup", {"batter": "A Bat", "bowler": "w1"}))["batter"]["ambiguous"]
    assert json.loads(sbox.call("matchup", {"batter": "b1", "bowler": "Nobody"}))["bowler"]["found"] is False
    assert "error" in json.loads(sbox.call("matchup", {"batter": "b1", "bowler": "w1", "phase": "late"}))
    out = json.loads(sbox.call("archetype_of", {"player": "B Bat"}))
    assert out["id"] == "b2" and out["batter"]["bin"] == 1 and out["bowler"] is None


def test_predict_next_ball_lists_assumed_inputs_and_rejects_bad_ones(sbox):
    args = {"batter": "b1", "bowler": "w1", "over": 17, "wickets_before": 3}
    out = json.loads(sbox.call("predict_next_ball", args))
    assert out["assumed"] == ["score_before"] and out["state"]["wickets_before"] == 3
    assert out["batter"]["train_balls"] == 3 and abs(sum(out["p"].values()) - 1) < 1e-3
    assert "error" in json.loads(sbox.call("predict_next_ball", {"batter": "b1", "bowler": "w1", "over": 25}))
    bad = {"batter": "b1", "bowler": "w1", "over": 1, "pitch": "green"}
    assert "error" in json.loads(sbox.call("predict_next_ball", bad))


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools, tool_choice=None):
        return {
            "message": self.replies.pop(0),
            "usage": {"total_tokens": 10},
            "latency": 0.1,
            "cached": False,
        }


def call_msg(name, args):
    tc = {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    return {"role": "assistant", "content": "", "tool_calls": [tc]}


def test_run_tool_loop(box):
    llm = Fake([call_msg("player_profile", {"player": "b1"}), {"role": "assistant", "content": "4 balls"}])
    res = run("q", llm, box)
    assert res["answer"] == "4 balls" and res["stopped"] is None
    assert [c["name"] for c in res["calls"]] == ["player_profile"] and res["tokens"] == 20


def test_run_stops_on_budget_and_bad_arguments(box):
    bad = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "sql", "arguments": "not json"}}
        ],
    }
    res = run("q", Fake([bad, bad]), box, token_budget=15)
    assert res["answer"] == "" and res["stopped"] == "budget" and len(res["calls"]) == 2


Q = {
    "id": "q1",
    "category": "lookup",
    "truth": 125.0,
    "tolerance": 0.1,
    "refuse": False,
    "acceptable_first_tools": ["player_profile"],
}


def test_numbers_and_answer_ok():
    assert numbers("He scored 13,250 runs.") == [13250.0]
    assert answer_ok(Q, "strike rate 125.04") is True
    assert answer_ok(Q, "strike rate 130") is False
    assert answer_ok({**Q, "truth": None}, "x") is None
    assert answer_ok({**Q, "truth": "Rashid Khan"}, "It was rashid khan.") is True
    wi = {**Q, "category": "what-if", "truth": 0.074, "tolerance": 0.01}
    assert answer_ok(wi, "about 7.4%") is True


def test_score_and_summarise():
    ok = score_one(
        Q,
        {
            "answer": "125",
            "calls": [{"name": "player_profile", "args": {}}],
            "tokens": 10,
            "latency": 1.0,
            "stopped": None,
        },
    )
    refuse_q = {**Q, "id": "q2", "category": "refuse", "truth": None, "refuse": True}
    good = score_one(
        refuse_q,
        {"answer": "CANNOT_ANSWER: no handedness", "calls": [], "tokens": 5, "latency": 0.5, "stopped": None},
    )
    empty = score_one(refuse_q, {"answer": "", "calls": [], "tokens": 5, "latency": 0.5, "stopped": "steps"})
    assert ok["ok"] is True and ok["tool_ok"] is True
    assert good["refusal_ok"] is True and empty["refusal_ok"] is False
    s = summarise([ok, good, empty])
    assert s["numeric_accuracy"] == 1.0 and s["refusal_correctness"] == 0.5 and s["n_stopped_early"] == 1
    assert s["passes_80_gate"] is True
