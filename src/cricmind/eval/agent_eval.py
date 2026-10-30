"""Scores the agent on evals/agent_questions.jsonl -> reports/p6/model_comparison.json (D-017, Phase 6).

uv run --env-file .env python -m cricmind.eval.agent_eval [--limit N]

Rules fixed before any model run: a numeric answer is right if any number in it is within the question's
tolerance of the truth (what-if and next-ball questions also accept a number followed by %; a list truth
accepts any listed number); a name answer must contain the truth name; a refusal is right iff the answer
starts with CANNOT_ANSWER:; tool choice is right iff the first tool called is in the question's
`acceptable_first_tools`. What-if truths come from the saved D-058 chain; a question with no truth is left
out of numeric accuracy and counted in `n_unscored`. `--set tools` scores the D-059 set of 20 questions.
"""

import argparse
import json
import os
import re
import unicodedata

import numpy as np

from cricmind import config
from cricmind.agent.agent import run
from cricmind.agent.llm import LLM
from cricmind.agent.prompts import REFUSAL
from cricmind.agent.serving import Serving
from cricmind.agent.tools import Toolbox
from cricmind.data.db import connect

QUESTIONS = config.ROOT / "evals" / "agent_questions.jsonl"
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
PERCENT = re.compile(r"(-?\d[\d,]*\.?\d*)\s*%")
RATIO = re.compile(r"(\d[\d,]*\.?\d*)\s*/\s*(\d[\d,]*\.?\d*)")
FRACTION = ("what-if", "next-ball", "mixed")
GATE = 0.8
# D-059: the second set tests archetype_of and predict_next_ball; it has its own report, not the 80% gate's.
SETS = {
    "main": (QUESTIONS, "model_comparison.json", ""),
    "tools": (config.ROOT / "evals" / "agent_questions_tools.jsonl", "tool_comparison.json", "_tools"),
}


def numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUMBER.findall(text) if m.replace(",", "").strip(".")]


def answer_ok(q: dict, answer: str) -> bool | None:
    """None = no truth to score against."""
    truth = q["truth"]
    if truth is None:
        return None
    answer = unicodedata.normalize("NFKC", answer)  # models write no-break spaces and hyphens in names
    if isinstance(truth, str):
        return truth.lower() in answer.lower()
    tol = q["tolerance"] + 1e-9
    values = numbers(answer)
    if q["category"] in FRACTION:  # asked as a fraction, a model may answer "7.4%" or "78/1000"
        values += [float(m.replace(",", "")) / 100 for m in PERCENT.findall(answer)]  # a bare "3" is not 3%
        ratios = [(float(a.replace(",", "")), float(b.replace(",", ""))) for a, b in RATIO.findall(answer)]
        values += [a / b for a, b in ratios if b]
    targets = truth if isinstance(truth, list) else [truth]  # a list: quoting any listed number is right
    hit = [any(abs(v - t) <= tol for v in values) for t in targets]
    return all(hit) if q.get("all_of") else any(hit)  # all_of: a two-part question needs both numbers


def score_one(q: dict, res: dict) -> dict:
    first = res["calls"][0]["name"] if res["calls"] else None
    refused = res["answer"].strip().startswith(REFUSAL)
    return {
        "id": q["id"],
        "category": q["category"],
        "answer": res["answer"],
        "first_tool": first,
        "n_calls": len(res["calls"]),
        "calls": res["calls"],  # name + arguments of every call, to see why a question went wrong
        "tokens": res["tokens"],
        "latency": res["latency"],
        "stopped": res["stopped"],
        "refused": refused,
        "ok": None if q["refuse"] else answer_ok(q, res["answer"]),
        "tool_ok": None if q["refuse"] else first in q["acceptable_first_tools"],
        # mixed questions need every listed tool at some point, in any order
        "tools_ok": set(q["required_tools"]) <= {c["name"] for c in res["calls"]}
        if q.get("required_tools")
        else None,
        "refusal_ok": refused if q["refuse"] else None,
    }


def mean(rows: list[dict], key: str, where=lambda r: True) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None and where(r)]
    return round(float(np.mean(vals)), 4) if vals else None


def summarise(rows: list[dict]) -> dict:
    lat = [r["latency"] for r in rows]
    numeric = mean(rows, "ok")
    price_in = float(os.environ.get("LLM_PRICE_PER_MTOK", 0))  # blended USD per 1M tokens; free tier = 0
    return {
        "n_questions": len(rows),
        "numeric_accuracy": numeric,
        "numeric_accuracy_by_category": {
            c: mean(rows, "ok", lambda r, c=c: r["category"] == c)
            for c in ("lookup", "matchup", "what-if", "archetype", "next-ball", "mixed")
        },
        "required_tools_used": mean(rows, "tools_ok"),
        "n_unscored": sum(r["ok"] is None and r["category"] != "refuse" for r in rows),
        "tool_choice_accuracy": mean(rows, "tool_ok"),
        "refusal_correctness": mean(rows, "refusal_ok"),
        "false_refusal_rate": mean(rows, "refused", lambda r: r["category"] != "refuse"),
        "tokens_per_question": round(float(np.mean([r["tokens"] for r in rows])), 1),
        "latency_p50_s": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
        "n_stopped_early": sum(r["stopped"] is not None for r in rows),
        "cost_usd": round(sum(r["tokens"] for r in rows) / 1e6 * price_in, 4),
        "passes_80_gate": numeric is not None and numeric >= GATE,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", choices=sorted(SETS), default="main")
    args = ap.parse_args()
    questions_path, report_name, suffix = SETS[args.set]
    questions = [json.loads(line) for line in questions_path.read_text().splitlines()][: args.limit]
    llm, box = LLM(), Toolbox(connect(), Serving())
    out = config.REPORTS / "p6"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", llm.model)
    rows = []
    # Each row is written as soon as it is scored, so a run cut short by a rate limit keeps its answers.
    with (out / "runs" / f"{slug}{suffix}.jsonl").open("w") as f:
        for q in questions:
            rows.append(score_one(q, run(q["question"], llm, box)))
            f.write(json.dumps(rows[-1]) + "\n")
            f.flush()
            r = rows[-1]
            print(f"{q['id']} {q['category']:8s} ok={r['ok']} tool={r['first_tool']}", flush=True)
            if r["ok"] is False or r["tool_ok"] is False or r["refusal_ok"] is False:
                print(f"    truth={q['truth']!r} answer={r['answer'][:240]!r}", flush=True)
            elif q.get("rubric"):  # scored on the numbers it quotes; the reading is for a human to judge
                print(f"    rubric: {q['rubric']}\n    answer={r['answer'][:300]!r}", flush=True)
    path = out / report_name
    report = json.loads(path.read_text()) if path.exists() else {}
    report[llm.model] = summarise(rows)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report[llm.model], indent=2))


if __name__ == "__main__":
    main()
