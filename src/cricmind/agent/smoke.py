"""Smoke test (D-017): does this provider/model do tool calling, and what rate limits does it report?

uv run --env-file .env python -m cricmind.agent.smoke

Three live calls, no cache: a forced tool call, an unforced one, and the tool-result round trip.
Writes reports/p6/smoke_<model>.json. Measured, not assumed.
"""

import json
import re

from cricmind import config
from cricmind.agent.llm import LLM

TOOL = {
    "type": "function",
    "function": {
        "name": "get_total_runs",
        "description": "Total runs scored by a named player.",
        "parameters": {
            "type": "object",
            "properties": {"player": {"type": "string"}},
            "required": ["player"],
        },
    },
}
ASK = {"role": "user", "content": "How many runs has V Kohli scored? Use the tool."}


def main() -> None:
    llm = LLM(cache=False)
    rep = {"model": llm.model, "forced_call": False, "auto_call": False, "round_trip": False, "error": None}
    try:
        forced = llm.chat(
            [ASK], [TOOL], tool_choice={"type": "function", "function": {"name": "get_total_runs"}}
        )
        rep["ratelimit_headers"] = forced["ratelimit"]
        rep["forced_call"] = bool(forced["message"].get("tool_calls"))
        auto = llm.chat([ASK], [TOOL])
        rep["auto_call"] = bool(auto["message"].get("tool_calls"))
        if rep["auto_call"]:
            call = auto["message"]["tool_calls"][0]
            tool_msg = {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"runs": 13250})}
            final = llm.chat([ASK, auto["message"], tool_msg], [TOOL])
            rep["round_trip"] = "13250" in final["message"]["content"].replace(",", "")
            rep["final_answer"] = final["message"]["content"][:200]
    except Exception as e:  # noqa: BLE001 - the point is to record any provider failure
        rep["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    out = config.REPORTS / "p6"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"smoke_{re.sub(r'[^A-Za-z0-9._-]+', '_', llm.model)}.json").write_text(
        json.dumps(rep, indent=2) + "\n"
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
