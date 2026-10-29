"""The tool loop (D-017): own loop, no framework. One question in, answer + the tool calls it made out.

from cricmind.agent.agent import run
res = run("How many runs has V Kohli scored?", LLM(), Toolbox(connect()))
"""

import json

from cricmind.agent.prompts import SYSTEM
from cricmind.agent.tools import Toolbox


def run(question: str, llm, box: Toolbox, max_steps: int = 6, token_budget: int = 20_000) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls, tokens, latency, answer, stopped = [], 0, 0.0, "", "steps"
    for _ in range(max_steps):
        r = llm.chat(messages, box.schemas())
        tokens += r["usage"]["total_tokens"]
        latency += r["latency"]
        msg = r["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            answer, stopped = msg["content"], None
            break
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                out = box.call(name, args)
            except (json.JSONDecodeError, TypeError):
                args, out = {}, json.dumps({"error": "arguments must be a JSON object"})
            calls.append({"name": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        if tokens > token_budget:
            stopped = "budget"
            break
    # An empty answer with `stopped` set is a failure, never a refusal (the scorer does not credit it).
    return {
        "answer": answer,
        "calls": calls,
        "tokens": tokens,
        "latency": round(latency, 3),
        "stopped": stopped,
    }
