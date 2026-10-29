"""One provider-agnostic chat client (D-017): OpenAI-compatible chat + tool calling.

LLM_BASE_URL, LLM_MODEL, LLM_API_KEY come from the environment (`uv run --env-file .env ...`). Retry with
exponential backoff on 429/5xx is the SDK's (`max_retries`); responses are cached on disk by request hash so a
rerun of the eval costs no requests (free tiers have daily caps). `chat` returns plain dicts, not SDK objects.
"""

import hashlib
import json
import os
import time

from openai import OpenAI

from cricmind import config

CACHE_DIR = config.ARTIFACTS / "p6" / "cache"


def message_dict(m) -> dict:
    out = {"role": "assistant", "content": m.content or ""}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": t.id,
                "type": "function",
                "function": {"name": t.function.name, "arguments": t.function.arguments},
            }
            for t in m.tool_calls
        ]
    return out


class LLM:
    def __init__(self, cache: bool = True):
        try:
            self.model, base, key = (
                os.environ["LLM_MODEL"],
                os.environ["LLM_BASE_URL"],
                os.environ["LLM_API_KEY"],
            )
        except KeyError as e:
            raise RuntimeError(f"set {e.args[0]} in .env (see .env.example)") from e
        self.client = OpenAI(base_url=base, api_key=key, max_retries=5, timeout=90)
        self.cache = cache
        # Free tiers allow ~20 requests/minute: LLM_MIN_INTERVAL=3.5 spaces live calls under that.
        self.gap, self._last = float(os.environ.get("LLM_MIN_INTERVAL", 0)), 0.0

    def chat(self, messages: list[dict], tools: list[dict], tool_choice: dict | str | None = None) -> dict:
        key = hashlib.sha256(json.dumps([self.model, messages, tools, tool_choice], sort_keys=True).encode())
        path = CACHE_DIR / f"{key.hexdigest()}.json"
        if self.cache and path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        if (wait := self._last + self.gap - time.monotonic()) > 0:
            time.sleep(wait)
        extra = {"tool_choice": tool_choice} if tool_choice else {}
        t0 = time.monotonic()
        raw = self.client.chat.completions.with_raw_response.create(
            model=self.model, messages=messages, tools=tools, temperature=0, **extra
        )
        self._last = time.monotonic()
        resp = raw.parse()
        if not resp.choices:  # some gateways answer 200 with an error body; never cache that
            raise RuntimeError(f"provider returned no choices: {getattr(resp, 'error', None)}")
        out = {
            "message": message_dict(resp.choices[0].message),
            "usage": {"total_tokens": resp.usage.total_tokens if resp.usage else 0},
            "latency": round(self._last - t0, 3),
            "ratelimit": {k: v for k, v in raw.headers.items() if "ratelimit" in k.lower()},
        }
        if self.cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
        return {**out, "cached": False}
