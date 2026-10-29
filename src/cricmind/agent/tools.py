"""Agent tools (D-011): typed read-only functions over the DuckDB views, plus one SELECT-only SQL tool.

Every tool returns compact JSON (top-k rows, rounded numbers, the ball counts behind them) because free-tier
token limits are small and answers must cite sample sizes. Tool output is data, never instructions: the system
prompt says so and tools never echo free text from the question. `matchup`, `archetype_of` and
`predict_next_ball` are added through `Toolbox.add` once the serving artifact exists.

from cricmind.agent.tools import Toolbox
box = Toolbox(connect())
"""

import json
import re
from collections.abc import Callable

import duckdb
import pandas as pd

from cricmind import config

ROW_LIMIT = 20
MAX_CHARS = 4000
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|call|vacuum"
    r"|truncate|checkpoint|getenv|current_setting)\b|\bread_\w+|\bglob\b|\bduckdb_\w+|\bpragma_\w+",
    re.I,
)
PATH_LITERAL = re.compile(r"'[^']*[/\\][^']*'|\bfrom\s+['\"]", re.I)  # replacement scans read files by path
PHASE_SQL = (
    f"case when \"over\" < {config.PHASE_EDGES[0]} then 'powerplay' "
    f"when \"over\" < {config.PHASE_EDGES[1]} then 'middle' else 'death' end"
)
METRICS = {
    "batter": {
        "runs": "sum(runs_batter)",
        "strike_rate": "100.0 * sum(runs_batter) / count(*)",
        "boundary_pct": "100.0 * avg((outcome in ('4', '6'))::int)",
    },
    "bowler": {"wickets": "sum((outcome = 'W')::int)", "dot_pct": "100.0 * avg((outcome = 'dot')::int)"},
}
RATE_METRICS = {"strike_rate", "boundary_pct", "dot_pct"}


def guard_sql(sql: str, limit: int = ROW_LIMIT) -> str:
    """One SELECT/WITH statement, no comments, no file access; wrapped to return at most limit + 1 rows."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise ValueError("empty query")
    if ";" in s or "--" in s or "/*" in s:
        raise ValueError("send exactly one statement with no comments")
    if not re.match(r"(select|with)\b", s, re.I):
        raise ValueError("only SELECT queries are allowed")
    if FORBIDDEN.search(s) or PATH_LITERAL.search(s):
        raise ValueError("query uses a forbidden keyword, function or file path")
    return f"select * from ({s}) as q limit {limit + 1}"


def _dump(obj) -> str:
    text = json.dumps(obj, separators=(",", ":"), default=lambda o: o.item())
    return text if len(text) <= MAX_CHARS else json.dumps({"error": "result too large; narrow the query"})


def _rates(df: pd.DataFrame, wkey: str) -> dict:
    n = int(df["balls"].sum())
    runs = int(df["runs"].sum())
    return {
        "balls": n,
        "runs": runs,
        wkey: int(df["wkts"].sum()),
        "strike_rate": round(100 * runs / n, 2) if n else None,
        "boundary_pct": round(100 * df["bnd"].sum() / n, 2) if n else None,
        "dot_pct": round(100 * df["dots"].sum() / n, 2) if n else None,
    }


class Toolbox:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.tools: dict[str, tuple[str, dict, Callable[..., dict]]] = {}
        self.add(
            "player_profile",
            "Career T20 numbers for one player by name or registry id: batting and bowling totals and by "
            "phase (powerplay, middle, death), with ball counts. Returns candidates if the name is "
            "ambiguous.",
            {"player": {"type": "string", "description": "player name or registry id"}},
            self.player_profile,
        )
        self.add(
            "top_players",
            "Leaderboard. role batter: metric runs | strike_rate | boundary_pct; "
            "role bowler: metric wickets | dot_pct. Rate metrics need min_balls (default 300). "
            "Optional season_year.",
            {
                "role": {"type": "string", "enum": ["batter", "bowler"]},
                "metric": {"type": "string"},
                "season_year": {"type": "integer"},
                "k": {"type": "integer", "description": "rows to return, max 10"},
                "min_balls": {"type": "integer"},
            },
            self.top_players,
            required=["role", "metric"],
        )
        self.add(
            "sql",
            'Read-only DuckDB SELECT over views legal_deliveries(match_id, batter_id, bowler_id, "over", '
            "runs_batter, runs_total, outcome in dot|1|2|3+|4|6|W), "
            "matches(match_id, season_year, competition, venue) and players(player_id, name). "
            "Max 20 rows. Use only if no other tool fits.",
            {"query": {"type": "string"}},
            self.sql,
        )

    def add(self, name: str, description: str, props: dict, fn: Callable[..., dict], required=None) -> None:
        params = {"type": "object", "properties": props, "required": required or list(props)}
        self.tools[name] = (description, params, fn)

    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": n, "description": d, "parameters": p}}
            for n, (d, p, _) in self.tools.items()
        ]

    def call(self, name: str, args: dict) -> str:
        if name not in self.tools:
            return _dump({"error": f"unknown tool {name}"})
        try:
            return _dump(self.tools[name][2](**args))
        except (TypeError, ValueError, duckdb.Error) as e:
            return _dump({"error": str(e)[:300]})

    def sql(self, query: str) -> dict:
        df = self.con.execute(guard_sql(query)).df()
        return {"rows": df.head(ROW_LIMIT).round(3).to_dict("records"), "truncated": len(df) > ROW_LIMIT}

    def _find(self, player: str) -> pd.DataFrame:
        return self.con.execute(
            "with m as (select player_id as id, name from players where player_id = ? or name ilike ?), "
            "b as (select batter_id as id, count(*) as n from legal_deliveries group by 1), "
            "w as (select bowler_id as id, count(*) as n from legal_deliveries group by 1) "
            "select m.id, m.name, coalesce(b.n, 0) + coalesce(w.n, 0) as balls from m "
            "left join b using (id) left join w using (id) "
            "where coalesce(b.n, 0) + coalesce(w.n, 0) > 0 order by balls desc limit 5",
            [player, f"%{player}%"],
        ).df()

    def _role(self, col: str, pid: str, wkey: str) -> dict | None:
        df = self.con.execute(
            f"select {PHASE_SQL} as phase, count(*) as balls, sum(runs_batter) as runs, "
            f"sum((outcome = 'W')::int) as wkts, sum((outcome in ('4', '6'))::int) as bnd, "
            f"sum((outcome = 'dot')::int) as dots from legal_deliveries where {col} = ? group by phase",
            [pid],
        ).df()
        if df.empty:
            return None
        by_phase = {r.phase: _rates(df[df["phase"] == r.phase], wkey) for r in df.itertuples()}
        return {**_rates(df, wkey), "by_phase": by_phase}

    def player_profile(self, player: str) -> dict:
        found = self._find(player)
        exact = found[(found["name"].str.lower() == player.lower()) | (found["id"] == player)]
        if len(found) == 0:
            return {"found": False}
        if len(found) > 1 and len(exact) != 1:
            return {
                "ambiguous": True,
                "candidates": found.to_dict("records"),
                "hint": "call again with the id",
            }
        pick = exact.iloc[0] if len(exact) == 1 else found.iloc[0]
        return {
            "id": pick["id"],
            "name": pick["name"],
            "note": "counts are legal balls only; by_phase uses the over of the delivery",
            "batting": self._role("batter_id", pick["id"], "dismissals"),
            "bowling": self._role("bowler_id", pick["id"], "wickets"),
        }

    def top_players(
        self, role: str, metric: str, season_year: int | None = None, k: int = 5, min_balls: int = 300
    ) -> dict:
        if role not in METRICS or metric not in METRICS[role]:
            raise ValueError(f"metric must be one of {sorted(METRICS.get(role, {}))} for role {role!r}")
        k = max(1, min(int(k), 10))
        join = "join matches m using (match_id) " if season_year else ""
        where = "where m.season_year = ? " if season_year else ""
        having = f"having count(*) >= {int(min_balls)} " if metric in RATE_METRICS else ""
        df = self.con.execute(
            f"select p.player_id as id, p.name, round({METRICS[role][metric]}, 2) as value, "
            f"count(*) as balls "
            f"from legal_deliveries d {join}join players p on p.player_id = d.{role}_id {where}"
            f"group by p.player_id, p.name {having}order by value desc limit {k}",
            [int(season_year)] if season_year else [],
        ).df()
        return {"role": role, "metric": metric, "season_year": season_year, "rows": df.to_dict("records")}
