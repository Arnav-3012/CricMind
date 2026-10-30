"""Agent tools (D-011): typed read-only functions over the DuckDB views, plus one SELECT-only SQL tool.

Every tool returns compact JSON (top-k rows, rounded numbers, the ball counts behind them) because free-tier
token limits are small and answers must cite sample sizes. Tool output is data, never instructions: the system
prompt says so and tools never echo free text from the question. `matchup`, `archetype_of` and
`predict_next_ball` read the D-058 serving artifact and are only offered when one is passed in.

from cricmind.agent.tools import Toolbox
box = Toolbox(connect(), Serving())
"""

import json
import re
from collections.abc import Callable

import duckdb
import pandas as pd

from cricmind import config
from cricmind.agent.serving import Serving
from cricmind.features.context import PHASES

ROW_LIMIT = 20
MAX_CHARS = 4000
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|call|vacuum"
    r"|truncate|checkpoint|getenv|current_setting|sniff_csv|query|query_table)\b|\bread_\w+|\bglob\b"
    r"|\bduckdb_\w+|\bpragma_\w+|\bparquet_\w+|\w+_scan\b",
    re.I,
)
# D-062: file-reading table functions take a bare file name too ('x.parquet' has no slash), so no function
# call is allowed in FROM or JOIN at all; the agent only ever queries the views.
TABLE_FUNCTION = re.compile(r"\b(from|join)\s+\w+\s*\(|,\s*\w+\s*\(\s*'", re.I)
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
STATE_INPUTS = [  # optional predict_next_ball inputs; the names match serving.STATE
    "score_before", "wickets_before", "target", "batter_runs", "batter_balls_faced", "bowler_balls_bowled",
]  # fmt: skip
MODEL_NOTE = (
    "per-ball class probabilities from models fitted on seasons up to 2023 and scored on later seasons; "
    "train_balls = the train balls behind each input"
)


def guard_sql(sql: str, limit: int = ROW_LIMIT) -> str:
    """One SELECT/WITH statement, no comments, no file access; wrapped to return at most limit + 1 rows."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise ValueError("empty query")
    if ";" in s or "--" in s or "/*" in s:
        raise ValueError("send exactly one statement with no comments")
    if not re.match(r"(select|with)\b", s, re.I):
        raise ValueError("only SELECT queries are allowed")
    if FORBIDDEN.search(s) or PATH_LITERAL.search(s) or TABLE_FUNCTION.search(s):
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
    def __init__(self, con: duckdb.DuckDBPyConnection, serving: Serving | None = None):
        self.con, self.srv = con, serving
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
        if serving is not None:
            self._add_serving_tools()

    def _add_serving_tools(self) -> None:
        self.add(
            "matchup",
            "Batter vs bowler. `observed`: their head-to-head counts over all seasons (balls, runs, "
            "dismissals, strike rate, dot %, boundary %, by phase). `model`: per-ball probabilities of "
            "dot|1|2|3+|4|6|W per phase from the matchup model, with the train balls behind it.",
            {
                "batter": {"type": "string", "description": "batter name or registry id"},
                "bowler": {"type": "string", "description": "bowler name or registry id"},
                "phase": {"type": "string", "enum": PHASES},
            },
            self.matchup,
            required=["batter", "bowler"],
        )
        self.add(
            "archetype_of",
            "A player's archetype bin per role (batter, bowler): numbered clusters of similar players with "
            "their 3 most distinctive traits, train balls, and whether the clustering is stable.",
            {"player": {"type": "string", "description": "player name or registry id"}},
            self.archetype_of,
        )
        state = {c: {"type": "integer"} for c in STATE_INPUTS}
        self.add(
            "predict_next_ball",
            "Probabilities of dot|1|2|3+|4|6|W for the next ball of a batter vs bowler in a given match "
            "situation. over is 0-based (0 = first over, 19 = last). Inputs left out get typical values "
            "and are listed in `assumed`. target only in innings 2.",
            {
                "batter": {"type": "string"},
                "bowler": {"type": "string"},
                "over": {"type": "integer", "minimum": 0, "maximum": 19},
                "innings": {"type": "integer", "enum": [1, 2]},
                **state,
                "competition": {"type": "string"},
            },
            self.predict_next_ball,
            required=["batter", "bowler", "over"],
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

    def players(self, role: str = "any", min_balls: int = 50) -> list[dict]:
        """Players with at least `min_balls` legal balls in the role, most balls first. Not an agent tool:
        it feeds the dashboard's dropdowns (names collide, so callers keep the id)."""
        cols = {"batter": ["batter_id"], "bowler": ["bowler_id"], "any": ["batter_id", "bowler_id"]}[role]
        balls = " union all ".join(f"select {c} as id from legal_deliveries" for c in cols)
        return (
            self.con.execute(
                f"with b as ({balls}) select p.player_id as id, p.name, count(*) as balls "
                "from b join players p on p.player_id = b.id group by 1, 2 having count(*) >= ? "
                "order by balls desc, id",
                [int(min_balls)],
            )
            .df()
            .to_dict("records")
        )

    def pairs(self, k: int = 100, min_balls: int = 30) -> list[dict]:
        """The batter-bowler pairs with the most shared legal balls (unique tie-break), for the dashboard."""
        return (
            self.con.execute(
                "select d.batter_id as batter_id, d.bowler_id as bowler_id, pb.name as batter, "
                "pw.name as bowler, count(*) as balls from legal_deliveries d "
                "join players pb on pb.player_id = d.batter_id join players pw on pw.player_id = d.bowler_id "
                "group by 1, 2, 3, 4 having count(*) >= ? order by balls desc, batter_id, bowler_id limit ?",
                [int(min_balls), int(k)],
            )
            .df()
            .to_dict("records")
        )

    def _find(self, player: str) -> pd.DataFrame:
        parts = player.split()
        # Registry names are "V Kohli" but people type "Virat Kohli": retry as first initial + surname.
        patterns = [f"%{player}%"] + ([f"{parts[0][0]}% {parts[-1]}"] if len(parts) > 1 else [])
        for pattern in patterns:
            found = self.con.execute(
                "with m as (select player_id as id, name from players where player_id = ? or name ilike ?), "
                "b as (select batter_id as id, count(*) as n from legal_deliveries group by 1), "
                "w as (select bowler_id as id, count(*) as n from legal_deliveries group by 1) "
                "select m.id, m.name, coalesce(b.n, 0) + coalesce(w.n, 0) as balls from m "
                "left join b using (id) left join w using (id) "
                "where coalesce(b.n, 0) + coalesce(w.n, 0) > 0 order by balls desc limit 5",
                [player, pattern],
            ).df()
            if len(found):
                break
        return found

    def _role(self, where: str, params: list, wkey: str) -> dict | None:
        df = self.con.execute(
            f"select {PHASE_SQL} as phase, count(*) as balls, sum(runs_batter) as runs, "
            f"sum((outcome = 'W')::int) as wkts, sum((outcome in ('4', '6'))::int) as bnd, "
            f"sum((outcome = 'dot')::int) as dots from legal_deliveries where {where} group by phase",
            params,
        ).df()
        if df.empty:
            return None
        by_phase = {r.phase: _rates(df[df["phase"] == r.phase], wkey) for r in df.itertuples()}
        return {**_rates(df, wkey), "by_phase": by_phase}

    def _resolve(self, player: str) -> tuple[pd.Series | None, dict | None]:
        """(row with id and name, None), or (None, the reply to send back: not found or the candidates)."""
        found = self._find(player)
        exact = found[(found["name"].str.lower() == player.lower()) | (found["id"] == player)]
        if len(found) == 0:
            return None, {"found": False, "query": player}
        if len(found) > 1 and len(exact) != 1:
            return None, {
                "ambiguous": True,
                "candidates": found.to_dict("records"),
                "hint": "call again with the id",
            }
        return (exact.iloc[0] if len(exact) == 1 else found.iloc[0]), None

    def _pair(self, batter: str, bowler: str) -> tuple[pd.Series | None, pd.Series | None, dict | None]:
        bat, miss = self._resolve(batter)
        if miss:
            return None, None, {"batter": miss}
        bowl, miss = self._resolve(bowler)
        if miss:
            return None, None, {"bowler": miss}
        return bat, bowl, None

    def player_profile(self, player: str) -> dict:
        pick, miss = self._resolve(player)
        if miss:
            return {k: v for k, v in miss.items() if k != "query"}
        return {
            "id": pick["id"],
            "name": pick["name"],
            "note": "counts are legal balls only; by_phase uses the over of the delivery",
            "batting": self._role("batter_id = ?", [pick["id"]], "dismissals"),
            "bowling": self._role("bowler_id = ?", [pick["id"]], "wickets"),
        }

    def matchup(self, batter: str, bowler: str, phase: str | None = None) -> dict:
        if phase is not None and phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        bat, bowl, miss = self._pair(batter, bowler)
        if miss:
            return miss
        ids = [bat["id"], bowl["id"]]
        observed = self._role("batter_id = ? and bowler_id = ?", ids, "dismissals")
        if observed and phase:
            observed = observed["by_phase"].get(phase)
        phases = [phase] if phase else PHASES
        frame = pd.DataFrame({"batter_id": ids[0], "bowler_id": ids[1], "phase": phases})
        P, n = self.srv.chain_proba(frame)
        model = {
            ph: {
                "p": dict(zip(config.OUTCOME_CLASSES, P[i].round(4).tolist(), strict=True)),
                "train_balls": {level: int(v[i]) for level, v in n.items()},
            }
            for i, ph in enumerate(phases)
        }
        return {
            "batter": {"id": ids[0], "name": bat["name"]},
            "bowler": {"id": ids[1], "name": bowl["name"]},
            "observed": observed or {"balls": 0},
            "model": model,
            "model_note": MODEL_NOTE + " (levels: archetype cell, batter, bowler, this pair; 0 = no data)",
        }

    def archetype_of(self, player: str) -> dict:
        pick, miss = self._resolve(player)
        if miss:
            return miss
        return {
            "id": pick["id"],
            "name": pick["name"],
            **self.srv.archetype(pick["id"]),
            "note": "bins are numbered, not named; trait z-scores vs all clustered players; seasons <= 2023",
        }

    def predict_next_ball(
        self,
        batter: str,
        bowler: str,
        over: int,
        innings: int = 1,
        competition: str | None = None,
        **state,
    ) -> dict:
        unknown = set(state) - set(STATE_INPUTS)
        if unknown:
            raise ValueError(f"unknown inputs {sorted(unknown)}")
        bat, bowl, miss = self._pair(batter, bowler)
        if miss:
            return miss
        frame, info = self.srv.state_frame(bat["id"], bowl["id"], over, innings or 1, competition, **state)
        p = self.srv.next_ball_proba(frame)[0]
        who = {
            role: {"id": r["id"], "name": r["name"], "train_balls": self.srv.train_balls(prefix, r["id"])}
            for role, prefix, r in (("batter", "bat", bat), ("bowler", "bowl", bowl))
        }
        return {
            **who,
            **info,
            "p": dict(zip(config.OUTCOME_CLASSES, p.round(4).tolist(), strict=True)),
            "model_note": MODEL_NOTE,
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
