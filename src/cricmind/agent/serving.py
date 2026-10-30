"""Serving artifact behind the `matchup`, `archetype_of` and `predict_next_ball` tools (D-058).

Reads artifacts/p6/serving/ (built by `matchups/build_serving.py`): the D-057 role_arch chain as count tables,
the role labels, and the LightGBM + stats next-ball model. Everything was fitted on train seasons only, so the
numbers are those of the evaluated models. Imports pandas, numpy and lightgbm only: no sklearn, no torch, so
the agent and the Phase 8 API can load it in a small process.

from cricmind.agent.serving import Serving
srv = Serving()
P, n = srv.chain_proba(frame)  # frame: batter_id, bowler_id, phase
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cricmind import config
from cricmind.features.context import OTHER, PHASES, REQ_RATE_CLIP
from cricmind.features.stats import SPECS, RateTable, apply_all

DIR = config.ARTIFACTS / "p6" / "serving"
K = len(config.OUTCOME_CLASSES)
COUNTS = [f"c{i}" for i in range(K)]
PROBS = [f"p{i}" for i in range(K)]
NO_PHASE = "__none__"  # prior row for rows without a phase (reduced matches): the global class shares
NO_VENUE = "__none__"  # D-058(5): venue is not a tool input, so its rates sit at the train prior (n = 0)
LEVEL_KEYS = {  # D-057 order, broad to narrow; the manifest's s values follow it
    "cell": ["bat_arch", "bowl_arch", "phase"],
    "bat": ["batter_id", "phase"],
    "bowl": ["bowler_id", "phase"],
    "pair": ["batter_id", "bowler_id", "phase"],
}
ARCH = ["bat_arch", "bowl_arch"]
STATE = [
    "score_before", "wickets_before", "batter_runs", "batter_balls_faced", "bowler_balls_bowled", "target",
]  # fmt: skip
INNINGS_BALLS = config.SCHEDULED_OVERS * config.BALLS_PER_OVER


def phase_of(over: int) -> str:
    powerplay_end, middle_end = config.PHASE_EDGES
    return PHASES[0] if over < powerplay_end else PHASES[1] if over < middle_end else PHASES[2]


class Serving:
    def __init__(self, path: Path = DIR):
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"{path} has no manifest: run python -m cricmind.matchups.build_serving")
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.prior = pd.read_parquet(path / "phase_prior.parquet").set_index("phase")[PROBS]
        self.levels = {}
        for name in self.manifest["chain"]["s"]:
            table = pd.read_parquet(path / f"chain_{name}.parquet")
            # labels are compared as floats on both sides (a mapped label column with gaps is float)
            self.levels[name] = table.astype({c: "float64" for c in ARCH if c in table})
        self.labels = pd.read_parquet(path / "labels.parquet")
        self.label_of = {r: g.set_index("player_id")["label"] for r, g in self.labels.groupby("role")}
        nb = self.manifest["next_ball"]
        self.booster = lgb.Booster(model_file=str(path / "next_ball.txt"))
        self.rates = {
            p: RateTable(
                pd.read_parquet(path / f"rates_{p}.parquet").set_index(keys)[COUNTS],
                np.array(nb["rates"][p]["prior"]),
                nb["rates"][p]["m"],
            )
            for p, keys in SPECS.items()
        }
        self.defaults = pd.read_parquet(path / "state_defaults.parquet").set_index(["innings", "over"])

    def chain_proba(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """D-057 chain on rows with batter_id, bowler_id, phase: (rows, K) probabilities and, per level, the
        train balls behind each row's key (0 = level skipped, the parent passes through, as in `Chain`)."""
        keys = df[["batter_id", "bowler_id", "phase"]].reset_index(drop=True)
        keys["bat_arch"] = keys["batter_id"].map(self.label_of["batter"]).astype("float64")
        keys["bowl_arch"] = keys["bowler_id"].map(self.label_of["bowler"]).astype("float64")
        P = self.prior.loc[keys["phase"].fillna(NO_PHASE)].to_numpy().copy()  # the levels write into P
        n = {}
        for name, s in self.manifest["chain"]["s"].items():
            # left merge keeps row order; tables hold no missing keys, so an incomplete key never matches
            hit = keys.merge(self.levels[name], how="left", on=LEVEL_KEYS[name])
            counts = hit[COUNTS].fillna(0.0).to_numpy()
            n[name] = counts.sum(axis=1)
            has = n[name] > 0
            P[has] = (counts[has] + s * P[has]) / (n[name][has, None] + s)
        return P, n

    def next_ball_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """LightGBM + stats (D-058(5)) on context rows: NUMERIC, phase, competition and the three ids."""
        nb = self.manifest["next_ball"]
        frame = frame.reset_index(drop=True)
        X = pd.concat([frame, apply_all(frame, self.rates)], axis=1)[nb["features"]]
        for col, levels in nb["levels"].items():
            X[col] = pd.Categorical(X[col], categories=levels)
        return self.booster.predict(X)

    def train_balls(self, prefix: str, key: str) -> int:
        counts = self.rates[prefix].counts
        return int(counts.loc[key].sum()) if key in counts.index else 0

    def state_frame(
        self,
        batter_id: str,
        bowler_id: str,
        over: int,
        innings: int = 1,
        competition: str | None = None,
        **given,
    ) -> tuple[pd.DataFrame, dict]:
        """One hypothetical context row at ball 1 of `over`. Inputs in STATE left as None get the train median
        for that (innings, over) and are listed in `assumed`."""
        over, innings = int(over), int(innings)
        if not 0 <= over < config.SCHEDULED_OVERS or innings not in (1, 2):
            raise ValueError(f"over must be 0-{config.SCHEDULED_OVERS - 1} and innings 1 or 2")
        typical = self.defaults.loc[(innings, over)]
        cols = STATE if innings == 2 else STATE[:-1]
        assumed = [c for c in cols if given.get(c) is None]
        state = {c: int(round(typical[c])) if c in assumed else int(given[c]) for c in cols}
        if not 0 <= state["wickets_before"] <= 9 or min(state.values()) < 0:
            raise ValueError("wickets_before must be 0-9 and counts non-negative")
        if innings == 2 and state["target"] <= state["score_before"]:
            raise ValueError("target must exceed score_before in the 2nd innings")
        levels = self.manifest["next_ball"]["levels"]["competition"]
        comp = competition or self.manifest["next_ball"]["default_competition"]
        comp = comp if comp in levels else OTHER  # rare competitions share __other__, as in build_context
        balls = over * config.BALLS_PER_OVER  # exact for a hypothetical 6-ball-over innings (D-058(5))
        left = INNINGS_BALLS - balls
        target = float(state["target"]) if innings == 2 else np.nan
        row = {
            **{c: float(v) for c, v in state.items()},
            "innings": innings,
            "over": over,
            "ball_in_over": 1.0,
            "balls_bowled": balls,
            "crr": state["score_before"] * 6 / balls if balls else np.nan,
            "legal_balls_remaining": float(left),
            "target": target,
            "req_rate": min(max((target - state["score_before"]) * 6 / left, 0.0), REQ_RATE_CLIP)
            if innings == 2
            else np.nan,
            "phase": phase_of(over),
            "competition": comp,
            "batter_id": batter_id,
            "bowler_id": bowler_id,
            "venue_key": NO_VENUE,
        }
        info = {"state": {"innings": innings, "over": over, **state, "competition": comp}, "assumed": assumed}
        return pd.DataFrame([row]), info

    def archetype(self, player_id: str) -> dict:
        """Role bin per role (D-052 labels, D-050 extension) with its description; None without a label."""
        out = {}
        for role, info in self.manifest["roles"].items():
            row = self.labels[(self.labels["player_id"] == player_id) & (self.labels["role"] == role)]
            if row.empty:
                out[role] = None
                continue
            r = row.iloc[0]
            cluster = info["clusters"][str(int(r["label"]))]
            out[role] = {
                "bin": int(r["label"]),
                "placed": r["placed"],
                "train_balls": int(r["train_balls"]),
                "clustered_players_in_bin": cluster["size"],
                "bin_traits": cluster["top"],
                "clustering_stable": info["stable"],
                "bootstrap_ari": round(info["boot_ari"], 3),
            }
        return out
