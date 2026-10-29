"""Hierarchical backoff chain for batter-vs-bowler outcomes (D-053).

Each level estimates the 7 class probabilities from its own train counts and is pulled toward the level above:
    p_level = (counts + s * p_parent) / (n + s)
so a thin key stays near its parent and a thick one trusts its own counts. A key with no train counts, or a
row with a missing key part (an unlabelled player, a reduced-match row with no phase), passes the parent
through unchanged. The top parent is the phase baseline. Levels, broad to narrow:
    cell  (batter archetype, bowler archetype, phase)   -- the Phase 5 table, optional
    bat   (batter, phase)      bowl  (bowler, phase)      pair  (batter, bowler, phase)
Each s is chosen greedily on val, level by level, with the earlier levels fixed.

from cricmind.matchups.chain import Chain, cell_keys, bat_keys, bowl_keys, pair_keys
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.eval.metrics import row_log_loss
from cricmind.matchups.table import search_strength
from cricmind.models.baselines import PhaseFreq

K = len(config.OUTCOME_CLASSES)
KeyFn = Callable[[pd.DataFrame], pd.DataFrame]


def bat_keys(df: pd.DataFrame) -> pd.DataFrame:
    return df[["batter_id", "phase"]]


def bowl_keys(df: pd.DataFrame) -> pd.DataFrame:
    return df[["bowler_id", "phase"]]


def pair_keys(df: pd.DataFrame) -> pd.DataFrame:
    return df[["batter_id", "bowler_id", "phase"]]


def cell_keys(labels: dict[str, pd.Series]) -> KeyFn:
    def keys(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bat_arch": df["batter_id"].map(labels["batter"]),
                "bowl_arch": df["bowler_id"].map(labels["bowler"]),
                "phase": df["phase"],
            }
        )

    return keys


class Level:
    """Class counts per key, fitted on train rows whose key is complete."""

    def __init__(self, name: str, key_fn: KeyFn):
        self.name, self.key_fn = name, key_fn

    def fit(self, train: pd.DataFrame) -> "Level":
        keys = self.key_fn(train)
        ok = keys.notna().all(axis=1).to_numpy()
        codes, self.index = pd.MultiIndex.from_frame(keys[ok]).factorize()
        flat = codes * K + train["y"].to_numpy()[ok]
        self.counts = np.bincount(flat, minlength=len(self.index) * K).reshape(len(self.index), K)
        return self

    def lookup(self, df: pd.DataFrame) -> np.ndarray:
        """(rows, K) train counts of each row's key, zeros where the key is incomplete or never seen."""
        keys = self.key_fn(df)
        ok = keys.notna().all(axis=1).to_numpy()
        found = self.index.get_indexer(pd.MultiIndex.from_frame(keys[ok]))
        out = np.zeros((len(df), K))
        hit = found >= 0
        out[np.flatnonzero(ok)[hit]] = self.counts[found[hit]]
        return out


def shrink(counts: np.ndarray, prior: np.ndarray, s: float) -> np.ndarray:
    return (counts + s * prior) / (counts.sum(axis=1, keepdims=True) + s)


class Chain:
    def __init__(self, specs: list[tuple[str, KeyFn]], phase_freq: PhaseFreq):
        self.levels = [Level(name, fn) for name, fn in specs]
        self.phase_freq = phase_freq
        self.s: list[float] = []

    def fit(self, train: pd.DataFrame, tune: pd.DataFrame) -> dict:
        """Counts from `train`; each level's s from the `tune` rows, greedily. Returns the staircase of
        `tune` log-loss after each level and the share of `tune` rows the level had counts for."""
        y = tune["y"].to_numpy()
        P = self.phase_freq.predict_proba(tune)
        steps = [{"level": "phase_prior", "tune_log_loss": float(row_log_loss(P, y).mean())}]
        self.s = []
        for level in self.levels:
            level.fit(train)
            counts = level.lookup(tune)
            has = counts.sum(axis=1) > 0
            found = search_strength(counts[has], P[has], y[has])
            self.s.append(found["s"])
            P[has] = shrink(counts[has], P[has], found["s"])
            steps.append(
                {
                    "level": level.name,
                    "s": found["s"],
                    "at_grid_edge": found["at_grid_edge"],
                    "tune_rows_with_counts": float(has.mean()),
                    "tune_log_loss": float(row_log_loss(P, y).mean()),
                }
            )
        return {"steps": steps}

    def refit(self, train: pd.DataFrame, phase_freq: PhaseFreq) -> "Chain":
        """New counts and phase prior from `train`, the s values kept: tune on an earlier year, then score a
        later one without that later year choosing anything (D-032)."""
        self.phase_freq = phase_freq
        for level in self.levels:
            level.fit(train)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        P = self.phase_freq.predict_proba(df)
        for level, s in zip(self.levels, self.s, strict=True):
            counts = level.lookup(df)
            has = counts.sum(axis=1) > 0
            P[has] = shrink(counts[has], P[has], s)
        return P
