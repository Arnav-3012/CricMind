"""Archetype matchup table with Dirichlet-multinomial shrinkage toward the phase prior (D-010, D-048).

Cell = (batter archetype, bowler archetype, phase). A cell's class counts come from train rows only; its
probabilities are (counts + s * prior_phase) / (n + s), so a thin cell falls back to the phase shares and a
thick one trusts its own counts. Rows the table cannot serve (a player without a label, or a reduced-match
row with no phase) get the phase baseline, so every predictor scores the same rows.

from cricmind.matchups.table import MatchupTable, load_labels, fit_strength
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cricmind import config
from cricmind.archetypes.roles import role_vectors
from cricmind.archetypes.stability import make_model
from cricmind.archetypes.vectors import ROLES, embedding_rows, stats_vectors
from cricmind.eval.metrics import row_log_loss
from cricmind.features.context import PHASES
from cricmind.models.baselines import PhaseFreq

K = len(config.OUTCOME_CLASSES)
S_GRID = np.logspace(0, 6, 37)  # D-051 (widens D-048(3)): 1 .. 1,000,000 pseudo-balls, same 6 points a decade
EXT_MIN_BALLS = config.MIN_TRAIN_BALLS_EMBEDDING  # D-050: lighter players need an embedding row to be placed
ARM_NAMES = ("stats", "emb", "stats_ext", "emb_ext")  # D-050: *_ext = same clusters, lighter players too


def load_labels(rep: str) -> dict[str, pd.Series]:
    """player_id -> archetype label per role from the Phase 4 k-means clustering of `rep` (stats | emb)."""
    labels = pd.read_parquet(config.ARTIFACTS / "p4" / f"archetypes_{rep}.parquet")
    labels = labels[labels["method"] == "kmeans"]
    return {role: g.set_index("player_id")["label"] for role, g in labels.groupby("role")}


def nearest_cluster(model, X: np.ndarray, ids: pd.Index) -> pd.Series:
    """Label of the nearest cluster centre, in the fitted pipeline's own space (scaler, PCA, centroids)."""
    return pd.Series(model.predict(X), index=ids)


def _vectors(rep: str, frame: pd.DataFrame, role: str, ids: pd.Index) -> np.ndarray:
    if rep == "stats":
        return stats_vectors(frame, role, ids).to_numpy()
    if rep == "role":
        return role_vectors(frame, role, ids).to_numpy()
    return embedding_rows(config.ARTIFACTS / "p3", role, ids)


def chosen_spec(rep: str) -> dict[str, dict]:
    """The k, eligibility floor and PCA flag each role's clustering ended with (Phase 4 / D-052 reports)."""
    path = config.REPORTS / ("p5b/role_clusters.json" if rep == "role" else "p4/clusters.json")
    cells = json.loads(path.read_text())["cells"]
    out = {}
    for role in ROLES:
        cell = cells[role if rep == "role" else f"{rep}_{role}"]
        out[role] = {key: cell[key] for key in ("k", "min_balls", "pca")}
    return out


def refit_labels(
    rep: str, part: pd.DataFrame, spec: dict[str, dict]
) -> tuple[dict[str, pd.Series], dict[str, Pipeline]]:
    """D-055: the same clustering recipe (k, floor, PCA from `spec`) fitted on `part` only, e.g. the years
    before the tuning year, so a holdout year never decides who is in which cluster. Not for embeddings."""
    labels, models = {}, {}
    for role, col in ROLES.items():
        counts = part[col].value_counts()
        ids = pd.Index(sorted(counts[counts >= spec[role]["min_balls"]].index))
        model = make_model(spec[role]["k"], spec[role]["pca"]).fit(_vectors(rep, part, role, ids))
        labels[role], models[role] = pd.Series(model[-1].labels_, index=ids), model
    return labels, models


def extend_labels(
    rep: str,
    train: pd.DataFrame,
    base: dict[str, pd.Series],
    min_balls: int = EXT_MIN_BALLS,
    models: dict[str, Pipeline] | None = None,
) -> tuple[dict[str, pd.Series], dict]:
    """D-050: place every player with at least `min_balls` train balls on the frozen Phase 4 clustering.
    Clusters, scaler and PCA are not refit; the Phase 4 label wins for the players who have one. Train rows
    only, so test players are never looked at. `models` replaces the saved pipelines (D-055 holdout refit)."""
    labels, info = {}, {}
    for role, col in ROLES.items():
        counts = train[col].value_counts()
        ids = pd.Index(sorted(counts[counts >= min_balls].index))
        saved = config.ARTIFACTS / "p4" / f"kmeans_{rep}_{role}.joblib"
        model = models[role] if models else joblib.load(saved)
        assigned = nearest_cluster(model, _vectors(rep, train, role, ids), ids)
        extra = assigned[~assigned.index.isin(base[role].index)]
        labels[role] = pd.concat([base[role], extra])
        info[role] = {
            "base_players": len(base[role]),
            "players": len(labels[role]),
            # sanity: re-assigning the Phase 4 players must give their Phase 4 labels back
            "agrees_with_phase4": float((assigned.reindex(base[role].index) == base[role]).mean()),
        }
    return labels, info


def build_arm(name: str, train: pd.DataFrame, phase_freq: PhaseFreq) -> tuple["MatchupTable", dict]:
    """One of ARM_NAMES: the Phase 4 labels, or (`*_ext`) those extended to lighter players."""
    rep, _, ext = name.partition("_")
    labels, info = load_labels(rep), {}
    if ext:
        labels, info = extend_labels(rep, train, labels)
    return MatchupTable(labels).fit(train, phase_freq), info


class MatchupTable:
    def __init__(self, labels: dict[str, pd.Series]):
        self.labels = labels
        self.k_bat = int(labels["batter"].max()) + 1
        self.k_bowl = int(labels["bowler"].max()) + 1

    @property
    def n_cells(self) -> int:
        return self.k_bat * self.k_bowl * len(PHASES)

    def cells(self, df: pd.DataFrame) -> np.ndarray:
        """Flat cell id per row, -1 where the table cannot serve the row."""
        bat = df["batter_id"].map(self.labels["batter"])
        bowl = df["bowler_id"].map(self.labels["bowler"])
        phase = pd.Categorical(df["phase"], categories=PHASES).codes
        ok = bat.notna().to_numpy() & bowl.notna().to_numpy() & (phase >= 0)
        b, w = bat.fillna(0).to_numpy("int64"), bowl.fillna(0).to_numpy("int64")
        return np.where(ok, (b * self.k_bowl + w) * len(PHASES) + phase, -1)

    def served(self, df: pd.DataFrame) -> np.ndarray:
        return self.cells(df) >= 0

    def fit(self, train: pd.DataFrame, phase_freq: PhaseFreq) -> "MatchupTable":
        self.phase_freq = phase_freq
        cell = self.cells(train)
        ok = cell >= 0
        flat = cell[ok] * K + train["y"].to_numpy()[ok]
        self.counts = np.bincount(flat, minlength=self.n_cells * K).reshape(self.n_cells, K)
        return self

    def predict_proba(self, df: pd.DataFrame, s: float) -> np.ndarray:
        out = self.phase_freq.predict_proba(df)  # the fallback, and the prior of every served row
        ok = self.cells(df)
        served = ok >= 0
        counts = self.counts[ok[served]]
        out[served] = (counts + s * out[served]) / (counts.sum(axis=1, keepdims=True) + s)
        return out

    def to_frame(self, s: float) -> pd.DataFrame:
        """The architecture.md artifact: one row per cell with shrunk probabilities."""
        cell = np.arange(self.n_cells)
        phase = cell % len(PHASES)
        prior = self.phase_freq.table[phase]
        n = self.counts.sum(axis=1, keepdims=True)
        P = (self.counts + s * prior) / (n + s)
        frame = pd.DataFrame(
            {
                "bat_arch": cell // (len(PHASES) * self.k_bowl),
                "bowl_arch": (cell // len(PHASES)) % self.k_bowl,
                "phase": [PHASES[i] for i in phase],
                "n": n[:, 0],
            }
        )
        probs = pd.DataFrame(P, columns=[f"p_{c}" for c in config.OUTCOME_CLASSES])
        return pd.concat([frame, probs], axis=1).assign(prior_strength=s)


def _val_loss(counts: np.ndarray, prior: np.ndarray, y: np.ndarray, s: float) -> float:
    P = (counts + s * prior) / (counts.sum(axis=1, keepdims=True) + s)
    return float(row_log_loss(P, y).mean())


def fit_strength(table: MatchupTable, val: pd.DataFrame) -> dict:
    """Pick the prior strength s on the val rows the table serves (D-048(3)). Val picks only s; counts come
    from train."""
    served = table.served(val)
    counts = table.counts[table.cells(val)[served]]
    prior = table.phase_freq.predict_proba(val)[served]
    return {**search_strength(counts, prior, val["y"].to_numpy()[served]), "served_rows": int(served.sum())}


def search_strength(counts: np.ndarray, prior: np.ndarray, y: np.ndarray) -> dict:
    """Best s for rows with their cell `counts` and `prior` probabilities: a log grid, then a local refine
    between the best point's neighbours. Shared by the Phase 5 table and the Phase 5b chain."""
    grid_loss = np.array([_val_loss(counts, prior, y, s) for s in S_GRID])
    i = int(np.argmin(grid_loss))
    fine = np.logspace(np.log10(S_GRID[max(i - 1, 0)]), np.log10(S_GRID[min(i + 1, len(S_GRID) - 1)]), 25)
    fine_loss = np.array([_val_loss(counts, prior, y, s) for s in fine])
    j = int(np.argmin(fine_loss))
    return {
        "s": float(fine[j]),
        "val_log_loss_served": float(fine_loss[j]),
        "at_grid_edge": bool(i in (0, len(S_GRID) - 1)),  # then widen the grid in a new D-entry, not silently
        "grid": [{"s": float(s), "val_log_loss": float(v)} for s, v in zip(S_GRID, grid_loss, strict=True)],
    }
