"""Phase 5b role + performance vectors and their clustering (D-052). sklearn only.

uv run python -m cricmind.archetypes.roles   # artifacts/p4/{archetypes_role.parquet, kmeans_role_*.joblib},
                                             # reports/p5b/role_clusters.json

A batter's vector mixes how they perform (strike rate, dot %, boundary % per phase) with how they are used
(share of balls faced per phase, and where they come in: over and wickets down at their first ball). A
bowler's mixes performance with the share of their balls bowled per phase. Usage is a captain's decision, so
it is far less noisy than an outcome rate. Train rows only; the same k rule, ladder and match-bootstrap
resamples as Phase 4.

from cricmind.archetypes.roles import role_vectors
"""

import joblib
import numpy as np
import pandas as pd

from cricmind import config
from cricmind.archetypes import vectors as vec
from cricmind.archetypes.stability import Cell, make_model, run_ladder
from cricmind.features.context import PHASES

M_ENTRY = 10  # innings of pseudo-evidence pulling a batter's entry point toward the population mean
PERF = vec.STAT_COLUMNS[:9]  # strike rate, dot %, boundary % x 3 phases; dismissal rate dropped (D-047)
USAGE = [f"share_{p}" for p in PHASES]
ENTRY = ["entry_over", "entry_wickets"]


def _bucket(over: pd.Series) -> np.ndarray:
    """0 / 1 / 2 = powerplay / middle / death from the over, so reduced-match rows (no phase) count too."""
    first, second = config.PHASE_EDGES
    return np.select([over < first, over < second], [0, 1], 2)


def usage_shares(
    frame: pd.DataFrame, role: str, ids: pd.Index, m: float = config.STATS_SHRINK_M
) -> pd.DataFrame:
    """Share of the player's balls in each phase, shrunk toward the population share with m pseudo-balls."""
    bucket = _bucket(frame["over"])
    counts = pd.crosstab(frame[vec.ROLES[role]], bucket)
    counts = counts.reindex(index=ids, columns=range(3), fill_value=0).to_numpy(dtype="float64")
    prior = np.bincount(bucket, minlength=3) / len(frame)
    shares = (counts + m * prior) / (counts.sum(axis=1, keepdims=True) + m)
    return pd.DataFrame(shares, index=ids, columns=USAGE)


def entry_features(frame: pd.DataFrame, ids: pd.Index, m: float = M_ENTRY) -> pd.DataFrame:
    """Mean over and wickets down at the batter's first ball of an innings, shrunk toward the population mean.
    A first ball is a row with no earlier ball faced; a resampled match drawn twice counts twice."""
    first = frame[frame["batter_balls_faced"] == 0]
    cols = ["over", "wickets_before"]
    sums = first.groupby("batter_id")[cols].sum().reindex(ids, fill_value=0.0)
    n = first.groupby("batter_id").size().reindex(ids, fill_value=0).to_numpy()[:, None]
    mean = (sums.to_numpy() + m * first[cols].mean().to_numpy()) / (n + m)
    return pd.DataFrame(mean, index=ids, columns=ENTRY)


def role_vectors(frame: pd.DataFrame, role: str, ids: pd.Index) -> pd.DataFrame:
    """Batter: 9 performance + 3 usage + 2 entry dims. Bowler: 9 performance + 3 usage dims."""
    parts = [vec.stats_vectors(frame, role, ids)[PERF], usage_shares(frame, role, ids)]
    if role == "batter":
        parts.append(entry_features(frame, ids))
    return pd.concat(parts, axis=1)


def role_bootstrap(train: pd.DataFrame, ids_by_role: dict[str, pd.Index], n_boot: int = vec.B) -> dict:
    """Per role, `n_boot` vector matrices recomputed on the shared Phase 4 resamples."""
    out: dict[str, list] = {role: [] for role in ids_by_role}
    for b in range(n_boot):
        res = vec.resample_matches(train, b)
        for role, ids in ids_by_role.items():
            out[role].append(role_vectors(res, role, ids).to_numpy())
    return out


def _profile(X: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """Per cluster: size, the 3 features furthest from the population (z) and their raw cluster means."""
    z = (X - X.mean()) / X.std(ddof=0).replace(0.0, 1.0)
    rows = []
    for c in sorted(set(labels)):
        zc = z[labels == c].mean()
        top = zc.abs().sort_values(ascending=False).index[:3]
        rows.append(
            {
                "cluster": int(c),
                "size": int((labels == c).sum()),
                "top": {f: {"z": float(zc[f]), "mean": float(X.loc[labels == c, f].mean())} for f in top},
            }
        )
    return rows


def run(train: pd.DataFrame, n_boot: int = vec.B) -> dict:
    floor = min(vec.SENSITIVITY_FLOORS)  # the superset; each ladder rung slices it
    eligible = {role: vec.eligibility(train, role, floor) for role in vec.ROLES}
    ids = {role: counts.index for role, counts in eligible.items()}
    boots = role_bootstrap(train, ids, n_boot)
    arch = config.ARTIFACTS / "p4"
    arch.mkdir(parents=True, exist_ok=True)

    report, frames = {"n_boot": n_boot, "cells": {}}, []
    for role in vec.ROLES:
        X = role_vectors(train, role, ids[role])
        cell = Cell(eligible[role].to_numpy(), X.to_numpy(), boots[role])
        result = run_ladder(cell)
        keep = cell.n_balls >= result["min_balls"]
        model = make_model(result["k"], result["pca"]).fit(cell.X[keep])
        labels = model[-1].labels_
        joblib.dump(model, arch / f"kmeans_role_{role}.joblib")
        result["profiles"] = _profile(X[keep], labels)
        result["features"] = list(X.columns)
        report["cells"][role] = result
        frames.append(
            pd.DataFrame(
                {"player_id": ids[role][keep], "role": role, "rep": "role", "method": "kmeans"}
                | {"k": result["k"], "label": labels, "prob": np.nan, "outlier": np.nan}
            )
        )
    pd.concat(frames, ignore_index=True).to_parquet(arch / "archetypes_role.parquet", index=False)
    vec.write_json(config.REPORTS / "p5b" / "role_clusters.json", report)
    return report


if __name__ == "__main__":
    r = run(vec.load_train()[0])
    for role, c in r["cells"].items():
        ch = c["chosen"]
        print(
            f"{role}: k={c['k']} step={c['ladder_step']} stable={c['stable']} players={c['n_players']} "
            f"bootARI={ch['boot_ari_mean']:.3f} silhouette={ch['silhouette']:.3f} "
            f"seedARI={ch['seed_ari_mean']:.3f}"
        )
