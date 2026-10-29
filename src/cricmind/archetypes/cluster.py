"""Phase 4 entry point: archetypes for batters and bowlers from the stats vectors and the cric2vec embeddings.

uv run python -m cricmind.archetypes.emb_boot   # first, in its own process (torch)
uv run python -m cricmind.archetypes.cluster    # then this (sklearn only)

Reads artifacts/p3/ (reference embeddings) and artifacts/p4/emb_boot/ (resampled embeddings); writes
reports/p4/{clusters,seeding,comparison,profiles}.json and
artifacts/p4/{archetypes_*.parquet, kmeans_*.joblib}.
Train rows only; val and test are never loaded (D-044(1)).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from cricmind import config
from cricmind.archetypes import vectors as vec
from cricmind.archetypes.profiles import profile
from cricmind.archetypes.stability import BOOT_ARI_MIN, KS, Cell, make_model, run_ladder, seeding_experiment
from cricmind.models.p3_data import artifact_dir as p3_artifact_dir

HDBSCAN_MIN_CLUSTER = 20  # D-044(9)


def _gmm(Z: np.ndarray, kmeans_labels: np.ndarray, k: int) -> dict:
    """GMM (full covariance) comparison: BIC by k and, at the chosen k, agreement with k-means (D-044(9))."""
    fits = {
        j: GaussianMixture(j, covariance_type="full", n_init=3, random_state=config.SEED).fit(Z) for j in KS
    }
    proba = fits[k].predict_proba(Z)
    return {
        "bic_by_k": {str(j): float(g.bic(Z)) for j, g in fits.items()},
        "ari_vs_kmeans": float(adjusted_rand_score(kmeans_labels, proba.argmax(axis=1))),
        "mean_max_prob": float(proba.max(axis=1).mean()),
        "labels": proba.argmax(axis=1),
        "prob": proba.max(axis=1),
    }


def _hdbscan(Z: np.ndarray, kmeans_labels: np.ndarray) -> dict:
    h = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER, copy=True).fit(Z)
    keep = h.labels_ >= 0
    ari = adjusted_rand_score(kmeans_labels[keep], h.labels_[keep]) if keep.sum() > 1 else None
    return {
        "n_clusters": int(h.labels_.max() + 1),
        "noise_share": float((~keep).mean()),
        "ari_vs_kmeans_non_noise": None if ari is None else float(ari),
        "labels": h.labels_,
        "outlier": 1.0 - h.probabilities_,  # sklearn gives membership strength, not GLOSH scores
    }


def _rows(ids: pd.Index, role: str, rep: str, method: str, k: int, labels, prob=None, outlier=None):
    return pd.DataFrame(
        {
            "player_id": ids,
            "role": role,
            "rep": rep,
            "method": method,
            "k": k,
            "label": np.asarray(labels, dtype="int64"),
            "prob": np.nan if prob is None else prob,
            "outlier": np.nan if outlier is None else outlier,
        }
    )


def run_cell(
    rep: str, role: str, cell: Cell, stats: pd.DataFrame, artifacts: Path
) -> tuple[dict, pd.Series, pd.DataFrame, list[dict]]:
    """Ladder, final fit, profiles and comparison-only methods for one (representation, role). Returns the
    cell's result, its k-means labels by player id, the archetype rows and the profiles."""
    result = run_ladder(cell)
    keep = cell.n_balls >= result["min_balls"]
    X, ids, k = cell.X[keep], stats.index[keep], result["k"]
    model = make_model(k, result["pca"]).fit(X)
    kmeans = model[-1]
    Z = model[:-1].transform(X)
    joblib.dump(model, artifacts / f"kmeans_{rep}_{role}.joblib")  # Phase 5 assigns by nearest centroid
    gmm, hdb = _gmm(Z, kmeans.labels_, k), _hdbscan(Z, kmeans.labels_)
    result["gate_pass"] = bool(result["chosen"]["boot_ari_mean"] >= BOOT_ARI_MIN)
    result["comparison_only"] = {
        "gmm": {key: v for key, v in gmm.items() if key not in ("labels", "prob")},
        "hdbscan": {key: v for key, v in hdb.items() if key not in ("labels", "outlier")},
    }
    rows = pd.concat(
        [
            _rows(ids, role, rep, "kmeans", k, kmeans.labels_),
            _rows(ids, role, rep, "gmm", k, gmm["labels"], prob=gmm["prob"]),
            _rows(ids, role, rep, "hdbscan", hdb["n_clusters"], hdb["labels"], outlier=hdb["outlier"]),
        ],
        ignore_index=True,
    )
    profiles = profile(stats[keep], Z, kmeans.labels_, kmeans.cluster_centers_)
    return result, pd.Series(kmeans.labels_, index=ids), rows, profiles


def build_cells(
    train: pd.DataFrame, n_boot: int
) -> tuple[dict[tuple[str, str], Cell], dict[str, pd.DataFrame], dict]:
    """The four (representation, role) cells over the superset of players, the reference stats vectors per
    role, and the D-045 UNK-fallback counts. Shared with `diagnostics`."""
    floor = min(vec.SENSITIVITY_FLOORS)  # the superset; each ladder rung slices it
    eligible = {role: vec.eligibility(train, role, floor) for role in vec.ROLES}
    ids = {role: counts.index for role, counts in eligible.items()}
    n_balls = {role: counts.to_numpy() for role, counts in eligible.items()}
    stats = {role: vec.stats_vectors(train, role, ids[role]) for role in vec.ROLES}
    stats_boots = vec.stats_bootstrap(train, ids, n_boot)
    boot_dirs = [vec.boot_dir(b) for b in range(n_boot)]
    cells = {
        ("stats", role): Cell(n_balls[role], stats[role].to_numpy(), stats_boots[role]) for role in vec.ROLES
    } | {
        ("emb", role): Cell(
            n_balls[role],
            vec.embedding_rows(p3_artifact_dir(), role, ids[role]),
            [vec.embedding_rows(d, role, ids[role], allow_unk=True) for d in boot_dirs],
        )
        for role in vec.ROLES
    }
    # D-045: eligible players that were UNK in a resample, per eligibility floor and resample
    unk_fallback = {
        role: {
            str(f): [
                int((~vec.has_own_embedding(d, role, ids[role]) & (n_balls[role] >= f)).sum())
                for d in boot_dirs
            ]
            for f in vec.SENSITIVITY_FLOORS
        }
        for role in vec.ROLES
    }
    return cells, stats, unk_fallback


def run(
    train: pd.DataFrame, n_boot: int = vec.B, artifacts: Path | None = None, reports: Path | None = None
) -> dict:
    artifacts, reports = artifacts or vec.artifact_dir(), reports or vec.report_dir()
    artifacts.mkdir(parents=True, exist_ok=True)
    cells, stats, unk_fallback = build_cells(train, n_boot)

    clusters, profiles, seeding, frames, labels = {}, {}, {}, {}, {}  # each keyed "<rep>_<role>"
    for (rep, role), cell in cells.items():
        key = f"{rep}_{role}"
        out = run_cell(rep, role, cell, stats[role], artifacts)
        clusters[key], labels[key], frames[key], profiles[key] = out
        base = cell.n_balls >= config.MIN_TRAIN_BALLS_ARCHETYPE
        seeding[key] = seeding_experiment(cell.X[base])
    for rep in ("stats", "emb"):
        pd.concat([frames[f"{rep}_{role}"] for role in vec.ROLES], ignore_index=True).to_parquet(
            artifacts / f"archetypes_{rep}.parquet", index=False
        )

    comparison = {}
    for role in vec.ROLES:  # stats vs embeddings side by side, and do they find the same groups?
        by_rep = {rep: clusters[f"{rep}_{role}"] for rep in ("stats", "emb")}
        a, b = labels[f"stats_{role}"], labels[f"emb_{role}"]
        shared = a.index.intersection(b.index)
        comparison[role] = {
            rep: {
                **{m: c[m] for m in ("k", "ladder_step", "stable", "gate_pass", "n_players")},
                "boot_ari_mean": c["chosen"]["boot_ari_mean"],
                "seed_ari_mean": c["chosen"]["seed_ari_mean"],
                "silhouette": c["chosen"]["silhouette"],
            }
            for rep, c in by_rep.items()
        } | {
            "n_players_shared": len(shared),
            "cross_rep_ari": float(adjusted_rand_score(a[shared], b[shared])),
        }
    report = {
        "n_boot": n_boot,
        "gate_min_boot_ari": BOOT_ARI_MIN,
        "emb_unk_fallback": unk_fallback,
        "cells": clusters,
    }
    vec.write_json(reports / "clusters.json", report)
    vec.write_json(reports / "seeding.json", seeding)
    vec.write_json(reports / "comparison.json", comparison)
    vec.write_json(reports / "profiles.json", profiles)
    return {"clusters": clusters, "comparison": comparison}


if __name__ == "__main__":
    train, _ = vec.load_train()
    out = run(train)
    for key, c in out["clusters"].items():
        ari = c["chosen"]["boot_ari_mean"]
        print(f"{key}: k={c['k']} ladder step={c['ladder_step']} boot ARI={ari:.3f} gate={c['gate_pass']}")
