"""D-046 after-the-fact diagnostics on the Phase 4 clusterings: clusterwise stability and one cleaner stats
vector. Train only, the same 20 resamples; the Phase 4 gate outcome does not change. sklearn only.

uv run python -m cricmind.archetypes.diagnostics   # reports/p4/diagnostics.json (needs cluster.py outputs)
"""

import json

import joblib
import numpy as np
import pandas as pd

from cricmind.archetypes import vectors as vec
from cricmind.archetypes.cluster import build_cells
from cricmind.archetypes.stability import BOOT_ARI_MIN, Cell, make_model

STABLE, PATTERN, DISSOLVED = 0.75, 0.6, 0.5  # Hennig / fpc clusterboot thresholds (D-046(a))
IMPROVES_BY = 0.1  # D-046(b)
KEEP_DIMS = [i for i, c in enumerate(vec.STAT_COLUMNS) if not c.startswith("dismissal")]  # 9 of 12


def jaccard_by_cluster(ref: np.ndarray, boot: np.ndarray) -> np.ndarray:
    """For each reference cluster, the best Jaccard overlap with any resample cluster. Labels are arbitrary
    ids, so the best match is searched, never assumed."""
    overlap = np.zeros((ref.max() + 1, boot.max() + 1))
    np.add.at(overlap, (ref, boot), 1.0)
    union = overlap.sum(axis=1)[:, None] + overlap.sum(axis=0)[None, :] - overlap
    return (overlap / union).max(axis=1)


def reading(mean_jaccard: float) -> str:
    if mean_jaccard >= STABLE:
        return "stable"
    if mean_jaccard >= PATTERN:
        return "pattern"
    return "dissolved" if mean_jaccard <= DISSOLVED else "untrusted"


def variant_verdict(base: list[dict], variant: list[dict]) -> dict:
    best_base = max(r["boot_ari_mean"] for r in base)
    best = max(variant, key=lambda r: r["boot_ari_mean"])
    verdict = (
        "fixes"
        if best["boot_ari_mean"] >= BOOT_ARI_MIN
        else "improves"
        if best["boot_ari_mean"] >= best_base + IMPROVES_BY
        else "no_change"
    )
    return {
        "verdict": verdict,
        "best_k": best["k"],
        "best_boot_ari": best["boot_ari_mean"],
        "base_best": best_base,
    }


def clusterwise(cell: Cell, cfg: dict, model) -> list[dict]:
    keep = cell.n_balls >= cfg["min_balls"]
    X, boots = cell.X[keep], [b[keep] for b in cell.boots]
    ref = model[-1].labels_  # the saved fit, so cluster ids match profiles.json and the parquet
    assert len(ref) == len(X)
    per_boot = np.array(
        [jaccard_by_cluster(ref, make_model(cfg["k"], cfg["pca"]).fit(Xb)[-1].labels_) for Xb in boots]
    )
    return [
        {
            "cluster": c,
            "size": int((ref == c).sum()),
            "mean_jaccard": float(per_boot[:, c].mean()),
            "p2_5_jaccard": float(np.percentile(per_boot[:, c], 2.5)),
            "reading": reading(per_boot[:, c].mean()),
        }
        for c in range(cfg["k"])
    ]


def run(train: pd.DataFrame, n_boot: int = vec.B) -> dict:
    clusters = json.loads((vec.report_dir() / "clusters.json").read_text())["cells"]
    cells, _, _ = build_cells(train, n_boot)
    out: dict = {"clusterwise": {}, "dismissal_free_vector": {}}
    for (rep, role), cell in cells.items():
        key, cfg = f"{rep}_{role}", clusters[f"{rep}_{role}"]
        model = joblib.load(vec.artifact_dir() / f"kmeans_{key}.joblib")
        out["clusterwise"][key] = clusterwise(cell, cfg, model)
        if rep == "stats":  # D-046(b): same k table, same eligibility and PCA as the chosen cell
            slim = Cell(cell.n_balls, cell.X[:, KEEP_DIMS], [b[:, KEEP_DIMS] for b in cell.boots])
            table = slim.table(cfg["min_balls"], cfg["pca"])
            base = cfg["tables"][f"{cfg['min_balls']}_{'pca' if cfg['pca'] else 'raw'}"]
            out["dismissal_free_vector"][role] = {
                "dims": [vec.STAT_COLUMNS[i] for i in KEEP_DIMS],
                **variant_verdict(base, table),
                "table": table,
            }
    vec.write_json(vec.report_dir() / "diagnostics.json", out)
    return out


if __name__ == "__main__":
    train, _ = vec.load_train()
    result = run(train)
    for key, clusters in result["clusterwise"].items():
        print(key, [(c["cluster"], round(c["mean_jaccard"], 2), c["reading"]) for c in clusters])
    for role, v in result["dismissal_free_vector"].items():
        print(
            f"9-dim {role}: {v['verdict']} "
            f"(best k={v['best_k']}, ARI {v['best_boot_ari']:.3f} vs 12-dim {v['base_best']:.3f})"
        )
