"""D-042 diagnostic: how much identity do trees get from per-player rates? (2023 holdout only, LightGBM only)

uv run python -m cricmind.models.no_identity_lgbm   # reports/p3/no_identity_lgbm.json

Fits LightGBM on years before the holdout year, with and without the target-encoded stats, and scores the
holdout year. With the MLP numbers from `no_identity.json` this splits the holdout gate gap into
  C = MLP without ids - tree without ids   (the MLP reads the match context less well)
  I = G - E                                (the MLP gets less identity gain than the tree: G = tree gain from
                                            stats, E = MLP gain from embeddings)
and C + I = MLP with ids - tree with stats, the gap on this holdout. Val is never scored. Imports no torch.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.eval.metrics import paired_bootstrap_diff, row_log_loss
from cricmind.features.context import FEATURES
from cricmind.models.baselines import Lgbm
from cricmind.models.p3_data import BASE_COLS, holdout_split, load_frames, report_dir, with_stats

TIE = 0.001  # D-042: C and I closer than this are called a tie (half the 0.002 bar)


def _holdout_row_loss(
    features: list[str], fit: pd.DataFrame, holdout: pd.DataFrame
) -> tuple[np.ndarray, int]:
    """Early-stop on the holdout to pick the tree count, refit on `fit`, return the per-row holdout loss."""
    model = Lgbm(features=features)
    trees = model.tune(fit, holdout)["best_iteration"]
    model.fit(fit)
    return row_log_loss(model.predict_proba(holdout), holdout["y"].to_numpy()), trees


def split_gap(mlp_with: float, mlp_without: float, tree_without: float, tree_stats: float) -> dict:
    """The decomposition in the module docstring, from four holdout log-losses."""
    c = mlp_without - tree_without
    g = tree_without - tree_stats
    e = mlp_without - mlp_with
    i = g - e
    larger = "context" if c > i + TIE else "identity_extraction" if i > c + TIE else "tie"
    return {
        "C_context_gap": c,
        "G_tree_identity_gain": g,
        "E_mlp_identity_gain": e,
        "I_identity_extraction_gap": i,
        "gap_on_holdout": mlp_with - tree_stats,  # equals C + I
        "larger_source": larger,
    }


def run(processed_dir: Path | None = None) -> dict:
    train, _, _ = load_frames(processed_dir)  # val is built by the shared loader, never scored
    fit, holdout = holdout_split(train)
    plain, trees_plain = _holdout_row_loss(FEATURES, fit, holdout)
    fit_s, hold_s = with_stats(fit, holdout)
    stats, trees_stats = _holdout_row_loss(FEATURES + BASE_COLS, fit_s, hold_s)
    g = paired_bootstrap_diff(plain, stats, holdout["match_id"].to_numpy())  # positive: stats help the tree
    mlp = json.loads((report_dir() / "no_identity.json").read_text())["holdout_log_loss"]
    report = {
        "holdout_year": config.TUNE_HOLDOUT_YEAR,
        "holdout_log_loss": {
            "tree_without_stats": float(plain.mean()),
            "tree_with_stats": float(stats.mean()),
        },
        "trees": {"without_stats": trees_plain, "with_stats": trees_stats},
        "G_without_minus_with_stats": g,
        "mlp_holdout_log_loss": mlp,
        "split": split_gap(mlp["with_ids"], mlp["without_ids"], float(plain.mean()), float(stats.mean())),
        "tie_margin": TIE,
    }
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "no_identity_lgbm.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    print(f"holdout log-loss {r['holdout_log_loss']}  G {r['G_without_minus_with_stats']}")
    print(f"split {r['split']}")
