"""Phase 3 LightGBM + target-encoded stats: the strong non-embedding baseline (D-038(3), (4), (6)).

uv run python -m cricmind.models.lgbm_stats   # reports/p3/lgbm_stats.json, artifacts/p3/val_loss_*.npy

Runs in its own process, never with torch (see p3_data). Tree count and the player-at-venue ablation come from
the 2023 holdout; three seeds are refit on all of train and scored on val once.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.eval.metrics import row_log_loss
from cricmind.features.context import FEATURES
from cricmind.models.baselines import Lgbm
from cricmind.models.p3_data import (
    ABLATION_COLS,
    ABLATION_MIN_GAIN,
    BASE_COLS,
    SEEDS,
    artifact_dir,
    holdout_split,
    load_frames,
    report_dir,
    val_key,
    with_stats,
)


def features(ablation: bool) -> list[str]:
    return FEATURES + BASE_COLS + (ABLATION_COLS if ablation else [])


def tune(fit_part: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    fit_s, hold_s = with_stats(fit_part, holdout)
    arms = {a: Lgbm(features=features(a == "ablation")).tune(fit_s, hold_s) for a in ("base", "ablation")}
    gain = arms["base"]["holdout_log_loss"] - arms["ablation"]["holdout_log_loss"]
    return {
        "holdout_year": config.TUNE_HOLDOUT_YEAR,
        "fit_rows": len(fit_part),
        "holdout_rows": len(holdout),
        **arms,
        "ablation_gain": gain,
        "ablation_min_gain": ABLATION_MIN_GAIN,
        "use_ablation": bool(gain >= ABLATION_MIN_GAIN),
    }


def run(processed_dir: Path | None = None) -> dict:
    train, val, _ = load_frames(processed_dir)
    tuning = tune(*holdout_split(train))
    use = tuning["use_ablation"]
    trees = tuning["ablation" if use else "base"]["best_iteration"]

    train_s, val_s = with_stats(train, val)  # train rows out-of-fold, val rows from all of train
    y = val["y"].to_numpy()
    losses = np.array(
        [
            row_log_loss(Lgbm(trees, features(use), seed).fit(train_s).predict_proba(val_s), y)
            for seed in SEEDS
        ]
    )
    report = {
        "seeds": list(SEEDS),
        "val_key": val_key(val),
        "tuning": tuning,
        "final": {"trees": trees, "ablation": use},
        "per_seed_val_log_loss": [float(x) for x in losses.mean(axis=1)],
    }
    artifact_dir().mkdir(parents=True, exist_ok=True)
    report_dir().mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir() / "val_loss_lgbm_stats.npy", losses)  # (seeds, val rows)
    (report_dir() / "lgbm_stats.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    print(f"lgbm+stats per-seed val log-loss {r['per_seed_val_log_loss']}  final {r['final']}")
