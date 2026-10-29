"""D-042 diagnostic: would a smaller embedding than the grid's floor do better? (holdout only, torch only)

uv run python -m cricmind.models.small_dims   # reports/p3/small_dims.json (needs reports/p3/cric2vec.json)

The D-038 grid was {8, 16, 32}, holdout loss rose with size, and the chosen size 8 is the grid's floor. This
tries sizes 2 and 4 with the same recipe (seed, optimiser, early stopping on the holdout, trained on years
before it) and compares each with the chosen size's holdout loss from `cric2vec.json`. Val is never scored and
nothing is refit: a gain only opens a new D-entry (D-042), it does not change the gate model (D-040).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.models.p3_data import SEEDS, holdout_split, load_frames, report_dir
from cricmind.models.train import fit_net

SMALL_DIMS = (2, 4)
MIN_GAIN = 0.002  # nats on the holdout, the D-038(4) bar


def small_dim_losses(
    fit: pd.DataFrame, holdout: pd.DataFrame, competitions: list[str], dims: tuple[int, ...], seed: int
) -> dict[int, dict]:
    """Best holdout log-loss and the epoch that reached it, per embedding size."""
    out = {}
    for dim in dims:
        curve = fit_net(fit, competitions, dim, seed, holdout=holdout)[1]
        out[dim] = {"holdout_log_loss": float(min(curve)), "epochs": int(np.argmin(curve)) + 1}
    return out


def decision(base_loss: float, results: dict[int, dict]) -> dict:
    """Pre-registered reading (D-042): a smaller size matters if it beats the chosen size by MIN_GAIN."""
    best = min(results, key=lambda d: results[d]["holdout_log_loss"])
    gain = base_loss - results[best]["holdout_log_loss"]
    return {
        "best_small_dim": best,
        "gain_vs_chosen": gain,
        "min_gain": MIN_GAIN,
        "smaller_dim_helps": bool(gain >= MIN_GAIN),
    }


def run(processed_dir: Path | None = None) -> dict:
    train, _, competitions = load_frames(processed_dir)  # val is built by the shared loader, never scored
    fit, holdout = holdout_split(train)
    tuning = json.loads((report_dir() / "cric2vec.json").read_text())["tuning"]
    base = tuning["holdout_log_loss_by_dim"][str(tuning["dim"])]
    results = small_dim_losses(fit, holdout, competitions, SMALL_DIMS, SEEDS[0])
    report = {
        "holdout_year": config.TUNE_HOLDOUT_YEAR,
        "seed": SEEDS[0],
        "chosen_dim": tuning["dim"],
        "chosen_dim_holdout_log_loss": base,
        "small_dims": {str(d): v for d, v in results.items()},
        "decision": decision(base, results),
    }
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "small_dims.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    print(f"chosen {r['chosen_dim']}: {r['chosen_dim_holdout_log_loss']:.5f}  small {r['small_dims']}")
    print(f"decision {r['decision']}")
