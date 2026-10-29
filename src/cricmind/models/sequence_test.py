"""Phase 3 step 5, the D-016 sequence test: do the last 12 balls help the MLP beyond the context features?

uv run python -m cricmind.models.sequence_test   # reports/p3/sequence_test.json (needs the train.py outputs)

Arm "without" is the final cric2vec recipe, so its val losses are the saved `val_loss_cric2vec.npy` (same dim,
epochs and seeds, nothing retrained). Arm "with" adds runs and wickets in the previous 12 legal balls and the
last-ball class; it picks its own epoch on the 2023 holdout, is refit on all train with the same three
seeds and scored on val once (D-038(8)). Pass: seed-mean gain <= -THRESHOLD and the paired CI excludes 0,
which builds Phase 3b. Torch only (see p3_data).
"""

import json
from pathlib import Path

import numpy as np

from cricmind import config
from cricmind.eval.metrics import paired_bootstrap_diff, row_log_loss
from cricmind.models.p3_data import (
    SEEDS,
    artifact_dir,
    holdout_split,
    load_frames,
    report_dir,
    val_key,
    with_stats,
)
from cricmind.models.train import _extra, fit_net

THRESHOLD = 0.002  # nats; D-016, kept by D-035(3)
SEED_SPREAD_LIMIT = 0.001  # D-038(8): above this in either arm, revisit the threshold (D-035)


def verdict(diff: dict, spread: dict) -> dict:
    """`diff` is the paired bootstrap of (with - without); negative = sequence helps."""
    return {
        "gain_at_least_threshold": bool(diff["value"] <= -THRESHOLD),
        "ci_excludes_0": bool(diff["hi"] < 0),
        "seed_spread_ok": bool(max(spread.values()) <= SEED_SPREAD_LIMIT),
        "build_phase_3b": bool(diff["value"] <= -THRESHOLD and diff["hi"] < 0),
    }


def run(processed_dir: Path | None = None) -> dict:
    train, val, competitions = load_frames(processed_dir, sequence=True)
    cric = json.loads((report_dir() / "cric2vec.json").read_text())
    if cric["val_key"] != val_key(val):
        raise ValueError("val rows differ from the cric2vec run; re-run train.py")
    dim, use_abl = cric["final"]["dim"], cric["final"]["ablation"]
    without = np.load(artifact_dir() / "val_loss_cric2vec.npy")

    fit_part, holdout = holdout_split(train)
    if use_abl:  # the player-at-venue columns, only if the cric2vec run kept them
        fit_part, holdout = with_stats(fit_part, holdout)
        train, val = with_stats(train, val)
    _, curve = fit_net(
        fit_part, competitions, dim, SEEDS[0], sequence=True, extra=_extra(use_abl), holdout=holdout
    )
    epochs = int(np.argmin(curve)) + 1

    y = val["y"].to_numpy()
    with_seq = np.array(
        [
            row_log_loss(
                fit_net(train, competitions, dim, s, sequence=True, extra=_extra(use_abl), epochs=epochs)[
                    0
                ].predict_proba(val),
                y,
            )
            for s in SEEDS
        ]
    )
    assert with_seq.shape == without.shape == (len(SEEDS), len(val))
    diff = paired_bootstrap_diff(with_seq.mean(axis=0), without.mean(axis=0), val["match_id"].to_numpy())
    spread = {
        "without_sequence": float(np.ptp(without.mean(axis=1))),  # ndarray.ptp is gone in NumPy 2
        "with_sequence": float(np.ptp(with_seq.mean(axis=1))),
    }
    report = {
        "snapshot": config.DATA_SNAPSHOT,
        "seeds": list(SEEDS),
        "threshold": THRESHOLD,
        "dim": dim,
        "epochs": {"without_sequence": cric["final"]["epochs"], "with_sequence": epochs},
        "holdout_log_loss_with_sequence": float(min(curve)),
        "per_seed_val_log_loss": {
            "without_sequence": [float(x) for x in without.mean(axis=1)],
            "with_sequence": [float(x) for x in with_seq.mean(axis=1)],
        },
        "val_log_loss": {
            "without_sequence": float(without.mean()),
            "with_sequence": float(with_seq.mean()),
        },
        "with_minus_without": diff,
        "seed_spread": spread,
        "verdict": verdict(diff, spread),
    }
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "sequence_test.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    print(f"val log-loss {r['val_log_loss']}  with - without {r['with_minus_without']}")
    print(f"verdict {r['verdict']}")
