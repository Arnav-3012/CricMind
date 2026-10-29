"""Phase 5 step 1: fit the matchup tables on train, choose the prior strength s on val (D-048, D-050).

No test data is read.

uv run python -m cricmind.matchups.val_fit   # reports/p5/val_fit.json, artifacts/matchups_{arm}.parquet

`s` is frozen here before `headline` opens the test seasons. Val log-losses below were used to choose s, so
they are optimistic for the tables; the test numbers in `headline` are the result. Arms: stats and emb (the
D-048 labels) plus stats_ext and emb_ext (D-050: the same clusters, labels also for lighter players).
"""

from pathlib import Path

import numpy as np

from cricmind import config
from cricmind.archetypes.vectors import write_json
from cricmind.eval.metrics import row_log_loss
from cricmind.matchups.guard import report_dir
from cricmind.matchups.table import ARM_NAMES, build_arm, fit_strength
from cricmind.models.baselines import GlobalFreq, PhaseFreq
from cricmind.models.p3_data import load_frames, val_key


def run(processed_dir: Path | None = None) -> dict:
    train, val, _ = load_frames(processed_dir)  # train and val only; the test seasons stay unread
    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    y = val["y"].to_numpy()
    phase_loss = row_log_loss(phase_freq.predict_proba(val), y)

    report: dict = {"val_key": val_key(val), "val_rows": len(val), "arms": {}}
    for name in ARM_NAMES:
        table, labels_info = build_arm(name, train, phase_freq)
        fit = fit_strength(table, val)
        served = table.served(val)
        loss = row_log_loss(table.predict_proba(val, fit["s"]), y)
        n_cell = table.counts.sum(axis=1)
        report["arms"][name] = {
            "k_batter": table.k_bat,
            "k_bowler": table.k_bowl,
            "cells": table.n_cells,
            "cells_empty": int((n_cell == 0).sum()),
            "cell_n_min_nonzero": int(n_cell[n_cell > 0].min()),
            "cell_n_median": float(np.median(n_cell)),
            "train_rows_in_cells": int(n_cell.sum()),
            "val_served_share": float(served.mean()),
            "labels": labels_info,
            **fit,
            "val_log_loss_all_rows": float(loss.mean()),
            "phase_val_log_loss_all_rows": float(phase_loss.mean()),
            "phase_val_log_loss_served": float(phase_loss[served].mean()),
        }
        config.ARTIFACTS.mkdir(exist_ok=True)
        table.to_frame(fit["s"]).to_parquet(config.ARTIFACTS / f"matchups_{name}.parquet", index=False)
    write_json(report_dir() / "val_fit.json", report)
    return report


if __name__ == "__main__":
    r = run()
    for name, a in r["arms"].items():
        agree = [v["agrees_with_phase4"] for v in a["labels"].values()]
        print(
            f"{name}: s={a['s']:.1f} edge={a['at_grid_edge']} served={a['val_served_share']:.3f} "
            f"table {a['val_log_loss_served']:.5f} vs phase {a['phase_val_log_loss_served']:.5f} (served)"
            + (f" agree={min(agree):.3f}" if agree else "")
        )
