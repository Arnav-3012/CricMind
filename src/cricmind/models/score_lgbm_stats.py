"""Phase 5: refit LightGBM + target-encoded stats on all of train and score the test seasons, once (D-048(6)).

uv run python -m cricmind.models.score_lgbm_stats   # artifacts/p5/test_P_lgbm_stats.npy + reports/p5/

LightGBM only (mistakes.md OpenMP rule). Recipe = the Phase 3 final one from `reports/p3/lgbm_stats.json`:
tree count and ablation flag are read, nothing is retuned. Train rows get out-of-fold stats, test rows get
stats from all of train, exactly as val did in Phase 3.
"""

import json
from pathlib import Path

import numpy as np

from cricmind.archetypes.vectors import write_json
from cricmind.matchups.guard import artifact_dir, load_train_test, report_dir
from cricmind.models import p3_data
from cricmind.models.baselines import Lgbm
from cricmind.models.lgbm_stats import features


def run(processed_dir: Path | None = None) -> dict:
    final = json.loads((p3_data.report_dir() / "lgbm_stats.json").read_text())["final"]
    train, test, _ = load_train_test("lgbm_stats", processed_dir)  # claims the test seasons
    train_s, test_s = p3_data.with_stats(train, test)
    model = Lgbm(final["trees"], features(final["ablation"]), p3_data.SEEDS[0]).fit(train_s)
    P = model.predict_proba(test_s)

    artifact_dir().mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir() / "test_P_lgbm_stats.npy", P)
    report = {"test_key": p3_data.val_key(test), "test_rows": len(test), "seed": p3_data.SEEDS[0], **final}
    write_json(report_dir() / "lgbm_stats_test.json", report)
    return report


if __name__ == "__main__":
    print(run())
