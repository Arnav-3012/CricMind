"""Phase 5: score the saved Phase 3 cric2vec (seed 42) on the test seasons, once (D-048(5), (6), (8)).

uv run python -m cricmind.models.score_cric2vec   # artifacts/p5/test_P_cric2vec.npy + reports/p5/

Torch only (mistakes.md OpenMP rule). The checkpoint stores weights, not the encoder; the encoder is refit on
the same train rows with the same recipe, which is deterministic, and `load_state_dict` fails loudly if the
vocab sizes differ from the checkpoint.
"""

import json
from pathlib import Path

import numpy as np
import torch

from cricmind.archetypes.vectors import write_json
from cricmind.matchups.guard import artifact_dir, load_train_test, report_dir
from cricmind.models import p3_data
from cricmind.models.cric2vec import Cric2Vec, Encoder
from cricmind.models.train import DEVICE, Net


def run(processed_dir: Path | None = None) -> dict:
    final = json.loads((p3_data.report_dir() / "cric2vec.json").read_text())["final"]
    if final["ablation"]:
        raise ValueError("the saved net used the ablation inputs; this scorer rebuilds the base encoder")
    checkpoint = torch.load(p3_data.artifact_dir() / "cric2vec_seed42.pt", map_location=DEVICE)
    if checkpoint["dim"] != final["dim"] or checkpoint["epochs"] != final["epochs"]:
        raise ValueError("checkpoint and reports/p3/cric2vec.json disagree on dim or epochs")

    train, test, competitions = load_train_test("cric2vec", processed_dir)  # claims the test seasons
    encoder = Encoder().fit(train, competitions)
    prior = np.bincount(train["y"], minlength=7) / len(train)
    model = Cric2Vec(encoder, final["dim"], prior).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    P = Net(encoder, model).predict_proba(test)

    artifact_dir().mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir() / "test_P_cric2vec.npy", P)
    report = {"test_key": p3_data.val_key(test), "test_rows": len(test), "seed": 42, "dim": final["dim"]}
    write_json(report_dir() / "cric2vec_test.json", report)
    return report


if __name__ == "__main__":
    print(run())
