"""Phase 4 embedding bootstrap worker: retrain cric2vec on each match-bootstrap resample (D-044(6)).

uv run python -m cricmind.archetypes.emb_boot   # artifacts/p4/emb_boot/bNN/, reports/p4/emb_boot.json

Recipe = the Phase 3 final one read from reports/p3/cric2vec.json (dim, epochs, no ablation), seed 42, no
early stopping, so no validation rows are needed. Resumable: a resample whose embeddings exist is skipped.
Torch only: the clustering (cluster.py) runs in another process (mistakes.md OpenMP rule, D-044(12)).
"""

import json
import time
from pathlib import Path

from cricmind import config
from cricmind.archetypes.vectors import B, boot_dir, load_train, report_dir, resample_matches, write_json
from cricmind.models.p3_data import report_dir as p3_report_dir
from cricmind.models.train import DEVICE, fit_net, save_artifacts


def recipe() -> dict:
    final = json.loads((p3_report_dir() / "cric2vec.json").read_text())["final"]
    if final["ablation"]:
        raise ValueError("D-044 retrains the base cric2vec; the Phase 3 final used the ablation inputs")
    return final


def run(processed_dir: Path | None = None, n_boot: int = B) -> dict:
    final = recipe()
    train, competitions = load_train(processed_dir)  # competition vocab fixed from the real train set
    path = report_dir() / "emb_boot.json"
    seconds = json.loads(path.read_text())["seconds"] if path.exists() else {}
    for b in range(n_boot):
        out = boot_dir(b)
        if (out / "embeddings.npy").exists():
            continue
        start = time.time()
        res = resample_matches(train, b)
        net, _ = fit_net(res, competitions, final["dim"], config.SEED, epochs=final["epochs"])
        save_artifacts(net, res, final["dim"], final["epochs"], out)
        seconds[str(b)] = round(time.time() - start, 1)
        write_json(path, _report(final, n_boot, seconds))  # partial progress survives an interrupted run
        print(f"resample {b + 1}/{n_boot} done in {seconds[str(b)]}s")
    report = _report(final, n_boot, seconds)
    write_json(path, report)
    return report


def _report(final: dict, n_boot: int, seconds: dict) -> dict:
    return {
        "n_boot": n_boot,
        "dim": final["dim"],
        "epochs": final["epochs"],
        "seed": config.SEED,
        "device": str(DEVICE),
        "resample_seeds": [config.SEED + b for b in range(n_boot)],
        "seconds": seconds,
    }


if __name__ == "__main__":
    r = run()
    print(f"emb_boot: {len(r['seconds'])} resamples on record, dim {r['dim']}, epochs {r['epochs']}")
