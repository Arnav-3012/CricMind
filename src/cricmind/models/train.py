"""Phase 3 cric2vec worker: tune on the 2023 holdout, refit on all train with 3 seeds, score val (D-038).

uv run python -m cricmind.models.train   # reports/p3/cric2vec.json, artifacts/p3/ (val losses, embeddings)
uv run python -m cricmind.models.train --lite   # D-041: reports/p3/cric2vec_lite.json, artifacts/p3/lite/

Embedding dim, epochs and the player-at-venue ablation are chosen on the holdout year, trained on the years
before it; val is scored once and never picks anything (D-032, D-036). The test seasons are never loaded.
Imports no LightGBM: the two libraries crash together on macOS (p3_data). The comparison with
LightGBM + stats is `outcome_compare`.

from cricmind.models.train import fit_net
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from cricmind import config
from cricmind.eval.metrics import log_loss, row_log_loss
from cricmind.models.cric2vec import LITE_NUMERIC, Cric2Vec, Encoder
from cricmind.models.p3_data import (
    ABLATION_COLS,
    ABLATION_MIN_GAIN,
    SEEDS,
    artifact_dir,
    holdout_split,
    lite_artifact_dir,
    load_frames,
    report_dir,
    val_key,
    with_stats,
)

K = len(config.OUTCOME_CLASSES)
DIMS = (8, 16, 32)  # D-038(5)
BATCH, LR, WEIGHT_DECAY = 4096, 1e-3, 1e-5
MAX_EPOCHS, PATIENCE = 30, 3
CHUNK = 65_536  # prediction batch; only memory, not results
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _tensors(encoder: Encoder, frame: pd.DataFrame) -> tuple[torch.Tensor, ...]:
    out = encoder.transform(frame)
    return tuple(torch.as_tensor(out[k], device=DEVICE) for k in ("bat", "bowl", "venue", "x"))


def _predict(model: Cric2Vec, tensors: tuple[torch.Tensor, ...]) -> np.ndarray:
    model.eval()
    n = len(tensors[0])
    with torch.no_grad():
        parts = [
            F.softmax(model(*(t[i : i + CHUNK] for t in tensors)), dim=1).cpu().numpy()
            for i in range(0, n, CHUNK)
        ]
    return np.vstack(parts).astype("float64")


@dataclass
class Net:
    encoder: Encoder
    model: Cric2Vec

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return _predict(self.model, _tensors(self.encoder, frame))


def fit_net(
    fit: pd.DataFrame,
    competitions: list[str],
    dim: int,
    seed: int,
    *,
    sequence: bool = False,
    extra: tuple[str, ...] = (),
    epochs: int | None = None,
    holdout: pd.DataFrame | None = None,
    lite: bool = False,
) -> tuple[Net, list[float]]:
    """Train on `fit`. With `holdout`: early stopping (patience PATIENCE, at most MAX_EPOCHS) and the
    per-epoch holdout log-loss comes back as the curve. With `epochs`: exactly that many (the final refit).
    `lite`: ids + venue + the D-041 situation inputs only, no competition one-hot."""
    torch.manual_seed(seed)
    encoder = Encoder(sequence, extra, LITE_NUMERIC if lite else None, competition=not lite)
    encoder.fit(fit, competitions)
    prior = np.bincount(fit["y"], minlength=K) / len(fit)
    model = Cric2Vec(encoder, dim, prior).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_t = _tensors(encoder, fit)
    y = torch.as_tensor(fit["y"].to_numpy(dtype="int64"), device=DEVICE)
    hold_t = _tensors(encoder, holdout) if holdout is not None else None
    gen = torch.Generator().manual_seed(seed)
    curve: list[float] = []
    for epoch in range(epochs or MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(y), generator=gen).to(DEVICE)
        for i in range(0, len(y), BATCH):
            idx = order[i : i + BATCH]
            loss = F.cross_entropy(model(*(t[idx] for t in train_t)), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if hold_t is not None:
            curve.append(log_loss(_predict(model, hold_t), holdout["y"].to_numpy()))
            if epoch - int(np.argmin(curve)) >= PATIENCE:
                break
    return Net(encoder, model), curve


def _extra(ablation: bool) -> tuple[str, ...]:
    return tuple(ABLATION_COLS) if ablation else ()


def tune(fit_part: pd.DataFrame, holdout: pd.DataFrame, competitions: list[str]) -> dict:
    """Embedding dim and epochs, and whether player-at-venue rates stay, all from the 2023 holdout."""
    fit_s, hold_s = with_stats(fit_part, holdout)
    seed = SEEDS[0]  # lean: one seed for tuning; the final fit uses three (the dim grid is coarse)
    curves = {dim: fit_net(fit_s, competitions, dim, seed, holdout=hold_s)[1] for dim in DIMS}
    best_dim = min(curves, key=lambda d: min(curves[d]))
    abl = fit_net(fit_s, competitions, best_dim, seed, extra=_extra(True), holdout=hold_s)[1]
    gain = min(curves[best_dim]) - min(abl)
    return {
        "holdout_year": config.TUNE_HOLDOUT_YEAR,
        "fit_rows": len(fit_part),
        "holdout_rows": len(holdout),
        "holdout_log_loss_by_dim": {str(d): min(c) for d, c in curves.items()},
        "epochs_by_dim": {str(d): int(np.argmin(c)) + 1 for d, c in curves.items()},
        "dim": best_dim,
        "dim_at_grid_edge": best_dim == max(DIMS),  # D-038 revisit trigger: extend the grid to 64
        "epochs": int(np.argmin(curves[best_dim])) + 1,
        "ablation_epochs": int(np.argmin(abl)) + 1,
        "ablation_gain": gain,
        "ablation_min_gain": ABLATION_MIN_GAIN,
        "use_ablation": bool(gain >= ABLATION_MIN_GAIN),
    }


def tune_lite(fit_part: pd.DataFrame, holdout: pd.DataFrame, competitions: list[str]) -> dict:
    """D-041(2): embedding dim and epochs from the 2023 holdout, same grid and stopping rule as `tune`. No
    target-encoded stats and no ablation: the lite inputs carry none."""
    seed = SEEDS[0]
    curves = {dim: fit_net(fit_part, competitions, dim, seed, holdout=holdout, lite=True)[1] for dim in DIMS}
    best_dim = min(curves, key=lambda d: min(curves[d]))
    return {
        "holdout_year": config.TUNE_HOLDOUT_YEAR,
        "fit_rows": len(fit_part),
        "holdout_rows": len(holdout),
        "holdout_log_loss_by_dim": {str(d): min(c) for d, c in curves.items()},
        "epochs_by_dim": {str(d): int(np.argmin(c)) + 1 for d, c in curves.items()},
        "dim": best_dim,
        "dim_at_grid_edge": best_dim == max(DIMS),  # D-041 revisit trigger: extend to 64, holdout only
        "epochs": int(np.argmin(curves[best_dim])) + 1,
    }


def save_artifacts(net: Net, train: pd.DataFrame, dim: int, epochs: int, out_dir: Path) -> None:
    """Seed-42 model and the batter/bowler embedding tables for Phase 4 (architecture.md contract)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": net.model.state_dict(), "dim": dim, "epochs": epochs}, out_dir / "cric2vec_seed42.pt"
    )
    tables = net.model.embeddings()
    rows = []
    for role, col, vocab in (
        ("batter", "batter_id", net.encoder.bat_vocab),
        ("bowler", "bowler_id", net.encoder.bowl_vocab),
    ):
        counts = train[col].value_counts()
        ids = ["__UNK__", *sorted(vocab, key=vocab.get)]  # row i of the table is index i
        n_unk = int(counts[~counts.index.isin(vocab)].sum())
        rows.append(
            pd.DataFrame(
                {"player_id": ids, "role": role, "n_train_balls": [n_unk, *(counts[i] for i in ids[1:])]}
            )
        )
    np.save(out_dir / "embeddings.npy", np.vstack([tables["batter"], tables["bowler"]]))
    pd.concat(rows, ignore_index=True).to_parquet(out_dir / "embedding_index.parquet", index=False)


def run(processed_dir: Path | None = None) -> dict:
    train, val, competitions = load_frames(processed_dir)
    tuning = tune(*holdout_split(train), competitions)
    use = tuning["use_ablation"]
    epochs = tuning["ablation_epochs"] if use else tuning["epochs"]

    train_s, val_s = with_stats(train, val)  # train rows out-of-fold, val rows from all of train
    y = val["y"].to_numpy()
    losses = []
    for seed in SEEDS:
        net, _ = fit_net(train_s, competitions, tuning["dim"], seed, extra=_extra(use), epochs=epochs)
        losses.append(row_log_loss(net.predict_proba(val_s), y))
        if seed == SEEDS[0]:
            save_artifacts(net, train, tuning["dim"], epochs, artifact_dir())
    losses = np.array(losses)
    report = {
        "seeds": list(SEEDS),
        "device": str(DEVICE),
        "val_key": val_key(val),
        "tuning": tuning,
        "final": {"dim": tuning["dim"], "epochs": epochs, "ablation": use},
        "per_seed_val_log_loss": [float(x) for x in losses.mean(axis=1)],
    }
    report_dir().mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir() / "val_loss_cric2vec.npy", losses)  # (seeds, val rows)
    (report_dir() / "cric2vec.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def run_lite(processed_dir: Path | None = None) -> dict:
    """D-041: lite-context cric2vec, seed 42 only (artifacts for the Phase 4 embedding choice, not a gate
    model). Val log-loss is reported, never used to choose; the choice is the embedding probe's."""
    train, val, competitions = load_frames(processed_dir)
    tuning = tune_lite(*holdout_split(train), competitions)
    net, _ = fit_net(train, competitions, tuning["dim"], SEEDS[0], epochs=tuning["epochs"], lite=True)
    save_artifacts(net, train, tuning["dim"], tuning["epochs"], lite_artifact_dir())
    report = {
        "seed": SEEDS[0],
        "device": str(DEVICE),
        "inputs": list(LITE_NUMERIC),
        "val_key": val_key(val),
        "tuning": tuning,
        "final": {"dim": tuning["dim"], "epochs": tuning["epochs"]},
        "val_log_loss": log_loss(net.predict_proba(val), val["y"].to_numpy()),
    }
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "cric2vec_lite.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    if "--lite" in sys.argv[1:]:
        r = run_lite()
        print(f"cric2vec lite val log-loss {r['val_log_loss']:.5f}  final {r['final']}")
    else:
        r = run()
        print(f"cric2vec per-seed val log-loss {r['per_seed_val_log_loss']}  final {r['final']}")
