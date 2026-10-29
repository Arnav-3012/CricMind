"""D-040 diagnostic: what do the batter, bowler and venue embeddings add to the MLP? (2023 holdout only)

uv run python -m cricmind.models.no_identity   # reports/p3/no_identity.json (needs reports/p3/cric2vec.json)

Same MLP, seed and embedding dim, trained on years before the holdout and scored on the holdout year, with and
without player identity. Without: every batter, bowler and venue id is one constant, so the net sees only the
match situation. E = loss without - loss with: what identity adds. Val is never loaded or scored.
Torch only (see p3_data).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.eval.metrics import paired_bootstrap_diff, row_log_loss
from cricmind.models.p3_data import ABLATION_MIN_GAIN, SEEDS, holdout_split, load_frames, report_dir
from cricmind.models.train import fit_net

IDS = ["batter_id", "bowler_id", "venue_key"]


def _holdout_row_loss(
    fit: pd.DataFrame, holdout: pd.DataFrame, competitions: list[str], dim: int, seed: int
) -> tuple[np.ndarray, int]:
    """Early-stop on the holdout to pick the epoch, refit to exactly that epoch, return its per-row loss."""
    _, curve = fit_net(fit, competitions, dim, seed, holdout=holdout)
    epochs = int(np.argmin(curve)) + 1
    net, _ = fit_net(fit, competitions, dim, seed, epochs=epochs)
    return row_log_loss(net.predict_proba(holdout), holdout["y"].to_numpy()), epochs


def run(processed_dir: Path | None = None) -> dict:
    train, _, competitions = load_frames(processed_dir)  # val is built by the shared loader, never scored
    fit, holdout = holdout_split(train)
    dim = json.loads((report_dir() / "cric2vec.json").read_text())["tuning"]["dim"]
    blank = {c: "__none__" for c in IDS}
    with_ids, ep_with = _holdout_row_loss(fit, holdout, competitions, dim, SEEDS[0])
    without, ep_without = _holdout_row_loss(
        fit.assign(**blank), holdout.assign(**blank), competitions, dim, SEEDS[0]
    )
    e = paired_bootstrap_diff(without, with_ids, holdout["match_id"].to_numpy())  # positive: ids help
    report = {
        "holdout_year": config.TUNE_HOLDOUT_YEAR,
        "seed": SEEDS[0],
        "dim": dim,
        "holdout_log_loss": {"with_ids": float(with_ids.mean()), "without_ids": float(without.mean())},
        "epochs": {"with_ids": ep_with, "without_ids": ep_without},
        "E_without_minus_with": e,
        "threshold": ABLATION_MIN_GAIN,
        # D-040(3), pre-registered: signal only if E clears the threshold and its CI excludes 0
        "embeddings_carry_signal": bool(e["value"] >= ABLATION_MIN_GAIN and e["lo"] > 0),
    }
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "no_identity.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    print(f"holdout log-loss {r['holdout_log_loss']}  E {r['E_without_minus_with']}")
    print(f"embeddings_carry_signal: {r['embeddings_carry_signal']}")
