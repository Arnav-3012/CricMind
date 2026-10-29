"""Phase 3 gate: cric2vec vs LightGBM + stats on val, from the two workers' per-row losses (D-038(7)).

uv run python -m cricmind.models.train          # cric2vec worker (torch)
uv run python -m cricmind.models.lgbm_stats     # LightGBM worker (no torch)
uv run python -m cricmind.models.outcome_compare   # reads both, writes reports/p3/outcome_models.json

Never imports torch or lightgbm (see p3_data).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.eval.metrics import bootstrap_ci, paired_bootstrap_diff
from cricmind.features.players import seen_pairs, unseen_pair_mask
from cricmind.models.p3_data import SEEDS, artifact_dir, load_frames, report_dir, val_key


def _compare(mask: np.ndarray, cric: np.ndarray, lgbm: np.ndarray, ids: np.ndarray) -> dict:
    """Seed-mean per-row losses of both models on one val slice, plus the paired difference."""
    if not mask.any():
        return {"n": 0}
    c, g, m = cric[mask], lgbm[mask], ids[mask]
    return {
        "n": int(mask.sum()),
        "cric2vec": bootstrap_ci(c, m),
        "lightgbm_stats": bootstrap_ci(g, m),
        "cric2vec_minus_lightgbm_stats": paired_bootstrap_diff(c, g, m),
    }


def slices(val: pd.DataFrame, train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Val-row masks. 'Unseen' means never in train by raw id (not the UNK floor), as in D-038(7)."""
    reduced = val["is_reduced"].to_numpy(dtype=bool)
    out = {"all": np.ones(len(val), dtype=bool), "unreduced": ~reduced, "reduced": reduced}
    for phase in ("powerplay", "middle", "death"):
        out[f"phase_{phase}"] = (val["phase"] == phase).to_numpy()  # reduced rows have no phase
    out["unseen_pair"] = unseen_pair_mask(val, seen_pairs(train))
    out["unseen_batter"] = ~val["batter_id"].isin(train["batter_id"]).to_numpy()
    out["unseen_bowler"] = ~val["bowler_id"].isin(train["bowler_id"]).to_numpy()
    return out


def run(processed_dir: Path | None = None) -> dict:
    train, val, _ = load_frames(processed_dir)
    cric = json.loads((report_dir() / "cric2vec.json").read_text())
    lgbm = json.loads((report_dir() / "lgbm_stats.json").read_text())
    if not (cric["val_key"] == lgbm["val_key"] == val_key(val)):
        raise ValueError("the workers scored different val rows; re-run both")
    loss_c = np.load(artifact_dir() / "val_loss_cric2vec.npy")
    loss_g = np.load(artifact_dir() / "val_loss_lgbm_stats.npy")
    assert loss_c.shape == loss_g.shape == (len(SEEDS), len(val))
    mean_c, mean_g = loss_c.mean(axis=0), loss_g.mean(axis=0)  # D-038(6): 3-seed mean of row losses

    ids = val["match_id"].to_numpy()
    results = {name: _compare(m, mean_c, mean_g, ids) for name, m in slices(val, train).items()}
    per_seed = {"cric2vec": loss_c.mean(axis=1), "lightgbm_stats": loss_g.mean(axis=1)}
    spread = {k: float(v.max() - v.min()) for k, v in per_seed.items()}
    diff = results["all"]["cric2vec_minus_lightgbm_stats"]
    unreduced = results["unreduced"]["cric2vec_minus_lightgbm_stats"]
    report = {
        "snapshot": config.DATA_SNAPSHOT,
        "seeds": list(SEEDS),
        "classes": config.OUTCOME_CLASSES,
        "rows": {"train": len(train), "val": len(val), "val_matches": int(val["match_id"].nunique())},
        "cric2vec": {"tuning": cric["tuning"], "final": cric["final"], "device": cric["device"]},
        "lightgbm_stats": {"tuning": lgbm["tuning"], "final": lgbm["final"]},
        "per_seed_val_log_loss": {k: [float(x) for x in v] for k, v in per_seed.items()},
        "seed_spread": spread,  # > 0.001 in either arm triggers the D-035 revisit
        "seed_spread_exceeds_0.001": bool(max(spread.values()) > 0.001),
        "val": results,
        "gate": {  # D-038(7): all rows, and the unreduced slice (no reduced-match tell) must agree
            "all_point_negative": bool(diff["value"] < 0),
            "all_ci_hi_negative": bool(diff["hi"] < 0),
            "unreduced_sign_agrees": bool(unreduced["value"] < 0),
        },
    }
    report["gate"]["passed"] = all(report["gate"].values())
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "outcome_models.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    d = r["val"]["all"]
    print(f"cric2vec {d['cric2vec']['value']:.4f}  lgbm+stats {d['lightgbm_stats']['value']:.4f}")
    print(f"gate {r['gate']}")
