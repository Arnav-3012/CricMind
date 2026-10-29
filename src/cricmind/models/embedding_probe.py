"""Probe: do the cric2vec embeddings encode player style? No training, train rows only (val is never used).

uv run python -m cricmind.models.embedding_probe   # reports/p3/embedding_probe.json (needs train.py outputs)
uv run python -m cricmind.models.embedding_probe --lite   # D-041: reads artifacts/p3/lite/

For eligible players (>= MIN_TRAIN_BALLS_ARCHETYPE train balls in the role), regress each per-phase style stat
(strike rate, dot %, boundary %, wicket rate) on the seed-42 embedding with a ridge model and report the
cross-validated R^2 over players. Two references keep the number honest:
  * a shuffled-target R^2 (should sit at or below 0), and
  * the stat's own reliability: split the player's balls by match parity, correlate the two halves, then
    Spearman-Brown to full length. R^2 cannot beat it, because the stat itself is noisy.
Pre-registered reading (logs.md 2026-10-29, written before the run), on the mean R^2 / reliability ratio per
role: >= 0.5 the embeddings encode style; < 0.2 they do not; in between Phase 4 stability decides.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cricmind import config
from cricmind.features.context import PHASES
from cricmind.models.p3_data import artifact_dir, lite_artifact_dir, load_frames, report_dir

K = len(config.OUTCOME_CLASSES)
RUNS = np.array([0, 1, 2, 3, 4, 6, 0], dtype="float64")  # runs off the bat per class; "3+" counted as 3
MIN_PHASE_BALLS = 50  # a player's phase stat needs at least this many balls in that phase
FOLDS, RIDGE_ALPHA = 5, 1.0
STRONG, WEAK = 0.5, 0.2  # ratio cut-offs of the pre-registered reading
ROLES = {"batter": "batter_id", "bowler": "bowler_id"}


def player_stats(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """One row per player, columns `<stat>_<phase>`; NaN where the player has < MIN_PHASE_BALLS in that phase.
    Reduced-match rows have no phase and drop out."""
    onehot = pd.DataFrame(np.eye(K)[frame["y"].to_numpy()], index=frame.index)
    counts = onehot.groupby([frame[id_col], frame["phase"]]).sum()
    n = counts.sum(axis=1).to_numpy()
    c = counts.to_numpy()
    stats = pd.DataFrame(
        {
            "strike_rate": c @ RUNS / n * 100,
            "dot_pct": c[:, 0] / n * 100,
            "boundary_pct": (c[:, 4] + c[:, 5]) / n * 100,
            "wicket_rate": c[:, 6] / n * 100,
        },
        index=counts.index,
    )
    stats[n < MIN_PHASE_BALLS] = np.nan
    wide = stats.unstack("phase")
    wide.columns = [f"{stat}_{phase}" for stat, phase in wide.columns]
    return wide


def probe_r2(X: np.ndarray, y: np.ndarray, seed: int = config.SEED) -> float:
    """Cross-validated R^2 of a ridge regression of `y` on `X` (standardised inside each fold)."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    pred = cross_val_predict(model, X, y, cv=KFold(FOLDS, shuffle=True, random_state=seed))
    return float(r2_score(y, pred))


def reliability(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman-Brown full-length reliability from two half-sample measurements of the same stat."""
    r = np.corrcoef(a, b)[0, 1]
    return float(2 * r / (1 + r))


def verdict(ratio: float) -> str:
    return "encodes_style" if ratio >= STRONG else "does_not" if ratio < WEAK else "partial_phase4_decides"


def _role(train: pd.DataFrame, id_col: str, X_all: np.ndarray, ids: pd.Series, half: np.ndarray) -> dict:
    total = train[id_col].value_counts()
    keep = ids.isin(total[total >= config.MIN_TRAIN_BALLS_ARCHETYPE].index).to_numpy()
    X, players = X_all[keep], pd.Index(ids[keep])
    full = player_stats(train, id_col).reindex(players)
    a = player_stats(train[half], id_col).reindex(players)
    b = player_stats(train[~half], id_col).reindex(players)
    rng = np.random.default_rng(config.SEED)
    targets = {}
    for t in full.columns:
        ok = (full[t].notna() & a[t].notna() & b[t].notna()).to_numpy()
        y = full[t].to_numpy()[ok]
        rel = reliability(a[t].to_numpy()[ok], b[t].to_numpy()[ok])
        r2 = probe_r2(X[ok], y)
        corr = [abs(np.corrcoef(X[ok][:, d], y)[0, 1]) for d in range(X.shape[1])]
        targets[t] = {
            "n_players": int(ok.sum()),
            "r2": r2,
            "r2_shuffled": probe_r2(X[ok], rng.permutation(y)),
            "reliability": rel,
            "r2_over_reliability": r2 / rel if rel > 0 else None,
            "best_dim": int(np.argmax(corr)),
            "best_dim_abs_corr": float(max(corr)),
        }
    ratios = [v["r2_over_reliability"] for v in targets.values() if v["r2_over_reliability"] is not None]
    mean_ratio = float(np.mean(ratios))
    return {
        "eligible_players": int(keep.sum()),
        "mean_r2": float(np.mean([v["r2"] for v in targets.values()])),
        "mean_r2_over_reliability": mean_ratio,
        "verdict": verdict(mean_ratio),
        "targets": targets,
    }


def run(
    processed_dir: Path | None = None, emb_dir: Path | None = None, report_path: Path | None = None
) -> dict:
    """`emb_dir` holds embeddings.npy and embedding_index.parquet (default: the current model's);
    `report_path` is where the JSON goes. The lite variant (D-041) passes its own, so both reports coexist."""
    emb_dir = emb_dir or artifact_dir()
    train, _, _ = load_frames(processed_dir)  # only train rows are used below
    emb = np.load(emb_dir / "embeddings.npy")
    index = pd.read_parquet(emb_dir / "embedding_index.parquet")  # row i of emb = row i of index
    half = pd.factorize(train["match_id"])[0] % 2 == 0  # split by match, not by ball
    out = {}
    for role, id_col in ROLES.items():
        rows = index[index["role"] == role]
        out[role] = _role(train, id_col, emb[rows.index.to_numpy()], rows["player_id"], half)
    report = {
        "dim": int(emb.shape[1]),
        "phases": PHASES,
        "min_phase_balls": MIN_PHASE_BALLS,
        "min_train_balls": config.MIN_TRAIN_BALLS_ARCHETYPE,
        "cut_offs": {"strong": STRONG, "weak": WEAK},
        # D-041(3): the number Phase 4 compares between the current and the lite embeddings
        "mean_ratio_batter_bowler": float(np.mean([out[r]["mean_r2_over_reliability"] for r in ROLES])),
        **out,
    }
    report_path = report_path or report_dir() / "embedding_probe.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    if "--lite" in sys.argv[1:]:
        r = run(emb_dir=lite_artifact_dir(), report_path=report_dir() / "embedding_probe_lite.json")
    else:
        r = run()
    for role in ROLES:
        d = r[role]
        print(
            f"{role}: players {d['eligible_players']}  mean R2 {d['mean_r2']:.3f}  "
            f"R2/reliability {d['mean_r2_over_reliability']:.2f}  -> {d['verdict']}"
        )
    print(f"mean ratio (batter, bowler) {r['mean_ratio_batter_bowler']:.3f}")
