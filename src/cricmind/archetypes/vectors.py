"""Phase 4 inputs: train-only loading, eligibility, the shared match bootstrap, the two player representations
(D-044). Imports no torch, so the sklearn clustering and the torch worker stay in separate processes.

from cricmind.archetypes.vectors import load_train, eligibility, resample_matches, stats_vectors
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.features.context import PHASES, build_context, competition_vocab
from cricmind.features.splits import load_deliveries
from cricmind.models.embedding_probe import RUNS

K = len(config.OUTCOME_CLASSES)
ROLES = {"batter": "batter_id", "bowler": "bowler_id"}
B = 20  # D-044(6): match-bootstrap resamples, shared by both representations
SENSITIVITY_FLOORS = (150, config.MIN_TRAIN_BALLS_ARCHETYPE, 500)  # D-044(2); the smallest is the superset
STAT_NAMES = ("strike_rate", "dot_pct", "boundary_pct", "dismissal_rate")
STAT_COLUMNS = [f"{s}_{p}" for s in STAT_NAMES for p in PHASES]  # 12 dims, stat-major


def artifact_dir() -> Path:
    return config.ARTIFACTS / "p4"


def report_dir() -> Path:
    return config.REPORTS / "p4"


def boot_dir(b: int) -> Path:
    return artifact_dir() / "emb_boot" / f"b{b:02d}"


def write_json(path: Path, obj: dict) -> None:
    """Reports are written by code only; numpy scalars become plain numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item()) + "\n")


def load_train(processed_dir: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Train context frame and the train-only competition vocab. Val and test are never loaded (D-044(1))."""
    deliveries = load_deliveries(("train",), processed_dir)
    competitions = competition_vocab(deliveries)
    train = build_context(deliveries, competitions)
    assert set(train["split"]) == {"train"}
    return train, competitions


def eligibility(train: pd.DataFrame, role: str, floor: int) -> pd.Series:
    """Legal train balls per player with at least `floor` of them, sorted by registry id."""
    counts = train[ROLES[role]].value_counts()
    return counts[counts >= floor].sort_index()


def resample_matches(frame: pd.DataFrame, b: int) -> pd.DataFrame:
    """Whole matches drawn with replacement, seed SEED + b. One function for both arms, so the stats and the
    retrained embeddings see the same resamples (D-044(6)). A match drawn twice appears twice."""
    codes, uniques = pd.factorize(frame["match_id"])
    draws = np.random.default_rng(config.SEED + b).integers(0, len(uniques), len(uniques))
    times = np.bincount(draws, minlength=len(uniques))
    return frame.iloc[np.repeat(np.arange(len(frame)), times[codes])].reset_index(drop=True)


def stats_vectors(
    frame: pd.DataFrame, role: str, ids: pd.Index, m: float = config.STATS_SHRINK_M
) -> pd.DataFrame:
    """Raw 12-dim style vector per player in `ids` (D-044(3)). The 7 class counts of each (player, phase) are
    shrunk toward that phase's class shares in `frame`, so a player with a thin phase keeps a finite, honest
    value instead of NaN. Reduced-match rows have no phase (D-030) and drop out."""
    d = frame[frame["phase"].notna()]
    onehot = pd.DataFrame(np.eye(K)[d["y"].to_numpy()], index=d.index)
    counts = onehot.groupby([d[ROLES[role]], d["phase"]]).sum()
    cells = pd.MultiIndex.from_product([ids, PHASES], names=[ROLES[role], "phase"])
    counts = counts.reindex(cells, fill_value=0.0)
    share = onehot.groupby(d["phase"]).sum()
    share = share.div(share.sum(axis=1), axis=0)
    prior = share.reindex(counts.index.get_level_values("phase")).to_numpy()
    n = counts.sum(axis=1).to_numpy()[:, None]
    p = (counts.to_numpy() + m * prior) / (n + m)
    stats = [p @ RUNS * 100, p[:, 0] * 100, (p[:, 4] + p[:, 5]) * 100, p[:, 6] * 100]
    wide = np.hstack([s.reshape(len(ids), len(PHASES)) for s in stats])  # player x (stat, phase)
    return pd.DataFrame(wide, index=ids, columns=STAT_COLUMNS)


def stats_bootstrap(
    train: pd.DataFrame, ids_by_role: dict[str, pd.Index], n_boot: int = B
) -> dict[str, list]:
    """Per role, `n_boot` stats matrices (players x 12) recomputed on the shared resamples."""
    out: dict[str, list] = {role: [] for role in ids_by_role}
    for b in range(n_boot):
        res = resample_matches(train, b)
        for role, ids in ids_by_role.items():
            out[role].append(stats_vectors(res, role, ids).to_numpy())
    return out


def _positions(emb_dir: Path, role: str) -> pd.Series:
    index = pd.read_parquet(emb_dir / "embedding_index.parquet")
    chosen = index[index["role"] == role]
    return pd.Series(chosen.index.to_numpy(), index=chosen["player_id"])


def has_own_embedding(emb_dir: Path, role: str, ids: pd.Index) -> np.ndarray:
    """False for players the model gave no embedding of their own (UNK)."""
    return ids.isin(_positions(emb_dir, role).index)


def embedding_rows(emb_dir: Path, role: str, ids: pd.Index, allow_unk: bool = False) -> np.ndarray:
    """Rows of `embeddings.npy` for `ids` (architecture.md contract). A player below the embedding vocab floor
    in a resample is UNK there, exactly as the retrained model sees them; with `allow_unk` that player gets
    the shared UNK row (D-045), otherwise it is an error (the real train set must cover every id)."""
    position = _positions(emb_dir, role)
    own = ids.isin(position.index)
    if not own.all() and not allow_unk:
        raise ValueError(f"{(~own).sum()} eligible {role}s have no embedding in {emb_dir}")
    rows = position.reindex(ids, fill_value=position["__UNK__"]).to_numpy()
    return np.load(emb_dir / "embeddings.npy")[rows]
