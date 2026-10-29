"""D-005 temporal split by match date. Years come from config.py; never re-tune after seeing results.

from cricmind.features.splits import load_deliveries
train = load_deliveries(("train",))
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config

# Inclusive date bounds per split. The loader filters on these, so rows of a split that was not asked
# for (the test seasons, in Phase 2) never enter memory.
BOUNDS = {
    "train": (dt.date.min, dt.date(config.TRAIN_END_YEAR, 12, 31)),
    "val": (dt.date(config.VAL_YEAR, 1, 1), dt.date(config.VAL_YEAR, 12, 31)),
    "test": (dt.date(config.TEST_START_YEAR, 1, 1), dt.date(config.TEST_END_YEAR, 12, 31)),
}
# Everything the context features need, nothing else (the parquet has 34 columns).
COLUMNS = [
    "match_id", "match_date", "competition", "venue_key", "innings", "is_super_over", "over", "ball_seq",
    "legal_ball_in_over", "batter_id", "bowler_id", "runs_batter", "wides", "is_legal", "outcome",
    "score_before", "wickets_before", "target", "legal_balls_remaining", "is_reduced",
]  # fmt: skip


def assign_split(match_date: pd.Series) -> pd.Series:
    """'train' / 'val' / 'test' / 'other' from the match year; vectorised."""
    year = pd.to_datetime(match_date).dt.year
    labels = np.select(
        [
            year <= config.TRAIN_END_YEAR,
            year == config.VAL_YEAR,
            year.between(config.TEST_START_YEAR, config.TEST_END_YEAR),
        ],
        ["train", "val", "test"],
        default="other",
    )
    return pd.Series(labels, index=match_date.index, name="split")


def load_deliveries(splits: tuple[str, ...], processed_dir: Path | None = None) -> pd.DataFrame:
    """All deliveries (wides and no-balls too: the running counts need them) of matches in `splits`.

    A match sits wholly in one split, so per-match running counts computed on this frame are the same as
    on the full file.
    """
    unknown = set(splits) - set(BOUNDS)
    if unknown:
        raise ValueError(f"unknown split(s) {sorted(unknown)}; choose from {sorted(BOUNDS)}")
    path = (processed_dir or config.DATA_PROCESSED) / "deliveries.parquet"
    filters = [[("match_date", ">=", BOUNDS[s][0]), ("match_date", "<=", BOUNDS[s][1])] for s in splits]
    return pd.read_parquet(path, columns=COLUMNS, filters=filters)
