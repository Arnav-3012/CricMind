"""Phase 3 inputs shared by the cric2vec worker, the LightGBM worker and the comparison (D-038).

Imports neither torch nor lightgbm on purpose: the two libraries bring separate OpenMP runtimes and crash when
they share a process on macOS (mistakes.md 2026-10-29), so each model trains in its own process and the
comparison reads their per-row val losses from disk.

from cricmind.models.p3_data import load_frames, with_stats
"""

from pathlib import Path

import pandas as pd

from cricmind import config
from cricmind.features.context import build_context, competition_vocab
from cricmind.features.splits import load_deliveries
from cricmind.features.stats import ABLATION_SPECS, SPECS, apply_all, fit_all, oof_all, stat_columns

SEEDS = (42, 43, 44)  # D-038(6); artifacts come from the first
ABLATION_MIN_GAIN = 0.002  # D-038(4), nats on the holdout
BASE_COLS = stat_columns(SPECS)
ABLATION_COLS = stat_columns(ABLATION_SPECS)
ALL_SPECS = {**SPECS, **ABLATION_SPECS}


def artifact_dir() -> Path:
    return config.ARTIFACTS / "p3"


def report_dir() -> Path:
    return config.REPORTS / "p3"


def lite_artifact_dir() -> Path:
    return artifact_dir() / "lite"  # D-041(2): same contract as artifact_dir()


def check_split_integrity(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Whole matches on one side and every train match before every val match (D-005). Checks the loaded
    rows, not the config, so a bad parquet or a changed bound fails here instead of leaking silently."""
    if set(train["match_id"]) & set(val["match_id"]):
        raise ValueError("a match is in both train and val")
    if train["match_date"].max() >= val["match_date"].min():
        raise ValueError("train and val overlap in time")


def load_frames(
    processed_dir: Path | None = None, sequence: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train and val context frames and the train-only competition vocab. The test seasons stay unread.
    `sequence=True` adds the recent-ball columns; rows and their order are the same either way."""
    train_d = load_deliveries(("train",), processed_dir)
    competitions = competition_vocab(train_d)
    train = build_context(train_d, competitions, sequence)
    val = build_context(load_deliveries(("val",), processed_dir), competitions, sequence)
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}
    check_split_integrity(train, val)
    return train, val, competitions


def holdout_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Years before the tuning year, and the tuning year (D-032)."""
    return train[train["year"] < config.TUNE_HOLDOUT_YEAR], train[train["year"] == config.TUNE_HOLDOUT_YEAR]


def with_stats(fit: pd.DataFrame, apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`fit` gets out-of-fold rates, `apply` (holdout or val) gets rates from all of `fit` (D-038(3))."""
    tables = fit_all(fit, ALL_SPECS)
    fit_s = pd.concat([fit, oof_all(fit, ALL_SPECS)], axis=1)
    apply_s = pd.concat([apply, apply_all(apply, tables, ALL_SPECS)], axis=1)
    return fit_s, apply_s


def val_key(val: pd.DataFrame) -> int:
    """Fingerprint of the val rows and their order; the comparison refuses losses from different rows."""
    return int(pd.util.hash_pandas_object(val[["match_id", "ball_seq"]], index=False).sum() % 2**62)
