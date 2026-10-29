"""Id vocabularies for the embedding models and the unseen-matchup subset (D-008, D-038).

Index 0 is UNK: an id with fewer than `floor` modelled balls in the frame it is built from, or one never
seen there. Vocabs are built from the fit set of each run (train, or train before the holdout year), so a
debutant is UNK in the holdout exactly as a val debutant is.

from cricmind.features.players import build_vocab, encode
"""

import numpy as np
import pandas as pd

from cricmind import config

UNK = 0


def _check_train_rows(frame: pd.DataFrame, what: str) -> None:
    if frame["year"].max() > config.TRAIN_END_YEAR:
        raise ValueError(f"{what} is fitted on train rows only (D-005)")


def build_vocab(frame: pd.DataFrame, col: str, floor: int) -> dict[str, int]:
    """id -> index (1..n, sorted) for ids with at least `floor` rows of `col`. `frame` is a context frame
    (one row per modelled ball), so a count of rows is a count of legal balls."""
    _check_train_rows(frame, "build_vocab")
    counts = frame[col].value_counts()
    return {i: n + 1 for n, i in enumerate(sorted(counts[counts >= floor].index))}


def encode(ids: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    """Index per row; ids outside the vocab (rare or unseen) become UNK."""
    return ids.map(vocab).fillna(UNK).to_numpy(dtype="int64")


def seen_pairs(frame: pd.DataFrame) -> pd.MultiIndex:
    """Distinct (batter, bowler) pairs that met in `frame`. Raw ids, not UNK-mapped: 'never in train' means
    the two players never faced each other, even if both are known individually."""
    _check_train_rows(frame, "seen_pairs")
    return pd.MultiIndex.from_frame(frame[["batter_id", "bowler_id"]].drop_duplicates())


def unseen_pair_mask(frame: pd.DataFrame, pairs: pd.MultiIndex) -> np.ndarray:
    """True for rows whose (batter, bowler) pair is not in `pairs` (the unseen-matchup subset)."""
    return ~pd.MultiIndex.from_frame(frame[["batter_id", "bowler_id"]]).isin(pairs)
