"""Target-encoded class rates per batter, bowler, venue (D-038(3)): the strong non-embedding baseline.

A key's rates are (counts + m * prior) / (n + m): few balls pull toward the class shares, many balls trust the
key. Inside train a row's rates come from the other folds only (folds are whole matches), so a tree can never
read the row's own label; val gets rates from all of train. An unseen key gets the prior and n = 0.

from cricmind.features.stats import SPECS, fit_all, oof_all, apply_all
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cricmind import config

K = len(config.OUTCOME_CLASSES)
SPECS = {  # prefix -> key columns of the context frame (IDS)
    "bat": ["batter_id"],
    "bowl": ["bowler_id"],
    "venue": ["venue_key"],
}
ABLATION_SPECS = {  # D-038(4): player-at-venue form, kept only if the 2023-holdout gain is >= 0.002
    "bat_venue": ["batter_id", "venue_key"],
    "bowl_venue": ["bowler_id", "venue_key"],
}


def stat_columns(specs: dict[str, list[str]]) -> list[str]:
    return [c for prefix in specs for c in (f"{prefix}_n", *(f"{prefix}_p{i}" for i in range(K)))]


def _onehot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), K))
    out[np.arange(len(y)), y] = 1.0
    return out


def _index(frame: pd.DataFrame, keys: list[str]) -> pd.Index:
    """Row keys as an Index that matches what groupby(keys) produces (flat for one key, multi for several)."""
    return pd.Index(frame[keys[0]]) if len(keys) == 1 else pd.MultiIndex.from_frame(frame[keys])


def _columns(prefix: str, counts: np.ndarray, prior: np.ndarray, m: float) -> pd.DataFrame:
    """`counts` (rows, K) and `prior` (K,) or (rows, K) -> n and shrunk rates."""
    n = counts.sum(axis=1, keepdims=True)
    rates = (counts + m * prior) / (n + m)
    return pd.DataFrame(np.hstack([n, rates]), columns=stat_columns({prefix: []}))


def fold_of(match_ids: pd.Series, folds: int = config.STATS_FOLDS, seed: int = config.SEED) -> np.ndarray:
    """Fold id per row; a whole match is in one fold and the folds are balanced in number of matches."""
    codes, uniques = pd.factorize(np.asarray(match_ids))
    per_match = np.random.default_rng(seed).permutation(len(uniques)) % folds
    return per_match[codes]


@dataclass
class RateTable:
    counts: pd.DataFrame  # one row per key, K class counts
    prior: np.ndarray  # class shares of the frame it was fitted on
    m: float


def fit_rates(frame: pd.DataFrame, keys: list[str], m: float = config.STATS_SHRINK_M) -> RateTable:
    """Counts per key over all of `frame` (train). Used for val/test rows, never for train rows themselves."""
    onehot = pd.DataFrame(_onehot(frame["y"].to_numpy()), index=frame.index)
    counts = onehot.groupby([frame[k] for k in keys]).sum()
    return RateTable(counts, counts.sum().to_numpy() / len(frame), m)


def apply_rates(frame: pd.DataFrame, table: RateTable, keys: list[str], prefix: str) -> pd.DataFrame:
    counts = table.counts.reindex(_index(frame, keys)).fillna(0.0).to_numpy()  # unseen key: n = 0
    out = _columns(prefix, counts, table.prior, table.m)
    out.index = frame.index
    return out


def oof_rates(
    frame: pd.DataFrame,
    keys: list[str],
    prefix: str,
    m: float = config.STATS_SHRINK_M,
    folds: int = config.STATS_FOLDS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Rates for the rows of `frame` computed without their own fold: total counts minus the fold's counts,
    for the key and for the prior alike (the prior would otherwise carry a sliver of the row's own label)."""
    fold = fold_of(frame["match_id"], folds, seed)
    y = frame["y"].to_numpy()
    cells = frame[keys].copy()
    cells["fold"] = fold
    classes = [f"c{i}" for i in range(K)]
    cells[classes] = _onehot(y)
    per_fold = cells.groupby([*keys, "fold"])[classes].sum()
    per_key = per_fold.groupby(level=list(range(len(keys)))).sum()
    total = per_key.reindex(_index(frame, keys)).to_numpy()
    own_fold = per_fold.reindex(pd.MultiIndex.from_frame(cells[[*keys, "fold"]])).to_numpy()
    counts = total - own_fold
    class_in_fold = np.zeros((folds, K))
    np.add.at(class_in_fold, (fold, y), 1.0)
    prior = class_in_fold.sum(axis=0) - class_in_fold[fold]
    prior /= prior.sum(axis=1, keepdims=True)
    out = _columns(prefix, counts, prior, m)
    out.index = frame.index
    return out


def fit_all(train: pd.DataFrame, specs: dict[str, list[str]] = SPECS) -> dict[str, RateTable]:
    return {prefix: fit_rates(train, keys) for prefix, keys in specs.items()}


def apply_all(
    frame: pd.DataFrame, tables: dict[str, RateTable], specs: dict[str, list[str]] = SPECS
) -> pd.DataFrame:
    return pd.concat([apply_rates(frame, tables[p], keys, p) for p, keys in specs.items()], axis=1)


def oof_all(train: pd.DataFrame, specs: dict[str, list[str]] = SPECS) -> pd.DataFrame:
    return pd.concat([oof_rates(train, keys, p) for p, keys in specs.items()], axis=1)
