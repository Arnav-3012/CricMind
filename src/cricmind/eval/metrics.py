"""Log-loss, Brier, ECE and match-level bootstrap CIs (D-032). `P` is (n, 7) probabilities, `y` class ids.

from cricmind.eval.metrics import log_loss, bootstrap_ci
"""

import numpy as np
import pandas as pd

from cricmind import config

EPS = 1e-15  # keeps ln(0) finite; a model that needs this clip is already wrong


def row_log_loss(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """-ln p(true class) per row."""
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))


def log_loss(P: np.ndarray, y: np.ndarray) -> float:
    return float(row_log_loss(P, y).mean())


def brier(P: np.ndarray, y: np.ndarray) -> float:
    """Mean over rows of the squared error summed over the classes (0 perfect, 2 worst)."""
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def ece(P: np.ndarray, y: np.ndarray, bins: int = config.ECE_BINS) -> float:
    """Top-label expected calibration error: bin by the model's top probability, compare it with how often
    that top class was right. sum_b (n_b / N) * |accuracy_b - confidence_b|."""
    conf = P.max(axis=1)
    hit = (P.argmax(axis=1) == y).astype("float64")
    idx = np.minimum((conf * bins).astype(int), bins - 1)  # conf == 1.0 goes in the last bin
    gap = np.bincount(idx, weights=hit, minlength=bins) - np.bincount(idx, weights=conf, minlength=bins)
    return float(np.abs(gap).sum() / len(y))


def _match_totals(match_ids: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(np.asarray(match_ids))
    return codes, np.bincount(codes, minlength=len(uniques))


def _resampled_means(row_values: np.ndarray, match_ids, n_boot: int, seed: int) -> tuple[float, np.ndarray]:
    """Mean of `row_values` and its bootstrap distribution when whole matches are resampled with
    replacement (balls in one match are not independent, so rows are never resampled)."""
    codes, counts = _match_totals(match_ids)
    sums = np.bincount(codes, weights=row_values, minlength=len(counts))
    m = len(counts)
    draws = np.random.default_rng(seed).multinomial(m, np.full(m, 1.0 / m), size=n_boot)
    return float(sums.sum() / counts.sum()), (draws @ sums) / (draws @ counts)


def _summary(point: float, boot: np.ndarray, level: float) -> dict[str, float]:
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"value": point, "lo": float(lo), "hi": float(hi), "width": float(hi - lo)}


def bootstrap_ci(
    row_loss: np.ndarray,
    match_ids,
    n_boot: int = config.BOOTSTRAP_N,
    seed: int = config.SEED,
    level: float = 0.95,
) -> dict[str, float]:
    """Percentile CI of the mean per-row loss, resampling matches."""
    point, boot = _resampled_means(np.asarray(row_loss, dtype="float64"), match_ids, n_boot, seed)
    return _summary(point, boot, level)


def paired_bootstrap_diff(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    match_ids,
    n_boot: int = config.BOOTSTRAP_N,
    seed: int = config.SEED,
    level: float = 0.95,
) -> dict[str, float]:
    """CI of mean(loss_a - loss_b) on the same resampled matches. Negative = model a is better;
    `hi < 0` means the CI excludes 0 in a's favour."""
    diff = np.asarray(loss_a, dtype="float64") - np.asarray(loss_b, dtype="float64")
    point, boot = _resampled_means(diff, match_ids, n_boot, seed)
    return _summary(point, boot, level)
