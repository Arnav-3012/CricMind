import numpy as np
import pytest

from cricmind import config
from cricmind.eval import metrics

K = len(config.OUTCOME_CLASSES)


def _uniform(n):
    return np.full((n, K), 1.0 / K)


def _top(n, conf, correct):
    """n rows whose top class (0) has probability `conf`; `correct` of them have y = 0."""
    P = np.full((n, K), (1 - conf) / (K - 1))
    P[:, 0] = conf
    y = np.array([0] * correct + [1] * (n - correct))
    return P, y


def test_uniform_log_loss_is_ln_k():
    y = np.arange(14) % K
    assert metrics.log_loss(_uniform(14), y) == pytest.approx(np.log(K))


def test_perfect_prediction_has_zero_loss_and_brier():
    y = np.array([0, 3, 6])
    P = np.eye(K)[y]
    assert metrics.log_loss(P, y) == pytest.approx(0, abs=1e-12)
    assert metrics.brier(P, y) == 0


def test_log_loss_is_finite_when_the_true_class_gets_probability_zero():
    P = np.eye(K)[[1]]
    assert np.isfinite(metrics.log_loss(P, np.array([0])))


def test_brier_of_uniform_is_one_minus_one_over_k():
    assert metrics.brier(_uniform(7), np.arange(7)) == pytest.approx(1 - 1 / K)


def test_ece_is_zero_when_confidence_matches_accuracy():
    P, y = _top(10, 0.9, correct=9)
    assert metrics.ece(P, y) == pytest.approx(0, abs=1e-12)


def test_ece_of_an_overconfident_model():
    P, y = _top(10, 0.9, correct=5)  # says 90%, right 50% of the time
    assert metrics.ece(P, y) == pytest.approx(0.4)


def test_ece_handles_full_confidence():
    P = np.eye(K)[[0, 0]]
    assert metrics.ece(P, np.array([0, 1])) == pytest.approx(0.5)


def test_bootstrap_resamples_matches_and_is_seeded():
    loss = np.array([0.0, 0.0, 1.0, 1.0])
    ids = np.array(["a", "a", "b", "b"])
    ci = metrics.bootstrap_ci(loss, ids)
    assert ci["value"] == 0.5
    assert (ci["lo"], ci["hi"]) == (0.0, 1.0)  # two matches: the mean can only be 0, 0.5 or 1
    assert metrics.bootstrap_ci(loss, ids) == ci  # same seed, same CI


def test_bootstrap_of_constant_loss_has_zero_width():
    ci = metrics.bootstrap_ci(np.full(6, 2.0), np.array(list("aabbcc")))
    assert ci["width"] == 0 and ci["value"] == 2.0


def test_paired_diff_sign_and_ci():
    ids = np.array(list("aabbccdd"))
    loss_b = np.linspace(1, 2, 8)
    diff = metrics.paired_bootstrap_diff(loss_b - 0.25, loss_b, ids)  # model a is better by 0.25 everywhere
    assert diff["value"] == pytest.approx(-0.25)
    assert diff["hi"] < 0
    same = metrics.paired_bootstrap_diff(loss_b, loss_b, ids)
    assert same["value"] == 0 and same["width"] == 0
