import numpy as np
import pandas as pd
import pytest

from cricmind.models import baselines as bl
from cricmind.models import reduced_gap as rg
from test_baselines import _frame


def test_entropy_and_kl():
    p = np.array([0.5, 0.5, 0, 0, 0, 0, 0])
    assert rg.entropy(p) == pytest.approx(np.log(2))
    assert rg.kl(p, p) == pytest.approx(0)
    q = np.full(7, 1 / 7)
    assert rg.entropy(p) + rg.kl(p, q) == pytest.approx(np.log(7))  # log-loss under q = entropy + KL


def test_standardising_reweights_unreduced_cells_to_the_reduced_mix():
    df = pd.DataFrame({
        "cell": ["1/powerplay"] * 3 + ["2/death"] * 1 + ["1/powerplay"] * 1 + ["2/death"] * 3,
        "reduced": [False] * 4 + [True] * 4,
        "loss": [1.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 2.0],
    })  # fmt: skip
    got = rg.standardised_unreduced(df)
    # unreduced: powerplay 1.0, death 3.0; the reduced rows are 25% powerplay / 75% death
    assert got["unreduced_loss_at_reduced_mix"] == pytest.approx(0.25 * 1.0 + 0.75 * 3.0)
    assert got["unreduced_loss"] == pytest.approx(1.5)
    assert got["reduced_loss"] == pytest.approx(1.75)


def test_lgbm_can_drop_the_masked_columns():
    train, val = _frame(1000, 0, 2022), _frame(300, 1, 2024)
    model = bl.Lgbm(n_estimators=5, features=rg.NO_POSITION).fit(train)
    P = model.predict_proba(val)
    bl._check_proba(P, len(val), "no_position")
    assert not set(rg.POSITION) & set(rg.NO_POSITION) and "phase" not in rg.ALLOW
