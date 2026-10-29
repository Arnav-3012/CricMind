import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cricmind.features import context
from test_parse import _d, _derived

METHOD = {"winner": "X", "by": {"runs": 5}, "method": "D/L"}


def _ctx(by_over, vocab=("Some League",), **kw):
    return context.build_context(_derived(by_over, **kw), list(vocab))


def _chase(target_overs, n_balls=3, runs=100):
    return [{"team": "Y", "target": {"overs": target_overs, "runs": runs},
             "overs": [{"over": 0, "deliveries": [_d(runs=1)] + [_d()] * (n_balls - 1)}]}]  # fmt: skip


def test_balls_bowled_is_a_running_count_not_over_times_six():
    inn = {"miscounted_overs": {"0": {"balls": "7"}}}  # D-031(2): a 7-ball over
    ctx = _ctx([[_d()] * 7, [_d()]], innings_extra=inn)
    assert ctx["balls_bowled"].tolist() == list(range(8))
    assert ctx["ball_in_over"].tolist() == [1, 2, 3, 4, 5, 6, 7, 1]
    last = ctx.iloc[-1]
    assert last["over"] * 6 + last["ball_in_over"] - 1 == 6  # the formula D-031 forbids is off by one


def test_reduced_chase_counts_balls_from_zero_and_masks_the_revised_inputs():
    ctx = _ctx([[_d()]], extra_innings=_chase(12))  # D-031(1): a 12-over chase, the budget is 72
    chase = ctx[ctx["innings"] == 2]
    assert chase["balls_bowled"].tolist() == [0, 1, 2]  # never 72 - legal_balls_remaining
    for col in context.MASKED:
        assert ctx[col].isna().all(), col  # innings 1 of a reduced match is masked too
    assert np.isnan(chase["crr"].iloc[0])
    assert chase["crr"].tolist()[1:] == [6.0, 3.0]  # 1 run after 1 ball, after 2 balls


def test_unreduced_chase_keeps_target_based_inputs():
    ctx = _ctx([[_d()]], extra_innings=_chase(20))
    assert ctx.loc[ctx["innings"] == 1, "req_rate"].isna().all()  # no target in the first innings
    chase = ctx[ctx["innings"] == 2]
    assert chase["legal_balls_remaining"].tolist() == [120, 119, 118]
    assert chase["target"].tolist() == [100, 100, 100]
    assert chase["phase"].tolist() == ["powerplay"] * 3
    assert chase["req_rate"].iloc[0] == pytest.approx(100 * 6 / 120)
    assert chase["req_rate"].iloc[1] == pytest.approx(99 * 6 / 119)


def test_wide_is_not_faced_and_no_ball_is_faced():
    ctx = _ctx([[_d(runs=1), _d(extras={"wides": 1}), _d(runs=2, extras={"noballs": 1}), _d()]])
    assert ctx["ball_seq"].tolist() == [0, 3]  # wide and no-ball are not rows, but they are counted
    last = ctx.iloc[1]
    assert (last["batter_balls_faced"], last["batter_runs"]) == (2, 3)  # D-031(3): ball 0 + the no-ball
    assert (last["balls_bowled"], last["bowler_balls_bowled"]) == (1, 1)  # legal balls only
    assert last["score_before"] == 5  # the extras count towards the score


def test_reduced_rows_keep_only_the_allow_list():
    kw = {"extra_innings": _chase(20), "outcome": METHOD}
    ctx = _ctx([[_d()]], **kw)
    allowed = [f for f in context.FEATURES if f not in context.MASKED]
    seen = ctx[ctx["balls_bowled"] > 0]
    assert len(seen) == 2
    assert seen[allowed].notna().all().all()  # D-030: position, score, workload stay
    assert ctx[context.MASKED].isna().all().all()
    assert ctx["is_reduced"].all()


def test_unreduced_rows_are_not_masked():
    chase = _ctx([[_d()]], extra_innings=_chase(20))
    chase = chase[chase["innings"] == 2]
    assert chase[context.MASKED].notna().all().all()


def test_is_reduced_and_identities_are_never_model_inputs():
    forbidden = {"is_reduced", "match_id", "batter_id", "bowler_id", "non_striker_id", "venue_key",
                 "venue_id", "y", "outcome"}  # fmt: skip
    # D-029: reduced is only known after the match; Phase 2 baselines carry no player or venue identity
    assert not forbidden & set(context.FEATURES)
    assert set(context.MASKED) <= set(context.FEATURES)


def test_features_do_not_look_ahead():
    a = _ctx([[_d(), _d(), _d(runs=0), _d()]])
    b = _ctx([[_d(), _d(), _d(runs=6), _d()]])  # only ball 2 differs
    assert_frame_equal(a[context.FEATURES].iloc[:3], b[context.FEATURES].iloc[:3])
    assert a["y"].tolist() != b["y"].tolist()
    assert a["score_before"].iloc[3] != b["score_before"].iloc[3]  # it does show up on the next ball


def test_labels_and_rows_are_modelled_balls_only():
    so = {"team": "Y", "super_over": True, "overs": [{"over": 0, "deliveries": [_d(runs=1)]}]}
    ctx = _ctx([[_d(runs=4), _d(extras={"wides": 1}), _d(runs=6), _d()]], extra_innings=[so])
    assert ctx["y"].tolist() == [4, 5, 0]  # the wide and the super over are not rows
    assert set(ctx["split"]) == {"val"} and set(ctx["year"]) == {2024}
    assert ctx["y"].dtype == "int8"


def test_competition_floor_is_fitted_on_train_and_unknowns_become_other():
    train = _derived([[_d()] * 3], dates=["2022-05-01"])
    assert context.competition_vocab(train, floor=3) == ["Some League"]
    assert context.competition_vocab(train, floor=4) == []
    ctx = context.build_context(_derived([[_d()]]), [])
    assert ctx["competition"].tolist() == [context.OTHER]


def test_competition_vocab_refuses_validation_rows():
    with pytest.raises(ValueError, match="train rows only"):
        context.competition_vocab(_derived([[_d()]]))  # 2024 = val


def test_context_frame_columns():
    ctx = _ctx([[_d()]])
    assert isinstance(ctx, pd.DataFrame)
    assert list(ctx.columns) == context.BOOKKEEPING + context.FEATURES + context.IDS
    assert ctx["venue_key"].notna().all()


def _seq_ctx(runs_by_ball, wicket_balls=(), **kw):
    balls = [
        _d(runs=r, wickets=[{"player_out": "A", "kind": "bowled"}] if i in wicket_balls else None)
        for i, r in enumerate(runs_by_ball)
    ]
    overs = [balls[i : i + 6] for i in range(0, len(balls), 6)]
    return context.build_context(_derived(overs, **kw), ["Some League"], sequence=True)


def test_sequence_columns_only_appear_on_request():
    assert not set(context.SEQUENCE) & set(_ctx([[_d()]]).columns)
    assert list(_seq_ctx([1]).columns)[-3:] == context.SEQUENCE


def test_sequence_sums_the_previous_12_legal_balls_and_excludes_the_current_ball():
    runs = [1, 1, 4, 1, 1, 0] + [1] * 13 + [6]  # 20 balls; ball 5 is a wicket
    ctx = _seq_ctx(runs, wicket_balls={5})
    for i in range(len(runs)):
        window = range(max(0, i - 12), i)  # an independent loop, not the cumsum trick
        assert ctx["runs_last12"].iloc[i] == sum(runs[j] for j in window), i
        assert ctx["wkts_last12"].iloc[i] == (1 if 5 in window else 0), i
    wkts = ctx["wkts_last12"].tolist()
    assert (wkts[5], wkts[6], wkts[17], wkts[18]) == (0, 1, 1, 0)  # not itself; in the window; out at ball 18


def test_last_outcome_is_the_previous_legal_balls_class_and_resets_each_innings():
    ctx = _seq_ctx([1, 4, 0], extra_innings=_chase(20))
    first, chase = ctx[ctx["innings"] == 1], ctx[ctx["innings"] == 2]
    none = context.NO_LAST_OUTCOME
    assert first["last_outcome"].tolist() == [none, first["y"].iloc[0], first["y"].iloc[1]]
    assert chase["last_outcome"].tolist() == [none, chase["y"].iloc[0], chase["y"].iloc[1]]
    assert chase["runs_last12"].tolist() == [0, 1, 1]  # nothing carries over from innings 1


def test_a_wide_does_not_enter_the_sequence_window():
    overs = [[_d(runs=2), _d(extras={"wides": 1}), _d(runs=1), _d()]]
    ctx = context.build_context(_derived(overs), ["Some League"], sequence=True)
    assert ctx["runs_last12"].tolist() == [0, 2, 3]  # legal balls only (lean: wide runs are not counted)
    assert ctx["last_outcome"].iloc[1] == ctx["y"].iloc[0]


def test_sequence_features_do_not_look_ahead():
    a = _seq_ctx([1, 1, 0, 1])
    b = _seq_ctx([1, 1, 6, 1])  # only ball 2 differs
    cols = context.FEATURES + context.SEQUENCE
    assert_frame_equal(a[cols].iloc[:3], b[cols].iloc[:3])
    assert a["runs_last12"].iloc[3] != b["runs_last12"].iloc[3]
