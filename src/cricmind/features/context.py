"""Game-state context features for the no-player-identity baselines (Phase 2, D-029 to D-032).

Every running count is "before the ball" (exclusive cumsum), so a row never sees its own outcome or anything
later. Inputs that depend on the final `target` are set to NaN on `is_reduced` rows (D-030 allow-list).

from cricmind.features.context import build_context, competition_vocab
"""

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.features.splits import assign_split

OTHER = "__other__"  # competitions below the train-ball floor share this level
PHASES = ["powerplay", "middle", "death"]

# D-030: these carry the post-hoc revised target (or the label built from the over), so they are NaN on
# reduced rows. Everything else below is fully determined by balls already bowled.
MASKED = ["phase", "legal_balls_remaining", "req_rate", "target"]
NUMERIC = [
    "innings", "over", "ball_in_over", "balls_bowled", "score_before", "wickets_before", "crr",
    "batter_runs", "batter_balls_faced", "bowler_balls_bowled", "legal_balls_remaining", "req_rate", "target",
]  # fmt: skip
CATEGORICAL = ["phase", "competition"]
FEATURES = NUMERIC + CATEGORICAL  # `is_reduced` is deliberately absent: it is only known after the match
BOOKKEEPING = ["match_id", "ball_seq", "match_date", "year", "split", "is_reduced", "y"]
REQ_RATE_CLIP = 36.0  # runs per over; above this the chase is lost anyway


def competition_vocab(deliveries: pd.DataFrame, floor: int = config.MIN_TRAIN_BALLS_COMPETITION) -> list[str]:
    """Competitions with at least `floor` modelled balls. Train only, so it refuses later years."""
    years = pd.to_datetime(deliveries["match_date"]).dt.year
    if years.max() > config.TRAIN_END_YEAR:
        raise ValueError("competition_vocab is fitted on train rows only (D-005)")
    modelled = deliveries[deliveries["is_legal"] & ~deliveries["is_super_over"]]
    counts = modelled["competition"].value_counts()
    return sorted(counts[counts >= floor].index)


def _before(df: pd.DataFrame, values: pd.Series, keys: list[str]) -> pd.Series:
    """Running sum of `values` within `keys`, excluding the current row."""
    v = values.astype("int64")
    return v.groupby([df[k] for k in keys]).cumsum() - v


def build_context(deliveries: pd.DataFrame, competitions: list[str]) -> pd.DataFrame:
    """One row per modelled ball (legal, not a super over): FEATURES + BOOKKEEPING.

    `deliveries` must hold every delivery of its matches, wides and no-balls included: the batter's balls
    faced and the innings' balls bowled are counted over all of them (D-031).
    """
    inn = ["match_id", "innings"]
    d = deliveries[~deliveries["is_super_over"]].sort_values([*inn, "ball_seq"], kind="stable")
    d = d.reset_index(drop=True)
    d["balls_bowled"] = _before(d, d["is_legal"], inn)  # D-031(1): never budget - legal_balls_remaining
    d["bowler_balls_bowled"] = _before(d, d["is_legal"], [*inn, "bowler_id"])  # legal balls only
    d["batter_balls_faced"] = _before(d, d["wides"] == 0, [*inn, "batter_id"])  # D-031(3): not wides
    d["batter_runs"] = _before(d, d["runs_batter"], [*inn, "batter_id"])

    d = d[d["is_legal"]].reset_index(drop=True)  # modelled rows only from here
    d["ball_in_over"] = d["legal_ball_in_over"].astype("float64")  # D-031(2): counted, never over * 6 + ball
    d["target"] = d["target"].astype("float64")
    d["legal_balls_remaining"] = d["legal_balls_remaining"].astype("float64")
    bowled = d["balls_bowled"].to_numpy(dtype="float64")
    d["crr"] = np.where(bowled > 0, d["score_before"] * 6 / np.where(bowled > 0, bowled, 1), np.nan)

    left = d["legal_balls_remaining"].to_numpy()
    needed = (d["target"] - d["score_before"]).to_numpy()
    ok = (d["innings"].to_numpy() == 2) & np.isfinite(needed) & (left > 0)
    d["req_rate"] = np.where(ok, np.clip(needed * 6 / np.where(ok, left, 1), 0, REQ_RATE_CLIP), np.nan)

    powerplay_end, middle_end = config.PHASE_EDGES
    d["phase"] = pd.Series(
        np.select([d["over"] < powerplay_end, d["over"] < middle_end], PHASES[:2], PHASES[2]), index=d.index
    )
    d["competition"] = d["competition"].where(d["competition"].isin(competitions), OTHER)

    reduced = d["is_reduced"].to_numpy(dtype=bool)
    for col in MASKED:  # D-030: set missing, never fill with a revised value
        d.loc[reduced, col] = np.nan

    d["y"] = d["outcome"].map({c: i for i, c in enumerate(config.OUTCOME_CLASSES)}).astype("int8")
    d["split"] = assign_split(d["match_date"])
    d["year"] = pd.to_datetime(d["match_date"]).dt.year
    return d[BOOKKEEPING + FEATURES]
