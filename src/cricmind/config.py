"""Single source of truth: paths, seed, split years, thresholds. Constants only."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

SEED = 42

# D-021: Cricsheet zips update daily, so one snapshot is pinned; bump only via a new D-entry.
DATA_SNAPSHOT = "2026-10-28"
CRICSHEET_URLS = {
    "all_male_json.zip": "https://cricsheet.org/downloads/all_male_json.zip",  # D-020
    "people.csv": "https://cricsheet.org/register/people.csv",
}

# D-020: Phase 1 scope filter.
MATCH_TYPES = frozenset({"T20", "IT20"})
GENDER = "male"
BALLS_PER_OVER = 6
SCHEDULED_OVERS = 20
# Cricsheet event.name renamed over the years -> one label (IT20 is always "T20I", done in parse.py).
COMPETITION_RENAMES = {
    "NatWest T20 Blast": "Vitality Blast",
    "Vitality Blast Men": "Vitality Blast",
    "ICC World Twenty20": "ICC Men's T20 World Cup",
    "World T20": "ICC Men's T20 World Cup",
    "ICC World Twenty20 Qualifier": "ICC Men's T20 World Cup Qualifier",
    "Ram Slam T20 Challenge": "CSA T20 Challenge",
    "MiWAY T20 Challenge": "CSA T20 Challenge",
    "HRV Twenty20": "Super Smash",
}

# D-005: temporal split by match date (season year). Never re-tune after seeing results.
TRAIN_END_YEAR = 2023  # train: <= 2023
VAL_YEAR = 2024
TEST_START_YEAR = 2025  # test: 2025-2026
TEST_END_YEAR = 2026

# D-008: players with fewer train balls share the UNK embedding.
MIN_TRAIN_BALLS_EMBEDDING = 50
# D-038: venues with fewer train balls share a venue UNK (a guess, like the competition floor).
MIN_TRAIN_BALLS_VENUE = 1_000
# D-009: archetype eligibility, legal train balls (batters and bowlers separately).
MIN_TRAIN_BALLS_ARCHETYPE = 300

# D-004: class order is the label id (y = index) for every model and metric.
OUTCOME_CLASSES = ["dot", "1", "2", "3+", "4", "6", "W"]

# Phase 2 (D-032).
TUNE_HOLDOUT_YEAR = TRAIN_END_YEAR  # tuning fits on earlier train years and scores on this one; val unused
PHASE_EDGES = (6, 15)  # over < 6 powerplay, < 15 middle, else death (same as data/profile.py)
WICKET_BAND_EDGES = (3, 6)  # wickets_before 0-2 / 3-5 / 6+ (a guess, not tuned)
MIN_TRAIN_BALLS_COMPETITION = 20_000  # competitions below this share "other" in the context models (a guess)
ECE_BINS = 15
BOOTSTRAP_N = 1000  # resamples of matches for every CI
