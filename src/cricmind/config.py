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

# D-005: temporal split by match date (season year). Never re-tune after seeing results.
TRAIN_END_YEAR = 2023  # train: <= 2023
VAL_YEAR = 2024
TEST_START_YEAR = 2025  # test: 2025-2026
TEST_END_YEAR = 2026

# D-008: players with fewer train balls share the UNK embedding.
MIN_TRAIN_BALLS_EMBEDDING = 50
# D-009: archetype eligibility, legal train balls (batters and bowlers separately).
MIN_TRAIN_BALLS_ARCHETYPE = 300
