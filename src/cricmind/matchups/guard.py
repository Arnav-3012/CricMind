"""The test seasons are read once per scorer (D-048(5)).

Each scorer claims the test seasons in `reports/p5/test_lock.json` before it loads them; a second claim by the
same scorer raises. The lock is a file, so clearing one entry is a manual, visible act and needs a
`logs.md` row.
Imports neither torch nor lightgbm (mistakes.md OpenMP rule), so every scorer process can use it.

from cricmind.matchups.guard import load_train_test
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from cricmind import config
from cricmind.features.context import build_context, competition_vocab
from cricmind.features.splits import load_deliveries
from cricmind.models.p3_data import check_split_integrity

SCORERS = ("tables", "cric2vec", "lgbm_stats", "second_look")  # second_look: D-054, labelled as such


def artifact_dir() -> Path:
    return config.ARTIFACTS / "p5"


def report_dir() -> Path:
    return config.REPORTS / "p5"


def lock_path() -> Path:
    return report_dir() / "test_lock.json"


def claim_test(scorer: str) -> None:
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}; choose from {SCORERS}")
    path = lock_path()
    claims = json.loads(path.read_text()) if path.exists() else {}
    if scorer in claims:
        raise RuntimeError(
            f"{scorer} already read the test seasons at {claims[scorer]} (D-048(5)); "
            "clearing the claim by hand needs a logs.md row"
        )
    claims[scorer] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(claims, indent=2) + "\n")


def load_train_test(
    scorer: str, processed_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train and test context frames and the train-only competition vocab. Claims the test seasons first."""
    claim_test(scorer)
    train_d = load_deliveries(("train",), processed_dir)
    competitions = competition_vocab(train_d)
    train = build_context(train_d, competitions)
    test = build_context(load_deliveries(("test",), processed_dir), competitions)
    assert set(train["split"]) == {"train"} and set(test["split"]) == {"test"}
    check_split_integrity(train, test)
    return train, test, competitions
