"""Phase 5b: does the hierarchical chain beat player-only shrinkage, and which archetypes help it?

uv run python -m cricmind.matchups.chain_val   # reports/p5b/chain_val.json   (D-052, D-053)

The test seasons (2025-26) are never loaded: the Phase 5 headline spent them (D-048(5)), so no test-season
claim can come from here. Following D-032, each level's s is tuned on the 2023 holdout with counts from years
before 2023; the chain is then refit on all of train with s frozen and 2024 val is scored once, so val
chooses nothing. Val was used in Phase 3 to compare models but not to tune any of these chains. D-055: the
Phase 4 / D-052 clusters saw 2023 rows, which made the `cell` level look too good on the holdout, so the
tuning stage uses clusters refit on the years before 2023 (same k, floor and PCA).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.archetypes.vectors import write_json
from cricmind.eval.metrics import bootstrap_ci, paired_bootstrap_diff, row_log_loss
from cricmind.matchups.chain import Chain, bat_keys, bowl_keys, cell_keys, pair_keys
from cricmind.matchups.headline import verdict
from cricmind.matchups.table import chosen_spec, extend_labels, load_labels, refit_labels
from cricmind.models import p3_data
from cricmind.models.baselines import GlobalFreq, PhaseFreq

THIN_BALLS = 500  # D-053(5): a player under this many train balls is "thin"
MIN_GAIN = 0.002  # D-053(6), nats; the same bar as the D-016 sequence test
PLAYER_LEVELS = [("bat", bat_keys), ("bowl", bowl_keys), ("pair", pair_keys)]
# (a, b): the paired CI is on mean(loss_a - loss_b); negative favours a
PAIRS = [
    ("stats_arch", "player_only"),
    ("role_arch", "player_only"),
    ("role_arch", "stats_arch"),
    ("player_only", "phase_freq"),
    ("stats_arch", "phase_freq"),
    ("role_arch", "phase_freq"),
    ("lgbm_stats", "player_only"),
    ("lgbm_stats", "stats_arch"),
    ("lgbm_stats", "role_arch"),
]


def reference_losses(val: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-row val losses of the Phase 3 models (mean of 3 seeds); refuses losses from different val rows."""
    key = p3_data.val_key(val)
    out = {}
    for name in ("cric2vec", "lgbm_stats"):
        if json.loads((p3_data.report_dir() / f"{name}.json").read_text())["val_key"] != key:
            raise ValueError(f"{name} val losses are from different val rows")
        out[name] = np.load(p3_data.artifact_dir() / f"val_loss_{name}.npy").mean(axis=0)
    return out


def run(processed_dir: Path | None = None) -> dict:
    train, val, _ = p3_data.load_frames(processed_dir)
    fit_part, holdout = p3_data.holdout_split(train)  # D-032: s is tuned on 2023, counts before 2023
    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    tune_freq = PhaseFreq(GlobalFreq()).fit(fit_part)
    y, mids = val["y"].to_numpy(), val["match_id"].to_numpy()

    # D-055: the tuning chain's clusters are refit on the years before 2023 (same k / floor / PCA), so the
    # holdout year cannot decide cluster membership; the scored chain uses the clusters fitted on all train.
    tune_specs, final_specs, label_info = {"player_only": PLAYER_LEVELS}, {"player_only": PLAYER_LEVELS}, {}
    for name, rep in (("stats_arch", "stats"), ("role_arch", "role")):
        base, models = refit_labels(rep, fit_part, chosen_spec(rep))
        tune_labels, _ = extend_labels(rep, fit_part, base, models=models)
        labels, label_info[name] = extend_labels(rep, train, load_labels(rep))
        tune_specs[name] = [("cell", cell_keys(tune_labels)), *PLAYER_LEVELS]
        final_specs[name] = [("cell", cell_keys(labels)), *PLAYER_LEVELS]

    report: dict = {
        "val_key": p3_data.val_key(val),
        "val_rows": len(val),
        "leak_free_tuning": True,  # D-055
        "variants": {},
    }
    losses = {"phase_freq": row_log_loss(phase_freq.predict_proba(val), y)}
    for name in final_specs:
        tuned = Chain(tune_specs[name], tune_freq)
        tuning = tuned.fit(fit_part, holdout)  # s from the 2023 holdout only
        chain = Chain(final_specs[name], phase_freq)
        chain.s = list(tuned.s)  # s frozen
        chain.refit(train, phase_freq)  # counts and clusters from all of train; val is scored once
        report["variants"][name] = {**tuning, "labels": label_info.get(name)}
        losses[name] = row_log_loss(chain.predict_proba(val), y)
    losses.update(reference_losses(val))

    balls = {col: train[col].value_counts() for col in ("batter_id", "bowler_id")}
    thin = (
        (val["batter_id"].map(balls["batter_id"]).fillna(0) < THIN_BALLS)
        | (val["bowler_id"].map(balls["bowler_id"]).fillna(0) < THIN_BALLS)
    ).to_numpy()
    slices = {"all": np.ones(len(val), dtype=bool), "thin": thin, "thick": ~thin}
    report["slices"] = {
        s: {"rows": int(m.sum()), "log_loss": {arm: bootstrap_ci(v[m], mids[m]) for arm, v in losses.items()}}
        for s, m in slices.items()
    }
    report["paired"] = {
        s: {
            f"{a}__minus__{b}": {**d, "verdict": verdict(d)}
            for a, b in PAIRS
            for d in [paired_bootstrap_diff(losses[a][m], losses[b][m], mids[m])]
        }
        for s, m in slices.items()
    }

    arch = report["paired"]["thin"]["stats_arch__minus__player_only"]
    role = report["paired"]["all"]["role_arch__minus__stats_arch"]
    role_report = json.loads((config.REPORTS / "p5b" / "role_clusters.json").read_text())
    report["decisions"] = {  # D-053(6), fixed before the run
        "min_gain": MIN_GAIN,
        "archetype_level_kept": bool(arch["value"] <= -MIN_GAIN and arch["hi"] < 0),
        "role_beats_stats": bool(role["value"] <= -MIN_GAIN and role["hi"] < 0),
        "role_clusters_stable": {r: c["stable"] for r, c in role_report["cells"].items()},
    }
    write_json(config.REPORTS / "p5b" / "chain_val.json", report)
    return report


if __name__ == "__main__":
    r = run()
    for name, v in r["variants"].items():
        print(name, [(s["level"], round(s["tune_log_loss"], 5)) for s in v["steps"]])
    for s in ("all", "thin"):
        for k, d in r["paired"][s].items():
            print(f"{s} {k}: {d['value']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}] {d['verdict']}")
    print(r["decisions"])
