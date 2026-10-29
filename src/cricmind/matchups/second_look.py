"""Phase 5b SECOND LOOK at the spent 2025-26 test seasons (D-054). Exploratory, not a held-out claim.

uv run python -m cricmind.matchups.second_look   # reports/p5b/second_look.json

The Phase 5 headline already read these seasons once (D-048(5)), and Phase 5b exists because of what it
showed, so this is a second look to see whether the val improvement (chain_val) shows up on the same rows
again. It runs once (its own lock claim), only after chain_val has written its decisions, and scores only the
pre-registered variants with the s values frozen in chain_val.json: no new variants, no retuning. Counts are
train-only, as in the headline, so the numbers line up with it. Quote every number with the report's label.
"""

import json
from pathlib import Path

import numpy as np

from cricmind import config
from cricmind.archetypes.vectors import write_json
from cricmind.eval.metrics import bootstrap_ci, paired_bootstrap_diff, row_log_loss
from cricmind.matchups.chain import Chain, cell_keys
from cricmind.matchups.chain_val import MIN_GAIN, PLAYER_LEVELS, THIN_BALLS
from cricmind.matchups.guard import artifact_dir, load_train_test, report_dir
from cricmind.matchups.headline import load_model_probs, verdict
from cricmind.matchups.table import build_arm, extend_labels, load_labels
from cricmind.models.baselines import GlobalFreq, PhaseFreq
from cricmind.models.p3_data import val_key

LABEL = "SECOND LOOK at spent test seasons (D-054): exploratory, not a held-out claim"
# (a, b): the paired CI is on mean(loss_a - loss_b); negative favours a
PAIRS = [
    ("stats_arch", "player_only"),
    ("role_arch", "player_only"),
    ("role_arch", "stats_arch"),
    ("player_only", "stats_table"),
    ("player_only", "phase_freq"),
    ("lgbm_stats", "player_only"),
    ("lgbm_stats", "role_arch"),
    ("cric2vec", "player_only"),
]


def run(processed_dir: Path | None = None) -> dict:
    p5b = config.REPORTS / "p5b"
    chain_val = json.loads((p5b / "chain_val.json").read_text())  # fails here, before the claim, if missing
    val_fit = json.loads((report_dir() / "val_fit.json").read_text())
    for name in ("cric2vec", "lgbm_stats"):
        for path in (report_dir() / f"{name}_test.json", artifact_dir() / f"test_P_{name}.npy"):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: the Phase 5 test scorers must have run")

    train, test, _ = load_train_test("second_look", processed_dir)  # claims the test seasons
    y, mids = test["y"].to_numpy(), test["match_id"].to_numpy()
    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    preds = {"phase_freq": phase_freq.predict_proba(test)}
    table, _ = build_arm("stats", train, phase_freq)
    preds["stats_table"] = table.predict_proba(test, val_fit["arms"]["stats"]["s"])

    specs = {"player_only": PLAYER_LEVELS}
    for name, rep in (("stats_arch", "stats"), ("role_arch", "role")):
        labels, _ = extend_labels(rep, train, load_labels(rep))
        specs[name] = [("cell", cell_keys(labels)), *PLAYER_LEVELS]
    for name, spec in specs.items():
        chain = Chain(spec, phase_freq)
        chain.s = [step["s"] for step in chain_val["variants"][name]["steps"][1:]]  # frozen, never retuned
        preds[name] = chain.refit(train, phase_freq).predict_proba(test)
    key = val_key(test)
    for name in ("cric2vec", "lgbm_stats"):
        preds[name] = load_model_probs(name, key, len(test))

    losses = {arm: row_log_loss(P, y) for arm, P in preds.items()}
    balls = {col: train[col].value_counts() for col in ("batter_id", "bowler_id")}
    thin = (
        (test["batter_id"].map(balls["batter_id"]).fillna(0) < THIN_BALLS)
        | (test["bowler_id"].map(balls["bowler_id"]).fillna(0) < THIN_BALLS)
    ).to_numpy()
    slices = {
        "all": np.ones(len(test), dtype=bool),
        "thin": thin,
        "thick": ~thin,
        "labelled_pairs": table.served(test),
    }
    paired = {
        s: {
            f"{a}__minus__{b}": {**d, "verdict": verdict(d)}
            for a, b in PAIRS
            for d in [paired_bootstrap_diff(losses[a][m], losses[b][m], mids[m])]
        }
        for s, m in slices.items()
    }
    arch = paired["thin"]["stats_arch__minus__player_only"]
    role = paired["all"]["role_arch__minus__stats_arch"]
    decisions = {  # the D-053(5) rules, applied to the second look
        "archetype_level_kept": bool(arch["value"] <= -MIN_GAIN and arch["hi"] < 0),
        "role_beats_stats": bool(role["value"] <= -MIN_GAIN and role["hi"] < 0),
    }
    report = {
        "label": LABEL,
        "test_key": key,
        "test_rows": len(test),
        "slices": {
            s: {
                "rows": int(m.sum()),
                "log_loss": {arm: bootstrap_ci(v[m], mids[m]) for arm, v in losses.items()},
            }
            for s, m in slices.items()
        },
        "paired": paired,
        "decisions": decisions,
        "val_decisions": chain_val["decisions"],  # the pre-registered ones; the second look only checks them
        "agrees_with_val": {k: decisions[k] == chain_val["decisions"][k] for k in decisions},
    }
    write_json(p5b / "second_look.json", report)
    return report


if __name__ == "__main__":
    r = run()
    print(r["label"])
    for s in ("all", "thin"):
        for k, d in r["paired"][s].items():
            print(f"{s} {k}: {d['value']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}] {d['verdict']}")
    print(r["decisions"], "agrees with val:", r["agrees_with_val"])
