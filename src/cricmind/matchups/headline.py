"""Phase 5 step 3: the head-to-head test-season result (D-048). Run after val_fit and both model scorers.

uv run python -m cricmind.matchups.headline    # reports/p5/headline.json

Scores the phase baseline and the four archetype tables here (claim `tables`; the *_ext pair is the D-050
extra); cric2vec and LightGBM+stats come from the per-row probabilities their own processes saved. All are
checked to be on the same test rows.
pandas / numpy / sklearn only, so it never shares a process with torch.
"""

import json
from pathlib import Path

import numpy as np

from cricmind import config
from cricmind.archetypes.vectors import write_json
from cricmind.eval.metrics import bootstrap_ci, brier, ece, paired_bootstrap_diff, row_log_loss
from cricmind.features.players import seen_pairs, unseen_pair_mask
from cricmind.matchups.guard import artifact_dir, load_train_test, report_dir
from cricmind.matchups.table import ARM_NAMES, build_arm
from cricmind.models.baselines import GlobalFreq, PhaseFreq
from cricmind.models.p3_data import val_key

ARMS = [
    "phase_freq",
    "stats_table",
    "emb_table",
    "stats_ext_table",
    "emb_ext_table",
    "cric2vec",
    "lgbm_stats",
]  # the *_ext tables are the D-050 extras
# (a, b): the paired CI is on mean(loss_a - loss_b); negative favours a (D-048(6)).
PAIRS = [
    ("stats_table", "phase_freq"),
    ("emb_table", "phase_freq"),
    ("emb_table", "stats_table"),
    ("cric2vec", "phase_freq"),
    ("lgbm_stats", "cric2vec"),
    ("lgbm_stats", "stats_table"),
    ("stats_ext_table", "phase_freq"),  # D-050 extras
    ("emb_ext_table", "phase_freq"),
    ("emb_ext_table", "stats_ext_table"),
    ("stats_ext_table", "stats_table"),
    ("emb_ext_table", "emb_table"),
]
PAIRED_SLICES = ("all", "labelled_pairs", "ext_labelled_pairs", "unseen_pair")
VERDICT_SLICE = "labelled_pairs"  # D-049
EXT_VERDICT_SLICE = "ext_labelled_pairs"  # D-050: where an *_ext table differs from the phase baseline


def verdict(diff: dict) -> str:
    """D-048(7): a is better only if the CI excludes 0 in its favour, worse only if it excludes 0 against."""
    return "a_better" if diff["hi"] < 0 else "a_worse" if diff["lo"] > 0 else "no_difference"


def load_model_probs(name: str, test_key: int, n: int) -> np.ndarray:
    meta = json.loads((report_dir() / f"{name}_test.json").read_text())
    if meta["test_key"] != test_key or meta["test_rows"] != n:
        raise ValueError(f"{name} was scored on different test rows than this run")
    P = np.load(artifact_dir() / f"test_P_{name}.npy")
    if P.shape != (n, len(config.OUTCOME_CLASSES)):
        raise ValueError(f"{name} probabilities have shape {P.shape}")
    return P


def slice_report(preds: dict, y: np.ndarray, mask: np.ndarray, match_ids: np.ndarray) -> dict:
    out = {"rows": int(mask.sum()), "matches": int(len(np.unique(match_ids[mask]))), "arms": {}}
    if not mask.any():
        return out
    for arm, P in preds.items():
        loss = row_log_loss(P[mask], y[mask])
        out["arms"][arm] = {
            "log_loss": bootstrap_ci(loss, match_ids[mask]),
            "brier": brier(P[mask], y[mask]),
            "ece": ece(P[mask], y[mask]),
        }
    return out


def paired_report(losses: dict, mask: np.ndarray, match_ids: np.ndarray) -> dict:
    out = {}
    for a, b in PAIRS:
        diff = paired_bootstrap_diff(losses[a][mask], losses[b][mask], match_ids[mask])
        out[f"{a}__minus__{b}"] = {**diff, "verdict": verdict(diff)}
    return out


def run(processed_dir: Path | None = None) -> dict:
    val_fit = json.loads((report_dir() / "val_fit.json").read_text())  # s is frozen here, before test opens
    for name in ("cric2vec", "lgbm_stats"):  # check inputs before the claim, so a missing file costs nothing
        for path in (report_dir() / f"{name}_test.json", artifact_dir() / f"test_P_{name}.npy"):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run the {name} scorer first")
    train, test, _ = load_train_test("tables", processed_dir)  # claims the test seasons
    key = val_key(test)
    y, mids = test["y"].to_numpy(), test["match_id"].to_numpy()

    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    preds = {"phase_freq": phase_freq.predict_proba(test)}
    tables = {}
    for name in ARM_NAMES:
        tables[name], _ = build_arm(name, train, phase_freq)
        preds[f"{name}_table"] = tables[name].predict_proba(test, val_fit["arms"][name]["s"])
    for name in ("cric2vec", "lgbm_stats"):
        preds[name] = load_model_probs(name, key, len(test))
    for arm, P in preds.items():
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{arm} rows do not sum to 1")

    losses = {arm: row_log_loss(P, y) for arm, P in preds.items()}
    served = {rep: t.served(test) for rep, t in tables.items()}
    labelled, ext_labelled = served["stats"], served["stats_ext"]  # D-048(1): same players in both reps
    if not ((served["emb"] == labelled).all() and (served["emb_ext"] == ext_labelled).all()):
        raise ValueError("stats and emb tables serve different rows; D-048(1) assumed the same players")
    unseen = unseen_pair_mask(test, seen_pairs(train))
    phase = test["phase"].to_numpy(dtype=object)
    everything = np.ones(len(test), dtype=bool)
    slices = {
        "all": everything,
        "labelled_pairs": labelled,
        "ext_labelled_pairs": ext_labelled,
        "unlabelled_pairs": ~labelled,
        "unseen_pair": unseen,
        "seen_pair": ~unseen,
        "reduced": test["is_reduced"].to_numpy(dtype=bool),
        **{f"phase_{p}": phase == p for p in ("powerplay", "middle", "death")},
    }

    # Sanity (D-048): the phase baseline recomputed by hand, and the table at a huge s collapses onto it.
    big = {rep: t.predict_proba(test, 1e12) for rep, t in tables.items()}
    sanity = {
        "phase_loss_recomputed": float(-np.log(preds["phase_freq"][np.arange(len(y)), y]).mean()),
        "phase_loss_reported": float(losses["phase_freq"].mean()),
        "table_large_s_max_abs_diff": {
            name: float(np.abs(big[name] - preds["phase_freq"]).max()) for name in ARM_NAMES
        },
    }

    report = {
        "test_key": key,
        "test_rows": len(test),
        "test_matches": int(len(np.unique(mids))),
        "labelled_pair_share": float(labelled.mean()),
        "ext_labelled_pair_share": float(ext_labelled.mean()),
        "prior_strength": {name: val_fit["arms"][name]["s"] for name in ARM_NAMES},
        "archetype_k": {
            name: {"batter": tables[name].k_bat, "bowler": tables[name].k_bowl} for name in ARM_NAMES
        },
        "arms_unstable": True,  # D-047: every archetype cell ended at ladder step 3
        "slices": {name: slice_report(preds, y, m, mids) for name, m in slices.items()},
        "paired": {name: paired_report(losses, slices[name], mids) for name in PAIRED_SLICES},
        "verdict_slice": VERDICT_SLICE,  # D-049: the slice that decides; "all" is the coverage number
        "ext_verdict_slice": EXT_VERDICT_SLICE,  # D-050: for the *_ext extras only
        "sanity": sanity,
        "notes": [
            "cric2vec and LightGBM+stats are single-seed (42) on test; val used the 3-seed mean (D-048(8)).",
            "Rows with an unlabelled player or no phase get the phase baseline in both tables (D-048(4)).",
            "Counts come from train only; s was fitted on val and frozen in val_fit.json (D-048(3)).",
        ],
    }
    write_json(report_dir() / "headline.json", report)
    return report


if __name__ == "__main__":
    r = run()
    for arm in ARMS:
        v = r["slices"]["all"]["arms"][arm]["log_loss"]
        print(f"{arm:12s} test log-loss {v['value']:.5f} [{v['lo']:.5f}, {v['hi']:.5f}]")
    print(f"labelled-pair share {r['labelled_pair_share']:.3f}, extended {r['ext_labelled_pair_share']:.3f}")
    for sl in (VERDICT_SLICE, EXT_VERDICT_SLICE):
        for name, d in r["paired"][sl].items():
            print(f"{sl} {name}: {d['value']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}] {d['verdict']}")
