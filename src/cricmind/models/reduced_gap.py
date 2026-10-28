"""Is the reduced-row log-loss gap outcome hardness or the features? (D-030 trigger, logs 2026-10-28)

uv run python -m cricmind.models.reduced_gap    # reads reports/p2/baselines.json, writes reduced_gap.json

Goal link: reduced matches are 4.3% of the modelled balls the matchup model trains and is scored on. If the
gap is outcome hardness (not the innings-phase mix, which is checked separately), nothing needs to change; if
the inputs cost accuracy there, D-030's fallback applies. The tree count comes from the 2023 tuning, but the
D-030 decision this supports reads val results (D-035 and the D-032 amendment say so).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricmind import config
from cricmind.eval.metrics import bootstrap_ci, paired_bootstrap_diff, row_log_loss
from cricmind.features.context import FEATURES, MASKED, PHASES, build_context, competition_vocab
from cricmind.features.splits import load_deliveries
from cricmind.models.baselines import Lgbm

K = len(config.OUTCOME_CLASSES)
ALLOW = [f for f in FEATURES if f not in MASKED]  # what a reduced row is allowed to show (D-030)
POSITION = ["over", "ball_in_over", "balls_bowled", "crr"]  # crr = score / balls bowled: encodes position
NO_POSITION = [f for f in ALLOW if f not in POSITION]
VARIANTS = {"A_full": FEATURES, "B_allow_list_only": ALLOW, "C_no_position": NO_POSITION}


def _shares(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=K) / len(y)


def entropy(p: np.ndarray) -> float:
    """Log-loss of the best constant forecast for data with class shares p."""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """Extra log-loss of forecasting with shares q when the data has shares p."""
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def standardised_unreduced(df: pd.DataFrame) -> dict:
    """Unreduced mean loss re-weighted to the reduced rows' (innings, phase) mix.

    `df` has the columns cell, reduced, loss."""
    g = df.groupby(["cell", "reduced"])["loss"].agg(["size", "mean"]).unstack("reduced")
    both = g.dropna()  # cells present in both groups
    w = both[("size", True)] / both[("size", True)].sum()
    size = g["size"].fillna(0)
    share = size / size.sum()  # per group: the cell's share of that group's rows
    mean = g["mean"]  # NaN where a cell has no rows in that group; reported as null below
    return {
        "reduced_loss": float(df.loc[df["reduced"], "loss"].mean()),
        "unreduced_loss": float(df.loc[~df["reduced"], "loss"].mean()),
        "unreduced_loss_at_reduced_mix": float((w * both[("mean", False)]).sum()),
        "cells": {
            c: {
                "reduced_share": float(share.loc[c, True]),
                "unreduced_share": float(share.loc[c, False]),
                "reduced_loss": None if pd.isna(mean.loc[c, True]) else float(mean.loc[c, True]),
                "unreduced_loss": None if pd.isna(mean.loc[c, False]) else float(mean.loc[c, False]),
            }
            for c in g.index
        },
    }


def run(processed_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or config.REPORTS / "p2"
    n_trees = json.loads((out_dir / "baselines.json").read_text())["tuning"]["lightgbm"]["best_iteration"]
    train_d = load_deliveries(("train",), processed_dir)
    vocab = competition_vocab(train_d)
    train = build_context(train_d, vocab)
    val = build_context(load_deliveries(("val",), processed_dir), vocab)
    del train_d
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}  # the test seasons stay unread

    y = val["y"].to_numpy()
    ids = val["match_id"].to_numpy()
    red = val["is_reduced"].to_numpy(dtype=bool)
    loss = {
        name: row_log_loss(Lgbm(n_trees, features=feats).fit(train).predict_proba(val), y)
        for name, feats in VARIANTS.items()
    }

    def pair(a: str, b: str, mask: np.ndarray) -> dict:
        return paired_bootstrap_diff(loss[a][mask], loss[b][mask], ids[mask])

    p_train, p_red, p_un = _shares(train["y"].to_numpy()), _shares(y[red]), _shares(y[~red])
    # The true phase is a report label only (reduced rows have the model's phase input masked).
    first, second = config.PHASE_EDGES
    phase = np.select([val["over"] < first, val["over"] < second], PHASES[:2], PHASES[2])
    cell = val["innings"].astype(int).astype(str) + "/" + pd.Series(phase, index=val.index)
    report = {
        "lightgbm_trees": n_trees,
        "val_rows": {"reduced": int(red.sum()), "unreduced": int((~red).sum())},
        "val_matches": {
            "reduced": int(val.loc[red, "match_id"].nunique()),
            "unreduced": int(val.loc[~red, "match_id"].nunique()),
        },
        # global log-loss on a group = entropy of the group + KL from the train shares:
        # entropy = how hard the group is for any constant forecast, KL = how different it is from train.
        "mix": {
            "class_shares": {
                "classes": config.OUTCOME_CLASSES,
                "train": p_train.tolist(),
                "val_reduced": p_red.tolist(),
                "val_unreduced": p_un.tolist(),
            },
            "entropy_reduced": entropy(p_red),
            "entropy_unreduced": entropy(p_un),
            "kl_reduced_from_train": kl(p_red, p_train),
            "kl_unreduced_from_train": kl(p_un, p_train),
        },
        "position_mix_lightgbm_A": standardised_unreduced(
            pd.DataFrame({"cell": cell.to_numpy(), "reduced": red, "loss": loss["A_full"]})
        ),
        "variants": {
            name: {
                "n_features": len(VARIANTS[name]),
                "reduced": bootstrap_ci(loss[name][red], ids[red]),
                "unreduced": bootstrap_ci(loss[name][~red], ids[~red]),
            }
            for name in VARIANTS
        },
        # Negative = the first model is better. B-A: what never seeing target-based inputs costs.
        # C-B: the D-030 fallback, dropping ball position from the allow-list.
        "paired": {
            "B_minus_A_unreduced": pair("B_allow_list_only", "A_full", ~red),
            "B_minus_A_reduced": pair("B_allow_list_only", "A_full", red),
            "C_minus_B_reduced": pair("C_no_position", "B_allow_list_only", red),
            "C_minus_B_unreduced": pair("C_no_position", "B_allow_list_only", ~red),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reduced_gap.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    for name, v in r["variants"].items():
        print(f"{name:20s} reduced {v['reduced']['value']:.4f}  unreduced {v['unreduced']['value']:.4f}")
    mix = r["mix"]
    print(f"entropy reduced {mix['entropy_reduced']:.4f}  unreduced {mix['entropy_unreduced']:.4f}")
