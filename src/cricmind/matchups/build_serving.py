"""Phase 6 serving artifact (D-058): the evaluated models saved as plain tables for the agent and API.

uv run python -m cricmind.matchups.build_serving   # artifacts/p6/serving/, reports/p6/serving.json

Everything is fitted on the train seasons (<= 2023) exactly as it was scored: the D-057 role_arch chain
with the s values frozen in `chain_val.json`, and the Phase 3 LightGBM + stats recipe (trees from
`lgbm_stats.json`, seed 42). The test seasons are never loaded. Val is read for one reason: the saved
files, loaded through `agent/serving.py`, must give back the val log-loss already reported (D-058(2)), so
drift between the saved artifact and the reported model fails the build. Val chooses nothing here.
"""

import json
from pathlib import Path

import pandas as pd

from cricmind import config
from cricmind.agent.serving import COUNTS, DIR, LEVEL_KEYS, NO_PHASE, PROBS, STATE, Serving
from cricmind.archetypes.vectors import ROLES, write_json
from cricmind.eval.metrics import row_log_loss
from cricmind.features.context import FEATURES, OTHER, PHASES
from cricmind.features.stats import SPECS, fit_all, oof_all
from cricmind.matchups.chain import Chain, bat_keys, bowl_keys, cell_keys, pair_keys
from cricmind.matchups.table import extend_labels, load_labels
from cricmind.models import p3_data
from cricmind.models.baselines import GlobalFreq, Lgbm, PhaseFreq

VARIANT = "role_arch"  # D-057
SEED = p3_data.SEEDS[0]  # D-058(5): seed 42, the first Phase 3 seed
TOL = 1e-6  # D-058(2)


def write_chain(chain: Chain, phase_freq: PhaseFreq, out: Path) -> dict[str, float]:
    """Each level as key columns + train class counts; the phase prior as one row per phase plus the global
    shares for rows with no phase. Returns s per level, in chain order."""
    for level in chain.levels:
        # factorize() leaves the level index as a flat Index of key tuples; the names come from the loader
        keys = pd.DataFrame(list(level.index), columns=LEVEL_KEYS[level.name])
        counts = pd.DataFrame(level.counts, columns=COUNTS)
        pd.concat([keys, counts], axis=1).to_parquet(out / f"chain_{level.name}.parquet", index=False)
    prior = pd.DataFrame([*phase_freq.table, phase_freq.parent.p], columns=PROBS)
    prior.insert(0, "phase", [*PHASES, NO_PHASE])
    prior.to_parquet(out / "phase_prior.parquet", index=False)
    return {level.name: float(s) for level, s in zip(chain.levels, chain.s, strict=True)}


def write_labels(train: pd.DataFrame, labels: dict, base: dict, out: Path) -> None:
    frames = []
    for role, col in ROLES.items():
        lab = labels[role]
        frames.append(
            pd.DataFrame(
                {
                    "player_id": lab.index,
                    "role": role,
                    "label": lab.to_numpy("int64"),
                    "placed": lab.index.isin(base[role].index),
                    "train_balls": train[col].value_counts().reindex(lab.index).to_numpy("int64"),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["placed"] = df["placed"].map({True: "clustered", False: "nearest_centre"})
    df.to_parquet(out / "labels.parquet", index=False)


def roles_manifest(role_report: dict) -> dict:
    """Per role: stability and each bin's size and top-3 features vs the population (D-058(4))."""
    out = {}
    for role in ROLES:
        cell = role_report["cells"][role]
        out[role] = {
            "k": cell["k"],
            "stable": cell["stable"],
            "boot_ari": cell["chosen"]["boot_ari_mean"],
            "clusters": {
                str(p["cluster"]): {
                    "size": p["size"],
                    "top": {
                        f: {"z": round(v["z"], 2), "mean": round(v["mean"], 2)} for f, v in p["top"].items()
                    },
                }
                for p in cell["profiles"]
            },
        }
    return out


def write_next_ball(train: pd.DataFrame, trees: int, out: Path, seed: int = SEED) -> dict:
    """Phase 3 final recipe: out-of-fold rates on the training rows, rates from all of train for serving."""
    model = Lgbm(trees, FEATURES + p3_data.BASE_COLS, seed).fit(pd.concat([train, oof_all(train)], axis=1))
    model.model.booster_.save_model(str(out / "next_ball.txt"))
    tables = fit_all(train)
    for prefix, table in tables.items():
        counts = table.counts.set_axis(COUNTS, axis=1).reset_index()
        counts.to_parquet(out / f"rates_{prefix}.parquet", index=False)
    comps = train["competition"].value_counts().drop(OTHER, errors="ignore")
    return {
        "trees": trees,
        "seed": seed,
        "features": model.features,
        "levels": model.levels,
        "rates": {p: {"prior": t.prior.tolist(), "m": t.m, "keys": SPECS[p]} for p, t in tables.items()},
        # lean: a fixed default league for tool calls without one - venue-free hypotheticals need some level
        "default_competition": comps.idxmax(),
    }


def write_state_defaults(train: pd.DataFrame, out: Path) -> None:
    """Median match state at the first legal ball of each (innings, over), unreduced rows (D-058(5))."""
    first = train[(train["ball_in_over"] == 1) & ~train["is_reduced"] & train["innings"].isin([1, 2])]
    med = first.groupby(["innings", "over"])[STATE].median()
    full = pd.MultiIndex.from_product([[1, 2], range(config.SCHEDULED_OVERS)], names=["innings", "over"])
    if not full.isin(med.index).all():
        raise ValueError("some (innings, over) has no unreduced first ball in train")
    med.reset_index().to_parquet(out / "state_defaults.parquet", index=False)


def build(
    train: pd.DataFrame,
    labels: dict[str, pd.Series],
    base: dict[str, pd.Series],
    s: list[float],
    trees: int,
    role_report: dict,
    out: Path = DIR,
) -> dict:
    """Fit the chain (s frozen) and the next-ball model on `train` and write the artifact. Returns the
    manifest."""
    if train["year"].max() > config.TRAIN_END_YEAR:
        raise ValueError("the serving artifact is fitted on train seasons only (D-058(1))")
    out.mkdir(parents=True, exist_ok=True)
    phase_freq = PhaseFreq(GlobalFreq()).fit(train)
    specs = [("cell", cell_keys(labels)), ("bat", bat_keys), ("bowl", bowl_keys), ("pair", pair_keys)]
    chain = Chain(specs, phase_freq)
    chain.s = list(s)
    chain.refit(train, phase_freq)
    write_labels(train, labels, base, out)
    write_state_defaults(train, out)
    manifest = {
        "train_end_year": config.TRAIN_END_YEAR,
        "data_snapshot": config.DATA_SNAPSHOT,
        "classes": config.OUTCOME_CLASSES,
        "chain": {"variant": VARIANT, "s": write_chain(chain, phase_freq, out)},
        "roles": roles_manifest(role_report),
        "next_ball": write_next_ball(train, trees, out),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def run(processed_dir: Path | None = None) -> dict:
    train, val, _ = p3_data.load_frames(processed_dir)
    chain_report = json.loads((config.REPORTS / "p5b" / "chain_val.json").read_text())
    lgbm_report = json.loads((p3_data.report_dir() / "lgbm_stats.json").read_text())
    key = p3_data.val_key(val)
    if chain_report["val_key"] != key or lgbm_report["val_key"] != key:
        raise ValueError("the reports were scored on different val rows; rerun them before building")
    if lgbm_report["final"]["ablation"]:
        raise ValueError("the loader serves SPECS rates only; the ablation recipe needs a new D-entry")
    steps = chain_report["variants"][VARIANT]["steps"][1:]  # step 0 is the phase prior
    if [st["level"] for st in steps] != list(LEVEL_KEYS):
        raise ValueError("chain_val level order differs from the loader's")
    base = load_labels("role")
    labels, _ = extend_labels("role", train, base)  # D-050 rule, as chain_val built its final chain
    role_report = json.loads((config.REPORTS / "p5b" / "role_clusters.json").read_text())
    build(train, labels, base, [st["s"] for st in steps], lgbm_report["final"]["trees"], role_report)

    srv, y = Serving(), val["y"].to_numpy()
    checks = {
        "chain_role_arch": (
            chain_report["slices"]["all"]["log_loss"][VARIANT]["value"],
            row_log_loss(srv.chain_proba(val)[0], y).mean(),
        ),
        f"lgbm_stats_seed{SEED}": (
            lgbm_report["per_seed_val_log_loss"][lgbm_report["seeds"].index(SEED)],
            row_log_loss(srv.next_ball_proba(val), y).mean(),
        ),
    }
    report = {
        "val_key": key,
        "tolerance": TOL,
        "checks": {
            name: {"reported": float(a), "reproduced": float(b), "abs_diff": float(abs(a - b))}
            for name, (a, b) in checks.items()
        },
        "files": sorted(p.name for p in DIR.iterdir()),
    }
    report["passed"] = all(c["abs_diff"] <= TOL for c in report["checks"].values())
    write_json(config.REPORTS / "p6" / "serving.json", report)
    if not report["passed"]:
        raise ValueError(f"serving artifact does not reproduce the reported val log-loss: {report['checks']}")
    return report


if __name__ == "__main__":
    r = run()
    for name, c in r["checks"].items():
        print(f"{name}: reported {c['reported']:.8f} reproduced {c['reproduced']:.8f} ", end="")
        print(f"diff {c['abs_diff']:.1e}")
    print("passed" if r["passed"] else "FAILED")
