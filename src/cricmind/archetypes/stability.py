"""Stability of k-means clusterings: bootstrap and seed ARI, silhouette, the k rule, the ladder, the seeding
experiment (D-044(5) to (8)). sklearn only.

from cricmind.archetypes.stability import Cell, run_ladder, seeding_experiment
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from cricmind import config

KS = range(3, 11)
BOOT_ARI_MIN = 0.6  # the Phase 4 gate and the k-rule cut-off
SILHOUETTE_TIE = 0.005  # within this of the best silhouette, the smaller k wins
PCA_VARIANCE = 0.90  # ladder step 2
SEED_ARI_SEEDS = range(20)
SEEDING_SEEDS = range(50)
# (min train balls, PCA): step 0 is the base clustering, 1 and 2 are the D-044(7) remedies
LADDER = ((config.MIN_TRAIN_BALLS_ARCHETYPE, False), (500, False), (500, True))


def make_model(k: int, pca: bool, seed: int = config.SEED) -> Pipeline:
    """Scaler (+ PCA) + KMeans in one pipeline, so every fit refits the scaler and PCA on its own vectors."""
    steps: list = [StandardScaler()]
    if pca:
        steps.append(PCA(PCA_VARIANCE, svd_solver="full", random_state=config.SEED))
    steps.append(KMeans(k, init="k-means++", n_init=10, random_state=seed))
    return make_pipeline(*steps)


def evaluate_k(X: np.ndarray, boots: list[np.ndarray], k: int, pca: bool) -> dict:
    """Bootstrap ARI over the shared resamples, seed ARI over random_state 0..19, silhouette on the reference
    vectors in the clustering's own space. All ARIs are against the reference labels on the same players."""
    ref = make_model(k, pca).fit(X)
    labels = ref[-1].labels_
    boot = np.array([adjusted_rand_score(labels, make_model(k, pca).fit(Xb)[-1].labels_) for Xb in boots])
    seed = np.array(
        [adjusted_rand_score(labels, make_model(k, pca, s).fit(X)[-1].labels_) for s in SEED_ARI_SEEDS]
    )
    return {
        "k": k,
        "silhouette": float(silhouette_score(ref[:-1].transform(X), labels)),
        "boot_ari_mean": float(boot.mean()),
        "boot_ari_p2_5": float(np.percentile(boot, 2.5)),
        "boot_ari_p97_5": float(np.percentile(boot, 97.5)),
        "seed_ari_mean": float(seed.mean()),
        "inertia": float(ref[-1].inertia_),
    }


def choose_k(table: list[dict]) -> dict | None:
    """D-044(7): candidates have mean bootstrap ARI >= BOOT_ARI_MIN; highest silhouette wins, and within
    SILHOUETTE_TIE of it the smaller k. None when nothing qualifies."""
    ok = [r for r in table if r["boot_ari_mean"] >= BOOT_ARI_MIN]
    if not ok:
        return None
    best = max(r["silhouette"] for r in ok)
    return min((r for r in ok if r["silhouette"] >= best - SILHOUETTE_TIE), key=lambda r: r["k"])


@dataclass
class Cell:
    """One (representation, role): vectors for a superset of players and the same players on each resample."""

    n_balls: np.ndarray  # legal train balls per player, aligned with the rows of X
    X: np.ndarray
    boots: list[np.ndarray]

    def table(self, min_balls: int, pca: bool) -> list[dict]:
        keep = self.n_balls >= min_balls
        return [evaluate_k(self.X[keep], [Xb[keep] for Xb in self.boots], k, pca) for k in KS]


def run_ladder(cell: Cell) -> dict:
    """Walk the D-044(7) ladder for one cell, re-running the selection at each step; the method never
    switches. `ladder_step` 0 = base, 1 = eligibility 500, 2 = plus PCA, 3 = nothing reached the cut-off, so
    the k with the highest mean bootstrap ARI of the last table is reported with `stable` false. Sensitivity
    (150 and 500 balls, no PCA) is always reported; it never changes the choice."""
    cache: dict[tuple[int, bool], list[dict]] = {}

    def table(min_balls: int, pca: bool) -> list[dict]:
        if (min_balls, pca) not in cache:
            cache[(min_balls, pca)] = cell.table(min_balls, pca)
        return cache[(min_balls, pca)]

    for rung, (min_balls, pca) in enumerate(LADDER):
        chosen = choose_k(table(min_balls, pca))
        if chosen:
            step = rung
            break
    else:  # loop variables now hold the last rung
        step = len(LADDER)
        chosen = max(table(min_balls, pca), key=lambda r: (r["boot_ari_mean"], -r["k"]))
    sensitivity = {str(floor): choose_k(table(floor, False)) for floor in (150, 500)}
    return {
        "ladder_step": step,
        "min_balls": min_balls,
        "pca": pca,
        "k": chosen["k"],
        "stable": step < len(LADDER),
        "chosen": chosen,
        "n_players": int((cell.n_balls >= min_balls).sum()),
        "tables": {f"{mb}_{'pca' if p else 'raw'}": t for (mb, p), t in cache.items()},
        "sensitivity_choice": sensitivity,
    }


def seeding_experiment(X: np.ndarray, ks=KS, seeds=SEEDING_SEEDS) -> dict:
    """D-044(8): random vs k-means++ starts with one start each (n_init=1) over `seeds`; spread of the final
    inertia and iterations to converge, per k."""
    Z = StandardScaler().fit_transform(X)
    out: dict = {}
    for init in ("random", "k-means++"):
        out[init] = {}
        for k in ks:
            fits = [KMeans(k, init=init, n_init=1, random_state=s).fit(Z) for s in seeds]
            inertia = np.array([f.inertia_ for f in fits])
            iters = np.array([f.n_iter_ for f in fits])
            out[init][str(k)] = {
                "inertia_mean": float(inertia.mean()),
                "inertia_std": float(inertia.std()),
                "inertia_min": float(inertia.min()),
                "inertia_max": float(inertia.max()),
                "n_iter_mean": float(iters.mean()),
                "n_iter_max": int(iters.max()),
            }
    return out
