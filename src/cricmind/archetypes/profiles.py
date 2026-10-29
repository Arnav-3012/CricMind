"""Cluster profiles in stats units, for both representations (D-044(10)). Names are written by hand after the
run, so `name` is None here.

from cricmind.archetypes.profiles import profile
"""

import numpy as np
import pandas as pd

N_CENTRAL = 5
N_TOP = 3


def profile(stats: pd.DataFrame, space: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> list[dict]:
    """Per cluster: size, mean raw stats, z of each stat against the eligible population, the top-3 stats by
    |z|, and the 5 players nearest the centroid in `space` (the clustering's own space, so "central" means
    what the clustering means). `stats` is indexed by registry id and aligned with `space` and `labels`."""
    mean, std = stats.mean(), stats.std(ddof=0).replace(0.0, 1.0)
    out = []
    for c in np.unique(labels):
        mask = labels == c
        cluster_mean = stats[mask].mean()
        z = (cluster_mean - mean) / std
        nearest = np.argsort(np.linalg.norm(space[mask] - centers[c], axis=1))[:N_CENTRAL]
        out.append(
            {
                "cluster": int(c),
                "size": int(mask.sum()),
                "name": None,
                "mean_stats": cluster_mean.to_dict(),
                "z_stats": z.to_dict(),
                "top_stats": [
                    {"stat": s, "z": float(z[s]), "mean": float(cluster_mean[s])}
                    for s in z.abs().nlargest(N_TOP).index
                ],
                "central_players": stats.index[mask][nearest].tolist(),
            }
        )
    return out
