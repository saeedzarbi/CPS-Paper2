"""Cluster bootstrap CIs (trace-level resampling) for Tube-v2 reporting."""
from __future__ import annotations

import numpy as np


def cluster_bootstrap_mean_ci(
    values_by_cluster: dict[str, list[float]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Resample clusters (e.g. trace_id) with replacement; pool episodes and take the mean.
    Returns (point_mean, ci_lo, ci_hi) at ``1 - alpha`` nominal coverage.
    """
    if not values_by_cluster:
        return float("nan"), float("nan"), float("nan")
    clusters = list(values_by_cluster.keys())
    flat = [v for vs in values_by_cluster.values() for v in vs]
    point = float(np.mean(flat))
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        pool = [x for c in pick for x in values_by_cluster[c]]
        boots.append(float(np.mean(pool)))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
