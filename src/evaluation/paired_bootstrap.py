"""Paired episode bootstrap CIs and win/tie/loss vs a reference arm."""
from __future__ import annotations

import numpy as np


def paired_episode_values(rows: list[dict], key: str) -> np.ndarray:
    vals: list[float] = []
    for r in sorted(rows, key=lambda x: int(x["episode"])):
        if key not in r or r.get(key, "") in ("", "nan"):
            continue
        vals.append(float(r[key]))
    return np.asarray(vals, float)


def paired_bootstrap_mean_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a - b) with paired resampling."""
    if len(a) != len(b) or len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    diff = a - b
    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(diff)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(diff[idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return obs, float(lo), float(hi)


def win_tie_loss(
    a: np.ndarray,
    b: np.ndarray,
    *,
    higher_is_better: bool = True,
    eps: float = 1e-9,
) -> tuple[int, int, int]:
    if len(a) != len(b):
        raise ValueError("paired arrays must match length")
    wins = ties = losses = 0
    for x, y in zip(a, b, strict=True):
        d = (x - y) if higher_is_better else (y - x)
        if abs(d) <= eps:
            ties += 1
        elif d > 0:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def format_ci(mu: float, lo: float, hi: float, digits: int = 2) -> str:
    if mu != mu:
        return "—"
    return f"{mu:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"
