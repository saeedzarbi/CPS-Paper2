"""Paired episode bootstrap CIs and win/tie/loss vs a reference arm."""
from __future__ import annotations

import numpy as np


def episode_pair_key(row: dict) -> tuple[str, int, int]:
    trace = str(row.get("trace_file") or row.get("trace_id", ""))
    return (trace, int(row["episode"]), int(row.get("episode_seed", 0)))


def _metric_value(row: dict, key: str) -> float | None:
    if key not in row or row.get(key, "") in ("", "nan"):
        return None
    return float(row[key])


def align_paired_metric(
    rows_a: list[dict],
    rows_b: list[dict],
    key: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Inner-join episodes on (trace_file, episode, episode_seed); assert equal key sets."""
    map_a: dict[tuple[str, int, int], float] = {}
    map_b: dict[tuple[str, int, int], float] = {}
    for r in rows_a:
        v = _metric_value(r, key)
        if v is not None:
            map_a[episode_pair_key(r)] = v
    for r in rows_b:
        v = _metric_value(r, key)
        if v is not None:
            map_b[episode_pair_key(r)] = v
    keys_a = set(map_a)
    keys_b = set(map_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)[:5]
        only_b = sorted(keys_b - keys_a)[:5]
        raise ValueError(
            f"paired episode keys differ for metric {key}: "
            f"only_a={only_a} only_b={only_b} |a|={len(keys_a)} |b|={len(keys_b)}"
        )
    if not keys_a:
        return np.asarray([], float), np.asarray([], float), 0
    keys = sorted(keys_a)
    a = np.asarray([map_a[k] for k in keys], float)
    b = np.asarray([map_b[k] for k in keys], float)
    return a, b, len(keys)


def paired_episode_values(rows: list[dict], key: str) -> np.ndarray:
    """Legacy helper: metric values sorted by episode (prefer ``align_paired_metric``)."""
    vals: list[float] = []
    for r in sorted(rows, key=lambda x: (episode_pair_key(x))):
        v = _metric_value(r, key)
        if v is not None:
            vals.append(v)
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
