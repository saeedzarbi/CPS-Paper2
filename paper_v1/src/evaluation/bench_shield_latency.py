"""Microbenchmark: per-decision wall-clock latency of the Certified Perceptual
Shield projection. Times only `certified_safe_action` over a realistic rollout
so the number reflects the runtime overhead added per chunk.

Run from `new/`:
    python src/evaluation/bench_shield_latency.py --steps 20000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

sys_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(sys_root))

from src.training.certified_perceptual_shield import (
    CPShieldConfig, ConformalConfig, ConformalThroughputEstimator,
    certified_safe_action)
from src.evaluation.eval_certified_shield_v18 import build_base_env


def build_env(trace_dir, buffer_max):
    return build_base_env(str(trace_dir), float(buffer_max), blind=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--trace-dir", default="data/standardized/test_traces_5g_v18")
    ap.add_argument("--buffer", type=float, default=12.0)
    ap.add_argument("--predictive", action="store_true", default=True)
    args = ap.parse_args()

    env = build_env(args.trace_dir, args.buffer)
    cfg = CPShieldConfig(
        epsilon_vmaf=1.0, enabled=True, enable_conformal=True, enable_banking=True,
        predictive=True, lookahead=6, forecast_dips=True,
        conformal=ConformalConfig(alpha=0.10),
    )
    est = ConformalThroughputEstimator(cfg.conformal)

    rng = np.random.default_rng(0)
    n_levels = len(env.BITRATE_LEVELS)
    env.reset()

    times = []
    done = True
    for _ in range(args.steps):
        if done:
            env.reset()
            est = ConformalThroughputEstimator(cfg.conformal)
        a = int(rng.integers(0, n_levels))
        t0 = time.perf_counter()
        safe, interv, sinfo = certified_safe_action(env, a, cfg, est)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        obs, r, term, trunc, info = env.step(safe)
        actual = float(getattr(env, "last_raw_throughput", 2000.0))
        est.update(actual)
        done = bool(term or trunc)

    us = np.array(times) * 1e6
    print(f"shield decisions timed : {us.size}")
    print(f"ladder rungs (L)       : {n_levels}")
    print(f"per-decision mean (us) : {us.mean():.2f}")
    print(f"per-decision median(us): {np.percentile(us, 50):.2f}")
    print(f"per-decision p95   (us): {np.percentile(us, 95):.2f}")
    print(f"per-decision p99   (us): {np.percentile(us, 99):.2f}")
    print(f"chunk duration (s)     : {float(getattr(env, 'CHUNK_DURATION', 4.0))}")


if __name__ == "__main__":
    main()
