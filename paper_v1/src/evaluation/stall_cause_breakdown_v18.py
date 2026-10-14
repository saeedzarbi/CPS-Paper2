"""
Per-chunk stall attribution for the Certified arm (greedy, primary 5G).

Replays the same paired seeds as eval_certified_shield_v18.py and attributes
each second of rebuffering to one exclusive bucket (priority order):

  emergency   : B_k <= B_crit (force lowest rung; no Prop. 1 claim)
  cold_start  : residual window shorter than min_calib (fallback scale)
  bound_miss  : realized throughput below the conformal lower bound
  bound_held  : bound cleared (Prop. 1 says stall should be 0 if feasible)

Usage (from new/):
  python src/evaluation/stall_cause_breakdown_v18.py \
      --trace-dir data/standardized/test_traces_5g_v18 \
      --episodes 204 --epsilon 1.0 --alpha 0.10 \
      --out results/v18_certified/greedy_5g/stall_breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from configs.paths import get_paths
from configs.videos import CPS_EPISODES
from src.evaluation.eval_certified_shield_v18 import build_base_env, make_policy
from src.training.certified_perceptual_shield import (
    CertifiedPerceptualShieldWrapper, CPShieldConfig, ConformalConfig)

P = get_paths()
BUCKETS = ("emergency", "cold_start", "bound_miss", "bound_held")


def _bucket(info: dict) -> str:
    if int(info.get("emergency", 0)):
        return "emergency"
    if int(info.get("cold_start", 0)):
        return "cold_start"
    if int(info.get("bound_held", 0)):
        return "bound_held"
    return "bound_miss"


def run(trace_dir: str, buffer_max: float | None, episodes: int,
        epsilon: float, alpha: float) -> tuple[list[dict], dict]:
    base = build_base_env(trace_dir, buffer_max, blind=False)
    cfg = CPShieldConfig(
        enabled=True, enable_banking=True, epsilon_vmaf=epsilon,
        enable_conformal=True,
        conformal=ConformalConfig(alpha=alpha, window=200, k_predict=5),
        safety_margin=0.5, min_buffer=0.3,
        predictive=False, forecast_dips=False,
    )
    env = CertifiedPerceptualShieldWrapper(base, cfg)
    policy = make_policy("greedy", base, ckpt=None, blind=False)

    rows: list[dict] = []
    totals = {b: 0.0 for b in BUCKETS}
    stall_chunks = {b: 0 for b in BUCKETS}
    n_chunks = 0

    for ep in range(episodes):
        random.seed(1000 + ep)
        obs, info = env.reset(seed=1000 + ep)
        done, chunk = False, 0
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc
            rb = float(info.get("rebuffer", 0.0))
            buck = _bucket(info)
            n_chunks += 1
            if rb > 1e-9:
                totals[buck] += rb
                stall_chunks[buck] += 1
            rows.append({
                "episode": ep,
                "chunk": chunk,
                "rebuffer_s": rb,
                "bucket": buck,
                "emergency": int(info.get("emergency", 0)),
                "cold_start": int(info.get("cold_start", 0)),
                "bound_held": int(info.get("bound_held", 0)),
                "buffer_before": float(info.get("buffer_before", float("nan"))),
                "tp_lb_kbps": float(info.get("tp_lb_kbps", float("nan"))),
                "throughput_kbps": float(info.get("throughput", float("nan"))),
            })
            chunk += 1

    total_reb = sum(totals.values())
    summary = {
        "policy": "greedy",
        "arm": "certified",
        "episodes": episodes,
        "epsilon": epsilon,
        "alpha": alpha,
        "n_chunks": n_chunks,
        "rebuffer_total_s": total_reb,
        "rebuffer_mean_per_episode_s": total_reb / max(episodes, 1),
        "seconds": totals,
        "seconds_pct": {b: (100.0 * totals[b] / total_reb if total_reb > 0 else 0.0)
                        for b in BUCKETS},
        "stall_chunks": stall_chunks,
        "priority": "emergency > cold_start > bound_miss > bound_held",
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Certified-arm stall cause breakdown (v18).")
    ap.add_argument("--trace-dir", type=str,
                    default=str(P.get("test_traces_5g_v18", P.get("test_traces"))))
    ap.add_argument("--buffer", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=CPS_EPISODES)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", type=str,
                    default="results/v18_certified/greedy_5g/stall_breakdown")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[stall-breakdown] greedy certified episodes={args.episodes} "
          f"trace_dir={args.trace_dir}")
    rows, summary = run(args.trace_dir, args.buffer, args.episodes,
                        args.epsilon, args.alpha)

    chunk_path = out / "chunks.csv"
    with chunk_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["episode", "chunk", "rebuffer_s", "bucket"])
        w.writeheader()
        w.writerows(rows)

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {chunk_path} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
