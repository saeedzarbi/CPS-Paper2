#!/usr/bin/env python3
"""
Clean-slate ABR campaign: Markov 5G + broadband, four conformal arms, four hosts.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.paths import STANDARDIZED_DIR, get_paths
from configs.videos import CPS_EPISODES, EVAL_VIDEOS, MAX_CHUNKS
from src.control.two_timescale_conformal import ArmName, ControlArmConfig, configure_env_for_arm, make_controlled_policy
from src.environment.clean_abr_env import CleanABREnv
from src.evaluation.eval_certified_shield_v18 import _bootstrap_ci, _wilcoxon, make_policy

P = get_paths()
ARMS: list[ArmName] = ["raw", "static_conformal", "tube_conformal", "two_timescale_conformal"]
TRACE_SETS = {
    "5g_markov": STANDARDIZED_DIR / "test_traces_5g_markov",
    "bb": STANDARDIZED_DIR / "test_traces_v18",
}


def ensure_markov_traces() -> None:
    test = TRACE_SETS["5g_markov"]
    if test.is_dir() and any(test.glob("*.json")):
        return
    script = ROOT / "data" / "prepare_markov_5g_traces.py"
    subprocess.check_call([sys.executable, str(script)], cwd=str(ROOT))


def build_env(trace_dir: Path, reward_mode: str, arm: ArmName) -> CleanABREnv:
    env = CleanABREnv(
        video_names=list(EVAL_VIDEOS),
        trace_dir=str(trace_dir),
        vmaf_dir=str(P["vmaf_scores"]),
        siti_dir=str(P["content_features"]),
        max_chunks=MAX_CHUNKS,
        random_seed=0,
        use_lyapunov=True,
        use_future=True,
        reward_mode=reward_mode,
        enable_early_abort=False,
    )
    configure_env_for_arm(env, arm)
    return env


def run_episode(env: CleanABREnv, policy, arm: ArmName, ep: int, cfg: ControlArmConfig):
    random.seed(1000 + ep)
    obs, _ = env.reset(seed=1000 + ep)
    if hasattr(policy, "on_episode_start"):
        policy.on_episode_start()
    chunk_rebufs = []
    log_utils = []
    vmafs = []
    smooths = []
    cover_hits, cover_n = 0, 0
    aborted = 0

    proj = None
    if arm == "static_conformal" and hasattr(policy, "predict"):
        pass

    done = False
    while not done:
        a, _ = policy.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        done = term or trunc
        chunk_rebufs.append(float(info.get("rebuffer", 0.0)))
        log_utils.append(float(info.get("log_utility", 0.0)))
        vmafs.append(float(info.get("vmaf", 0.0)))
        smooths.append(float(info.get("smooth_penalty", 0.0)))
        aborted += int(info.get("aborted", 0))
        tp = float(info.get("throughput", 0.0))
        if tp > 0 and hasattr(policy, "tube"):
            lb = policy.tube.point_lower_bound_kbps()
            cover_n += 1
            cover_hits += int(tp >= lb)

    reb_total = float(sum(chunk_rebufs))
    reb_p95 = float(np.percentile(chunk_rebufs, 95)) if chunk_rebufs else 0.0
    stallfree = float(np.mean([1.0 if r <= 1e-6 else 0.0 for r in chunk_rebufs]))
    return {
        "episode": ep,
        "rebuffer_total": reb_total,
        "rebuffer_p95_chunk": reb_p95,
        "stallfree_rate": stallfree,
        "log_utility_mean": float(np.mean(log_utils)) if log_utils else 0.0,
        "vmaf_mean": float(np.mean(vmafs)) if vmafs else 0.0,
        "smooth_penalty_mean": float(np.mean(smooths)) if smooths else 0.0,
        "abort_rate": aborted / max(len(chunk_rebufs), 1),
        "conformal_coverage": (cover_hits / cover_n) if cover_n else float("nan"),
    }


def run_arm(policy_kind: str, trace_key: str, arm: ArmName, episodes: int, reward_mode: str):
    trace_dir = TRACE_SETS[trace_key]
    env = build_env(trace_dir, reward_mode, arm)
    base = make_policy(policy_kind, env, None, False)
    cfg = ControlArmConfig()
    controlled = make_controlled_policy(env, arm, base, cfg)
    rows = []
    for ep in range(episodes):
        rows.append(run_episode(env, controlled, arm, ep, cfg))
    return rows


def summarize(rows, tag: str, policy: str, trace_key: str, arm: str):
    def col(k):
        return [r[k] for r in rows]

    summary = {"tag": tag, "policy": policy, "trace": trace_key, "arm": arm, "metrics": {}}
    for k in (
        "rebuffer_total",
        "rebuffer_p95_chunk",
        "stallfree_rate",
        "log_utility_mean",
        "vmaf_mean",
        "smooth_penalty_mean",
        "abort_rate",
        "conformal_coverage",
    ):
        m, lo, hi = _bootstrap_ci(col(k))
        summary["metrics"][k] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    return summary


def write_out(out_dir: Path, all_rows: dict[str, list], summaries: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "arm",
                "episode",
                "rebuffer_total",
                "rebuffer_p95_chunk",
                "stallfree_rate",
                "log_utility_mean",
                "vmaf_mean",
                "smooth_penalty_mean",
                "abort_rate",
                "conformal_coverage",
            ]
        )
        metric_cols = [
            "rebuffer_total",
            "rebuffer_p95_chunk",
            "stallfree_rate",
            "log_utility_mean",
            "vmaf_mean",
            "smooth_penalty_mean",
            "abort_rate",
            "conformal_coverage",
        ]
        for arm, rows in all_rows.items():
            for r in rows:
                w.writerow([arm, r["episode"], *[r[k] for k in metric_cols]])
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["mpc", "greedy", "bba", "bola"], default="mpc")
    ap.add_argument("--trace", choices=list(TRACE_SETS.keys()), default=None)
    ap.add_argument("--episodes", type=int, default=CPS_EPISODES)
    ap.add_argument("--reward-mode", choices=["log_utility", "vmaf"], default="log_utility")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "clean_v2")
    args = ap.parse_args()

    ensure_markov_traces()
    traces = [args.trace] if args.trace else list(TRACE_SETS.keys())

    for trace_key in traces:
        out_dir = args.out / f"{args.policy}_{trace_key}"
        all_rows = {}
        summaries = {"policy": args.policy, "trace": trace_key, "arms": {}}
        for arm in ARMS:
            print(f"[clean] policy={args.policy} trace={trace_key} arm={arm} episodes={args.episodes}")
            rows = run_arm(args.policy, trace_key, arm, args.episodes, args.reward_mode)
            all_rows[arm] = rows
            summaries["arms"][arm] = summarize(rows, trace_key, args.policy, trace_key, arm)["metrics"]
        write_out(out_dir, all_rows, summaries)
        print(f"Wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
