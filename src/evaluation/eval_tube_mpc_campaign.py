"""
Tube-MPC campaign evaluation (log-utility; no VMAF in metrics).

Arms: raw_mpc, tube_static (split-conformal tube), tube_aci (online ACI tube).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from configs.videos import CPS_EPISODES, EVAL_VIDEOS, MAX_CHUNKS
from src.environment.paper2_abr_env import Paper2StreamingEnv
from src.evaluation.eval_certified_shield_v18 import RobustMPCPolicy, _bootstrap_ci, _pair_report, _wilcoxon
from src.training.conformal_tube import ConformalTubeConfig
from src.training.tube_mpc_policy import TubeMPCConfig, TubeMPCPolicy

P = get_paths()
ARMS_DEFAULT = "raw_mpc,tube_static,tube_aci"


def build_env(trace_dir):
    return Paper2StreamingEnv(
        video_names=list(EVAL_VIDEOS),
        trace_dir=str(trace_dir),
        max_chunks=MAX_CHUNKS,
        random_seed=0,
        use_lyapunov=True,
        use_future=True,
    )


def make_policy(arm: str, env):
    if arm == "raw_mpc":
        return RobustMPCPolicy(env, horizon=5, past=5)
    if arm == "tube_static":
        cfg = TubeMPCConfig(tube=ConformalTubeConfig(horizon=5, use_aci=False))
        return TubeMPCPolicy(env, cfg)
    if arm == "tube_aci":
        cfg = TubeMPCConfig(tube=ConformalTubeConfig(horizon=5, use_aci=True))
        return TubeMPCPolicy(env, cfg)
    raise ValueError(arm)


def log_utility_mean(env, bitrates_kbps: list[float]) -> float:
    br0 = float(env.BITRATE_LEVELS[0])
    u = [float(np.log(max(b / br0, 1e-6))) for b in bitrates_kbps]
    return float(np.mean(u)) if u else 0.0


def run_arm(arm: str, trace_dir, episodes: int):
    env = build_env(trace_dir)
    policy = make_policy(arm, env)
    rows = []
    br_levels = np.asarray(env.BITRATE_LEVELS, dtype=float)
    for ep in range(episodes):
        random.seed(1000 + ep)
        obs, _ = env.reset(seed=1000 + ep)
        if hasattr(policy, "on_episode_start"):
            policy.on_episode_start()
        reb, bitrates, log_u, cover_hits, cover_n = 0.0, [], [], 0, 0
        done = False
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc
            rb = float(info.get("rebuffer", 0.0))
            reb += rb
            b = float(info.get("bitrate", 0.0))
            bitrates.append(b)
            tp = float(info.get("throughput", getattr(env, "last_raw_throughput", 0.0)))
            if arm != "raw_mpc" and hasattr(policy, "tube"):
                lb = policy.tube.point_lower_bound_kbps()
                if lb > 0 and tp > 0:
                    cover_n += 1
                    cover_hits += int(tp >= lb)
        rows.append(
            {
                "episode": ep,
                "rebuffer_total": reb,
                "log_utility_mean": log_utility_mean(env, bitrates),
                "bitrate_mean_kbps": float(np.mean(bitrates)) if bitrates else 0.0,
                "tube_coverage": (cover_hits / cover_n) if cover_n else float("nan"),
                "alpha_t": float(policy.tube.alpha_t) if arm == "tube_aci" and hasattr(policy, "tube") else float("nan"),
            }
        )
    return rows


def summarize(all_rows, out_dir: Path, tag: str):
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "episode", "rebuffer_total", "log_utility_mean", "bitrate_mean_kbps", "tube_coverage", "alpha_t"])
        for arm, rows in all_rows.items():
            for r in rows:
                w.writerow([arm, r["episode"], r["rebuffer_total"], r["log_utility_mean"], r["bitrate_mean_kbps"], r["tube_coverage"], r["alpha_t"]])

    summary = {"tag": tag, "arms": {}, "comparisons": {}}
    for arm, rows in all_rows.items():
        reb = [r["rebuffer_total"] for r in rows]
        lu = [r["log_utility_mean"] for r in rows]
        m_reb, lo, hi = _bootstrap_ci(reb)
        m_lu, lu_lo, lu_hi = _bootstrap_ci(lu)
        cov = [c for c in [r["tube_coverage"] for r in rows] if not np.isnan(c)]
        summary["arms"][arm] = {
            "rebuffer_total": {"mean": m_reb, "ci_lo": lo, "ci_hi": hi},
            "log_utility_mean": {"mean": m_lu, "ci_lo": lu_lo, "ci_hi": lu_hi},
            "tube_coverage_mean": float(np.mean(cov)) if cov else float("nan"),
        }
    if "tube_aci" in all_rows and "raw_mpc" in all_rows:
        summary["comparisons"]["tube_aci_vs_raw_mpc"] = _pair_report_log(all_rows, "tube_aci", "raw_mpc")
    if "tube_aci" in all_rows and "tube_static" in all_rows:
        summary["comparisons"]["tube_aci_vs_tube_static"] = _pair_report_log(all_rows, "tube_aci", "tube_static")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _pair_report_log(all_rows, treat, ref):
    rt = np.asarray([r["rebuffer_total"] for r in all_rows[treat]], float)
    rr = np.asarray([r["rebuffer_total"] for r in all_rows[ref]], float)
    lt = np.asarray([r["log_utility_mean"] for r in all_rows[treat]], float)
    lr = np.asarray([r["log_utility_mean"] for r in all_rows[ref]], float)
    _, p_reb = _wilcoxon(rt, rr)
    _, p_lu = _wilcoxon(lt, lr)
    return {
        "rebuffer_diff_s": float(rt.mean() - rr.mean()),
        "rebuffer_wilcoxon_p": p_reb,
        "log_utility_diff": float(lt.mean() - lr.mean()),
        "log_utility_wilcoxon_p": p_lu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--episodes", type=int, default=CPS_EPISODES)
    ap.add_argument("--arms", default=ARMS_DEFAULT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_rows = {}
    for arm in arms:
        print(f"[tube-mpc] arm={arm} episodes={args.episodes} trace={args.trace_dir}")
        all_rows[arm] = run_arm(arm, args.trace_dir, args.episodes)
    tag = Path(args.trace_dir).name
    summarize(all_rows, Path(args.out), tag)
    print(f"Wrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
