"""
OACS (Paper 2) four-arm evaluation on paired episode seeds.

Arms (see configs.paper2_release):
  raw, static_cps, aci_feasibility, aci_oacs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paper2_release import (
    ARM_ACI_FEASIBILITY,
    ARM_ACI_FULL,
    ARM_P1_STATIC,
    ARM_RAW,
    PAPER2_RELEASE,
    campaign_dir,
)
from configs.paths import get_paths
from configs.videos import CPS_EPISODES, EVAL_VIDEOS, MAX_CHUNKS
from src.environment.paper2_abr_env import Paper2StreamingEnv
from src.evaluation.eval_certified_shield_v18 import (
    _pair_report,
    make_policy,
    summarize,
)
from src.training.aci_perceptual_shield import ACIPerceptualShieldWrapper, ACIShieldConfig, ACIConfig
from src.training.certified_perceptual_shield import (
    CertifiedPerceptualShieldWrapper,
    CPShieldConfig,
    ConformalConfig,
)

from src.utils.mattermost_notify import notify as mm_notify

P = get_paths()


def _mm(msg: str) -> None:
    mm_notify(msg)
DEFAULT_ARMS = f"{ARM_RAW},{ARM_P1_STATIC},{ARM_ACI_FEASIBILITY},{ARM_ACI_FULL}"


def build_base_env(trace_dir, buffer_max, blind):
    class Env(Paper2StreamingEnv):
        pass

    if buffer_max is not None:
        Env.BUFFER_MAX = float(buffer_max)
        Env.BUFFER_TARGET = float(buffer_max) / 2.0
        Env.B_REF = max(2.0, float(buffer_max) / 3.0)
    return Env(
        video_names=list(EVAL_VIDEOS),
        trace_dir=str(trace_dir),
        vmaf_dir=str(P["vmaf_scores"]),
        siti_dir=str(P["content_features"]),
        max_chunks=MAX_CHUNKS,
        random_seed=0,
        use_lyapunov=True,
        use_future=True,
    )


def wrap_env(base, arm: str, epsilon: float, alpha: float):
    if arm == ARM_RAW:
        return base
    if arm == ARM_P1_STATIC:
        cfg = CPShieldConfig(
            enabled=True,
            enable_banking=True,
            epsilon_vmaf=epsilon,
            enable_conformal=True,
            conformal=ConformalConfig(alpha=alpha, window=200, k_predict=5, min_calib=20),
            safety_margin=0.5,
            min_buffer=0.3,
        )
        return CertifiedPerceptualShieldWrapper(base, cfg)
    if arm in (ARM_ACI_FEASIBILITY, ARM_ACI_FULL):
        cfg = ACIShieldConfig(
            enabled=True,
            enable_banking=(arm == ARM_ACI_FULL),
            epsilon_vmaf=epsilon,
            aci=ACIConfig(alpha=alpha),
            safety_margin=0.5,
            min_buffer=0.3,
        )
        return ACIPerceptualShieldWrapper(base, cfg)
    raise ValueError(f"unknown arm: {arm}")


def run_arm(arm, kind, ckpt, blind, trace_dir, buffer_max, episodes, epsilon, alpha):
    base = build_base_env(trace_dir, buffer_max, blind)
    env = wrap_env(base, arm, epsilon, alpha)
    policy = make_policy(kind, base, ckpt, blind)
    rows = []
    for ep in range(episodes):
        random.seed(1000 + ep)
        obs, _info = env.reset(seed=1000 + ep)
        reb, vmafs, bitrates, buffers, chunk_reb = 0.0, [], [], [], []
        interv, done = 0, False
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc
            rb = float(info.get("rebuffer", 0.0))
            reb += rb
            chunk_reb.append(rb)
            vmafs.append(float(info.get("vmaf", 0.0)))
            bitrates.append(float(info.get("bitrate", 0.0)))
            buffers.append(float(info.get("buffer", 0.0)))
            interv += int(info.get("shield_intervened", 0))
        rows.append(
            {
                "episode": ep,
                "rebuffer_total": reb,
                "rebuffer_p95_chunk": float(np.percentile(chunk_reb, 95)),
                "rebuffer_p99_chunk": float(np.percentile(chunk_reb, 99)),
                "stallfree": 1.0 if reb <= 1e-6 else 0.0,
                "vmaf_mean": float(np.mean(vmafs)),
                "bitrate_mean_kbps": float(np.mean(bitrates)),
                "buffer_mean": float(np.mean(buffers)),
                "interv_rate": interv / max(len(chunk_reb), 1),
                "conformal_coverage": float(info.get("conformal_coverage", float("nan"))),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=f"OACS campaign eval ({PAPER2_RELEASE})")
    ap.add_argument("--policy", choices=["greedy", "bba", "bola", "mpc"], required=True)
    ap.add_argument("--trace-dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=CPS_EPISODES)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--arms", type=str, default=DEFAULT_ARMS)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else campaign_dir() / f"{args.policy}_{Path(args.trace_dir).name}"
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_rows = {}
    trace_tag = Path(args.trace_dir).name
    for arm in arms:
        print(f"[oacs] arm={arm} policy={args.policy} episodes={args.episodes} trace={args.trace_dir}")
        all_rows[arm] = run_arm(
            arm, args.policy, None, False, args.trace_dir, None, args.episodes, args.epsilon, args.alpha
        )

    tag = f"{args.policy}_{Path(args.trace_dir).name}"
    summary = summarize(all_rows, args.epsilon, args.alpha, out, tag)
    summary["release"] = PAPER2_RELEASE
    summary["comparisons"]["aci_oacs_vs_static_cps"] = _pair_report(
        all_rows, ARM_ACI_FULL, ARM_P1_STATIC, args.epsilon
    )
    summary["comparisons"]["aci_oacs_vs_aci_feasibility"] = _pair_report(
        all_rows, ARM_ACI_FULL, ARM_ACI_FEASIBILITY, args.epsilon
    )
    summary["comparisons"]["aci_oacs_vs_raw"] = _pair_report(all_rows, ARM_ACI_FULL, ARM_RAW, args.epsilon)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}/summary.json")
    aci = summary["arms"].get(ARM_ACI_FULL, {})
    static = summary["arms"].get(ARM_P1_STATIC, {})
    reb_oacs = aci.get("rebuffer_total", {}).get("mean", float("nan"))
    reb_cps = static.get("rebuffer_total", {}).get("mean", float("nan"))
    _mm(
        f":white_check_mark: `{tag}` | reb: aci_oacs={reb_oacs:.2f}s static_cps={reb_cps:.2f}s"
    )


if __name__ == "__main__":
    main()
