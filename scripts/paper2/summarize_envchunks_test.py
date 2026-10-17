"""Quick summary of adaptive_test_envchunks primary arms."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results" / "tube_v3" / "adaptive_test_envchunks"
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1])

PRIMARY = {
    "5g_markov": "tube_v2_maxscore_a0.15_safe_fs",
    "5g_harsh": "tube_v2_maxscore_a0.2_safe_fs",
    "bb": "tube_v2_maxscore_a0.2_safe_fs",
    "5g_irish": "tube_v2_maxscore_a0.1_safe_fs",
}
NOM = {"5g_markov": 0.85, "5g_harsh": 0.80, "bb": 0.80, "5g_irish": 0.90}

print("| set | primary | QoE | rebuf | joint | nom 1-a |")
print("|---|---|---|---|---|---|")
for ts, arm in PRIMARY.items():
    d = pd.read_csv(ROOT / ts / f"{arm}.csv", comment="#")
    j = float(d.joint_coverage_ep.mean())
    print(
        f"| {ts} | `{arm}` | {d.episodic_qoe.mean():.1f} | "
        f"{d.rebuffer_total.mean():.2f} | {j:.3f} | {NOM[ts]:.2f} |"
    )
print("\nd vs RobustMPC (episode mean):")
for ts, arm in PRIMARY.items():
    a = pd.read_csv(ROOT / ts / f"{arm}.csv", comment="#")
    r = pd.read_csv(ROOT / ts / "robust_mpc_logu.csv", comment="#")
    m = a.merge(r, on=["trace_file", "episode"], suffixes=("", "_r"))
    print(
        f"  {ts}: dQoE {(m.episodic_qoe - m.episodic_qoe_r).mean():+.1f}, "
        f"drebuf {(m.rebuffer_total - m.rebuffer_total_r).mean():+.2f}s"
    )
p15 = ROOT / "bb" / "tube_v2_maxscore_a0.15_safe_fs.csv"
if p15.is_file():
    d = pd.read_csv(p15, comment="#")
    print(f"\nbb legacy a0.15_safe_fs joint: {d.joint_coverage_ep.mean():.3f}")
