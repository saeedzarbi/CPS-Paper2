#!/usr/bin/env python3
"""Replay trace throughputs and report per-step tube coverage (Tube-v2)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.paths import STANDARDIZED_DIR
from src.training.conformal_tube_v2 import ConformalThroughputTubeV2, ConformalTubeV2Config

TRACE_DIRS = {
    "5g_markov": STANDARDIZED_DIR / "test_traces_5g_markov",
    "bb": STANDARDIZED_DIR / "test_traces_v18",
    "5g_v18": STANDARDIZED_DIR / "test_traces_5g_v18",
    "fcc_fixed": STANDARDIZED_DIR / "eval_fcc_fixed",
    "5g_irish": STANDARDIZED_DIR / "test_traces_5g_irish_mmsys20",
}


def replay_trace(tp: np.ndarray, cfg: ConformalTubeV2Config) -> dict[int, list[bool]]:
    tube = ConformalThroughputTubeV2(cfg)
    h = max(int(cfg.horizon), 1)
    bounds_hist: list[np.ndarray] = []
    for t, actual in enumerate(tp):
        bounds_hist.append(
            tube.horizon_lower_bounds_kbps(h, decision_chunk=t).copy()
        )
        tube.observe(float(actual), realized_chunk=t)
    covered: dict[int, list[bool]] = {i: [] for i in range(h)}
    for c in range(len(tp)):
        for step in range(h):
            j = c - step - 1
            if j < 0 or j >= len(bounds_hist):
                continue
            covered[step].append(float(tp[c]) >= float(bounds_hist[j][step]) - 1e-6)
    return covered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "tube_v2" / "coverage")
    ap.add_argument("--trace-set", choices=list(TRACE_DIRS), nargs="*", default=list(TRACE_DIRS))
    ap.add_argument("--modes", nargs="*", default=["per_step", "bonferroni"])
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for set_name in args.trace_set:
        tdir = TRACE_DIRS[set_name]
        if not tdir.is_dir():
            continue
        for mode in args.modes:
            for use_aci in (False, True):
                cfg = ConformalTubeV2Config(
                    mode=mode,
                    alpha=args.alpha,
                    horizon=args.horizon,
                    use_aci=use_aci,
                )
                alpha_h = cfg.alpha / cfg.horizon if mode == "bonferroni" else cfg.alpha
                agg: dict[int, list[bool]] = {}
                for path in sorted(tdir.glob("*.json")):
                    data = json.loads(path.read_text(encoding="utf-8"))
                    tp = np.asarray(data["throughput_kbps"], dtype=float)
                    cov = replay_trace(tp, cfg)
                    for h_idx, flags in cov.items():
                        agg.setdefault(h_idx, []).extend(flags)
                        rate = float(np.mean(flags)) if flags else float("nan")
                        rows.append(
                            {
                                "trace_id": path.stem,
                                "trace_set": set_name,
                                "h": h_idx + 1,
                                "alpha_h": alpha_h,
                                "mode": mode,
                                "use_aci": use_aci,
                                "covered_rate": rate,
                                "n": len(flags),
                            }
                        )
                if agg:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    hs = sorted(agg)
                    rates = [float(np.mean(agg[i])) for i in hs]
                    ax.plot([i + 1 for i in hs], rates, marker="o", label=f"{set_name} {mode} aci={use_aci}")
                    ax.axhline(1 - args.alpha, ls="--", color="gray", label="1-alpha")
                    if mode == "bonferroni":
                        ax.axhline(1 - args.alpha / args.horizon, ls=":", color="gray", label="1-alpha/H")
                    ax.set_xlabel("horizon step h")
                    ax.set_ylabel("empirical coverage")
                    ax.legend(fontsize=7)
                    fig.tight_layout()
                    tag = f"{set_name}_{mode}_aci{int(use_aci)}"
                    fig.savefig(args.out_dir / f"coverage_{tag}.png", dpi=120)
                    fig.savefig(args.out_dir / f"coverage_{tag}.pdf")
                    plt.close(fig)

    csv_path = args.out_dir / "coverage_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
