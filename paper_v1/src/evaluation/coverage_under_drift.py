"""
Coverage-under-drift stress test for the conformal throughput lower bound.

Injects a synthetic LOS/NLOS-style throughput shock mid-episode on the primary
5G trace pool and reports per-chunk conformal coverage before, during, and after
the shock.  Complements the exchangeability discussion in Section 4 (Method).

Usage (from ``new/``):
  python src/evaluation/coverage_under_drift.py \\
      --trace-dir data/standardized/test_traces_5g \\
      --episodes 204 --alpha 0.10 \\
      --out results/v18_certified/coverage_drift

Outputs:
  - chunks.csv   per-chunk coverage indicators
  - summary.json phase-wise coverage + rolling window stats
  - fig_coverage_drift.pdf (optional figure for the paper)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paths import get_paths
from configs.videos import CPS_EPISODES, EVAL_VIDEOS, MAX_CHUNKS
from src.environment.abr_multi_env_v18 import ABREnv
from src.evaluation.eval_certified_shield_v18 import GreedyPolicy, build_base_env
from src.training.certified_perceptual_shield import (
    CertifiedPerceptualShieldWrapper,
    CPShieldConfig,
    ConformalConfig,
)

P = get_paths()


def _phase(chunk_idx: int, shock_start: int, shock_len: int, warmup: int) -> str:
    if chunk_idx < warmup:
        return "warmup"
    if chunk_idx < shock_start:
        return "pre"
    if chunk_idx < shock_start + shock_len:
        return "shock"
    return "post"


def _inject_shock(env, shock_start: int, shock_len: int, shock_scale: float) -> None:
    """Scale trace throughput over [shock_start, shock_start+shock_len) chunks."""
    trace = getattr(env, "current_trace", None)
    if not trace or "throughput_kbps" not in trace:
        return
    patched = copy.deepcopy(trace)
    tp = patched["throughput_kbps"]
    cd = int(getattr(env, "CHUNK_DURATION", 4))
    lo = shock_start * cd
    hi = (shock_start + shock_len) * cd
    floor = float(getattr(env, "MIN_NETWORK_THROUGHPUT", 50.0))
    for i in range(len(tp)):
        if lo <= i < hi:
            tp[i] = max(float(tp[i]) * shock_scale, floor)
    env.current_trace = patched


def run_episode(
    env,
    policy,
    episode: int,
    shock_start: int,
    shock_len: int,
    shock_scale: float,
    rolling: int,
    warmup: int,
):
    random.seed(1000 + episode)
    obs, _ = env.reset(seed=1000 + episode)
    _inject_shock(env.unwrapped, shock_start, shock_len, shock_scale)

    rows = []
    roll = deque(maxlen=rolling)
    done = False
    while not done:
        chunk_idx = int(getattr(env.unwrapped, "chunk_idx", 0))
        a, _ = policy.predict(obs, deterministic=True)
        tp_lb = float(env.est.lower_bound()) if hasattr(env, "est") else float("nan")
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        realized = float(info.get("throughput", 0.0))
        covered = int(realized >= tp_lb) if tp_lb > 0 and realized > 0 else 0
        valid = int(tp_lb > 0 and realized > 0)
        ph = _phase(chunk_idx, shock_start, shock_len, warmup)
        if valid:
            roll.append(covered)
        rows.append({
            "episode": episode,
            "chunk": chunk_idx,
            "phase": ph,
            "covered": covered,
            "valid": valid,
            "tp_lb_kbps": tp_lb,
            "throughput_kbps": realized,
            "rolling_coverage": float(np.mean(roll)) if roll else float("nan"),
        })
    return rows


def _summarize(rows: list[dict], alpha: float) -> dict:
    def cov(sub):
        v = [r for r in sub if r["valid"]]
        if not v:
            return {"n": 0, "coverage": float("nan")}
        return {"n": len(v), "coverage": float(np.mean([r["covered"] for r in v]))}

    by_phase = {
        ph: cov([r for r in rows if r["phase"] == ph])
        for ph in ("warmup", "pre", "shock", "post")
    }
    steady = [r for r in rows if r["phase"] in ("pre", "shock", "post") and r["valid"]]
    overall = cov(steady)
    return {
        "target_coverage": 1.0 - alpha,
        "overall": overall,
        "by_phase": by_phase,
        "n_chunks_valid": overall["n"],
        "n_episodes": len({r["episode"] for r in rows}),
    }


def _plot(
    rows: list[dict],
    out_pdf: Path,
    alpha: float,
    shock_start: int,
    shock_len: int,
    warmup: int = 20,
):
    import matplotlib.pyplot as plt

    # Mean rolling coverage vs chunk index (across episodes)
    by_chunk: dict[int, list[float]] = {}
    for r in rows:
        if not np.isnan(r["rolling_coverage"]):
            by_chunk.setdefault(r["chunk"], []).append(r["rolling_coverage"])
    xs = sorted(by_chunk)
    ys = [float(np.mean(by_chunk[x])) for x in xs]

    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    # Cold-start first so shock draws on top in the legend order we want.
    ax.axvspan(0, warmup, color="#d9d9d9", alpha=0.85, label="Conformal warm-up")
    ax.axvspan(shock_start, shock_start + shock_len, color="#f4cccc", alpha=0.75, label="Injected shock")
    ax.plot(xs, ys, color="#1f4e79", lw=1.2, label=r"Rolling mean ($W{=}20$)")
    ax.axhline(1.0 - alpha, color="#888888", ls="--", lw=0.9,
               label=rf"Target $1-\alpha={1 - alpha:.2f}$")
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Conformal coverage")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.92)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def replot_from_csv(out_dir: Path, alpha: float, shock_start: int, shock_len: int, warmup: int, rolling: int) -> Path:
    """Regenerate the paper figure from an existing chunks.csv (no re-simulation)."""
    import csv as _csv

    csv_path = out_dir / "chunks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    # coerce types used by _plot
    for r in rows:
        r["chunk"] = int(r["chunk"])
        r["rolling_coverage"] = float(r["rolling_coverage"]) if r["rolling_coverage"] not in ("", "nan") else float("nan")
    fig_path = out_dir / "fig_coverage_drift.pdf"
    _plot(rows, fig_path, alpha, shock_start, shock_len, warmup=warmup)
    # also copy into paper figure dirs if present
    paper_figs = [
        _ROOT / "src" / "paper" / "overleaf_upload" / "figures" / "fig_cps_coverage_drift.pdf",
        _ROOT / "src" / "paper" / "figures" / "fig_cps_coverage_drift.pdf",
    ]
    import shutil
    for dest in paper_figs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, dest)
        print(f"[copy] {dest}")
    return fig_path


def main():
    ap = argparse.ArgumentParser(description="Conformal coverage under injected throughput drift")
    ap.add_argument("--trace-dir", type=Path, default=P["project_root"] / "data/standardized/test_traces_5g")
    ap.add_argument("--episodes", type=int, default=CPS_EPISODES)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--shock-start", type=int, default=28, help="chunk index where shock begins (after warm-up)")
    ap.add_argument("--shock-len", type=int, default=10, help="shock duration in chunks")
    ap.add_argument("--shock-scale", type=float, default=0.30, help="throughput multiplier during shock")
    ap.add_argument("--warmup", type=int, default=20, help="chunks excluded as conformal cold-start")
    ap.add_argument("--rolling", type=int, default=20, help="rolling window for plot (chunks)")
    ap.add_argument("--out", type=Path, default=P["project_root"] / "results/v18_certified/coverage_drift")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--replot-only",
        action="store_true",
        help="Rebuild fig from existing chunks.csv (SCI-04 Path C); do not re-run episodes.",
    )
    args = ap.parse_args()

    if args.replot_only:
        fig = replot_from_csv(
            args.out, args.alpha, args.shock_start, args.shock_len, args.warmup, args.rolling,
        )
        print(f"Replotted {fig}")
        return

    base = build_base_env(args.trace_dir, buffer_max=None, blind=False)
    cfg = CPShieldConfig(
        enabled=True,
        enable_banking=True,
        epsilon_vmaf=args.epsilon,
        enable_conformal=True,
        conformal=ConformalConfig(alpha=args.alpha, window=200, k_predict=5),
    )
    env = CertifiedPerceptualShieldWrapper(base, cfg)
    policy = GreedyPolicy(len(base.BITRATE_LEVELS))

    all_rows: list[dict] = []
    for ep in range(args.episodes):
        all_rows.extend(
            run_episode(
                env, policy, ep,
                args.shock_start, args.shock_len, args.shock_scale,
                args.rolling, args.warmup,
            )
        )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    summary = {
        "config": {
            "trace_dir": str(args.trace_dir),
            "episodes": args.episodes,
            "alpha": args.alpha,
            "shock_start_chunk": args.shock_start,
            "shock_len_chunks": args.shock_len,
            "shock_scale": args.shock_scale,
            "warmup_chunks": args.warmup,
            "policy": "greedy",
            "arm": "certified",
        },
        **_summarize(all_rows, args.alpha),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig_path = out_dir / "fig_coverage_drift.pdf"
    if not args.no_plot:
        _plot(all_rows, fig_path, args.alpha, args.shock_start, args.shock_len, warmup=args.warmup)
        summary["figure"] = str(fig_path)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}/chunks.csv, summary.json", end="")
    if not args.no_plot:
        print(f", {fig_path.name}")
    else:
        print()


if __name__ == "__main__":
    main()
