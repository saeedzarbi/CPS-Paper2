#!/usr/bin/env python3
"""Phase 2: Tube-v2 ACI on/off ablation on validation trace splits."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.trace_sets_v2 import resolve_trace_dir
from configs.videos import CPS_EPISODES
from src.control.two_timescale_conformal import TUBE_V2_ARMS, ControlArmConfig
from src.evaluation.clean_abr_eval import run_arm, write_job_out
from src.evaluation.cluster_stats import cluster_bootstrap_mean_ci

OUT_ROOT = ROOT / "results" / "tube_v2" / "phase2"
REPORT_PATH = ROOT / "docs" / "reports" / "tube_v2_phase2_aci_ablation.md"

METRICS = (
    "rebuffer_total",
    "log_utility_mean",
    "smooth_penalty_mean",
    "episodic_qoe",
    "joint_coverage_ep",
    "joint_tube_coverage_decision",
)

PHASE2_ARMS = (
    "tube_v2_bonf_static",
    "tube_v2_bonf_aci",
    "tube_v2_step_static",
    "tube_v2_step_aci",
)


def _by_trace(rows: list[dict], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        tid = str(r.get("trace_id", "0"))
        out.setdefault(tid, []).append(float(r[key]))
    return out


def _md_table(
    trace_set: str,
    host: str,
    summaries: dict[str, dict[str, tuple[float, float, float]]],
) -> list[str]:
    lines = [
        f"### {trace_set} · host `{host}` (cluster bootstrap 95% CI)",
        "",
        "| arm | rebuffer | log utility | smoothness | QoE | joint cov |",
        "|-----|----------|-------------|------------|-----|-----------|",
    ]
    for arm in PHASE2_ARMS:
        if arm not in summaries:
            continue
        cells = []
        for m in METRICS:
            if m not in summaries[arm]:
                cells.append("—")
                continue
            mu, lo, hi = summaries[arm][m]
            if m == "joint_coverage_ep":
                cells.append(f"{mu:.3f} [{lo:.3f},{hi:.3f}]")
            else:
                cells.append(f"{mu:.2f} [{lo:.2f},{hi:.2f}]")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def run_job(
    trace_set: str,
    host: str,
    episodes: int,
    workers: int,
    cfg: ControlArmConfig,
) -> dict[str, list[dict]]:
    trace_dir = resolve_trace_dir(trace_set, "val")
    all_rows: dict[str, list[dict]] = {}
    for arm in PHASE2_ARMS:
        all_rows[arm] = run_arm(host, trace_dir, arm, episodes, cfg, workers=workers)
    return all_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-sets", nargs="+", default=["5g_markov", "5g_irish"])
    ap.add_argument("--hosts", nargs="+", default=["bba", "mpc"])
    ap.add_argument("--episodes", type=int, default=min(CPS_EPISODES, 102))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = ControlArmConfig()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        meta = {
            "episodes": args.episodes,
            "workers": args.workers,
            "cfg": cfg.__dict__,
            "arms": list(PHASE2_ARMS),
        }
        t0 = time.time()
        for trace_set in args.trace_sets:
            for host in args.hosts:
                try:
                    rows = run_job(trace_set, host, args.episodes, args.workers, cfg)
                except FileNotFoundError as e:
                    print("SKIP", trace_set, e)
                    continue
                out = OUT_ROOT / trace_set / host
                write_job_out(
                    out,
                    rows,
                    host,
                    f"{trace_set}_val",
                    meta={**meta, "wall_s": time.time() - t0},
                    protocol="tube_v2_phase2_v1",
                )
                for arm, ep_rows in rows.items():
                    csv_path = out / f"{arm}_episodes.csv"
                    with csv_path.open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
                        w.writeheader()
                        w.writerows(ep_rows)

    lines = [
        "# Tube-v2 Phase 2 — ACI on/off (validation)",
        "",
        f"Episodes per job (default cap): {args.episodes}. CIs: **trace-cluster bootstrap** (10k, seed 0).",
        "",
    ]
    for trace_set in args.trace_sets:
        for host in args.hosts:
            job_dir = OUT_ROOT / trace_set / host
            summ_path = job_dir / "summary.json"
            if not summ_path.is_file():
                continue
            summaries: dict[str, dict[str, tuple[float, float, float]]] = {}
            for arm in PHASE2_ARMS:
                csv_path = job_dir / f"{arm}_episodes.csv"
                if not csv_path.is_file():
                    continue
                with csv_path.open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                summaries[arm] = {}
                for m in METRICS:
                    if m not in rows[0]:
                        continue
                    summaries[arm][m] = cluster_bootstrap_mean_ci(_by_trace(rows, m))
            lines.extend(_md_table(trace_set, host, summaries))

    lines.extend(
        [
            "## Recommendation (draft)",
            "",
            "Compare `*_aci` vs `*_static` at matched mode (`bonf` / `step`). "
            "If ACI does not improve QoE or coverage on val, de-emphasize it in claims.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
