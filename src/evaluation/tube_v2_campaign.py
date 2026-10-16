#!/usr/bin/env python3
"""Tube-v2 campaign driver (incremental CSV, resume, phases)."""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.paths import STANDARDIZED_DIR
from configs.trace_sets_v2 import TRACE_SETS_V2, calibration_dirs_for_eval
from configs.videos import CPS_EPISODES
from src.control.two_timescale_conformal import (
    SMOKE_ARMS,
    TUBE_V2_ARMS,
    ControlArmConfig,
)
from src.evaluation.clean_abr_eval import build_env, run_episode, write_job_out
from src.evaluation.eval_certified_shield_v18 import make_policy
from src.evaluation.eval_seeds import DEFAULT_MASTER_SEED
from src.control.two_timescale_conformal import make_controlled_policy
from src.training.fixed_quantile_mpc import load_or_fit_quantile, log_fitted_quantile, quantile_cache_path

RESULTS_ROOT = ROOT / "results" / "tube_v2"
SMOKE_TRACE_SETS = {
    "5g_harsh": STANDARDIZED_DIR / "test_traces_5g_harsh_v15",
    "5g_v18": STANDARDIZED_DIR / "test_traces_5g_v18",
    "5g_markov": STANDARDIZED_DIR / "test_traces_5g_markov",
    "bb": STANDARDIZED_DIR / "test_traces_v18",
}
SMOKE_EPISODES_DEFAULT = 12
HORIZON = 5

PHASE_ARMS: dict[int, tuple[str, ...]] = {
    2: tuple(
        a
        for a in TUBE_V2_ARMS
        if a
        in (
            "tube_v2_bonf_static",
            "tube_v2_bonf_aci",
            "tube_v2_step_static",
            "tube_v2_step_aci",
        )
    ),
    3: SMOKE_ARMS,
}


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _episode_csv(phase: int, trace_set: str, host: str, arm: str) -> Path:
    return RESULTS_ROOT / f"phase{phase}" / trace_set / host / f"{arm}.csv"


def _read_done(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as f:
        return {int(r["episode"]) for r in csv.DictReader(f)}


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _prefit_quantile(trace_sets: list[str], force: bool) -> dict[str, float]:
    qs: dict[str, float] = {}
    for ts in trace_sets:
        try:
            dirs = calibration_dirs_for_eval(ts)
        except (FileNotFoundError, KeyError) as e:
            print("SKIP quantile prefit", ts, e)
            continue
        q = load_or_fit_quantile(
            dirs,
            trace_set=ts,
            split="calib",
            alpha=0.10,
            k_predict=5,
            force_refit=force,
        )
        qs[ts] = q
        log_fitted_quantile(q, 0.10, quantile_cache_path(ts, "calib", 0.10, 5))
    return qs


def run_incremental(
    phase: int,
    trace_dir: Path,
    trace_key: str,
    host: str,
    arm: str,
    episodes: int,
    cfg: ControlArmConfig,
    *,
    resume: bool = True,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> list[dict]:
    csv_path = _episode_csv(phase, trace_key, host, arm)
    done = _read_done(csv_path) if resume else set()
    rows: list[dict] = []
    env = build_env(trace_dir, arm)  # type: ignore[arg-type]
    base = make_policy(host, env, None, False)
    policy = make_controlled_policy(
        env, arm, base, cfg, trace_set_key=trace_key
    )  # type: ignore[arg-type]
    for ep in range(episodes):
        if ep in done:
            continue
        if hasattr(policy, "on_episode_start"):
            policy.on_episode_start()
        row = run_episode(
            env,
            policy,
            ep,
            master_seed=master_seed,
            trace_key=trace_key,
        )
        row["phase"] = phase
        row["trace_set"] = trace_key
        row["host"] = host
        row["arm"] = arm
        _append_row(csv_path, row)
        rows.append(row)
    return rows


def _resolve_trace(trace_set: str, split: str, data_dir: Path | None) -> Path:
    if data_dir is not None:
        os.environ["CPS_DATA_DIR"] = str(data_dir)
        os.environ["CPS_PAPER2_DATA_ROOT"] = str(data_dir)
    if trace_set in SMOKE_TRACE_SETS:
        p = SMOKE_TRACE_SETS[trace_set]
        if p.is_dir():
            return p
    from configs.trace_sets_v2 import resolve_trace_dir

    return resolve_trace_dir(trace_set, split)


def run_phases(
    phases: list[int],
    *,
    trace_sets: list[str],
    hosts: list[str],
    split: str,
    episodes: int,
    workers: int,
    resume: bool,
    data_dir: Path | None,
    master_seed: int,
    refit_quantile: bool,
    arms_filter: tuple[str, ...] | None = None,
) -> dict:
    if refit_quantile:
        _prefit_quantile(trace_sets, force=True)
    cfg = ControlArmConfig()
    meta = {
        "git_sha": _git_sha(),
        "episodes": episodes,
        "workers": workers,
        "phases": phases,
        "split": split,
        "master_seed": master_seed,
    }
    t0 = time.time()
    summary: dict = {"jobs": []}
    for phase in phases:
        arms = PHASE_ARMS.get(phase, ())
        if arms_filter:
            arms = tuple(a for a in arms if a in arms_filter)
        for trace_set in trace_sets:
            try:
                trace_dir = _resolve_trace(trace_set, split, data_dir)
            except (FileNotFoundError, KeyError) as e:
                print("SKIP trace", trace_set, e)
                continue
            for host in hosts:
                all_rows: dict[str, list[dict]] = {}
                for arm in arms:
                    if workers <= 1:
                        all_rows[arm] = run_incremental(
                            phase,
                            trace_dir,
                            trace_set,
                            host,
                            arm,
                            episodes,
                            cfg,
                            resume=resume,
                            master_seed=master_seed,
                        )
                    else:
                        from src.evaluation.clean_abr_eval import run_arm as run_arm_par

                        all_rows[arm] = run_arm_par(
                            host,
                            trace_dir,
                            arm,
                            episodes,
                            cfg,
                            workers=workers,
                            master_seed=master_seed,
                            trace_key=trace_set,
                        )
                out = RESULTS_ROOT / f"phase{phase}" / trace_set / host
                write_job_out(
                    out,
                    all_rows,
                    host,
                    f"{trace_set}_{split}",
                    meta={**meta, "wall_s": time.time() - t0},
                    protocol=f"tube_v2_phase{phase}",
                )
                summary["jobs"].append(str(out))
    summary["wall_s"] = time.time() - t0
    (RESULTS_ROOT / "last_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _mean_col(rows: list[dict], key: str) -> str:
    if not rows or key not in rows[0]:
        return "—"
    vals = [float(r[key]) for r in rows if r.get(key, "") not in ("", "nan")]
    return f"{sum(vals) / len(vals):.3f}" if vals else "—"


def _cov_cols(h: int) -> str:
    return " | ".join(f"cov_h{i + 1}" for i in range(h))


def smoke_table(
    arms: tuple[str, ...],
    trace_sets: tuple[str, ...],
    episodes: int,
    *,
    fitted_q: dict[str, float] | None = None,
    determinism_note: str = "",
) -> str:
    lines = [
        "# Tube-v2 smoke (Phase 3 baselines + tube_v2 ACI)",
        "",
        f"Episodes per arm: {episodes}. Seeds: `eval_seeds.episode_rng(master={DEFAULT_MASTER_SEED}, trace, ep)`.",
        "",
    ]
    if fitted_q:
        for ts, q in sorted(fitted_q.items()):
            lines.append(f"`mpc_fixed_quantile` q={q:.4f} (trace_set={ts}, calib train/val).")
        lines.append("")
    if determinism_note:
        lines.extend([determinism_note, ""])
    header = (
        "| trace | arm | QoE | rebuf | smooth | logU | VMAF | cold% | joint | "
        + _cov_cols(HORIZON)
        + " | mean_α_t |"
    )
    sep = "|-------|-----|-----|-------|--------|------|------|-------|-------|" + "|".join(
        ["------"] * HORIZON
    ) + "|--------|"
    lines.extend([header, sep])
    for trace_set in trace_sets:
        for arm in arms:
            path = _episode_csv(3, trace_set, "bba", arm)
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            cov_cells = " | ".join(_mean_col(rows, f"cov_h{i + 1}") for i in range(HORIZON))
            lines.append(
                "| {trace} | `{arm}` | {qoe} | {reb} | {sm} | {lu} | {vmaf} | {cold} | {jc} | {cov} | {at} |".format(
                    trace=trace_set,
                    arm=arm,
                    qoe=_mean_col(rows, "episodic_qoe"),
                    reb=_mean_col(rows, "rebuffer_total"),
                    sm=_mean_col(rows, "smooth_penalty_mean"),
                    lu=_mean_col(rows, "log_utility_mean"),
                    vmaf=_mean_col(rows, "mean_vmaf"),
                    cold=_mean_col(rows, "tube_cold_start_frac"),
                    jc=_mean_col(rows, "joint_tube_coverage_decision"),
                    cov=cov_cells,
                    at=_mean_col(rows, "tube_mean_alpha_t"),
                )
            )
    return "\n".join(lines) + "\n"


def _compare_smoke_csv_trees(a: Path, b: Path) -> str:
    diffs: list[str] = []
    for p in sorted(a.rglob("*.csv")):
        rel = p.relative_to(a)
        q = b / rel
        if not q.is_file():
            diffs.append(f"missing in run B: {rel}")
            continue
        if not filecmp.cmp(p, q, shallow=False):
            diffs.append(f"byte mismatch: {rel}")
    if not diffs:
        return "Smoke determinism: two consecutive runs produced **byte-identical** phase3 CSVs."
    return "Smoke determinism: **FAILED** — " + "; ".join(diffs[:8])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="3", help="Comma-separated phase ids (2,3,...)")
    ap.add_argument("--trace-sets", default="5g_harsh,bb")
    ap.add_argument("--hosts", default="bba")
    ap.add_argument("--arms", default="", help="Comma-separated subset of phase arms")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    ap.add_argument("--refit-quantile", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke-report", action="store_true", help="Print smoke markdown from existing CSVs")
    ap.add_argument("--smoke-determinism-check", action="store_true")
    args = ap.parse_args()

    smoke_traces = tuple(t.strip() for t in args.trace_sets.split(",") if t.strip())
    arms_filter: tuple[str, ...] | None = None
    if args.arms.strip():
        arms_filter = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    if args.smoke_report:
        eps = args.episodes or SMOKE_EPISODES_DEFAULT
        print(smoke_table(SMOKE_ARMS, smoke_traces, eps))
        return 0

    episodes = args.episodes
    if args.smoke or args.pilot:
        episodes = episodes or SMOKE_EPISODES_DEFAULT
    else:
        episodes = episodes or CPS_EPISODES

    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    trace_sets = list(smoke_traces)
    hosts = [x.strip() for x in args.hosts.split(",") if x.strip()]

    resume = args.resume
    refit = args.refit_quantile
    if args.smoke or args.smoke_determinism_check:
        phases = [3]
        if not args.trace_sets or args.trace_sets == "5g_harsh,bb":
            trace_sets = ["5g_harsh", "bb"]
            if not SMOKE_TRACE_SETS["5g_harsh"].is_dir():
                trace_sets = ["5g_v18", "bb"]
        hosts = ["bba"]
        resume = False
        refit = True

    determinism_note = ""
    fitted_q: dict[str, float] = {}
    snapshot_a: Path | None = None

    if args.smoke:
        shutil.rmtree(RESULTS_ROOT / "phase3", ignore_errors=True)

    if args.smoke_determinism_check:
        shutil.rmtree(RESULTS_ROOT / "phase3", ignore_errors=True)

    summary = run_phases(
        phases,
        trace_sets=trace_sets,
        hosts=hosts,
        split="test" if (args.smoke or args.smoke_determinism_check) else args.split,
        episodes=episodes,
        workers=args.workers,
        resume=resume,
        data_dir=args.data_dir,
        master_seed=args.master_seed,
        refit_quantile=refit,
        arms_filter=arms_filter,
    )
    print(json.dumps(summary, indent=2))

    if args.smoke:
        fitted_q = _prefit_quantile(trace_sets, force=False)
        snapshot_a = RESULTS_ROOT / "_smoke_snapshot_a"
        if snapshot_a.exists():
            shutil.rmtree(snapshot_a)
        shutil.copytree(RESULTS_ROOT / "phase3", snapshot_a)
        shutil.rmtree(RESULTS_ROOT / "phase3", ignore_errors=True)
        run_phases(
            phases,
            trace_sets=trace_sets,
            hosts=hosts,
            split="test",
            episodes=episodes,
            workers=args.workers,
            resume=False,
            data_dir=args.data_dir,
            master_seed=args.master_seed,
            refit_quantile=False,
            arms_filter=arms_filter,
        )
        determinism_note = _compare_smoke_csv_trees(snapshot_a, RESULTS_ROOT / "phase3")
        shutil.rmtree(snapshot_a, ignore_errors=True)

    if args.smoke or args.smoke_determinism_check:
        if not fitted_q and refit:
            fitted_q = _prefit_quantile(trace_sets, force=False)
        report = ROOT / "docs" / "reports" / "tube_v2_smoke_phase3.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        arms_for_table = arms_filter or SMOKE_ARMS
        report.write_text(
            smoke_table(
                arms_for_table,
                tuple(trace_sets),
                episodes,
                fitted_q=fitted_q,
                determinism_note=determinism_note,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
