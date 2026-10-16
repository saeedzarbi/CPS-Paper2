#!/usr/bin/env python3
"""Phase 2: Tube-v2 ACI ablation (val splits, gamma grid, paired bootstrap CIs)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.trace_sets_v2 import resolve_trace_dir, split_inventory
from src.control.two_timescale_conformal import (
    RHO_SWEEP_ARMS,
    ControlArmConfig,
    make_controlled_policy,
    parse_tube_v2_arm,
)
from src.evaluation.clean_abr_eval import build_env, run_episode
from src.evaluation.cluster_stats import cluster_bootstrap_mean_ci
from src.evaluation.eval_certified_shield_v18 import make_policy
from src.evaluation.eval_seeds import DEFAULT_MASTER_SEED
from src.evaluation.paired_bootstrap import (
    align_paired_metric,
    format_ci,
    paired_bootstrap_mean_ci,
    win_tie_loss,
)
from src.evaluation.trace_provenance import load_canonical_manifest, verify_or_exit

OUT_ROOT = ROOT / "results" / "tube_v2" / "phase2"
REPORT_PATH = ROOT / "docs" / "reports" / "tube_v2_phase2_pilot.md"
DONE_PATH = OUT_ROOT / "DONE.json"
RUN_LOG_DEFAULT = OUT_ROOT / "run.log"

PROVENANCE_VERIFIED = "VERIFIED"
PROVENANCE_UNVERIFIED = "UNVERIFIED"

GAMMAS = (0.002, 0.005, 0.01, 0.02)
TUBE_STATIC_ARMS = (
    "tube_v2_bonf_static",
    "tube_v2_step_static",
    "tube_v2_step_static_nocap",
)
TUBE_ACI_ARMS = (
    "tube_v2_bonf_aci",
    "tube_v2_step_aci",
    "tube_v2_step_aci_nocap",
)
BASELINE_ARMS = ("robust_mpc_logu", "mpc_fixed_quantile_native", *RHO_SWEEP_ARMS)

REF_ARMS = ("robust_mpc_logu", "mpc_fixed_quantile_native")


def _job_key(arm: str, gamma: float | None) -> str:
    if gamma is None:
        return arm
    return f"{arm}@g{gamma:.3f}"


def phase2_arm_list() -> list[tuple[str, float | None]]:
    jobs: list[tuple[str, float | None]] = [(a, None) for a in TUBE_STATIC_ARMS]
    for arm in TUBE_ACI_ARMS:
        for g in GAMMAS:
            jobs.append((arm, g))
    for arm in BASELINE_ARMS:
        jobs.append((arm, None))
    return jobs


def _csv_path(trace_set: str, host: str, job_key: str) -> Path:
    safe = job_key.replace("@", "_at_").replace(".", "p")
    return OUT_ROOT / trace_set / host / f"{safe}.csv"


def _provenance_comment(status: str) -> str:
    return f"# provenance_status={status}"


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and lines[0].startswith("#"):
        lines = lines[1:]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def _episodes_done(path: Path) -> set[int]:
    return {int(r["episode"]) for r in _read_csv_rows(path)}


def _job_complete(path: Path, episodes: int) -> bool:
    done = _episodes_done(path)
    return len(done) >= episodes and all(ep in done for ep in range(episodes))


def _append_csv_row(path: Path, row: dict, provenance: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if not path.is_file() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write(_provenance_comment(provenance) + "\n")
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run_job(
    trace_set: str,
    host: str,
    episodes: int,
    jobs: list[tuple[str, float | None]],
    *,
    resume: bool,
    provenance: str,
) -> dict[str, list[dict]]:
    trace_dir = resolve_trace_dir(trace_set, "val")
    all_rows: dict[str, list[dict]] = {}
    for arm, gamma in jobs:
        cfg = ControlArmConfig(aci_gamma=float(gamma if gamma is not None else 0.005))
        key = _job_key(arm, gamma)
        path = _csv_path(trace_set, host, key)
        if resume and _job_complete(path, episodes):
            all_rows[key] = _read_csv_rows(path)
            continue
        done = _episodes_done(path) if resume and path.is_file() else set()
        if resume and path.is_file() and not done:
            pass
        elif path.is_file() and not resume:
            path.unlink()

        env = build_env(trace_dir, arm)  # type: ignore[arg-type]
        base = make_policy(host, env, None, False)
        policy = make_controlled_policy(env, arm, base, cfg, trace_set_key=trace_set)  # type: ignore[arg-type]
        rows: list[dict] = _read_csv_rows(path) if resume else []
        for ep in range(episodes):
            if ep in done:
                continue
            if hasattr(policy, "on_episode_start"):
                policy.on_episode_start()
            row = run_episode(
                env,
                policy,
                ep,
                master_seed=DEFAULT_MASTER_SEED,
                trace_key=trace_set,
            )
            row["arm"] = arm
            row["aci_gamma"] = "" if gamma is None else f"{gamma:.4f}"
            row["job_key"] = key
            row["provenance_status"] = provenance
            _append_csv_row(path, row, provenance)
            rows.append(row)
        all_rows[key] = rows if rows else _read_csv_rows(path)
    return all_rows


def _by_trace(rows: list[dict], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        tid = str(r.get("trace_id", "0"))
        if r.get(key, "") in ("", "nan"):
            continue
        out.setdefault(tid, []).append(float(r[key]))
    return out


def _pareto_rows(rows_by_arm: dict[str, list[dict]], keys: list[str]) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for key in keys:
        if key not in rows_by_arm or not rows_by_arm[key]:
            continue
        rs = rows_by_arm[key]
        label = rs[0].get("job_key", key)
        qoe = float(np.mean([float(r["episodic_qoe"]) for r in rs]))
        reb = float(np.mean([float(r["rebuffer_total"]) for r in rs]))
        out.append((str(label), qoe, reb))
    return out


def _rho_monotone_flags(rows_by_arm: dict[str, list[dict]]) -> list[str]:
    rhos = [0.5, 0.55, 0.6, 0.65, 0.7]
    qoes: list[tuple[float, float, float]] = []
    for rho in rhos:
        arm = f"mpc_fixed_rho_{rho}"
        key = next((k for k, rs in rows_by_arm.items() if rs and rs[0].get("arm") == arm), None)
        if key is None:
            continue
        rs = rows_by_arm[key]
        mu, lo, hi = cluster_bootstrap_mean_ci(_by_trace(rs, "episodic_qoe"), n_boot=2000)
        qoes.append((rho, mu, hi - lo))
    flags: list[str] = []
    for i in range(len(qoes) - 1):
        r0, m0, w0 = qoes[i]
        r1, m1, w1 = qoes[i + 1]
        if m1 > m0 and (m1 - m0) > 0.5 * (w0 + w1):
            flags.append(f"QoE rises {r0}→{r1} ({m0:.1f}→{m1:.1f}) despite higher rho (check non-monotone)")
    return flags


def _split_inventory_lines(trace_sets: list[str]) -> list[str]:
    lines = [
        "### Trace split inventory (files per split)",
        "",
        "| trace_set | n_calib | n_val | n_test | n_test_pool | n_removed (basename filter) |",
        "|-----------|---------|-------|--------|-------------|------------------------------|",
    ]
    for ts in trace_sets:
        try:
            inv = split_inventory(ts)
        except (FileNotFoundError, KeyError) as e:
            lines.append(f"| `{ts}` | — | — | — | — | skip ({e}) |")
            continue
        lines.append(
            f"| `{ts}` | {inv.n_calib} | {inv.n_val} | {inv.n_test} | "
            f"{inv.n_test_pool} | {inv.n_test_removed_by_basename_filter} |"
        )
    lines.append("")
    return lines


def build_report(
    trace_sets: list[str],
    hosts: list[str],
    episodes: int,
    pilot: bool,
    wall_by_job: dict[str, float],
    *,
    provenance: str,
    manifest_build: dict | None,
) -> str:
    lines = [
        "# Tube-v2 Phase 2 — ACI ablation (validation)",
        "",
        f"**Provenance: {provenance}**",
        "",
        f"Mode: **{'pilot' if pilot else 'full'}**. Episodes/arm: {episodes}.",
        "CIs: trace-cluster bootstrap (QoE aggregates) and **paired episode bootstrap** vs references.",
        "",
    ]
    if manifest_build:
        lines.append(
            f"Canonical manifest build: python {manifest_build.get('python', '?')}, "
            f"numpy {manifest_build.get('numpy', '?')}, "
            f"git `{manifest_build.get('git_commit', '?')[:12]}` "
            f"({manifest_build.get('built_at_utc', '')})."
        )
        lines.append("")
    lines.extend(_split_inventory_lines(trace_sets))
    wall_path = OUT_ROOT / "pilot_wall.json"
    if not wall_by_job and wall_path.is_file():
        wall_by_job = json.loads(wall_path.read_text(encoding="utf-8"))
    if wall_by_job:
        total = sum(wall_by_job.values())
        n_sets = max(len(wall_by_job), 1)
        per_set = total / n_sets
        scale = (100 / max(episodes, 1)) if pilot else 1.0
        full_sets = len(trace_sets) * len(hosts)
        eta_h = per_set * scale * full_sets / 3600
        lines.extend(
            [
                f"Pilot wall time (this run): {total:.1f}s. Rough full-run ETA (100 ep, all jobs): **{eta_h:.1f} h** on this host.",
                "",
            ]
        )
    for trace_set in trace_sets:
        for host in hosts:
            lines.append(f"## {trace_set} · `{host}`")
            lines.append("")
            job_dir = OUT_ROOT / trace_set / host
            if not job_dir.is_dir():
                lines.append("_no results_")
                continue
            rows_by_arm: dict[str, list[dict]] = {}
            for path in sorted(job_dir.glob("*.csv")):
                rows = _read_csv_rows(path)
                if not rows:
                    continue
                key = rows[0].get("job_key", path.stem)
                rows_by_arm[key] = rows
            lines.append("### Means (cluster bootstrap 95% CI)")
            lines.append("")
            lines.append("| job | QoE | rebuf | VMAF | joint_cov | mean_α_t | clip% |")
            lines.append("|-----|-----|-------|------|-----------|----------|-------|")
            for key, rows in sorted(rows_by_arm.items()):
                if not rows:
                    continue
                q_mu, q_lo, q_hi = cluster_bootstrap_mean_ci(_by_trace(rows, "episodic_qoe"), n_boot=2000)
                r_mu, r_lo, r_hi = cluster_bootstrap_mean_ci(_by_trace(rows, "rebuffer_total"), n_boot=2000)
                v_mu, v_lo, v_hi = cluster_bootstrap_mean_ci(_by_trace(rows, "mean_vmaf"), n_boot=2000)
                j_mu, j_lo, j_hi = cluster_bootstrap_mean_ci(
                    _by_trace(rows, "joint_tube_coverage_decision"), n_boot=2000
                )
                mat = np.nanmean([float(r.get("tube_mean_alpha_t", "nan")) for r in rows])
                clip = np.nanmean([float(r.get("tube_aci_clip_frac", "nan")) for r in rows])
                lines.append(
                    f"| `{key}` | {format_ci(q_mu, q_lo, q_hi)} | {format_ci(r_mu, r_lo, r_hi)} | "
                    f"{format_ci(v_mu, v_lo, v_hi)} | {format_ci(j_mu, j_lo, j_hi, 3)} | "
                    f"{mat:.4f} | {clip:.3f} |"
                )
            lines.append("")
            for ref_name in REF_ARMS:
                ref_key = next((k for k, rs in rows_by_arm.items() if rs and rs[0].get("arm") == ref_name), None)
                if not ref_key:
                    continue
                lines.append(f"### Paired vs `{ref_name}` (episode bootstrap Δ, 95% CI; W/T/L)")
                lines.append("")
                lines.append("| job | n_pairs | ΔQoE | Δrebuf | ΔVMAF | Δjoint_cov | W/T/L QoE |")
                lines.append("|-----|---------|------|--------|-------|------------|-----------|")
                ref_rows = rows_by_arm[ref_key]
                for key, rows in sorted(rows_by_arm.items()):
                    if key == ref_key or not parse_tube_v2_arm(str(rows[0].get("arm", ""))):
                        continue
                    cells = []
                    wtl = ""
                    n_pairs = 0
                    for m, _hib in (
                        ("episodic_qoe", True),
                        ("rebuffer_total", False),
                        ("mean_vmaf", True),
                        ("joint_tube_coverage_decision", True),
                    ):
                        try:
                            a, b, n_pairs = align_paired_metric(rows, ref_rows, m)
                        except ValueError:
                            cells.append("—")
                            continue
                        if n_pairs == 0:
                            cells.append("—")
                            continue
                        d, lo, hi = paired_bootstrap_mean_ci(a, b)
                        cells.append(format_ci(d, lo, hi))
                        if m == "episodic_qoe":
                            w, t, l = win_tie_loss(a, b, higher_is_better=True)
                            wtl = f"{w}/{t}/{l}"
                    lines.append(
                        f"| `{key}` | {n_pairs} | " + " | ".join(cells) + f" | {wtl} |"
                    )
                lines.append("")
            lines.append("### QoE–rebuffer (tube + baselines)")
            lines.append("")
            lines.append("| arm | QoE | rebuf |")
            lines.append("|-----|-----|-------|")
            for arm, qoe, reb in sorted(_pareto_rows(rows_by_arm, list(rows_by_arm.keys())), key=lambda x: -x[1]):
                lines.append(f"| `{arm}` | {qoe:.2f} | {reb:.3f} |")
            flags = _rho_monotone_flags(rows_by_arm)
            if flags:
                lines.append("")
                lines.append("**Rho monotonicity flags:** " + "; ".join(flags))
            lines.append("")
    return "\n".join(lines) + "\n"


def _write_done_marker(
    *,
    provenance: str,
    episodes: int,
    pilot: bool,
    trace_sets: list[str],
    hosts: list[str],
    wall_by_job: dict[str, float],
) -> None:
    payload = {
        "finished_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "provenance_status": provenance,
        "pilot": pilot,
        "episodes_per_arm": episodes,
        "trace_sets": trace_sets,
        "hosts": hosts,
        "jobs_per_trace_set": len(phase2_arm_list()),
        "wall_seconds_by_key": wall_by_job,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    DONE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace-sets",
        nargs="+",
        default=["5g_markov", "5g_harsh", "bb"],
    )
    ap.add_argument("--hosts", nargs="+", default=["bba"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--pilot", action="store_true", help="Short run for runtime ETA (5 episodes)")
    ap.add_argument("--workers", type=int, default=1, help="Reserved; Phase 2 runs sequentially per job")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    episodes = 5 if args.pilot else args.episodes
    provenance = PROVENANCE_UNVERIFIED if args.skip_verify else PROVENANCE_VERIFIED
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_build: dict | None = None
    try:
        manifest_build = load_canonical_manifest().get("build")
    except FileNotFoundError:
        pass

    if args.skip_verify and not args.report_only:
        print(
            f"WARNING: --skip-verify → all outputs tagged {PROVENANCE_UNVERIFIED}; not for reported runs.",
            file=sys.stderr,
        )
    if not args.skip_verify and not args.report_only:
        verify_or_exit()

    jobs = phase2_arm_list()
    wall_by_job: dict[str, float] = {}
    if not args.report_only:
        if DONE_PATH.is_file() and not args.resume:
            DONE_PATH.unlink()
        meta = {
            "episodes": episodes,
            "pilot": args.pilot,
            "gammas": list(GAMMAS),
            "provenance_status": provenance,
            "manifest_build": manifest_build,
            "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        (OUT_ROOT / "phase2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for trace_set in args.trace_sets:
            for host in args.hosts:
                try:
                    t0 = time.time()
                    run_job(
                        trace_set,
                        host,
                        episodes,
                        jobs,
                        resume=args.resume,
                        provenance=provenance,
                    )
                    wall_by_job[f"{trace_set}/{host}"] = time.time() - t0
                except (FileNotFoundError, KeyError) as e:
                    print("SKIP", trace_set, e)
        (OUT_ROOT / "pilot_wall.json").write_text(json.dumps(wall_by_job, indent=2), encoding="utf-8")
        _write_done_marker(
            provenance=provenance,
            episodes=episodes,
            pilot=args.pilot,
            trace_sets=args.trace_sets,
            hosts=args.hosts,
            wall_by_job=wall_by_job,
        )

    if not wall_by_job and (OUT_ROOT / "pilot_wall.json").is_file():
        wall_by_job = json.loads((OUT_ROOT / "pilot_wall.json").read_text(encoding="utf-8"))

    report = build_report(
        args.trace_sets,
        args.hosts,
        episodes,
        args.pilot,
        wall_by_job,
        provenance=provenance,
        manifest_build=manifest_build,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    if not args.report_only:
        print(f"Wrote {DONE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
