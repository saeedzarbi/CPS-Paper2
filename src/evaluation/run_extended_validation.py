#!/usr/bin/env python3
"""
Extended validation for Paper 2 (OOD traces, content strata, ablations, extended stats).

Headline benchmark remains ``results/clean_v3/``. This script writes to
``results/clean_v3_extended/`` and ``docs/reports/clean_v3_extended_validation.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.paths import get_paths
from configs.videos import CPS_EPISODES, EVAL_VIDEOS
from src.control.two_timescale_conformal import ABLATION_ARMS, HEADLINE_ARMS, ControlArmConfig
from src.evaluation.clean_abr_eval import (
    EXTENDED_TRACE_SETS,
    HEADLINE_TRACE_SETS,
    build_env,
    default_eval_workers,
    episode_cvar95,
    load_episodes_csv,
    load_summary,
    paired_bootstrap_ci,
    resolve_video_for_episode,
    run_arm,
    write_job_out,
)
from src.utils.mattermost_notify import notify as mattermost_notify

P = get_paths()
RESULTS_EXT = ROOT / "results" / "clean_v3_extended"
RESULTS_V3 = ROOT / "results" / "clean_v3"
REPORT = ROOT / "docs" / "reports" / "clean_v3_extended_validation.md"

OOD_ARMS = tuple(HEADLINE_ARMS)
OOD_POLICIES = ("mpc", "greedy", "bba", "bola")
ABLATION_POLICIES = ("bba", "bola")
# H=8 omitted: full tube MPC enumerates 6^8 sequences per chunk (infeasible at 204 eps).
HORIZONS = (1, 2, 3, 5)
ALPHAS = (0.05, 0.10, 0.15, 0.20)
REPORT_PROTOCOL = "clean_v3_extended_v3"
MM_USER = "clean_v3_extended"

HEADLINE_ARM_LABELS = {
    "raw": "raw",
    "static_conformal": "static",
    "tube_conformal": "tube",
}
ABLATION_COL_LABELS = {
    "raw": "raw",
    "heuristic_85": "heur. 0.85",
    "static_conformal": "static H=1",
    "symmetric_tube": "sym. tube",
    "fixed_alpha_tube": "fixed-α tube",
    "tube_conformal": "tube (proposed)",
}


def _mm(text: str) -> None:
    mattermost_notify(text, username=MM_USER)


def _notify_job_done(label: str, out_dir: Path) -> None:
    summary = load_summary(out_dir / "summary.json")
    if not summary:
        _mm(f":white_check_mark: `{label}` done")
        return
    arms = summary.get("arms", {})
    tube = arms.get("tube_conformal", {}).get("rebuffer_total", {}).get("mean")
    static = arms.get("static_conformal", {}).get("rebuffer_total", {}).get("mean")
    if tube is not None and static is not None:
        _mm(f":white_check_mark: `{label}` reb tube={tube:.2f}s static={static:.2f}s")
    elif tube is not None:
        _mm(f":white_check_mark: `{label}` tube reb={tube:.2f}s")
    else:
        _mm(f":white_check_mark: `{label}` done")


def _ensure_subsets() -> None:
    script = ROOT / "data" / "prepare_eval_trace_subsets.py"
    if script.is_file():
        subprocess.check_call([sys.executable, str(script)], cwd=str(ROOT))


def _video_vbr_cv(slug: str) -> float:
    env = build_env(HEADLINE_TRACE_SETS["5g_markov"], "raw")
    prof = env.video_assets[slug].get("vbr_profile", [])
    if prof is None or len(prof) == 0:
        return float("nan")
    sizes = [float(x) for x in list(prof)[:48] if float(x) > 0]
    if not sizes:
        return float("nan")
    m = float(np.mean(sizes))
    return float(np.std(sizes) / m) if m > 0 else 0.0


def _content_strata() -> dict[str, str]:
    """Map video slug -> 'low_medium' | 'high_vbr_motion'."""
    siti_path = P["content_features"] / "siti_summary.csv"
    complexity: dict[str, str] = {}
    if siti_path.is_file():
        with open(siti_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                complexity[row["video"]] = row.get("complexity", "")

    cv_by_video = {v: _video_vbr_cv(v) for v in EVAL_VIDEOS}
    valid_cv = [c for c in cv_by_video.values() if c == c]
    thresh = float(np.percentile(valid_cv, 66)) if valid_cv else 0.0

    strata: dict[str, str] = {}
    for v in EVAL_VIDEOS:
        comp = complexity.get(v, complexity.get(f"{v}_short", ""))
        high_motion = any(
            x in comp
            for x in ("High (Complex + Fast Motion)", "High-TI (Fast Motion)", "Fast Motion")
        )
        high_cv = cv_by_video.get(v, 0.0) >= thresh
        strata[v] = "high_vbr_motion" if (high_motion or high_cv) else "low_medium"
    return strata


def _job_done(out_dir: Path, protocol: str | None = None) -> bool:
    p = out_dir / "summary.json"
    if not p.is_file() or p.stat().st_size <= 20:
        return False
    if protocol is None:
        return True
    summary = load_summary(p)
    return bool(summary and summary.get("meta", {}).get("protocol") == protocol)


def _metric_mean(summary: dict, arm: str, key: str) -> float:
    try:
        return float(summary["arms"][arm][key]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def _best_arm(values: dict[str, float], *, lower_better: bool) -> str | None:
    valid = {a: v for a, v in values.items() if v == v}
    if not valid:
        return None
    if lower_better:
        return min(valid, key=valid.get)
    return max(valid, key=valid.get)


def _fmt_val(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v == v else "—"


def _fmt_arm_cells(
    arms: tuple[str, ...],
    values: dict[str, float],
    *,
    lower_better: bool,
    decimals: int = 2,
) -> list[str]:
    best = _best_arm(values, lower_better=lower_better)
    out = []
    for arm in arms:
        v = values.get(arm, float("nan"))
        txt = _fmt_val(v, decimals)
        if arm == best and v == v:
            txt = f"**{txt}**"
        out.append(txt)
    return out


def run_ood_suite(episodes: int, workers: int) -> None:
    _ensure_subsets()
    for trace_key, trace_dir in EXTENDED_TRACE_SETS.items():
        if not trace_dir.is_dir() or not any(trace_dir.glob("*.json")):
            print(f"[skip] missing traces: {trace_key}")
            continue
        for policy in OOD_POLICIES:
            out = RESULTS_EXT / "ood" / trace_key / policy
            need_proto = trace_key == "5g_v18"
            if _job_done(out, REPORT_PROTOCOL if need_proto else None):
                print(f"[skip] done: ood/{trace_key}/{policy}")
                continue
            all_rows: dict[str, list] = {}
            for arm in OOD_ARMS:
                print(f"[ood] {trace_key} {policy} {arm} n={episodes}")
                all_rows[arm] = run_arm(policy, trace_dir, arm, episodes, workers=workers)
            write_job_out(
                out,
                all_rows,
                policy,
                trace_key,
                protocol=REPORT_PROTOCOL if need_proto else None,
            )
            _notify_job_done(f"ood/{trace_key}/{policy}", out)


def run_ablation(episodes: int, workers: int) -> None:
    trace_dir = HEADLINE_TRACE_SETS["5g_markov"]
    for policy in ABLATION_POLICIES:
        out = RESULTS_EXT / "ablation" / policy
        if _job_done(out, REPORT_PROTOCOL):
            print(f"[skip] done: ablation/{policy}")
            continue
        all_rows: dict[str, list] = {}
        for arm in ABLATION_ARMS:
            print(f"[ablation] {policy} {arm} n={episodes}")
            all_rows[arm] = run_arm(policy, trace_dir, arm, episodes, workers=workers)
        write_job_out(out, all_rows, policy, "5g_markov", protocol=REPORT_PROTOCOL)
        _notify_job_done(f"ablation/{policy}", out)


def run_sensitivity(episodes: int, workers: int) -> None:
    trace_dir = HEADLINE_TRACE_SETS["5g_markov"]
    for policy in ABLATION_POLICIES:
        for h in HORIZONS:
            out = RESULTS_EXT / "sensitivity" / "horizon" / f"{policy}_H{h}"
            if _job_done(out, REPORT_PROTOCOL):
                print(f"[skip] done: sensitivity/horizon/{policy}_H{h}")
                continue
            if h == 1:
                cfg = ControlArmConfig(horizon=1, alpha=0.10)
                rows = run_arm(
                    policy, trace_dir, "static_conformal", episodes, cfg, workers=workers
                )
                meta = {"horizon": h, "alpha": 0.10, "arm": "static_conformal"}
            else:
                cfg = ControlArmConfig(horizon=h, alpha=0.10, horizon_ref=5)
                rows = run_arm(
                    policy, trace_dir, "tube_conformal", episodes, cfg, workers=workers
                )
                meta = {"horizon": h, "alpha": 0.10, "arm": "tube_conformal"}
            write_job_out(
                out,
                {"tube_conformal": rows},
                policy,
                "5g_markov",
                meta=meta,
                protocol=REPORT_PROTOCOL,
            )
            _notify_job_done(f"sensitivity/H{h}/{policy}", out)
        for alpha in ALPHAS:
            cfg = ControlArmConfig(horizon=5, alpha=alpha)
            out = RESULTS_EXT / "sensitivity" / "alpha" / f"{policy}_a{alpha:.2f}"
            if _job_done(out):
                print(f"[skip] done: sensitivity/alpha/{policy}_a{alpha:.2f}")
                continue
            rows = run_arm(
                policy, trace_dir, "tube_conformal", episodes, cfg, workers=workers
            )
            write_job_out(
                out,
                {"tube_conformal": rows},
                policy,
                "5g_markov",
                meta={"horizon": 5, "alpha": alpha},
                protocol=REPORT_PROTOCOL,
            )
            _notify_job_done(f"sensitivity/a={alpha:.2f}/{policy}", out)


def _load_v3_episode_pairs(job: str) -> list[tuple[int, float, float, float, float]]:
    """episode, tube reb, static reb, tube smooth, static smooth."""
    path = RESULTS_V3 / job / "episodes.csv"
    rows = load_episodes_csv(path)
    if not rows:
        return []
    by_ep_arm: dict[tuple[int, str], dict] = {}
    for r in rows:
        by_ep_arm[(int(r["episode"]), r["arm"])] = r
    out = []
    for ep in range(CPS_EPISODES):
        ts = by_ep_arm.get((ep, "tube_conformal"))
        ss = by_ep_arm.get((ep, "static_conformal"))
        if not ts or not ss:
            continue
        out.append(
            (
                ep,
                float(ts["rebuffer_total"]),
                float(ss["rebuffer_total"]),
                float(ts["smooth_penalty_mean"]),
                float(ss["smooth_penalty_mean"]),
            )
        )
    return out


def content_stratification_section() -> list[str]:
    for ep in range(CPS_EPISODES):
        resolve_video_for_episode(ep)
    strata = _content_strata()
    lines = [
        "## Task 3 — Content complexity (VBR / SI-TI strata)",
        "",
        "Videos grouped by `siti_summary.csv` motion/complexity labels and VBR chunk-size CV (top tertile → high).",
        "",
        "| Stratum | Videos | Δ rebuffer (tube−static) s/ep | Δ smoothness (tube−static) | n episodes |",
        "|---|---|---|---|---|",
    ]
    for stratum in ("low_medium", "high_vbr_motion"):
        vids = {v for v, s in strata.items() if s == stratum}
        reb_t, reb_s, sm_t, sm_s = [], [], [], []
        for job in [f"{p}_{t}" for p in OOD_POLICIES for t in ("5g_markov", "bb")]:
            for ep, rt, rs, st, ss in _load_v3_episode_pairs(job):
                vid = resolve_video_for_episode(ep)
                if vid not in vids:
                    continue
                reb_t.append(rt)
                reb_s.append(rs)
                sm_t.append(st)
                sm_s.append(ss)
        if reb_t:
            mr, lo, hi = paired_bootstrap_ci(reb_t, reb_s, n_boot=10_000)
            ds, _, _ = paired_bootstrap_ci(sm_t, sm_s, n_boot=10_000)
            ms = ds
            vid_list = ", ".join(sorted(vids))
            lines.append(
                f"| `{stratum}` | {vid_list} | {mr:.3f} [{lo:.3f}, {hi:.3f}] | {ms:.4f} | {len(reb_t)} |"
            )
        else:
            lines.append(f"| `{stratum}` | — | — | — | 0 |")
    lines.append("")
    return lines


def _trace_dir_for_v3_job(job: str) -> Path:
    return HEADLINE_TRACE_SETS["bb"] if job.endswith("_bb") else HEADLINE_TRACE_SETS["5g_markov"]


def extended_stats_section(latency_episodes: int, workers: int) -> list[str]:
    lines = [
        "## Task 4 — Extended statistics (`results/clean_v3/`)",
        "",
        "Paired bootstrap 10k resamples for tube − static; CVaR₉₅ on episode rebuffer; zero-stall rate; "
        "decision latency (`perf_counter_ns`, mean μs per chunk).",
        "",
        "| Job | Δ rebuffer [95% CI] | Δ smoothness [95% CI] | CVaR₉₅ tube | CVaR₉₅ static | "
        "Zero-stall % tube | Latency μs (tube) [static] |",
        "|---|---|---|---|---|---|---|",
    ]
    jobs = [f"{p}_{t}" for p in OOD_POLICIES for t in ("5g_markov", "bb")]
    lat_cache: dict[tuple[str, str], tuple[float, float]] = {}

    for job in jobs:
        pairs = _load_v3_episode_pairs(job)
        if not pairs:
            continue
        reb_t = [p[1] for p in pairs]
        reb_s = [p[2] for p in pairs]
        sm_t = [p[3] for p in pairs]
        sm_s = [p[4] for p in pairs]
        dr, lo, hi = paired_bootstrap_ci(reb_t, reb_s, n_boot=10_000)
        ds, slo, shi = paired_bootstrap_ci(sm_t, sm_s, n_boot=10_000)
        cvar_t = episode_cvar95(reb_t)
        cvar_s = episode_cvar95(reb_s)
        zst = 100.0 * float(np.mean([r <= 1e-6 for r in reb_t]))

        if job.endswith("_bb"):
            pol, trace_key = job[: -len("_bb")], "bb"
        elif job.endswith("_5g_markov"):
            pol, trace_key = job[: -len("_5g_markov")], "5g_markov"
        else:
            pol, trace_key = job, "5g_markov"
        lat_key = (pol, trace_key)
        if lat_key not in lat_cache and latency_episodes > 0:
            cfg = ControlArmConfig()
            n_eps = min(latency_episodes, CPS_EPISODES)
            trace_dir = _trace_dir_for_v3_job(job)
            tube_rows = run_arm(
                pol,
                trace_dir,
                "tube_conformal",
                n_eps,
                cfg,
                time_decisions=True,
                workers=workers,
            )
            static_rows = run_arm(
                pol,
                trace_dir,
                "static_conformal",
                n_eps,
                cfg,
                time_decisions=True,
                workers=workers,
            )
            lat_t = float(
                np.mean([r.get("decision_latency_us_mean", float("nan")) for r in tube_rows])
            )
            lat_s = float(
                np.mean([r.get("decision_latency_us_mean", float("nan")) for r in static_rows])
            )
            lat_cache[lat_key] = (lat_t, lat_s)
        lat_t, lat_s = lat_cache.get(lat_key, (float("nan"), float("nan")))
        if lat_t == lat_t and lat_s == lat_s:
            lat_cell = f"{lat_t:.1f} [{lat_s:.1f}]"
        elif lat_t == lat_t:
            lat_cell = f"{lat_t:.1f} [—]"
        else:
            lat_cell = "—"
        lines.append(
            f"| `{job}` | {dr:.3f} [{lo:.3f}, {hi:.3f}] | {ds:.4f} [{slo:.4f}, {shi:.4f}] | "
            f"{cvar_t:.2f} | {cvar_s:.2f} | {zst:.1f}% | {lat_cell} |"
        )
    lines.append("")
    return lines


def _ood_metric_table(metric_key: str, title: str, *, lower_better: bool, decimals: int = 2) -> list[str]:
    lines = [f"#### {title}", ""]
    hdr = "| Trace | Policy | " + " | ".join(HEADLINE_ARM_LABELS[a] for a in OOD_ARMS) + " | p(tube vs static) |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(OOD_ARMS) + 3))
    for trace_key in EXTENDED_TRACE_SETS:
        for policy in OOD_POLICIES:
            s = load_summary(RESULTS_EXT / "ood" / trace_key / policy / "summary.json")
            if not s:
                continue
            vals = {a: _metric_mean(s, a, metric_key) for a in OOD_ARMS}
            cells = _fmt_arm_cells(OOD_ARMS, vals, lower_better=lower_better, decimals=decimals)
            comp = s.get("comparisons", {}).get("tube_conformal_vs_static_conformal", {})
            p = comp.get(metric_key, {}).get("wilcoxon_p", float("nan"))
            ps = f"{p:.2e}" if p == p else "—"
            lines.append(f"| `{trace_key}` | {policy} | " + " | ".join(cells) + f" | {ps} |")
    lines.append("")
    return lines


def ood_section() -> list[str]:
    lines = [
        "## Task 2 — OOD & mobility generalization",
        "",
        "Headline arms on extended trace sets. **Bold** = best in row (rebuffer & smoothness: minimum; QoE: maximum).",
        "",
    ]
    lines += _ood_metric_table("rebuffer_total", "Mean rebuffer (s/ep)", lower_better=True)
    lines += _ood_metric_table("smooth_penalty_mean", "Mean smoothness (|Δlog u|)", lower_better=True, decimals=2)
    lines += _ood_metric_table("episodic_qoe", "Mean episodic QoE", lower_better=False)
    return lines


def _ablation_metric_table(metric_key: str, title: str, *, lower_better: bool, decimals: int = 2) -> list[str]:
    labels = [ABLATION_COL_LABELS[a] for a in ABLATION_ARMS]
    lines = [f"#### {title}", "", "| Policy | " + " | ".join(labels) + " |", "|---|" + "---|" * len(ABLATION_ARMS)]
    for policy in ABLATION_POLICIES:
        s = load_summary(RESULTS_EXT / "ablation" / policy / "summary.json")
        if not s:
            continue
        vals = {a: _metric_mean(s, a, metric_key) for a in ABLATION_ARMS}
        cells = _fmt_arm_cells(ABLATION_ARMS, vals, lower_better=lower_better, decimals=decimals)
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def ablation_section() -> list[str]:
    lines = [
        "## Task 4 — Component ablation (`test_traces_5g_markov`, bba & bola)",
        "",
        "`sym. tube`: symmetric ±1-tier rate limits (no static emergency cap). "
        "`fixed-α tube`: fixed α₀ bounds without online ACI updates.",
        "",
    ]
    lines += _ablation_metric_table("rebuffer_total", "Mean rebuffer (s/ep)", lower_better=True)
    lines += _ablation_metric_table("smooth_penalty_mean", "Mean smoothness (|Δlog u|)", lower_better=True)
    lines += _ablation_metric_table("episodic_qoe", "Mean episodic QoE", lower_better=False)
    return lines


def _horizon_metric(policy: str, h: int, metric_key: str) -> float:
    s = load_summary(RESULTS_EXT / "sensitivity" / "horizon" / f"{policy}_H{h}" / "summary.json")
    if not s:
        return float("nan")
    return _metric_mean(s, "tube_conformal", metric_key)


def sensitivity_section() -> list[str]:
    lines = [
        "### Horizon sweep (α=0.10)",
        "",
        "H=1: exact 1-step `static_conformal`. H∈{2,3,5}: `tube_conformal` with reserve ramping to full H=5.",
        "",
    ]
    for metric_key, title, lower_better, decimals in (
        ("rebuffer_total", "Mean rebuffer (s/ep)", True, 2),
        ("smooth_penalty_mean", "Mean smoothness (|Δlog u|)", True, 2),
        ("episodic_qoe", "Mean episodic QoE", False, 2),
    ):
        lines.append(f"#### {title}")
        lines.append("")
        hdr = "| Policy | " + " | ".join(f"H={h}" for h in HORIZONS) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(HORIZONS) + 1))
        for policy in ABLATION_POLICIES:
            vals = {str(h): _horizon_metric(policy, h, metric_key) for h in HORIZONS}
            cells = []
            best = _best_arm(vals, lower_better=lower_better)
            for h in HORIZONS:
                v = vals[str(h)]
                txt = _fmt_val(v, decimals)
                if str(h) == best and v == v:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append(f"| {policy} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["### α sweep (tube, H=5)", "", "| Policy | α=0.05 | α=0.10 | α=0.15 | α=0.20 |", "|---|---|---|---|---|"]
    for policy in ABLATION_POLICIES:
        cells = []
        for alpha in ALPHAS:
            s = load_summary(RESULTS_EXT / "sensitivity" / "alpha" / f"{policy}_a{alpha:.2f}" / "summary.json")
            v = (
                s.get("arms", {}).get("tube_conformal", {}).get("rebuffer_total", {}).get("mean", float("nan"))
                if s
                else float("nan")
            )
            cells.append(f"{v:.2f}" if v == v else "—")
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_report(latency_episodes: int, workers: int) -> None:
    lines = [
        "# Clean ABR — extended validation (`clean_v3_extended`)",
        "",
        "Headline benchmark unchanged: [`clean_v3_full_metrics.md`](clean_v3_full_metrics.md).",
        f"Extended artifacts: `{RESULTS_EXT.relative_to(ROOT).as_posix()}/`.",
        "",
    ]
    lines += ood_section()
    lines += content_stratification_section()
    lines += ablation_section()
    lines += sensitivity_section()
    lines.append("### Extended statistics on headline `clean_v3`")
    lines.append("")
    lines += extended_stats_section(latency_episodes, workers)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=CPS_EPISODES)
    ap.add_argument(
        "--latency-episodes",
        type=int,
        default=17,
        help="Episodes per policy×trace for decision latency profiling (0 = skip)",
    )
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel episode workers (default: env CLEAN_V3_EXT_* caps; see server_runs.md)",
    )
    args = ap.parse_args()
    workers = args.workers if args.workers is not None else default_eval_workers()
    (RESULTS_EXT / "logs").mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        _mm(
            f":rocket: **clean_v3_extended** start | episodes={args.episodes} "
            f"workers={workers} latency_eps={args.latency_episodes}"
        )
        if not args.skip_ood:
            run_ood_suite(args.episodes, workers)
        if not args.skip_ablation:
            run_ablation(args.episodes, workers)
        if not args.skip_sweeps:
            run_sensitivity(args.episodes, workers)

    write_report(args.latency_episodes, workers)
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "paper2" / "generate_clean_v3_report.py")],
        cwd=str(ROOT),
    )
    if not args.report_only:
        n = len(list(RESULTS_EXT.rglob("summary.json")))
        _mm(f":checkered_flag: **clean_v3_extended** finished | summaries={n} report=`{REPORT.name}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
