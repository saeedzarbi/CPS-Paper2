"""Sensitivity ablation over the perceptual budget (epsilon) and conformal
miscoverage (alpha) for the Certified Perceptual Shield.

Runs the real greedy-host shield evaluation (no learned components, no server) on
the synthetic 5G test traces for:

  * epsilon in {0.5, 1.0, 2.0, 4.0} at fixed alpha = 0.10
  * alpha   in {0.05, 0.10, 0.20} at fixed epsilon = 1.0

and emits LaTeX tables, macros, and a two-panel figure showing that (i) banking
magnitude and its (small) perceptual cost scale smoothly with epsilon, and
(ii) empirical conformal coverage tracks the finite-sample target 1 - alpha,
so the headline results are not overfit to a single (epsilon, alpha) choice.

Path A (SCI-02): the default cell (eps=1.0, alpha=0.10) is taken from the
headline evaluation ``results/v18_certified/greedy_5g`` so Table 5 and the
sensitivity default row are the same run. Other sweep cells are evaluated
separately under the same seeding protocol.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
NEW_ROOT = HERE.parents[2]
if str(NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_ROOT))
from configs.videos import CPS_EPISODES

EVAL = HERE.parent / "eval_certified_shield_v18.py"
TRACE_DIR = "data/standardized/test_traces_5g_v18"
OUT = NEW_ROOT / "results" / "ablation_eps_alpha"
HEADLINE = NEW_ROOT / "results" / "v18_certified" / "greedy_5g"
PAPER = NEW_ROOT / "src" / "paper"
TABLES = [PAPER / "overleaf_upload" / "tables", PAPER / "tables"]
FIGURES = [PAPER / "overleaf_upload" / "figures", PAPER / "figures"]

EPISODES = CPS_EPISODES
EPS_LIST = [0.5, 1.0, 2.0, 4.0]
ALPHA_LIST = [0.05, 0.10, 0.20]
EPS_FIXED_ALPHA = 0.10
ALPHA_FIXED_EPS = 1.0
PYEXE = sys.executable


def run_eval(eps: float, alpha: float, tag: str) -> Path:
    out = OUT / tag
    cmd = [
        PYEXE, str(EVAL),
        "--policy", "greedy",
        "--trace-dir", TRACE_DIR,
        "--episodes", str(EPISODES),
        "--epsilon", str(eps),
        "--alpha", str(alpha),
        "--arms", "raw,safety,certified",
        "--out", str(out),
    ]
    print(f"[run] eps={eps} alpha={alpha} -> {out}")
    subprocess.run(cmd, cwd=str(NEW_ROOT), check=True)
    return out / "summary.json"


def sync_headline_default() -> Path:
    """Copy headline greedy_5g artifacts into ablation eps_1.0 (default cell)."""
    if not (HEADLINE / "summary.json").exists():
        raise FileNotFoundError(f"missing headline summary: {HEADLINE}")
    dest = OUT / f"eps_{ALPHA_FIXED_EPS}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "episodes.csv"):
        src = HEADLINE / name
        if src.exists():
            shutil.copy2(src, dest / name)
            print(f"[reuse] {src} -> {dest / name}")
    return dest / "summary.json"


def load(summary: Path) -> dict:
    d = json.loads(summary.read_text())
    cs = d["comparisons"]["certified_vs_safety"]
    cert = d["arms"]["certified"]

    def scal(x):
        return x["mean"] if isinstance(x, dict) else x

    return {
        "bw_cs": cs["bandwidth_reduction_pct"],
        "reb_cs": cs["rebuffer_change_pct"],
        "vmaf_cs": cs["vmaf_mean_diff"],
        "coverage": scal(cert.get("conformal_coverage")),
        "interv": 100.0 * scal(cert.get("interv_rate")),
        "bitrate": scal(cert.get("bitrate_mean_kbps")),
        "vmaf": scal(cert.get("vmaf_mean")),
    }


def collect_rows(*, run_missing: bool) -> tuple[list[dict], list[dict]]:
    """Build eps/alpha rows; default cell always from headline greedy_5g."""
    OUT.mkdir(parents=True, exist_ok=True)
    default_summary = sync_headline_default()

    eps_rows: list[dict] = []
    for eps in EPS_LIST:
        if abs(eps - ALPHA_FIXED_EPS) < 1e-9:
            s = default_summary
        else:
            s = OUT / f"eps_{eps}" / "summary.json"
            if not s.exists():
                if not run_missing:
                    raise FileNotFoundError(s)
                s = run_eval(eps, EPS_FIXED_ALPHA, f"eps_{eps}")
        r = load(s)
        r["epsilon"] = eps
        eps_rows.append(r)

    alpha_rows: list[dict] = []
    for alpha in ALPHA_LIST:
        if abs(alpha - EPS_FIXED_ALPHA) < 1e-9:
            s = default_summary
        else:
            s = OUT / f"alpha_{alpha}" / "summary.json"
            if not s.exists():
                if not run_missing:
                    raise FileNotFoundError(s)
                s = run_eval(ALPHA_FIXED_EPS, alpha, f"alpha_{alpha}")
        r = load(s)
        r["alpha"] = alpha
        alpha_rows.append(r)

    return eps_rows, alpha_rows


def emit(eps_rows: list[dict], alpha_rows: list[dict]) -> None:
    write_eps_table(eps_rows)
    write_alpha_table(alpha_rows)
    write_macros(eps_rows, alpha_rows)
    write_figure(eps_rows, alpha_rows)
    print("\n== epsilon sweep ==")
    for r in eps_rows:
        print(r)
    print("\n== alpha sweep ==")
    for r in alpha_rows:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Reuse existing non-default summaries + headline greedy_5g; do not run evals.",
    )
    args = ap.parse_args()
    eps_rows, alpha_rows = collect_rows(run_missing=not args.rebuild_only)
    emit(eps_rows, alpha_rows)


def write_eps_table(rows):
    lines = [
        r"% Auto-generated by ablation_eps_alpha.py",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$\varepsilon$ (pts) & Bitrate saved (\%) & Rebuffer $\Delta$ (\%) & "
        r"VMAF $\Delta$ (pts) & Coverage \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['epsilon']:.1f} & {-r['bw_cs']:.1f} & {r['reb_cs']:.1f} & "
            f"{r['vmaf_cs']:+.2f} & {r['coverage']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines) + "\n"
    for d in TABLES:
        d.mkdir(parents=True, exist_ok=True)
        (d / "table_ablation_eps.tex").write_text(body, encoding="utf-8")


def write_alpha_table(rows):
    lines = [
        r"% Auto-generated by ablation_eps_alpha.py",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$\alpha$ & Target $1{-}\alpha$ & Coverage & Interv. (\%) & Bitrate saved (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['alpha']:.2f} & {1 - r['alpha']:.2f} & {r['coverage']:.3f} & "
            f"{r['interv']:.1f} & {-r['bw_cs']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines) + "\n"
    for d in TABLES:
        (d / "table_ablation_alpha.tex").write_text(body, encoding="utf-8")


def write_macros(eps_rows, alpha_rows):
    def cmd(n, v):
        return f"\\newcommand{{\\{n}}}{{{v}}}"
    e_lo, e_hi = eps_rows[0], eps_rows[-1]
    # coverage across all runs
    covs = [r["coverage"] for r in eps_rows] + [r["coverage"] for r in alpha_rows]
    m = [
        r"% Auto-generated eps/alpha ablation macros",
        cmd("AblEpsLo", f"{e_lo['epsilon']:.1f}"),
        cmd("AblEpsHi", f"{e_hi['epsilon']:.1f}"),
        cmd("AblBwLo", f"{-e_lo['bw_cs']:.1f}"),
        cmd("AblBwHi", f"{-e_hi['bw_cs']:.1f}"),
        cmd("AblVmMaxAbs", f"{max(abs(r['vmaf_cs']) for r in eps_rows):.2f}"),
        cmd("AblCovMin", f"{min(covs):.3f}"),
        cmd("AblCovMax", f"{max(covs):.3f}"),
    ]
    body = "\n".join(m) + "\n"
    for d in TABLES:
        (d / "macros_ablation.tex").write_text(body, encoding="utf-8")


def write_figure(eps_rows, alpha_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")

    eps = [r["epsilon"] for r in eps_rows]
    bw = [-r["bw_cs"] for r in eps_rows]
    ax1.plot(eps, bw, "o-", color="#4C72B0")
    ax1.set_xlabel(r"perceptual budget $\varepsilon$ (VMAF pts)")
    ax1.set_ylabel("bitrate saved (%)")
    ax1.grid(axis="y", alpha=0.3)

    al = [r["alpha"] for r in alpha_rows]
    cov = [r["coverage"] for r in alpha_rows]
    tgt = [1 - a for a in al]
    x = np.arange(len(al))
    w = 0.38
    ax2.bar(x - w / 2, tgt, w, label=r"target $1{-}\alpha$", color="#BBBBBB")
    ax2.bar(x + w / 2, cov, w, label="empirical", color="#4C72B0")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{a:.2f}" for a in al])
    ax2.set_xlabel(r"miscoverage $\alpha$ ($\varepsilon{=}1.0$)")
    ax2.set_ylabel("conformal coverage")
    ax2.set_ylim(0.70, 1.02)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    for d in FIGURES:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_cps_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
