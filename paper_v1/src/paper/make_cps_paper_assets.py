"""Generate CPS paper tables, macros, and hero figure. Run from `new/`:

    python src/paper/make_cps_paper_assets.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "v18_certified"
PAPER = Path(__file__).resolve().parent
OUT_DIRS = [PAPER / "tables", PAPER / "overleaf_upload" / "tables"]
FIG_DIRS = [PAPER / "figures", PAPER / "overleaf_upload" / "figures"]


def load(name):
    p = RES / name / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def arm(s, a, k):
    v = s["arms"][a][k]
    return v["mean"] if isinstance(v, dict) else v


def cmp_(s, name):
    return s["comparisons"][name]


def pct(x):
    return f"{x:+.1f}"


def num(x, d=2):
    return f"{x:.{d}f}"


def pval(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    if p <= 0:
        return "<10^{-300}"
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(math.floor(math.log10(p)))
    m = p / (10 ** e)
    return f"{m:.1f}\\!\\times\\!10^{{{e}}}"


def wilcoxon_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        rng = np.random.default_rng(0)
        obs = abs(d.mean())
        perm = np.abs((rng.choice([-1.0, 1.0], size=(20000, d.size)) * d).mean(1))
        return float(max((perm >= obs - 1e-12).mean(), 1 / 20000))


def certified_mean(summary: dict, key: str) -> float:
    x = summary["arms"]["certified"][key]
    return float(x["mean"] if isinstance(x, dict) else x)


def codesign_delta_from_summary(reg: dict, cps: dict) -> tuple[float, float, float]:
    """Table-6 Δ row: difference of certified-arm means in summary.json."""
    v_a = certified_mean(reg, "vmaf_mean")
    r_a = certified_mean(reg, "rebuffer_total")
    v_b = certified_mean(cps, "vmaf_mean")
    r_b = certified_mean(cps, "rebuffer_total")
    d_v = v_b - v_a
    d_r = r_b - r_a
    wstar = d_v / d_r if abs(d_r) > 1e-9 else float("nan")
    return d_v, d_r, wstar


def _episodes_certified_col(name: str, key: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    path = RES / name / "episodes.csv"
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] == "certified":
                rows[int(r["episode"])] = float(r[key])
    return rows


def verify_codesign_episodes(reg: dict, cps: dict, tol_vmaf: float = 0.05, tol_reb: float = 0.08):
    """Ensure episodes.csv matches summary.json and both runs share episode count."""
    issues: list[str] = []
    for tag, summary in [("proposed_5g_regime", reg), ("proposed_cps_5g", cps)]:
        path = RES / tag / "episodes.csv"
        if not path.exists():
            issues.append(f"{tag}: missing episodes.csv")
            continue
        v_col = _episodes_certified_col(tag, "vmaf_mean")
        r_col = _episodes_certified_col(tag, "rebuffer_total")
        if not v_col:
            issues.append(f"{tag}: episodes.csv has no certified rows")
            continue
        csv_v = float(np.mean(list(v_col.values())))
        csv_r = float(np.mean(list(r_col.values())))
        sum_v = certified_mean(summary, "vmaf_mean")
        sum_r = certified_mean(summary, "rebuffer_total")
        if abs(csv_v - sum_v) > tol_vmaf or abs(csv_r - sum_r) > tol_reb:
            issues.append(
                f"{tag}: episodes.csv stale vs summary.json "
                f"(csv VMAF {csv_v:.2f} vs {sum_v:.2f}; reb {csv_r:.2f} vs {sum_r:.2f}; n={len(v_col)})"
            )
    n_reg = len(_episodes_certified_col("proposed_5g_regime", "vmaf_mean"))
    n_cps = len(_episodes_certified_col("proposed_cps_5g", "vmaf_mean"))
    if n_reg and n_cps and n_reg != n_cps:
        issues.append(f"co-design episode-count mismatch: regime n={n_reg} vs cps n={n_cps}")
    return issues


def codesign_stats_paired():
    """Wilcoxon p-values on paired certified episodes (requires synced CSVs)."""
    v_a = _episodes_certified_col("proposed_5g_regime", "vmaf_mean")
    r_a = _episodes_certified_col("proposed_5g_regime", "rebuffer_total")
    v_b = _episodes_certified_col("proposed_cps_5g", "vmaf_mean")
    r_b = _episodes_certified_col("proposed_cps_5g", "rebuffer_total")
    eps = sorted(set(v_a) & set(v_b))
    va = np.array([v_a[e] for e in eps])
    vb = np.array([v_b[e] for e in eps])
    ra = np.array([r_a[e] for e in eps])
    rb = np.array([r_b[e] for e in eps])
    return wilcoxon_p(vb, va), wilcoxon_p(rb, ra)


def codesign_bootstrap(n_boot=20000, seed=0):
    """Paired-episode bootstrap CIs for the co-design contrast (co - eval).

    Resamples the shared episode indices with replacement and recomputes the
    mean VMAF gain, mean rebuffer change, and the QoE crossover w* on each
    replicate. Returns 95% percentile intervals plus P(dVMAF>0)."""
    def col(name, key, arm="certified"):
        rows = {}
        with open(RES / name / "episodes.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] == arm:
                    rows[int(r["episode"])] = float(r[key])
        return rows
    vA, rA = col("proposed_5g_regime", "vmaf_mean"), col("proposed_5g_regime", "rebuffer_total")
    vB, rB = col("proposed_cps_5g", "vmaf_mean"), col("proposed_cps_5g", "rebuffer_total")
    eps = sorted(set(vA) & set(vB))
    dv = np.array([vB[e] - vA[e] for e in eps])
    dr = np.array([rB[e] - rA[e] for e in eps])
    n = dv.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    mdv = dv[idx].mean(axis=1)
    mdr = dr[idx].mean(axis=1)
    ok = mdr > 0
    w = mdv[ok] / mdr[ok]
    q = lambda a, lo, hi: (float(np.percentile(a, lo)), float(np.percentile(a, hi)))
    dv_ci = q(mdv, 2.5, 97.5)
    dr_ci = q(mdr, 2.5, 97.5)
    w_ci = q(w, 2.5, 97.5)
    return {
        "dv_ci": dv_ci, "dr_ci": dr_ci, "w_ci": w_ci,
        "w_med": float(np.median(w)),
        "p_dv_pos": float((mdv > 0).mean()),
        "n": n,
    }


def write_all(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def gen_macros():
    M = []

    def macro(name, val):
        M.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    g5 = load("greedy_5g")
    n_ep = 204
    ep_path = RES / "greedy_5g" / "episodes.csv"
    if ep_path.exists():
        with open(ep_path, encoding="utf-8") as f:
            n_ep = sum(1 for _ in csv.DictReader(f)) // 3  # three arms per episode
    macro("CPSepisodes", str(n_ep))
    macro("CPSepsilon", num(g5["epsilon"], 1))
    macro("CPSalpha", num(g5["alpha"], 2))
    macro("CPScovTarget", num(g5.get("coverage_target", 1 - g5["alpha"]), 2))

    hosts = [
        ("Greedy", "greedy_5g"), ("Bba", "bba_5g"), ("Pen", "pensieve_5g"),
        ("Bola", "bola_5g"), ("Mpc", "mpc_5g"),
    ]
    for prefix, tag in hosts:
        s = load(tag)
        if not s:
            continue
        for a in ("raw", "safety", "certified"):
            suf = {"raw": "Raw", "safety": "Saf", "certified": "Cert"}[a]
            macro(f"CPS{prefix}{suf}Reb", num(arm(s, a, "rebuffer_total"), 1))
            macro(f"CPS{prefix}{suf}VM", num(arm(s, a, "vmaf_mean"), 1))
            macro(f"CPS{prefix}{suf}Br", f"{arm(s, a, 'bitrate_mean_kbps'):.0f}")
            cov = arm(s, a, "conformal_coverage")
            if isinstance(cov, float) and not math.isnan(cov):
                macro(f"CPS{prefix}{suf}Cov", num(cov, 3))
            macro(f"CPS{prefix}{suf}Int", num(arm(s, a, "interv_rate") * 100, 1))

    for tag, s in [("greedy", load("greedy_5g")), ("bba", load("bba_5g")),
                   ("bola", load("bola_5g")), ("mpc", load("mpc_5g"))]:
        if s is None:
            continue
        cs = cmp_(s, "certified_vs_safety")
        cr = cmp_(s, "certified_vs_raw")
        T = tag.capitalize()
        macro(f"CPS{T}BWcs", pct(cs["bandwidth_reduction_pct"]))
        macro(f"CPS{T}RebCs", pct(cs["rebuffer_change_pct"]))
        macro(f"CPS{T}VMcs", num(cs["vmaf_mean_diff"]))
        macro(f"CPS{T}RebCr", pct(cr["rebuffer_change_pct"]))
        macro(f"CPS{T}Cov", num(arm(s, "certified", "conformal_coverage"), 3))
        macro(f"CPS{T}PBWcs", pval(cs["bandwidth_wilcoxon_p"]))
        macro(f"CPS{T}PRebCs", pval(cs["rebuffer_wilcoxon_p"]))

    for tag, s in [("greedy", load("greedy_5g_improved")), ("bba", load("bba_5g_improved"))]:
        cs = cmp_(s, "certified_vs_safety")
        T = tag.capitalize()
        macro(f"CPS{T}ImpBWcs", pct(cs["bandwidth_reduction_pct"]))
        macro(f"CPS{T}ImpRebCs", pct(cs["rebuffer_change_pct"]))
        macro(f"CPS{T}ImpVMcs", num(cs["vmaf_mean_diff"]))
        macro(f"CPS{T}ImpCov", num(arm(s, "certified", "conformal_coverage"), 3))

    bb = load("greedy_bb")
    macro("CPSbbBWcs", pct(cmp_(bb, "certified_vs_safety")["bandwidth_reduction_pct"]))
    macro("CPSbbRebSr", pct(cmp_(bb, "safety_vs_raw")["rebuffer_change_pct"]))
    cov_bb = arm(bb, "certified", "conformal_coverage")
    if isinstance(cov_bb, float) and not math.isnan(cov_bb):
        macro("CPSbbCov", num(cov_bb, 3))

    for tag, s in [("Prop", load("proposed_5g")), ("PropImp", load("proposed_5g_improved"))]:
        cs = cmp_(s, "certified_vs_safety")
        macro(f"CPS{tag}BWcs", pct(cs["bandwidth_reduction_pct"]))
        macro(f"CPS{tag}RebCs", pct(cs["rebuffer_change_pct"]))
        macro(f"CPS{tag}VMcs", num(cs["vmaf_mean_diff"]))
        macro(f"CPS{tag}PRebCs", pval(cs["rebuffer_wilcoxon_p"]))

    reg, cps = load("proposed_5g_regime"), load("proposed_cps_5g")
    macro("CPScoRegVM", num(arm(reg, "certified", "vmaf_mean")))
    macro("CPScoRegReb", num(arm(reg, "certified", "rebuffer_total")))
    macro("CPScoRegBr", f"{arm(reg, 'certified', 'bitrate_mean_kbps'):.0f}")
    macro("CPScoRegCov", num(arm(reg, "certified", "conformal_coverage"), 3))
    macro("CPScoCpsVM", num(arm(cps, "certified", "vmaf_mean")))
    macro("CPScoCpsReb", num(arm(cps, "certified", "rebuffer_total")))
    macro("CPScoCpsBr", f"{arm(cps, 'certified', 'bitrate_mean_kbps'):.0f}")
    macro("CPScoCpsCov", num(arm(cps, "certified", "conformal_coverage"), 3))
    d_v, d_r, wstar = codesign_delta_from_summary(reg, cps)
    macro("CPScoDVM", num(d_v))
    macro("CPScoDReb", num(d_r))
    macro("CPScoWstar", num(wstar))
    co_issues = verify_codesign_episodes(reg, cps)
    if co_issues:
        for msg in co_issues:
            print(f"[ERROR] co-design: {msg}")
        print("[ERROR] Re-run BOTH proposed_5g_regime and proposed_cps_5g at CPS_EPISODES,")
        print("        then pull episodes.csv + summary.json before regenerating paper assets.")
        if os.environ.get("CODESIGN_STRICT", "1") != "0":
            raise SystemExit(1)
        p_vm, p_reb = float("nan"), float("nan")
        bs = None
    else:
        p_vm, p_reb = codesign_stats_paired()
        bs = codesign_bootstrap()
    macro("CPScoPVM", pval(p_vm))
    macro("CPScoPReb", pval(p_reb))

    def _bs(name, key, sub=None):
        if bs is None:
            macro(name, "---")
        elif sub is None:
            macro(name, num(bs[key]))
        else:
            macro(name, num(bs[key][sub]))

    _bs("CPScoDVMlo", "dv_ci", 0)
    _bs("CPScoDVMhi", "dv_ci", 1)
    _bs("CPScoDRebLo", "dr_ci", 0)
    _bs("CPScoDRebHi", "dr_ci", 1)
    _bs("CPScoWstarLo", "w_ci", 0)
    _bs("CPScoWstarHi", "w_ci", 1)
    _bs("CPScoWstarMed", "w_med")
    if bs is None:
        macro("CPScoDVMposPct", "---")
        macro("CPScoBootN", str(int(n_ep)))
    else:
        macro("CPScoDVMposPct", f"{100 * bs['p_dv_pos']:.1f}")
        macro("CPScoBootN", str(bs["n"]))

    header = ("% Auto-generated by make_cps_paper_assets.py -- DO NOT EDIT BY HAND.\n"
              "% Certified Perceptual Shield (V18) numbers.\n")
    text = header + "\n".join(M) + "\n"
    for d in OUT_DIRS:
        write_all(d / "macros_cps.tex", text)
    print(f"macros: {len(M)} entries")


def gen_table_full():
    rows = []
    for host, raw, saf, cert in [
        ("Greedy", "CPSGreedyRaw", "CPSGreedySaf", "CPSGreedyCert"),
        ("BBA", "CPSBbaRaw", "CPSBbaSaf", "CPSBbaCert"),
        ("BOLA", "CPSBolaRaw", "CPSBolaSaf", "CPSBolaCert"),
        ("RobustMPC", "CPSMpcRaw", "CPSMpcSaf", "CPSMpcCert"),
        ("Pensieve", "CPSPenRaw", "CPSPenSaf", "CPSPenCert"),
    ]:
        rows.append(
            f"{host} & \\textsc{{Raw}} & \\{raw}Reb{{}}\\,s & \\{raw}VM{{}} & \\{raw}Br{{}} & --- & 0.0\\\\\n"
            f" & \\textsc{{Safety}} & \\{saf}Reb{{}}\\,s & \\{saf}VM{{}} & \\{saf}Br{{}} & \\{saf}Cov{{}} & \\{saf}Int{{}}\\%\\\\\n"
            f" & \\textsc{{Certified}} & \\{cert}Reb{{}}\\,s & \\{cert}VM{{}} & \\{cert}Br{{}} & \\{cert}Cov{{}} & \\{cert}Int{{}}\\%\\\\"
        )
    body = "\n\\midrule\n".join(rows)
    tex = f"""% Full paired arms on 5G test (macros from make_cps_paper_assets.py).
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}>{{\\raggedright\\arraybackslash}}X c ccccc@{{}}}}
\\toprule
\\textbf{{Host}} & \\textbf{{Arm}} & \\textbf{{Reb.}} & \\textbf{{VMAF}} & \\textbf{{Bitrate}} & \\textbf{{Cover.}} & \\textbf{{Interv.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
"""
    for d in OUT_DIRS:
        write_all(d / "table_cps_full.tex", tex)
    print("wrote table_cps_full.tex")


def gen_table_codesign():
    tex = r"""% Co-design: certified arm only (5G test, improved shield at eval).
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X cccc@{}}
\toprule
\textbf{Policy} & \textbf{VMAF} & \textbf{Reb. (s)} & \textbf{Bitrate} & \textbf{Cover.} \\
\midrule
Eval-only shield & \CPScoRegVM & \CPScoRegReb & \CPScoRegBr & \CPScoRegCov \\
Co-trained (CPS) & \CPScoCpsVM & \CPScoCpsReb & \CPScoCpsBr & \CPScoCpsCov \\
\midrule
$\Delta$ (co $-$ eval) & $+\CPScoDVM$ & $+\CPScoDReb$ & --- & --- \\
\bottomrule
\end{tabularx}
"""
    for d in OUT_DIRS:
        write_all(d / "table_cps_codesign.tex", tex)
    print("wrote table_cps_codesign.tex")


COLORS = {
    "raw": "#D55E00",
    "safety": "#0072B2",
    "certified": "#009E73",
    "eval": "#0072B2",
    "codesign": "#E69F00",
}
ARM_LABEL = {"raw": "Raw", "safety": "Safety", "certified": "Certified"}


def _plt():
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        })
        return plt
    except ImportError:
        print("matplotlib not available; skip figures")
        return None


def _save(fig, name, plt):
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def load_episodes(name):
    path = RES / name / "episodes.csv"
    rows = {"raw": [], "safety": [], "certified": []}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["arm"]
            if a in rows:
                rows[a].append({
                    "reb": float(r["rebuffer_total"]),
                    "vmaf": float(r["vmaf_mean"]),
                    "br": float(r["bitrate_mean_kbps"]),
                    "ep": int(r["episode"]),
                })
    for a in rows:
        rows[a].sort(key=lambda x: x["ep"])
    return rows


def gen_hero_figure():
    plt = _plt()
    if plt is None:
        return

    data = []
    for label, tag in [("Greedy", "greedy_5g"), ("BBA", "bba_5g")]:
        s = load(tag)
        cs = cmp_(s, "certified_vs_safety")
        data.append({
            "host": label,
            "bw": -cs["bandwidth_reduction_pct"],
            "reb": -cs["rebuffer_change_pct"],
            "vmaf_loss": max(0.0, -cs["vmaf_mean_diff"]),
        })

    hosts = [d["host"] for d in data]
    x = np.arange(len(hosts))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
    b1 = ax.bar(x - w, [d["bw"] for d in data], w, label="Bitrate saved (%)", color="#0072B2")
    b2 = ax.bar(x, [d["reb"] for d in data], w, label="Rebuffer reduced (%)", color="#009E73")
    b3 = ax.bar(x + w, [d["vmaf_loss"] for d in data], w, label="VMAF cost (pts)", color="#E69F00")
    for b in (b1, b2, b3):
        ax.bar_label(b, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(hosts)
    ax.set_ylabel("Relative change (%)")
    ax.set_ylim(0, max(d["reb"] for d in data) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.legend(loc="outside lower center", ncol=3, fontsize=8.5, frameon=False)
    _save(fig, "fig_cps_hero.pdf", plt)


def gen_tradeoff_figure():
    """Scatter: rebuffer vs VMAF for Raw / Safety / Certified (Greedy, BBA)."""
    plt = _plt()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=False,
                             constrained_layout=True)
    handles = None
    for ax, (title, tag) in zip(axes, [("Greedy", "greedy_5g"), ("BBA", "bba_5g")]):
        rows = load_episodes(tag)
        for arm in ("raw", "safety", "certified"):
            xs = [r["reb"] for r in rows[arm]]
            ys = [r["vmaf"] for r in rows[arm]]
            ax.scatter(xs, ys, s=14, alpha=0.55, c=COLORS[arm],
                       label=ARM_LABEL[arm], edgecolors="none")
            ax.scatter([np.mean(xs)], [np.mean(ys)], s=90, c=COLORS[arm],
                       marker="D", edgecolors="k", linewidths=0.6, zorder=5)
        ax.set_xlabel("Rebuffering (s)")
        ax.grid(alpha=0.25)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Mean VMAF")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    _save(fig, "fig_cps_tradeoff.pdf", plt)


def gen_cdf_figure():
    """CDF of episode rebuffering for Greedy and BBA."""
    plt = _plt()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True,
                             constrained_layout=True)
    handles = None
    for ax, (title, tag) in zip(axes, [("Greedy", "greedy_5g"), ("BBA", "bba_5g")]):
        rows = load_episodes(tag)
        for arm in ("raw", "safety", "certified"):
            vals = np.sort([r["reb"] for r in rows[arm]])
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, y, color=COLORS[arm], lw=1.8, label=ARM_LABEL[arm])
        ax.set_xlabel("Rebuffering (s)")
        ax.set_xlim(left=0)
        ax.grid(alpha=0.25)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("CDF")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    _save(fig, "fig_cps_cdf_reb.pdf", plt)


def gen_regime_figure():
    """5G vs broadband: Certified vs Safety bitrate / rebuffer savings."""
    plt = _plt()
    if plt is None:
        return

    rows = []
    for label, tag in [("5G greedy", "greedy_5g"), ("5G BBA", "bba_5g"),
                       ("BB greedy", "greedy_bb")]:
        s = load(tag)
        if s is None:
            continue
        cs = cmp_(s, "certified_vs_safety")
        rows.append({
            "label": label,
            "bw": -cs["bandwidth_reduction_pct"],
            "reb": -cs["rebuffer_change_pct"],
        })

    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    b1 = ax.bar(x - w / 2, [r["bw"] for r in rows], w, label="Bitrate saved (%)", color="#0072B2")
    b2 = ax.bar(x + w / 2, [r["reb"] for r in rows], w, label="Rebuffer reduced (%)", color="#009E73")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Savings (%)")
    ax.set_ylim(0, max(max(r["bw"], r["reb"]) for r in rows) * 1.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    _save(fig, "fig_cps_regime.pdf", plt)


def gen_codesign_figure():
    """Scatter + QoE crossover for eval-only vs co-trained certified arms."""
    plt = _plt()
    if plt is None:
        return

    def col(name, key):
        out = {}
        with open(RES / name / "episodes.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] == "certified":
                    out[int(r["episode"])] = float(r[key])
        return out

    vA, rA = col("proposed_5g_regime", "vmaf_mean"), col("proposed_5g_regime", "rebuffer_total")
    vB, rB = col("proposed_cps_5g", "vmaf_mean"), col("proposed_cps_5g", "rebuffer_total")
    eps = sorted(set(vA) & set(vB))
    va = np.array([vA[e] for e in eps])
    vb = np.array([vB[e] for e in eps])
    ra = np.array([rA[e] for e in eps])
    rb = np.array([rB[e] for e in eps])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(ra, va, s=16, alpha=0.55, c=COLORS["eval"], label="Eval-only shield", edgecolors="none")
    ax.scatter(rb, vb, s=16, alpha=0.55, c=COLORS["codesign"], label="Co-trained CPS", edgecolors="none")
    ax.scatter([ra.mean()], [va.mean()], s=100, c=COLORS["eval"], marker="D", edgecolors="k", zorder=5)
    ax.scatter([rb.mean()], [vb.mean()], s=100, c=COLORS["codesign"], marker="D", edgecolors="k", zorder=5)
    ax.set_xlabel("Rebuffering (s)")
    ax.set_ylabel("Mean VMAF")
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.92)
    ax.grid(alpha=0.25)

    ax = axes[1]
    # QoE(w) = mean VMAF - w * mean reb; plot mean QoE difference vs w
    ws = np.linspace(0, 20, 201)
    qA = va.mean() - ws * ra.mean()
    qB = vb.mean() - ws * rb.mean()
    dQ = qB - qA
    wstar = (vb.mean() - va.mean()) / max(rb.mean() - ra.mean(), 1e-12)
    ax.plot(ws, dQ, color="#009E73", lw=2.0, label=r"$\Delta$QoE (co $-$ eval)")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(wstar, color="#D55E00", ls="--", lw=1.4,
               label=rf"$w^\star={wstar:.2f}$")
    ax.set_xlabel(r"Stall weight $w$ (VMAF pts / s)")
    ax.set_ylabel(r"$\Delta$QoE")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.25)

    _save(fig, "fig_cps_codesign.pdf", plt)


def gen_overview_figure():
    """Single-panel overview merging the former hero and regime figures:
    certified-vs-safety bitrate saved, rebuffer reduced, and VMAF cost for
    two 5G hosts and a broadband host, so banking magnitude, near-zero
    perceptual cost, and the regime collapse all read from one chart."""
    plt = _plt()
    if plt is None:
        return

    groups = []
    for label, tag in [("5G greedy", "greedy_5g"), ("5G BBA", "bba_5g"),
                       ("BB greedy", "greedy_bb")]:
        s = load(tag)
        if s is None:
            continue
        cs = cmp_(s, "certified_vs_safety")
        groups.append({
            "label": label,
            "bw": -cs["bandwidth_reduction_pct"],
            "reb": -cs["rebuffer_change_pct"],
            "vmaf_loss": max(0.0, -cs["vmaf_mean_diff"]),
        })

    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    b1 = ax.bar(x - w, [g["bw"] for g in groups], w, color="#0072B2", label="Bitrate saved (%)")
    b2 = ax.bar(x, [g["reb"] for g in groups], w, color="#009E73", label="Rebuffer reduced (%)")
    b3 = ax.bar(x + w, [g["vmaf_loss"] for g in groups], w, color="#E69F00", label="VMAF cost (pts)")
    for b in (b1, b2, b3):
        ax.bar_label(b, fmt="%.1f", padding=2, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups])
    ax.set_ylabel("Change vs. safety (%)")
    ax.set_ylim(0, max(g["reb"] for g in groups) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.legend(loc="outside lower center", ncol=3, fontsize=8.5, frameon=False)
    _save(fig, "fig_cps_overview.pdf", plt)


def gen_coverage_figure():
    """Per-episode conformal coverage (certified arm) across hosts vs target."""
    plt = _plt()
    if plt is None:
        return

    def per_episode_cov(name):
        vals = []
        p = RES / name / "episodes.csv"
        if not p.exists():
            return vals
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] != "certified":
                    continue
                c = r.get("conformal_coverage", "")
                if c not in ("", "nan"):
                    try:
                        vals.append(float(c))
                    except ValueError:
                        pass
        return vals

    hosts = [("Greedy", "greedy_5g"), ("BBA", "bba_5g"), ("BOLA", "bola_5g"),
             ("MPC", "mpc_5g"), ("Pensieve", "pensieve_5g"),
             ("Proposed", "proposed_5g_improved"), ("Co-trained", "proposed_cps_5g")]
    data, labels = [], []
    for label, tag in hosts:
        v = per_episode_cov(tag)
        if v:
            data.append(v)
            labels.append(label)

    target = 0.90
    fig, ax = plt.subplots(figsize=(7.4, 3.4), constrained_layout=True)
    parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#0072B2")
        pc.set_alpha(0.30)
    bp = ax.boxplot(data, widths=0.28, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.3))
    for box in bp["boxes"]:
        box.set(facecolor="#009E73", alpha=0.65)
    ax.axhline(target, color="#D55E00", ls="--", lw=1.5,
               label=rf"Target $1-\alpha={target:.2f}$")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.5)
    ax.set_ylabel("Per-episode coverage")
    ax.set_ylim(min(0.86, target - 0.02), 1.005)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    _save(fig, "fig_cps_coverage.pdf", plt)


def gen_ladder_table():
    """Session-mean ladder summary for all twelve v19 titles."""
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from configs.videos import VIDEO_SPECS

    vmaf = pd.read_csv(ROOT / "data/vmaf_scores/vmaf_summary.csv")
    siti_path = ROOT / "data/content_features/siti_summary.csv"
    siti = pd.read_csv(siti_path).set_index("video") if siti_path.exists() else None

    rows = []
    for spec in VIDEO_SPECS:
        slug = spec["slug"]
        g = vmaf[vmaf["video"] == slug].sort_values("bitrate_kbps")
        if len(g) < 6:
            continue
        v = {int(r.bitrate_kbps): float(r.vmaf) for r in g.itertuples()}
        top_gain = v[6000] - v[2850]
        span = v[6000] - v[300]
        vals = [v[b] for b in sorted(v)]
        mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        si = float(siti.loc[slug, "mean_si"]) if siti is not None and slug in siti.index else float("nan")
        ti = float(siti.loc[slug, "mean_ti"]) if siti is not None and slug in siti.index else float("nan")
        ho = r"\checkmark" if spec["held_out"] else "---"
        name = spec["display_name"].replace("&", r"\&")
        rows.append(
            f"{name} & {ho} & {si:.1f} & {ti:.1f} & {top_gain:+.2f} & {span:.1f} & "
            f"{'Yes' if mono else 'No'} \\\\"
        )

    body = "\n".join(rows)
    tex = f"""% Session-mean ladder summary (v19, twelve titles) — make_cps_paper_assets.py
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}>{{\\raggedright\\arraybackslash}}X c c c c c c@{{}}}}
\\toprule
\\textbf{{Video}} & \\textbf{{Held-out}} & \\textbf{{SI}} & \\textbf{{TI}} &
\\textbf{{Top gain}} & \\textbf{{Span}} & \\textbf{{Mono.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
"""
    for d in OUT_DIRS:
        write_all(d / "table_ladder_spacing.tex", tex)
    print("wrote table_ladder_spacing.tex (12 titles, session-mean)")


def gen_ladder_macros():
    """Headline ladder macros from session-mean VMAF CSV."""
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from configs.videos import ALL_VIDEOS

    vmaf = pd.read_csv(ROOT / "data/vmaf_scores/vmaf_summary.csv")
    n_nonmono = 0
    top_gains = []
    for slug in ALL_VIDEOS:
        g = vmaf[vmaf["video"] == slug].sort_values("bitrate_kbps")
        if len(g) < 6:
            continue
        v = g["vmaf"].tolist()
        if not all(v[i] <= v[i + 1] for i in range(len(v) - 1)):
            n_nonmono += 1
        top_gains.append(float(g.iloc[-1]["vmaf"] - g.iloc[-2]["vmaf"]))

    M = [
        r"\providecommand{\LadNVideos}{12}",
        rf"\providecommand{{\LadPctNonMonotone}}{{{100 * n_nonmono / max(len(top_gains), 1):.0f}}}",
        rf"\providecommand{{\LadTopGainMean}}{{{np.mean(top_gains):.2f}}}",
    ]
    header = "% Session-mean ladder macros (v19) — make_cps_paper_assets.py\n"
    text = header + "\n".join(M) + "\n"
    for d in OUT_DIRS:
        write_all(d / "macros_ladder_v19.tex", text)
    print("wrote macros_ladder_v19.tex")


if __name__ == "__main__":
    gen_macros()
    gen_table_full()
    gen_table_codesign()
    gen_ladder_table()
    gen_ladder_macros()
    gen_overview_figure()
    gen_tradeoff_figure()
    gen_cdf_figure()
    gen_codesign_figure()
    gen_coverage_figure()
