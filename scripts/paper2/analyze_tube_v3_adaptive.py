"""Tables + figures for the Tube-v3 ADAPTIVE frozen test.

usage: python analyze_tube_v3_adaptive.py <adaptive_test_dir> <prev_final_test_dir> <fig_dir> <out_md> [freeze_yaml]
"""
import glob, os, sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R, PREV, FIG, OUT = sys.argv[1:5]
FREEZE_YAML = sys.argv[5] if len(sys.argv) > 5 else "configs/tube_v3_adaptive_final.yaml"
os.makedirs(FIG, exist_ok=True)
SETS = ["5g_markov", "5g_harsh", "bb", "5g_irish"]


def _load_freeze_yaml(path: str) -> dict:
    cfg: dict = {}
    cur_list: str | None = None
    root = Path(__file__).resolve().parents[2]
    for raw in (root / path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and cur_list is not None:
            cfg[cur_list].append(line[4:].strip())
            continue
        cur_list = None
        k, v = line.split(":", 1)
        v = v.strip()
        if v == "":
            cur_list = k
            cfg.setdefault(k, [])
        elif v.startswith("["):
            cfg[k] = [x.strip() for x in v.strip("[]").split(",") if x.strip()]
        else:
            cfg[k] = v
    return cfg


def _primary_arms(cfg: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for ts in SETS:
        arms = cfg.get(f"arms_{ts}") or []
        if not arms:
            raise KeyError(f"missing arms_{ts} in {FREEZE_YAML}")
        out[ts] = str(arms[0]).split("#", 1)[0].strip()
    return out


PRIMARY = _primary_arms(_load_freeze_yaml(FREEZE_YAML))
GLOBAL = "tube_v2_maxscore_a0.1_safe_fs"
LEG = "tube_v2_step_static_a0.15"
rng0 = lambda: np.random.default_rng(0)

def clus(d, c, n=4000):
    g = d.groupby("trace_file")[c].apply(np.array).tolist(); r = rng0(); k = len(g)
    bs = [np.concatenate([g[i] for i in r.integers(0, k, k)]).mean() for _ in range(n)]
    return d[c].mean(), *np.percentile(bs, [2.5, 97.5])

def pclus(m, c, n=4000):
    m = m.assign(dd=m[c] - m[c + "_r"]); g = m.groupby("trace_file").dd.apply(np.array).tolist(); r = rng0(); k = len(g)
    bs = [np.concatenate([g[i] for i in r.integers(0, k, k)]).mean() for _ in range(n)]
    return m.dd.mean(), *np.percentile(bs, [2.5, 97.5])

def pair(m, c, n=4000):
    dd = (m[c] - m[c + "_r"]).values; r = rng0()
    bs = [dd[r.integers(0, len(dd), len(dd))].mean() for _ in range(n)]
    return dd.mean(), *np.percentile(bs, [2.5, 97.5])

F = lambda t, p=1: f"{t[0]:+.{p}f} [{t[1]:+.{p}f},{t[2]:+.{p}f}]"
G = lambda t, p=1: f"{t[0]:.{p}f} [{t[1]:.{p}f},{t[2]:.{p}f}]"
load = lambda root, ts: {os.path.basename(p)[:-4].replace("_at_", "@"): pd.read_csv(p, comment="#")
                         for p in sorted(glob.glob(f"{root}/{ts}/*.csv"))}
L = []; D = {}; KEY = []
for ts in SETS:
    A = load(R, ts); D[ts] = A; d0 = A["robust_mpc_logu"]
    L.append(f"\n## {ts} (test) — n_traces={d0.trace_file.nunique()}, episodes/arm={len(d0)}\n\n"
             "Means with trace-cluster bootstrap 95% CI. ★ = QoE–rebuffer Pareto-efficient. P = rule-selected primary.\n\n"
             "| arm | QoE | rebuf (s) | p90 rebuf | VMAF | cov h1..h5 | joint |\n|---|---|---|---|---|---|---|")
    pts = {k: (d.episodic_qoe.mean(), d.rebuffer_total.mean()) for k, d in A.items()}
    for k, d in sorted(A.items(), key=lambda kv: -kv[1].episodic_qoe.mean()):
        q, r = pts[k]; par = not any((q2 >= q and r2 <= r) and (q2 > q or r2 < r) for q2, r2 in pts.values())
        cov = "/".join(f"{d[f'cov_h{h}'].mean():.2f}" for h in range(1, 6)) if "cov_h1" in d else "—"
        j = f"{d.joint_coverage_ep.mean():.3f}" if "joint_coverage_ep" in d else "—"
        tag = (" ★" if par else "") + (" **P**" if k == PRIMARY[ts] else "")
        L.append(f"| `{k}`{tag} | {G(clus(d,'episodic_qoe'))} | {G(clus(d,'rebuffer_total'),2)} | {d.rebuffer_total.quantile(.9):.2f} | "
                 f"{G(clus(d,'mean_vmaf'))} | {cov} | {j} |")
    for ref in ["robust_mpc_logu", "mpc_fixed_quantile_native", LEG]:
        L.append(f"\n**Δ vs `{ref}`** (arm − ref). Cluster CI (primary) / paired-episode CI; W/T/L on QoE.\n\n"
                 "| arm | ΔQoE cluster | ΔQoE paired | Δrebuf cluster | ΔVMAF cluster | W/T/L |\n|---|---|---|---|---|---|")
        for k in A:
            if not k.startswith("tube") or k == ref: continue
            m = A[k].merge(A[ref], on=["trace_file", "episode"], suffixes=("", "_r"))
            dq = (m.episodic_qoe - m.episodic_qoe_r).values; w = (dq > 1e-9).sum(); l = (dq < -1e-9).sum()
            row = (F(pclus(m, "episodic_qoe")), F(pair(m, "episodic_qoe")), F(pclus(m, "rebuffer_total"), 2), F(pclus(m, "mean_vmaf")))
            L.append(f"| `{k}` | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {w}/{len(m)-w-l}/{l} |")
            if k == PRIMARY[ts] and ref == "robust_mpc_logu":
                KEY.append((ts, k, A[k], A[ref], row))
    # reproducibility vs previous frozen test
    P = load(PREV, ts)
    L.append("\n**Reproducibility: legacy arms here vs previous frozen test (configs/tube_v3_final.yaml)**\n\n| arm | QoE now | QoE prev | identical CSV values |\n|---|---|---|---|")
    for k in [LEG, "tube_v2_maxscore_a0.1", "robust_mpc_logu", "mpc_fixed_quantile_native"]:
        if k in P and k in A:
            same = np.allclose(A[k].episodic_qoe.values, P[k].episodic_qoe.values)
            L.append(f"| `{k}` | {A[k].episodic_qoe.mean():.2f} | {P[k].episodic_qoe.mean():.2f} | {same} |")
_hdr = (
    f"# Tube-v3 ADAPTIVE frozen TEST report\n\n"
    f"- Results: `{R}`\n"
    f"- Freeze config: `{FREEZE_YAML}`\n"
    f"- Primary arms (first per `arms_<set>`): {PRIMARY}\n"
    f"- Bootstrap: trace-cluster (4000), paired-episode for reference\n\n"
)
KT = [_hdr, "## Key table (primary = rule-selected max-score `_safe_fs`) vs RobustMPC, test\n",
      "| set | primary arm | QoE | rebuf (s) | joint cov | ΔQoE vs RobustMPC (cluster) | Δrebuf (cluster) |", "|---|---|---|---|---|---|---|"]
for ts, k, a, r, row in KEY:
    KT.append(f"| {ts} | `{k}` | {a.episodic_qoe.mean():.1f} | {a.rebuffer_total.mean():.2f} | {a.joint_coverage_ep.mean():.3f} | {row[0]} | {row[2]} |")
open(OUT, "w").write("\n".join(KT + L) + "\n")

NICE = lambda k: (k.replace("tube_v2_", "").replace("_safe_fs", " safe+fs").replace("_safe", " safe")
                  .replace("step_static", "per-step").replace("_a", " α=").replace("robust_mpc_logu", "RobustMPC")
                  .replace("mpc_fixed_quantile_native", "fixed-quantile MPC").replace("mpc_fixed_rho_", "MPC ρ="))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
for a, ts in zip(ax, SETS):
    for k, d in D[ts].items():
        if k == "raw": continue
        q, r = d.episodic_qoe.mean(), d.rebuffer_total.mean()
        prim = k == PRIMARY[ts]; new = "_safe" in k
        a.scatter(r, q, s=150 if prim else 55, marker="*" if prim else ("o" if k.startswith("tube") else "s"),
                  c="tab:red" if prim else ("tab:green" if new else ("tab:blue" if k.startswith("tube") else "tab:gray")), zorder=3)
        a.annotate(NICE(k), (r, q), fontsize=6, xytext=(3, 2), textcoords="offset points")
    a.set_title(ts); a.set_xlabel("mean rebuffering per episode (s)"); a.grid(alpha=.3)
ax[0].set_ylabel("mean episodic QoE")
fig.suptitle("Adaptive frozen TEST: QoE vs rebuffering (red★ rule-selected primary; green = new _safe/_fs; blue = legacy tube; gray = baselines; raw omitted)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/pareto_qoe_rebuffer.png", dpi=200); plt.close(fig)
fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for a, ts in zip(ax, SETS):
    for k, d in D[ts].items():
        if "cov_h1" not in d or "maxscore" not in k and k != LEG: continue
        a.plot(range(1, 6), [d[f"cov_h{h}"].mean() for h in range(1, 6)], marker="o", lw=2.5 if k == PRIMARY[ts] else 1,
               label=f"{NICE(k)} (joint {d.joint_coverage_ep.mean():.2f})")
    for t in (0.8, 0.85, 0.9): a.axhline(t, ls="--", c="k", lw=.6)
    a.set_title(ts); a.set_xlabel("horizon step h"); a.set_xticks(range(1, 6)); a.grid(alpha=.3); a.legend(fontsize=6)
ax[0].set_ylabel("empirical per-step coverage")
fig.suptitle("Adaptive frozen TEST: per-step coverage vs horizon (joint coverage in legend)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/coverage_vs_horizon.png", dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 3.8)); xs = np.arange(4); w = .27
for i, k in enumerate(["robust_mpc_logu", LEG, "PRIMARY"]):
    vals = [D[ts][PRIMARY[ts] if k == "PRIMARY" else k].rebuffer_total.quantile(.9) for ts in SETS]
    ax.bar(xs + (i - 1) * w, vals, w, label={"PRIMARY": "max-score safe+fs (rule α)", LEG: "legacy per-step α=0.15"}.get(k, "RobustMPC"))
ax.set_xticks(xs); ax.set_xticklabels(SETS); ax.set_ylabel("p90 per-episode rebuffer (s)"); ax.set_yscale("symlog"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
ax.set_title("Adaptive frozen TEST: rebuffer tail", fontsize=9); fig.tight_layout(); fig.savefig(f"{FIG}/rebuffer_tail_p90.png", dpi=200); plt.close(fig)
print("ok")
