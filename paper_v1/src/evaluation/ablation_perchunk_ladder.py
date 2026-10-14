"""Per-chunk (non-monotone) vs pooled (session-mean) ladder ablation for CPS banking.

Reviewer concern addressed:
    The headline evaluation runs the VMAF-knee banking rule on the *pooled*
    session-mean ladder, which is almost always monotone (so Proposition 2 applies
    trivially). This script re-runs the banking stage on the *true per-chunk* ladders
    -- which are frequently non-monotone / inverted (Table 3) -- and quantifies:

      (a) the banking rule remains well defined on non-monotone ladders;
      (b) a shield that sees the per-chunk ladder keeps the realized per-chunk
          perceptual cost within the budget epsilon *by construction*
          (zero budget violations);
      (c) a shield that only sees the pooled ladder incurs a small, quantified
          per-chunk budget-violation rate -- the price of the pooling simplification;
      (d) the banked byte savings are comparable (indeed slightly larger) under the
          per-chunk ladder, because local inversions enlarge banking opportunity.

Only the *banking* stage depends on the VMAF ladder; the feasibility projection is
VMAF-independent (it depends on download time), so it is unaffected by inversions.

No RL environment, checkpoints, or server access are required: the ablation replays
the deterministic knee rule over the real per-chunk VMAF logs shipped in
new/data/vmaf_scores/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ paths
HERE = Path(__file__).resolve()
NEW_ROOT = HERE.parents[2]                       # .../new
VMAF_DIR = NEW_ROOT / "data" / "vmaf_scores"
PAPER = NEW_ROOT / "src" / "paper"
TABLES = [PAPER / "overleaf_upload" / "tables", PAPER / "tables"]
FIGURES = [PAPER / "overleaf_upload" / "figures", PAPER / "figures"]
RESULTS = NEW_ROOT / "results" / "perchunk_ablation"

BITRATES = [300, 750, 1200, 1850, 2850, 6000]    # ladder rungs (kbps)
L = len(BITRATES)
TOP = L - 1                                       # greedy proposal = top rung
EPS_SWEEP = [0.5, 1.0, 2.0, 4.0]
EPS_HEADLINE = 1.0

VIDEO_LABEL = {
    "bigbuckbunny": "BigBuckBunny",
    "sintel": "Sintel",
    "tearsofsteel_short": "TearsOfSteel",
    "crowd_run": "CrowdRun",
}


def knee(vmaf_ladder: np.ndarray, proposal: int, eps: float) -> int:
    """Smallest rung j<=proposal within eps of the proposal's VMAF (eq. 7)."""
    v_prop = vmaf_ladder[proposal]
    for j in range(proposal + 1):
        if v_prop - vmaf_ladder[j] <= eps:
            return j
    return proposal


def load_ladders():
    """Return per-chunk ladders and the pooled (session-mean) ladder per video."""
    pc = pd.read_csv(VMAF_DIR / "vmaf_perchunk.csv")
    pooled = pd.read_csv(VMAF_DIR / "vmaf_summary.csv")

    per_chunk = {}   # video -> list of np.array(L) indexed by rung
    for (video, chunk), grp in pc.groupby(["video", "chunk"]):
        row = grp.set_index("bitrate_kbps")["vmaf"]
        if not all(b in row.index for b in BITRATES):
            continue
        per_chunk.setdefault(video, []).append(
            np.array([float(row[b]) for b in BITRATES], dtype=float)
        )

    pooled_ladder = {}
    for video, grp in pooled.groupby("video"):
        row = grp.set_index("bitrate_kbps")["vmaf"]
        if not all(b in row.index for b in BITRATES):
            continue
        pooled_ladder[video] = np.array([float(row[b]) for b in BITRATES], dtype=float)

    return per_chunk, pooled_ladder


def run():
    per_chunk, pooled_ladder = load_ladders()
    bitr = np.array(BITRATES, dtype=float)

    rows = []            # sweep summary rows
    n_chunks_total = 0
    n_inverted_chunks = 0

    # count inverted chunks once (adjacent-pair inversion on the per-chunk ladder)
    for video, ladders in per_chunk.items():
        for v in ladders:
            n_chunks_total += 1
            if np.any(np.diff(v) < 0):
                n_inverted_chunks += 1

    for eps in EPS_SWEEP:
        knee_pool_idx, knee_pc_idx = [], []
        bw_pool, bw_pc = [], []
        cost_pc_shield, cost_pool_shield = [], []
        viol_pool = 0
        disagree = 0
        n = 0
        for video, ladders in per_chunk.items():
            vp = pooled_ladder.get(video)
            if vp is None:
                continue
            for vc in ladders:
                n += 1
                jp = knee(vp, TOP, eps)          # decision from pooled ladder
                jc = knee(vc, TOP, eps)          # decision from per-chunk ladder
                knee_pool_idx.append(jp)
                knee_pc_idx.append(jc)
                # banked byte savings vs top rung (%)
                bw_pool.append(100.0 * (bitr[TOP] - bitr[jp]) / bitr[TOP])
                bw_pc.append(100.0 * (bitr[TOP] - bitr[jc]) / bitr[TOP])
                # realized per-chunk perceptual cost = true VMAF(top) - true VMAF(chosen)
                c_pc = vc[TOP] - vc[jc]          # per-chunk shield: <= eps by construction
                c_pool = vc[TOP] - vc[jp]        # pooled shield judged on true per-chunk VMAF
                cost_pc_shield.append(c_pc)
                cost_pool_shield.append(c_pool)
                if c_pool > eps + 1e-9:
                    viol_pool += 1
                if jp != jc:
                    disagree += 1
        rows.append({
            "epsilon": eps,
            "n_chunks": n,
            "knee_idx_pooled": float(np.mean(knee_pool_idx)),
            "knee_idx_perchunk": float(np.mean(knee_pc_idx)),
            "bitrate_saved_pct_pooled": float(np.mean(bw_pool)),
            "bitrate_saved_pct_perchunk": float(np.mean(bw_pc)),
            "vmaf_cost_perchunk_shield": float(np.mean(cost_pc_shield)),
            "vmaf_cost_pooled_shield": float(np.mean(cost_pool_shield)),
            "budget_violation_pct_pooled": 100.0 * viol_pool / max(n, 1),
            "decision_disagree_pct": 100.0 * disagree / max(n, 1),
        })

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "perchunk_ladder_ablation.csv", index=False)

    inv_pct = 100.0 * n_inverted_chunks / max(n_chunks_total, 1)
    head = df[df.epsilon == EPS_HEADLINE].iloc[0]
    write_table(df)
    write_macros(head, inv_pct, n_chunks_total)
    write_figure(df)

    print(df.to_string(index=False))
    print(f"\nchunks total={n_chunks_total}  inverted={n_inverted_chunks} ({inv_pct:.1f}%)")
    return df


def write_table(df: pd.DataFrame):
    lines = [
        r"% Auto-generated by ablation_perchunk_ladder.py",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$\varepsilon$ & \multicolumn{2}{c}{Mean knee rung} & "
        r"\multicolumn{2}{c}{Bitrate saved (\%)} & "
        r"\shortstack{Per-chunk cost\\(pts)} & \shortstack{Budget viol.\\pooled (\%)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & pooled & per-chunk & pooled & per-chunk & per-chunk & \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r.epsilon:.1f} & {r.knee_idx_pooled:.2f} & {r.knee_idx_perchunk:.2f} & "
            f"{r.bitrate_saved_pct_pooled:.1f} & {r.bitrate_saved_pct_perchunk:.1f} & "
            f"{r.vmaf_cost_perchunk_shield:.2f} & {r.budget_violation_pct_pooled:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines) + "\n"
    for d in TABLES:
        d.mkdir(parents=True, exist_ok=True)
        (d / "table_perchunk_ablation.tex").write_text(body, encoding="utf-8")


def write_macros(head, inv_pct: float, n_chunks: int):
    def cmd(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    m = [
        r"% Auto-generated per-chunk ablation macros",
        cmd("PCnChunks", f"{n_chunks}"),
        cmd("PCinvPct", f"{inv_pct:.1f}"),
        cmd("PCeps", f"{EPS_HEADLINE:.1f}"),
        cmd("PCkneePooled", f"{head.knee_idx_pooled:.2f}"),
        cmd("PCkneePerchunk", f"{head.knee_idx_perchunk:.2f}"),
        cmd("PCbwPooled", f"{head.bitrate_saved_pct_pooled:.1f}"),
        cmd("PCbwPerchunk", f"{head.bitrate_saved_pct_perchunk:.1f}"),
        cmd("PCcostPerchunk", f"{head.vmaf_cost_perchunk_shield:.2f}"),
        cmd("PCcostPooled", f"{head.vmaf_cost_pooled_shield:.2f}"),
        cmd("PCviolPooled", f"{head.budget_violation_pct_pooled:.1f}"),
        cmd("PCdisagree", f"{head.decision_disagree_pct:.1f}"),
    ]
    body = "\n".join(m) + "\n"
    for d in TABLES:
        (d / "macros_perchunk.tex").write_text(body, encoding="utf-8")


def write_figure(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    x = np.arange(len(df))
    w = 0.38
    ax1.bar(x - w / 2, df.bitrate_saved_pct_pooled, w, label="pooled", color="#4C72B0")
    ax1.bar(x + w / 2, df.bitrate_saved_pct_perchunk, w, label="per-chunk", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{e:.1f}" for e in df.epsilon])
    ax1.set_xlabel(r"perceptual budget $\varepsilon$ (VMAF pts)")
    ax1.set_ylabel("bitrate saved (%)")
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(df.epsilon, df.vmaf_cost_perchunk_shield, "o-",
             label="per-chunk shield", color="#DD8452")
    ax2.plot(df.epsilon, df.vmaf_cost_pooled_shield, "s--",
             label="pooled shield", color="#4C72B0")
    ax2.plot(df.epsilon, df.epsilon, ":", color="gray", label=r"budget $\varepsilon$")
    ax2.set_xlabel(r"perceptual budget $\varepsilon$ (VMAF pts)")
    ax2.set_ylabel("realized per-chunk cost (pts)")
    ax2.legend(frameon=False, fontsize=8)

    for d in FIGURES:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_cps_perchunk.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
