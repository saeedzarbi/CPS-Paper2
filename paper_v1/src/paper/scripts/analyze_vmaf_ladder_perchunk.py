"""
Per-chunk VMAF ladder analysis.

Builds the chunk-level map V_k(j) from the per-frame VMAF logs on disk and
quantifies how far it departs from the per-video pooled ladder that the
evaluated shield actually uses.

Outputs
  new/src/paper/tables/macros_ladder.tex        headline numbers as LaTeX macros
  new/src/paper/tables/table_ladder_spacing.tex adjacent-rung spacing table
  new/src/paper/figures/fig_ladder_perchunk.pdf two-panel figure
  new/results/vmaf_ladder_perchunk.csv          the per-chunk ladder itself
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
    }
)

REPO = Path(__file__).resolve().parents[4]
NEW = REPO / "new"
sys.path.insert(0, str(NEW))
from configs.videos import DISPLAY_NAMES, resolve_slug  # noqa: E402

VMAF_DIR = NEW / "data" / "vmaf_scores"
SITI_DIR = NEW / "data" / "content_features"
TABLES = NEW / "src" / "paper" / "tables"
FIGURES = NEW / "src" / "paper" / "figures"
RESULTS = NEW / "results"

# Frame-level VMAF JSON on disk for §3 / fig:ladder only (excludes auxiliary crowd_run).
LADDER_FRAMELOG_SLUGS = ["bigbuckbunny", "tearsofsteel_short", "sintel"]
VIDEOS = list(LADDER_FRAMELOG_SLUGS)
VIDEO_LABEL = {
    slug: DISPLAY_NAMES.get(resolve_slug(slug), slug) for slug in VIDEOS
}
BITRATES = [300, 750, 1200, 1850, 2850, 6000]
CHUNK_DURATION = 4.0
# Adjacent rungs are treated as perceptually indistinguishable below this gap;
# 1.0 VMAF point is well inside the reported just-noticeable difference.
FLAT_GAP = 1.0


def frame_scores(video: str, bitrate: int) -> np.ndarray:
    path = VMAF_DIR / video / f"vmaf_{bitrate}kbps.json"
    data = json.loads(path.read_text())
    scores = [f["metrics"]["vmaf"] for f in data["frames"]]
    pooled = float(data["pooled_metrics"]["vmaf"]["mean"])
    return np.asarray(scores, dtype=float), pooled


def frames_per_chunk(video: str) -> int:
    meta = json.loads((SITI_DIR / f"{video}_siti.json").read_text())
    return int(round(float(meta["fps"]) * CHUNK_DURATION))


def build_ladders() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-chunk ladder, pooled ladder) in long form."""
    rows, pooled_rows = [], []
    for video in VIDEOS:
        fpc = frames_per_chunk(video)
        per_bitrate, n_chunks = {}, None
        for bitrate in BITRATES:
            scores, pooled = frame_scores(video, bitrate)
            usable = len(scores) // fpc
            n_chunks = usable if n_chunks is None else min(n_chunks, usable)
            per_bitrate[bitrate] = (scores, pooled)

        for bitrate, (scores, pooled) in per_bitrate.items():
            trimmed = scores[: n_chunks * fpc].reshape(n_chunks, fpc)
            chunk_means = trimmed.mean(axis=1)
            pooled_rows.append(
                {
                    "video": video,
                    "bitrate_kbps": bitrate,
                    "pooled_vmaf": pooled,
                    "frame_mean": float(scores.mean()),
                    "n_chunks": n_chunks,
                    "frames_per_chunk": fpc,
                }
            )
            for k, value in enumerate(chunk_means):
                rows.append(
                    {"video": video, "chunk": k, "bitrate_kbps": bitrate, "vmaf": float(value)}
                )
    return pd.DataFrame(rows), pd.DataFrame(pooled_rows)


def spacing_stats(ladder: pd.DataFrame) -> pd.DataFrame:
    """Adjacent-rung VMAF gaps, per video and rung pair, across chunks."""
    wide = ladder.pivot_table(index=["video", "chunk"], columns="bitrate_kbps", values="vmaf")
    records = []
    for lo, hi in zip(BITRATES[:-1], BITRATES[1:]):
        gaps = (wide[hi] - wide[lo]).groupby(level="video")
        for video, series in gaps:
            records.append(
                {
                    "video": video,
                    "pair": f"{lo}$\\to${hi}",
                    "lo": lo,
                    "hi": hi,
                    "mean": series.mean(),
                    "min": series.min(),
                    "max": series.max(),
                    "std": series.std(ddof=1) if len(series) > 1 else 0.0,
                    "n": len(series),
                    "n_inverted": int((series < 0).sum()),
                    "n_flat": int((series.abs() <= FLAT_GAP).sum()),
                }
            )
    return pd.DataFrame(records)


def monotonicity_stats(ladder: pd.DataFrame) -> dict:
    wide = ladder.pivot_table(index=["video", "chunk"], columns="bitrate_kbps", values="vmaf")
    values = wide[BITRATES].to_numpy()
    diffs = np.diff(values, axis=1)
    inverted_chunks = (diffs < 0).any(axis=1)
    flat_chunks = (np.abs(diffs) <= FLAT_GAP).any(axis=1)
    return {
        "n_chunks": int(len(values)),
        "n_pairs": int(diffs.size),
        "pct_chunks_inverted": 100.0 * inverted_chunks.mean(),
        "pct_chunks_with_flat_pair": 100.0 * flat_chunks.mean(),
        "pct_pairs_inverted": 100.0 * (diffs < 0).mean(),
        "pct_pairs_flat": 100.0 * (np.abs(diffs) <= FLAT_GAP).mean(),
    }


def pooled_gap_error(ladder: pd.DataFrame, pooled: pd.DataFrame) -> dict:
    """How wrong is the per-video gap as a predictor of the chunk-level gap?"""
    wide = ladder.pivot_table(index=["video", "chunk"], columns="bitrate_kbps", values="vmaf")
    pooled_wide = pooled.pivot_table(index="video", columns="bitrate_kbps", values="frame_mean")
    errors = []
    for lo, hi in zip(BITRATES[:-1], BITRATES[1:]):
        chunk_gap = wide[hi] - wide[lo]
        for video, series in chunk_gap.groupby(level="video"):
            reference = pooled_wide.loc[video, hi] - pooled_wide.loc[video, lo]
            errors.append(np.abs(series.to_numpy() - reference))
    stacked = np.concatenate(errors)
    return {
        "mad_mean": float(stacked.mean()),
        "mad_p90": float(np.percentile(stacked, 90)),
        "mad_max": float(stacked.max()),
    }


def make_figure(ladder: pd.DataFrame, pooled: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), layout="constrained")

    ax = axes[0]
    wide = ladder.pivot_table(index=["video", "chunk"], columns="bitrate_kbps", values="vmaf")
    pooled_wide = pooled.pivot_table(index="video", columns="bitrate_kbps", values="frame_mean")
    x = np.arange(len(BITRATES))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, video in enumerate(VIDEOS):
        block = wide.loc[video]
        ax.fill_between(
            x,
            block[BITRATES].min(axis=0).to_numpy(),
            block[BITRATES].max(axis=0).to_numpy(),
            alpha=0.20,
            color=colors[idx],
            linewidth=0,
        )
        ax.plot(
            x,
            pooled_wide.loc[video, BITRATES].to_numpy(),
            marker="o",
            markersize=5,
            color=colors[idx],
            label=VIDEO_LABEL[video],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BITRATES], rotation=30)
    ax.set_xlabel("Representation bitrate (kb/s)")
    ax.set_ylabel("VMAF")

    ax = axes[1]
    gaps, labels, positions = [], [], []
    for pos, (lo, hi) in enumerate(zip(BITRATES[:-1], BITRATES[1:])):
        series = (wide[hi] - wide[lo]).to_numpy()
        gaps.append(series)
        labels.append(f"{lo}\u2192{hi}")
        positions.append(pos)
    parts = ax.boxplot(gaps, positions=positions, widths=0.6, showfliers=True,
                       patch_artist=True, medianprops={"color": "black"})
    for patch in parts["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.8)
    ax.axhline(0.0, color="crimson", linewidth=1.2, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_xlabel("Adjacent representation pair (kb/s)")
    ax.set_ylabel("Per-chunk VMAF gap")

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.savefig(path)
    plt.close(fig)


def write_macros(mono: dict, gap_err: dict, spacing: pd.DataFrame, path: Path) -> None:
    widest = spacing.loc[spacing["max"].sub(spacing["min"]).idxmax()]
    cheapest = spacing.loc[spacing["mean"].idxmin()]
    lines = [
        "% Generated by analyze_vmaf_ladder_perchunk.py --- do not edit by hand.",
        _macro("LadNChunks", f"{mono['n_chunks']}"),
        _macro("LadNPairs", f"{mono['n_pairs']}"),
        _macro("LadPctChunksInverted", f"{mono['pct_chunks_inverted']:.1f}"),
        _macro("LadPctChunksFlat", f"{mono['pct_chunks_with_flat_pair']:.1f}"),
        _macro("LadPctPairsInverted", f"{mono['pct_pairs_inverted']:.1f}"),
        _macro("LadPctPairsFlat", f"{mono['pct_pairs_flat']:.1f}"),
        _macro("LadGapErrMean", f"{gap_err['mad_mean']:.2f}"),
        _macro("LadGapErrPNinety", f"{gap_err['mad_p90']:.2f}"),
        _macro("LadGapErrMax", f"{gap_err['mad_max']:.2f}"),
        _macro("LadFlatThresh", f"{FLAT_GAP:.1f}"),
        _macro("LadWidestVideo", VIDEO_LABEL[widest["video"]]),
        _macro("LadWidestPair", f"{widest['lo']}\\to{widest['hi']}"),
        _macro("LadWidestMin", f"{widest['min']:.2f}"),
        _macro("LadWidestMax", f"{widest['max']:.2f}"),
        _macro("LadCheapVideo", VIDEO_LABEL[cheapest["video"]]),
        _macro("LadCheapPair", f"{cheapest['lo']}\\to{cheapest['hi']}"),
        _macro("LadCheapMean", f"{cheapest['mean']:.2f}"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _macro(name: str, value: str) -> str:
    return f"\\providecommand{{\\{name}}}{{{value}}}"


def write_spacing_table(ladder: pd.DataFrame, spacing: pd.DataFrame, path: Path) -> None:
    wide = ladder.pivot_table(index=["video", "chunk"], columns="bitrate_kbps", values="vmaf")
    head = [
        "% Generated by analyze_vmaf_ladder_perchunk.py --- do not edit by hand.",
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xccccc@{}}",
        "\\toprule",
        "\\textbf{Video} & \\textbf{Chunks} & \\textbf{Inv.\\ chunks (\\%)} & "
        "\\textbf{Flat pairs (\\%)} & \\textbf{Widest pair (kb/s)} & \\textbf{Gap range} \\\\",
        "\\midrule",
    ]
    body = []
    for video in VIDEOS:
        block = spacing[spacing.video == video]
        widest = block.loc[block["max"].sub(block["min"]).idxmax()]
        diffs = np.diff(wide.loc[video][BITRATES].to_numpy(), axis=1)
        inverted_pct = 100.0 * (diffs < 0).any(axis=1).mean()
        flat_pct = 100.0 * block["n_flat"].sum() / block["n"].sum()
        body.append(
            f"{VIDEO_LABEL[video]} & {int(widest['n'])} & {inverted_pct:.1f} & "
            f"{flat_pct:.1f} & {int(widest['lo'])}$\\to${int(widest['hi'])} & "
            f"[{widest['min']:.2f}, {widest['max']:.2f}] \\\\"
        )
    tail = ["\\bottomrule", "\\end{tabularx}"]
    path.write_text("\n".join(head + body + tail) + "\n", encoding="utf-8")


def main() -> None:
    ladder, pooled = build_ladders()

    # The JSON frame list covers a prefix of each encode while pooled_metrics
    # covers the full clip, so the two per-video references differ slightly.
    # Every statistic below is computed from the frame list alone.
    drift = (pooled.pooled_vmaf - pooled.frame_mean).abs().max()
    print(f"note: max |pooled_metrics.mean - frame mean| = {drift:.2f} VMAF points")

    mono = monotonicity_stats(ladder)
    gap_err = pooled_gap_error(ladder, pooled)
    spacing = spacing_stats(ladder)

    print("\nchunks per video:")
    print(pooled.groupby("video").n_chunks.first().to_string())
    print("\nmonotonicity:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in mono.items()})
    print("pooled-gap error:", {k: round(v, 2) for k, v in gap_err.items()})
    print("\nadjacent-rung spacing:")
    print(spacing.drop(columns=["pair"]).to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    ladder.to_csv(RESULTS / "vmaf_ladder_perchunk.csv", index=False)
    write_macros(mono, gap_err, spacing, TABLES / "macros_ladder.tex")
    write_spacing_table(ladder, spacing, TABLES / "table_ladder_spacing.tex")
    make_figure(ladder, pooled, FIGURES / "fig_ladder_perchunk.pdf")
    print("\nwrote macros_ladder.tex, table_ladder_spacing.tex, fig_ladder_perchunk.pdf")


if __name__ == "__main__":
    main()
