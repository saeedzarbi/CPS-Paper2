"""
Build a PER-CHUNK, per-bitrate VMAF ladder from the raw per-frame libvmaf output
(reviewer-response V14/V15).

Why
---
The whole "VMAF-awareness is inert" result stems from the pooled, per-VIDEO VMAF
ladder being monotone in the bitrate index. But the manuscript already claimed
per-chunk quality inversions exist (LadPctChunksInverted etc.). The raw libvmaf
JSONs (data/vmaf_scores/<video>/vmaf_<b>kbps.json) contain PER-FRAME VMAF, so we
can reconstruct the true per-chunk ladder and measure whether, for individual
chunks, a higher bitrate rung can yield LOWER VMAF (a quality inversion). Where
such inversions occur, "highest-feasible-index" != "highest-VMAF", so a
VMAF-aware shield is no longer equivalent to a content-blind one.

This script:
  1. aggregates per-frame VMAF into 4-second chunks (fps from SI/TI features),
  2. writes a tidy per-chunk ladder CSV,
  3. reports inversion statistics (the empirical justification for perceptual
     awareness).

Reads : data/vmaf_scores/<video>/vmaf_<bitrate>kbps.json
        data/content_features/<video>_siti.json   (for fps)
Writes: data/vmaf_scores/vmaf_perchunk.csv
        data/vmaf_scores/vmaf_perchunk_inversions.csv

Usage:
  cd new
  python data/build_perchunk_vmaf.py --chunk-seconds 4 --max-chunks 48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BITRATES = [300, 750, 1200, 1850, 2850, 6000]
DEFAULT_FPS = 24.0


def _load_fps(siti_path: Path) -> float:
    try:
        return float(json.loads(siti_path.read_text(encoding="utf-8")).get("fps", DEFAULT_FPS))
    except Exception:
        return DEFAULT_FPS


def _perchunk_for_bitrate(vmaf_json: Path, fps: float, chunk_seconds: float, max_chunks: int) -> dict[int, float]:
    data = json.loads(vmaf_json.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    frames_per_chunk = max(1, int(round(fps * chunk_seconds)))
    buckets: dict[int, list[float]] = {}
    for fr in frames:
        fn = int(fr.get("frameNum", 0))
        v = float(fr.get("metrics", {}).get("vmaf", np.nan))
        if np.isnan(v):
            continue
        c = fn // frames_per_chunk
        if c >= max_chunks:
            continue
        buckets.setdefault(c, []).append(v)
    return {c: float(np.mean(vs)) for c, vs in buckets.items() if vs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--max-chunks", type=int, default=48)
    parser.add_argument("--vmaf-dir", type=str, default="data/vmaf_scores")
    parser.add_argument("--siti-dir", type=str, default="data/content_features")
    args = parser.parse_args()

    vmaf_dir = Path(args.vmaf_dir)
    siti_dir = Path(args.siti_dir)
    videos = sorted([p.name for p in vmaf_dir.iterdir() if p.is_dir()])
    print(f"Videos: {videos}")

    rows = []
    for video in videos:
        fps = _load_fps(siti_dir / f"{video}_siti.json")
        for b in BITRATES:
            jf = vmaf_dir / video / f"vmaf_{b}kbps.json"
            if not jf.exists():
                print(f"[WARN] missing {jf}")
                continue
            perchunk = _perchunk_for_bitrate(jf, fps, args.chunk_seconds, args.max_chunks)
            for c, v in perchunk.items():
                rows.append({"video": video, "chunk": c, "bitrate_kbps": b, "vmaf": round(v, 4)})
        print(f"  {video}: fps={fps}")

    df = pd.DataFrame(rows).sort_values(["video", "chunk", "bitrate_kbps"]).reset_index(drop=True)
    out_csv = vmaf_dir / "vmaf_perchunk.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-chunk ladder -> {out_csv}  ({len(df)} rows)")

    # ---- Inversion analysis --------------------------------------------------
    inv_rows = []
    total_chunks = 0
    chunks_with_inversion = 0
    total_pairs = 0
    inverted_pairs = 0
    inv_magnitudes = []
    for (video, chunk), g in df.groupby(["video", "chunk"]):
        g = g.sort_values("bitrate_kbps")
        v = g["vmaf"].to_numpy(float)
        total_chunks += 1
        # adjacent-rung inversions: higher bitrate -> lower VMAF
        diffs = v[1:] - v[:-1]
        n_pairs = len(diffs)
        n_inv = int((diffs < 0).sum())
        total_pairs += n_pairs
        inverted_pairs += n_inv
        if n_inv > 0:
            chunks_with_inversion += 1
            inv_magnitudes.extend([float(-d) for d in diffs if d < 0])
        # "regret" of content-blind highest-index vs true argmax VMAF for this chunk
        blind_choice_vmaf = float(v[-1])          # highest bitrate index
        best_vmaf = float(v.max())                # true best VMAF
        regret = best_vmaf - blind_choice_vmaf    # >0 means highest index is NOT best
        inv_rows.append({
            "video": video, "chunk": int(chunk), "n_inverted_pairs": n_inv,
            "highest_index_vmaf": round(blind_choice_vmaf, 3),
            "best_vmaf": round(best_vmaf, 3),
            "blind_regret": round(regret, 3),
        })

    inv_df = pd.DataFrame(inv_rows)
    inv_csv = vmaf_dir / "vmaf_perchunk_inversions.csv"
    inv_df.to_csv(inv_csv, index=False)

    pct_chunks = 100.0 * chunks_with_inversion / max(1, total_chunks)
    pct_pairs = 100.0 * inverted_pairs / max(1, total_pairs)
    mean_mag = float(np.mean(inv_magnitudes)) if inv_magnitudes else 0.0
    chunks_blind_suboptimal = int((inv_df["blind_regret"] > 1e-6).sum())
    pct_blind_suboptimal = 100.0 * chunks_blind_suboptimal / max(1, len(inv_df))
    mean_regret = float(inv_df.loc[inv_df["blind_regret"] > 1e-6, "blind_regret"].mean()) if chunks_blind_suboptimal else 0.0

    print("\n================ PER-CHUNK INVERSION DIAGNOSTICS ================")
    print(f"total (video,chunk) cells        : {total_chunks}")
    print(f"chunks with >=1 rung inversion   : {chunks_with_inversion}  ({pct_chunks:.1f}%)")
    print(f"adjacent rung pairs inverted     : {inverted_pairs}/{total_pairs}  ({pct_pairs:.1f}%)")
    print(f"mean inversion magnitude (VMAF)  : {mean_mag:.2f}")
    print(f"chunks where highest-index is NOT best VMAF : {chunks_blind_suboptimal}  ({pct_blind_suboptimal:.1f}%)")
    print(f"mean VMAF regret on those chunks : {mean_regret:.2f}")
    print(f"\nSaved inversion table -> {inv_csv}")
    if pct_blind_suboptimal < 1.0:
        print("\n[VERDICT] Per-chunk ladder is still ~monotone; VMAF-awareness would "
              "remain (nearly) inert. Perceptual gains need a different lever.")
    else:
        print("\n[VERDICT] Real per-chunk inversions exist: a VMAF-aware selector can "
              "strictly beat highest-feasible-index on these chunks. Wiring per-chunk "
              "VMAF into the env/shield is justified.")


if __name__ == "__main__":
    main()
