"""
Build a genuinely NON-MONOTONE, per-chunk VMAF ladder via multi-resolution
encoding + libvmaf. This is the only legitimate way to make the VMAF-aware shield
non-inert: on a single-resolution ladder VMAF is monotone in the bitrate index,
so "highest-feasible-VMAF" == "highest-feasible-index" (proven, and confirmed
empirically: gain is exactly 0 at every realistic downgrade depth).

Why multi-resolution creates real inversions
--------------------------------------------
Each rung is encoded at a specific (bitrate, resolution). VMAF is then computed
against the full-resolution reference (both scaled to 1080p, exactly like the
project's vmaf_calculator). At LOW bitrates a LOWER-resolution encode looks
BETTER than a higher-resolution one at similar bitrate (fewer compression
artifacts once upscaled); at HIGH bitrates the higher resolution wins. With a
fixed per-title bitrate->resolution ladder (as real HLS/DASH deployments use),
the per-chunk quality ordering therefore FLIPS on some scenes -- a real,
content-dependent, perceptually meaningful crossover (the convex-hull effect).

Inputs (same layout as vmaf_calculator.py):
  raw_videos/{video}.mp4                      (reference, native resolution)

Outputs (written under --out-dir, default data/vmaf_scores):
  vmaf_perchunk_multires.csv                  (video,chunk,bitrate_kbps,vmaf)
  vmaf_ladder_multires.json                   (index -> {bitrate, width, height})
  vmaf_perchunk_multires_inversions.csv       (per-chunk inversion diagnostics)
  encoded_multires/{video}/{video}_{br}kbps_{H}p.mp4 (kept for audit)

Requires: ffmpeg built with libx264 and libvmaf, plus ffprobe.

Usage (on the server, after installing ffmpeg+libvmaf):
  cd new
  python data/build_multires_vmaf.py --raw-dir raw_videos \
      --videos bigbuckbunny,crowd_run,tearsofsteel_short,sintel
  # inspect: it PRINTS how many chunks are non-monotone and the max/mean gain a
  # VMAF-aware shield could obtain. If that is ~0, do NOT bother training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.videos import EVAL_VIDEOS_CSV  # noqa: E402

CHUNK_SECONDS = 4.0
BITRATES = [300, 750, 1200, 1850, 2850, 6000]

# Per-title bitrate -> resolution ladder (realistic convex-hull-style assignment).
# Heights map to 16:9 widths. This fixed assignment is optimal on AVERAGE but not
# per scene, which is exactly what produces honest per-chunk crossovers.
DEFAULT_LADDER = {
    300:  (384, 216),
    750:  (512, 288),
    1200: (768, 432),
    1850: (1024, 576),
    2850: (1280, 720),
    6000: (1920, 1080),
}

VMAF_REF_W, VMAF_REF_H = 1920, 1080


def _check_ffmpeg() -> bool:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("[ERROR] ffmpeg/ffprobe not found on PATH.")
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True, check=True).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] could not run ffmpeg: {exc}")
        return False
    if "libvmaf" not in out:
        print("[ERROR] this ffmpeg has no libvmaf filter. Install an ffmpeg built with --enable-libvmaf.")
        return False
    return True


def _probe_res(path: Path) -> tuple[int, int]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "0", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception:
        return VMAF_REF_W, VMAF_REF_H


def _cap_ladder(ladder: dict, native_w: int, native_h: int) -> dict:
    """Never upscale: cap each rung's height to the source height, keeping even,
    16:9-consistent dimensions."""
    capped = {}
    for br, (w, h) in ladder.items():
        if h > native_h:
            w, h = native_w, native_h
        capped[br] = (int(w) - int(w) % 2, int(h) - int(h) % 2)
    return capped


def _probe_fps(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "0", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        num, den = out.split("/") if "/" in out else (out, "1")
        fps = float(num) / float(den)
        return fps if fps > 1e-3 else 24.0
    except Exception:
        return 24.0


def _encode(ref: Path, out: Path, br: int, w: int, h: int, fps: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    gop = max(1, int(round(fps * CHUNK_SECONDS)))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(ref),
        "-an", "-c:v", "libx264", "-preset", "medium",
        "-b:v", f"{br}k", "-maxrate", f"{int(br * 1.2)}k", "-bufsize", f"{int(br * 2)}k",
        "-vf", f"scale={w}:{h}:flags=bicubic",
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def _perframe_vmaf(ref: Path, dist: Path, log_json: Path, threads: int) -> list[float]:
    log_json.parent.mkdir(parents=True, exist_ok=True)
    vfilter = (
        f"[0:v]scale={VMAF_REF_W}:{VMAF_REF_H}:flags=bicubic,setpts=PTS-STARTPTS[dist];"
        f"[1:v]scale={VMAF_REF_W}:{VMAF_REF_H}:flags=bicubic,setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={log_json}:n_threads={threads}"
    )
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(dist), "-i", str(ref), "-filter_complex", vfilter, "-f", "null", "-"]
    subprocess.run(cmd, check=True)
    data = json.loads(log_json.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    vals = []
    for fr in frames:
        m = fr.get("metrics", fr)
        v = m.get("vmaf")
        if v is not None:
            vals.append(float(v))
    return vals


def _chunk_means(perframe: list[float], fps: float) -> list[float]:
    fpc = max(1, int(round(fps * CHUNK_SECONDS)))
    arr = np.asarray(perframe, dtype=float)
    n_chunks = int(np.ceil(len(arr) / fpc)) if arr.size else 0
    return [float(arr[c * fpc:(c + 1) * fpc].mean()) for c in range(n_chunks)]


def build_for_video(video: str, ref: Path, ladder: dict, enc_dir: Path,
                    log_dir: Path, threads: int) -> list[dict]:
    fps = _probe_fps(ref)
    nat_w, nat_h = _probe_res(ref)
    ladder = _cap_ladder(ladder, nat_w, nat_h)
    print(f"\n=== {video} (fps={fps:.3f}, native={nat_w}x{nat_h}) ref={ref.name} ===")
    print(f"    effective ladder: {[(b, ladder[b]) for b in BITRATES]}")
    rows = []
    for br in BITRATES:
        w, h = ladder[br]
        enc = enc_dir / video / f"{video}_{br}kbps_{h}p.mp4"
        log = log_dir / video / f"vmaf_{br}kbps_{h}p.json"
        if not enc.exists():
            print(f"  encode {br:>4}kbps @ {w}x{h} ...", flush=True)
            _encode(ref, enc, br, w, h, fps)
        print(f"  vmaf   {br:>4}kbps @ {w}x{h} ...", flush=True)
        perframe = _perframe_vmaf(ref, enc, log, threads)
        chunks = _chunk_means(perframe, fps)
        for c, v in enumerate(chunks):
            rows.append({"video": video, "chunk": c, "bitrate_kbps": br, "vmaf": round(v, 4)})
        print(f"    -> {len(chunks)} chunks, mean VMAF {np.mean(chunks):.2f}")
    return rows


def inversion_report(df: pd.DataFrame) -> pd.DataFrame:
    """For each (video,chunk) and each highest-feasible index k, VMAF-aware gain =
    max VMAF over rungs 0..k  minus  VMAF of rung k. This is exactly the quantity
    the VMAF-aware shield can exploit."""
    idx_of = {b: i for i, b in enumerate(BITRATES)}
    out_rows = []
    per_depth = {k: [] for k in range(len(BITRATES))}
    for (vid, ch), g in df.groupby(["video", "chunk"]):
        g = g.sort_values("bitrate_kbps")
        v = g.set_index("bitrate_kbps")["vmaf"].reindex(BITRATES).to_numpy()
        if np.isnan(v).any():
            continue
        for k in range(len(BITRATES)):
            gain = float(v[: k + 1].max() - v[k])
            per_depth[k].append(gain)
            if gain > 1e-6:
                out_rows.append({"video": vid, "chunk": int(ch), "k": k,
                                 "bitrate": BITRATES[k], "index_vmaf": round(float(v[k]), 3),
                                 "aware_vmaf": round(float(v[: k + 1].max()), 3),
                                 "gain": round(gain, 3)})
    print("\n=== VMAF-aware minus index gain, by highest-feasible index k ===")
    print(f"{'k':>3} {'bitrate':>8} {'n':>5} {'n_gain>0':>9} {'mean_gain':>10} {'max_gain':>9}")
    for k in range(len(BITRATES)):
        a = np.array(per_depth[k]) if per_depth[k] else np.array([0.0])
        print(f"{k:>3} {BITRATES[k]:>8} {a.size:>5} {(a > 1e-6).sum():>9} {a.mean():>10.4f} {a.max():>9.4f}")
    low = np.concatenate([np.array(per_depth[k]) for k in range(3)]) if any(per_depth[k] for k in range(3)) else np.array([0.0])
    print(f"\nDeep downgrades (k<=2, the stress regime): n={low.size} mean={low.mean():.4f} max={low.max():.4f}")
    print("If the numbers above are ~0, the multi-res ladder did not create usable")
    print("crossovers on this content and the VMAF-aware claim remains unsupportable.")
    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser(description="Build non-monotone per-chunk VMAF ladder (multi-resolution).")
    ap.add_argument("--raw-dir", type=str, default="data/raw_videos")
    ap.add_argument("--videos", type=str, default=EVAL_VIDEOS_CSV)
    ap.add_argument("--out-dir", type=str, default=str(Path("data") / "vmaf_scores"))
    ap.add_argument("--enc-dir", type=str, default=str(Path("data") / "encoded_multires"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    if not _check_ffmpeg():
        sys.exit(2)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_dir = Path(args.enc_dir)
    log_dir = out_dir / "_multires_logs"

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    all_rows = []
    for video in videos:
        ref = None
        for ext in (".mp4", ".y4m", ".mkv", ".webm", ".mov"):
            cand = raw_dir / f"{video}{ext}"
            if cand.exists():
                ref = cand
                break
        if ref is None:
            print(f"[WARN] reference for '{video}' not found in {raw_dir}; skipping.")
            continue
        all_rows += build_for_video(video, ref, DEFAULT_LADDER, enc_dir, log_dir, args.threads)

    if not all_rows:
        print("[ERROR] no rows produced; check --raw-dir and video names.")
        sys.exit(1)

    df = pd.DataFrame(all_rows).sort_values(["video", "chunk", "bitrate_kbps"]).reset_index(drop=True)
    perchunk_csv = out_dir / "vmaf_perchunk_multires.csv"
    df.to_csv(perchunk_csv, index=False)
    print(f"\nSaved per-chunk multires ladder -> {perchunk_csv}")

    ladder_json = out_dir / "vmaf_ladder_multires.json"
    ladder_json.write_text(json.dumps(
        {str(i): {"bitrate": b, "width": DEFAULT_LADDER[b][0], "height": DEFAULT_LADDER[b][1]}
         for i, b in enumerate(BITRATES)}, indent=2), encoding="utf-8")
    print(f"Saved ladder definition        -> {ladder_json}")

    inv = inversion_report(df)
    inv_csv = out_dir / "vmaf_perchunk_multires_inversions.csv"
    inv.to_csv(inv_csv, index=False)
    print(f"Saved inversion diagnostics    -> {inv_csv}")


if __name__ == "__main__":
    main()
