#!/usr/bin/env python3
"""
Convert the Irish commercial 5G production traces (MMSys 2020) to env JSON.

Source (open, ~2 MB):
  https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip

Citation:
  Raca et al., "Beyond Throughput, The Next Generation: A 5G Dataset with
  Channel and Context Metrics," MMSys 2020.

Each output file: {"throughput_kbps": [...]} at 1 Hz from DL_bitrate (G-NetTrack).
Zeros are treated as missing and forward-filled within the session.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paths import STANDARDIZED_DIR

DEFAULT_SRC = ROOT / "data" / "external" / "5g_irish_mmsys2020" / "5G-production-dataset"
OUT_TEST = STANDARDIZED_DIR / "test_traces_5g_irish_mmsys20"
OUT_TRAIN = STANDARDIZED_DIR / "train_traces_5g_irish_mmsys20"
MIN_SECONDS = 200  # 48 chunks * 4 s = 192 s playback + margin
SPLIT_SEED = 2026
TEST_FRACTION = 0.25


def _session_key(path: Path) -> str:
    """Group by scenario (app / mobility / show) for disjoint train-test split."""
    parts = path.parts
    try:
        idx = parts.index("5G-production-dataset")
        rel = parts[idx + 1 :]
    except ValueError:
        rel = path.parts[-4:]
    return "/".join(rel[:-1])


def _csv_to_trace(csv_path: Path) -> np.ndarray | None:
    df = pd.read_csv(csv_path)
    if "DL_bitrate" not in df.columns or "Timestamp" not in df.columns:
        return None
    if "NetworkMode" in df.columns:
        df = df[df["NetworkMode"].astype(str).str.upper() == "5G"]
        if df.empty:
            return None
    df["ts"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return None
    dl = pd.to_numeric(df["DL_bitrate"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df.assign(dl=dl)
    sec = df.groupby(df["ts"].dt.floor("s"))["dl"].max()
    tp = sec.replace(0.0, np.nan).ffill().bfill().to_numpy(dtype=float)
    if len(tp) < MIN_SECONDS:
        return None
    tp = np.clip(tp, 80.0, 500_000.0)
    return tp


def _safe_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(rel.with_suffix("")))
    return stem[:180]


def build(
    src: Path,
    test_dir: Path,
    train_dir: Path,
    test_fraction: float,
    all_test: bool = False,
) -> dict:
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if train_dir.exists():
        shutil.rmtree(train_dir)
    test_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)

    csv_files = sorted(src.rglob("*.csv"))
    by_group: dict[str, list[Path]] = {}
    for f in csv_files:
        by_group.setdefault(_session_key(f), []).append(f)

    groups = sorted(by_group)
    if all_test:
        test_groups = set(groups)
    else:
        rng = np.random.default_rng(SPLIT_SEED)
        rng.shuffle(groups)
        n_test_g = max(1, int(round(len(groups) * test_fraction)))
        test_groups = set(groups[:n_test_g])

    stats = {"train": 0, "test": 0, "skipped_short": 0, "skipped_parse": 0}
    meta_sessions = []

    for g in groups:
        side = "test" if g in test_groups else "train"
        out_root = test_dir if side == "test" else train_dir
        for csv_path in by_group[g]:
            tp = _csv_to_trace(csv_path)
            if tp is None:
                if csv_path.stat().st_size > 100:
                    stats["skipped_short"] += 1
                else:
                    stats["skipped_parse"] += 1
                continue
            name = _safe_name(csv_path, src) + ".json"
            payload = {
                "throughput_kbps": [float(x) for x in tp],
                "provenance": {
                    "source": "uccmisl/5Gdataset",
                    "file": str(csv_path.relative_to(src)),
                    "column": "DL_bitrate",
                    "resample_hz": 1,
                    "network_mode_filter": "5G",
                },
            }
            (out_root / name).write_text(json.dumps(payload), encoding="utf-8")
            stats[side] += 1
            meta_sessions.append({"group": g, "side": side, "file": name, "seconds": len(tp)})

    summary = {
        "dataset": "irish_5g_mmsys2020",
        "source_zip": "https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip",
        "citation": "Raca et al., MMSys 2020",
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "stats": stats,
        "sessions": meta_sessions,
    }
    (test_dir.parent / "trace_stats_5g_irish_mmsys20.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Extracted 5G-production-dataset/")
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument(
        "--all-test",
        action="store_true",
        help="Put every valid session in test/ (for eval campaigns; train/ empty).",
    )
    args = ap.parse_args()
    if not args.src.is_dir():
        print(f"Missing source tree: {args.src}", file=sys.stderr)
        print("Download:", "https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip")
        return 1
    summary = build(args.src, OUT_TEST, OUT_TRAIN, args.test_fraction, all_test=args.all_test)
    print(json.dumps(summary["stats"], indent=2))
    print(f"Wrote test  -> {OUT_TEST} ({summary['stats']['test']} traces)")
    print(f"Wrote train -> {OUT_TRAIN} ({summary['stats']['train']} traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
