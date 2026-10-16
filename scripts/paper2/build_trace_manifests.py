#!/usr/bin/env python3
"""Build / verify canonical trace manifests (sha256 per JSON trace file)."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.paths import (
    TEST_TRACES_5G_IRISH_MMSYS20_DIR,
    TEST_TRACES_5G_MARKOV_DIR,
    TEST_TRACES_V18_DIR,
    TRAIN_TRACES_5G_IRISH_MMSYS20_DIR,
    TRAIN_TRACES_5G_MARKOV_DIR,
    TRAIN_TRACES_V18_DIR,
)
from configs.trace_sets_v2 import (
    HARSH_CALIB_FRACTION,
    HARSH_FAMILY_SEED,
    HARSH_SOURCE_DIR,
    HARSH_TEST_FRACTION,
    HARSH_VAL_FRACTION,
    VAL_FRACTION,
    VAL_SEED,
    eval_test_dir,
    harsh_family_three_way_dirs,
    train_minus_val_dir,
    val_trace_dir,
)
from src.evaluation.trace_provenance import (
    CANONICAL_MANIFEST,
    manifest_entries_for_dir,
    verify_canonical_manifest,
)

MARKOV_SEED = 4242
N_TRAIN = 24
N_TEST = 12
TRACE_LEN = 1200

MARKOV_GEN = {
    "script": "scripts/paper2/gen_markov_traces.py",
    "seed": MARKOV_SEED,
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "trace_len": TRACE_LEN,
    "json": "sort_keys=True,separators=(',', ':')",
}


def _collect_directories() -> list[tuple[str, Path, dict | None]]:
    dirs: list[tuple[str, Path, dict | None]] = [
        ("train_traces_5g_markov", TRAIN_TRACES_5G_MARKOV_DIR, MARKOV_GEN),
        ("test_traces_5g_markov", TEST_TRACES_5G_MARKOV_DIR, MARKOV_GEN),
        ("train_traces_v18", TRAIN_TRACES_V18_DIR, None),
        ("test_traces_v18", TEST_TRACES_V18_DIR, None),
        ("test_traces_5g_harsh_v15", HARSH_SOURCE_DIR, None),
    ]
    if TRAIN_TRACES_5G_IRISH_MMSYS20_DIR.is_dir():
        dirs.append(("train_traces_5g_irish_mmsys20", TRAIN_TRACES_5G_IRISH_MMSYS20_DIR, None))
    if TEST_TRACES_5G_IRISH_MMSYS20_DIR.is_dir():
        dirs.append(("test_traces_5g_irish_mmsys20", TEST_TRACES_5G_IRISH_MMSYS20_DIR, None))
    for key in ("5g_markov", "5g_irish"):
        try:
            dirs.append((f"split_{key}_val", val_trace_dir(key), {"seed": VAL_SEED, "fraction": VAL_FRACTION}))
            dirs.append(
                (
                    f"split_{key}_train_minus_val",
                    train_minus_val_dir(key),
                    {"seed": VAL_SEED, "fraction": VAL_FRACTION},
                )
            )
        except (FileNotFoundError, KeyError):
            pass
    try:
        calib, val, test = harsh_family_three_way_dirs()
        dirs.append(
            (
                "split_5g_harsh_calib",
                calib,
                {
                    "seed": HARSH_FAMILY_SEED,
                    "fractions": {
                        "calib": HARSH_CALIB_FRACTION,
                        "val": HARSH_VAL_FRACTION,
                        "test": HARSH_TEST_FRACTION,
                    },
                },
            )
        )
        dirs.append(
            (
                "split_5g_harsh_val",
                val,
                {
                    "seed": HARSH_FAMILY_SEED,
                    "fractions": {
                        "calib": HARSH_CALIB_FRACTION,
                        "val": HARSH_VAL_FRACTION,
                        "test": HARSH_TEST_FRACTION,
                    },
                },
            )
        )
        dirs.append(
            (
                "split_5g_harsh_test",
                test,
                {
                    "seed": HARSH_FAMILY_SEED,
                    "fractions": {
                        "calib": HARSH_CALIB_FRACTION,
                        "val": HARSH_VAL_FRACTION,
                        "test": HARSH_TEST_FRACTION,
                    },
                },
            )
        )
    except FileNotFoundError:
        pass
    for key in ("5g_markov", "bb", "5g_irish"):
        try:
            test_eval = eval_test_dir(key)
            dirs.append((f"split_{key}_test_eval", test_eval, {"basename_disjoint_from_train": True}))
        except (FileNotFoundError, KeyError):
            pass
    return dirs


def build_environment_metadata() -> dict:
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = ""
    return {
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": git_commit,
    }


def build_manifest() -> dict:
    entries = []
    for logical, path, gen in _collect_directories():
        entries.append(manifest_entries_for_dir(path, logical_name=logical, generator=gen))
    return {
        "version": 2,
        "build": build_environment_metadata(),
        "directories": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Write configs/trace_manifests/canonical_traces.json")
    ap.add_argument("--verify", action="store_true", help="Verify local dirs against canonical manifest")
    args = ap.parse_args()
    if args.write:
        payload = build_manifest()
        CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        CANONICAL_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n_files = sum(len(d.get("files", {})) for d in payload["directories"])
        print(f"Wrote {CANONICAL_MANIFEST} ({n_files} trace files hashed)")
        return 0
    if args.verify:
        errors = verify_canonical_manifest()
        if errors:
            print("VERIFY FAILED:")
            for e in errors:
                print(" ", e)
            return 2
        print("VERIFY OK")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
