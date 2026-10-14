#!/usr/bin/env python3
"""Exit 0 if Paper 2 data layout is present (for server CI / preflight)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paper2_release import release_banner
from configs.paths import (
    CONTENT_FEATURES_DIR,
    TEST_TRACES_5G_V18_DIR,
    TEST_TRACES_V18_DIR,
    VMAF_SCORES_DIR,
)

REQUIRED = [
    ("test_traces_5g_v18", TEST_TRACES_5G_V18_DIR),
    ("test_traces_v18", TEST_TRACES_V18_DIR),
    ("vmaf_scores", VMAF_SCORES_DIR),
]

def main() -> int:
    print(release_banner())
    ok = True
    for name, path in REQUIRED:
        exists = path.exists()
        print(f"  [{'OK' if exists else 'MISSING':7s}] {name}: {path}")
        ok = ok and exists
    vmaf_pc = VMAF_SCORES_DIR / "vmaf_perchunk.csv"
    if vmaf_pc.exists():
        print(f"  [OK     ] vmaf_perchunk.csv")
    else:
        print(f"  [WARN   ] vmaf_perchunk.csv missing (per-chunk knee falls back to env ladder)")
    if CONTENT_FEATURES_DIR.exists():
        print(f"  [OK     ] content_features/")
    if not ok:
        print("\nSet CPS_PAPER2_DATA_ROOT or symlink data/ — see docs/server_runs.md")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
