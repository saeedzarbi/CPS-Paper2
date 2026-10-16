"""Offline calibration / warm-start for Tube-v2 (train+val only, never test)."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.conformal_tube_v2 import ConformalThroughputTubeV2


def replay_throughputs_on_tube(tube: ConformalThroughputTubeV2, trace_dirs: list[Path]) -> int:
    """Append scores from trace JSONs without registering MPC decision bounds."""
    n_obs = 0
    for tdir in trace_dirs:
        if not tdir.is_dir():
            continue
        for path in sorted(tdir.glob("*.json")):
            if path.name.startswith("trace_stats"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tp = data.get("throughput_kbps", [])
            except (json.JSONDecodeError, OSError):
                continue
            for actual in tp:
                tube.ingest_calibration_sample(float(actual))
                n_obs += 1
            tube.reset_episode_state()
    return n_obs


__all__ = ["replay_throughputs_on_tube"]
