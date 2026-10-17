"""Offline calibration / warm-start for Tube-v2 (train+val only, never test)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from configs.paths import get_paths
from configs.videos import EVAL_VIDEOS, MAX_CHUNKS
from src.environment.clean_abr_env import CleanABREnv
from src.training.conformal_tube_v2 import ConformalThroughputTubeV2

FsCalibMode = Literal["raw_samples", "env_chunks", "fixed_window"]
RefCalibPolicy = Literal["lowest", "rate_match"]

_P = get_paths()
_CALIB_VIDEO = EVAL_VIDEOS[0]
_RATE_MATCH_SAFETY = 0.85


def _reference_action(env: CleanABREnv, recent_tps: list[float], policy: RefCalibPolicy) -> int:
    if policy == "lowest":
        return 0
    if not recent_tps:
        return 0
    w = recent_tps[-5:]
    hm = float(len(w) / np.sum(1.0 / np.maximum(w, 1e-6)))
    safe = _RATE_MATCH_SAFETY * hm
    best = 0
    for i, br in enumerate(env.BITRATE_LEVELS):
        if float(br) <= safe:
            best = int(i)
    return best


def _build_calib_env(trace_dir: Path) -> CleanABREnv:
    return CleanABREnv(
        video_names=[_CALIB_VIDEO],
        trace_dir=str(trace_dir),
        vmaf_dir=str(_P["vmaf_scores"]),
        siti_dir=str(_P["content_features"]),
        max_chunks=MAX_CHUNKS,
        random_seed=0,
        use_lyapunov=True,
        use_future=True,
        reward_mode="log_utility",
        enable_early_abort=False,
    )


def chunk_effective_tps_from_trace_json(
    trace_data: dict,
    *,
    ref_policy: RefCalibPolicy = "rate_match",
) -> list[float]:
    """Replay one trace JSON through CleanABREnv; return per-chunk effective throughput (kbps)."""
    with tempfile.TemporaryDirectory(prefix="tube_fs_calib_") as td:
        path = Path(td) / "calib_trace.json"
        path.write_text(json.dumps(trace_data), encoding="utf-8")
        env = _build_calib_env(Path(td))
        rng = np.random.default_rng(0)
        _obs, _ = env.reset(seed=0, options={"rng": rng})
        recent: list[float] = []
        out: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = _reference_action(env, recent, ref_policy)
            _obs, _, terminated, truncated, info = env.step(action)
            tp = float(info.get("throughput", env.last_raw_throughput))
            if tp > 0:
                out.append(tp)
                recent.append(tp)
        return out


def fixed_window_throughputs_from_json(trace_data: dict, chunk_duration_s: float = 4.0) -> list[float]:
    raw = [float(x) for x in trace_data.get("throughput_kbps", []) if float(x) > 0]
    if not raw:
        return []
    dt = float(trace_data.get("sample_duration_s", trace_data.get("dt_s", 1.0)))
    steps = max(1, int(round(chunk_duration_s / max(dt, 1e-6))))
    out: list[float] = []
    for start in range(0, len(raw) - steps + 1, steps):
        window = raw[start : start + steps]
        out.append(float(np.mean(window)))
    return out


def replay_throughputs_on_tube(
    tube: ConformalThroughputTubeV2,
    trace_dirs: list[Path],
    *,
    fs_calib_mode: FsCalibMode = "raw_samples",
    ref_policy: RefCalibPolicy = "rate_match",
) -> int:
    """Warm-start tube scores from calibration traces (never test split)."""
    n_obs = 0
    for tdir in trace_dirs:
        if not tdir.is_dir():
            continue
        for path in sorted(tdir.glob("*.json")):
            if path.name.startswith("trace_stats"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            tube.begin_calibration_trace()
            if fs_calib_mode == "raw_samples":
                for actual in data.get("throughput_kbps", []):
                    tube.ingest_calibration_sample(float(actual))
                    n_obs += 1
            elif fs_calib_mode == "fixed_window":
                for actual in fixed_window_throughputs_from_json(data):
                    tube.ingest_calibration_sample(actual)
                    n_obs += 1
            elif fs_calib_mode == "env_chunks":
                for actual in chunk_effective_tps_from_trace_json(data, ref_policy=ref_policy):
                    tube.ingest_calibration_sample(actual)
                    n_obs += 1
            else:
                raise ValueError(f"unknown fs_calib_mode: {fs_calib_mode}")
            tube.reset_episode_state()
    return n_obs


__all__ = [
    "FsCalibMode",
    "RefCalibPolicy",
    "chunk_effective_tps_from_trace_json",
    "fixed_window_throughputs_from_json",
    "replay_throughputs_on_tube",
]
