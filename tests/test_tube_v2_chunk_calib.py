"""Offline _fs calibration modes (raw vs env chunk replay)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from src.training.conformal_tube_v2 import ConformalTubeV2Config, ConformalThroughputTubeV2
from src.training.conformal_tube_v3 import MaxScoreConformalTube
from src.training.tube_v2_calib import (
    chunk_effective_tps_from_trace_json,
    fixed_window_throughputs_from_json,
    replay_throughputs_on_tube,
)


def _piecewise_bb_trace() -> dict:
    """Fast raw ticks on plateaus; chunk downloads span multiple plateaus in the env."""
    tps: list[float] = []
    for level in (8000.0, 2000.0, 6000.0, 1500.0):
        tps.extend([level] * 24)  # 12 s at 0.5 s per sample
    return {"throughput_kbps": tps, "sample_duration_s": 0.5}


def test_begin_calibration_trace_clears_forecaster() -> None:
    tube = ConformalThroughputTubeV2(ConformalTubeV2Config(horizon=3))
    for x in (5000.0, 5000.0, 5000.0):
        tube.ingest_calibration_sample(x)
    assert len(tube._obs) == 3
    tube.begin_calibration_trace()
    assert len(tube._obs) == 0
    tube.ingest_calibration_sample(100.0)
    hat = tube.predict_point_kbps()
    assert abs(hat - 100.0) < 1.0


def test_env_chunks_shorter_series_than_raw_ticks() -> None:
    data = _piecewise_bb_trace()
    raw_n = len(data["throughput_kbps"])
    chunk_n = len(chunk_effective_tps_from_trace_json(data))
    assert chunk_n > 0
    assert chunk_n <= raw_n // 2


def test_fixed_window_coarser_than_raw() -> None:
    data = _piecewise_bb_trace()
    assert len(fixed_window_throughputs_from_json(data)) < len(data["throughput_kbps"])


def test_fs_finalize_differs_raw_vs_env_chunks() -> None:
    data = _piecewise_bb_trace()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        t_raw = MaxScoreConformalTube(
            ConformalTubeV2Config(alpha=0.15, horizon=5, freeze_scales=True)
        )
        t_env = MaxScoreConformalTube(
            ConformalTubeV2Config(alpha=0.15, horizon=5, freeze_scales=True)
        )
        replay_throughputs_on_tube(t_raw, [Path(td)], fs_calib_mode="raw_samples")
        replay_throughputs_on_tube(t_env, [Path(td)], fs_calib_mode="env_chunks")
        t_raw.finalize_frozen_calibration()
        t_env.finalize_frozen_calibration()
        sig_r = np.asarray(t_raw._frozen_sigma, float)
        sig_e = np.asarray(t_env._frozen_sigma, float)
        assert sig_e.shape == sig_r.shape
        assert float(np.max(sig_e)) >= float(np.max(sig_r)) * 0.5
