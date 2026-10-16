"""Tube-v3: max-score simultaneous tube and split-objective MPC."""
from __future__ import annotations

import numpy as np
import pytest

from src.control.two_timescale_conformal import parse_tube_v2_arm, parse_tube_v3_arm
from src.training.conformal_tube_v2 import ConformalTubeV2Config
from src.training.conformal_tube_v3 import MaxScoreConformalTube


def _run_iid(alpha: float, n: int = 20_000, H: int = 5, seed: int = 3) -> MaxScoreConformalTube:
    rng = np.random.default_rng(seed)
    tube = MaxScoreConformalTube(
        ConformalTubeV2Config(alpha=alpha, horizon=H, window=500, cold_start_cap=0.5)
    )
    tp = rng.lognormal(7.0, 0.4, n)
    for t, a in enumerate(tp):
        tube.horizon_lower_bounds_kbps(H, decision_chunk=t)
        tube.observe(float(a), realized_chunk=t)
    return tube


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_maxscore_joint_coverage_iid(alpha: float) -> None:
    tube = _run_iid(alpha)
    flags = np.asarray(tube._joint_decision_flags[1000:], float)
    assert abs(flags.mean() - (1 - alpha)) <= 0.02, flags.mean()
    # per-step coverage must be at least the joint coverage
    assert np.all(tube.empirical_coverage_per_step() >= flags.mean() - 0.02)


def test_maxscore_bounds_are_pre_update() -> None:
    tube = MaxScoreConformalTube(ConformalTubeV2Config(alpha=0.1, horizon=3, cold_start_cap=0.5))
    rng = np.random.default_rng(0)
    for t in range(300):
        tube.horizon_lower_bounds_kbps(3, decision_chunk=t)
        tube.observe(float(rng.lognormal(7, 0.3)), realized_chunk=t)
    lb = tube.horizon_lower_bounds_kbps(3, decision_chunk=300)
    stored = tube._bounds_by_decision[300].copy()
    tube.observe(1.0, realized_chunk=300)  # extreme miss must not alter stored bounds
    assert np.allclose(tube._bounds_by_decision[300], stored)
    assert np.allclose(lb, stored)


def test_maxscore_warm_start_fills_joint_scores() -> None:
    tube = MaxScoreConformalTube(ConformalTubeV2Config(alpha=0.1, horizon=5))
    rng = np.random.default_rng(1)
    for a in rng.lognormal(7, 0.3, 400):
        tube.ingest_calibration_sample(float(a))
    assert len(tube._joint_scores) == tube.cfg.window
    assert tube._joint_ready()


def test_arm_parsing() -> None:
    assert parse_tube_v3_arm("tube_v2_step_static") is None
    assert parse_tube_v2_arm("tube_v2_step_static") is not None
    old = {"safe": False, "freeze": False}  # legacy arms keep their exact semantics
    assert parse_tube_v3_arm("tube_v2_step_split") == {"kind": "split", "alpha": None, **old}
    assert parse_tube_v3_arm("tube_v2_step_static_a0.2") == {"kind": "static", "alpha": 0.2, **old}
    assert parse_tube_v3_arm("tube_v2_maxscore_a0.1") == {"kind": "maxscore", "alpha": 0.1, **old}
    assert parse_tube_v2_arm("tube_v2_step_split") is None
    assert parse_tube_v3_arm("tube_v2_step_aci_a0.15") == {"kind": "aci", "alpha": 0.15, **old}
    assert parse_tube_v3_arm("tube_v2_step_aci") is None


class _FakeEnv:
    BITRATE_LEVELS = [300, 750, 1200, 1850, 2850, 4300]
    CHUNK_DURATION = 4.0
    BUFFER_MAX = 60.0

    def __init__(self, buf: float):
        self.buffer_level = buf
        self.chunk_idx = 10
        self.max_chunks = 48
        self.last_bitrate_idx = 2
        self.current_video_name = "v"
        self.last_raw_throughput = 0.0

    def get_chunk_size_bits(self, br: int, k: int) -> float:
        return float(br) * 1000.0 * self.CHUNK_DURATION


@pytest.mark.parametrize("buf", [2.0, 6.0, 15.0])
@pytest.mark.parametrize("tp", [400.0, 1500.0, 5000.0])
def test_split_reduces_to_static_when_forecast_equals_bound(buf: float, tp: float) -> None:
    from src.training.tube_mpc_policy_v2 import TubeV2MPCConfig, TubeV2MPCPolicy

    actions = []
    for split in (False, True):
        env = _FakeEnv(buf)
        pol = TubeV2MPCPolicy(env, TubeV2MPCConfig(horizon=4, pruning="monotone", split_objective=split))
        pol._last_video = "v"
        pol.tube.horizon_lower_bounds_kbps = lambda h, decision_chunk=None: np.full(h, tp)
        pol.tube.predict_point_kbps = lambda: tp
        actions.append(pol.plan_action())
    assert actions[0] == actions[1]


def test_split_is_less_conservative_with_optimistic_point() -> None:
    from src.training.tube_mpc_policy_v2 import TubeV2MPCConfig, TubeV2MPCPolicy

    acts = []
    for split in (False, True):
        env = _FakeEnv(20.0)
        pol = TubeV2MPCPolicy(env, TubeV2MPCConfig(horizon=4, pruning="monotone", split_objective=split))
        pol._last_video = "v"
        pol.tube.horizon_lower_bounds_kbps = lambda h, decision_chunk=None: np.full(h, 1000.0)
        pol.tube.predict_point_kbps = lambda: 3000.0
        acts.append(pol.plan_action())
    assert acts[1] >= acts[0]


@pytest.mark.parametrize("cls_name", ["v2", "maxscore"])
def test_horizon_alignment_pinned(cls_name: str) -> None:
    """Planner convention: decision j is made after chunk j-1 is observed; step h covers chunk j+h-1.

    Trace alternates 1000/100 kbps. Bounds are set to 900 for chunks the
    planner believes are 'high' and 90 for 'low'. Correct alignment => every
    step covered (coverage 1.0); off-by-one => all odd-step comparisons hit a
    900 bound against a 100 chunk => coverage 0.5 per step, joint 0.0.
    """
    from src.training.conformal_tube_v2 import ConformalThroughputTubeV2

    H = 4
    def val(c: int) -> float:
        return 1000.0 if c % 2 == 0 else 100.0

    def run(offset: int):
        cfg = ConformalTubeV2Config(alpha=0.1, horizon=H, legacy_step_offset=offset)
        tube = MaxScoreConformalTube(cfg) if cls_name == "maxscore" else ConformalThroughputTubeV2(cfg)
        if cls_name == "maxscore":
            tube._sigmas_by_decision = {}
        for j in range(200):
            b = np.array([0.9 * val(j + h - 1) for h in range(1, H + 1)])
            tube._hats_by_decision[j] = np.full(H, 1000.0)
            tube._bounds_by_decision[j] = b
            if cls_name == "maxscore":
                tube._sigmas_by_decision[j] = np.full(H, 0.5)
            tube.observe(val(j), realized_chunk=j)
        return tube.empirical_coverage_per_step(), tube.joint_coverage()

    cov, joint = run(0)
    assert np.allclose(cov, 1.0) and joint == 1.0
    cov_bad, joint_bad = run(1)
    assert np.allclose(cov_bad, 0.5, atol=0.02)
    assert joint_bad == 0.0
