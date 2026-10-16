"""Tube-v3 adaptive: pre-refill safe planner (_safe) and frozen split calibration (_fs)."""
from __future__ import annotations

import numpy as np
import pytest

from src.control.two_timescale_conformal import parse_tube_v3_arm
from src.training.conformal_tube_v2 import ConformalThroughputTubeV2, ConformalTubeV2Config
from src.training.conformal_tube_v3 import MaxScoreConformalTube
from src.training.tube_mpc_policy_v2 import TubeV2MPCConfig, TubeV2MPCPolicy


class _Env:
    BITRATE_LEVELS = [300, 750, 1200, 1850, 2850, 4300]
    CHUNK_DURATION = 4.0
    BUFFER_MAX = 12.0

    def __init__(self, buf: float, n: int):
        self.buffer_level = buf
        self.chunk_idx = 0
        self.max_chunks = n
        self.last_bitrate_idx = 0
        self.current_video_name = "v"
        self.last_raw_throughput = 0.0

    def get_chunk_size_bits(self, br: int, k: int) -> float:
        return float(br) * 1000.0 * self.CHUNK_DURATION


def _closed_loop(safe: bool, reb_penalty: float, seed: int = 0, n: int = 60) -> tuple[float, list[float]]:
    """Real buffer dynamics; tube bounds are 0.8*actual (perfect coverage: actual >= bound)."""
    rng = np.random.default_rng(seed)
    y = rng.choice([500.0, 1500.0, 4000.0, 900.0], size=n + 10)
    env = _Env(buf=4.0, n=n)
    pol = TubeV2MPCPolicy(env, TubeV2MPCConfig(horizon=5, pruning="none", safety_margin=0.5,
                                               reb_penalty=reb_penalty, safe_prerefill=safe))
    pol._last_video = "v"
    pol._observe_throughput = lambda: None
    total_reb, plan_reb = 0.0, []
    for k in range(n):
        env.chunk_idx = k
        pol.tube.horizon_lower_bounds_kbps = lambda h, decision_chunk=None, k=k: 0.8 * y[k : k + h]
        a = pol.plan_action()
        d = env.get_chunk_size_bits(env.BITRATE_LEVELS[a], k) / (y[k] * 1000.0)
        reb = max(0.0, d - env.buffer_level)
        total_reb += reb
        plan_reb.append(reb)
        env.buffer_level = min(max(0.0, env.buffer_level - d) + env.CHUNK_DURATION, env.BUFFER_MAX)
        env.last_bitrate_idx = a
    return total_reb, plan_reb


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_safe_zero_rebuffer_under_perfect_coverage(seed: int) -> None:
    reb, _ = _closed_loop(safe=True, reb_penalty=0.01, seed=seed)
    assert reb == 0.0


def test_legacy_constraint_is_vacuous_and_can_rebuffer() -> None:
    # Same setting with a weak rebuffer penalty: legacy post-refill check (B^{(h)} >= margin) never binds.
    rebs = [_closed_loop(safe=False, reb_penalty=0.01, seed=s)[0] for s in range(3)]
    assert max(rebs) > 0.0


def test_safe_open_loop_plan_has_zero_rebuffer_on_bound_and_actual() -> None:
    """Every horizon step of the chosen plan satisfies d_h <= B^{(h-1)} - margin under L, hence under y >= L."""
    from itertools import product

    env = _Env(buf=6.0, n=40)
    env.chunk_idx = 5
    L = np.array([1200.0, 600.0, 2500.0, 400.0, 1800.0])
    pol = TubeV2MPCPolicy(env, TubeV2MPCConfig(horizon=5, pruning="none", reb_penalty=0.01, safe_prerefill=True))
    pol._last_video = "v"
    pol._observe_throughput = lambda: None
    pol.tube.horizon_lower_bounds_kbps = lambda h, decision_chunk=None: L[:h]
    a1 = pol.plan_action()
    # there exists a completion a2..a5 that is safe under L; check it stays rebuffer-free for any y >= L
    rng = np.random.default_rng(0)
    found = False
    for rest in product(range(6), repeat=4):
        seq = (a1, *rest)
        b, ok = env.buffer_level, True
        for h, a in enumerate(seq):
            d = env.BITRATE_LEVELS[a] * 4.0 / L[h]
            if d > b - 0.5:
                ok = False
                break
            b = min(b - d + 4.0, 12.0)
        if ok:
            found = True
            for _ in range(50):
                yy = L * (1.0 + rng.exponential(0.5, 5))
                b = env.buffer_level
                for h, a in enumerate(seq):
                    d = env.BITRATE_LEVELS[a] * 4.0 / yy[h]
                    assert d <= b
                    b = min(b - d + 4.0, 12.0)
            break
    assert found


def test_safe_fallback_lowest_when_infeasible() -> None:
    env = _Env(buf=0.0, n=40)
    env.chunk_idx = 3
    env.last_bitrate_idx = 5
    pol = TubeV2MPCPolicy(env, TubeV2MPCConfig(horizon=5, pruning="monotone", safe_prerefill=True))
    pol._last_video = "v"
    pol._observe_throughput = lambda: None
    pol.tube.horizon_lower_bounds_kbps = lambda h, decision_chunk=None: np.full(h, 1000.0)
    assert pol.plan_action() == 0


def test_arm_parsing_safe_fs() -> None:
    assert parse_tube_v3_arm("tube_v2_step_static_a0.15_safe") == {"kind": "static", "alpha": 0.15, "safe": True, "freeze": False}
    assert parse_tube_v3_arm("tube_v2_step_static_a0.1_safe_fs") == {"kind": "static", "alpha": 0.1, "safe": True, "freeze": True}
    assert parse_tube_v3_arm("tube_v2_maxscore_a0.2_safe_fs")["freeze"] is True
    assert parse_tube_v3_arm("tube_v2_step_aci_a0.1_fs") is None


def _ingest(tube, traces):
    for tr in traces:
        for v in tr:
            tube.ingest_calibration_sample(float(v))
        tube.reset_episode_state()
    tube.finalize_frozen_calibration()


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_fs_maxscore_joint_coverage_iid_traces(alpha: float) -> None:
    """Frozen sigma+Q from independent calib traces => joint coverage >= 1-alpha on new traces."""
    rng = np.random.default_rng(7)
    H = 5
    cal = [rng.lognormal(7.0, 0.4, 300) for _ in range(60)]
    tube = MaxScoreConformalTube(ConformalTubeV2Config(alpha=alpha, horizon=H, freeze_scales=True))
    _ingest(tube, cal)
    sig0, q0 = tube._frozen_sigma.copy(), tube._frozen_Q
    hits = n = 0
    for _ in range(40):
        tube.reset_episode_state()
        tube._obs.clear()
        y = rng.lognormal(7.0, 0.4, 300)
        for t, v in enumerate(y):
            if len(tube._obs) == 5 and t + H <= len(y):
                L = tube.horizon_lower_bounds_kbps(H, decision_chunk=t)
                if t % H == 0:
                    n += 1
                    hits += bool(np.all(y[t : t + H] >= L - 1e-9))
            tube.observe(float(v), realized_chunk=t)
    assert np.allclose(tube._frozen_sigma, sig0) and tube._frozen_Q == q0  # never updated online
    assert hits / n >= 1 - alpha - 0.025


def test_fs_step_static_per_step_coverage() -> None:
    rng = np.random.default_rng(11)
    H, a = 5, 0.15
    tube = ConformalThroughputTubeV2(ConformalTubeV2Config(alpha=a, horizon=H, freeze_scales=True))
    _ingest(tube, [rng.lognormal(7.0, 0.4, 300) for _ in range(40)])
    tube.reset_episode_state()
    tube._obs.clear()
    y = rng.lognormal(7.0, 0.4, 20000)
    for t, v in enumerate(y):
        if len(tube._obs) == 5:
            tube.horizon_lower_bounds_kbps(H, decision_chunk=t)
        tube.observe(float(v), realized_chunk=t)
    cov = tube.empirical_coverage_per_step()
    assert np.all(cov >= 1 - a - 0.015)
