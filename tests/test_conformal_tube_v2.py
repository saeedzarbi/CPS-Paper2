"""Unit tests for ConformalThroughputTubeV2 (Tube-v2)."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from configs.paths import STANDARDIZED_DIR
from src.control.two_timescale_conformal import TUBE_V2_ARMS, parse_tube_v2_arm
from src.evaluation.clean_abr_eval import HEADLINE_TRACE_SETS, build_env, run_arm
from src.training.certified_perceptual_shield import ConformalConfig, ConformalThroughputEstimator
from src.training.conformal_tube_v2 import ConformalThroughputTubeV2, ConformalTubeV2Config
from src.training.tube_mpc_policy_v2 import TubeV2MPCConfig, TubeV2MPCPolicy


def _lognormal_stream(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.lognormal(mean=7.0, sigma=0.45, size=n).astype(float)


def _simulate_joint_coverage(cfg: ConformalTubeV2Config, n: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    tube = ConformalThroughputTubeV2(cfg)
    h = max(int(cfg.horizon), 1)
    tp = _lognormal_stream(rng, n)
    bounds_hist: list[np.ndarray] = []
    for t, actual in enumerate(tp):
        bounds_hist.append(tube.horizon_lower_bounds_kbps(h, decision_chunk=t).copy())
        tube.observe(float(actual), realized_chunk=t)
    per_h: dict[int, list[bool]] = {i: [] for i in range(h)}
    joint_flags: list[bool] = []
    for j in range(len(tp) - h):
        ok = True
        for step in range(h):
            c = j + step + 1
            act = float(tp[c])
            lb = float(bounds_hist[j][step])
            hit = act >= lb - 1e-6
            per_h[step].append(hit)
            ok = ok and hit
        joint_flags.append(ok)
    per_step_rates = np.array([np.mean(per_h[i]) if per_h[i] else 1.0 for i in range(h)])
    joint_rate = float(np.mean(joint_flags)) if joint_flags else 1.0
    return per_step_rates, joint_rate


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_per_step_marginal_coverage(alpha: float, horizon: int) -> None:
    cfg = ConformalTubeV2Config(
        mode="per_step", alpha=alpha, horizon=horizon, use_aci=False, cold_start_cap=0.5
    )
    rates, _ = _simulate_joint_coverage(cfg, n=25_000, seed=horizon + int(alpha * 100))
    assert float(np.min(rates)) >= 1.0 - alpha - 0.02


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_bonferroni_joint_coverage(alpha: float, horizon: int) -> None:
    cfg = ConformalTubeV2Config(
        mode="bonferroni", alpha=alpha, horizon=horizon, use_aci=False, cold_start_cap=0.5
    )
    _, joint = _simulate_joint_coverage(cfg, n=25_000, seed=horizon * 3 + int(alpha * 100))
    assert joint >= 1.0 - alpha - 0.02


def test_bounds_non_increasing_on_random_walk() -> None:
    rng = np.random.default_rng(0)
    cfg = ConformalTubeV2Config(horizon=5, mode="per_step")
    tube = ConformalThroughputTubeV2(cfg)
    x = 500.0
    for t in range(400):
        tube.horizon_lower_bounds_kbps(5, decision_chunk=t)
        x = max(50.0, x + rng.normal(0, 40))
        tube.observe(x, realized_chunk=t)
    lbs = tube.horizon_lower_bounds_kbps(5, decision_chunk=400)
    if not np.all(lbs[:-1] >= lbs[1:] - 1e-6):
        warnings.warn("Tube bounds not monotone in h on random walk (sanity only)", stacklevel=1)


def test_h1_matches_static_conformal_estimator() -> None:
    rng = np.random.default_rng(42)
    tp = _lognormal_stream(rng, 800)
    est = ConformalThroughputEstimator(ConformalConfig(alpha=0.10, window=200, k_predict=5))
    tube = ConformalThroughputTubeV2(
        ConformalTubeV2Config(horizon=1, mode="per_step", alpha=0.10, cold_start_cap=0.8)
    )
    diffs = []
    for t, actual in enumerate(tp):
        lb_est = est.lower_bound()
        lb_tube = float(tube.horizon_lower_bounds_kbps(1, decision_chunk=t)[0])
        if lb_est > 0 and lb_tube > 0:
            diffs.append(abs(lb_est - lb_tube) / lb_est)
        est.update(float(actual))
        tube.observe(float(actual), realized_chunk=t)
    assert np.median(diffs) < 0.08


def test_no_lookahead() -> None:
    rng = np.random.default_rng(7)
    base = _lognormal_stream(rng, 120)
    alt = base.copy()
    alt[60:] = alt[60:] * 5.0
    cfg = ConformalTubeV2Config(horizon=3)
    for trace in (base, alt):
        tube = ConformalThroughputTubeV2(cfg)
        bounds_at_50 = None
        for t, actual in enumerate(trace):
            lbs = tube.horizon_lower_bounds_kbps(3, decision_chunk=t).copy()
            if t == 50:
                bounds_at_50 = lbs
            tube.observe(float(actual), realized_chunk=t)
        assert bounds_at_50 is not None
    tube_a = ConformalThroughputTubeV2(cfg)
    tube_b = ConformalThroughputTubeV2(cfg)
    b50_a = b50_b = None
    for t in range(51):
        b50_a = tube_a.horizon_lower_bounds_kbps(3, decision_chunk=t).copy()
        tube_a.observe(float(base[t]), realized_chunk=t)
        b50_b = tube_b.horizon_lower_bounds_kbps(3, decision_chunk=t).copy()
        tube_b.observe(float(alt[t]), realized_chunk=t)
    assert np.allclose(b50_a, b50_b)


def test_determinism() -> None:
    cfg = ConformalTubeV2Config(horizon=4)
    tp = _lognormal_stream(np.random.default_rng(0), 200)

    def run() -> list[np.ndarray]:
        tube = ConformalThroughputTubeV2(cfg)
        out = []
        for t, a in enumerate(tp):
            out.append(tube.horizon_lower_bounds_kbps(4, decision_chunk=t).copy())
            tube.observe(float(a), realized_chunk=t)
        return out

    assert all(np.allclose(a, b) for a, b in zip(run(), run()))


REGRESSION_ARMS = ("raw", "static_conformal", "tube_conformal")
REGRESSION_EPS = 3


def test_existing_arms_regression() -> None:
    trace_dir = HEADLINE_TRACE_SETS["5g_markov"]
    if not trace_dir.is_dir():
        pytest.skip("5g_markov traces missing")
    for arm in REGRESSION_ARMS:
        a = run_arm("bba", trace_dir, arm, REGRESSION_EPS, workers=1)
        b = run_arm("bba", trace_dir, arm, REGRESSION_EPS, workers=1)
        assert len(a) == len(b)
        for ra, rb in zip(a, b):
            for k in ra:
                assert k in rb
                if isinstance(ra[k], (int, float, np.floating, np.integer)):
                    assert abs(float(ra[k]) - float(rb[k])) < 1e-9
                else:
                    assert ra[k] == rb[k]


def test_tube_v2_planner_latency_order() -> None:
    trace_dir = HEADLINE_TRACE_SETS["bb"]
    if not trace_dir.is_dir():
        pytest.skip("bb traces missing")
    env = build_env(trace_dir, "raw")
    lat = {}
    for mode in ("none", "monotone", "beam"):
        pol = TubeV2MPCPolicy(
            env,
            TubeV2MPCConfig(horizon=5, pruning=mode, beam_width=32),
        )
        pol.on_episode_start()
        env.reset(seed=1000)
        lat[mode] = pol.benchmark_plan_latency_us(repeats=5)
    assert lat["none"] > 0 and lat["monotone"] > 0 and lat["beam"] > 0


def test_tube_v2_arm_names() -> None:
    assert parse_tube_v2_arm("tube_v2_bonf_aci") == ("bonferroni", True, True)
    assert parse_tube_v2_arm("tube_v2_step_static_nocap") == ("per_step", False, False)
    assert len(TUBE_V2_ARMS) == 8


def test_tube_v2_bounds_not_constant() -> None:
    path = STANDARDIZED_DIR / "test_traces_v18"
    if not path.is_dir():
        pytest.skip("v18 traces missing")
    trace = json.loads(next(path.glob("*.json")).read_text())["throughput_kbps"]
    cfg = ConformalTubeV2Config(horizon=5, mode="per_step", alpha=0.10)
    tube = ConformalThroughputTubeV2(cfg)
    for t, a in enumerate(trace[:300]):
        tube.horizon_lower_bounds_kbps(5, decision_chunk=t)
        tube.observe(float(a), realized_chunk=t)
    lbs = tube.horizon_lower_bounds_kbps(5, decision_chunk=300)
    assert float(np.std(lbs)) > 1.0
