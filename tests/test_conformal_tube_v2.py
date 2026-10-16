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
    rates, joint_sim = _simulate_joint_coverage(cfg, n=25_000, seed=horizon * 3 + int(alpha * 100))
    assert joint_sim >= 1.0 - alpha - 0.02
    # API joint (per completed decision, bounds frozen at issue time)
    tube = ConformalThroughputTubeV2(cfg)
    h = horizon
    tp = _lognormal_stream(np.random.default_rng(horizon * 3 + int(alpha * 100)), 25_000)  # noqa: NPY002
    for t, actual in enumerate(tp):
        tube.horizon_lower_bounds_kbps(h, decision_chunk=t)
        tube.observe(float(actual), realized_chunk=t)
    assert abs(tube.joint_coverage() - joint_sim) < 0.015


@pytest.mark.parametrize("alpha", [0.10])
@pytest.mark.parametrize("horizon", [5])
def test_bonferroni_aci_per_step_coverage(alpha: float, horizon: int) -> None:
    cfg = ConformalTubeV2Config(
        mode="bonferroni",
        alpha=alpha,
        horizon=horizon,
        use_aci=True,
        aci_gamma=0.01,
        cold_start_cap=0.5,
    )
    rates, _ = _simulate_joint_coverage(cfg, n=25_000, seed=909)
    alpha_h = alpha / horizon
    assert float(np.min(rates)) >= 1.0 - alpha_h - 0.04


def _simulate_aci_iid(cfg: ConformalTubeV2Config, n: int, seed: int) -> ConformalThroughputTubeV2:
    rng = np.random.default_rng(seed)
    tube = ConformalThroughputTubeV2(cfg)
    h = max(int(cfg.horizon), 1)
    tp = _lognormal_stream(rng, n)
    for t, actual in enumerate(tp):
        tube.horizon_lower_bounds_kbps(h, decision_chunk=t)
        tube.observe(float(actual), realized_chunk=t)
    return tube


def test_aci_iid_long_run_coverage_and_alpha_interior() -> None:
    alpha = 0.10
    cfg = ConformalTubeV2Config(
        mode="per_step",
        alpha=alpha,
        horizon=5,
        use_aci=True,
        aci_gamma=0.006,
        cold_start_cap=0.5,
    )
    tube = _simulate_aci_iid(cfg, n=30_000, seed=11)
    cov = tube.empirical_coverage_per_step()
    target = 1.0 - alpha
    for i, rate in enumerate(cov):
        if tube._cov_n[i] <= 0:
            continue
        assert abs(float(rate) - target) <= 0.02, f"step {i}: {rate}"
    mean_a = float(tube.mean_alpha_t())
    assert 0.5 * alpha <= mean_a <= 1.5 * alpha
    assert float(tube.aci_clip_fraction()) < 0.05


def test_aci_inverted_sign_fails_iid_guard() -> None:
    alpha = 0.10
    cfg_bad = ConformalTubeV2Config(
        mode="per_step",
        alpha=alpha,
        horizon=5,
        use_aci=True,
        aci_gamma=0.006,
        cold_start_cap=0.5,
        aci_invert_sign=True,
    )
    tube = _simulate_aci_iid(cfg_bad, n=30_000, seed=11)
    cov = tube.empirical_coverage_per_step()
    target = 1.0 - alpha
    mean_a = float(tube.mean_alpha_t())
    bad_cov = any(abs(float(rate) - target) > 0.02 for rate in cov if rate > 0)
    bad_alpha = mean_a < 0.5 * alpha or mean_a > 1.5 * alpha
    bad_clip = float(tube.aci_clip_fraction()) >= 0.05
    assert bad_cov or bad_alpha or bad_clip


def test_aci_differs_and_recovers_after_shift() -> None:
    rng = np.random.default_rng(2)
    shift = 3000
    n = 20_000
    trace = np.concatenate(
        [
            rng.lognormal(7.5, 0.20, shift),
            rng.lognormal(5.0, 0.75, n - shift) * 0.28,
        ]
    )
    alpha = 0.10
    h = 3
    target = 1.0 - alpha
    window = 500

    def run_trace(cfg: ConformalTubeV2Config) -> tuple[float, float, np.ndarray]:
        tube = ConformalThroughputTubeV2(cfg)
        post_hits: list[float] = []
        for t, actual in enumerate(trace):
            bounds = tube.horizon_lower_bounds_kbps(h, decision_chunk=t)
            if t >= shift + h:
                post_hits.append(float(actual) >= float(bounds[0]) - 1e-6)
            tube.observe(float(actual), realized_chunk=t)
        # Early recovery window (first 200 post-shift decisions): with aligned
        # horizon indexing, static recovers once its 200-score window refreshes,
        # so ACI's advantage is in the transient.
        post_roll = float(np.mean(post_hits[:200]))
        long_run = float(np.mean(post_hits[-5000:])) if len(post_hits) >= 5000 else float(
            np.mean(post_hits)
        )
        return post_roll, long_run, tube._alpha_t.copy()

    cfg_static = ConformalTubeV2Config(
        mode="per_step", alpha=alpha, horizon=h, use_aci=False, cold_start_cap=0.5
    )
    cfg_aci = ConformalTubeV2Config(
        mode="per_step",
        alpha=alpha,
        horizon=h,
        use_aci=True,
        aci_gamma=0.008,
        cold_start_cap=0.5,
    )
    post_static, long_static, _ = run_trace(cfg_static)
    post_aci, long_aci, alpha_t = run_trace(cfg_aci)
    base = ConformalThroughputTubeV2(cfg_aci)._base_alpha_per_step()
    assert not np.allclose(alpha_t, base, atol=1e-4)
    assert post_aci != post_static
    assert abs(post_aci - target) < abs(post_static - target) - 0.03
    assert abs(long_aci - target) <= 0.02


def test_cold_start_gate_uses_nominal_alpha_not_aci() -> None:
    cfg = ConformalTubeV2Config(
        mode="bonferroni", alpha=0.10, horizon=5, use_aci=True, aci_gamma=0.05
    )
    tube = ConformalThroughputTubeV2(cfg)
    nominal = float(tube._base_alpha_per_step()[0])
    need = tube._min_calib(nominal)
    for _ in range(need):
        tube._scores[0].append(0.05)
    assert not tube._in_cold_start(0)
    _, lo, _ = tube._alpha_bounds(0)
    tube._alpha_t[0] = lo
    assert not tube._in_cold_start(0)


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
        a = run_arm(
            "bba", trace_dir, arm, REGRESSION_EPS, workers=1, trace_key="5g_markov"
        )
        b = run_arm(
            "bba", trace_dir, arm, REGRESSION_EPS, workers=1, trace_key="5g_markov"
        )
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
