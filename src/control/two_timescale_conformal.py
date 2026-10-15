"""
Conformal control arms for CleanABREnv (Paper 2 headline: raw / static / tube).

`two_timescale_conformal` (early abort) removed from headline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.training.certified_perceptual_shield import ConformalConfig, ConformalThroughputEstimator
from src.training.conformal_tube import ConformalTubeConfig
from src.training.tube_mpc_policy import TubeMPCConfig, TubeMPCPolicy

HeadlineArm = Literal["raw", "static_conformal", "tube_conformal"]
ArmName = HeadlineArm  # backward-compatible alias

HEADLINE_ARMS: tuple[HeadlineArm, ...] = ("raw", "static_conformal", "tube_conformal")


@dataclass
class ControlArmConfig:
    alpha: float = 0.10
    safety_margin: float = 0.5
    horizon: int = 5
    cold_start_cap: float = 0.80
    cold_start_chunks: int = 20
    upward_smooth_weight: float = 2.0


class StaticConformalProjector:
    """Single-step feasibility under harmonic prediction + split conformal lb."""

    def __init__(self, env, cfg: ControlArmConfig | None = None):
        self.env = env
        self.cfg = cfg or ControlArmConfig()
        self._est = ConformalThroughputEstimator(
            ConformalConfig(alpha=self.cfg.alpha, window=200, k_predict=5)
        )
        self._tp_hist: list[float] = []

    def on_episode_start(self):
        self._est = ConformalThroughputEstimator(
            ConformalConfig(alpha=self.cfg.alpha, window=200, k_predict=5)
        )
        self._tp_hist = []

    def observe(self):
        env = self.env
        if env.chunk_idx == 0:
            self.on_episode_start()
            return
        tp = float(getattr(env, "last_raw_throughput", 0.0))
        if tp > 0:
            self._tp_hist.append(tp)
            self._est.update(tp)

    def _harmonic_kbps(self) -> float:
        h = self._tp_hist[-5:]
        if not h:
            return float(max(getattr(self.env, "last_raw_throughput", 500.0), 10.0))
        return float(len(h) / np.sum(1.0 / np.maximum(h, 1e-6)))

    def lower_bound_kbps(self) -> float:
        lb = float(self._est.lower_bound())
        if self.env.chunk_idx < self.cfg.cold_start_chunks:
            pred = self._harmonic_kbps()
            lb = min(lb, self.cfg.cold_start_cap * pred)
        return max(lb, 10.0)

    def max_safe_action(self, proposed: int | None = None) -> int:
        """Highest rung index feasible under 1-step conformal bound (<= proposed if given)."""
        self.observe()
        lb = self.lower_bound_kbps()
        env = self.env
        k = int(env.chunk_idx)
        buf = float(env.buffer_level)
        cap = max(buf - self.cfg.safety_margin, 0.1)
        hi = int(proposed) if proposed is not None else len(env.BITRATE_LEVELS) - 1
        best = 0
        for a in range(hi, -1, -1):
            br = int(env.BITRATE_LEVELS[a])
            bits = env.get_chunk_size_bits(br, k)
            dl = bits / (lb * 1000.0 + 1e-6)
            if dl <= cap:
                best = a
                break
        return best

    def project_action(self, proposed: int) -> int:
        return self.max_safe_action(proposed)


class AsymmetricTubeMPCPolicy(TubeMPCPolicy):
    """
    Multi-horizon tube MPC with asymmetric smoothness:
    - Up-switch / hold-high: full H-step tube + upward smoothness penalty.
    - Down-switch / low buffer: no smoothness block on safety steps.
    """

    def __init__(self, env, cfg: ControlArmConfig | None = None, tube_cfg: TubeMPCConfig | None = None):
        cfg = cfg or ControlArmConfig()
        tube_cfg = tube_cfg or TubeMPCConfig(
            horizon=cfg.horizon,
            safety_margin=cfg.safety_margin,
            tube=ConformalTubeConfig(horizon=cfg.horizon, use_aci=False),
        )
        super().__init__(env, tube_cfg)
        self.arm_cfg = cfg

    def _asymmetric_smooth(self, action_idx: np.ndarray, prev_idx: int, buf: float) -> np.ndarray:
        """Per-sequence smoothness penalty at the first planned chunk."""
        env = self.env
        b_tgt = float(getattr(env, "BUFFER_TARGET", 6.0))
        lu = self.log_u[action_idx]
        u_prev = float(self.log_u[prev_idx])
        pen = np.zeros(len(action_idx), dtype=float)
        for i, aj in enumerate(action_idx):
            if aj < prev_idx or buf < b_tgt:
                pen[i] = 0.0
            elif aj >= prev_idx and aj > 0:
                pen[i] = self.arm_cfg.upward_smooth_weight * max(0.0, lu[i] - u_prev)
                pen[i] += abs(lu[i] - u_prev)
            else:
                pen[i] = abs(lu[i] - u_prev)
        return pen

    def plan_action(self) -> int:
        env = self.env
        self._observe_throughput()
        k = int(env.chunk_idx)
        a_prev = int(getattr(env, "last_bitrate_idx", 0))
        h = min(self.cfg.horizon, int(env.max_chunks) - k)
        if h <= 0:
            return self.n - 1

        tp_tube = self.tube.horizon_lower_bounds_kbps(h)
        sizes = self._chunk_sizes(k, h)
        seqs = self._seqs(h)
        n_seq = seqs.shape[0]
        dl = sizes[np.arange(h)[None, :], seqs] / (tp_tube[np.arange(h)] * 1000.0 + 1e-6)

        buf0 = float(env.buffer_level)
        buf = np.full(n_seq, buf0)
        prev_u = np.full(n_seq, self.log_u[a_prev])
        total = np.zeros(n_seq)
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        margin = float(self.cfg.safety_margin)
        safe = np.ones(n_seq, dtype=bool)
        b_tgt = float(getattr(env, "BUFFER_TARGET", 6.0))

        for j in range(h):
            reb = np.maximum(0.0, dl[:, j] - buf)
            buf = np.minimum(np.maximum(0.0, buf - dl[:, j]) + p, b_max)
            safe &= buf >= margin
            aj = seqs[:, j]
            lu = self.log_u[aj]
            if j == 0:
                sm = self._asymmetric_smooth(aj, a_prev, buf0)
            else:
                sm = np.zeros(n_seq, dtype=float)
                for i in range(n_seq):
                    pidx = int(seqs[i, j - 1])
                    cidx = int(aj[i])
                    if cidx < pidx or buf0 < b_tgt:
                        sm[i] = 0.0
                    elif cidx >= pidx and cidx > 0:
                        sm[i] = self.arm_cfg.upward_smooth_weight * max(0.0, lu[i] - self.log_u[pidx])
                        sm[i] += abs(lu[i] - self.log_u[pidx])
                    else:
                        sm[i] = abs(lu[i] - self.log_u[pidx])
            total += lu - self.cfg.reb_penalty * reb - self.cfg.smooth_weight * sm
            prev_u = lu

        total[~safe] = -1e30
        return int(seqs[int(np.argmax(total)), 0])


class TubeConformalPolicy:
    """Tube MPC + asymmetric planning, capped by static 1-step safe rung."""

    def __init__(self, env, base_policy, cfg: ControlArmConfig | None = None):
        self.env = env
        self.base = base_policy
        self.cfg = cfg or ControlArmConfig()
        self.static = StaticConformalProjector(env, self.cfg)
        self.tube = AsymmetricTubeMPCPolicy(env, self.cfg)

    def on_episode_start(self):
        self.static.on_episode_start()
        self.tube.on_episode_start()

    def predict(self, obs, deterministic=True):
        a_base, _ = self.base.predict(obs, deterministic=deterministic)
        a_static = self.static.max_safe_action(int(a_base))
        a_tube = self.tube.plan_action()
        return int(min(a_tube, a_static)), None


def configure_env_for_arm(env, arm: HeadlineArm) -> None:
    env.enable_early_abort = False


def make_controlled_policy(env, arm: HeadlineArm, base_policy, cfg: ControlArmConfig | None = None):
    cfg = cfg or ControlArmConfig()
    configure_env_for_arm(env, arm)

    if arm == "raw":
        return base_policy

    if arm == "static_conformal":
        proj = StaticConformalProjector(env, cfg)

        class _Wrapped:
            def on_episode_start(self):
                proj.on_episode_start()

            def predict(self, obs, deterministic=True):
                a, _ = base_policy.predict(obs, deterministic=deterministic)
                return proj.project_action(int(a)), None

        return _Wrapped()

    if arm == "tube_conformal":
        return TubeConformalPolicy(env, base_policy, cfg)

    raise ValueError(f"unknown arm: {arm}")


def arm_display_name(arm: HeadlineArm) -> str:
    return arm
