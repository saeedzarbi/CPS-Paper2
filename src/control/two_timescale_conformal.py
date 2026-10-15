"""
Two-timescale conformal control arms for CleanABREnv (h3 spec).

Arms: raw, static_conformal, tube_conformal, two_timescale_conformal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.training.certified_perceptual_shield import ConformalConfig, ConformalThroughputEstimator
from src.training.conformal_tube import ConformalTubeConfig
from src.training.tube_mpc_policy import TubeMPCConfig, TubeMPCPolicy

ArmName = Literal["raw", "static_conformal", "tube_conformal", "two_timescale_conformal"]


@dataclass
class ControlArmConfig:
    alpha: float = 0.10
    safety_margin: float = 0.5
    horizon: int = 5
    cold_start_cap: float = 0.80
    cold_start_chunks: int = 20


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

    def project_action(self, proposed: int) -> int:
        self.observe()
        lb = self.lower_bound_kbps()
        env = self.env
        k = int(env.chunk_idx)
        buf = float(env.buffer_level)
        cap = max(buf - self.cfg.safety_margin, 0.1)
        best = 0
        for a in range(proposed, -1, -1):
            br = int(env.BITRATE_LEVELS[a])
            bits = env.get_chunk_size_bits(br, k)
            dl = bits / (lb * 1000.0 + 1e-6)
            if dl <= cap:
                best = a
                break
        return best

    def coverage_hit(self) -> bool | None:
        tp = float(getattr(self.env, "last_raw_throughput", 0.0))
        if tp <= 0 or self.env.chunk_idx == 0:
            return None
        return tp >= self.lower_bound_kbps()


def configure_env_for_arm(env, arm: ArmName) -> None:
    env.enable_early_abort = arm == "two_timescale_conformal"


def make_controlled_policy(env, arm: ArmName, base_policy, cfg: ControlArmConfig | None = None):
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

    tube_cfg = TubeMPCConfig(
        horizon=cfg.horizon,
        safety_margin=cfg.safety_margin,
        tube=ConformalTubeConfig(horizon=cfg.horizon, use_aci=False),
    )
    if arm == "tube_conformal":
        return TubeMPCPolicy(env, tube_cfg)

    if arm == "two_timescale_conformal":
        return TubeMPCPolicy(env, tube_cfg)

    raise ValueError(arm)


def arm_display_name(arm: ArmName) -> str:
    return arm
