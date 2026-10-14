"""
Multi-horizon conformal throughput tube for Tube-MPC (Paper 2, no VMAF).

Wraps online ACI or static split-conformal estimators and exposes a horizon
vector of pessimistic throughput lower bounds (kbps) for MPC simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.training.aci_perceptual_shield import ACIConfig, OnlineACIEstimator
from src.training.certified_perceptual_shield import ConformalConfig, ConformalThroughputEstimator


@dataclass
class ConformalTubeConfig:
    horizon: int = 5
    use_aci: bool = True
    aci: ACIConfig = field(default_factory=ACIConfig)
    static_conformal: ConformalConfig = field(default_factory=ConformalConfig)
    min_kbps: float = 10.0


class ConformalThroughputTube:
    """Throughput lower bound(s) for tube-MPC planning."""

    def __init__(self, cfg: ConformalTubeConfig | None = None):
        self.cfg = cfg or ConformalTubeConfig()
        if self.cfg.use_aci:
            self._aci = OnlineACIEstimator(self.cfg.aci)
            self._static: ConformalThroughputEstimator | None = None
        else:
            self._aci = None
            self._static = ConformalThroughputEstimator(self.cfg.static_conformal)

    def reset(self):
        if self._aci is not None:
            self._aci = OnlineACIEstimator(self.cfg.aci)
        if self._static is not None:
            self._static = ConformalThroughputEstimator(self.cfg.static_conformal)

    @property
    def alpha_t(self) -> float:
        if self._aci is not None:
            return float(self._aci.alpha_t)
        return float(self.cfg.static_conformal.alpha)

    def observe(self, actual_kbps: float):
        """Register realized throughput after a chunk (updates calibration)."""
        actual = float(max(actual_kbps, 1e-6))
        if self._aci is not None:
            self._aci.update(actual)
        elif self._static is not None:
            self._static.update(actual)

    def point_lower_bound_kbps(self) -> float:
        if self._aci is not None:
            lb = self._aci.lower_bound()
        else:
            lb = self._static.lower_bound()  # type: ignore[union-attr]
        return float(max(lb, self.cfg.min_kbps))

    def horizon_lower_bounds_kbps(self, horizon: int | None = None) -> np.ndarray:
        """Conservative tube: repeat the current conformal lower bound over H steps."""
        h = int(horizon if horizon is not None else self.cfg.horizon)
        h = max(h, 1)
        lb = self.point_lower_bound_kbps()
        return np.full(h, lb, dtype=float)
