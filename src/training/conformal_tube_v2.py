"""
Multi-step conformal throughput tube (Tube-v2).

Unlike ``ConformalThroughputTube``, each horizon step h maintains its own
nonconformity score buffer and quantile, so ``horizon_lower_bounds_kbps(h)``
is generally **non-constant** in h.

See ``docs/tube_v2_theory.md`` for the finite-sample coverage proposition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.training.certified_perceptual_shield import ConformalConfig

ScoreKind = Literal["relative", "log"]
TubeMode = Literal["bonferroni", "per_step"]


@dataclass
class ConformalTubeV2Config:
    mode: TubeMode = "per_step"
    alpha: float = 0.10
    horizon: int = 5
    window: int = 200
    score: ScoreKind = "relative"
    use_aci: bool = False
    aci_gamma: float = 0.005
    min_kbps: float = 10.0
    cold_start_cap: float = 0.80
    conformal: ConformalConfig = field(
        default_factory=lambda: ConformalConfig(alpha=0.10, window=200, k_predict=5)
    )


class ConformalThroughputTubeV2:
    """Per-step split conformal bounds over a planning horizon."""

    def __init__(self, cfg: ConformalTubeV2Config | None = None):
        self.cfg = cfg or ConformalTubeV2Config()
        h = max(int(self.cfg.horizon), 1)
        self._scores: list[deque[float]] = [deque(maxlen=self.cfg.window) for _ in range(h)]
        self._alpha_t: np.ndarray = self._base_alpha_per_step().copy()
        self._obs: deque[float] = deque(maxlen=max(self.cfg.conformal.k_predict, 1))
        self._pred_by_decision: dict[int, np.ndarray] = {}
        self._n_realized: int = 0
        self._cov_hits = np.zeros(h, dtype=np.int64)
        self._cov_n = np.zeros(h, dtype=np.int64)
        self._joint_hits = 0
        self._joint_n = 0

    def _base_alpha_per_step(self) -> np.ndarray:
        h = max(int(self.cfg.horizon), 1)
        a = float(self.cfg.alpha)
        if self.cfg.mode == "bonferroni":
            return np.full(h, a / h, dtype=float)
        return np.full(h, a, dtype=float)

    def reset(self) -> None:
        h = max(int(self.cfg.horizon), 1)
        self._scores = [deque(maxlen=self.cfg.window) for _ in range(h)]
        self._alpha_t = self._base_alpha_per_step().copy()
        self._obs.clear()
        self._pred_by_decision.clear()
        self._n_realized = 0
        self._cov_hits[:] = 0
        self._cov_n[:] = 0
        self._joint_hits = 0
        self._joint_n = 0

    def predict_point_kbps(self) -> float:
        if not self._obs:
            return 0.0
        vals = np.clip(np.asarray(self._obs, dtype=float), 1e-6, None)
        return float(len(vals) / np.sum(1.0 / vals))

    def point_lower_bound_kbps(self) -> float:
        hat = self.predict_point_kbps()
        if hat <= 0:
            return float(self.cfg.min_kbps)
        return float(max(self._lower_from_hat(hat, 0), self.cfg.min_kbps))

    def _score(self, hat: float, actual: float) -> float:
        hat = max(float(hat), 1e-6)
        actual = max(float(actual), 1e-6)
        if self.cfg.score == "log":
            return float(max(0.0, np.log(hat) - np.log(actual)))
        return float(max(0.0, (hat - actual) / hat))

    @staticmethod
    def _finite_sample_quantile(scores: deque[float], alpha: float) -> float:
        """(1 - alpha) quantile with split-conformal rank ceil((n+1)(1-alpha))."""
        if not scores:
            return 0.0
        arr = np.sort(np.asarray(scores, dtype=float))
        n = arr.size
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        rank = min(max(rank, 1), n)
        return float(arr[rank - 1])

    def _min_calib(self, alpha_h: float) -> int:
        return max(int(np.ceil(1.0 / max(alpha_h, 1e-9)) - 1), 1)

    def _lower_from_hat(self, hat: float, step_idx: int) -> float:
        alpha_h = float(self._alpha_t[step_idx])
        n = len(self._scores[step_idx])
        if n < self._min_calib(alpha_h):
            return float(self.cfg.cold_start_cap * hat)
        q = self._finite_sample_quantile(self._scores[step_idx], alpha_h)
        if self.cfg.score == "log":
            return float(hat * np.exp(-q))
        return float(hat * (1.0 - q))

    def horizon_lower_bounds_kbps(
        self,
        horizon: int | None = None,
        *,
        decision_chunk: int | None = None,
    ) -> np.ndarray:
        h = int(horizon if horizon is not None else self.cfg.horizon)
        h = max(min(h, len(self._scores)), 1)
        hat = self.predict_point_kbps()
        if hat <= 0:
            hat = float(self.cfg.min_kbps)
        hats = np.full(h, hat, dtype=float)
        lbs = np.array([self._lower_from_hat(hat, j) for j in range(h)], dtype=float)
        lbs = np.maximum(lbs, self.cfg.min_kbps)
        if decision_chunk is not None:
            self._pred_by_decision[int(decision_chunk)] = hats.copy()
        return lbs

    def _register_score(self, h_idx: int, hat: float, actual: float) -> None:
        lb = self._lower_from_hat(hat, h_idx)
        s = self._score(hat, actual)
        self._scores[h_idx].append(s)
        covered = actual >= lb - 1e-6
        self._cov_n[h_idx] += 1
        if covered:
            self._cov_hits[h_idx] += 1
        if self.cfg.use_aci:
            err = 0.0 if covered else 1.0
            base = float(self._base_alpha_per_step()[h_idx])
            self._alpha_t[h_idx] = float(
                np.clip(
                    self._alpha_t[h_idx] + self.cfg.aci_gamma * (base - err),
                    1e-4,
                    0.99,
                )
            )

    def observe(self, actual_kbps: float, *, realized_chunk: int | None = None) -> None:
        actual = float(max(actual_kbps, 1e-6))
        self._register_pending_scores(actual, realized_chunk)
        self._obs.append(actual)

    def _register_pending_scores(self, actual: float, realized_chunk: int | None) -> None:
        c = int(realized_chunk if realized_chunk is not None else self._n_realized)
        self._n_realized = max(self._n_realized, c + 1)
        h_max = len(self._scores)
        joint_ok = True
        any_scored = False
        for h in range(1, h_max + 1):
            j = c - h
            hats = self._pred_by_decision.get(j)
            if hats is None or h > len(hats):
                continue
            hat = float(hats[h - 1])
            self._register_score(h - 1, hat, actual)
            any_scored = True
            lb = self._lower_from_hat(hat, h - 1)
            if actual < lb - 1e-6:
                joint_ok = False
        if any_scored and h_max > 0:
            self._joint_n += 1
            if joint_ok:
                self._joint_hits += 1

    def empirical_coverage_per_step(self) -> np.ndarray:
        out = np.zeros(len(self._scores), dtype=float)
        for i in range(len(self._scores)):
            if self._cov_n[i] > 0:
                out[i] = self._cov_hits[i] / self._cov_n[i]
        return out

    def joint_coverage(self) -> float:
        if self._joint_n <= 0:
            return float("nan")
        return float(self._joint_hits / self._joint_n)


__all__ = ["ConformalTubeV2Config", "ConformalThroughputTubeV2", "ScoreKind", "TubeMode"]
