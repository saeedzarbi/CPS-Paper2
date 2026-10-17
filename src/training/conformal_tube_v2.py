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
    aci_invert_sign: bool = False
    min_kbps: float = 10.0
    cold_start_cap: float = 0.80
    # Horizon alignment: step h (1-indexed) of decision j covers chunk j + h - 1 + legacy_step_offset.
    # 0 = aligned with the planner (h=1 is the chunk being downloaded at decision j).
    # 1 = legacy Tube-v2 behaviour (off by one); kept only for regression tests.
    legacy_step_offset: int = 0
    # Split-conformal freeze (arm suffix _fs): quantiles (and max-score sigma) are fitted once
    # on the calibration traces with true h-step scores and never updated online.
    freeze_scales: bool = False
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
        self._hats_by_decision: dict[int, np.ndarray] = {}
        self._bounds_by_decision: dict[int, np.ndarray] = {}
        self._actual_by_chunk: dict[int, float] = {}
        self._n_realized: int = 0
        self._cov_hits = np.zeros(h, dtype=np.int64)
        self._cov_n = np.zeros(h, dtype=np.int64)
        self._joint_decision_flags: list[bool] = []
        self.last_joint_decision_covered: bool | None = None
        self._n_decisions: int = 0
        self._n_cold_decisions: int = 0
        self._aci_updates: int = 0
        self._aci_at_clip: int = 0
        self._frozen: bool = False
        self._frozen_q_step: np.ndarray | None = None
        self._frozen_sigma: np.ndarray | None = None
        self._frozen_Q: float = float("nan")
        self._frozen_n: int = 0
        self._fs_cur: list[float] = []
        self._fs_traces: list[list[float]] = []

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
        self.reset_episode_state()
        self._n_decisions = 0
        self._n_cold_decisions = 0
        self._aci_updates = 0
        self._aci_at_clip = 0

    def _flush_fs_trace(self) -> None:
        if getattr(self, "_fs_cur", None):
            self._fs_traces.append(self._fs_cur)
            self._fs_cur = []

    def _h_step_scores(self, traces: list[list[float]], H: int, stride: int = 1) -> list[list[float]]:
        """True h-step scores: decision j uses HM of y[j-k:j], step h covers y[j+h-1]."""
        k = max(int(self.cfg.conformal.k_predict), 1)
        out: list[list[float]] = [[] for _ in range(H)]
        for tr in traces:
            y = np.clip(np.asarray(tr, dtype=float), 1e-6, None)
            for j in range(k, len(y) - H + 1, max(int(stride), 1)):
                hat = float(k / np.sum(1.0 / y[j - k : j]))
                for h in range(H):
                    out[h].append(self._score(hat, float(y[j + h])))
        return out

    def finalize_frozen_calibration(self) -> None:
        """Fit per-step split-conformal quantiles once (all calibration traces)."""
        self._flush_fs_trace()
        H = len(self._scores)
        sc = self._h_step_scores(self._fs_traces, H, stride=1)
        self._frozen_q_step = np.array(
            [self._finite_sample_quantile(sc[h], float(self._base_alpha_per_step()[h])) for h in range(H)]
        )
        self._frozen_n = len(sc[0])
        self._frozen = True

    def begin_calibration_trace(self) -> None:
        """Start a new calibration trace: clear forecaster history only (keep score / _fs buffers)."""
        self._obs.clear()

    def reset_episode_state(self) -> None:
        """Clear per-episode bookkeeping; keep score buffers and ACI state."""
        if not getattr(self, "_frozen", True):
            self._flush_fs_trace()
        self._hats_by_decision.clear()
        self._bounds_by_decision.clear()
        self._actual_by_chunk.clear()
        self._n_realized = 0
        self._cov_hits[:] = 0
        self._cov_n[:] = 0
        self._joint_decision_flags.clear()
        self.last_joint_decision_covered = None

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
    def _finite_sample_quantile(scores: deque[float], miscoverage_alpha: float) -> float:
        """Upper-tail score quantile at miscoverage rate ``miscoverage_alpha``."""
        if not scores:
            return 0.0
        arr = np.sort(np.asarray(scores, dtype=float))
        n = arr.size
        rank = int(np.ceil((n + 1) * (1.0 - miscoverage_alpha)))
        rank = min(max(rank, 1), n)
        return float(arr[rank - 1])

    def _min_calib(self, alpha_h: float) -> int:
        return max(int(np.ceil(1.0 / max(alpha_h, 1e-9)) - 1), 1)

    def _alpha_bounds(self, step_idx: int) -> tuple[float, float, float]:
        base = float(self._base_alpha_per_step()[step_idx])
        lo = base / 10.0
        hi = min(1.0, 3.0 * base)
        return base, lo, hi

    def _quantile_alpha(self, step_idx: int) -> float:
        base, lo, hi = self._alpha_bounds(step_idx)
        if not self.cfg.use_aci:
            return base
        return float(np.clip(self._alpha_t[step_idx], lo, hi))

    def _in_cold_start(self, step_idx: int) -> bool:
        if self._frozen:
            return False
        nominal = float(self._base_alpha_per_step()[step_idx])
        return len(self._scores[step_idx]) < self._min_calib(nominal)

    def _lower_from_hat(self, hat: float, step_idx: int) -> float:
        miscoverage = self._quantile_alpha(step_idx)
        if self._in_cold_start(step_idx):
            return float(self.cfg.cold_start_cap * hat)
        if self._frozen and self._frozen_q_step is not None:
            q = float(self._frozen_q_step[step_idx])
        else:
            q = self._finite_sample_quantile(self._scores[step_idx], miscoverage)
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
            self._n_decisions += 1
            if self._in_cold_start(0):
                self._n_cold_decisions += 1
            j = int(decision_chunk)
            self._hats_by_decision[j] = hats.copy()
            self._bounds_by_decision[j] = lbs.copy()
        return lbs

    def _register_score(self, h_idx: int, hat: float, actual: float, lb_at_prediction: float) -> None:
        covered = actual >= lb_at_prediction - 1e-6
        self._cov_n[h_idx] += 1
        if covered:
            self._cov_hits[h_idx] += 1
        s = self._score(hat, actual)
        self._scores[h_idx].append(s)
        if self.cfg.use_aci:
            base, lo, hi = self._alpha_bounds(h_idx)
            err = 1.0 if not covered else 0.0
            delta = (base - err) if not self.cfg.aci_invert_sign else (err - base)
            new_a = float(np.clip(self._alpha_t[h_idx] + self.cfg.aci_gamma * delta, lo, hi))
            self._aci_updates += 1
            if new_a <= lo + 1e-15 or new_a >= hi - 1e-15:
                self._aci_at_clip += 1
            self._alpha_t[h_idx] = new_a

    def _register_marginal_scores(self, actual: float, realized_chunk: int | None) -> None:
        c = int(realized_chunk if realized_chunk is not None else self._n_realized)
        self._n_realized = max(self._n_realized, c + 1)
        h_max = len(self._scores)
        off = int(self.cfg.legacy_step_offset)
        for h in range(1, h_max + 1):
            j = c - (h - 1) - off
            bounds = self._bounds_by_decision.get(j)
            hats = self._hats_by_decision.get(j)
            if bounds is None or hats is None or h > len(bounds):
                continue
            hat = float(hats[h - 1])
            lb = float(bounds[h - 1])
            self._register_score(h - 1, hat, actual, lb)

    def _maybe_complete_joint_decision(self, c: int) -> None:
        h_max = len(self._scores)
        off = int(self.cfg.legacy_step_offset)
        j = c - (h_max - 1) - off
        if j < 0 or j not in self._bounds_by_decision:
            return
        bounds = self._bounds_by_decision[j]
        ok = True
        for step in range(h_max):
            chunk_idx = j + step + off
            act = self._actual_by_chunk.get(chunk_idx)
            if act is None:
                ok = False
                break
            if act < float(bounds[step]) - 1e-6:
                ok = False
        self._joint_decision_flags.append(ok)
        self.last_joint_decision_covered = ok

    def ingest_calibration_sample(self, actual_kbps: float) -> None:
        """Offline warm-start: append scores from (harmonic hat, actual) without MPC bounds."""
        actual = float(max(actual_kbps, 1e-6))
        if self.cfg.freeze_scales:
            self._fs_cur.append(actual)
        hat = self.predict_point_kbps()
        if hat > 0:
            s = self._score(hat, actual)
            for buf in self._scores:
                buf.append(s)
        self._obs.append(actual)

    def observe(self, actual_kbps: float, *, realized_chunk: int | None = None) -> None:
        actual = float(max(actual_kbps, 1e-6))
        c = int(realized_chunk if realized_chunk is not None else self._n_realized)
        self._register_marginal_scores(actual, c)
        self._actual_by_chunk[c] = actual
        self._maybe_complete_joint_decision(c)
        self._obs.append(actual)

    def empirical_coverage_per_step(self) -> np.ndarray:
        out = np.zeros(len(self._scores), dtype=float)
        for i in range(len(self._scores)):
            if self._cov_n[i] > 0:
                out[i] = self._cov_hits[i] / self._cov_n[i]
        return out

    def joint_coverage(self) -> float:
        if not self._joint_decision_flags:
            return float("nan")
        return float(np.mean(self._joint_decision_flags))

    def cold_start_fraction(self) -> float:
        if self._n_decisions <= 0:
            return float("nan")
        return float(self._n_cold_decisions / self._n_decisions)

    def mean_alpha_t(self) -> float:
        if not self.cfg.use_aci:
            return float("nan")
        return float(np.mean(self._alpha_t))

    def aci_clip_fraction(self) -> float:
        if self._aci_updates <= 0:
            return float("nan")
        return float(self._aci_at_clip / self._aci_updates)


__all__ = ["ConformalTubeV2Config", "ConformalThroughputTubeV2", "ScoreKind", "TubeMode"]
