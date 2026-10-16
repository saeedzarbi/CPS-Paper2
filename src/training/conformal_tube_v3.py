"""Tube-v3: simultaneous (max-score) conformal joint tube.

Score for a completed decision j (bounds for chunks j+1..j+H, all built from the
same point forecast hat_j):

    S_j = max_h  s_{j,h} / sigma_{j,h},   s_{j,h} = max(0, (hat_j - y_{j+h}) / hat_j)

where sigma_{j,h} > 0 is a per-step scale frozen at decision time (the
per-step empirical (1-alpha) score quantile, floored).  One quantile
q = Quant_{1-alpha}(S) gives L_{j,h} = hat_j * (1 - q * sigma_{j,h}), so
P(all H steps covered) >= 1 - alpha under exchangeability of decisions.
Bounds are recorded *before* any update (pre-update bounds), and the
per-decision joint flag is evaluated against those stored bounds.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.training.conformal_tube_v2 import ConformalThroughputTubeV2, ConformalTubeV2Config

SIGMA_FLOOR = 0.02


class MaxScoreConformalTube(ConformalThroughputTubeV2):
    def __init__(self, cfg: ConformalTubeV2Config | None = None):
        cfg = cfg or ConformalTubeV2Config()
        cfg.mode = "per_step"
        cfg.use_aci = False
        super().__init__(cfg)
        self._joint_scores: deque[float] = deque(maxlen=self.cfg.window)
        self._sigmas_by_decision: dict[int, np.ndarray] = {}
        # offline warm-start state
        self._cal_hats: deque[float] = deque(maxlen=len(self._scores) + 1)
        self._cal_actuals: deque[float] = deque(maxlen=len(self._scores) + 1)
        self.last_joint_q: float = float("nan")

    def reset(self) -> None:
        super().reset()
        self._joint_scores.clear()
        self._sigmas_by_decision.clear()
        self._cal_hats.clear()
        self._cal_actuals.clear()

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        self._sigmas_by_decision = {}
        if hasattr(self, "_cal_hats"):
            self._cal_hats.clear()
            self._cal_actuals.clear()

    # ---- scales / quantile ----
    def _sigmas(self, h: int) -> np.ndarray:
        a = float(self.cfg.alpha)
        out = np.empty(h, dtype=float)
        for i in range(h):
            buf = self._scores[i]
            q = self._finite_sample_quantile(buf, a) if len(buf) else 0.0
            out[i] = max(q, SIGMA_FLOOR)
        return out

    def _joint_ready(self) -> bool:
        return len(self._joint_scores) >= self._min_calib(float(self.cfg.alpha))

    def horizon_lower_bounds_kbps(self, horizon=None, *, decision_chunk=None) -> np.ndarray:
        h = int(horizon if horizon is not None else self.cfg.horizon)
        h = max(min(h, len(self._scores)), 1)
        hat = self.predict_point_kbps()
        if hat <= 0:
            hat = float(self.cfg.min_kbps)
        hats = np.full(h, hat, dtype=float)
        sig = self._sigmas(len(self._scores))
        self._n_decisions += 1
        if not self._joint_ready():
            self._n_cold_decisions += 1
            lbs = np.full(h, float(self.cfg.cold_start_cap * hat))
            self.last_joint_q = float("nan")
        else:
            q = self._finite_sample_quantile(self._joint_scores, float(self.cfg.alpha))
            self.last_joint_q = q
            lbs = hat * (1.0 - q * sig[:h])
        lbs = np.maximum(lbs, self.cfg.min_kbps)
        if decision_chunk is not None:
            j = int(decision_chunk)
            # pre-update bounds and frozen scales
            self._hats_by_decision[j] = hats.copy()
            self._bounds_by_decision[j] = lbs.copy()
            self._sigmas_by_decision[j] = sig.copy()
        return lbs

    def _register_score(self, h_idx, hat, actual, lb_at_prediction) -> None:
        covered = actual >= lb_at_prediction - 1e-6
        self._cov_n[h_idx] += 1
        if covered:
            self._cov_hits[h_idx] += 1
        self._scores[h_idx].append(self._score(hat, actual))

    def _maybe_complete_joint_decision(self, c: int) -> None:
        h_max = len(self._scores)
        j = c - h_max
        if j < 0 or j not in self._bounds_by_decision:
            return
        bounds = self._bounds_by_decision[j]
        hat = float(self._hats_by_decision[j][0])
        sig = self._sigmas_by_decision.get(j)
        if len(bounds) < h_max or sig is None:
            return  # truncated horizon near episode end: not a full joint sample
        ok = True
        smax = 0.0
        for step in range(h_max):
            act = self._actual_by_chunk.get(j + step + 1)
            if act is None:
                return
            if act < float(bounds[step]) - 1e-6:
                ok = False
            smax = max(smax, self._score(hat, act) / float(sig[step]))
        self._joint_decision_flags.append(ok)
        self.last_joint_decision_covered = ok
        self._joint_scores.append(smax)  # update AFTER evaluating pre-update bounds
        for d in (self._bounds_by_decision, self._hats_by_decision, self._sigmas_by_decision):
            d.pop(j, None)

    def ingest_calibration_sample(self, actual_kbps: float) -> None:
        """Offline warm-start with true h-step scores and joint max-scores."""
        actual = float(max(actual_kbps, 1e-6))
        hat = self.predict_point_kbps()
        H = len(self._scores)
        self._cal_hats.append(hat)  # forecast made before seeing `actual`
        self._cal_actuals.append(actual)
        n = len(self._cal_hats)
        # h-step score: forecast made h-1 samples ago vs current actual
        for h in range(1, min(H, n) + 1):
            hh = self._cal_hats[n - h]
            if hh > 0:
                self._scores[h - 1].append(self._score(hh, actual))
        if n >= H:
            h0 = self._cal_hats[n - H]
            if h0 > 0:
                sig = self._sigmas(H)
                acts = list(self._cal_actuals)[n - H : n]
                s = max(self._score(h0, a) / float(sig[i]) for i, a in enumerate(acts))
                self._joint_scores.append(s)
        self._obs.append(actual)


__all__ = ["MaxScoreConformalTube"]
