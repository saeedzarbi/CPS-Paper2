"""Tube-MPC planner using ``ConformalThroughputTubeV2`` (Tube-v2)."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.training.conformal_tube_v2 import ConformalThroughputTubeV2, ConformalTubeV2Config

PruningMode = Literal["none", "monotone", "beam"]


@dataclass
class TubeV2MPCConfig:
    horizon: int = 5
    safety_margin: float = 0.5
    reb_penalty: float = 100.0
    smooth_weight: float = 1.0
    pruning: PruningMode = "none"
    beam_width: int = 64
    split_objective: bool = False
    tube_kind: str = "v2"
    # Pre-refill safety (arm suffix _safe): every step h must satisfy
    # d_h(a, L) <= B^{(h-1)}(a, L) - margin on the conformal lower-bound trajectory
    # (no planned rebuffer); if no sequence qualifies, fall back to the lowest bitrate.
    safe_prerefill: bool = False
    tube: ConformalTubeV2Config = field(default_factory=ConformalTubeV2Config)

    def __post_init__(self) -> None:
        self.tube.horizon = int(self.horizon)


class TubeV2MPCPolicy:
    """Finite-horizon MPC with a real multi-step conformal tube."""

    def __init__(self, env, cfg: TubeV2MPCConfig | None = None):
        self.env = env
        self.cfg = cfg or TubeV2MPCConfig()
        self.n = len(env.BITRATE_LEVELS)
        if self.cfg.tube_kind == "maxscore":
            from src.training.conformal_tube_v3 import MaxScoreConformalTube

            self.tube = MaxScoreConformalTube(self.cfg.tube)
        else:
            self.tube = ConformalThroughputTubeV2(self.cfg.tube)
        br = np.asarray(env.BITRATE_LEVELS, dtype=float)
        self.log_u = np.log(np.maximum(br / max(br[0], 1e-6), 1e-6))
        self._seq_cache: dict[int, np.ndarray] = {}
        self._last_video = None
        self.max_step_down = 1

    def on_episode_start(self) -> None:
        self.tube.reset_episode_state()
        self._last_video = None

    def _seqs(self, h: int) -> np.ndarray:
        if h not in self._seq_cache:
            self._seq_cache[h] = np.array(list(itertools.product(range(self.n), repeat=h)), dtype=int)
        return self._seq_cache[h]

    def _observe_throughput(self) -> None:
        env = self.env
        vid = getattr(env, "current_video_name", None)
        if env.chunk_idx == 0 or vid != self._last_video:
            self.on_episode_start()
            self._last_video = vid
            return
        actual = float(getattr(env, "last_raw_throughput", 0.0))
        if actual > 0:
            self.tube.observe(actual, realized_chunk=int(env.chunk_idx) - 1)

    def _chunk_sizes(self, k: int, h: int) -> np.ndarray:
        max_c = int(getattr(self.env, "max_chunks", k + h))
        return np.array(
            [
                [
                    self.env.get_chunk_size_bits(int(br), min(k + j, max_c - 1))
                    for br in self.env.BITRATE_LEVELS
                ]
                for j in range(h)
            ],
            dtype=float,
        )

    def _filter_monotone(self, seqs: np.ndarray, a_prev: int) -> np.ndarray:
        ok = []
        for row in seqs:
            prev = a_prev
            good = True
            for j in range(row.shape[0]):
                cur = int(row[j])
                if cur < prev - self.max_step_down or cur > prev + 1:
                    good = False
                    break
                prev = cur
            if good:
                ok.append(row)
        if not ok:
            return seqs[:1]
        return np.asarray(ok, dtype=int)

    def _beam_prune(self, seqs: np.ndarray, k: int, h: int, tp_tube: np.ndarray) -> np.ndarray:
        if h <= 1 or self.cfg.beam_width >= len(seqs):
            return seqs
        sizes = self._chunk_sizes(k, h)
        scores = []
        for row in seqs:
            dl0 = sizes[0, int(row[0])] / (tp_tube[0] * 1000.0 + 1e-6)
            scores.append((-dl0, row))
        scores.sort(key=lambda x: x[0])
        keep = [row for _, row in scores[: self.cfg.beam_width]]
        return np.asarray(keep, dtype=int)

    def _select_seqs(self, h: int, a_prev: int, k: int, tp_tube: np.ndarray) -> np.ndarray:
        seqs = self._seqs(h)
        mode = self.cfg.pruning
        if mode == "monotone":
            seqs = self._filter_monotone(seqs, a_prev)
        elif mode == "beam":
            seqs = self._beam_prune(seqs, k, h, tp_tube)
        return seqs

    def plan_action(self) -> int:
        env = self.env
        self._observe_throughput()
        k = int(env.chunk_idx)
        a_prev = int(getattr(env, "last_bitrate_idx", 0))
        h = min(self.cfg.horizon, int(env.max_chunks) - k)
        if h <= 0:
            return self.n - 1

        tp_tube = self.tube.horizon_lower_bounds_kbps(h, decision_chunk=k)
        sizes = self._chunk_sizes(k, h)
        seqs = self._select_seqs(h, a_prev, k, tp_tube)
        n_seq = seqs.shape[0]
        dl = sizes[np.arange(h)[None, :], seqs] / (tp_tube[np.arange(h)] * 1000.0 + 1e-6)

        buf = np.full(n_seq, float(env.buffer_level))
        prev_u = np.full(n_seq, self.log_u[a_prev])
        total = np.zeros(n_seq)
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        margin = float(self.cfg.safety_margin)
        safe = np.ones(n_seq, dtype=bool)

        if self.cfg.split_objective:
            # Objective (rebuffer penalty) uses the point forecast; the safety
            # constraint (buffer >= margin) uses the conformal lower bound.
            hat = float(self.tube.predict_point_kbps())
            tp_point = np.maximum(np.full(h, hat if hat > 0 else 0.0), tp_tube)
            dl_obj = sizes[np.arange(h)[None, :], seqs] / (tp_point[np.arange(h)] * 1000.0 + 1e-6)
        else:
            dl_obj = dl
        buf_obj = buf.copy()
        for j in range(h):
            reb = np.maximum(0.0, dl_obj[:, j] - buf_obj)
            buf_obj = np.minimum(np.maximum(0.0, buf_obj - dl_obj[:, j]) + p, b_max)
            if self.cfg.safe_prerefill:
                safe &= dl[:, j] <= buf - margin
            buf = np.minimum(np.maximum(0.0, buf - dl[:, j]) + p, b_max)
            safe &= buf >= margin
            aj = seqs[:, j]
            lu = self.log_u[aj]
            sm = np.abs(lu - prev_u)
            total += lu - self.cfg.reb_penalty * reb - self.cfg.smooth_weight * sm
            prev_u = lu

        if self.cfg.safe_prerefill and not bool(np.any(safe)):
            self.n_fallback = getattr(self, "n_fallback", 0) + 1
            return 0
        total[~safe] = -1e30
        return int(seqs[int(np.argmax(total)), 0])

    def predict(self, obs, deterministic=True):
        return self.plan_action(), None

    def benchmark_plan_latency_us(self, repeats: int = 20) -> float:
        """Mean plan_action latency (microseconds), single-threaded."""
        self._observe_throughput()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            self.plan_action()
            samples.append((time.perf_counter_ns() - t0) / 1000.0)
        return float(np.mean(samples))


__all__ = ["TubeV2MPCConfig", "TubeV2MPCPolicy", "PruningMode"]
