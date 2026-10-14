"""
Tube-MPC host policy: plan over H steps using a conformal throughput tube.

Quality objective uses log-rate utility (no VMAF). Safety: buffer must stay
above ``safety_margin`` under tube-implied download times.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from src.training.conformal_tube import ConformalThroughputTube, ConformalTubeConfig


@dataclass
class TubeMPCConfig:
    horizon: int = 5
    safety_margin: float = 0.5
    reb_penalty: float = 100.0
    smooth_weight: float = 1.0
    tube: ConformalTubeConfig = field(default_factory=ConformalTubeConfig)

    def __post_init__(self):
        self.tube.horizon = self.horizon


class TubeMPCPolicy:
    """Finite-horizon MPC with conformal throughput tube (log-utility QoE)."""

    def __init__(self, env, cfg: TubeMPCConfig | None = None):
        self.env = env
        self.cfg = cfg or TubeMPCConfig()
        self.n = len(env.BITRATE_LEVELS)
        self.tube = ConformalThroughputTube(self.cfg.tube)
        br = np.asarray(env.BITRATE_LEVELS, dtype=float)
        self.log_u = np.log(np.maximum(br / max(br[0], 1e-6), 1e-6))
        self._seq_cache: dict[int, np.ndarray] = {}
        self._last_video = None
        self._last_pred_for_observe: float | None = None

    def on_episode_start(self):
        self.tube.reset()
        self._last_video = None
        self._last_pred_for_observe = None

    def _seqs(self, h: int) -> np.ndarray:
        if h not in self._seq_cache:
            self._seq_cache[h] = np.array(list(itertools.product(range(self.n), repeat=h)), dtype=int)
        return self._seq_cache[h]

    def _observe_throughput(self):
        env = self.env
        vid = getattr(env, "current_video_name", None)
        if env.chunk_idx == 0 or vid != self._last_video:
            self.on_episode_start()
            self._last_video = vid
            return
        actual = float(getattr(env, "last_raw_throughput", 0.0))
        if actual > 0:
            self.tube.observe(actual)

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

    def predict(self, obs, deterministic=True):
        env = self.env
        self._observe_throughput()
        k = int(env.chunk_idx)
        h = min(self.cfg.horizon, int(env.max_chunks) - k)
        if h <= 0:
            return self.n - 1, None

        tp_tube = self.tube.horizon_lower_bounds_kbps(h)
        sizes = self._chunk_sizes(k, h)
        seqs = self._seqs(h)
        n_seq = seqs.shape[0]
        dl = sizes[np.arange(h)[None, :], seqs] / (tp_tube[np.arange(h)] * 1000.0 + 1e-6)

        buf = np.full(n_seq, float(env.buffer_level))
        prev_u = np.full(n_seq, self.log_u[0])
        total = np.zeros(n_seq)
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        margin = float(self.cfg.safety_margin)
        safe = np.ones(n_seq, dtype=bool)

        for j in range(h):
            reb = np.maximum(0.0, dl[:, j] - buf)
            buf = np.minimum(np.maximum(0.0, buf - dl[:, j]) + p, b_max)
            safe &= buf >= margin
            lu = self.log_u[seqs[:, j]]
            sm = np.abs(lu - prev_u) if (k + j) > 0 else np.zeros(n_seq)
            total += lu - self.cfg.reb_penalty * reb - self.cfg.smooth_weight * sm
            prev_u = lu

        total[~safe] = -1e30
        best = int(np.argmax(total))
        return int(seqs[best, 0]), None

    def action_mask(self) -> np.ndarray:
        """Feasible rungs at h=0 under tube + margin (for future MaskablePPO)."""
        env = self.env
        mask = np.zeros(self.n, dtype=bool)
        tp = float(self.tube.point_lower_bound_kbps())
        buf = float(env.buffer_level)
        cap = max(buf - self.cfg.safety_margin, 0.1)
        k = int(env.chunk_idx)
        for j in range(self.n):
            bits = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[j]), k)
            dt = bits / (tp * 1000.0 + 1e-6)
            if dt <= cap:
                mask[j] = True
        if not mask.any():
            mask[0] = True
        return mask


__all__ = ["TubeMPCConfig", "TubeMPCPolicy"]
