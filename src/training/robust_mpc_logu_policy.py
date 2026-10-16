"""Robust MPC (Yin et al. 2015) with log-utility objective (Tube-v2 baseline)."""

from __future__ import annotations

import itertools

import numpy as np


class RobustMPCLogUtilityPolicy:
    """Harmonic predictor / (1 + max recent rel. error); same MPC shell as TubeMPC."""

    def __init__(self, env, horizon: int = 5, past: int = 5, safety_margin: float = 0.5):
        self.env = env
        self.horizon = int(horizon)
        self.past = int(past)
        self.safety_margin = float(safety_margin)
        self.n = len(env.BITRATE_LEVELS)
        br = np.asarray(env.BITRATE_LEVELS, dtype=float)
        self.log_u = np.log(np.maximum(br / max(br[0], 1e-6), 1e-6))
        self.reb_penalty = 100.0
        self.smooth_weight = 1.0
        self._seq_cache: dict[int, np.ndarray] = {}
        self._tp_hist: list[float] = []
        self._err_hist: list[float] = []
        self._last_pred: float | None = None
        self._last_video = None

    def on_episode_start(self) -> None:
        self._tp_hist = []
        self._err_hist = []
        self._last_pred = None
        self._last_video = None

    def _seqs(self, h: int) -> np.ndarray:
        if h not in self._seq_cache:
            self._seq_cache[h] = np.array(list(itertools.product(range(self.n), repeat=h)), dtype=int)
        return self._seq_cache[h]

    def _observe(self) -> None:
        env = self.env
        vid = getattr(env, "current_video_name", None)
        if env.chunk_idx == 0 or vid != self._last_video:
            self.on_episode_start()
            self._last_video = vid
            return
        actual = float(getattr(env, "last_raw_throughput", 0.0))
        if actual > 0:
            self._tp_hist.append(actual)
            if self._last_pred is not None and actual > 1e-6:
                self._err_hist.append(abs(self._last_pred - actual) / actual)

    def _robust_tp_kbps(self) -> float:
        h = self._tp_hist[-self.past :]
        if not h:
            base = float(getattr(self.env, "last_raw_throughput", 500.0))
        else:
            base = float(len(h) / np.sum(1.0 / np.maximum(h, 1e-6)))
        err = max(self._err_hist[-self.past :], default=0.0)
        pred = base / (1.0 + err)
        self._last_pred = pred
        return max(pred, 10.0)

    def predict(self, obs, deterministic=True):
        env = self.env
        self._observe()
        pred = self._robust_tp_kbps()
        k = int(env.chunk_idx)
        h = min(self.horizon, int(env.max_chunks) - k)
        if h <= 0:
            return self.n - 1, None
        tp_tube = np.full(h, pred, dtype=float)
        sizes = np.array(
            [
                [
                    env.get_chunk_size_bits(int(br), min(k + j, int(env.max_chunks) - 1))
                    for br in env.BITRATE_LEVELS
                ]
                for j in range(h)
            ],
            dtype=float,
        )
        seqs = self._seqs(h)
        n_seq = seqs.shape[0]
        dl = sizes[np.arange(h)[None, :], seqs] / (tp_tube[np.arange(h)] * 1000.0 + 1e-6)
        buf = np.full(n_seq, float(env.buffer_level))
        prev_u = np.full(n_seq, self.log_u[int(getattr(env, "last_bitrate_idx", 0))])
        total = np.zeros(n_seq)
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        margin = self.safety_margin
        safe = np.ones(n_seq, dtype=bool)
        for j in range(h):
            reb = np.maximum(0.0, dl[:, j] - buf)
            buf = np.minimum(np.maximum(0.0, buf - dl[:, j]) + p, b_max)
            safe &= buf >= margin
            lu = self.log_u[seqs[:, j]]
            sm = np.abs(lu - prev_u)
            total += lu - self.reb_penalty * reb - self.smooth_weight * sm
            prev_u = lu
        total[~safe] = -1e30
        return int(seqs[int(np.argmax(total)), 0]), None
