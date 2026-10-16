"""Fixed-quantile / constant-rho MPC baselines (no online conformal)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "results" / "tube_v2"


def _harmonic(vals: list[float]) -> float:
    if not vals:
        return 0.0
    v = np.clip(np.asarray(vals, dtype=float), 1e-6, None)
    return float(len(v) / np.sum(1.0 / v))


def fit_relative_error_quantile(
    trace_dirs: list[Path],
    *,
    alpha: float = 0.10,
    k_predict: int = 5,
) -> float:
    scores: list[float] = []
    for tdir in trace_dirs:
        if not tdir.is_dir():
            continue
        for path in sorted(tdir.glob("*.json")):
            if path.name.startswith("trace_stats"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tp = data.get("throughput_kbps", [])
            except (json.JSONDecodeError, OSError):
                continue
            hist: list[float] = []
            for actual in tp:
                a = float(max(actual, 1e-6))
                hat = _harmonic(hist[-k_predict:])
                if hat > 0:
                    scores.append(max(0.0, (hat - a) / hat))
                hist.append(a)
    if not scores:
        return float(alpha)
    arr = np.sort(np.asarray(scores, dtype=float))
    n = arr.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(arr[rank - 1])


def quantile_cache_path(
    trace_set: str,
    split: str,
    alpha: float,
    k_predict: int,
) -> Path:
    safe = f"fixed_quantile_q_{trace_set}_{split}_a{alpha:.4f}_k{k_predict}.json"
    return CACHE_ROOT / safe


def load_or_fit_quantile(
    trace_dirs: list[Path],
    *,
    trace_set: str,
    split: str = "calib",
    alpha: float = 0.10,
    k_predict: int = 5,
    force_refit: bool = False,
) -> float:
    path = quantile_cache_path(trace_set, split, alpha, k_predict)
    if path.is_file() and not force_refit:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("q", alpha))
    q = fit_relative_error_quantile(trace_dirs, alpha=alpha, k_predict=k_predict)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "q": q,
        "alpha": alpha,
        "k_predict": k_predict,
        "trace_set": trace_set,
        "split": split,
        "trace_dirs": [str(p) for p in trace_dirs if p.is_dir()],
        "note": "L = hat*(1-q); compare to fixed_rho with rho = 1-q",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return q


def log_fitted_quantile(q: float, alpha: float, path: Path) -> None:
    print(
        f"[mpc_fixed_quantile] fitted q={q:.6f} (alpha={alpha}, implied rho={1.0 - q:.6f}) path={path}"
    )


class FixedQuantileMPCPolicy:
    def __init__(self, env, q: float, horizon: int = 5, safety_margin: float = 0.5, k_predict: int = 5):
        self.env = env
        self.q = float(q)
        self.horizon = int(horizon)
        self.safety_margin = float(safety_margin)
        self.k_predict = int(k_predict)
        self.n = len(env.BITRATE_LEVELS)
        br = np.asarray(env.BITRATE_LEVELS, dtype=float)
        self.log_u = np.log(np.maximum(br / max(br[0], 1e-6), 1e-6))
        self.reb_penalty = 100.0
        self.smooth_weight = 1.0
        self._hist: list[float] = []
        self._seq_cache: dict[int, np.ndarray] = {}
        self._last_video = None

    def on_episode_start(self) -> None:
        self._hist = []
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
            self._hist.append(actual)

    def _lb_kbps(self) -> float:
        hat = _harmonic(self._hist[-self.k_predict :])
        if hat <= 0:
            hat = float(getattr(self.env, "last_raw_throughput", 500.0))
        return max(hat * (1.0 - self.q), 10.0)

    def predict(self, obs, deterministic=True):
        env = self.env
        self._observe()
        pred = self._lb_kbps()
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


class FixedRhoMPCPolicy(FixedQuantileMPCPolicy):
    """Constant discount ``lb = rho * harmonic`` (no calibration)."""

    def __init__(self, env, rho: float, **kwargs):
        super().__init__(env, q=1.0 - float(rho), **kwargs)
        self.rho = float(rho)

    def _lb_kbps(self) -> float:
        hat = _harmonic(self._hist[-self.k_predict :])
        if hat <= 0:
            hat = float(getattr(self.env, "last_raw_throughput", 500.0))
        return max(self.rho * hat, 10.0)
