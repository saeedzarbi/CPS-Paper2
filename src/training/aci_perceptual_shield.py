"""
OACS shield core: online ACI + per-chunk VMAF-knee + action masks (Paper 2).

Release tag: ``oacs-r1`` (see ``configs/paper2_release.py``). Simulator entry:
``src.environment.paper2_abr_env.Paper2StreamingEnv``.

Extends the Paper 1 static CPS shield with:
  - Prior-seeded online adaptive conformal throughput bounds (no 20-chunk cold start)
  - Exponentially weighted residual quantiles for non-stationary traces
  - Per-chunk VMAF lookup for knee banking
  - ``action_masks()`` for MaskablePPO pre-shielding
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np

from configs.paths import VMAF_SCORES_DIR


# --------------------------------------------------------------------------- #
# Online ACI throughput estimator
# --------------------------------------------------------------------------- #
@dataclass
class ACIConfig:
    alpha: float = 0.10
    gamma_aci: float = 0.04
    decay: float = 0.90
    window: int = 200
    k_predict: int = 5
    prior_scale: float = 0.78
    prior_weight: int = 3
    alpha_min: float = 0.005
    alpha_max: float = 0.45


class OnlineACIEstimator:
    """Harmonic-mean predictor with exponentially weighted conformal lower bound."""

    def __init__(self, cfg: ACIConfig):
        self.cfg = cfg
        self.alpha_t: float = float(cfg.alpha)
        self._obs: deque[float] = deque(maxlen=max(cfg.k_predict, 1))
        self._ratios: deque[float] = deque(maxlen=cfg.window)
        self._last_pred: float | None = None
        self._last_lb: float | None = None
        for _ in range(max(cfg.prior_weight, 0)):
            self._ratios.append(float(cfg.prior_scale))

    def predict(self) -> float:
        if not self._obs:
            return 0.0
        vals = np.clip(np.asarray(self._obs, dtype=float), 1e-6, None)
        pred = float(len(vals) / np.sum(1.0 / vals))
        self._last_pred = pred
        return pred

    def _weighted_quantile(self, level: float) -> float:
        """Weighted lower-tail quantile of residual ratios at miscoverage ``level``."""
        if not self._ratios:
            return self.cfg.prior_scale
        ratios = np.asarray(self._ratios, dtype=float)
        n = ratios.size
        weights = np.array([self.cfg.decay ** (n - 1 - i) for i in range(n)], dtype=float)
        weights /= weights.sum()
        order = np.argsort(ratios)
        ratios_s = ratios[order]
        weights_s = weights[order]
        cum = np.cumsum(weights_s)
        rank = int(np.floor(level * (n + 1)))
        if rank < 1:
            return float(self.cfg.prior_scale)
        idx = min(rank - 1, n - 1)
        # Map finite-sample rank onto weighted order statistic
        target = cum[idx] if idx < cum.size else 1.0
        j = int(np.searchsorted(cum, target, side="left"))
        j = min(j, n - 1)
        return float(ratios_s[j])

    def lower_bound(self) -> float:
        pred = self.predict()
        if pred <= 0.0:
            self._last_lb = 0.0
            return 0.0
        q = min(self._weighted_quantile(self.alpha_t), 1.0)
        lb = pred * max(q, 1e-3)
        self._last_lb = lb
        return lb

    def update(self, actual_kbps: float):
        actual = float(max(actual_kbps, 1e-6))
        lb = self._last_lb if self._last_lb is not None else 0.0
        if lb > 0.0:
            err = float(actual < lb)
            self.alpha_t = float(
                np.clip(
                    self.alpha_t + self.cfg.gamma_aci * (self.cfg.alpha - err),
                    self.cfg.alpha_min,
                    self.cfg.alpha_max,
                )
            )
        if self._last_pred and self._last_pred > 0:
            self._ratios.append(actual / self._last_pred)
        self._obs.append(actual)


# --------------------------------------------------------------------------- #
# Per-chunk VMAF lookup
# --------------------------------------------------------------------------- #
class PerChunkVMAFLookup:
    """Fast (video, chunk, bitrate_kbps) -> VMAF with env fallback."""

    def __init__(self, csv_path: Path | None = None):
        path = csv_path or (VMAF_SCORES_DIR / "vmaf_perchunk.csv")
        self._table: dict[tuple[str, int, int], float] = {}
        self._loaded = False
        if path.exists():
            self._load(path)
            self._loaded = True

    def _load(self, path: Path):
        import pandas as pd

        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        vcol = cols.get("video", "video")
        ccol = cols.get("chunk", "chunk")
        bcol = cols.get("bitrate_kbps", "bitrate_kbps")
        scol = cols.get("vmaf", "vmaf")
        for _, row in df.iterrows():
            key = (str(row[vcol]), int(row[ccol]), int(row[bcol]))
            self._table[key] = float(row[scol])

    def get(self, env, rung_idx: int) -> float:
        try:
            br = int(env.BITRATE_LEVELS[rung_idx])
            slug = getattr(env, "current_video_name", None) or getattr(env, "video_name", None)
            if slug and self._table:
                key = (str(slug), int(env.chunk_idx), br)
                if key in self._table:
                    return self._table[key]
        except Exception:
            pass
        scores = getattr(env, "current_vmaf_scores", None) or {}
        try:
            br = int(env.BITRATE_LEVELS[rung_idx])
            return float(scores.get(br, 35.0))
        except Exception:
            return float(rung_idx) * 10.0

    @property
    def has_perchunk_data(self) -> bool:
        return self._loaded and bool(self._table)


# --------------------------------------------------------------------------- #
# ACI certified shield
# --------------------------------------------------------------------------- #
@dataclass
class ACIShieldConfig:
    enabled: bool = True
    enable_banking: bool = True
    epsilon_vmaf: float = 1.0
    aci: ACIConfig = field(default_factory=ACIConfig)
    safety_margin: float = 0.5
    min_buffer: float = 0.3
    never_upgrade: bool = True


def _max_feasible_rung(env, tp_lb: float, buf: float, margin: float) -> int:
    n = len(env.BITRATE_LEVELS)

    def dl(j: int) -> float:
        bits = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[j]), env.chunk_idx)
        return min(bits / (tp_lb * 1000.0 + 1e-6), 60.0)

    cap = max(buf - margin, 0.1)
    a_max = 0
    for j in range(n):
        if dl(j) <= cap:
            a_max = j
    return a_max


def _knee_rung(env, proposal: int, eps_eff: float, vmaf: PerChunkVMAFLookup) -> int:
    v_prop = vmaf.get(env, proposal)
    for j in range(proposal + 1):
        if v_prop - vmaf.get(env, j) <= eps_eff:
            return j
    return proposal


def compute_action_masks(
    env,
    cfg: ACIShieldConfig,
    est: OnlineACIEstimator,
    vmaf: PerChunkVMAFLookup,
) -> np.ndarray:
    n = len(env.BITRATE_LEVELS)
    mask = np.zeros(n, dtype=bool)
    if not cfg.enabled:
        mask[:] = True
        return mask

    buf = float(getattr(env, "buffer_level", 0.0))
    tp_lb = max(est.lower_bound(), float(getattr(env, "MIN_NETWORK_THROUGHPUT", 0.0)))
    if buf <= cfg.min_buffer:
        mask[0] = True
        return mask

    a_max = _max_feasible_rung(env, tp_lb, buf, cfg.safety_margin)
    j_knee_max = _knee_rung(env, a_max, cfg.epsilon_vmaf, vmaf) if cfg.enable_banking else a_max
    ceiling = j_knee_max if cfg.enable_banking else a_max

    def dl(j: int) -> float:
        bits = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[j]), env.chunk_idx)
        return min(bits / (tp_lb * 1000.0 + 1e-6), 60.0)

    cap = max(buf - cfg.safety_margin, 0.1)
    for j in range(n):
        if j > ceiling:
            continue
        if dl(j) <= cap:
            mask[j] = True
    if not mask.any():
        mask[0] = True
    return mask


def aci_certified_action(
    env,
    action,
    cfg: ACIShieldConfig,
    est: OnlineACIEstimator,
    vmaf: PerChunkVMAFLookup,
):
    info = {
        "banked_bits": 0.0,
        "vmaf_given_up": 0.0,
        "tp_lb_kbps": 0.0,
        "alpha_t": float(est.alpha_t),
        "banked": 0,
        "safety_downgrade": 0,
        "emergency": 0,
        "cold_start": 0,
        "bound_held": 0,
        "perchunk_vmaf_used": 0,
        "buffer_before": 0.0,
    }
    try:
        if not cfg.enabled:
            return int(action), 0, info

        n = len(env.BITRATE_LEVELS)
        a = max(0, min(int(action), n - 1))
        buf = float(getattr(env, "buffer_level", 0.0))
        tp_lb = est.lower_bound()
        tp_lb = max(tp_lb, float(getattr(env, "MIN_NETWORK_THROUGHPUT", 0.0)))
        info["tp_lb_kbps"] = tp_lb
        info["alpha_t"] = float(est.alpha_t)

        def dl(j: int) -> float:
            bits = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[j]), env.chunk_idx)
            return min(bits / (tp_lb * 1000.0 + 1e-6), 60.0)

        def vmaf_at(j: int) -> float:
            used_pc = False
            if vmaf.has_perchunk_data:
                slug = getattr(env, "current_video_name", None)
                br = int(env.BITRATE_LEVELS[j])
                key = (str(slug), int(env.chunk_idx), br)
                used_pc = key in vmaf._table
            v = vmaf.get(env, j)
            return v, used_pc

        if buf <= cfg.min_buffer:
            info["safety_downgrade"] = int(a != 0)
            info["emergency"] = 1
            info["buffer_before"] = buf
            return 0, int(a != 0), info

        va, _ = vmaf_at(a)
        eps_eff = cfg.epsilon_vmaf
        j_knee = a
        if cfg.enable_banking:
            for j in range(a + 1):
                vj, used = vmaf_at(j)
                if used:
                    info["perchunk_vmaf_used"] = 1
                if va - vj <= eps_eff:
                    j_knee = j
                    break

        target = j_knee
        while target > 0 and dl(target) > max(buf - cfg.safety_margin, 0.1):
            target -= 1

        safe = min(a, target) if cfg.never_upgrade else target
        intervened = int(safe != a)

        bits_a = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[a]), env.chunk_idx)
        bits_s = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[safe]), env.chunk_idx)
        info["banked_bits"] = float(max(0.0, bits_a - bits_s))
        vs, _ = vmaf_at(safe)
        info["vmaf_given_up"] = float(max(0.0, va - vs))
        info["banked"] = int(j_knee < a and safe == j_knee)
        info["safety_downgrade"] = int(target < j_knee)
        info["knee_idx"] = int(j_knee)
        info["buffer_before"] = buf
        return int(safe), intervened, info
    except Exception:
        return int(action), 0, info


class ACIPerceptualShieldWrapper(gym.Wrapper):
    """ABR env wrapper with online ACI shield and RL action masks."""

    def __init__(self, env: gym.Env, cfg: ACIShieldConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or ACIShieldConfig()
        self.est = OnlineACIEstimator(self.cfg.aci)
        self.vmaf = PerChunkVMAFLookup()
        self._reset_stats()

    def _reset_stats(self):
        self.interventions = 0
        self.bank_events = 0
        self.safety_events = 0
        self.steps = 0
        self.total_banked_bits = 0.0
        self.total_vmaf_given_up = 0.0
        self.cover_hits = 0
        self.cover_total = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def action_masks(self) -> np.ndarray:
        return compute_action_masks(self.env, self.cfg, self.est, self.vmaf)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.est = OnlineACIEstimator(self.cfg.aci)
        self._reset_stats()
        info = dict(info)
        info["shield_intervention_rate"] = 0.0
        info["alpha_t"] = float(self.est.alpha_t)
        return obs, info

    def step(self, action):
        safe_action, intervened, sinfo = aci_certified_action(
            self.env, action, self.cfg, self.est, self.vmaf
        )
        tp_lb = float(sinfo.get("tp_lb_kbps", 0.0))

        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        realized = float(info.get("throughput", getattr(self.env, "last_raw_throughput", 0.0)))
        if tp_lb > 0.0 and realized > 0.0:
            self.cover_total += 1
            self.cover_hits += int(realized >= tp_lb)
            sinfo["bound_held"] = int(realized >= tp_lb)
        self.est.update(realized)

        self.steps += 1
        self.interventions += int(intervened)
        self.bank_events += int(sinfo.get("banked", 0))
        self.safety_events += int(sinfo.get("safety_downgrade", 0))
        self.total_banked_bits += float(sinfo.get("banked_bits", 0.0))
        self.total_vmaf_given_up += float(sinfo.get("vmaf_given_up", 0.0))

        info = dict(info)
        info["shielded_action"] = int(safe_action)
        info["shield_intervened"] = int(intervened)
        info["shield_intervention_rate"] = self.interventions / max(self.steps, 1)
        info["banked_bits"] = float(sinfo.get("banked_bits", 0.0))
        info["vmaf_given_up"] = float(sinfo.get("vmaf_given_up", 0.0))
        info["tp_lb_kbps"] = tp_lb
        info["alpha_t"] = float(self.est.alpha_t)
        info["conformal_coverage"] = (self.cover_hits / self.cover_total) if self.cover_total else float("nan")
        info["bound_held"] = int(sinfo.get("bound_held", 0))
        info["emergency"] = int(sinfo.get("emergency", 0))
        info["cold_start"] = int(sinfo.get("cold_start", 0))
        info["perchunk_vmaf_used"] = int(sinfo.get("perchunk_vmaf_used", 0))
        info["buffer_before"] = float(sinfo.get("buffer_before", 0.0))
        info["safety_downgrade"] = int(sinfo.get("safety_downgrade", 0))
        return obs, reward, terminated, truncated, info


__all__ = [
    "ACIConfig",
    "OnlineACIEstimator",
    "PerChunkVMAFLookup",
    "ACIShieldConfig",
    "aci_certified_action",
    "compute_action_masks",
    "ACIPerceptualShieldWrapper",
]
