"""
Safety shield (V14, reviewer-response): deterministic action projection.

V14 is behaviourally identical to V12 for every existing configuration, and adds
ONE thing the review demands (P0.2 / fatal flaw F2): an explicit
*highest-feasible-index* selection rule that shares the exact same soft-safe
feasible set as the VMAF-aware rule.

------------------------------------------------------------------------------
Why this matters
------------------------------------------------------------------------------
The manuscript proves (Sec. 3.6) that on a VMAF ladder that is monotone in the
representation index, VMAF-ordered selection collapses onto highest-feasible-
index selection. The review's central objection is that *no experiment isolates
the perceptual component*: the reported "VMAF-aware vs. legacy" gain is really a
change in the feasibility predicate (multiplicative ``buf * soft_tolerance``
instead of additive ``buf - margin``), not a change in *how the feasible set is
ranked*.

The fix is a controlled A/B with the ranking rule as the ONLY difference:

    * ``selection="vmaf"``  -> pick argmax VMAF over the soft-safe set (V12 behaviour)
    * ``selection="index"`` -> pick the highest index over the *same* soft-safe set

On the per-video (monotone) ladder these two MUST return identical actions,
which turns the review's theoretical objection into a reproducible, quantitative
result: the paired difference between the two arms is exactly zero, proving the
perceptual ranking is inert on this ladder. On a per-chunk (non-monotone) ladder
they can diverge, which is where perceptual awareness becomes a live design
choice.

Everything else is a verbatim port of ``safety_shield_v12.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class ShieldConfig:
    level: str = "light"  # 'off' | 'light' | 'strong'
    catastrophic_ratio: float = 2.0
    safe_margin_light: float = 0.5
    safe_margin_strong: float = 1.5
    safety_tp_scale: float = 0.90

    # Risk gate (download-time vs buffer)
    only_when_risky: bool = False
    risky_dl_over_buf_ratio: float = 1.10

    # ---- VMAF-Aware Soft Projection (V12.1) ----
    vmaf_aware: bool = False
    soft_tolerance: float = 1.0
    vmaf_loss_budget: float = 8.0

    # ---- V14: ranking rule for the soft-safe candidate set ----
    # "vmaf"  -> argmax VMAF (perceptual, V12 default)
    # "index" -> highest feasible index over the SAME soft-safe set
    #            (content-blind isolation baseline; ignores the VMAF budget).
    selection: str = "vmaf"

    # ---- Lookahead-Rollout Gate (V12.2) ----
    lookahead_horizon: int = 0
    lookahead_min_buffer: float = 0.5
    lookahead_tp_scale: float = 0.0


def _vmaf_for_idx(env, idx: int) -> float:
    """Per-video VMAF lookup; falls back to a monotone proxy if unavailable."""
    try:
        br = int(env.BITRATE_LEVELS[idx])
        scores = getattr(env, "current_vmaf_scores", None)
        if scores:
            return float(scores.get(br, 35.0))
    except Exception:
        pass
    return float(idx) * 10.0


def safe_adjust_action(env, action: int, cfg: ShieldConfig) -> tuple[int, int]:
    """Returns (safe_action, intervened_flag)."""
    try:
        if cfg.level == "off":
            return int(action), 0

        buf = float(getattr(env, "buffer_level", 0.0))
        cur_idx = int(action)
        cur_idx = max(0, min(cur_idx, len(env.BITRATE_LEVELS) - 1))

        if buf <= 0.3:
            return 0, int(cur_idx != 0)

        trace_tp = getattr(env, "current_trace", None)
        if trace_tp and "throughput_kbps" in trace_tp:
            tp_idx = int(env.chunk_idx * env.CHUNK_DURATION) % len(trace_tp["throughput_kbps"])
            trace_tp_val = float(trace_tp["throughput_kbps"][tp_idx])
        else:
            trace_tp_val = 2000.0

        last_tp = getattr(env, "last_raw_throughput", 2000.0)
        tp_est = min(trace_tp_val, last_tp) * cfg.safety_tp_scale
        tp_est = max(tp_est, env.MIN_NETWORK_THROUGHPUT)

        def dl_time_for(idx: int) -> float:
            br = int(env.BITRATE_LEVELS[idx])
            chunk_bits = env.get_chunk_size_bits(br, env.chunk_idx)
            return min(chunk_bits / (tp_est * 1000.0 + 1e-6), 60.0)

        if cfg.only_when_risky:
            dt_req = dl_time_for(cur_idx)
            if dt_req <= max(buf, 0.1) * cfg.risky_dl_over_buf_ratio:
                return cur_idx, 0

        # ---- Lookahead-Rollout Gate (V12.2) ----
        if cfg.lookahead_horizon and cfg.lookahead_horizon > 0:
            la_scale = cfg.lookahead_tp_scale if cfg.lookahead_tp_scale > 0 else cfg.safety_tp_scale
            la_tp = min(trace_tp_val, last_tp) * la_scale
            la_tp = max(la_tp, env.MIN_NETWORK_THROUGHPUT)
            br_kbps = int(env.BITRATE_LEVELS[cur_idx])
            buf_sim = float(buf)
            safe_under_lookahead = True
            for h in range(int(cfg.lookahead_horizon)):
                cidx_h = min(env.chunk_idx + h, env.max_chunks - 1)
                bits_h = env.get_chunk_size_bits(br_kbps, cidx_h)
                dt_h = min(bits_h / (la_tp * 1000.0 + 1e-6), 60.0)
                if dt_h > buf_sim:
                    safe_under_lookahead = False
                    break
                buf_sim = max(0.0, buf_sim - dt_h) + env.CHUNK_DURATION
                if buf_sim < cfg.lookahead_min_buffer:
                    safe_under_lookahead = False
                    break
            if safe_under_lookahead:
                return cur_idx, 0

        # ---- VMAF-Aware Soft Projection (V12.1) with V14 selection rule ----
        if cfg.vmaf_aware:
            cur_dt = dl_time_for(cur_idx)
            if cur_dt <= buf * cfg.catastrophic_ratio:
                return cur_idx, 0

            cur_vmaf = _vmaf_for_idx(env, cur_idx)
            soft_thresh = buf * cfg.soft_tolerance

            # Build candidates over indices [0, cur_idx]; never upgrade.
            soft_safe = []  # (idx, dt, vmaf, vmaf_loss)
            for j in range(cur_idx + 1):
                dt_j = dl_time_for(j)
                if dt_j <= soft_thresh:
                    vmaf_j = _vmaf_for_idx(env, j)
                    soft_safe.append((j, dt_j, vmaf_j, max(0.0, cur_vmaf - vmaf_j)))

            if soft_safe:
                if cfg.selection == "index":
                    # Content-blind isolation baseline: highest feasible index
                    # over the SAME soft-safe set (VMAF budget does not apply).
                    best = max(soft_safe, key=lambda x: x[0])
                    return best[0], int(best[0] != cur_idx)
                # Default V12 behaviour: prefer within-budget, then argmax VMAF.
                in_budget = [c for c in soft_safe if c[3] <= cfg.vmaf_loss_budget]
                pool = in_budget if in_budget else soft_safe
                best = max(pool, key=lambda x: (x[2], x[0], -x[1]))
                return best[0], int(best[0] != cur_idx)

            # No soft-safe option in [0, cur_idx]: emergency fallback.
            return 0, int(cur_idx != 0)

        # ---- Legacy V12 logic (unchanged) ----
        if cfg.level == "light":
            dt = dl_time_for(cur_idx)
            if dt > buf * cfg.catastrophic_ratio:
                original = cur_idx
                cur_idx = max(0, cur_idx - 1)
                dt = dl_time_for(cur_idx)
                if dt > buf * cfg.catastrophic_ratio:
                    for fallback in range(cur_idx, -1, -1):
                        if dl_time_for(fallback) <= buf - cfg.safe_margin_light:
                            return fallback, 1
                    return 0, 1
                return cur_idx, int(cur_idx != original)
            return cur_idx, 0

        original = cur_idx
        margin = cfg.safe_margin_strong
        for _ in range(cur_idx):
            if dl_time_for(cur_idx) <= buf - margin:
                break
            cur_idx -= 1
        return cur_idx, int(cur_idx != original)
    except Exception:
        return int(action), 0


class SafetyShieldWrapper(gym.Wrapper):
    """Wraps an ABR env and projects the action before stepping."""

    def __init__(self, env: gym.Env, cfg: ShieldConfig):
        super().__init__(env)
        self.cfg = cfg
        self.interventions = 0
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.interventions = 0
        self.steps = 0
        info = dict(info)
        info["shield_intervention_rate"] = 0.0
        return obs, info

    def step(self, action):
        safe_action, intervened = safe_adjust_action(self.env, action, self.cfg)
        self.interventions += int(intervened)
        self.steps += 1

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        info = dict(info)
        info["shielded_action"] = int(safe_action)
        info["shield_intervened"] = int(intervened)
        info["shield_intervention_rate"] = float(self.interventions) / max(float(self.steps), 1.0)
        return obs, reward, terminated, truncated, info


__all__ = ["ShieldConfig", "SafetyShieldWrapper", "safe_adjust_action"]
