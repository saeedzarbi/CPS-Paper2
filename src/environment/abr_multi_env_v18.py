"""
5G low-latency environment, V18 (operating point for the Certified Perceptual Shield).

V18 == V14 dynamics and corrected reward scale, at a 5G low-latency operating
point: a moderate buffer (default 12 s) that, on high-throughput 5G links, keeps
the perceptually-saturated top rungs FEASIBLE. That is precisely the regime where
the certified perceptual shield's VMAF-knee bandwidth banking has headroom: it can
trade perceptually-worthless top-rung bytes for a protective buffer margin.

Only the buffer geometry changes vs. V14; the byte-driven download/rebuffering
dynamics are identical, isolating the shield's effect. The robust trace loader
(from V15) is reused so 5G trace directories containing a ``trace_stats.json``
sidecar load cleanly.
"""

from __future__ import annotations

from .abr_multi_env_v14 import ABREnv as _ABREnvV14
from .abr_multi_env_v15 import ABREnv as _ABREnvV15


class ABREnv(_ABREnvV14):
    """V14 dynamics at a 5G low-latency operating point (moderate buffer)."""

    BUFFER_MAX = 12.0
    BUFFER_TARGET = 6.0
    B_REF = 4.0

    # Robust loader: skip non-trace sidecar JSON (e.g. trace_stats.json).
    _load_traces = _ABREnvV15._load_traces


__all__ = ["ABREnv"]
