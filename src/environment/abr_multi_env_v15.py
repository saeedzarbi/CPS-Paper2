"""
Multi-Video ABR Environment V15 (low-latency operating point).

V15 == V14 dynamics and corrected reward scale, but at a LOW-LATENCY buffer
operating point. This is the regime where a runtime safety layer and a binding
rebuffering constraint are expected to actually matter.

Motivation
----------
Under the V14 broadband setup the playback buffer cap was 30 s. A well-trained
policy simply parks a large buffer, so it (a) almost never rebuffers and (b)
almost never proposes an action the shield needs to repair (<0.5% intervention
rate, zero measured effect). That makes both the shield and VMAF-awareness look
inert -- not because they are bad ideas, but because the operating point is too
forgiving to exercise them.

Low-latency live streaming is the opposite regime: the client holds only a few
seconds of buffer, so a single throughput dip can cause a stall, and decision-
time safety becomes a live concern. V15 models this by shrinking the buffer cap
(and the target / Lyapunov reference proportionally). Everything else -- state,
VBR chunk sizes, per-video VMAF, chunk duration, and the corrected rebuffering
penalty (100.0) -- is inherited unchanged from V14, so results are directly
comparable across operating points.

Only the buffer geometry changes:
  BUFFER_MAX    : 30 -> 6   (low-latency cap)
  BUFFER_TARGET : 15 -> 3   (aim for half the cap)
  B_REF         :  8 -> 2   (Lyapunov reference scaled with the cap)
"""

from __future__ import annotations

import json

from .abr_multi_env_v14 import ABREnv as _ABREnvV14


class ABREnv(_ABREnvV14):
    """V14 environment at a low-latency (small-buffer) operating point."""

    BUFFER_MAX = 6.0
    BUFFER_TARGET = 3.0
    B_REF = 2.0

    def _load_traces(self):
        """Load traces, robustly skipping non-trace sidecar JSON files.

        The trace generators write a ``trace_stats.json`` calibration record into
        the same directory. The base loader globs ``*.json`` indiscriminately, so
        that sidecar is picked up as a "trace" and later crashes on
        ``current_trace['throughput_kbps']``. Here we keep only dicts that expose
        a non-empty ``throughput_kbps`` series.
        """
        trace_files = sorted(self.trace_dir.glob("*.json"))
        loaded = []
        for f in trace_files:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if isinstance(d, dict) and d.get("throughput_kbps"):
                loaded.append(d)
        self.traces = loaded if loaded else [{"throughput_kbps": [2000] * 1000}]


__all__ = ["ABREnv"]
