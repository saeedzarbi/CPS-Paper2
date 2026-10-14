"""
Paper 2 streaming simulator entry point (not the v18 filename lineage).

Dynamics are identical to the proven Paper 1 operating point; Paper 2 code should
import ``Paper2StreamingEnv`` (or alias ``ABREnv``) from here instead of
``abr_multi_env_v18`` so experiments and server jobs read as OACS / Paper 2.
"""

from __future__ import annotations

from .abr_multi_env_v18 import ABREnv as _ABREnvV18


class Paper2StreamingEnv(_ABREnvV18):
    """ABR simulator for Paper 2 (OACS) campaigns — same physics as v18 baseline."""


# Preferred import name in new Paper 2 modules
ABREnv = Paper2StreamingEnv

__all__ = ["Paper2StreamingEnv", "ABREnv"]
