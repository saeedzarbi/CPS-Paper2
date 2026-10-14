"""
Multi-Video ABR Environment V9.

V9 is a versioned import target to make the "final" paper experiments
reproducible. For now it aliases V6 behavior (same dynamics/state),
but it gives you a stable hook to introduce true V9 environment changes later.
"""

from .abr_multi_env_v6 import ABREnv as _ABREnvV6


class ABREnv(_ABREnvV6):
    """V9 environment (currently identical to V6)."""


__all__ = ["ABREnv"]

