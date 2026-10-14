"""
Multi-Video ABR Environment V10.

V10 keeps V9 dynamics/state and is a versioned import target for reproducible
paper experiments. Environment changes (if any) should live here.
"""

from .abr_multi_env_v9 import ABREnv as _ABREnvV9


class ABREnv(_ABREnvV9):
    """V10 environment (currently identical to V9)."""


__all__ = ["ABREnv"]

