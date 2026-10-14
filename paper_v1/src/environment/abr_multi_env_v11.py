"""
Multi-Video ABR Environment V11.

V11 keeps V10/V9 dynamics and is a versioned import target for reproducible
paper experiments. If you introduce true V11 environment changes, implement
them here.
"""

from .abr_multi_env_v10 import ABREnv as _ABREnvV10


class ABREnv(_ABREnvV10):
    """V11 environment (currently identical to V10)."""


__all__ = ["ABREnv"]

