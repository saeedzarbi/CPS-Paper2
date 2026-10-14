"""
Multi-Video ABR Environment V12.

V12 keeps V11 dynamics/state and provides a versioned import target for the
"final" paper experiments. If you introduce true V12 environment changes,
implement them here.
"""

from .abr_multi_env_v11 import ABREnv as _ABREnvV11


class ABREnv(_ABREnvV11):
    """V12 environment (currently identical to V11)."""


__all__ = ["ABREnv"]

