"""Paper 2 streaming environment — clean wall-clock simulator (h3)."""

from .clean_abr_env import ABREnv, CleanABREnv

Paper2StreamingEnv = CleanABREnv

__all__ = ["CleanABREnv", "Paper2StreamingEnv", "ABREnv"]
