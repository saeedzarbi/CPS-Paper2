"""Deterministic episode seeds for paired ABR evaluation."""
from __future__ import annotations

import hashlib
import random

import numpy as np

DEFAULT_MASTER_SEED = 42


def episode_seed(master_seed: int, trace_key: str, episode: int) -> int:
    payload = f"{int(master_seed)}:{trace_key}:{int(episode)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def apply_episode_seeds(master_seed: int, trace_key: str, episode: int) -> int:
    s = episode_seed(master_seed, trace_key, episode)
    random.seed(s)
    np.random.seed(s % (2**32 - 1))
    return s
