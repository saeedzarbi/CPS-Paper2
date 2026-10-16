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


def episode_rng(master_seed: int, trace_key: str, episode: int) -> np.random.Generator:
    return np.random.default_rng(episode_seed(master_seed, trace_key, episode))


def apply_episode_seeds(master_seed: int, trace_key: str, episode: int) -> tuple[int, np.random.Generator]:
    s = episode_seed(master_seed, trace_key, episode)
    random.seed(s)
    gen = np.random.default_rng(s)
    return s, gen
