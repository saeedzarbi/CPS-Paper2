#!/usr/bin/env python3
"""Generate minimal 5G Markov train/test trace JSONs (fixed seed) for Tube-v2 smoke."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.paths import STANDARDIZED_DIR, TEST_TRACES_5G_MARKOV_DIR, TRAIN_TRACES_5G_MARKOV_DIR

SEED = 4242
N_TRAIN = 24
N_TEST = 12
TRACE_LEN = 1200


def _markov_trace(rng: np.random.Generator, length: int) -> list[float]:
    states = [200.0, 800.0, 2500.0, 6000.0]
    trans = np.array(
        [
            [0.85, 0.12, 0.02, 0.01],
            [0.10, 0.75, 0.12, 0.03],
            [0.05, 0.15, 0.65, 0.15],
            [0.08, 0.12, 0.20, 0.60],
        ]
    )
    s = 1
    out: list[float] = []
    for _ in range(length):
        s = int(rng.choice(len(states), p=trans[s]))
        base = states[s]
        out.append(float(base * rng.lognormal(0.0, 0.12)))
    return out


def _write_dir(path: Path, n: int, rng: np.random.Generator, prefix: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        payload = {
            "throughput_kbps": _markov_trace(rng, TRACE_LEN),
            "trace_id": f"{prefix}_{i:03d}",
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        (path / f"{prefix}_{i:03d}.json").write_text(text, encoding="utf-8")


def main() -> int:
    rng = np.random.default_rng(SEED)
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    _write_dir(TRAIN_TRACES_5G_MARKOV_DIR, N_TRAIN, rng, "markov_train")
    _write_dir(TEST_TRACES_5G_MARKOV_DIR, N_TEST, rng, "markov_test")
    print(f"Wrote {N_TRAIN} train + {N_TEST} test traces under {STANDARDIZED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
