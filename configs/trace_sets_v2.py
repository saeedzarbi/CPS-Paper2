"""Tube-v2 trace sets with train / val / test paths (additive; does not replace legacy dicts)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from configs.paths import (
    STANDARDIZED_DIR,
    TEST_TRACES_5G_IRISH_MMSYS20_DIR,
    TEST_TRACES_5G_MARKOV_DIR,
    TRAIN_TRACES_5G_IRISH_MMSYS20_DIR,
    TRAIN_TRACES_5G_MARKOV_DIR,
)

SPLIT_ROOT = Path(__file__).resolve().parents[1] / "results" / "tube_v2" / "splits"
VAL_SEED = 2026
VAL_FRACTION = 0.20


def _materialize_val(train_dir: Path, name: str, seed: int = VAL_SEED, fraction: float = VAL_FRACTION) -> Path:
    val_dir = SPLIT_ROOT / f"{name}_val"
    if val_dir.is_dir() and any(val_dir.glob("*.json")):
        return val_dir
    files = sorted(train_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No traces in {train_dir}")
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(files) * fraction)))
    pick = rng.choice(len(files), size=n_val, replace=False)
    val_dir.mkdir(parents=True, exist_ok=True)
    for i in pick:
        dst = val_dir / files[i].name
        if not dst.exists():
            shutil.copy2(files[i], dst)
    return val_dir


TRACE_SETS_V2: dict[str, dict[str, Path]] = {
    "5g_markov": {
        "train": TRAIN_TRACES_5G_MARKOV_DIR,
        "test": TEST_TRACES_5G_MARKOV_DIR,
    },
    "5g_irish": {
        "train": TRAIN_TRACES_5G_IRISH_MMSYS20_DIR,
        "test": TEST_TRACES_5G_IRISH_MMSYS20_DIR,
    },
}


def val_trace_dir(trace_set: str) -> Path:
    if trace_set not in TRACE_SETS_V2:
        raise KeyError(trace_set)
    train = TRACE_SETS_V2[trace_set]["train"]
    if not train.is_dir():
        raise FileNotFoundError(f"Missing train dir for {trace_set}: {train}")
    return _materialize_val(train, trace_set)


def resolve_trace_dir(trace_set: str, split: str) -> Path:
    if split == "val":
        return val_trace_dir(trace_set)
    if trace_set not in TRACE_SETS_V2 or split not in TRACE_SETS_V2[trace_set]:
        raise KeyError((trace_set, split))
    return TRACE_SETS_V2[trace_set][split]
