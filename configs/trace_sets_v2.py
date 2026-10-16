"""Tube-v2 trace sets with train / val / test paths (additive; does not replace legacy dicts)."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from configs.paths import (
    STANDARDIZED_DIR,
    TEST_TRACES_5G_IRISH_MMSYS20_DIR,
    TEST_TRACES_5G_MARKOV_DIR,
    TEST_TRACES_V18_DIR,
    TRAIN_TRACES_5G_IRISH_MMSYS20_DIR,
    TRAIN_TRACES_5G_MARKOV_DIR,
    TRAIN_TRACES_V18_DIR,
)

SPLIT_ROOT = Path(__file__).resolve().parents[1] / "results" / "tube_v2" / "splits"
VAL_SEED = 2026
VAL_FRACTION = 0.20
HARSH_FAMILY_SEED = 2027
HARSH_CALIB_FRACTION = 0.40
HARSH_SOURCE_DIR = STANDARDIZED_DIR / "test_traces_5g_harsh_v15"


@dataclass(frozen=True)
class CalibBundle:
    dirs: tuple[Path, ...]
    label: str
    cache_suffix: str


def _manifest_path(name: str) -> Path:
    return SPLIT_ROOT / f"{name}_manifest.json"


def _load_manifest(name: str) -> dict | None:
    path = _manifest_path(name)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(name: str, payload: dict) -> None:
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    _manifest_path(name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _materialize_subset(src_dir: Path, dst_dir: Path, names: set[str]) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        src = src_dir / name
        dst = dst_dir / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    return dst_dir


def _materialize_val(train_dir: Path, name: str, seed: int = VAL_SEED, fraction: float = VAL_FRACTION) -> Path:
    manifest_key = f"{name}_val"
    existing = _load_manifest(manifest_key)
    val_dir = SPLIT_ROOT / f"{name}_val"
    if existing and val_dir.is_dir() and any(val_dir.glob("*.json")):
        return val_dir
    files = sorted(train_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No traces in {train_dir}")
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(files) * fraction)))
    pick = rng.choice(len(files), size=n_val, replace=False)
    val_names = {files[int(i)].name for i in pick}
    _materialize_subset(train_dir, val_dir, val_names)
    _save_manifest(
        manifest_key,
        {"seed": seed, "fraction": fraction, "files": sorted(val_names), "source": str(train_dir)},
    )
    return val_dir


def val_trace_dir(trace_set: str) -> Path:
    if trace_set not in TRACE_SETS_V2:
        raise KeyError(trace_set)
    train = TRACE_SETS_V2[trace_set]["train"]
    if not train.is_dir():
        raise FileNotFoundError(f"Missing train dir for {trace_set}: {train}")
    return _materialize_val(train, trace_set)


def val_file_names(trace_set: str) -> set[str]:
    return {p.name for p in val_trace_dir(trace_set).glob("*.json")}


def train_minus_val_dir(trace_set: str) -> Path:
    if trace_set not in TRACE_SETS_V2:
        raise KeyError(trace_set)
    train = TRACE_SETS_V2[trace_set]["train"]
    val_names = val_file_names(trace_set)
    out = SPLIT_ROOT / f"{trace_set}_train_minus_val"
    train_names = {p.name for p in train.glob("*.json")}
    keep = train_names - val_names
    assert not (keep & val_names), f"train/val overlap for {trace_set}"
    _materialize_subset(train, out, keep)
    return out


def assert_disjoint_dirs(a: Path, b: Path) -> None:
    a_names = {p.name for p in a.glob("*.json")}
    b_names = {p.name for p in b.glob("*.json")}
    overlap = a_names & b_names
    assert not overlap, f"disjoint split violated: {overlap}"


TRACE_SETS_V2: dict[str, dict[str, Path]] = {
    "5g_markov": {
        "train": TRAIN_TRACES_5G_MARKOV_DIR,
        "test": TEST_TRACES_5G_MARKOV_DIR,
    },
    "5g_irish": {
        "train": TRAIN_TRACES_5G_IRISH_MMSYS20_DIR,
        "test": TEST_TRACES_5G_IRISH_MMSYS20_DIR,
    },
    "bb": {
        "train": TRAIN_TRACES_V18_DIR,
        "test": TEST_TRACES_V18_DIR,
    },
}


def harsh_family_calib_test_dirs(
    *,
    seed: int = HARSH_FAMILY_SEED,
    fraction: float = HARSH_CALIB_FRACTION,
    force: bool = False,
) -> tuple[Path, Path]:
    """Disjoint calib / eval split of the harsh trace family (never eval on calib files)."""
    manifest_key = "5g_harsh_family"
    calib_dir = SPLIT_ROOT / "5g_harsh_calib"
    test_dir = SPLIT_ROOT / "5g_harsh_eval_test"
    if (
        not force
        and _load_manifest(manifest_key)
        and calib_dir.is_dir()
        and test_dir.is_dir()
        and any(calib_dir.glob("*.json"))
        and any(test_dir.glob("*.json"))
    ):
        assert_disjoint_dirs(calib_dir, test_dir)
        return calib_dir, test_dir
    if not HARSH_SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Missing harsh trace dir: {HARSH_SOURCE_DIR}")
    files = sorted(HARSH_SOURCE_DIR.glob("*.json"))
    files = [f for f in files if not f.name.startswith("trace_stats")]
    if not files:
        raise FileNotFoundError(f"No JSON traces in {HARSH_SOURCE_DIR}")
    rng = np.random.default_rng(seed)
    n_calib = max(1, int(round(len(files) * fraction)))
    pick = rng.choice(len(files), size=n_calib, replace=False)
    calib_names = {files[int(i)].name for i in pick}
    test_names = {f.name for f in files} - calib_names
    if calib_dir.exists():
        shutil.rmtree(calib_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    _materialize_subset(HARSH_SOURCE_DIR, calib_dir, calib_names)
    _materialize_subset(HARSH_SOURCE_DIR, test_dir, test_names)
    assert_disjoint_dirs(calib_dir, test_dir)
    _save_manifest(
        manifest_key,
        {
            "seed": seed,
            "fraction": fraction,
            "calib": sorted(calib_names),
            "test": sorted(test_names),
            "source": str(HARSH_SOURCE_DIR),
        },
    )
    return calib_dir, test_dir


def resolve_trace_dir(trace_set: str, split: str) -> Path:
    if trace_set == "5g_harsh" and split in ("test", "val"):
        return harsh_family_calib_test_dirs()[1]
    if split == "val":
        return val_trace_dir(trace_set)
    if trace_set not in TRACE_SETS_V2 or split not in TRACE_SETS_V2[trace_set]:
        raise KeyError((trace_set, split))
    return TRACE_SETS_V2[trace_set][split]


def markov_train_available() -> bool:
    return TRAIN_TRACES_5G_MARKOV_DIR.is_dir() and any(TRAIN_TRACES_5G_MARKOV_DIR.glob("*.json"))


def calibration_bundles_for_eval(trace_set: str) -> list[CalibBundle]:
    """Calibration directories for fitting (never the eval test split)."""
    from configs.paths import TRAIN_TRACES_5G_V18_DIR, TRAIN_TRACES_V18_DIR

    bundles: list[CalibBundle] = []
    if trace_set == "5g_harsh":
        calib, _test = harsh_family_calib_test_dirs()
        bundles.append(
            CalibBundle(dirs=(calib,), label="calib=5g_harsh (held-out split)", cache_suffix="harsh_native")
        )
        if markov_train_available():
            key = "5g_markov"
            tmv = train_minus_val_dir(key)
            vd = val_trace_dir(key)
            assert_disjoint_dirs(tmv, vd)
            bundles.append(
                CalibBundle(
                    dirs=(tmv, vd),
                    label="calib=5g_markov (transfer)",
                    cache_suffix="markov_transfer",
                )
            )
        return bundles
    if trace_set == "bb":
        if TRAIN_TRACES_V18_DIR.is_dir():
            bundles.append(
                CalibBundle(
                    dirs=(TRAIN_TRACES_V18_DIR,),
                    label="calib=train_traces_v18",
                    cache_suffix="calib",
                )
            )
        return bundles
    key = trace_set if trace_set in TRACE_SETS_V2 else None
    if key is None:
        raise KeyError(f"No calibration mapping for trace_set={trace_set}")
    train = TRACE_SETS_V2[key]["train"]
    if not train.is_dir():
        raise FileNotFoundError(f"Missing train dir for {key}: {train}")
    tmv = train_minus_val_dir(key)
    vd = val_trace_dir(key)
    assert_disjoint_dirs(tmv, vd)
    bundles.append(
        CalibBundle(dirs=(tmv, vd), label=f"calib={key} train_minus_val+val", cache_suffix="calib")
    )
    return bundles


def calibration_dirs_for_eval(trace_set: str, *, variant: str = "default") -> list[Path]:
    """Backward-compatible: return dirs for the chosen calibration variant."""
    bundles = calibration_bundles_for_eval(trace_set)
    if trace_set == "5g_harsh":
        if variant in ("transfer", "markov_transfer"):
            for b in bundles:
                if b.cache_suffix == "markov_transfer":
                    return list(b.dirs)
            raise FileNotFoundError("5g_markov train not available for transfer calibration")
        if variant in ("native", "harsh_native", "default"):
            for b in bundles:
                if b.cache_suffix == "harsh_native":
                    return list(b.dirs)
    if not bundles:
        raise KeyError(trace_set)
    return list(bundles[0].dirs)


def ensure_smoke_calibration_data(trace_sets: list[str]) -> None:
    """Fail fast with a clear message when calibration data are missing."""
    missing: list[str] = []
    need_markov = any(ts in ("5g_harsh", "5g_markov", "5g_v18") for ts in trace_sets)
    if need_markov and not markov_train_available():
        missing.append(f"{TRAIN_TRACES_5G_MARKOV_DIR} (run scripts/paper2/gen_markov_traces.py)")
    if "bb" in trace_sets:
        from configs.paths import TRAIN_TRACES_V18_DIR

        if not TRAIN_TRACES_V18_DIR.is_dir():
            missing.append(str(TRAIN_TRACES_V18_DIR))
    if "5g_harsh" in trace_sets and not HARSH_SOURCE_DIR.is_dir():
        missing.append(str(HARSH_SOURCE_DIR))
    if missing:
        raise SystemExit(
            "Smoke calibration data missing:\n  - "
            + "\n  - ".join(missing)
            + "\nInstall traces or run scripts/paper2/gen_markov_traces.py (fixed seed)."
        )
