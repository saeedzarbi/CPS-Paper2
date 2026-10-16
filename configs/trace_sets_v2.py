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
HARSH_VAL_FRACTION = 0.20
HARSH_TEST_FRACTION = 0.40
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


def eval_test_dir(trace_set: str) -> Path:
    """Held-out test traces with basenames disjoint from the train pool (filename leakage guard)."""
    if trace_set not in TRACE_SETS_V2:
        raise KeyError(trace_set)
    train = TRACE_SETS_V2[trace_set]["train"]
    test_src = TRACE_SETS_V2[trace_set]["test"]
    train_names = {p.name for p in train.glob("*.json")}
    test_names = {
        p.name
        for p in test_src.glob("*.json")
        if not p.name.startswith("trace_stats")
    }
    exclusive = test_names - train_names
    if not exclusive:
        raise FileNotFoundError(f"No test traces disjoint from train for {trace_set}")
    if exclusive == test_names:
        return test_src
    out = SPLIT_ROOT / f"{trace_set}_test_disjoint"
    if out.is_dir() and {p.name for p in out.glob("*.json")} == exclusive:
        return out
    if out.exists():
        shutil.rmtree(out)
    _materialize_subset(test_src, out, exclusive)
    return out


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


def harsh_family_three_way_dirs(
    *,
    seed: int = HARSH_FAMILY_SEED,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Disjoint calib / val / test (40/20/40) from the harsh trace family."""
    manifest_key = "5g_harsh_family_3way"
    calib_dir = SPLIT_ROOT / "5g_harsh_calib"
    val_dir = SPLIT_ROOT / "5g_harsh_val"
    test_dir = SPLIT_ROOT / "5g_harsh_test"
    if (
        not force
        and _load_manifest(manifest_key)
        and calib_dir.is_dir()
        and val_dir.is_dir()
        and test_dir.is_dir()
        and any(calib_dir.glob("*.json"))
        and any(val_dir.glob("*.json"))
        and any(test_dir.glob("*.json"))
    ):
        assert_disjoint_dirs(calib_dir, val_dir)
        assert_disjoint_dirs(calib_dir, test_dir)
        assert_disjoint_dirs(val_dir, test_dir)
        return calib_dir, val_dir, test_dir
    if not HARSH_SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Missing harsh trace dir: {HARSH_SOURCE_DIR}")
    files = sorted(HARSH_SOURCE_DIR.glob("*.json"))
    files = [f for f in files if not f.name.startswith("trace_stats")]
    if not files:
        raise FileNotFoundError(f"No JSON traces in {HARSH_SOURCE_DIR}")
    n = len(files)
    n_calib = max(1, int(round(n * HARSH_CALIB_FRACTION)))
    n_val = max(1, int(round(n * HARSH_VAL_FRACTION)))
    n_test = max(1, n - n_calib - n_val)
    if n_calib + n_val + n_test > n:
        n_test = max(1, n - n_calib - n_val)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    calib_idx = perm[:n_calib]
    val_idx = perm[n_calib : n_calib + n_val]
    test_idx = perm[n_calib + n_val : n_calib + n_val + n_test]
    calib_names = {files[int(i)].name for i in calib_idx}
    val_names = {files[int(i)].name for i in val_idx}
    test_names = {files[int(i)].name for i in test_idx}
    for d in (calib_dir, val_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)
    _materialize_subset(HARSH_SOURCE_DIR, calib_dir, calib_names)
    _materialize_subset(HARSH_SOURCE_DIR, val_dir, val_names)
    _materialize_subset(HARSH_SOURCE_DIR, test_dir, test_names)
    assert_disjoint_dirs(calib_dir, val_dir)
    assert_disjoint_dirs(calib_dir, test_dir)
    assert_disjoint_dirs(val_dir, test_dir)
    _save_manifest(
        manifest_key,
        {
            "seed": seed,
            "fractions": {
                "calib": HARSH_CALIB_FRACTION,
                "val": HARSH_VAL_FRACTION,
                "test": HARSH_TEST_FRACTION,
            },
            "calib": sorted(calib_names),
            "val": sorted(val_names),
            "test": sorted(test_names),
            "source_logical": "test_traces_5g_harsh_v15",
        },
    )
    return calib_dir, val_dir, test_dir


def harsh_family_calib_test_dirs(**kwargs) -> tuple[Path, Path]:
    """Backward-compatible: (calib, test) — prefer ``harsh_family_three_way_dirs``."""
    calib, _val, test = harsh_family_three_way_dirs(**kwargs)
    return calib, test


def split_dirs_for_trace_set(trace_set: str) -> tuple[Path, Path, Path]:
    """(calib, val, test) directory paths used for leakage checks."""
    if trace_set == "5g_harsh":
        return harsh_family_three_way_dirs()
    if trace_set not in TRACE_SETS_V2:
        raise KeyError(trace_set)
    calib = train_minus_val_dir(trace_set)
    val = val_trace_dir(trace_set)
    test = eval_test_dir(trace_set)
    return calib, val, test


def resolve_trace_dir(trace_set: str, split: str) -> Path:
    if trace_set == "5g_harsh":
        calib, val, test = harsh_family_three_way_dirs()
        if split == "calib":
            return calib
        if split == "val":
            return val
        if split == "test":
            return test
        raise KeyError((trace_set, split))
    if split == "val":
        return val_trace_dir(trace_set)
    if split == "test":
        return eval_test_dir(trace_set)
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
        calib, _val, _test = harsh_family_three_way_dirs()
        bundles.append(
            CalibBundle(dirs=(calib,), label="calib=5g_harsh (40% calib split)", cache_suffix="harsh_native")
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
    if variant in ("transfer", "markov_transfer"):
        if trace_set != "5g_harsh":
            raise ValueError("markov_transfer calibration is only defined for 5g_harsh")
        for b in bundles:
            if b.cache_suffix == "markov_transfer":
                return list(b.dirs)
        raise FileNotFoundError("5g_markov train not available for transfer calibration")
    if variant in ("native", "harsh_native"):
        if trace_set == "5g_harsh":
            for b in bundles:
                if b.cache_suffix == "harsh_native":
                    return list(b.dirs)
        if not bundles:
            raise KeyError(trace_set)
        return list(bundles[0].dirs)
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
