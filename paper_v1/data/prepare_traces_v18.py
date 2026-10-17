"""
Prepare clean, provenance-correct trace datasets for the V18 pipeline.

Fixes two issues found in the existing standardized split:

  (A) SOURCE LEAKAGE. The old train_traces / test_traces share the SAME source
      sessions (e.g. FCC trace-id 10652, Norway route bus.ljansbakken-oslo appear
      in BOTH). Windows from one session are highly correlated -> optimistic test
      numbers. We rebuild a SOURCE-DISJOINT split: every source group (FCC id or
      Norway route) is assigned ENTIRELY to train or test, stratified by dataset.

  (B) REGIME MISMATCH for the certified shield. The shield's headline is on 5G
      links (saturated top rungs feasible -> banking has headroom), but policies
      were trained on FCC/Norway broadband. We generate matched 5G TRAIN and TEST
      trace sets with DISJOINT seeds so shield-aware co-design (proposed_cps) is
      trained and evaluated in the same regime, without train/test leakage.

Outputs (under data/standardized/):
  train_traces_v18/      test_traces_v18/         (source-disjoint broadband)
  train_traces_5g_v18/   test_traces_5g_v18/      (synthetic 5G, disjoint seeds)

Idempotent: safe to re-run. Originals (train_traces/, test_traces/) are untouched.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from configs.paths import get_paths

P = get_paths()
STD = P["data_dir"] / "standardized"

TEST_FRACTION = 0.20
SPLIT_SEED = 1234


# --------------------------------------------------------------------------- #
# (A) source-disjoint broadband split
# --------------------------------------------------------------------------- #
def _group_key(name: str) -> str:
    """Map a trace filename to its SOURCE group so all correlated windows of one
    session land on the same side of the split."""
    if name.startswith("fcc"):
        m = re.search(r"trace_(\d+)", name)
        return f"fcc:{m.group(1)}" if m else f"fcc:{name}"
    if name.startswith("norway"):
        # norway_<route>-report.<date>_<time>.json  ->  group by <route>
        stem = name[len("norway_"):]
        route = stem.split("-report")[0]
        return f"norway:{route}"
    return f"other:{name.split('_')[0]}"


def _dataset_of(group: str) -> str:
    return group.split(":", 1)[0]


def build_disjoint_split():
    src_dirs = [P["train_traces"], P["test_traces"]]
    # union of all valid trace files, keyed by group; keep every distinct file
    groups: dict[str, list[Path]] = defaultdict(list)
    seen_content: dict[str, set[int]] = defaultdict(set)
    for d in src_dirs:
        for f in sorted(d.glob("*.json")):
            if f.name == "trace_stats.json":
                continue
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
                if "throughput_kbps" not in j:
                    continue
                h = hash(tuple(j["throughput_kbps"]))
            except Exception:
                continue
            g = _group_key(f.name)
            if h in seen_content[g]:
                continue  # exact-duplicate window already captured
            seen_content[g].add(h)
            groups[g].append(f)

    # stratified, deterministic group-level split
    rng = np.random.default_rng(SPLIT_SEED)
    train_dir = STD / "train_traces_v18"
    test_dir = STD / "test_traces_v18"
    for d in (train_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    by_ds: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        by_ds[_dataset_of(g)].append(g)

    assign: dict[str, str] = {}
    for ds, glist in by_ds.items():
        glist = sorted(glist)
        rng.shuffle(glist)
        n_test = max(1, int(round(len(glist) * TEST_FRACTION)))
        for i, g in enumerate(glist):
            assign[g] = "test" if i < n_test else "train"

    counts = {"train": 0, "test": 0}
    used_names = {"train": set(), "test": set()}
    for g, files in groups.items():
        side = assign[g]
        dst_dir = train_dir if side == "train" else test_dir
        for f in files:
            name = f.name
            k = 1
            while name in used_names[side]:
                name = f"{f.stem}__d{k}{f.suffix}"
                k += 1
            used_names[side].add(name)
            shutil.copy2(f, dst_dir / name)
            counts[side] += 1

    # verify zero group overlap
    def side_groups(d):
        return {_group_key(f.name.split("__d")[0] + ".json") for f in d.glob("*.json")}
    overlap = side_groups(train_dir) & side_groups(test_dir)
    print(f"[A] source-disjoint broadband split:")
    print(f"    train files={counts['train']}  test files={counts['test']}")
    print(f"    train groups={len(side_groups(train_dir))}  "
          f"test groups={len(side_groups(test_dir))}  GROUP OVERLAP={len(overlap)}")
    assert not overlap, f"LEAKAGE: groups in both splits: {sorted(overlap)[:5]}"
    return train_dir, test_dir


# --------------------------------------------------------------------------- #
# (B) matched 5G train / test sets with disjoint seeds
# --------------------------------------------------------------------------- #
def _gen_5g(out_dir: Path, n: int, seed: int, dip_lo: float, dip_hi: float,
            offset_mod: int):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        tp = np.full(1000, 9000.0)
        for start in range(20 + (i % offset_mod), 1000, 40):
            tp[start:start + 4] = float(rng.uniform(dip_lo, dip_hi))
        tp *= rng.uniform(0.85, 1.15, size=tp.shape)
        tp = np.clip(tp, 80.0, 50000.0)
        (out_dir / f"5g_v18_{i:03d}.json").write_text(
            json.dumps({"throughput_kbps": tp.tolist()}), encoding="utf-8")
    return out_dir


def build_5g_sets():
    # TEST set: identical to the runbook's original generation (seed 42) so prior
    # certified-shield eval numbers remain reproducible.
    test5g = _gen_5g(STD / "test_traces_5g_v18", n=50, seed=42,
                     dip_lo=500.0, dip_hi=1200.0, offset_mod=17)
    # TRAIN set: DISJOINT seed + different dip distribution / phase so the policy
    # never sees the evaluation traces.
    train5g = _gen_5g(STD / "train_traces_5g_v18", n=80, seed=7000,
                      dip_lo=400.0, dip_hi=1300.0, offset_mod=23)
    print(f"[B] 5G sets: train={len(list(train5g.glob('*.json')))} (seed 7000)  "
          f"test={len(list(test5g.glob('*.json')))} (seed 42, unchanged)")
    return train5g, test5g


if __name__ == "__main__":
    print(f"Preparing V18 trace datasets under {STD}")
    build_disjoint_split()
    build_5g_sets()
    print("Done. Update training/eval to use *_v18 directories.")
