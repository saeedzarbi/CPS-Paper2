"""Determinism tests for Tube-v2 evaluation."""
from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.control.two_timescale_conformal import make_controlled_policy
from src.evaluation.clean_abr_eval import HEADLINE_TRACE_SETS, build_env, run_episode
from src.evaluation.eval_certified_shield_v18 import make_policy
from src.evaluation.eval_seeds import DEFAULT_MASTER_SEED

ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "results" / "tube_v2" / "phase3"


def _run_rows(arm: str, trace_key: str, episodes: int = 3) -> list[dict]:
    trace_dir = HEADLINE_TRACE_SETS.get(trace_key) or ROOT / "data" / "standardized" / "test_traces_v18"
    if not Path(trace_dir).is_dir():
        pytest.skip("traces missing")
    env = build_env(trace_dir, arm)  # type: ignore[arg-type]
    base = make_policy("bba", env, None, False)
    policy = make_controlled_policy(env, arm, base, None, trace_set_key=trace_key)  # type: ignore[arg-type]
    rows = []
    for ep in range(episodes):
        if hasattr(policy, "on_episode_start"):
            policy.on_episode_start()
        rows.append(
            run_episode(
                env,
                policy,
                ep,
                master_seed=DEFAULT_MASTER_SEED,
                trace_key=trace_key,
            )
        )
    return rows


@pytest.mark.parametrize("arm", ["raw", "static_conformal", "robust_mpc_logu"])
def test_episode_rows_byte_identical_across_reruns(arm: str) -> None:
    a = _run_rows(arm, "bb")
    b = _run_rows(arm, "bb")
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        for k in ra:
            if isinstance(ra[k], (int, float)):
                assert abs(float(ra[k]) - float(rb[k])) < 1e-9, k
            else:
                assert ra[k] == rb[k]


def _campaign_args() -> list[str]:
    return [
        sys.executable,
        str(ROOT / "src" / "evaluation" / "tube_v2_campaign.py"),
        "--phases",
        "3",
        "--trace-sets",
        "bb",
        "--episodes",
        "2",
        "--arms",
        "robust_mpc_logu,tube_v2_bonf_aci",
        "--refit-quantile",
        "--no-resume",
        "--workers",
        "1",
    ]


def test_smoke_campaign_csvs_identical_in_fresh_processes(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    shutil.rmtree(PHASE3, ignore_errors=True)
    subprocess.check_call(_campaign_args(), cwd=ROOT, env=env)
    snap = tmp_path / "run_a"
    if not PHASE3.is_dir():
        pytest.skip("phase3 results missing (traces?)")
    shutil.copytree(PHASE3, snap)
    shutil.rmtree(PHASE3, ignore_errors=True)
    subprocess.check_call(_campaign_args(), cwd=ROOT, env=env)
    mismatches = []
    for csv_path in sorted(snap.rglob("*.csv")):
        if csv_path.name == "episodes.csv":
            continue
        rel = csv_path.relative_to(snap)
        other = PHASE3 / rel
        if not other.is_file():
            mismatches.append(f"missing {rel}")
        elif not filecmp.cmp(csv_path, other, shallow=False):
            mismatches.append(str(rel))
    assert not mismatches, "CSV diffs: " + ", ".join(mismatches)
