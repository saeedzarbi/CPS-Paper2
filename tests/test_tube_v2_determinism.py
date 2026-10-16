"""Determinism tests for Tube-v2 evaluation."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.control.two_timescale_conformal import make_controlled_policy
from src.evaluation.clean_abr_eval import HEADLINE_TRACE_SETS, build_env, run_episode
from src.evaluation.eval_certified_shield_v18 import make_policy
from src.evaluation.eval_seeds import DEFAULT_MASTER_SEED

ROOT = Path(__file__).resolve().parents[1]


def _run_rows(arm: str, trace_key: str, episodes: int = 3) -> list[dict]:
    trace_dir = HEADLINE_TRACE_SETS.get(trace_key) or ROOT / "data" / "standardized" / "test_traces_v18"
    if not Path(trace_dir).is_dir():
        pytest.skip("traces missing")
    env = build_env(trace_dir, arm)  # type: ignore[arg-type]
    base = make_policy("bba", env, None, False)
    rows = []
    for ep in range(episodes):
        policy = make_controlled_policy(env, arm, base, None)  # type: ignore[arg-type]
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
