"""Shared clean-slate ABR evaluation helpers (clean_v3 / extended validation)."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from configs.paths import STANDARDIZED_DIR, get_paths
from configs.videos import CPS_EPISODES, EVAL_VIDEOS, MAX_CHUNKS
from src.control.two_timescale_conformal import ControlArmConfig, EvalArm, configure_env_for_arm, make_controlled_policy
from src.environment.clean_abr_env import CleanABREnv
from src.evaluation.eval_certified_shield_v18 import _bootstrap_ci, _wilcoxon, make_policy

P = get_paths()
QOE_REBUF_W = 4.3
QOE_SMOOTH_W = 1.0

HEADLINE_TRACE_SETS: dict[str, Path] = {
    "5g_markov": STANDARDIZED_DIR / "test_traces_5g_markov",
    "bb": STANDARDIZED_DIR / "test_traces_v18",
}

EXTENDED_TRACE_SETS: dict[str, Path] = {
    "5g_stress_v14": STANDARDIZED_DIR / "test_traces_5g_stress_v14",
    "5g_harsh": STANDARDIZED_DIR / "test_traces_5g_harsh_v15",
    "norway_vehicular": STANDARDIZED_DIR / "eval_norway_vehicular",
    "fcc_fixed": STANDARDIZED_DIR / "eval_fcc_fixed",
}


_EP_VIDEO_CACHE: dict[int, str] = {}


def resolve_video_for_episode(ep: int) -> str:
    if ep in _EP_VIDEO_CACHE:
        return _EP_VIDEO_CACHE[ep]
    env = CleanABREnv(
        video_names=list(EVAL_VIDEOS),
        trace_dir=str(HEADLINE_TRACE_SETS["5g_markov"]),
        vmaf_dir=str(P["vmaf_scores"]),
        siti_dir=str(P["content_features"]),
        max_chunks=MAX_CHUNKS,
        random_seed=0,
        enable_early_abort=False,
    )
    random.seed(1000 + ep)
    env.reset(seed=1000 + ep)
    name = str(env.current_video_name)
    _EP_VIDEO_CACHE[ep] = name
    return name


def build_env(trace_dir: Path, arm: EvalArm) -> CleanABREnv:
    env = CleanABREnv(
        video_names=list(EVAL_VIDEOS),
        trace_dir=str(trace_dir),
        vmaf_dir=str(P["vmaf_scores"]),
        siti_dir=str(P["content_features"]),
        max_chunks=MAX_CHUNKS,
        random_seed=0,
        use_lyapunov=True,
        use_future=True,
        reward_mode="log_utility",
        enable_early_abort=False,
    )
    configure_env_for_arm(env, arm)
    return env


def run_episode(
    env: CleanABREnv,
    policy,
    ep: int,
    *,
    time_decisions: bool = False,
) -> dict:
    random.seed(1000 + ep)
    obs, _ = env.reset(seed=1000 + ep)
    if hasattr(policy, "on_episode_start"):
        policy.on_episode_start()
    chunk_rebufs: list[float] = []
    log_utils: list[float] = []
    smooths: list[float] = []
    decision_us: list[float] = []

    done = False
    while not done:
        t0 = time.perf_counter_ns()
        a, _ = policy.predict(obs, deterministic=True)
        if time_decisions:
            decision_us.append((time.perf_counter_ns() - t0) / 1000.0)
        obs, _r, term, trunc, info = env.step(a)
        done = term or trunc
        chunk_rebufs.append(float(info.get("rebuffer", 0.0)))
        log_utils.append(float(info.get("log_utility", 0.0)))
        smooths.append(float(info.get("smooth_penalty", 0.0)))

    reb_total = float(sum(chunk_rebufs))
    reb_p95 = float(np.percentile(chunk_rebufs, 95)) if chunk_rebufs else 0.0
    stallfree = float(np.mean([1.0 if r <= 1e-6 else 0.0 for r in chunk_rebufs]))
    sum_u = float(sum(log_utils))
    sum_smooth = float(sum(smooths))
    episodic_qoe = sum_u - QOE_REBUF_W * reb_total - QOE_SMOOTH_W * sum_smooth
    zero_stall = float(reb_total <= 1e-6)

    row = {
        "episode": ep,
        "video_name": str(env.current_video_name),
        "rebuffer_total": reb_total,
        "rebuffer_p95_chunk": reb_p95,
        "stallfree_rate": stallfree,
        "zero_stall_episode": zero_stall,
        "log_utility_mean": float(np.mean(log_utils)) if log_utils else 0.0,
        "smooth_penalty_mean": float(np.mean(smooths)) if smooths else 0.0,
        "episodic_qoe": episodic_qoe,
    }
    if time_decisions and decision_us:
        row["decision_latency_us_mean"] = float(np.mean(decision_us))
    return row


def _run_arm_episode_batch(
    policy_kind: str,
    trace_dir: str,
    arm: str,
    episode_ids: list[int],
    cfg_dict: dict | None,
    time_decisions: bool,
) -> list[dict]:
    cfg = ControlArmConfig(**cfg_dict) if cfg_dict else None
    env = build_env(Path(trace_dir), arm)  # type: ignore[arg-type]
    base = make_policy(policy_kind, env, None, False)
    controlled = make_controlled_policy(env, arm, base, cfg)  # type: ignore[arg-type]
    return [
        run_episode(env, controlled, ep, time_decisions=time_decisions) for ep in episode_ids
    ]


def default_eval_workers() -> int:
    """Episode parallelism; keep headroom for other services on shared hosts."""
    raw = os.environ.get("CLEAN_V3_EXT_WORKERS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    cpus = os.cpu_count() or 4
    reserved = int(os.environ.get("CLEAN_V3_EXT_RESERVED_CPUS", "2"))
    cap = int(os.environ.get("CLEAN_V3_EXT_MAX_WORKERS", "2"))
    return max(1, min(cap, cpus - max(0, reserved)))


def run_arm(
    policy_kind: str,
    trace_dir: Path,
    arm: EvalArm,
    episodes: int,
    cfg: ControlArmConfig | None = None,
    *,
    time_decisions: bool = False,
    workers: int | None = None,
) -> list[dict]:
    n_workers = workers if workers is not None else default_eval_workers()
    if n_workers <= 1 or episodes <= 1:
        env = build_env(trace_dir, arm)
        base = make_policy(policy_kind, env, None, False)
        controlled = make_controlled_policy(env, arm, base, cfg)
        return [
            run_episode(env, controlled, ep, time_decisions=time_decisions)
            for ep in range(episodes)
        ]

    ep_ids = list(range(episodes))
    chunk = max(1, (episodes + n_workers - 1) // n_workers)
    chunks = [ep_ids[i : i + chunk] for i in range(0, episodes, chunk)]
    n_workers = min(n_workers, len(chunks))
    cfg_dict = asdict(cfg) if cfg else None
    task_args = [
        (policy_kind, str(trace_dir), arm, ids, cfg_dict, time_decisions) for ids in chunks
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_arm_episode_batch, *args) for args in task_args]
        for fut in as_completed(futures):
            rows.extend(fut.result())
    rows.sort(key=lambda r: int(r["episode"]))
    return rows


def pair_comp(all_rows: dict[str, list[dict]], treat: str, ref: str) -> dict:
    out = {}
    keys = (
        "rebuffer_total",
        "rebuffer_p95_chunk",
        "smooth_penalty_mean",
        "log_utility_mean",
        "episodic_qoe",
    )
    for k in keys:
        a = np.asarray([r[k] for r in all_rows[treat]], float)
        b = np.asarray([r[k] for r in all_rows[ref]], float)
        _, p = _wilcoxon(a, b)
        out[k] = {"diff": float(a.mean() - b.mean()), "wilcoxon_p": p}
    return out


def write_job_out(
    out_dir: Path,
    all_rows: dict[str, list[dict]],
    policy: str,
    trace_key: str,
    meta: dict | None = None,
    *,
    protocol: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict = {"policy": policy, "trace": trace_key, "arms": {}, "comparisons": {}}
    summaries["meta"] = dict(meta or {})
    if protocol:
        summaries["meta"]["protocol"] = protocol
    metric_cols = [
        "rebuffer_total",
        "rebuffer_p95_chunk",
        "stallfree_rate",
        "zero_stall_episode",
        "log_utility_mean",
        "smooth_penalty_mean",
        "episodic_qoe",
    ]
    for arm, rows in all_rows.items():
        summaries["arms"][arm] = {}
        for k in metric_cols:
            m, lo, hi = _bootstrap_ci([r[k] for r in rows])
            summaries["arms"][arm][k] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
        if rows and "decision_latency_us_mean" in rows[0]:
            lat = [r["decision_latency_us_mean"] for r in rows]
            m, lo, hi = _bootstrap_ci(lat)
            summaries["arms"][arm]["decision_latency_us_mean"] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    if "tube_conformal" in all_rows and "static_conformal" in all_rows:
        summaries["comparisons"]["tube_conformal_vs_static_conformal"] = pair_comp(
            all_rows, "tube_conformal", "static_conformal"
        )
    if "tube_conformal" in all_rows and "raw" in all_rows:
        summaries["comparisons"]["tube_conformal_vs_raw"] = pair_comp(
            all_rows, "tube_conformal", "raw"
        )

    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["arm", "episode", "video_name"] + metric_cols
        w.writerow(cols)
        for arm, rows in all_rows.items():
            for r in rows:
                w.writerow([arm, r["episode"], r.get("video_name", ""), *[r[k] for k in metric_cols]])

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a - b) with paired resampling."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = a - b
    obs = float(np.mean(diffs))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(diffs[idx])
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return obs, lo, hi


def episode_cvar95(values: Sequence[float]) -> float:
    x = np.asarray(values, float)
    if len(x) == 0:
        return float("nan")
    q = np.percentile(x, 95)
    tail = x[x >= q]
    return float(np.mean(tail)) if len(tail) else float(q)


def load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_episodes_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    import csv as _csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))
