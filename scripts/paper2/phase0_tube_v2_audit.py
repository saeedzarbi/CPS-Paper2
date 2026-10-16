#!/usr/bin/env python3
"""Phase 0 audit for Tube-v2 (resources, traces, ACI, latency, pilot timing)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.paths import STANDARDIZED_DIR
from src.control.two_timescale_conformal import ABLATION_ARMS, ControlArmConfig, make_controlled_policy
from src.evaluation.clean_abr_eval import HEADLINE_TRACE_SETS, build_env, run_arm
from src.evaluation.eval_certified_shield_v18 import make_policy

PILOT_EPS = 5
PILOT_ARMS = tuple(ABLATION_ARMS)
HOSTS = ("bba", "mpc", "greedy")


def _shell(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, cwd=str(ROOT)).strip()


def resource_report() -> dict:
    out = {}
    for key, cmd in [
        ("nproc", "nproc"),
        ("free_h", "free -h"),
        ("df", "df -h ."),
        ("uptime", "uptime"),
        ("lscpu_model", "lscpu | grep 'Model name' || true"),
    ]:
        try:
            out[key] = _shell(cmd)
        except subprocess.CalledProcessError as e:
            out[key] = str(e)
    try:
        out["nvidia_smi"] = _shell("nvidia-smi 2>&1 | head -3")
    except subprocess.CalledProcessError:
        out["nvidia_smi"] = "none"
    return out


def trace_inventory() -> list[dict]:
    sets = {
        **HEADLINE_TRACE_SETS,
        "5g_v18": STANDARDIZED_DIR / "test_traces_5g_v18",
        "fcc_fixed": STANDARDIZED_DIR / "eval_fcc_fixed",
        "5g_irish": STANDARDIZED_DIR / "test_traces_5g_irish_mmsys20",
        "train_5g_markov": STANDARDIZED_DIR / "train_traces_5g_markov",
    }
    rows = []
    for name, path in sorted(sets.items()):
        n = len(list(path.glob("*.json"))) if path.is_dir() else 0
        rows.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "exists": path.is_dir(),
                "n_traces": n,
            }
        )
    return rows


def aci_status() -> dict[str, bool | str]:
    trace_dir = HEADLINE_TRACE_SETS["5g_markov"]
    env = build_env(trace_dir, "tube_conformal")
    base = make_policy("bba", env, None, False)
    cfg = ControlArmConfig()

    def _tube_aci(arm: str) -> bool | str:
        pol = make_controlled_policy(env, arm, base, cfg)  # type: ignore[arg-type]
        if arm in ("raw", "heuristic_85", "static_conformal"):
            return "n/a"
        tube = getattr(pol, "tube", None)
        if tube is None:
            return "n/a"
        # TubeMPCPolicy.tube -> ConformalThroughputTube
        est = getattr(tube, "tube", None)
        if est is None:
            return "unknown"
        return getattr(est, "_aci", None) is not None

    return {arm: _tube_aci(arm) for arm in ABLATION_ARMS}


def decision_latencies_us() -> dict[str, float]:
    trace_dir = HEADLINE_TRACE_SETS["5g_markov"]
    out: dict[str, float] = {}
    for arm in PILOT_ARMS:
        rows = run_arm(
            "bba",
            trace_dir,
            arm,
            PILOT_EPS,
            time_decisions=True,
            workers=1,
        )
        lat = [r["decision_latency_us_mean"] for r in rows if "decision_latency_us_mean" in r]
        out[arm] = float(sum(lat) / len(lat)) if lat else -1.0
    from src.evaluation.clean_abr_eval import run_episode

    for host in HOSTS:
        env = build_env(trace_dir, "raw")
        pol = make_policy(host, env, None, False)
        us = []
        for ep in range(3):
            row = run_episode(env, pol, ep, time_decisions=True)
            us.append(row["decision_latency_us_mean"])
        out[f"host_{host}"] = float(sum(us) / len(us))
    return out


def pilot_wall_seconds() -> dict:
    trace_dir = HEADLINE_TRACE_SETS["5g_markov"]
    t0 = time.perf_counter()
    for arm in PILOT_ARMS:
        run_arm("bba", trace_dir, arm, PILOT_EPS, workers=1)
    elapsed = time.perf_counter() - t0
    per_ep_arm = elapsed / (len(PILOT_ARMS) * PILOT_EPS)
    # Extrapolate: extended OOD ~4 trace sets × 6 arms × 204 ep (from prompt phases)
    extrap_ood = per_ep_arm * 4 * 6 * 204
    extrap_headline = per_ep_arm * 2 * 3 * 204
    return {
        "pilot_episodes_per_arm": PILOT_EPS,
        "pilot_arms": len(PILOT_ARMS),
        "pilot_wall_s": round(elapsed, 2),
        "per_episode_per_arm_s": round(per_ep_arm, 4),
        "extrap_headline_2sets_3arms_204ep_s": round(extrap_headline, 0),
        "extrap_ood_4sets_6arms_204ep_s": round(extrap_ood, 0),
        "workers": 1,
    }


def main() -> int:
    out_dir = ROOT / "results" / "tube_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "resources": resource_report(),
        "traces": trace_inventory(),
        "aci_on": aci_status(),
        "decision_latency_us_mean": decision_latencies_us(),
        "pilot": pilot_wall_seconds(),
    }
    path = out_dir / "checkpoint0_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(report, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
