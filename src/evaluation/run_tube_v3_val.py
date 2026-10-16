"""Tube-v3 validation campaign (VAL split only): split objective, alpha sweep, max-score tube."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.trace_sets_v2 import resolve_trace_dir  # noqa: E402
from src.control.two_timescale_conformal import ControlArmConfig, make_controlled_policy  # noqa: E402
from src.evaluation.clean_abr_eval import build_env, run_episode  # noqa: E402
from src.evaluation.eval_certified_shield_v18 import make_policy  # noqa: E402
from src.evaluation.eval_seeds import DEFAULT_MASTER_SEED  # noqa: E402
from src.evaluation.run_tube_v2_phase2 import (  # noqa: E402
    _append_csv_row,
    _episodes_done,
    _job_complete,
)
from src.evaluation.trace_provenance import verify_or_exit  # noqa: E402

OUT_ROOT = ROOT / "results" / "tube_v3" / "val"
ALPHAS = (0.1, 0.15, 0.2, 0.3)


def job_list() -> list[tuple[str, float | None]]:
    jobs: list[tuple[str, float | None]] = [
        ("robust_mpc_logu", None),
        ("mpc_fixed_quantile_native", None),
        *[(f"mpc_fixed_rho_{r}", None) for r in ("0.5", "0.55", "0.6", "0.65", "0.7")],
        ("tube_v2_step_static", None),
        ("tube_v2_step_aci", 0.002),
    ]
    for a in ALPHAS:
        jobs.append((f"tube_v2_step_static_a{a}", None))
        jobs.append((f"tube_v2_step_split_a{a}", None))
    jobs += [("tube_v2_maxscore_a0.1", None), ("tube_v2_maxscore_a0.2", None)]
    return jobs


def _key(arm: str, gamma: float | None) -> str:
    return arm if gamma is None else f"{arm}@g{gamma:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-sets", nargs="+", default=["5g_markov", "5g_harsh", "bb"])
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    verify_or_exit()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {"split": "val", "episodes": args.episodes, "jobs": [_key(*j) for j in job_list()],
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "provenance_status": "VERIFIED"}
    (OUT_ROOT / "meta.json").write_text(json.dumps(meta, indent=2))
    for ts in args.trace_sets:
        trace_dir = resolve_trace_dir(ts, "val")  # VAL ONLY
        for arm, gamma in job_list():
            key = _key(arm, gamma)
            path = OUT_ROOT / ts / (key.replace("@", "_at_") + ".csv")
            if _job_complete(path, args.episodes):
                continue
            done = _episodes_done(path) if path.is_file() else set()
            t0 = time.time()
            cfg = ControlArmConfig(aci_gamma=float(gamma if gamma is not None else 0.005))
            env = build_env(trace_dir, arm)
            base = make_policy("bba", env, None, False)
            pol = make_controlled_policy(env, arm, base, cfg, trace_set_key=ts)
            for ep in range(args.episodes):
                if ep in done:
                    continue
                if hasattr(pol, "on_episode_start"):
                    pol.on_episode_start()
                row = run_episode(env, pol, ep, master_seed=DEFAULT_MASTER_SEED, trace_key=ts)
                row.update(arm=arm, job_key=key, aci_gamma="" if gamma is None else gamma, split="val",
                           provenance_status="VERIFIED")
                _append_csv_row(path, row, "VERIFIED")
            print(f"{ts} {key} {time.time() - t0:.1f}s", flush=True)
    (OUT_ROOT / "DONE").write_text(datetime.now(timezone.utc).isoformat())
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
