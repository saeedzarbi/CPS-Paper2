"""Run the FROZEN Tube-v3 configuration (configs/tube_v3_final.yaml) once on the TEST split."""
from __future__ import annotations

import argparse
import hashlib
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
from src.evaluation.run_tube_v2_phase2 import _append_csv_row, _episodes_done, _job_complete  # noqa: E402
from src.evaluation.trace_provenance import verify_or_exit  # noqa: E402

CFG = ROOT / "configs" / "tube_v3_final.yaml"
OUT_ROOT = ROOT / "results" / "tube_v3" / "final_test"


def load_cfg(path: Path = CFG) -> dict:
    """Minimal parser for the frozen yaml (scalars, inline lists, '- item' list under arms)."""
    cfg: dict = {"arms": []}
    cur_list: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and cur_list is not None:
            cfg[cur_list].append(line[4:].strip())
            continue
        cur_list = None
        k, v = line.split(":", 1)
        v = v.strip()
        if v == "":
            cur_list = k
            cfg.setdefault(k, [])
        elif v.startswith("["):
            cfg[k] = [x.strip() for x in v.strip("[]").split(",") if x.strip()]
        else:
            cfg[k] = v
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-sets", nargs="+", default=None)
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))
    out_root = Path(args.out)
    assert cfg["split"] == "test"
    man = ROOT / cfg["manifest_file"]
    sha = hashlib.sha256(man.read_bytes()).hexdigest()
    if sha != cfg["manifest_sha256"]:
        print(f"manifest sha mismatch: {sha}", file=sys.stderr)
        return 2
    verify_or_exit()
    episodes = int(cfg["episodes_per_arm"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta.json").write_text(json.dumps({**cfg, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                                    "provenance_status": "VERIFIED"}, indent=2))
    for ts in args.trace_sets or cfg["trace_sets"]:
        trace_dir = resolve_trace_dir(ts, "test")
        for job in list(cfg.get(f"arms_{ts}", [])) + list(cfg["arms"]):  # per-family arms first
            arm, _, g = job.partition("@g")
            gamma = float(g) if g else None
            path = out_root / ts / (job.replace("@", "_at_") + ".csv")
            if _job_complete(path, episodes):
                continue
            done = _episodes_done(path) if path.is_file() else set()
            t0 = time.time()
            ccfg = ControlArmConfig(
                aci_gamma=gamma if gamma is not None else 0.005,
                fs_calib_mode=str(cfg.get("fs_calib_mode", "raw_samples")),
            )
            env = build_env(trace_dir, arm)
            base = make_policy(cfg["host"], env, None, False)
            pol = make_controlled_policy(env, arm, base, ccfg, trace_set_key=ts)
            for ep in range(episodes):
                if ep in done:
                    continue
                if hasattr(pol, "on_episode_start"):
                    pol.on_episode_start()
                row = run_episode(env, pol, ep, master_seed=int(cfg["master_seed"]), trace_key=ts)
                row.update(arm=arm, job_key=job, aci_gamma="" if gamma is None else gamma, split="test",
                           provenance_status="VERIFIED")
                _append_csv_row(path, row, "VERIFIED")
            print(f"{ts} {job} {time.time() - t0:.1f}s", flush=True)
    (out_root / "DONE").write_text(datetime.now(timezone.utc).isoformat())
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
