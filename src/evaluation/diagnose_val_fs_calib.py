"""VAL-only: compare _fs calibration (raw_samples vs env_chunks) before freezing a new test config."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.trace_sets_v2 import calibration_dirs_for_eval, resolve_trace_dir  # noqa: E402
from src.training.conformal_tube_v2 import ConformalTubeV2Config  # noqa: E402
from src.training.conformal_tube_v3 import MaxScoreConformalTube  # noqa: E402
from src.training.tube_v2_calib import replay_throughputs_on_tube  # noqa: E402


def _fit_fs(alpha: float, dirs: list[Path], mode: str) -> dict:
    tube = MaxScoreConformalTube(ConformalTubeV2Config(alpha=alpha, horizon=5, freeze_scales=True))
    replay_throughputs_on_tube(tube, dirs, fs_calib_mode=mode)
    tube.finalize_frozen_calibration()
    sig = np.asarray(tube._frozen_sigma, float)
    return {
        "mode": mode,
        "alpha": alpha,
        "frozen_Q": float(tube._frozen_Q),
        "frozen_n_joint": int(tube._frozen_n),
        "sigma_h": sig.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-set", default="bb")
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.15, 0.2])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tube_v3" / "fs_calib_diagnose.json")
    args = ap.parse_args()
    variant = "native" if args.trace_set == "5g_harsh" else "default"
    if args.split == "val":
        resolve_trace_dir(args.trace_set, "val")  # ensure val exists
    dirs = list(calibration_dirs_for_eval(args.trace_set, variant=variant))
    report: dict = {"trace_set": args.trace_set, "split": args.split, "calib_dirs": [str(d) for d in dirs], "fits": []}
    for a in args.alphas:
        raw = _fit_fs(a, dirs, "raw_samples")
        env = _fit_fs(a, dirs, "env_chunks")
        ratio = [e / max(r, 1e-6) for r, e in zip(raw["sigma_h"], env["sigma_h"])]
        report["fits"].append(
            {
                "alpha": a,
                "raw": raw,
                "env_chunks": env,
                "sigma_ratio_env_over_raw": ratio,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
