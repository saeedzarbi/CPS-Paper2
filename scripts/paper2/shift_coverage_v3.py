"""Distribution-shift check (tube-only, offline): markov VAL -> harsh VAL throughput concatenated
per episode. Compares step_static vs step_aci (alpha=0.15, gamma=0.002, H=5) rolling per-step
coverage around the shift. Uses VAL traces only (never test). No ABR simulation => no rebuffer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.trace_sets_v2 import calibration_dirs_for_eval, resolve_trace_dir  # noqa: E402
from src.training.conformal_tube_v2 import ConformalThroughputTubeV2, ConformalTubeV2Config  # noqa: E402
from src.training.tube_v2_calib import replay_throughputs_on_tube  # noqa: E402

H, ALPHA, GAMMA, SEG = 5, 0.15, 0.002, 300


def tp(path: Path) -> np.ndarray:
    return np.asarray(json.loads(path.read_text())["throughput_kbps"], float)


def main() -> None:
    mk = sorted(resolve_trace_dir("5g_markov", "val").glob("*.json"))
    hs = sorted(resolve_trace_dir("5g_harsh", "val").glob("*.json"))
    out = {}
    for name, aci in (("step_static", False), ("step_aci", True)):
        hits = []  # [episode][t][h]
        for e in range(40):
            a = tp(mk[e % len(mk)]); b = tp(hs[e % len(hs)])
            a = a[:SEG] if len(a) >= SEG else np.resize(a, SEG)
            b = b[:SEG] if len(b) >= SEG else np.resize(b, SEG)
            trace = np.concatenate([a, b])
            tube = ConformalThroughputTubeV2(ConformalTubeV2Config(
                mode="per_step", alpha=ALPHA, horizon=H, use_aci=aci, aci_gamma=GAMMA))
            replay_throughputs_on_tube(tube, calibration_dirs_for_eval("5g_markov", variant="default"))
            bounds = {}
            ep = np.full((len(trace), H), np.nan)
            for j, y in enumerate(trace):
                if j > 0:
                    for h in range(1, H + 1):
                        d = j - (h - 1)
                        if d in bounds and d >= 1:
                            ep[j, h - 1] = float(y >= bounds[d][h - 1] - 1e-6)
                bounds[j] = tube.horizon_lower_bounds_kbps(H, decision_chunk=j)
                tube.observe(float(y), realized_chunk=j)
            hits.append(ep)
        H_ = np.stack(hits)  # E x T x H
        out[name] = H_
    res = {}
    for name, H_ in out.items():
        m = np.nanmean(H_, axis=0)  # T x H
        res[name] = {
            "pre_shift_cov": np.nanmean(m[SEG - 100:SEG], axis=0).round(3).tolist(),
            "post_0_50": np.nanmean(m[SEG:SEG + 50], axis=0).round(3).tolist(),
            "post_0_150": np.nanmean(m[SEG:SEG + 150], axis=0).round(3).tolist(),
            "post_150_300": np.nanmean(m[SEG + 150:], axis=0).round(3).tolist(),
            "rolling_h1": [float(np.nanmean(m[max(0, t - 20):t + 1, 0])) for t in range(0, 2 * SEG)],
        }
    Path(ROOT / "results/tube_v3/shift").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/tube_v3/shift/shift_coverage.json").write_text(json.dumps(res))
    for k, v in res.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "rolling_h1"})


if __name__ == "__main__":
    main()
