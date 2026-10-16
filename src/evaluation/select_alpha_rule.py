"""Apply the pre-registered rule configs/tube_v3_alpha_rule.yaml to VAL results (never test)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.run_tube_v3_final import load_cfg  # noqa: E402

def _p90(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, 100 * q))


def tail_upper(arm: pd.DataFrame, ref: pd.DataFrame, q: float, B: int, seed: int, level: float) -> tuple[float, float]:
    m = arm.merge(ref, on=["trace_file", "episode"], suffixes=("", "_r"))
    groups = [g for _, g in m.groupby("trace_file")]
    a = [g.rebuffer_total.to_numpy() for g in groups]
    r = [g.rebuffer_total_r.to_numpy() for g in groups]
    d0 = _p90(m.rebuffer_total.to_numpy(), q) - _p90(m.rebuffer_total_r.to_numpy(), q)
    rng = np.random.default_rng(seed)
    k = len(groups)
    bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, k, k)
        bs[b] = _p90(np.concatenate([a[i] for i in idx]), q) - _p90(np.concatenate([r[i] for i in idx]), q)
    return d0, float(np.percentile(bs, 100 * level))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default=str(ROOT / "results" / "tube_v3" / "adaptive_val"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tube_v3" / "adaptive_val" / "alpha_selection.json"))
    args = ap.parse_args()
    rule = load_cfg(ROOT / "configs" / "tube_v3_alpha_rule.yaml")
    assert rule["split"] == "val"
    grid = [float(x) for x in rule["alpha_grid"]]
    variants = list(rule["variants"])
    q, B, seed = float(rule["tail_quantile"]), int(rule["bootstrap_B"]), int(rule["bootstrap_seed"])
    level, tol = float(rule["upper_ci_level"]), float(rule["tail_tolerance_s"])
    root = Path(args.val_root)
    out: dict = {"rule_file": "configs/tube_v3_alpha_rule.yaml", "variants": {}}
    for v in variants:
        base, suf = ("maxscore", v[len("maxscore"):]) if v.startswith("maxscore") else ("step_static", v[len("step_static"):])
        per_fam: dict = {}
        for fam in rule["families"]:
            ref = pd.read_csv(root / fam / f"{rule['reference_arm']}.csv", comment="#")
            rows = []
            for a in grid:
                arm = f"tube_v2_{base}_a{a}{suf}"
                d = pd.read_csv(root / fam / f"{arm}.csv", comment="#")
                d0, U = tail_upper(d, ref, q, B, seed, level)
                rows.append(dict(alpha=a, arm=arm, qoe=float(d.episodic_qoe.mean()), rebuf=float(d.rebuffer_total.mean()),
                                 p90_rebuf=_p90(d.rebuffer_total.to_numpy(), q), dp90=d0, U=U, admissible=bool(U <= tol)))
            adm = [r for r in rows if r["admissible"]]
            if adm:
                best = sorted(adm, key=lambda r: (-r["qoe"], r["alpha"]))[0]
                flag = ""
            else:
                best = sorted(rows, key=lambda r: (r["U"], r["alpha"]))[0]
                flag = "no admissible"
            per_fam[fam] = {"candidates": rows, "chosen_alpha": best["alpha"], "flag": flag}
        fams = list(per_fam)
        all_adm = [a for a in grid if all(next(r for r in per_fam[f]["candidates"] if r["alpha"] == a)["admissible"] for f in fams)]
        macro = {a: float(np.mean([next(r for r in per_fam[f]["candidates"] if r["alpha"] == a)["qoe"] for f in fams])) for a in grid}
        if all_adm:
            g = sorted(all_adm, key=lambda a: (-macro[a], a))[0]
            gflag = ""
        else:
            maxU = {a: max(next(r for r in per_fam[f]["candidates"] if r["alpha"] == a)["U"] for f in fams) for a in grid}
            g = sorted(grid, key=lambda a: (maxU[a], a))[0]
            gflag = "no alpha admissible in every family"
        out["variants"][v] = {"per_family": per_fam, "global_alpha": g, "global_flag": gflag, "macro_qoe": macro}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for v, r in out["variants"].items():
        print(v, {f: (x["chosen_alpha"], x["flag"]) for f, x in r["per_family"].items()}, "global", r["global_alpha"], r["global_flag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
