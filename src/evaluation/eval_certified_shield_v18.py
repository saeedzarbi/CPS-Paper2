"""
Model-agnostic evaluation of the Certified Perceptual Shield (V18).

For a given base policy (a trained PPO checkpoint, or a checkpoint-free baseline
such as bitrate-greedy or buffer-based), this runs THREE arms on identical
(video, trace) episode pairs and reports the shield's runtime effect:

  RAW        : the base policy, unshielded.
  SAFETY     : base policy + conformal feasibility projection (banking OFF).
  CERTIFIED  : base policy + conformal feasibility + VMAF-knee bandwidth banking.

Paired across arms (same python-`random` seed per episode => same video+trace),
so Wilcoxon signed-rank and TOST equivalence tests are valid. Reports:

  * bandwidth (mean delivered bitrate) reduction:   CERTIFIED vs SAFETY / RAW,
  * rebuffering (mean, p95, p99) and stall-free rate,
  * VMAF non-inferiority via TOST within +/- epsilon (the perceptual budget),
  * empirical conformal coverage vs the 1 - alpha target.

Usage (run per policy; see run_certified_v18.sh):
  python src/evaluation/eval_certified_shield_v18.py \
      --policy ppo --ckpt results/models/master_v18_5g/proposed_v14/seed_0/final_model \
      --trace-dir data/standardized/test_traces_5g_v18 \
      --episodes 200 --epsilon 1.0 --alpha 0.10 \
      --out results/v18_certified/proposed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from configs.videos import (
    CPS_EPISODES, EVAL_VIDEOS, HELD_OUT_VIDEOS, MAX_CHUNKS, TRAIN_VIDEOS,
)
from src.environment.abr_multi_env_v18 import ABREnv
from src.training.certified_perceptual_shield import (
    CertifiedPerceptualShieldWrapper, CPShieldConfig, ConformalConfig)

P = get_paths()


# --------------------------------------------------------------------------- #
# Base policies (model-agnostic)
# --------------------------------------------------------------------------- #
class GreedyPolicy:
    """Always request the top rung (adversarial worst case; also what a pure
    bitrate-reward agent tends to do). No checkpoint required."""

    def __init__(self, n):
        self.n = n

    def predict(self, obs, deterministic=True):
        return self.n - 1, None


class BBAPolicy:
    """Buffer-based (BBA-like): rung grows linearly with buffer fill."""

    def __init__(self, env):
        self.env = env
        self.n = len(env.BITRATE_LEVELS)

    def predict(self, obs, deterministic=True):
        buf = float(getattr(self.env, "buffer_level", 0.0))
        frac = np.clip(buf / max(self.env.BUFFER_MAX, 1e-6), 0.0, 1.0)
        return int(round(frac * (self.n - 1))), None


class BOLAPolicy:
    """BOLA-BASIC (Spiteri et al., ``BOLA'' / ``From Theory to Practice''), the
    canonical buffer-based Lyapunov ABR used in the dash.js reference player.

    Pure buffer control (no throughput prediction). For each rung m it maximizes
        score_m = (Vp * (u_m + gp) - Q) / S_m,
    with normalized utility u_m = ln(bitrate_m / bitrate_0) + 1 (dash.js), buffer
    Q in seconds, and segment size S_m in bits. The parameters are the standard
    closed form anchoring the lowest rung at Q=B_min and the highest at Q=B_max:
        gp = (u_M - 1) / (B_max/B_min - 1),   Vp = B_min / gp.
    """

    def __init__(self, env, buffer_min=None):
        self.env = env
        self.n = len(env.BITRATE_LEVELS)
        br = np.asarray(env.BITRATE_LEVELS, dtype=float)
        v = np.log(br / br[0])            # v_0 = 0
        self.u = v + 1.0                  # dash.js normalized utilities
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        b_min = float(buffer_min) if buffer_min is not None else p  # one segment
        b_min = min(b_min, b_max - p)     # keep B_min < B_max
        self.gp = v[-1] / (b_max / b_min - 1.0)
        self.vp = b_min / self.gp

    def predict(self, obs, deterministic=True):
        env = self.env
        q = float(env.buffer_level)
        sizes = np.array(
            [env.get_chunk_size_bits(br, env.chunk_idx) for br in env.BITRATE_LEVELS],
            dtype=float,
        )
        score = (self.vp * (self.u + self.gp) - q) / np.maximum(sizes, 1.0)
        return int(np.argmax(score)), None


class RobustMPCPolicy:
    """Robust MPC (Yin et al., SIGCOMM 2015): plan over a finite horizon against a
    throughput prediction discounted by the recent worst-case prediction error.

    Prediction: harmonic mean of the last ``past`` observed per-chunk throughputs,
    divided by (1 + max recent relative error) for robustness. Only *past* observed
    throughputs are used (read from ``env.last_raw_throughput``), so there is no
    look-ahead into the trace. The planner enumerates rung sequences over the
    horizon and maximizes the environment's QoE (VMAF minus rebuffering and
    smoothness penalties), then plays the first action.
    """

    def __init__(self, env, horizon=5, past=5):
        self.env = env
        self.n = len(env.BITRATE_LEVELS)
        self.horizon = int(horizon)
        self.past = int(past)
        self.reb_pen = float(getattr(env, "REBUF_PENALTY_BASE", 100.0))
        self.smooth_w = float(getattr(env, "SMOOTH_PENALTY_WEIGHT", 1.0))
        self.vmaf = np.array(
            [env.current_vmaf_scores.get(br, 35.0) for br in env.BITRATE_LEVELS],
            dtype=float,
        )
        self._seq_cache = {}
        self._reset()

    def _reset(self):
        self._tp_hist = []
        self._err_hist = []
        self._last_pred = None
        self._last_video = getattr(self.env, "current_video_name", None)

    def _seqs(self, h):
        if h not in self._seq_cache:
            import itertools
            self._seq_cache[h] = np.array(
                list(itertools.product(range(self.n), repeat=h)), dtype=int)
        return self._seq_cache[h]

    def _observe(self):
        env = self.env
        vid = getattr(env, "current_video_name", None)
        if env.chunk_idx == 0 or vid != self._last_video:
            self._reset()
            self._last_video = vid
            return
        actual = float(env.last_raw_throughput)
        self._tp_hist.append(actual)
        if self._last_pred is not None and actual > 1e-6:
            self._err_hist.append(abs(self._last_pred - actual) / actual)

    def _predict_tp(self):
        env = self.env
        h = self._tp_hist[-self.past:]
        if not h:
            base = float(env.last_raw_throughput)
        else:
            base = len(h) / float(np.sum(1.0 / np.maximum(h, 1e-6)))  # harmonic
        err = max(self._err_hist[-self.past:], default=0.0)
        robust = base / (1.0 + err)
        self._last_pred = robust
        return max(robust, float(env.MIN_NETWORK_THROUGHPUT))

    def predict(self, obs, deterministic=True):
        env = self.env
        # refresh cached VMAF ladder in case the video changed
        if getattr(env, "current_video_name", None) != self._last_video:
            self.vmaf = np.array(
                [env.current_vmaf_scores.get(br, 35.0) for br in env.BITRATE_LEVELS],
                dtype=float,
            )
        self._observe()
        pred = self._predict_tp()  # kbps
        k = env.chunk_idx
        h = min(self.horizon, env.max_chunks - k)
        if h <= 0:
            return self.n - 1, None
        sizes = np.array(
            [[env.get_chunk_size_bits(br, min(k + j, env.max_chunks - 1))
              for br in env.BITRATE_LEVELS] for j in range(h)],
            dtype=float,
        )  # (h, n)
        seqs = self._seqs(h)                              # (N, h)
        dl = sizes[np.arange(h)[None, :], seqs] / (pred * 1000.0)  # (N, h)
        vm = self.vmaf[seqs]                              # (N, h)
        buf = np.full(seqs.shape[0], float(env.buffer_level))
        prev = np.full(seqs.shape[0], float(getattr(env, "last_vmaf", self.vmaf[0])))
        total = np.zeros(seqs.shape[0])
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        for j in range(h):
            reb = np.maximum(0.0, dl[:, j] - buf)
            buf = np.minimum(np.maximum(0.0, buf - dl[:, j]) + p, b_max)
            sm = np.abs(vm[:, j] - prev) if (k + j) > 0 else np.zeros_like(prev)
            total += vm[:, j] - self.reb_pen * reb - self.smooth_w * sm
            prev = vm[:, j]
        return int(seqs[int(np.argmax(total)), 0]), None


def _resolve_ppo_ckpt(ckpt: str | Path) -> str:
    """Normalize SB3 checkpoint path (strip .zip; SB3 appends it on load)."""
    p = Path(ckpt)
    candidate = p.with_suffix("") if p.suffix == ".zip" else p
    if candidate.with_suffix(".zip").exists():
        return str(candidate)
    raise FileNotFoundError(f"PPO checkpoint not found: {candidate}.zip")


class PPOWrapperPolicy:
    def __init__(self, ckpt, blind, obs_len):
        from stable_baselines3 import PPO
        self.model = PPO.load(_resolve_ppo_ckpt(ckpt), device="cpu")
        self.blind = blind
        self.obs_len = obs_len

    def predict(self, obs, deterministic=True):
        if self.blind:
            obs = obs.copy()
            obs[15:] = 0.0
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return int(a), None


def build_base_env(trace_dir, buffer_max, blind):
    class Env(ABREnv):
        pass
    if buffer_max is not None:
        Env.BUFFER_MAX = float(buffer_max)
        Env.BUFFER_TARGET = float(buffer_max) / 2.0
        Env.B_REF = max(2.0, float(buffer_max) / 3.0)
    env = Env(
        video_names=list(EVAL_VIDEOS),
        trace_dir=str(trace_dir),
        vmaf_dir=str(P["vmaf_scores"]),
        siti_dir=str(P["content_features"]),
        max_chunks=MAX_CHUNKS, random_seed=0, use_lyapunov=True, use_future=True,
    )
    return env


def make_policy(kind, env, ckpt, blind):
    if kind == "greedy":
        return GreedyPolicy(len(env.BITRATE_LEVELS))
    if kind == "bba":
        return BBAPolicy(env)
    if kind == "bola":
        return BOLAPolicy(env)
    if kind == "mpc":
        return RobustMPCPolicy(env)
    if kind == "ppo":
        if not ckpt:
            raise ValueError("--ckpt is required for --policy ppo")
        return PPOWrapperPolicy(ckpt, blind, env.observation_space.shape[0])
    raise ValueError(f"unknown policy kind: {kind}")


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def run_arm(arm, kind, ckpt, blind, trace_dir, buffer_max, episodes, epsilon, alpha,
            shield=None):
    shield = shield or {}
    base = build_base_env(trace_dir, buffer_max, blind)
    if arm == "raw":
        env = base
    else:
        banking = (arm == "certified")
        cfg = CPShieldConfig(
            enabled=True, enable_banking=banking, epsilon_vmaf=epsilon,
            enable_conformal=True,
            conformal=ConformalConfig(alpha=alpha, window=200, k_predict=5),
            safety_margin=0.5, min_buffer=0.3,
            # improvements (items 1/3/4): risk-aware budget + dip forecasting.
            # Only meaningful when banking is on; default off for back-compat.
            predictive=bool(banking and shield.get("predictive", False)),
            lookahead=int(shield.get("lookahead", 1)),
            epsilon_risk=float(shield.get("epsilon_risk", max(4.0, epsilon))),
            risk_buffer=float(shield.get("risk_buffer", 8.0)),
            forecast_dips=bool(banking and shield.get("forecast", False)),
            horizon_quantile=float(shield.get("horizon_quantile", 0.2)),
        )
        env = CertifiedPerceptualShieldWrapper(base, cfg)

    policy = make_policy(kind, base, ckpt, blind)
    n = len(base.BITRATE_LEVELS)

    rows = []
    for ep in range(episodes):
        random.seed(1000 + ep)          # pairs video+trace across arms
        obs, info = env.reset(seed=1000 + ep)
        reb, vmafs, bitrates, buffers, chunk_reb = 0.0, [], [], [], []
        interv, done = 0, False
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            rb = float(info.get("rebuffer", 0.0))
            reb += rb
            chunk_reb.append(rb)
            vmafs.append(float(info.get("vmaf", 0.0)))
            bitrates.append(float(info.get("bitrate", 0.0)))
            buffers.append(float(info.get("buffer", 0.0)))
            interv += int(info.get("shield_intervened", 0))
        rows.append({
            "episode": ep,
            "rebuffer_total": reb,
            "rebuffer_p95_chunk": float(np.percentile(chunk_reb, 95)),
            "rebuffer_p99_chunk": float(np.percentile(chunk_reb, 99)),
            "stallfree": 1.0 if reb <= 1e-6 else 0.0,
            "vmaf_mean": float(np.mean(vmafs)),
            "bitrate_mean_kbps": float(np.mean(bitrates)),
            "buffer_mean": float(np.mean(buffers)),
            "interv_rate": interv / max(len(chunk_reb), 1),
            "conformal_coverage": float(info.get("conformal_coverage", float("nan"))),
        })
    return rows


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _bootstrap_ci(x, n_boot=5000, ci=0.95, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return (float(x.mean()), float(lo), float(hi))


def _norm_cdf(z):
    import math
    return 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))


def _wilcoxon(a, b):
    """Paired two-sided test. Uses scipy's Wilcoxon signed-rank if available;
    otherwise a scipy-free paired sign-flip permutation test on the mean diff."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return float("nan"), 1.0
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        d = a - b
        rng = np.random.default_rng(0)
        obs = abs(d.mean())
        signs = rng.choice([-1.0, 1.0], size=(20000, len(d)))
        perm = np.abs((signs * d).mean(axis=1))
        p = float((perm >= obs - 1e-12).mean())
        return float("nan"), max(p, 1.0 / 20000)


def _tost_noninferiority(treat, ref, margin):
    """TOST equivalence of paired (treat - ref) within +/- margin.

    Returns (mean_diff, p_equiv). p_equiv < 0.05 => VMAF is statistically within
    the perceptual budget, i.e. banking is perceptually non-inferior.
    Uses Student-t if scipy is present, else a normal approximation."""
    d = np.asarray(treat, float) - np.asarray(ref, float)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0:
        return md, (0.0 if abs(md) < margin else 1.0)
    se = sd / np.sqrt(n)
    t_low = (md - (-margin)) / se     # H0: diff <= -margin
    t_high = (md - margin) / se       # H0: diff >= +margin
    try:
        from scipy.stats import t
        p_low = 1 - t.cdf(t_low, df=n - 1)
        p_high = t.cdf(t_high, df=n - 1)
    except Exception:
        p_low = 1 - _norm_cdf(t_low)
        p_high = _norm_cdf(t_high)
    return md, float(max(p_low, p_high))


def summarize(all_rows, epsilon, alpha, out_dir, tag):
    import csv
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-episode dump
    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "episode", "rebuffer_total", "rebuffer_p95_chunk",
                    "rebuffer_p99_chunk", "stallfree", "vmaf_mean",
                    "bitrate_mean_kbps", "buffer_mean", "interv_rate", "conformal_coverage"])
        for arm, rows in all_rows.items():
            for r in rows:
                w.writerow([arm, r["episode"], r["rebuffer_total"], r["rebuffer_p95_chunk"],
                            r["rebuffer_p99_chunk"], r["stallfree"], r["vmaf_mean"],
                            r["bitrate_mean_kbps"], r["buffer_mean"], r["interv_rate"],
                            r["conformal_coverage"]])

    def col(arm, key):
        return [r[key] for r in all_rows[arm]]

    cov_target = 1.0 - alpha
    summary = {"tag": tag, "epsilon": epsilon, "alpha": alpha,
               "coverage_target": cov_target,
               "conformal_method": "split-conformal (n+1) finite-sample lower bound",
               "arms": {}}
    for arm in all_rows:
        m = {}
        for key in ["rebuffer_total", "rebuffer_p95_chunk", "rebuffer_p99_chunk",
                    "stallfree", "vmaf_mean", "bitrate_mean_kbps", "buffer_mean", "interv_rate"]:
            mean, lo, hi = _bootstrap_ci(col(arm, key))
            m[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        cov = [c for c in col(arm, "conformal_coverage") if not np.isnan(c)]
        cov_mean = float(np.mean(cov)) if cov else float("nan")
        m["conformal_coverage"] = cov_mean
        m["conformal_coverage_meets_target"] = (
            bool(cov_mean >= cov_target) if cov else None)
        summary["arms"][arm] = m

    comps = {}
    if "certified" in all_rows and "safety" in all_rows:
        comps["certified_vs_safety"] = _pair_report(all_rows, "certified", "safety", epsilon)
    if "certified" in all_rows and "raw" in all_rows:
        comps["certified_vs_raw"] = _pair_report(all_rows, "certified", "raw", epsilon)
    if "safety" in all_rows and "raw" in all_rows:
        comps["safety_vs_raw"] = _pair_report(all_rows, "safety", "raw", epsilon)
    summary["comparisons"] = comps

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# QoE stall weights: VMAF points lost per second of rebuffering.
# ~1-3 fidelity-favoring, ~5-10 balanced, >=~20 stall-averse.
QOE_WEIGHTS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def _qoe_report(vm_t, vm_r, reb_t, reb_r, weights=QOE_WEIGHTS):
    """Paired fidelity-vs-stall QoE, QoE(w) = mean_VMAF - w * rebuffer_seconds.

    Reports the crossover weight ``w* = dVMAF / dReb`` (the stall weight at which
    the two arms have equal mean QoE) plus a paired Wilcoxon test of the QoE
    difference at reference weights, so a reader can see over what QoE regime the
    treatment (e.g. banking, or a co-trained policy) is the net winner."""
    d_vmaf = float(vm_t.mean() - vm_r.mean())
    d_reb = float(reb_t.mean() - reb_r.mean())
    w_star = float(d_vmaf / d_reb) if abs(d_reb) > 1e-12 else None
    table = []
    for w in weights:
        q_t, q_r = vm_t - w * reb_t, vm_r - w * reb_r
        _, p = _wilcoxon(q_t, q_r)
        d = float(q_t.mean() - q_r.mean())
        table.append({"w": float(w), "qoe_treat": float(q_t.mean()),
                      "qoe_ref": float(q_r.mean()), "qoe_diff": d,
                      "wilcoxon_p": float(p), "treat_wins": bool(d > 0)})
    return {"vmaf_diff": d_vmaf, "rebuffer_diff_s": d_reb,
            "qoe_crossover_w": w_star, "qoe_by_weight": table}


def _pair_report(all_rows, treat, ref, epsilon):
    def col(arm, key):
        return np.asarray([r[key] for r in all_rows[arm]], float)
    br_t, br_r = col(treat, "bitrate_mean_kbps"), col(ref, "bitrate_mean_kbps")
    reb_t, reb_r = col(treat, "rebuffer_total"), col(ref, "rebuffer_total")
    vm_t, vm_r = col(treat, "vmaf_mean"), col(ref, "vmaf_mean")

    _, p_reb = _wilcoxon(reb_t, reb_r)
    _, p_bw = _wilcoxon(br_t, br_r)
    md_vmaf, p_equiv = _tost_noninferiority(vm_t, vm_r, epsilon)
    bw_pct = 100.0 * (br_t.mean() - br_r.mean()) / max(br_r.mean(), 1e-9)
    reb_pct = 100.0 * (reb_t.mean() - reb_r.mean()) / max(reb_r.mean(), 1e-9)
    return {
        "bandwidth_reduction_pct": float(bw_pct),
        "bandwidth_wilcoxon_p": p_bw,
        "rebuffer_change_pct": float(reb_pct),
        "rebuffer_wilcoxon_p": p_reb,
        "vmaf_mean_diff": float(md_vmaf),
        "vmaf_tost_p_equiv": float(p_equiv),
        "vmaf_within_epsilon": bool(p_equiv < 0.05),
        "qoe": _qoe_report(vm_t, vm_r, reb_t, reb_r),
    }


def main():
    ap = argparse.ArgumentParser(description="Model-agnostic Certified Perceptual Shield evaluation (v18).")
    ap.add_argument("--policy", choices=["ppo", "greedy", "bba", "bola", "mpc"], required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--blind", action="store_true", help="mask content features (Pensieve).")
    ap.add_argument("--trace-dir", type=str, default=str(P["test_traces"]))
    ap.add_argument("--buffer", type=float, default=None, help="override BUFFER_MAX (default v18=12s).")
    ap.add_argument("--episodes", type=int, default=CPS_EPISODES)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--arms", type=str, default="raw,safety,certified")
    ap.add_argument("--out", type=str, required=True)
    # shield improvements (items 1/3/4): applied to the `certified` arm.
    ap.add_argument("--predictive", action="store_true",
                    help="risk-aware perceptual budget eps(B) + look-ahead pre-banking.")
    ap.add_argument("--forecast", action="store_true",
                    help="plan the look-ahead against a low quantile of recent throughput (tail).")
    ap.add_argument("--epsilon-risk", type=float, default=4.0)
    ap.add_argument("--risk-buffer", type=float, default=8.0)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--horizon-quantile", type=float, default=0.2)
    args = ap.parse_args()

    shield = {
        "predictive": args.predictive, "forecast": args.forecast,
        "epsilon_risk": args.epsilon_risk, "risk_buffer": args.risk_buffer,
        "lookahead": (max(args.lookahead, 6) if args.forecast else args.lookahead),
        "horizon_quantile": args.horizon_quantile,
    }

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_rows = {}
    for arm in arms:
        print(f"[v18-eval] arm={arm} policy={args.policy} episodes={args.episodes} "
              f"trace_dir={args.trace_dir} predictive={args.predictive} forecast={args.forecast}")
        all_rows[arm] = run_arm(arm, args.policy, args.ckpt, args.blind, args.trace_dir,
                                args.buffer, args.episodes, args.epsilon, args.alpha,
                                shield=shield)

    tag = f"{args.policy}{'_blind' if args.blind else ''}"
    summary = summarize(all_rows, args.epsilon, args.alpha, args.out, tag)

    print("\n================ V18 CERTIFIED PERCEPTUAL SHIELD ================")
    print(f"policy={tag} | episodes={args.episodes} | eps={args.epsilon} VMAF | "
          f"alpha={args.alpha} (coverage target {1.0 - args.alpha:.3f}, split-conformal n+1)")
    print(f"{'arm':>10}{'reb_mean':>10}{'reb_p95':>9}{'stallfree':>10}{'vmaf':>8}"
          f"{'bitrate':>9}{'buffer':>8}{'interv':>8}{'cover':>7}{'>=tgt':>7}")
    for arm in arms:
        a = summary["arms"][arm]
        meets = a.get("conformal_coverage_meets_target")
        meets_s = "-" if meets is None else ("yes" if meets else "NO")
        print(f"{arm:>10}{a['rebuffer_total']['mean']:>10.3f}{a['rebuffer_p95_chunk']['mean']:>9.3f}"
              f"{a['stallfree']['mean']:>10.3f}{a['vmaf_mean']['mean']:>8.2f}"
              f"{a['bitrate_mean_kbps']['mean']:>9.0f}{a['buffer_mean']['mean']:>8.2f}"
              f"{a['interv_rate']['mean']:>8.3f}{a['conformal_coverage']:>7.3f}{meets_s:>7}")
    for name, c in summary["comparisons"].items():
        print(f"\n[{name}]  bandwidth {c['bandwidth_reduction_pct']:+.1f}% (p={c['bandwidth_wilcoxon_p']:.1e})  "
              f"rebuffer {c['rebuffer_change_pct']:+.1f}% (p={c['rebuffer_wilcoxon_p']:.1e})")
        print(f"           VMAF diff {c['vmaf_mean_diff']:+.3f}  TOST p_equiv={c['vmaf_tost_p_equiv']:.1e}  "
              f"within +/-{args.epsilon}: {c['vmaf_within_epsilon']}")
        q = c.get("qoe", {})
        ws = q.get("qoe_crossover_w")
        ws_s = "n/a" if ws is None else f"{ws:.2f} VMAF/s"
        print(f"           QoE: dVMAF {q.get('vmaf_diff', float('nan')):+.2f}  dReb "
              f"{q.get('rebuffer_diff_s', float('nan')):+.2f}s  crossover w*={ws_s}")
        for row in q.get("qoe_by_weight", []):
            win = "treat" if row["treat_wins"] else "ref"
            print(f"             w={row['w']:>5.1f}  QoE_diff {row['qoe_diff']:+7.2f}  "
                  f"p={row['wilcoxon_p']:.1e}  winner={win}")
    print(f"\nWrote {args.out}/summary.json and episodes.csv")


if __name__ == "__main__":
    main()
