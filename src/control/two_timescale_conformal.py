"""
Conformal control arms for CleanABREnv (Paper 2 headline: raw / static / tube).

`two_timescale_conformal` (early abort) removed from headline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

from src.training.certified_perceptual_shield import ConformalConfig, ConformalThroughputEstimator
from src.training.conformal_tube import ConformalTubeConfig
from src.training.conformal_tube_v2 import ConformalTubeV2Config, TubeMode
from src.training.tube_mpc_policy import TubeMPCConfig, TubeMPCPolicy
from src.training.fixed_quantile_mpc import (
    FixedQuantileMPCPolicy,
    FixedRhoMPCPolicy,
    load_or_fit_quantile,
)
from src.training.robust_mpc_logu_policy import RobustMPCLogUtilityPolicy
from src.training.tube_mpc_policy_v2 import TubeV2MPCConfig, TubeV2MPCPolicy

HeadlineArm = Literal["raw", "static_conformal", "tube_conformal"]
AblationArm = Literal[
    "heuristic_85",
    "symmetric_tube",
    "fixed_alpha_tube",
]
BaselineArm = Literal[
    "robust_mpc_logu",
    "mpc_fixed_quantile",
    "mpc_fixed_quantile_native",
    "mpc_fixed_quantile_harsh",
    "mpc_fixed_rho_0.5",
    "mpc_fixed_rho_0.55",
    "mpc_fixed_rho_0.6",
    "mpc_fixed_rho_0.65",
    "mpc_fixed_rho_0.7",
]
TubeV2Arm = Literal[
    "tube_v2_bonf_static",
    "tube_v2_bonf_aci",
    "tube_v2_step_static",
    "tube_v2_step_aci",
    "tube_v2_bonf_static_nocap",
    "tube_v2_bonf_aci_nocap",
    "tube_v2_step_static_nocap",
    "tube_v2_step_aci_nocap",
]
EvalArm = Union[HeadlineArm, AblationArm, TubeV2Arm, BaselineArm]
ArmName = HeadlineArm  # backward-compatible alias

HEADLINE_ARMS: tuple[HeadlineArm, ...] = ("raw", "static_conformal", "tube_conformal")
ABLATION_ARMS: tuple[EvalArm, ...] = (
    "raw",
    "heuristic_85",
    "static_conformal",
    "symmetric_tube",
    "fixed_alpha_tube",
    "tube_conformal",
)

RHO_SWEEP_ARMS: tuple[str, ...] = tuple(
    f"mpc_fixed_rho_{rho}" for rho in ("0.5", "0.55", "0.6", "0.65", "0.7")
)

PHASE3_BASELINE_ARMS: tuple[str, ...] = (
    "raw",
    "static_conformal",
    "tube_conformal",
    "robust_mpc_logu",
    "mpc_fixed_quantile",
    "mpc_fixed_quantile_native",
    *RHO_SWEEP_ARMS,
    "tube_v2_step_static_nocap",
)

SMOKE_ARMS: tuple[str, ...] = PHASE3_BASELINE_ARMS + (
    "tube_v2_bonf_static",
    "tube_v2_bonf_aci",
    "tube_v2_step_static",
    "tube_v2_step_aci",
)

TUBE_V2_ARMS: tuple[TubeV2Arm, ...] = (
    "tube_v2_bonf_static",
    "tube_v2_bonf_aci",
    "tube_v2_step_static",
    "tube_v2_step_aci",
    "tube_v2_bonf_static_nocap",
    "tube_v2_bonf_aci_nocap",
    "tube_v2_step_static_nocap",
    "tube_v2_step_aci_nocap",
)


def parse_tube_v3_arm(arm: str) -> dict | None:
    """Tube-v3 arms: tube_v2_step_split[_aX], tube_v2_step_static_aX, tube_v2_maxscore_aX."""
    name = str(arm)
    alpha = None
    m = __import__("re").fullmatch(r"(tube_v2_(?:step_split|step_static|maxscore))(?:_a(\d*\.?\d+))?", name)
    if not m:
        return None
    base, a = m.group(1), m.group(2)
    if base == "tube_v2_step_static" and a is None:
        return None  # legacy arm handled by parse_tube_v2_arm (unchanged)
    alpha = float(a) if a is not None else None
    kind = {"tube_v2_step_split": "split", "tube_v2_step_static": "static", "tube_v2_maxscore": "maxscore"}[base]
    return {"kind": kind, "alpha": alpha}


def parse_tube_v2_arm(arm: str) -> tuple[TubeMode, bool, bool] | None:
    """Return (mode, use_aci, capped) or None if not a tube_v2 arm name."""
    if not arm.startswith("tube_v2_"):
        return None
    capped = True
    if arm.endswith("_nocap"):
        capped = False
        arm = arm[: -len("_nocap")]
    tail = arm[len("tube_v2_") :]
    mapping: dict[str, tuple[TubeMode, bool]] = {
        "bonf_static": ("bonferroni", False),
        "bonf_aci": ("bonferroni", True),
        "step_static": ("per_step", False),
        "step_aci": ("per_step", True),
    }
    if tail not in mapping:
        return None
    mode, aci = mapping[tail]
    return mode, aci, capped


@dataclass
class ControlArmConfig:
    alpha: float = 0.10
    safety_margin: float = 0.5
    horizon: int = 5
    horizon_ref: int = 5
    cold_start_cap: float = 0.80
    cold_start_chunks: int = 20
    upward_smooth_weight: float = 2.0
    aci_gamma: float = 0.005

    def effective_safety_margin(self) -> float:
        """Buffer reserve for tube arms; H=1 static uses full margin (horizon sweep baseline)."""
        h = max(int(self.horizon), 1)
        ref = max(int(self.horizon_ref), 1)
        if h <= 1:
            return float(self.safety_margin)
        # Ramp reserve with H (not H/ref): avoids an overly loose H=2 that collapses QoE.
        t = (h - 1) / max(ref - 1, 1)
        scale = 0.55 + 0.45 * t
        return float(self.safety_margin) * scale


class StaticConformalProjector:
    """Single-step feasibility under harmonic prediction + split conformal lb."""

    def __init__(self, env, cfg: ControlArmConfig | None = None, *, adapt: bool = True):
        self.env = env
        self.cfg = cfg or ControlArmConfig()
        self.adapt = adapt
        self._est = ConformalThroughputEstimator(
            ConformalConfig(alpha=self.cfg.alpha, window=200, k_predict=5)
        )
        self._tp_hist: list[float] = []

    def on_episode_start(self):
        self._est = ConformalThroughputEstimator(
            ConformalConfig(alpha=self.cfg.alpha, window=200, k_predict=5)
        )
        self._tp_hist = []

    def observe(self):
        env = self.env
        if env.chunk_idx == 0:
            self.on_episode_start()
            return
        tp = float(getattr(env, "last_raw_throughput", 0.0))
        if tp > 0:
            self._tp_hist.append(tp)
            if self.adapt:
                self._est.update(tp)

    def _harmonic_kbps(self) -> float:
        h = self._tp_hist[-5:]
        if not h:
            return float(max(getattr(self.env, "last_raw_throughput", 500.0), 10.0))
        return float(len(h) / np.sum(1.0 / np.maximum(h, 1e-6)))

    def lower_bound_kbps(self) -> float:
        if self.adapt:
            lb = float(self._est.lower_bound())
            if self.env.chunk_idx < self.cfg.cold_start_chunks:
                pred = self._harmonic_kbps()
                lb = min(lb, self.cfg.cold_start_cap * pred)
        else:
            scale = 1.0 - float(self.cfg.alpha)
            lb = scale * self._harmonic_kbps()
        return max(lb, 10.0)

    def max_safe_action(self, proposed: int | None = None) -> int:
        """Highest rung index feasible under 1-step conformal bound (<= proposed if given)."""
        self.observe()
        lb = self.lower_bound_kbps()
        env = self.env
        k = int(env.chunk_idx)
        buf = float(env.buffer_level)
        cap = max(buf - self.cfg.effective_safety_margin(), 0.1)
        hi = int(proposed) if proposed is not None else len(env.BITRATE_LEVELS) - 1
        best = 0
        for a in range(hi, -1, -1):
            br = int(env.BITRATE_LEVELS[a])
            bits = env.get_chunk_size_bits(br, k)
            dl = bits / (lb * 1000.0 + 1e-6)
            if dl <= cap:
                best = a
                break
        return best

    def project_action(self, proposed: int) -> int:
        return self.max_safe_action(proposed)


class Heuristic85Projector:
    """Fixed 0.85× harmonic throughput discount (no conformal quantile)."""

    def __init__(self, env, cfg: ControlArmConfig | None = None):
        self.env = env
        self.cfg = cfg or ControlArmConfig()
        self._tp_hist: list[float] = []

    def on_episode_start(self):
        self._tp_hist = []

    def observe(self):
        env = self.env
        if env.chunk_idx == 0:
            self.on_episode_start()
            return
        tp = float(getattr(env, "last_raw_throughput", 0.0))
        if tp > 0:
            self._tp_hist.append(tp)

    def _harmonic_kbps(self) -> float:
        h = self._tp_hist[-5:]
        if not h:
            return float(max(getattr(self.env, "last_raw_throughput", 500.0), 10.0))
        return float(len(h) / np.sum(1.0 / np.maximum(h, 1e-6)))

    def lower_bound_kbps(self) -> float:
        return max(0.85 * self._harmonic_kbps(), 10.0)

    def max_safe_action(self, proposed: int | None = None) -> int:
        self.observe()
        lb = self.lower_bound_kbps()
        env = self.env
        k = int(env.chunk_idx)
        buf = float(env.buffer_level)
        cap = max(buf - self.cfg.effective_safety_margin(), 0.1)
        hi = int(proposed) if proposed is not None else len(env.BITRATE_LEVELS) - 1
        for a in range(hi, -1, -1):
            br = int(env.BITRATE_LEVELS[a])
            bits = env.get_chunk_size_bits(br, k)
            dl = bits / (lb * 1000.0 + 1e-6)
            if dl <= cap:
                return a
        return 0

    def project_action(self, proposed: int) -> int:
        return self.max_safe_action(proposed)


class SymmetricTubeMPCPolicy(TubeMPCPolicy):
    """Multi-horizon tube MPC with symmetric smoothness and max_step_down=1."""

    max_step_down: int = 1

    def __init__(self, env, cfg: ControlArmConfig | None = None, tube_cfg: TubeMPCConfig | None = None):
        cfg = cfg or ControlArmConfig()
        tube_cfg = tube_cfg or TubeMPCConfig(
            horizon=cfg.horizon,
            safety_margin=cfg.effective_safety_margin(),
            tube=ConformalTubeConfig(horizon=cfg.horizon, use_aci=True),
        )
        super().__init__(env, tube_cfg)

    def _filtered_seqs(self, h: int, a_prev: int) -> np.ndarray:
        seqs = self._seqs(h)
        ok_rows = []
        for row in seqs:
            prev = a_prev
            good = True
            for j in range(h):
                cur = int(row[j])
                if cur < prev - self.max_step_down:
                    good = False
                    break
                if cur > prev + 1:
                    good = False
                    break
                prev = cur
            if good:
                ok_rows.append(row)
        if not ok_rows:
            return seqs[:1]
        return np.asarray(ok_rows, dtype=int)

    def plan_action(self) -> int:
        env = self.env
        self._observe_throughput()
        k = int(env.chunk_idx)
        a_prev = int(getattr(env, "last_bitrate_idx", 0))
        h = min(self.cfg.horizon, int(env.max_chunks) - k)
        if h <= 0:
            return self.n - 1

        tp_tube = self.tube.horizon_lower_bounds_kbps(h)
        sizes = self._chunk_sizes(k, h)
        seqs = self._filtered_seqs(h, a_prev)
        n_seq = seqs.shape[0]
        dl = sizes[np.arange(h)[None, :], seqs] / (tp_tube[np.arange(h)] * 1000.0 + 1e-6)

        buf = np.full(n_seq, float(env.buffer_level))
        prev_u = np.full(n_seq, self.log_u[a_prev])
        total = np.zeros(n_seq)
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        margin = float(self.cfg.safety_margin)
        safe = np.ones(n_seq, dtype=bool)

        for j in range(h):
            reb = np.maximum(0.0, dl[:, j] - buf)
            buf = np.minimum(np.maximum(0.0, buf - dl[:, j]) + p, b_max)
            safe &= buf >= margin
            aj = seqs[:, j]
            lu = self.log_u[aj]
            sm = np.abs(lu - prev_u)
            total += lu - self.cfg.reb_penalty * reb - self.cfg.smooth_weight * sm
            prev_u = lu

        total[~safe] = -1e30
        return int(seqs[int(np.argmax(total)), 0])


class AsymmetricTubeMPCPolicy(TubeMPCPolicy):
    """
    Multi-horizon tube MPC with asymmetric smoothness:
    - Up-switch / hold-high: full H-step tube + upward smoothness penalty.
    - Down-switch / low buffer: no smoothness block on safety steps.
    """

    def __init__(self, env, cfg: ControlArmConfig | None = None, tube_cfg: TubeMPCConfig | None = None):
        cfg = cfg or ControlArmConfig()
        tube_cfg = tube_cfg or TubeMPCConfig(
            horizon=cfg.horizon,
            safety_margin=cfg.effective_safety_margin(),
            tube=ConformalTubeConfig(horizon=cfg.horizon, use_aci=False),
        )
        super().__init__(env, tube_cfg)
        self.arm_cfg = cfg

    def _asymmetric_smooth(self, action_idx: np.ndarray, prev_idx: int, buf: float) -> np.ndarray:
        """Per-sequence smoothness penalty at the first planned chunk."""
        env = self.env
        b_tgt = float(getattr(env, "BUFFER_TARGET", 6.0))
        lu = self.log_u[action_idx]
        u_prev = float(self.log_u[prev_idx])
        pen = np.zeros(len(action_idx), dtype=float)
        for i, aj in enumerate(action_idx):
            if aj < prev_idx or buf < b_tgt:
                pen[i] = 0.0
            elif aj >= prev_idx and aj > 0:
                pen[i] = self.arm_cfg.upward_smooth_weight * max(0.0, lu[i] - u_prev)
                pen[i] += abs(lu[i] - u_prev)
            else:
                pen[i] = abs(lu[i] - u_prev)
        return pen

    def plan_action(self) -> int:
        env = self.env
        self._observe_throughput()
        k = int(env.chunk_idx)
        a_prev = int(getattr(env, "last_bitrate_idx", 0))
        h = min(self.cfg.horizon, int(env.max_chunks) - k)
        if h <= 0:
            return self.n - 1

        tp_tube = self.tube.horizon_lower_bounds_kbps(h)
        sizes = self._chunk_sizes(k, h)
        seqs = self._seqs(h)
        n_seq = seqs.shape[0]
        dl = sizes[np.arange(h)[None, :], seqs] / (tp_tube[np.arange(h)] * 1000.0 + 1e-6)

        buf0 = float(env.buffer_level)
        buf = np.full(n_seq, buf0)
        prev_u = np.full(n_seq, self.log_u[a_prev])
        total = np.zeros(n_seq)
        p = float(env.CHUNK_DURATION)
        b_max = float(env.BUFFER_MAX)
        margin = float(self.cfg.safety_margin)
        safe = np.ones(n_seq, dtype=bool)
        b_tgt = float(getattr(env, "BUFFER_TARGET", 6.0))

        for j in range(h):
            reb = np.maximum(0.0, dl[:, j] - buf)
            buf = np.minimum(np.maximum(0.0, buf - dl[:, j]) + p, b_max)
            safe &= buf >= margin
            aj = seqs[:, j]
            lu = self.log_u[aj]
            if j == 0:
                sm = self._asymmetric_smooth(aj, a_prev, buf0)
            else:
                sm = np.zeros(n_seq, dtype=float)
                for i in range(n_seq):
                    pidx = int(seqs[i, j - 1])
                    cidx = int(aj[i])
                    if cidx < pidx or buf0 < b_tgt:
                        sm[i] = 0.0
                    elif cidx >= pidx and cidx > 0:
                        sm[i] = self.arm_cfg.upward_smooth_weight * max(0.0, lu[i] - self.log_u[pidx])
                        sm[i] += abs(lu[i] - self.log_u[pidx])
                    else:
                        sm[i] = abs(lu[i] - self.log_u[pidx])
            total += lu - self.cfg.reb_penalty * reb - self.cfg.smooth_weight * sm
            prev_u = lu

        total[~safe] = -1e30
        return int(seqs[int(np.argmax(total)), 0])


class SymmetricTubeConformalPolicy:
    """Symmetric tube MPC only (no static emergency cap)."""

    def __init__(self, env, base_policy, cfg: ControlArmConfig | None = None):
        self.env = env
        self.base = base_policy
        self.cfg = cfg or ControlArmConfig()
        eff = self.cfg.effective_safety_margin()
        tube_cfg = TubeMPCConfig(
            horizon=self.cfg.horizon,
            safety_margin=eff,
            tube=ConformalTubeConfig(horizon=self.cfg.horizon, use_aci=True),
        )
        self.tube = SymmetricTubeMPCPolicy(env, self.cfg, tube_cfg)

    def on_episode_start(self):
        self.tube.on_episode_start()

    def predict(self, obs, deterministic=True):
        return self.tube.plan_action(), None


class TubeV2ConformalPolicy:
    """Tube-v2 MPC with optional static 1-step cap (``min(tube, static)``)."""

    def __init__(
        self,
        env,
        base_policy,
        cfg: ControlArmConfig | None = None,
        *,
        mode: TubeMode = "per_step",
        use_aci: bool = False,
        capped: bool = True,
        split_objective: bool = False,
        tube_kind: str = "v2",
    ):
        self.env = env
        self.base = base_policy
        self.cfg = cfg or ControlArmConfig()
        self.capped = capped
        self.static = StaticConformalProjector(env, self.cfg, adapt=True)
        eff = self.cfg.effective_safety_margin()
        tube_cfg = ConformalTubeV2Config(
            mode=mode,
            alpha=self.cfg.alpha,
            horizon=self.cfg.horizon,
            use_aci=use_aci,
            aci_gamma=float(self.cfg.aci_gamma),
            cold_start_cap=self.cfg.cold_start_cap,
            conformal=ConformalConfig(alpha=self.cfg.alpha, window=200, k_predict=5),
        )
        self.tube_mpc = TubeV2MPCPolicy(
            env,
            TubeV2MPCConfig(
                horizon=self.cfg.horizon,
                safety_margin=eff,
                tube=tube_cfg,
                pruning="monotone",
                split_objective=split_objective,
                tube_kind=tube_kind,
            ),
        )

    def on_episode_start(self) -> None:
        self.static.on_episode_start()
        self.tube_mpc.on_episode_start()

    def predict(self, obs, deterministic=True):
        a_base, _ = self.base.predict(obs, deterministic=deterministic)
        a_tube = int(self.tube_mpc.plan_action())
        if not self.capped:
            return a_tube, None
        a_static = int(self.static.max_safe_action(int(a_base)))
        return int(min(a_tube, a_static)), None


class TubeConformalPolicy:
    """Tube MPC + planning, capped by static 1-step safe rung."""

    def __init__(
        self,
        env,
        base_policy,
        cfg: ControlArmConfig | None = None,
        *,
        tube_planner: type = AsymmetricTubeMPCPolicy,
        use_aci: bool = True,
        static_adapt: bool = True,
    ):
        self.env = env
        self.base = base_policy
        self.cfg = cfg or ControlArmConfig()
        eff = self.cfg.effective_safety_margin()
        self.static = StaticConformalProjector(env, self.cfg, adapt=static_adapt)
        tube_cfg = TubeMPCConfig(
            horizon=self.cfg.horizon,
            safety_margin=eff,
            tube=ConformalTubeConfig(
                horizon=self.cfg.horizon,
                use_aci=use_aci,
                static_conformal=ConformalConfig(alpha=self.cfg.alpha, window=200, k_predict=5),
            ),
        )
        self.tube = tube_planner(env, self.cfg, tube_cfg)

    def on_episode_start(self):
        self.static.on_episode_start()
        self.tube.on_episode_start()

    def predict(self, obs, deterministic=True):
        a_base, _ = self.base.predict(obs, deterministic=deterministic)
        a_static = self.static.max_safe_action(int(a_base))
        a_tube = self.tube.plan_action()
        return int(min(a_tube, a_static)), None


def configure_env_for_arm(env, arm: EvalArm) -> None:
    env.enable_early_abort = False


def _wrap_projector(base_policy, proj):
    class _Wrapped:
        def on_episode_start(self):
            proj.on_episode_start()

        def predict(self, obs, deterministic=True):
            a, _ = base_policy.predict(obs, deterministic=deterministic)
            return proj.project_action(int(a)), None

    return _Wrapped()


def _warm_start_tube_v2(policy: TubeV2ConformalPolicy, trace_set_key: str) -> None:
    from configs.trace_sets_v2 import calibration_dirs_for_eval
    from src.training.tube_v2_calib import replay_throughputs_on_tube

    variant = "native" if trace_set_key == "5g_harsh" else "default"
    dirs = calibration_dirs_for_eval(trace_set_key, variant=variant)
    replay_throughputs_on_tube(policy.tube_mpc.tube, dirs)


def make_controlled_policy(
    env,
    arm: EvalArm,
    base_policy,
    cfg: ControlArmConfig | None = None,
    *,
    trace_set_key: str = "",
):
    cfg = cfg or ControlArmConfig()
    configure_env_for_arm(env, arm)

    if arm == "raw":
        return base_policy

    if arm == "heuristic_85":
        return _wrap_projector(base_policy, Heuristic85Projector(env, cfg))

    if arm == "static_conformal":
        return _wrap_projector(base_policy, StaticConformalProjector(env, cfg))

    if arm == "symmetric_tube":
        return SymmetricTubeConformalPolicy(env, base_policy, cfg)

    if arm == "fixed_alpha_tube":
        return TubeConformalPolicy(
            env,
            base_policy,
            cfg,
            tube_planner=AsymmetricTubeMPCPolicy,
            use_aci=False,
            static_adapt=False,
        )

    if arm == "tube_conformal":
        return TubeConformalPolicy(env, base_policy, cfg, use_aci=True)

    v3 = parse_tube_v3_arm(str(arm))
    if v3 is not None:
        from dataclasses import replace as _dc_replace

        cfg3 = _dc_replace(cfg, alpha=v3["alpha"]) if v3["alpha"] is not None else cfg
        pol = TubeV2ConformalPolicy(
            env,
            base_policy,
            cfg3,
            mode="per_step",
            use_aci=False,
            capped=True,
            split_objective=(v3["kind"] == "split"),
            tube_kind=("maxscore" if v3["kind"] == "maxscore" else "v2"),
        )
        if trace_set_key:
            _warm_start_tube_v2(pol, trace_set_key)
        return pol

    parsed = parse_tube_v2_arm(str(arm))
    if parsed is not None:
        mode, use_aci, capped = parsed
        pol = TubeV2ConformalPolicy(
            env,
            base_policy,
            cfg,
            mode=mode,
            use_aci=use_aci,
            capped=capped,
        )
        if trace_set_key:
            _warm_start_tube_v2(pol, trace_set_key)
        return pol

    if arm == "robust_mpc_logu":
        return RobustMPCLogUtilityPolicy(
            env,
            horizon=cfg.horizon,
            safety_margin=cfg.effective_safety_margin(),
        )

    if arm in ("mpc_fixed_quantile", "mpc_fixed_quantile_native", "mpc_fixed_quantile_harsh"):
        from configs.trace_sets_v2 import calibration_dirs_for_eval

        ts = trace_set_key or "5g_markov"
        if arm in ("mpc_fixed_quantile_native", "mpc_fixed_quantile_harsh"):
            calib_dirs = calibration_dirs_for_eval(ts, variant="native")
            variant = "harsh_native" if ts == "5g_harsh" else "calib"
        else:
            variant = "markov_transfer" if ts == "5g_harsh" else "calib"
            calib_dirs = calibration_dirs_for_eval(ts, variant=variant)
        q = load_or_fit_quantile(
            calib_dirs,
            trace_set=ts,
            variant=variant,
            alpha=cfg.alpha,
            k_predict=5,
        )
        return FixedQuantileMPCPolicy(
            env,
            q=q,
            horizon=cfg.horizon,
            safety_margin=cfg.effective_safety_margin(),
        )

    if str(arm).startswith("mpc_fixed_rho_"):
        rho = float(str(arm).split("_")[-1])
        return FixedRhoMPCPolicy(
            env,
            rho=rho,
            horizon=cfg.horizon,
            safety_margin=cfg.effective_safety_margin(),
        )

    raise ValueError(f"unknown arm: {arm}")


def arm_display_name(arm: HeadlineArm) -> str:
    return arm
