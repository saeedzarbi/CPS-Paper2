"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V14, reviewer-response).

This wrapper adapts the base Lagrangian machinery in ``constrained_abr.py`` to
the corrected reward scale introduced in ``abr_multi_env_v14.py`` and adds the
CMDP diagnostics the review asked for (P1.3).

------------------------------------------------------------------------------
Why the dual-variable range changed (P1.3 in the review)
------------------------------------------------------------------------------
On the OLD reward scale the rebuffering weight was ``lambda_r in [4.3, 40]``.
Because fidelity was on the VMAF 0-100 scale, that ceiling of 40 was far too
low to make stalls costly relative to fidelity, so ``lambda_r`` saturated at 40
and the constraint (rebuffering ratio <= 0.05) was violated by ~3x (achieved
~0.147). The review correctly flagged that a paper titled "Constrained Deep RL"
must show the constraint actually binding.

On the corrected scale (see ``abr_multi_env_v14.py``) one second of stall should
cost ~100 fidelity points, so ``lambda_r`` must be able to reach the same order
of magnitude. We therefore widen the range to ``[10, 400]`` and raise the dual
learning rate accordingly. With this range the dual ascent either (a) drives the
rebuffering ratio down to the target, or (b) saturates the ceiling, which is
itself an informative, reportable diagnostic rather than a hidden failure.

Diagnostics
-----------
``ConstraintDiagnosticsLogger`` writes, in addition to the multiplier trace,
the achieved-vs-target rebuffering ratio and smoothness at every log step so the
paper can include the "achieved vs. target constraint" table the review
requested. It reuses the same CSV as ``DualVariableLogger`` (superset of
columns), so downstream scripts keep working.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from .constrained_abr import DualVariableLogger, LagrangianRewardWrapper as _BaseLagrangianRewardWrapper


class LagrangianRewardWrapperV14(_BaseLagrangianRewardWrapper):
    """V14-tuned Lagrangian wrapper for the corrected (VMAF-scaled) objective.

    Key changes vs. V9/V12:
    - ``lambda_rebuf_range`` widened from (4.3, 40) to (10, 400) so the stall
      penalty can reach the same order of magnitude as the corrected fidelity
      scale and the rebuffering constraint can bind.
    - ``lambda_rebuf_init`` raised to 60 (mid-range) for a warm start.
    - ``dual_lr_rebuf`` raised to 40 because the constraint gap is measured in
      ratio units (~0.01-0.15) that must move a multiplier of order ~100.
    - Smoothness controls are essentially unchanged: smoothness is already on the
      VMAF-point scale, so its multiplier does not need rescaling.
    """

    def __init__(
        self,
        env: gym.Env,
        rebuf_target: float = 0.05,
        smooth_target: float = 3.5,
        dual_lr_rebuf: float = 40.0,
        dual_lr_smooth: float = 0.05,
        lambda_rebuf_init: float = 60.0,
        lambda_smooth_init: float = 0.7,
        lambda_rebuf_range: tuple = (10.0, 400.0),
        lambda_smooth_range: tuple = (0.3, 5.0),
        buffer_dev_weight: float = 0.03,
        lyapunov_weight: float = 0.4,
        warmup_episodes: int = 60,
    ):
        super().__init__(
            env=env,
            rebuf_target=rebuf_target,
            smooth_target=smooth_target,
            dual_lr_rebuf=dual_lr_rebuf,
            dual_lr_smooth=dual_lr_smooth,
            lambda_rebuf_init=lambda_rebuf_init,
            lambda_smooth_init=lambda_smooth_init,
            lambda_rebuf_range=lambda_rebuf_range,
            lambda_smooth_range=lambda_smooth_range,
            buffer_dev_weight=buffer_dev_weight,
            lyapunov_weight=lyapunov_weight,
            warmup_episodes=warmup_episodes,
        )


class ConstraintDiagnosticsLogger(DualVariableLogger):
    """Logs multipliers *and* achieved-vs-target constraint values.

    Writes ``dual_variables.csv`` with columns:
        step, lambda_rebuf, lambda_smooth, recent_rebuf_ratio, recent_avg_smooth,
        rebuf_target, smooth_target, rebuf_gap, smooth_gap

    The extra columns let the paper report whether the CMDP constraint was
    actually satisfied (achieved <= target) without re-running training.
    """

    def __init__(self, log_dir: str, log_freq: int = 5000,
                 rebuf_target: float = 0.05, smooth_target: float = 3.5,
                 verbose: int = 0):
        super().__init__(log_dir=log_dir, log_freq=log_freq, verbose=verbose)
        self.rebuf_target = float(rebuf_target)
        self.smooth_target = float(smooth_target)

    def _on_training_start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_dir / "dual_variables.csv", "w", encoding="utf-8")
        self._fh.write(
            "step,lambda_rebuf,lambda_smooth,recent_rebuf_ratio,recent_avg_smooth,"
            "rebuf_target,smooth_target,rebuf_gap,smooth_gap\n"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        for info in self.locals.get("infos", []):
            lr = info.get("lambda_rebuf")
            ls = info.get("lambda_smooth")
            if lr is None or self._fh is None:
                continue
            rebuf_r = info.get("total_rebuffer", 0.0) / max(info.get("chunk_idx", 1) * 4.0, 1.0)
            avg_sm = info.get("total_smoothness", 0.0) / max(info.get("chunk_idx", 1), 1)
            self._fh.write(
                f"{self.num_timesteps},{lr:.4f},{ls:.4f},{rebuf_r:.4f},{avg_sm:.2f},"
                f"{self.rebuf_target:.4f},{self.smooth_target:.2f},"
                f"{rebuf_r - self.rebuf_target:.4f},{avg_sm - self.smooth_target:.2f}\n"
            )
            self._fh.flush()
            break
        return True


def summarize_constraint_satisfaction(log_dir: str, tail_frac: float = 0.2) -> dict:
    """Read ``dual_variables.csv`` and summarise the final constraint state.

    Uses the last ``tail_frac`` of logged rows (post-convergence) to report the
    achieved rebuffering ratio / smoothness against target and whether each
    constraint is satisfied. Returns a dict; also usable from a notebook.
    """
    path = Path(log_dir) / "dual_variables.csv"
    if not path.exists():
        return {"error": f"missing {path}"}
    import pandas as pd

    df = pd.read_csv(path)
    if df.empty:
        return {"error": "empty log"}
    n_tail = max(1, int(len(df) * tail_frac))
    tail = df.tail(n_tail)
    rebuf_target = float(tail["rebuf_target"].iloc[-1]) if "rebuf_target" in tail else 0.05
    smooth_target = float(tail["smooth_target"].iloc[-1]) if "smooth_target" in tail else 3.5
    achieved_rebuf = float(tail["recent_rebuf_ratio"].mean())
    achieved_smooth = float(tail["recent_avg_smooth"].mean())
    return {
        "achieved_rebuf_ratio": achieved_rebuf,
        "rebuf_target": rebuf_target,
        "rebuf_satisfied": bool(achieved_rebuf <= rebuf_target),
        "achieved_smoothness": achieved_smooth,
        "smooth_target": smooth_target,
        "smooth_satisfied": bool(achieved_smooth <= smooth_target),
        "final_lambda_rebuf": float(tail["lambda_rebuf"].iloc[-1]),
        "final_lambda_smooth": float(tail["lambda_smooth"].iloc[-1]),
        "lambda_rebuf_saturated": bool(np.isclose(tail["lambda_rebuf"].iloc[-1], 400.0, atol=1.0)),
    }


__all__ = [
    "LagrangianRewardWrapperV14",
    "ConstraintDiagnosticsLogger",
    "DualVariableLogger",
    "summarize_constraint_satisfaction",
]
