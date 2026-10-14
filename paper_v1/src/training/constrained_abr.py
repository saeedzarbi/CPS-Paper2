"""
Constrained ABR via Lagrangian Primal-Dual Optimization.

Formulates the ABR bitrate selection problem as a Constrained MDP (CMDP):

    maximize   E[ Σ_t  VMAF(a_t) ]
    subject to
        (C1)  E[ total_rebuffer / T ]          ≤  δ_rebuf
        (C2)  E[ Σ |VMAF_t − VMAF_{t-1}| / K ] ≤  δ_smooth

Standard ABR-RL approaches (Pensieve, etc.) convert these constraints into a
fixed-weight penalty:  R = VMAF − α·rebuf − β·smooth.  The weights α, β must
be hand-tuned and a single set of weights cannot be optimal across all
network / content conditions.

We instead solve the Lagrangian relaxation:

    L(π, λ) = E[ Σ  VMAF − λ_r·rebuf − λ_s·smooth ]

    Primal step :  π* = argmax_π  L(π, λ)        (PPO update)
    Dual   step :  λ  ← Π_Λ [ λ + η · (g(π) − δ) ]   (projected gradient ascent)

where g(π) is the measured constraint value, δ is the target budget, η is
the dual learning rate, and Π_Λ projects onto the feasible set [λ_min, λ_max].

Convergence follows from standard CMDP / Lagrangian duality theory
(Altman 1999, Paternain et al. 2019).

Tuned targets (v5.2): rebuf_target=0.04, smooth_target=4.0,
lambda_rebuf_init=5.0, lyapunov_weight=0.7.  v5.1 over-relaxed
(rebuf_target=0.05, lambda_min=1.0, lyap_w=0.5) causing policy collapse
to 1200kbps.  v5.0 was too conservative (rebuf=4.84s vs Genie 9.07s).
v5.2 uses intermediate values to balance bitrate ambition with stability.

Components
----------
LagrangianRewardWrapper : gym.Wrapper
    Recomputes the reward using adaptive λ values.  Applied per-env
    (each SubprocVecEnv worker maintains its own dual variables, which
    converge to similar values via distributed dual ascent).

DualVariableLogger : BaseCallback
    Logs the evolution of λ_rebuf and λ_smooth during training to CSV.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
#  Lagrangian Reward Wrapper
# ---------------------------------------------------------------------------

class LagrangianRewardWrapper(gym.Wrapper):
    """Recomputes the ABR reward using adaptive Lagrangian dual variables.

    At each episode boundary the wrapper performs a projected dual gradient
    ascent step, increasing λ when the constraint is violated and decreasing
    it otherwise.  The result is that the policy sees a reward landscape
    whose penalty weights continuously adapt to satisfy the constraints.
    """

    def __init__(
        self,
        env: gym.Env,
        rebuf_target: float = 0.04,
        smooth_target: float = 4.0,
        dual_lr_rebuf: float = 0.007,
        dual_lr_smooth: float = 0.003,
        lambda_rebuf_init: float = 4.3,
        lambda_smooth_init: float = 0.7,
        lambda_rebuf_range: tuple = (0.8, 12.0),
        lambda_smooth_range: tuple = (0.2, 2.5),
        buffer_dev_weight: float = 0.05,
        lyapunov_weight: float = 0.5,
        warmup_episodes: int = 50,
    ):
        super().__init__(env)
        self.rebuf_target = rebuf_target
        self.smooth_target = smooth_target
        self.dual_lr_rebuf = dual_lr_rebuf
        self.dual_lr_smooth = dual_lr_smooth
        self.lambda_rebuf = lambda_rebuf_init
        self.lambda_smooth = lambda_smooth_init
        self.lambda_rebuf_range = lambda_rebuf_range
        self.lambda_smooth_range = lambda_smooth_range
        self.buffer_dev_weight = buffer_dev_weight
        self.lyapunov_weight = lyapunov_weight
        self.warmup_episodes = warmup_episodes

        self._ep_rebuf = 0.0
        self._ep_smooth = 0.0
        self._ep_steps = 0
        self._episode_count = 0

        self._recent_rebuf: list[float] = []
        self._recent_smooth: list[float] = []

    # -- gym API -----------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_rebuf = 0.0
        self._ep_smooth = 0.0
        self._ep_steps = 0
        return obs, info

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        vmaf = info["vmaf"]
        rebuf = info["rebuffer"]
        smooth = info["smooth_penalty"]
        buf_dev = info.get("buffer_dev", 0.0)
        lyap = info.get("lyapunov_penalty", 0.0)

        reward = (
            vmaf
            - self.lambda_rebuf * rebuf
            - self.lambda_smooth * smooth
            - self.buffer_dev_weight * buf_dev
            - self.lyapunov_weight * lyap
        ) / 100.0

        self._ep_rebuf += rebuf
        self._ep_smooth += smooth
        self._ep_steps += 1

        if terminated or truncated:
            self._dual_update()

        info["lambda_rebuf"] = self.lambda_rebuf
        info["lambda_smooth"] = self.lambda_smooth

        return obs, reward, terminated, truncated, info

    # -- dual update -------------------------------------------------------

    def _dual_update(self):
        if self._ep_steps == 0:
            return

        ep_duration = self._ep_steps * 4.0
        rebuf_ratio = self._ep_rebuf / ep_duration
        avg_smooth = self._ep_smooth / self._ep_steps

        self._recent_rebuf.append(rebuf_ratio)
        self._recent_smooth.append(avg_smooth)
        self._episode_count += 1

        if self._episode_count < self.warmup_episodes:
            return

        window = min(100, len(self._recent_rebuf))
        mean_rebuf = float(np.mean(self._recent_rebuf[-window:]))
        mean_smooth = float(np.mean(self._recent_smooth[-window:]))

        rebuf_gap = mean_rebuf - self.rebuf_target
        smooth_gap = mean_smooth - self.smooth_target

        # Asymmetric update: when over target (rebuf_gap > 0), increase lambda faster
        # so we react strongly to violations; when under target, decrease slowly
        # to avoid lambda drifting down and making the policy too aggressive.
        if rebuf_gap < 0:
            self.lambda_rebuf += 1.0 * self.dual_lr_rebuf * rebuf_gap
        else:
            self.lambda_rebuf += 2.0 * self.dual_lr_rebuf * rebuf_gap

        if smooth_gap < 0:
            self.lambda_smooth += 1.5 * self.dual_lr_smooth * smooth_gap
        else:
            self.lambda_smooth += self.dual_lr_smooth * smooth_gap

        self.lambda_rebuf = float(np.clip(
            self.lambda_rebuf, *self.lambda_rebuf_range
        ))
        self.lambda_smooth = float(np.clip(
            self.lambda_smooth, *self.lambda_smooth_range
        ))

        if len(self._recent_rebuf) > 500:
            self._recent_rebuf = self._recent_rebuf[-200:]
            self._recent_smooth = self._recent_smooth[-200:]


# ---------------------------------------------------------------------------
#  Dual-Variable Logger (SB3 Callback)
# ---------------------------------------------------------------------------

class DualVariableLogger(BaseCallback):
    """Periodically logs Lagrangian multiplier values to a CSV file."""

    def __init__(self, log_dir: str, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self._fh = None

    def _on_training_start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(
            self.log_dir / "dual_variables.csv", "w", encoding="utf-8"
        )
        self._fh.write(
            "step,lambda_rebuf,lambda_smooth,"
            "recent_rebuf_ratio,recent_avg_smooth\n"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        for info in self.locals.get("infos", []):
            lr = info.get("lambda_rebuf")
            ls = info.get("lambda_smooth")
            if lr is not None and self._fh is not None:
                rebuf_r = info.get("total_rebuffer", 0.0) / max(
                    info.get("chunk_idx", 1) * 4.0, 1.0
                )
                avg_sm = info.get("total_smoothness", 0.0) / max(
                    info.get("chunk_idx", 1), 1
                )
                self._fh.write(
                    f"{self.num_timesteps},{lr:.4f},{ls:.4f},"
                    f"{rebuf_r:.4f},{avg_sm:.2f}\n"
                )
                self._fh.flush()
                break
        return True

    def _on_training_end(self):
        if self._fh:
            self._fh.close()
