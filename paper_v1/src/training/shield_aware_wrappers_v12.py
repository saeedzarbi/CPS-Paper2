"""
Shield-aware training & hysteresis wrappers (V12).

Keep these effects *mild* and stable by default to avoid tuning loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


class ShieldAwarePenaltyWrapper(gym.Wrapper):
    """
    Reward shaping to improve QoE while respecting safety interventions.

    Penalizes:
    - Any shield intervention (binary)
    - Action deviation between agent action and applied (shielded/hysteresis) action
    """

    def __init__(
        self,
        env: gym.Env,
        beta_intervene: float = 0.08,
        gamma_deviation: float = 0.03,
    ):
        super().__init__(env)
        self.beta = float(beta_intervene)
        self.gamma = float(gamma_deviation)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        intervened = int(info.get("shield_intervened", 0))
        applied = int(info.get("shielded_action", action))
        agent_a = int(action)

        deviation = abs(applied - agent_a)
        shaped = float(reward) - self.beta * float(intervened) - self.gamma * float(deviation)

        info["reward_raw"] = float(reward)
        info["reward_shaped"] = float(shaped)
        info["shield_deviation"] = int(deviation)
        return obs, shaped, terminated, truncated, info


@dataclass(frozen=True)
class HysteresisConfig:
    max_step: int = 1
    min_buf_for_upswitch: float = 1.5


class HysteresisActionWrapper(gym.Wrapper):
    """
    Action post-processing that reduces switching and avoids risky up-switches.
    """

    def __init__(self, env: gym.Env, cfg: HysteresisConfig):
        super().__init__(env)
        self.cfg = cfg
        self._last_action = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_action = 0
        return obs, info

    def step(self, action):
        a = int(action)
        last = int(self._last_action)

        # Clamp per-step action changes
        if a > last + self.cfg.max_step:
            a = last + self.cfg.max_step
        elif a < last - self.cfg.max_step:
            a = last - self.cfg.max_step

        # Block up-switch on low buffer (safety/QoE stability)
        buf = float(getattr(self.env, "buffer_level", 0.0))
        if a > last and buf < self.cfg.min_buf_for_upswitch:
            a = last

        self._last_action = a
        obs, reward, terminated, truncated, info = self.env.step(a)
        info = dict(info)
        info["applied_action"] = int(a)
        return obs, reward, terminated, truncated, info


__all__ = ["ShieldAwarePenaltyWrapper", "HysteresisActionWrapper", "HysteresisConfig"]

