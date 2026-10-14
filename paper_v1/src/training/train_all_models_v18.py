"""
Unified training script (V18, 5G low-latency operating point).

Identical to train_all_models_v14.py except:
  * uses abr_multi_env_v18 (5G low-latency, moderate buffer), and
  * saves under models/master_v18_5g/.

Policies are trained WITHOUT the certified perceptual shield; the shield is a
model-agnostic RUNTIME wrapper applied at evaluation (see
src/evaluation/eval_certified_shield_v18.py). This keeps the central claim clean:
the shield improves ANY policy at run time, with a formal guarantee.

Run (on the 5G traces produced by the runbook):
  cd new
  python src/training/train_all_models_v18.py --models proposed pensieve --seeds 0 --num-envs 8
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import subprocess

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from configs.videos import HELD_OUT_VIDEOS, MAX_CHUNKS, TRAIN_VIDEOS
from src.environment.abr_multi_env_v18 import ABREnv
from src.training.certified_perceptual_shield import (
    CertifiedPerceptualShieldWrapper,
    CPShieldConfig,
    ConformalConfig,
)
from src.training.constrained_abr_v14 import (
    ConstraintDiagnosticsLogger,
    LagrangianRewardWrapperV14,
)
from src.training.safety_shield_v14 import SafetyShieldWrapper, ShieldConfig
from src.training.shield_aware_wrappers_v12 import (
    HysteresisActionWrapper,
    HysteresisConfig,
    ShieldAwarePenaltyWrapper,
)

PATHS = get_paths()

REBUF_TARGET = 0.05
SMOOTH_TARGET = 3.5
MODEL_TAG = "master_v18_5g"


def make_cps_cfg() -> CPShieldConfig:
    """Certified perceptual shield used FOR CO-TRAINING (item 5): banking +
    risk-aware perceptual budget + dip forecasting. Kept identical to the best
    evaluation-time shield so the co-designed policy sees the same dynamics it
    will be deployed under."""
    return CPShieldConfig(
        enabled=True, enable_banking=True, epsilon_vmaf=1.0,
        enable_conformal=True,
        conformal=ConformalConfig(alpha=0.10, window=200, k_predict=5),
        safety_margin=0.5, min_buffer=0.3,
        predictive=True, lookahead=6, epsilon_risk=4.0, risk_buffer=8.0,
        forecast_dips=True, horizon_quantile=0.2,
    )


def linear_schedule(initial_value: float, final_value: float = 1e-5):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func


class Config:
    TRAIN_VIDEOS = list(TRAIN_VIDEOS)
    TEST_VIDEOS = list(HELD_OUT_VIDEOS)
    MAX_CHUNKS = MAX_CHUNKS
    NUM_ENVS = 8

    LEARNING_RATE = linear_schedule(3e-4, 1e-5)
    N_STEPS = 4096
    BATCH_SIZE = 512
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    DEVICE = "cpu"

    SAVE_FREQ = 50_000
    EVAL_FREQ = 20_000


MODEL_SPECS: Dict[str, Dict] = {
    "proposed": {
        "folder": "proposed_v14", "use_lyapunov": True, "use_future": True,
        "blind_features": False, "use_lagrangian": True, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.04, "timesteps": 2_000_000,
    },
    # Shield-aware CO-DESIGN (item 5): identical to `proposed` but the certified
    # perceptual shield is active INSIDE the training loop, so the policy learns to
    # exploit the banked buffer (chase fidelity without causing stalls) instead of
    # collapsing to a conservative low-rung policy.
    "proposed_cps": {
        "folder": "proposed_cps_v18", "use_lyapunov": True, "use_future": True,
        "blind_features": False, "use_lagrangian": True, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.04, "timesteps": 2_000_000,
        "cps_train": True,
    },
    "pensieve": {
        "folder": "pensieve_v14", "use_lyapunov": False, "use_future": False,
        "blind_features": True, "use_lagrangian": False, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.03, "timesteps": 1_000_000,
    },
    "ablation_base": {
        "folder": "ablation_base_v14", "use_lyapunov": False, "use_future": False,
        "blind_features": False, "use_lagrangian": False, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.03, "timesteps": 1_000_000,
    },
}


class ContentBlindWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        modified = obs.copy()
        modified[15:] = 0.0
        return modified


def make_env(rank: int, model_key: str, base_seed: int = 0, is_eval: bool = False,
             train_dir: str | None = None, test_dir: str | None = None):
    spec = MODEL_SPECS[model_key]
    # Resolve here so the (picklable) strings are captured in the closure -- on
    # Windows 'spawn', the subprocess re-imports this module with default globals,
    # so trace-dir overrides MUST travel inside the closure, not via globals.
    train_dir = train_dir if train_dir is not None else str(PATHS["train_traces"])
    test_dir = test_dir if test_dir is not None else str(PATHS["test_traces"])

    def _init():
        videos = Config.TEST_VIDEOS if is_eval else Config.TRAIN_VIDEOS
        traces = test_dir if is_eval else train_dir
        env = ABREnv(
            video_names=videos,
            trace_dir=str(traces),
            vmaf_dir=str(PATHS["vmaf_scores"]),
            siti_dir=str(PATHS["content_features"]),
            max_chunks=Config.MAX_CHUNKS,
            random_seed=(base_seed + 1000 if is_eval else base_seed) + rank,
            use_lyapunov=spec["use_lyapunov"],
            use_future=spec["use_future"],
        )

        if spec.get("use_lagrangian") and not is_eval:
            env = LagrangianRewardWrapperV14(
                env, rebuf_target=REBUF_TARGET, smooth_target=SMOOTH_TARGET
            )

        if spec["blind_features"]:
            env = ContentBlindWrapper(env)

        # Shield-aware co-design: apply the certified perceptual shield inside the
        # loop so the policy is trained (and model-selected) under the deployment
        # dynamics. The policy still proposes actions; the shield projects/banks and
        # the resulting reward flows back -> the policy learns a shield-exploiting
        # strategy. Applied to BOTH train and eval envs for consistent selection.
        if spec.get("cps_train"):
            env = CertifiedPerceptualShieldWrapper(env, make_cps_cfg())

        return Monitor(env, info_keywords=("avg_quality", "total_rebuffer"))

    return _init


def train_one_model(model_key: str, seed: int, timesteps_scale: float, num_envs: int,
                    train_dir: str | None = None, test_dir: str | None = None,
                    model_tag: str = MODEL_TAG):
    spec = MODEL_SPECS[model_key]
    seed_tag = f"seed_{seed}"
    model_root = PATHS["models"] / model_tag / spec["folder"] / seed_tag
    log_root = PATHS["logs"] / model_tag / spec["folder"] / seed_tag
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    total_timesteps = max(1, int(spec["timesteps"] * timesteps_scale))

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "=" * 72)
    print(f"Training {model_key.upper()} (v18 5G) seed={seed} | "
          f"steps={total_timesteps:,} | envs={num_envs}")
    print(f"Model dir: {model_root}")
    print("=" * 72)

    train_env = SubprocVecEnv(
        [make_env(i, model_key, base_seed=seed, is_eval=False,
                  train_dir=train_dir, test_dir=test_dir) for i in range(num_envs)]
    )
    eval_env = SubprocVecEnv([make_env(0, model_key, base_seed=seed, is_eval=True,
                                       train_dir=train_dir, test_dir=test_dir)])

    policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": torch.nn.Tanh}

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=Config.LEARNING_RATE, n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE, n_epochs=Config.N_EPOCHS,
        gamma=Config.GAMMA, gae_lambda=Config.GAE_LAMBDA, clip_range=Config.CLIP_RANGE,
        ent_coef=spec["ent_coef"], vf_coef=Config.VF_COEF, max_grad_norm=Config.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs, verbose=1, seed=seed, device=Config.DEVICE,
        tensorboard_log=str(log_root),
    )

    cb_list = [
        CheckpointCallback(save_freq=max(1, Config.SAVE_FREQ // num_envs),
                           save_path=str(model_root / "checkpoints"), name_prefix=spec["folder"]),
        EvalCallback(eval_env, best_model_save_path=str(model_root / "best_model"),
                     log_path=str(log_root / "eval"), eval_freq=max(1, Config.EVAL_FREQ // num_envs),
                     n_eval_episodes=10, deterministic=True),
    ]
    if spec.get("use_lagrangian"):
        cb_list.append(ConstraintDiagnosticsLogger(
            log_dir=str(log_root), log_freq=5000,
            rebuf_target=REBUF_TARGET, smooth_target=SMOOTH_TARGET))

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(cb_list), progress_bar=True)
        model.save(model_root / "final_model")
        print(f"[DONE] {model_key} v18 seed={seed}.")
    except KeyboardInterrupt:
        model.save(model_root / "interrupted_model")
        print(f"[INTERRUPTED] {model_key} v18 seed={seed} checkpoint saved.")
    finally:
        train_env.close()
        eval_env.close()


def parse_model_list(value: str) -> List[str]:
    keys = [v.strip().lower() for v in value.split(",") if v.strip()]
    invalid = [k for k in keys if k not in MODEL_SPECS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown model keys: {invalid}. Valid: {list(MODEL_SPECS)}")
    return keys


def parse_seed_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def main():
    parser = argparse.ArgumentParser(description="Train ABR models at the 5G low-latency operating point (v18).")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--models", type=parse_model_list, default=["proposed", "pensieve"])
    parser.add_argument("--seeds", type=parse_seed_list, default=[0])
    parser.add_argument("--timesteps-scale", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=Config.NUM_ENVS)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--trace-dir", type=str, default=None,
                        help="override TRAIN trace directory (default: source-disjoint broadband).")
    parser.add_argument("--test-trace-dir", type=str, default=None,
                        help="override in-training EVAL trace directory.")
    parser.add_argument("--tag", type=str, default=MODEL_TAG,
                        help="output tag under results/models/<tag>/ (isolate regimes).")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    order = list(MODEL_SPECS) if args.all else args.models

    # Default to the provenance-correct broadband split; fall back to legacy if the
    # V18 datasets have not been prepared yet.
    train_dir = args.trace_dir
    test_dir = args.test_trace_dir
    if train_dir is None:
        cand = PATHS["train_traces_v18"]
        train_dir = str(cand) if cand.exists() else str(PATHS["train_traces"])
    if test_dir is None:
        cand = PATHS["test_traces_v18"]
        test_dir = str(cand) if cand.exists() else str(PATHS["test_traces"])
    print(f"[v18-train] train_dir={train_dir}\n[v18-train] test_dir={test_dir}\n[v18-train] tag={args.tag}")

    for k in order:
        for s in args.seeds:
            train_one_model(k, seed=s, timesteps_scale=args.timesteps_scale,
                            num_envs=args.num_envs, train_dir=train_dir,
                            test_dir=test_dir, model_tag=args.tag)
    print("\nAll requested v18 trainings finished.")


if __name__ == "__main__":
    main()
