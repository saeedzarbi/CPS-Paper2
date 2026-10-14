#!/usr/bin/env bash
# OACS (Paper 2) install + shield smoke test — run on Linux server or WSL.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export PYTHONIOENCODING=utf-8

echo "== ${ROOT} =="
python3 configs/paths.py
python3 configs/videos.py
python3 -c "
from configs.paper2_release import release_banner
print(release_banner())
"

python3 -c "
from src.environment.paper2_abr_env import Paper2StreamingEnv
from src.training.aci_perceptual_shield import ACIPerceptualShieldWrapper
env = ACIPerceptualShieldWrapper(Paper2StreamingEnv())
obs, _ = env.reset(seed=1000)
print('Mask:', env.action_masks())
obs, r, d, t, info = env.step(5)
print('Step OK, alpha_t:', info['alpha_t'], 'tp_lb:', info['tp_lb_kbps'])
"

echo "OACS smoke test passed."
