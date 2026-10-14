#!/usr/bin/env bash
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export PYTHONIOENCODING=utf-8

python3 - <<'PY'
from configs.paths import get_paths
from configs.videos import EVAL_VIDEOS, MAX_CHUNKS
from src.environment.paper2_abr_env import Paper2StreamingEnv
from src.training.tube_mpc_policy import TubeMPCPolicy, TubeMPCConfig
from src.training.conformal_tube import ConformalTubeConfig

P = get_paths()
trace = P["test_traces_5g_v18"]
if not trace.exists():
    raise SystemExit(f"Missing traces: {trace}")

env = Paper2StreamingEnv(
    video_names=list(EVAL_VIDEOS),
    trace_dir=str(trace),
    max_chunks=MAX_CHUNKS,
    random_seed=0,
    use_lyapunov=True,
    use_future=True,
)
cfg = TubeMPCConfig(tube=ConformalTubeConfig(horizon=5, use_aci=True))
policy = TubeMPCPolicy(env, cfg)
obs, _ = env.reset(seed=1000)
reb = 0.0
for step in range(20):
    a, _ = policy.predict(obs)
    obs, r, term, trunc, info = env.step(a)
    reb += float(info.get("rebuffer", 0.0))
    if term or trunc:
        break
print("Tube-MPC smoke OK: steps=", step + 1, "rebuffer=", round(reb, 3), "alpha_t=", round(policy.tube.alpha_t, 4))
print("action_mask:", policy.action_mask())
PY
