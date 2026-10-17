#!/usr/bin/env bash
set -eu
cd /root/CPS-Paper2
source .venv-tube-v2/bin/activate
export PYTHONPATH=/root/CPS-Paper2
export OMP_NUM_THREADS=1
LOG=results/tube_v3/adaptive_test_envchunks_run.log
mkdir -p results/tube_v3
{
echo "=== TEST START $(date -Is) ==="
python3 -m src.evaluation.run_tube_v3_final \
  --config configs/tube_v3_adaptive_final_envchunks.yaml \
  --out results/tube_v3/adaptive_test_envchunks
echo "=== TEST FINISH $(date -Is) ==="
} >> "$LOG" 2>&1
