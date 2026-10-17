#!/usr/bin/env bash
set -eu
cd /root/CPS-Paper2
source .venv-tube-v2/bin/activate
export PYTHONPATH=/root/CPS-Paper2
export OMP_NUM_THREADS=1
LOG=results/tube_v3/adaptive_val_envchunks_run.log
mkdir -p results/tube_v3
{
echo "=== START $(date -Is) ==="
python3 -m src.evaluation.diagnose_val_fs_calib \
  --trace-set bb --split val \
  --out results/tube_v3/fs_calib_diagnose_bb_val.json
echo "=== diagnose done $(date -Is) ==="
python3 -m src.evaluation.run_tube_v3_adaptive_val --fs-calib-mode env_chunks
echo "=== FINISH $(date -Is) ==="
} >> "$LOG" 2>&1
