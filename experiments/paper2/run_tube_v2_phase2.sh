#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv-tube-v2/bin/activate 2>/dev/null || source .venv/bin/activate
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/evaluation/run_tube_v2_phase2.py "$@"
