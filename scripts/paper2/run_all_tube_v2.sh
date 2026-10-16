#!/usr/bin/env bash
# Tube-v2 unified driver (venv, logging, phases).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

if [[ -f "${ROOT}/.venv-tube-v2/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv-tube-v2/bin/activate"
elif [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ -n "${CPS_DATA_DIR:-}" ]]; then
  export CPS_PAPER2_DATA_ROOT="${CPS_DATA_DIR}"
fi

RUN_NAME="tube_v2_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
echo "=== ${RUN_NAME} ===" | tee -a "$LOG_FILE"
echo "argv: $*" | tee -a "$LOG_FILE"

python "${ROOT}/src/evaluation/tube_v2_campaign.py" "$@" 2>&1 | tee -a "$LOG_FILE"
