#!/usr/bin/env bash
# Master OACS evaluation runner (Paper 2). Wires in full eval modules as they land.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export PYTHONIOENCODING=utf-8

CAMPAIGN="${OACS_CAMPAIGN:-oacs-r1_main}"
OUT="${ROOT}/results/oacs/campaigns/${CAMPAIGN}"
mkdir -p "$OUT"

PY=python3
POLICIES=(greedy bba bola mpc)
TRACE_5G="data/standardized/test_traces_5g_v18"
TRACE_BB="data/standardized/test_traces_v18"

echo "OACS campaign=${CAMPAIGN} out=${OUT}"

EVAL="${ROOT}/src/evaluation/eval_oacs_campaign.py"
if [[ ! -f "$EVAL" ]]; then
  echo "NOTE: ${EVAL} not implemented yet (Phase 3)."
  echo "      Run: bash experiments/paper2/run_smoke.sh"
  exit 0
fi

for pol in "${POLICIES[@]}"; do
  "$PY" "$EVAL" --policy "$pol" --trace-dir "$TRACE_5G" --out "${OUT}/${pol}_5g"
  "$PY" "$EVAL" --policy "$pol" --trace-dir "$TRACE_BB" --out "${OUT}/${pol}_bb"
done

for extra in stall_attribution_oacs coverage_drift_oacs; do
  script="${ROOT}/src/evaluation/${extra}.py"
  if [[ -f "$script" ]]; then
    "$PY" "$script" --out "${OUT}/${extra}"
  fi
done

echo "Campaign finished: ${OUT}"
