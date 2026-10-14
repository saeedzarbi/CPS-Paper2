#!/usr/bin/env bash
# Tube-MPC eval on real Irish 5G traces (MMSys 2020). See docs/data_5g_real.md
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export PYTHONIOENCODING=utf-8
if [[ -f "${ROOT}/mattermost_webhook.url" ]]; then
  export MATTERMOST_WEBHOOK_URL="$(tr -d '\r\n' < "${ROOT}/mattermost_webhook.url")"
fi
mm() { python3 "${ROOT}/scripts/paper2/mattermost_notify.py" "$1" || true; }

CAMPAIGN="${TUBE_CAMPAIGN:-tube_mpc_r1_irish5g}"
TRACE_DIR="${TUBE_IRISH_TRACE_DIR:-data/standardized/test_traces_5g_irish_mmsys20}"
OUT="${ROOT}/results/tube_mpc/campaigns/${CAMPAIGN}"
mkdir -p "$OUT"
EPISODES="${TUBE_EPISODES:-204}"
PY=python3
EVAL="${ROOT}/src/evaluation/eval_tube_mpc_campaign.py"

if [[ ! -d "${ROOT}/${TRACE_DIR}" ]]; then
  echo "Missing ${TRACE_DIR} — run scripts/paper2/prepare_traces_5g_irish_mmsys20.py --all-test"
  exit 1
fi

mm ":rocket: **Tube-MPC Irish 5G** \`${CAMPAIGN}\` (${EPISODES} ep, real traces)"

echo "=== tube-mpc 5g_irish ==="
"$PY" "$EVAL" --trace-dir "$TRACE_DIR" --episodes "$EPISODES" --out "${OUT}/5g_irish"
mm ":white_check_mark: Tube-MPC \`5g_irish\` (real 5G) done → \`${OUT}/5g_irish\`"

mm ":tada: **Tube-MPC Irish 5G finished** \`${CAMPAIGN}\`"
echo "Done: ${OUT}"
