#!/usr/bin/env bash
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export PYTHONIOENCODING=utf-8
if [[ -f "${ROOT}/mattermost_webhook.url" ]]; then
  export MATTERMOST_WEBHOOK_URL="$(tr -d '\r\n' < "${ROOT}/mattermost_webhook.url")"
fi
mm() { python3 "${ROOT}/scripts/paper2/mattermost_notify.py" "$1" || true; }

CAMPAIGN="${TUBE_CAMPAIGN:-tube_mpc_r1}"
OUT="${ROOT}/results/tube_mpc/campaigns/${CAMPAIGN}"
mkdir -p "$OUT"
EPISODES="${TUBE_EPISODES:-204}"
PY=python3
EVAL="${ROOT}/src/evaluation/eval_tube_mpc_campaign.py"

mm ":rocket: **Tube-MPC campaign** \`${CAMPAIGN}\` (${EPISODES} ep)"

for pair in "5g:data/standardized/test_traces_5g_v18" "bb:data/standardized/test_traces_v18"; do
  name="${pair%%:*}"
  dir="${pair#*:}"
  echo "=== tube-mpc ${name} ==="
  "$PY" "$EVAL" --trace-dir "$dir" --episodes "$EPISODES" --out "${OUT}/${name}"
  mm ":white_check_mark: Tube-MPC \`${name}\` done → \`${OUT}/${name}\`"
done

mm ":tada: **Tube-MPC campaign finished** \`${CAMPAIGN}\`"
echo "Done: ${OUT}"
