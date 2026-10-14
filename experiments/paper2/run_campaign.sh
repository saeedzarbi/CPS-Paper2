#!/usr/bin/env bash
# Master OACS evaluation runner (Paper 2). Wires in full eval modules as they land.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export PYTHONIOENCODING=utf-8
if [[ -f "${ROOT}/mattermost_webhook.url" ]]; then
  export MATTERMOST_WEBHOOK_URL="$(tr -d '\r\n' < "${ROOT}/mattermost_webhook.url")"
fi

MM="${PY:-python3} ${ROOT}/scripts/paper2/mattermost_notify.py"
mm() { "${PY:-python3}" "${ROOT}/scripts/paper2/mattermost_notify.py" "$1" || true; }

CAMPAIGN="${OACS_CAMPAIGN:-oacs-r1_main}"
OUT="${ROOT}/results/oacs/campaigns/${CAMPAIGN}"
mkdir -p "$OUT"

POLICIES=(greedy bba bola mpc)
TRACE_5G="data/standardized/test_traces_5g_v18"
TRACE_BB="data/standardized/test_traces_v18"

PY=python3
echo "OACS campaign=${CAMPAIGN} out=${OUT}"
mm ":rocket: **OACS campaign start** \`${CAMPAIGN}\` | episodes=\`${OACS_EPISODES:-204}\` | host=\`$(hostname)\`"

EVAL="${ROOT}/src/evaluation/eval_oacs_campaign.py"
if [[ ! -f "$EVAL" ]]; then
  echo "ERROR: missing ${EVAL}"
  exit 1
fi

EPISODES="${OACS_EPISODES:-204}"
for pol in "${POLICIES[@]}"; do
  echo "=== ${pol} 5G (${EPISODES} ep) ==="
  "$PY" "$EVAL" --policy "$pol" --trace-dir "$TRACE_5G" --episodes "$EPISODES" --out "${OUT}/${pol}_5g"
  echo "=== ${pol} broadband ==="
  "$PY" "$EVAL" --policy "$pol" --trace-dir "$TRACE_BB" --episodes "$EPISODES" --out "${OUT}/${pol}_bb"
done

for extra in stall_attribution_oacs coverage_drift_oacs; do
  script="${ROOT}/src/evaluation/${extra}.py"
  if [[ -f "$script" ]]; then
    "$PY" "$script" --out "${OUT}/${extra}"
  fi
done

echo "Campaign finished: ${OUT}"
mm ":tada: **OACS campaign finished** \`${CAMPAIGN}\` | out=\`${OUT}\`"
