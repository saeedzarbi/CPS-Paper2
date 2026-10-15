#!/usr/bin/env bash
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [ -f "${ROOT}/mattermost_webhook.url" ]; then
  export MATTERMOST_WEBHOOK_URL="$(tr -d '\r\n' < "${ROOT}/mattermost_webhook.url")"
fi
export PYTHONPATH="${ROOT}"
export PYTHONIOENCODING=utf-8
EPISODES="${CLEAN_V3_EXT_EPISODES:-204}"
LATENCY_EPS="${CLEAN_V3_EXT_LATENCY_EPS:-17}"
export CLEAN_V3_EXT_RESERVED_CPUS="${CLEAN_V3_EXT_RESERVED_CPUS:-2}"
export CLEAN_V3_EXT_MAX_WORKERS="${CLEAN_V3_EXT_MAX_WORKERS:-2}"
NICE_LEVEL="${CLEAN_V3_EXT_NICE:-12}"
if [ -x "${ROOT}/.venv/bin/python" ]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY=python3
fi
WORKER_ARGS=()
if [ -n "${CLEAN_V3_EXT_WORKERS:-}" ]; then
  WORKER_ARGS=(--workers "${CLEAN_V3_EXT_WORKERS}")
fi
RUN=(nice -n "${NICE_LEVEL}" "${PY}" "${ROOT}/src/evaluation/run_extended_validation.py"
  --episodes "${EPISODES}"
  --latency-episodes "${LATENCY_EPS}"
  "${WORKER_ARGS[@]}"
  "$@")
exec "${RUN[@]}"
