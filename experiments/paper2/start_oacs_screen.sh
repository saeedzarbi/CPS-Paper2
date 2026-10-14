#!/usr/bin/env bash
# Detached screen job for OACS (run once on server: bash experiments/paper2/start_oacs_screen.sh)
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/oacs/logs
LOG="results/oacs/logs/run_$(date +%Y%m%d_%H%M%S).log"

if screen -list | grep -q '\.oacs'; then
  echo "Screen session 'oacs' already exists. Attach: screen -r oacs"
  exit 0
fi

screen -dmS oacs bash -lc "
  cd '$ROOT' &&
  source .venv/bin/activate &&
  export PYTHONPATH='$ROOT' PYTHONIOENCODING=utf-8 &&
  exec > >(tee -a '$LOG') 2>&1 &&
  echo START \$(date) &&
  bash experiments/paper2/run_smoke.sh &&
  bash experiments/paper2/run_campaign.sh &&
  echo END \$(date)
"
echo "Started screen 'oacs'. Log: $LOG"
echo "Attach: screen -r oacs"
