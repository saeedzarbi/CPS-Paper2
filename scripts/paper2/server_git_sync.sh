#!/usr/bin/env bash
# Reset server working tree to match GitHub (after a messy manual deploy).
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
git fetch origin
git checkout main 2>/dev/null || git checkout -b main origin/main
git reset --hard origin/main
git clean -fd
echo "Synced to: $(git rev-parse --short HEAD) $(git log -1 --oneline)"
python3 configs/paths.py || true
python3 scripts/paper2/verify_data.py || true
