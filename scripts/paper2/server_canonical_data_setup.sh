#!/usr/bin/env bash
# Run on the Tube-v2 VPS: canonical standardized data, manifest, verified Phase 2 pilot.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
LEGACY="${ROOT}/data/_legacy/5g_markov_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LEGACY"
echo "Moving legacy 5g_markov_*.json → $LEGACY"
while IFS= read -r -d '' f; do
  mv "$f" "$LEGACY/"
done < <(find "${ROOT}/data/standardized" -name '5g_markov_*.json' -print0 2>/dev/null || true)

rm -rf "${ROOT}/results/tube_v2/splits"
source .venv-tube-v2/bin/activate
export OMP_NUM_THREADS=1
python scripts/paper2/gen_markov_traces.py
python -c 'from configs.trace_sets_v2 import harsh_family_three_way_dirs; harsh_family_three_way_dirs(force=True)'
python scripts/paper2/build_trace_manifests.py --write
python scripts/paper2/build_trace_manifests.py --verify

rm -rf "${ROOT}/results/tube_v2/phase2"
mkdir -p "${ROOT}/results/tube_v2/phase2"

echo "Setup OK. Launch verified pilot (detached; tmux optional if installed):"
echo "  nohup bash -lc 'source .venv-tube-v2/bin/activate && export OMP_NUM_THREADS=1 && python src/evaluation/run_tube_v2_phase2.py --pilot --trace-sets 5g_markov 5g_harsh bb --workers 1 --no-resume' > results/tube_v2/phase2/run.log 2>&1 &"
echo "  tail -f results/tube_v2/phase2/run.log"
echo "  test -f results/tube_v2/phase2/DONE.json && echo finished"
