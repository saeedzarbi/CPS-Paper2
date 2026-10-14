#!/usr/bin/env bash
# =============================================================================
# Finalize v19 eval: refresh any 200-episode stale runs + ablation tables 8–9.
#
# Run on the Linux server (or locally if checkpoints/traces exist):
#   cd new
#   bash run_v19_finalize.sh
#
# What it does:
#   1. Re-eval BOLA + RobustMPC at CPS_EPISODES (204)
#   2. Re-eval proposed_5g_improved (primary PPO; proposed_5g skip if fresh)
#   3. Optional: pensieve negative control (REDO_PENSIEVE=1)
#   4. Re-run epsilon/alpha ablation (Tables 8–9) at 204 episodes
#   5. Regenerate all paper macros/tables/figures
#
# Skips co-design (already synced). To re-run co-design too:
#   REDO_CODESIGN=1 bash run_v19_remaining_eval.sh
#
# Afterward, pull to your laptop:
#   results/v18_certified/{bola_5g,mpc_5g,proposed_5g_improved}/summary.json
#   results/v18_certified/{bola_5g,mpc_5g,proposed_5g_improved}/episodes.csv
#   src/paper/overleaf_upload/tables/macros_cps.tex
#   src/paper/overleaf_upload/tables/table_ablation_*.tex
#   src/paper/overleaf_upload/tables/macros_ablation.tex
#   src/paper/overleaf_upload/figures/fig_cps_ablation.pdf
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PY="${PYTHON:-python3}"
OUT="results/v18_certified"
TRACE_5G="data/standardized/test_traces_5g_v18"
EPISODES="$("${PY}" -c "from configs.videos import CPS_EPISODES; print(CPS_EPISODES)")"
EPSILON="${EPSILON:-1.0}"
ALPHA="${ALPHA:-0.10}"
REDO_PENSIEVE="${REDO_PENSIEVE:-0}"
SKIP_ABLATION="${SKIP_ABLATION:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=== [1/4] BOLA + RobustMPC @ ${EPISODES} episodes ==="
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy bola --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/bola_5g"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy mpc --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/mpc_5g"

echo "=== [2/4] Primary PPO improved shield @ ${EPISODES} episodes ==="
TAG_PRIMARY="master_v18_5g"
SEED0="${SEEDS:-0}"
SEED0="${SEED0%%,*}"
CKPT_PROP="${ROOT_DIR}/results/models/${TAG_PRIMARY}/proposed_v14/seed_${SEED0}/final_model"
if [ -f "${CKPT_PROP}.zip" ]; then
    "${PY}" src/evaluation/eval_certified_shield_v18.py \
        --policy ppo --ckpt "${CKPT_PROP}" --episodes "${EPISODES}" \
        --epsilon "${EPSILON}" --alpha "${ALPHA}" \
        --trace-dir "${TRACE_5G}" --buffer 12 \
        --predictive --forecast \
        --out "${OUT}/proposed_5g_improved"
else
    echo "[WARN] Skip proposed_5g_improved: missing ${CKPT_PROP}.zip"
fi

if [ "${REDO_PENSIEVE}" = "1" ]; then
    echo "=== [2b] Pensieve negative control @ ${EPISODES} episodes ==="
    CKPT_PEN="${ROOT_DIR}/results/models/${TAG_PRIMARY}/pensieve_v14/seed_${SEED0}/final_model"
    if [ -f "${CKPT_PEN}.zip" ]; then
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy ppo --ckpt "${CKPT_PEN}" --blind --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_5G}" --buffer 12 \
            --out "${OUT}/pensieve_5g"
    else
        echo "[WARN] Skip pensieve_5g: missing ${CKPT_PEN}.zip"
    fi
fi

if [ "${SKIP_ABLATION}" != "1" ]; then
    echo "=== [3/4] Ablation eps/alpha (Tables 8–9) @ ${EPISODES} episodes ==="
    echo "    (7 greedy eval runs — may take 30–90 min depending on hardware)"
    "${PY}" src/evaluation/ablation_eps_alpha.py
else
    echo "=== [3/4] Skipping ablation (SKIP_ABLATION=1) ==="
fi

echo "=== [4/4] Regenerate paper assets ==="
"${PY}" src/paper/make_cps_paper_assets.py

echo ""
echo "DONE. Sync overleaf_upload/ to Overleaf and recompile main.tex."
