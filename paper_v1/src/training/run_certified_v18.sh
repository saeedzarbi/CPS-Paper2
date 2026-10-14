#!/usr/bin/env bash
# =============================================================================
# ABR-Awareness V18 : Certified Perceptual Shield
# -----------------------------------------------------------------------------
# Novel claim (model-agnostic runtime shield):
#   (1) VMAF-knee bandwidth banking  -- perceptually-lossless downshift on
#       saturated chunks (e.g. 6000->2850 buys ~0 VMAF, ~2x fewer bytes).
#   (2) Conformal throughput lower bound -- distribution-free coverage 1-alpha.
#   (3) Certified feasibility under that bound -> rebuffering guarantee.
#
# Unlike the V12-V16 "VMAF-aware ranking" shield (proven inert on monotone
# ladders), this exploits RATE-LADDER SATURATION, which the data genuinely has.
#
# Quick (no training; greedy + BBA, real + 5G traces):
#   cd new
#   bash src/training/run_certified_v18.sh
#
# Full (train proposed+pensieve overnight, then evaluate with the CPS):
#   FULL=1 NUM_ENVS=8 SEEDS=0 bash src/training/run_certified_v18.sh
#
# Env vars: PYTHON, SEEDS, NUM_ENVS, TS_SCALE, EPISODES, EPSILON, ALPHA, FULL
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0}"
NUM_ENVS="${NUM_ENVS:-8}"
TS_SCALE="${TS_SCALE:-1.0}"
EPISODES="${EPISODES:-$("${PY}" -c "from configs.videos import CPS_EPISODES; print(CPS_EPISODES)" 2>/dev/null || echo 204)}"
EPSILON="${EPSILON:-1.0}"
ALPHA="${ALPHA:-0.10}"
FULL="${FULL:-0}"

OUT="results/v18_certified"
# Provenance-correct datasets (built by data/prepare_traces_v18.py):
TRACE_5G="data/standardized/test_traces_5g_v18"          # 5G eval (seed 42)
TRACE_5G_TRAIN="data/standardized/train_traces_5g_v18"   # 5G co-design training (seed 7000)
TRACE_BB="data/standardized/test_traces_v18"             # source-disjoint broadband eval
TAG_PRIMARY="master_v18_5g"          # proposed/pensieve trained on broadband
TAG_CODESIGN="master_v18_5gtrain"    # proposed/proposed_cps trained on 5G (item 5)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=============================================================="
echo "V18 Certified Perceptual Shield"
echo "Root=${ROOT_DIR}  eps=${EPSILON}  alpha=${ALPHA}  FULL=${FULL}"
echo "=============================================================="

mkdir -p "${OUT}"

# ---- [0] Torch-free PoC (isolates banking vs safety-only) ----
echo ""
echo "[0] PoC: bitrate-greedy under RAW / SAFETY / CERTIFIED"
"${PY}" _poc_v18.py | tee "${OUT}/poc_console.txt"

# ---- [1] Prepare provenance-correct datasets ----
#  (A) source-disjoint broadband split (no FCC id / Norway route in both splits)
#  (B) matched 5G train/test sets with DISJOINT seeds (regime-consistent co-design)
echo ""
echo "[1] Prepare datasets (source-disjoint broadband split + 5G train/test)"
"${PY}" data/prepare_traces_v18.py

# ---- [2] Model-agnostic eval: GREEDY (no training needed) ----
echo ""
echo "[2] Eval GREEDY on 5G traces (the adversarial / bitrate-hungry case)"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy greedy --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/greedy_5g"

echo ""
echo "[3] Eval GREEDY on broadband test traces"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy greedy --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_BB}" --buffer 12 \
    --out "${OUT}/greedy_bb"

echo ""
echo "[4] Eval BBA on 5G traces"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy bba --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/bba_5g"

# ---- [4b] IMPROVED shield (items 1/3/4): risk-aware budget + dip forecasting ----
echo ""
echo "[4b] Eval GREEDY + BBA on 5G traces with the IMPROVED shield (predictive+forecast)"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy greedy --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --predictive --forecast \
    --out "${OUT}/greedy_5g_improved"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy bba --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --predictive --forecast \
    --out "${OUT}/bba_5g_improved"

echo ""
echo "[4c] Eval BOLA + RobustMPC on 5G traces (model-agnostic contrast)"
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

echo ""
echo "=============================================================="
echo "QUICK SIGNAL READY. Inspect:"
echo "  ${OUT}/greedy_5g/summary.json"
echo "  ${OUT}/bba_5g/summary.json"
echo "Success criteria for the novel claim:"
echo "  * CERTIFIED vs SAFETY: bandwidth DOWN, VMAF TOST within +/-eps"
echo "  * CERTIFIED vs RAW:    rebuffer DOWN (p<0.05), conformal cover ~ 1-alpha"
echo "=============================================================="

if [ "${FULL}" != "1" ]; then
    echo ""
    echo "QUICK mode done. Re-run with FULL=1 to train proposed+pensieve overnight"
    echo "and evaluate the CPS on those RL policies."
    exit 0
fi

# ---- FULL: train RL policies ----
#  PRIMARY (broadband, source-disjoint split): proposed + pensieve, NO shield in
#  the loop (the shield is a model-agnostic runtime wrapper applied at eval).
echo ""
echo "[5] FULL: train PRIMARY proposed + pensieve on source-disjoint broadband"
"${PY}" src/training/train_all_models_v18.py \
    --models proposed,pensieve --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}" \
    --trace-dir "${TRACE_BB/test_/train_}" --test-trace-dir "${TRACE_BB}" \
    --tag "${TAG_PRIMARY}"

#  CO-DESIGN (item 5), 5G regime: proposed (shield at eval only) vs proposed_cps
#  (shield active INSIDE training). Both trained on the SAME 5G train traces so the
#  comparison isolates co-design; in-training eval uses the 5G TRAIN set (no test
#  leakage). Final numbers come from the eval script on the 5G TEST set.
echo ""
echo "[5b] FULL: train CO-DESIGN proposed + proposed_cps on 5G (regime-matched)"
"${PY}" src/training/train_all_models_v18.py \
    --models proposed,proposed_cps --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}" \
    --trace-dir "${TRACE_5G_TRAIN}" --test-trace-dir "${TRACE_5G_TRAIN}" \
    --tag "${TAG_CODESIGN}"

# SB3 save() writes final_model.zip under results/models (see configs/paths.py).
# Pass path without .zip — PPO.load appends it. Prefer best_model if present.
_resolve_ckpt() {
    local base="$1"
    if [ -f "${base}/best_model/best_model.zip" ]; then
        echo "${base}/best_model/best_model"
    elif [ -f "${base}/final_model.zip" ]; then
        echo "${base}/final_model"
    else
        echo "${base}/final_model"
    fi
}
SEED0="${SEEDS%%,*}"
CKPT_PROP="$(_resolve_ckpt "results/models/${TAG_PRIMARY}/proposed_v14/seed_${SEED0}")"
CKPT_PEN="$(_resolve_ckpt "results/models/${TAG_PRIMARY}/pensieve_v14/seed_${SEED0}")"
CKPT_PROP_5G="$(_resolve_ckpt "results/models/${TAG_CODESIGN}/proposed_v14/seed_${SEED0}")"
CKPT_PROP_CPS="$(_resolve_ckpt "results/models/${TAG_CODESIGN}/proposed_cps_v18/seed_${SEED0}")"
echo "CKPT proposed(primary)=${CKPT_PROP}"
echo "CKPT pensieve(primary)=${CKPT_PEN}"
echo "CKPT proposed(5g)=${CKPT_PROP_5G}"
echo "CKPT proposed_cps(5g)=${CKPT_PROP_CPS}"
for _c in "${CKPT_PROP}" "${CKPT_PEN}" "${CKPT_PROP_5G}" "${CKPT_PROP_CPS}"; do
    if [ ! -f "${_c}.zip" ]; then
        echo "ERROR: missing checkpoint ${_c}.zip (did FULL training finish?)"
        exit 1
    fi
done

echo ""
echo "[6] FULL: CPS eval on proposed (base banking + IMPROVED shield)"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy ppo --ckpt "${CKPT_PROP}" --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/proposed_5g"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy ppo --ckpt "${CKPT_PROP}" --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --predictive --forecast \
    --out "${OUT}/proposed_5g_improved"

echo ""
echo "[7] FULL: CPS eval on pensieve (content-blind)"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy ppo --ckpt "${CKPT_PEN}" --blind --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/pensieve_5g"

# ---- [8] CO-DESIGN comparison (item 5): shield-aware vs shield-at-eval ----
# Fair, regime-matched: BOTH policies trained on the SAME 5G traces and deployed
# under the IMPROVED shield. Only difference = shield-in-the-loop during training.
echo ""
echo "[8] FULL: CO-DESIGN eval on 5G test (both IMPROVED shield)"
echo "    (8a) proposed trained in-regime, shield at eval only (baseline)"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy ppo --ckpt "${CKPT_PROP_5G}" --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --predictive --forecast \
    --out "${OUT}/proposed_5g_regime"
echo "    (8b) proposed_cps CO-TRAINED with the shield"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy ppo --ckpt "${CKPT_PROP_CPS}" --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --predictive --forecast \
    --out "${OUT}/proposed_cps_5g"

echo ""
echo "DONE (FULL). Key outputs under ${OUT}/"
echo "  PRIMARY claim (model-agnostic shield on broadband-trained policy):"
echo "    ${OUT}/proposed_5g{,_improved}/summary.json   ${OUT}/pensieve_5g/summary.json"
echo "  CO-DESIGN (item 5), regime-matched, IMPROVED shield -- compare CERTIFIED arm:"
echo "    ${OUT}/proposed_5g_regime/summary.json   (shield at eval only)"
echo "    ${OUT}/proposed_cps_5g/summary.json      (shield CO-TRAINED)"
echo "  Expect proposed_cps to reach higher VMAF / lower rebuffer in the certified arm."
