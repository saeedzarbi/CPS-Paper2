#!/usr/bin/env bash
# =============================================================================
# ABR-Awareness v19 — full server runbook (step-by-step + Mattermost alerts)
#
# Runs the entire Stage-1 pipeline on a Linux server:
#   preflight -> download 12 videos -> encode -> VMAF -> SI/TI
#   -> (optional) multires ladder -> prepare traces -> CPS quick eval
#   -> (optional) FULL PPO retrain + eval
#
# Mattermost notifications are sent:
#   * when each step STARTS and COMPLETES successfully
#   * immediately on any failure (then the script exits)
#
# Usage:
#   cd new
#   bash run_v19_full_server.sh
#
# Options (environment variables):
#   MATTERMOST_WEBHOOK   webhook URL (default: project hook below)
#   PYTHON               python binary (default: python3, fallback python)
#   SKIP_DOWNLOAD=1      skip raw video download
#   SKIP_ENCODE=1        skip encode step (raw + encoded already done)
#   SKIP_VMAF=1          skip VMAF step (vmaf_summary.csv already done)
#   FAST_VMAF=1          faster VMAF pass (720p subsampling on low rungs)
#   PARALLEL_VMAF=1      parallel VMAF workers
#   BUILD_MULTIRES=1     build per-chunk multires VMAF ladder (slow)
#   SKIP_CPS_EVAL=1      stop after data pipeline (no shield eval)
#   FULL=1               also run overnight PPO retrain via run_certified_v18.sh
#                        (re-runs CPS quick eval inside that script — expect overlap)
#   HOST_LABEL           hostname tag in notifications (default: $(hostname))
#
# Test Mattermost webhook only:
#   bash run_v19_full_server.sh --test-notify
#
# Log: results/v19_server_run.log (appended)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# --- Mattermost ----------------------------------------------------------------
MATTERMOST_WEBHOOK="${MATTERMOST_WEBHOOK:-https://mychat.moshaver-amlak.com/hooks/nmofp9ywif857pgjy7xy6shwwy}"
HOST_LABEL="${HOST_LABEL:-$(hostname -s 2>/dev/null || hostname)}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

# --- Pipeline toggles ----------------------------------------------------------
PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3
    elif command -v python  >/dev/null 2>&1; then PY=python
    else PY=python3
    fi
fi
RAW_DIR="${RAW_DIR:-data/raw_videos}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_ENCODE="${SKIP_ENCODE:-0}"
SKIP_VMAF="${SKIP_VMAF:-0}"
FAST_VMAF="${FAST_VMAF:-0}"
PARALLEL_VMAF="${PARALLEL_VMAF:-0}"
BUILD_MULTIRES="${BUILD_MULTIRES:-0}"
SKIP_CPS_EVAL="${SKIP_CPS_EVAL:-0}"
FULL="${FULL:-0}"

LOG_DIR="${LOG_DIR:-results}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/v19_server_run.log"

# --- Notify helpers ------------------------------------------------------------
notify_mm() {
    # notify_mm "message text"
    local text="$1"
    local payload
    payload="$("${PY}" -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "${text}" 2>/dev/null)" \
        || payload="{\"text\": $(printf '%s' "${text}" | "${PY}" -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}"

    if ! curl -sS -m 30 -X POST \
        -H 'Content-Type: application/json' \
        -d "${payload}" \
        "${MATTERMOST_WEBHOOK}" >/dev/null 2>&1; then
        echo "[WARN] Mattermost notification failed (continuing pipeline)." | tee -a "${LOG_FILE}"
    fi
}

notify_start() {
    notify_mm "**[ABR v19 | ${HOST_LABEL}]** ▶️ START step **${CURRENT_STEP}/${TOTAL_STEPS}**: ${STEP_NAME}  \`run=${RUN_ID}\`"
}

notify_done() {
    notify_mm "**[ABR v19 | ${HOST_LABEL}]** ✅ DONE step **${CURRENT_STEP}/${TOTAL_STEPS}**: ${STEP_NAME}  \`run=${RUN_ID}\`"
}

notify_fail() {
    local exit_code="${1:-1}"
    notify_mm "**[ABR v19 | ${HOST_LABEL}]** ❌ **FAILED** step **${CURRENT_STEP}/${TOTAL_STEPS}**: ${STEP_NAME}  \`exit=${exit_code}\`  \`run=${RUN_ID}\`  — check \`${LOG_FILE}\` on the server."
}

on_err() {
    local code=$?
    notify_fail "${code}"
    echo "[FATAL] Pipeline aborted at step ${CURRENT_STEP}: ${STEP_NAME} (exit ${code})" | tee -a "${LOG_FILE}"
    exit "${code}"
}
trap on_err ERR

# --- Step runner ---------------------------------------------------------------
CURRENT_STEP=0
TOTAL_STEPS=0
STEP_NAME=""

# Count steps dynamically once we know video count
count_steps() {
    local n_videos
    n_videos="$("${PY}" -c "from configs.videos import ALL_VIDEOS; print(len(ALL_VIDEOS))")"
    # preflight + download + vmaf + siti + traces + provenance + final = 7 base
    TOTAL_STEPS=$((7 + n_videos))
    if [ "${BUILD_MULTIRES}" = "1" ]; then TOTAL_STEPS=$((TOTAL_STEPS + 1)); fi
    if [ "${SKIP_CPS_EVAL}" != "1" ]; then
        local cps_steps=5   # greedy_5g, greedy_bb, bba_5g, greedy_improved, bba_improved
        [ -f "_poc_v18.py" ] && cps_steps=$((cps_steps + 1))
        TOTAL_STEPS=$((TOTAL_STEPS + cps_steps))
        [ "${FULL}" = "1" ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))
    fi
    if [ "${SKIP_DOWNLOAD}" = "1" ]; then TOTAL_STEPS=$((TOTAL_STEPS - 1)); fi
    if [ "${SKIP_ENCODE}" = "1" ]; then TOTAL_STEPS=$((TOTAL_STEPS - n_videos)); fi
    if [ "${SKIP_VMAF}" = "1" ]; then TOTAL_STEPS=$((TOTAL_STEPS - 1)); fi
}

run_step() {
    STEP_NAME="$1"
    shift
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "" | tee -a "${LOG_FILE}"
    echo "========== [${CURRENT_STEP}/${TOTAL_STEPS}] ${STEP_NAME} ==========" | tee -a "${LOG_FILE}"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}"
    notify_start
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}"
    notify_done
}

# --- Preflight -----------------------------------------------------------------
preflight() {
    command -v ffmpeg >/dev/null 2>&1 || { echo "[FATAL] ffmpeg not found"; exit 2; }
    command -v curl   >/dev/null 2>&1 || { echo "[FATAL] curl not found (needed for Mattermost)"; exit 2; }
    local ff
    ff="$(ffmpeg -hide_banner -filters 2>/dev/null || true)"
    case "${ff}" in *libvmaf*) : ;; *)
        echo "[FATAL] ffmpeg lacks libvmaf filter"; exit 2 ;;
    esac
    if [ "${SKIP_DOWNLOAD}" != "1" ]; then
        command -v xz >/dev/null 2>&1 || { echo "[FATAL] xz not found (needed for elephants_dream)"; exit 2; }
    fi
    "${PY}" -c "import cv2; from configs.videos import ALL_VIDEOS, CPS_EPISODES; print(f'videos={len(ALL_VIDEOS)} episodes={CPS_EPISODES} opencv={cv2.__version__}')"
}

# --- Main ----------------------------------------------------------------------
main() {
    count_steps

    notify_mm "**[ABR v19 | ${HOST_LABEL}]** 🚀 **Pipeline launched** (${TOTAL_STEPS} steps)  \`run=${RUN_ID}\`  \`FULL=${FULL}\`  \`FAST_VMAF=${FAST_VMAF}\`"

    run_step "Preflight checks (ffmpeg, libvmaf, python, configs)" preflight

    VIDEOS="$("${PY}" -c "from configs.videos import EVAL_VIDEOS_CSV; print(EVAL_VIDEOS_CSV)")"

    if [ "${SKIP_DOWNLOAD}" != "1" ]; then
        run_step "Download and prepare 12 raw reference videos" \
            env RAW_DIR="${RAW_DIR}" bash data/download_raw_videos.sh
    else
        echo "[INFO] SKIP_DOWNLOAD=1 — skipping download" | tee -a "${LOG_FILE}"
    fi

    for v in $(echo "${VIDEOS}" | tr ',' ' '); do
        if [ "${SKIP_ENCODE}" = "1" ]; then
            echo "[INFO] SKIP_ENCODE=1 — skipping encode for ${v}" | tee -a "${LOG_FILE}"
            continue
        fi
        run_step "Encode 6-rung ladder: ${v}" \
            "${PY}" data/video_encoder.py --video "${v}" \
                --input-dir "${RAW_DIR}" --output-dir data/encoded_videos
    done

    if [ "${SKIP_VMAF}" != "1" ]; then
        VMAF_CMD=("${PY}" data/vmaf_calculator.py --videos "${VIDEOS}")
        [ "${FAST_VMAF}" = "1" ]     && VMAF_CMD+=(--fast)
        [ "${PARALLEL_VMAF}" = "1" ] && VMAF_CMD+=(--parallel)
        run_step "Session-mean VMAF (12 videos x 6 rungs)" "${VMAF_CMD[@]}"
    else
        echo "[INFO] SKIP_VMAF=1 — skipping VMAF" | tee -a "${LOG_FILE}"
    fi

    run_step "Extract SI/TI content features" \
        "${PY}" data/si_ti_extractor.py --videos "${VIDEOS}" --video-dir "${RAW_DIR}"

    if [ "${BUILD_MULTIRES}" = "1" ]; then
        run_step "Build per-chunk multi-resolution VMAF ladder" \
            "${PY}" data/build_multires_vmaf.py --raw-dir "${RAW_DIR}" --videos "${VIDEOS}"
    fi

    run_step "Prepare v18 trace datasets (broadband + 5G splits)" \
        "${PY}" data/prepare_traces_v18.py

    run_step "Write provenance record (train vs held-out videos)" \
        "${PY}" src/paper/scripts/make_provenance_v14.py

    if [ "${SKIP_CPS_EVAL}" = "1" ]; then
        notify_mm "**[ABR v19 | ${HOST_LABEL}]** 🏁 **Data pipeline complete** (CPS eval skipped)  \`run=${RUN_ID}\`"
        return 0
    fi

    OUT="results/v18_certified"
    mkdir -p "${OUT}"
    TRACE_5G="data/standardized/test_traces_5g_v18"
    TRACE_BB="data/standardized/test_traces_v18"
    EPISODES="$("${PY}" -c "from configs.videos import CPS_EPISODES; print(CPS_EPISODES)")"
    EPSILON="${EPSILON:-1.0}"
    ALPHA="${ALPHA:-0.10}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

    if [ -f "_poc_v18.py" ]; then
        run_step "PoC: greedy RAW / SAFETY / CERTIFIED (torch-free)" \
            bash -c "\"${PY}\" _poc_v18.py | tee \"${OUT}/poc_console.txt\""
    else
        echo "[INFO] _poc_v18.py not found — skipping PoC" | tee -a "${LOG_FILE}"
    fi

    run_step "CPS eval: GREEDY on 5G test traces (${EPISODES} episodes)" \
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy greedy --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_5G}" --buffer 12 \
            --out "${OUT}/greedy_5g"

    run_step "CPS eval: GREEDY on broadband test traces (${EPISODES} episodes)" \
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy greedy --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_BB}" --buffer 12 \
            --out "${OUT}/greedy_bb"

    run_step "CPS eval: BBA on 5G test traces (${EPISODES} episodes)" \
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy bba --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_5G}" --buffer 12 \
            --out "${OUT}/bba_5g"

    run_step "CPS eval: GREEDY improved shield (predictive + forecast)" \
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy greedy --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_5G}" --buffer 12 \
            --predictive --forecast \
            --out "${OUT}/greedy_5g_improved"

    run_step "CPS eval: BBA improved shield (predictive + forecast)" \
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy bba --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_5G}" --buffer 12 \
            --predictive --forecast \
            --out "${OUT}/bba_5g_improved"

    run_step "CPS eval: BOLA on 5G test traces (${EPISODES} episodes)" \
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy bola --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_5G}" --buffer 12 \
            --out "${OUT}/bola_5g"

    run_step "CPS eval: RobustMPC on 5G test traces (${EPISODES} episodes)" \
        "${PY}" src/evaluation/eval_certified_shield_v18.py \
            --policy mpc --episodes "${EPISODES}" \
            --epsilon "${EPSILON}" --alpha "${ALPHA}" \
            --trace-dir "${TRACE_5G}" --buffer 12 \
            --out "${OUT}/mpc_5g"

    if [ "${FULL}" = "1" ]; then
        run_step "FULL: PPO training + CPS eval (proposed, pensieve, co-design)" \
            env FULL=1 SEEDS="${SEEDS:-0}" NUM_ENVS="${NUM_ENVS:-8}" \
                EPISODES="${EPISODES}" EPSILON="${EPSILON}" ALPHA="${ALPHA}" \
                PYTHON="${PY}" bash src/training/run_certified_v18.sh
    fi

    notify_mm "**[ABR v19 | ${HOST_LABEL}]** 🎉 **All steps completed successfully**  \`run=${RUN_ID}\`  \`steps=${TOTAL_STEPS}\`  \`log=${LOG_FILE}\`"
    echo ""
    echo "=============================================================="
    echo "SUCCESS. Full log: ${LOG_FILE}"
    echo "CPS summaries: ${OUT}/greedy_5g/summary.json  (and siblings)"
    echo "=============================================================="
}

# --- Entry ---------------------------------------------------------------------
if [ "${1:-}" = "--test-notify" ]; then
    notify_mm "**[ABR v19 | ${HOST_LABEL}]** 🔔 Webhook test OK  \`run=${RUN_ID}\`"
    echo "Test notification sent to Mattermost."
    exit 0
fi

main "$@"
