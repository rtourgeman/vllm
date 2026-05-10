#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"

export VLLM_HOST_IP="$(get_routable_ip)"

my_node="$(hostname -s)"
vllm_pid=""
ROLE="${ROLE:-primary_head}"
TAG=""; [[ "${ROLE}" == "secondary_head" ]] && TAG="[B] "

if [[ "${ROLE}" == "primary_head" ]]; then
    MY_RAY_PORT="${RAY_PORT}"
    MY_RAY_IP="${HEAD_NODE_IP}"
    MY_METRICS_PORT=9090
    MY_VLLM_LOG="${RUN_DIR}/vllm_server.log"
    MY_WAIT_NODES="${INITIAL_NODES}"
else
    MY_RAY_PORT="${RAY_PORT_B}"
    MY_RAY_IP="${SECONDARY_HEAD_IP}"
    MY_METRICS_PORT=9091
    MY_VLLM_LOG="${RUN_DIR}/vllm_server_B.log"
    MY_WAIT_NODES=$((TARGET_NODES - INITIAL_NODES))
    SECONDARY_MODEL_NAME="${SECONDARY_MODEL_NAME:-/rtourgeman/models/DeepSeek-V2-Lite-Chat}"
fi

export RAY_ADDRESS="${MY_RAY_IP}:${MY_RAY_PORT}"

save_ray_logs() {
    local ray_log_dir
    ray_log_dir="$(ls -td /data/tmp/ray/session_*/logs 2>/dev/null | head -1)"
    if [[ -n "${ray_log_dir}" && -d "${ray_log_dir}" ]]; then
        local dest="${RUN_DIR}/ray_logs"
        mkdir -p "${dest}"
        echo "[${my_node}] ${TAG}copying Ray logs to ${dest}"
        cp -r "${ray_log_dir}/"worker-*.err "${dest}/" 2>/dev/null || true
        cp -r "${ray_log_dir}/"worker-*.out "${dest}/" 2>/dev/null || true
        cp "${ray_log_dir}/raylet.err" "${dest}/" 2>/dev/null || true
        cp "${ray_log_dir}/ray_process_exit.log" "${dest}/" 2>/dev/null || true
    fi
}

cleanup() {
    [[ "${ROLE}" == "primary_head" ]] && save_ray_logs
    if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" 2>&1; then
        echo "[${my_node}] ${TAG}stopping vLLM pid=${vllm_pid}"
        kill "${vllm_pid}" 2>&1 || true
        sleep 5
        kill -9 "${vllm_pid}" 2>&1 || true
    fi
    [[ "${ROLE}" == "primary_head" ]] && rm -f "${SIGNAL_FILE}" 2>&1 || true
    ray stop -f 2>&1 || true
}
trap cleanup EXIT

cd "${VLLM_WORKDIR}"

start_ray_head() {
    for ((attempt=1; attempt<=5; attempt++)); do
        ray stop -f 2>&1 || true
        if ray start --head \
            --port="${MY_RAY_PORT}" \
            --node-ip-address="${MY_RAY_IP}" \
            --num-gpus="${GPUS_PER_NODE}" \
            --metrics-export-port="${MY_METRICS_PORT}" \
            --dashboard-agent-grpc-port=9094 \
            --runtime-env-agent-port=9095 \
            --min-worker-port=20000 \
            --max-worker-port=29999; then
            return 0
        fi
        echo "[${my_node}] ${TAG}Ray head attempt ${attempt}/5 failed, retrying"
        sleep 5
    done
    return 1
}

launch_vllm() {
    local log_file="${1}"
    local dp_size="${2}"
    echo "[${my_node}] launching vLLM (dp=${dp_size}), log=${log_file}"
    DATA_PARALLEL_SIZE="${dp_size}" \
        bash "${SCRIPT_DIR}/serve.sh" >"${log_file}" 2>&1 &
    vllm_pid="$!"
}

wait_vllm_ready() {
    local log_file="${1}"
    local waited=0
    echo "[${my_node}] waiting for vLLM to start"
    while ! grep -q "Application startup complete" "${log_file}" 2>/dev/null; do
        if ! kill -0 "${vllm_pid}" >/dev/null 2>&1; then
            echo "[${my_node}] vLLM exited before healthy" >&2
            tail -20 "${log_file}" >&2 || true
            wait "${vllm_pid}" || true
            return 1
        fi
        if (( waited >= SERVER_WAIT_TIMEOUT )); then
            echo "[${my_node}] timed out waiting for vLLM" >&2
            tail -20 "${log_file}" >&2 || true
            return 1
        fi
        sleep 15
        waited=$((waited + 15))
        echo "[${my_node}] still waiting for vLLM startup... ${waited}s"
    done
    echo "[${my_node}] vLLM is running"
}

stop_vllm() {
    if [[ -n "${vllm_pid}" ]]; then
        echo "[${my_node}] stopping vLLM pid=${vllm_pid}"
        kill "${vllm_pid}" 2>&1 || true
        sleep 5
        kill -9 "${vllm_pid}" 2>&1 || true
        wait "${vllm_pid}" 2>/dev/null || true
        vllm_pid=""
    fi
}

run_bench() {
    local label="${1}"
    local dp="${2}"
    local log="${RUN_DIR}/bench_${label}_${dp}gpu_${BENCH_TAG}.log"
    echo "[${my_node}] running benchmark '${label}' at ${dp} GPUs -> ${log}"
    BENCH_LOG_FILE="${log}" bash "${SCRIPT_DIR}/bench.sh"
}

# ── secondary head: unchanged ──
if [[ "${ROLE}" == "secondary_head" ]]; then
    start_ray_head

    echo "[${my_node}] ${TAG}waiting for ${MY_WAIT_NODES} Ray node(s)"
    wait_for_ray_nodes "${MY_WAIT_NODES}" "${RAY_WAIT_TIMEOUT}" || true
    ray status || true

    warm_lustre_cache "${SECONDARY_MODEL_NAME}"
    echo "[${my_node}] ${TAG}launching vLLM (secondary, model=${SECONDARY_MODEL_NAME}), log=${MY_VLLM_LOG}"
    MODEL_NAME="${SECONDARY_MODEL_NAME}" \
    DATA_PARALLEL_SIZE="${SECONDARY_DP_SIZE}" \
    DATA_PARALLEL_SIZE_LOCAL="${GPUS_PER_NODE}" \
    DATA_PARALLEL_ADDRESS="${SECONDARY_HEAD_IP}" \
    DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT_B}" \
    VLLM_NIXL_EP_MAX_NUM_RANKS="${SECONDARY_DP_SIZE}" \
    PORT="${PORT_B}" \
        bash "${SCRIPT_DIR}/serve.sh" >"${MY_VLLM_LOG}" 2>&1 &
    vllm_pid="$!"

    wait_vllm_ready "${MY_VLLM_LOG}"

    echo "[${my_node}] ${TAG}running small benchmark (${SECONDARY_NUM_PROMPTS} prompts, model=${SECONDARY_MODEL_NAME})"
    MODEL_NAME="${SECONDARY_MODEL_NAME}" \
    NUM_PROMPTS="${SECONDARY_NUM_PROMPTS}" \
    MAX_CONCURRENCY="${SECONDARY_NUM_PROMPTS}" \
    BENCH_HOST="localhost" \
    PORT="${PORT_B}" \
    WAIT_FOR_SERVER="false" \
    BENCH_LOG_FILE="${RUN_DIR}/bench_secondary_${SECONDARY_DP_SIZE}gpu_np${SECONDARY_NUM_PROMPTS}.log" \
        bash "${SCRIPT_DIR}/bench.sh" || true

    echo "[${my_node}] ${TAG}secondary benchmark done, waiting for scale-up signal"
    waited=0
    while [[ ! -f "${SIGNAL_FILE}" ]]; do
        if (( waited >= SERVER_WAIT_TIMEOUT )); then
            echo "[${my_node}] ${TAG}timed out waiting for signal" >&2
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done

    echo "[${my_node}] ${TAG}tearing down secondary vLLM + Ray"
    kill "${vllm_pid}" 2>&1 || true
    sleep 3
    kill -9 "${vllm_pid}" 2>&1 || true
    vllm_pid=""
    ray stop -f 2>&1 || true
    sleep 5
    pkill -9 -f "runtime_env_agent" 2>&1 || true
    pkill -9 -f "raylet" 2>&1 || true
    sleep 10
    ray stop -f 2>&1 || true
    sleep 5

    echo "[${my_node}] ${TAG}joining primary Ray cluster at ${HEAD_NODE_IP}:${RAY_PORT}"
    export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
    join_ray_with_retry "${HEAD_NODE_IP}:${RAY_PORT}" "${GPUS_PER_NODE}"
    exit 0
fi

# ── primary head: 3-phase apple-to-apple benchmark ──

BENCH_TAG="np${NUM_PROMPTS}_c${MAX_CONCURRENCY}_i${RANDOM_INPUT_LEN}_o${RANDOM_OUTPUT_LEN}"
warm_lustre_cache "${MODEL_NAME}"

restart_ray() {
    echo "[${my_node}] restarting Ray cluster (clean session)"
    stop_vllm
    ray stop -f 2>&1 || true
    pkill -9 -f "raylet|runtime_env_agent" 2>&1 || true
    rm -rf /data/tmp/ray/session_* 2>/dev/null || true
    sleep 5
    start_ray_head
    echo "[${my_node}] waiting for ${TARGET_NODES} Ray node(s)"
    wait_for_ray_nodes "${TARGET_NODES}" "${RAY_WAIT_TIMEOUT}" || true
    ray status || true
}

# ────────────────────────────────────────────────────
# Phase 1: 32-GPU baseline (clean EPLB)
# ────────────────────────────────────────────────────
echo ""
echo "========== PHASE 1: ${INITIAL_DP_SIZE}-GPU baseline =========="
restart_ray

launch_vllm "${RUN_DIR}/vllm_phase1_${INITIAL_DP_SIZE}gpu.log" "${INITIAL_DP_SIZE}"
wait_vllm_ready "${RUN_DIR}/vllm_phase1_${INITIAL_DP_SIZE}gpu.log"

run_bench "phase1" "${INITIAL_DP_SIZE}"

stop_vllm
echo "[${my_node}] Phase 1 done"

if [[ "${RUN_ELASTIC_SCALE}" != "true" ]]; then
    echo "[${my_node}] no elastic scale requested, exiting"
    ray stop -f 2>&1 || true
    touch "${RUN_DIR}/job_done.signal"
    exit 0
fi

# ────────────────────────────────────────────────────
# Phase 2: 32→40 elastic scale-up (clean EPLB)
# ────────────────────────────────────────────────────
echo ""
echo "========== PHASE 2: ${INITIAL_DP_SIZE}→${TARGET_DP_SIZE} elastic scale-up =========="
restart_ray

launch_vllm "${RUN_DIR}/vllm_phase2_${INITIAL_DP_SIZE}to${TARGET_DP_SIZE}gpu.log" "${INITIAL_DP_SIZE}"
wait_vllm_ready "${RUN_DIR}/vllm_phase2_${INITIAL_DP_SIZE}to${TARGET_DP_SIZE}gpu.log"

echo "[${my_node}] signaling additional nodes to join Ray"
touch "${SIGNAL_FILE}"

echo "[${my_node}] waiting for ${TARGET_NODES} Ray node(s) before scale-up"
if ! wait_for_ray_nodes "${TARGET_NODES}" "${RAY_WAIT_TIMEOUT}"; then
    echo "[${my_node}] scale-up aborted: not enough Ray nodes" >&2
    exit 1
fi

ray status || true
echo "[${my_node}] scaling vLLM to ${TARGET_DP_SIZE} GPUs"
scale_start=$(date +%s)
python3 examples/online_serving/elastic_ep/scale.py \
    --host "localhost" \
    --port "${PORT}" \
    --new-dp-size "${TARGET_DP_SIZE}" \
    --num-redundant-experts 24
scale_end=$(date +%s)
echo "[${my_node}] scale-up completed in $((scale_end - scale_start))s"

echo "[${my_node}] waiting 30s for scale-up to stabilize"
sleep 30

run_bench "phase2_scaled" "${TARGET_DP_SIZE}"

stop_vllm
rm -f "${SIGNAL_FILE}"
echo "[${my_node}] Phase 2 done"

# ────────────────────────────────────────────────────
# Phase 3: 40-GPU static (clean EPLB, 24 redundant)
# ────────────────────────────────────────────────────
echo ""
echo "========== PHASE 3: ${TARGET_DP_SIZE}-GPU static (24 redundant experts) =========="
NUM_REDUNDANT_EXPERTS=24
restart_ray

launch_vllm "${RUN_DIR}/vllm_phase3_${TARGET_DP_SIZE}gpu.log" "${TARGET_DP_SIZE}"
wait_vllm_ready "${RUN_DIR}/vllm_phase3_${TARGET_DP_SIZE}gpu.log"

run_bench "phase3" "${TARGET_DP_SIZE}"

stop_vllm
save_ray_logs
ray stop -f 2>&1 || true
echo "[${my_node}] Phase 3 done"

echo ""
echo "========== All phases complete =========="
echo "[${my_node}] benchmark finished, shutting down"
touch "${RUN_DIR}/job_done.signal"
exit 0
