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
        break
    fi
    echo "[${my_node}] ${TAG}Ray head attempt ${attempt}/5 failed, retrying"
    sleep 5
done

echo "[${my_node}] ${TAG}waiting for ${MY_WAIT_NODES} Ray node(s)"
wait_for_ray_nodes "${MY_WAIT_NODES}" "${RAY_WAIT_TIMEOUT}" || true
ray status || true

if [[ "${ROLE}" == "secondary_head" ]]; then
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
else
    warm_lustre_cache "${MODEL_NAME}"
    echo "[${my_node}] ${TAG}launching vLLM, log=${MY_VLLM_LOG}"
    bash "${SCRIPT_DIR}/serve.sh" >"${MY_VLLM_LOG}" 2>&1 &
fi
vllm_pid="$!"

echo "[${my_node}] ${TAG}waiting for vLLM to start"
waited=0
while ! grep -q "Application startup complete" "${MY_VLLM_LOG}" 2>/dev/null; do
    if ! kill -0 "${vllm_pid}" >/dev/null 2>&1; then
        echo "[${my_node}] ${TAG}vLLM exited before healthy" >&2
        tail -20 "${MY_VLLM_LOG}" >&2 || true
        wait "${vllm_pid}" || true
        exit 1
    fi
    if (( waited >= SERVER_WAIT_TIMEOUT )); then
        echo "[${my_node}] ${TAG}timed out waiting for vLLM" >&2
        tail -20 "${MY_VLLM_LOG}" >&2 || true
        exit 1
    fi
    sleep 15
    waited=$((waited + 15))
    echo "[${my_node}] ${TAG}still waiting for vLLM startup... ${waited}s"
done
echo "[${my_node}] ${TAG}vLLM is running"

if [[ "${ROLE}" == "primary_head" ]]; then
    if [[ "${RUN_BASELINE_BENCH}" == "true" ]]; then
        echo "[${my_node}] running baseline benchmark at ${INITIAL_DP_SIZE} GPUs"
        BENCH_LOG_FILE="${RUN_DIR}/benchmark_initial_${INITIAL_DP_SIZE}gpu.log" \
            bash "${SCRIPT_DIR}/bench.sh"
    fi

    if [[ "${RUN_ELASTIC_SCALE}" == "true" ]]; then
        echo "[${my_node}] signaling additional nodes to join Ray"
        touch "${SIGNAL_FILE}"

        echo "[${my_node}] waiting for ${TARGET_NODES} Ray node(s) before scale-up"
        if ! wait_for_ray_nodes "${TARGET_NODES}" "${RAY_WAIT_TIMEOUT}"; then
            echo "[${my_node}] scale-up aborted: not enough Ray nodes" >&2
            exit 1
        fi

        ray status || true
        echo "[${my_node}] scaling vLLM to ${TARGET_DP_SIZE} GPUs"
        python3 examples/online_serving/elastic_ep/scale.py \
            --host "localhost" \
            --port "${PORT}" \
            --new-dp-size "${TARGET_DP_SIZE}" \
            --num-redundant-experts 24

        echo "[${my_node}] waiting 30s for scale-up to stabilize"
        sleep 30
    fi

    echo "[${my_node}] running final benchmark at ${TARGET_DP_SIZE} GPUs"
    BENCH_LOG_FILE="${RUN_DIR}/benchmark_final_${TARGET_DP_SIZE}gpu.log" \
        bash "${SCRIPT_DIR}/bench.sh"
    echo "[${my_node}] benchmark finished, shutting down"
    kill "${vllm_pid}" 2>&1 || true
    vllm_pid=""
    sleep 3
    ray stop -f 2>&1 || true

else
    echo "[${my_node}] ${TAG}running small benchmark (${SECONDARY_NUM_PROMPTS} prompts, model=${SECONDARY_MODEL_NAME})"
    MODEL_NAME="${SECONDARY_MODEL_NAME}" \
    NUM_PROMPTS="${SECONDARY_NUM_PROMPTS}" \
    MAX_CONCURRENCY="${SECONDARY_NUM_PROMPTS}" \
    BENCH_HOST="localhost" \
    PORT="${PORT_B}" \
    WAIT_FOR_SERVER="false" \
    BENCH_LOG_FILE="${RUN_DIR}/benchmark_secondary_${SECONDARY_DP_SIZE}gpu.log" \
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
fi
