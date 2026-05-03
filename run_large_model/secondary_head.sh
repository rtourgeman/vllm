#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"

export VLLM_HOST_IP="$(get_routable_ip)"
export RAY_ADDRESS="${SECONDARY_HEAD_IP}:${RAY_PORT_B}"

my_node="$(hostname -s)"
vllm_pid=""

cleanup() {
    if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" 2>&1; then
        echo "[${my_node}][B] stopping vLLM pid=${vllm_pid}"
        kill "${vllm_pid}" 2>&1 || true
        sleep 5
        kill -9 "${vllm_pid}" 2>&1 || true
    fi
    ray stop -f 2>&1 || true
}
trap cleanup EXIT

cd "${VLLM_WORKDIR}"

SECONDARY_NODES_COUNT=$((TARGET_NODES - INITIAL_NODES))

for ((attempt=1; attempt<=5; attempt++)); do
    ray stop -f 2>&1 || true
    if ray start --head \
        --port="${RAY_PORT_B}" \
        --node-ip-address="${SECONDARY_HEAD_IP}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --metrics-export-port=9091; then
        break
    fi
    echo "[${my_node}][B] Ray head attempt ${attempt}/5 failed, retrying"
    sleep 5
done

echo "[${my_node}][B] waiting for ${SECONDARY_NODES_COUNT} Ray node(s)"
wait_for_ray_nodes "${SECONDARY_NODES_COUNT}" "${RAY_WAIT_TIMEOUT}" || true
ray status || true

echo "[${my_node}][B] launching vLLM (secondary), log=${RUN_DIR}/vllm_server_B.log"
DATA_PARALLEL_SIZE="${SECONDARY_DP_SIZE}" \
DATA_PARALLEL_SIZE_LOCAL="${GPUS_PER_NODE}" \
DATA_PARALLEL_ADDRESS="${SECONDARY_HEAD_IP}" \
DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT_B}" \
VLLM_NIXL_EP_MAX_NUM_RANKS="${SECONDARY_DP_SIZE}" \
PORT="${PORT_B}" \
    bash "${SCRIPT_DIR}/serve.sh" >"${RUN_DIR}/vllm_server_B.log" 2>&1 &
vllm_pid="$!"

VLLM_LOG="${RUN_DIR}/vllm_server_B.log"
echo "[${my_node}][B] waiting for vLLM to start"
waited=0
while ! grep -q "Application startup complete" "${VLLM_LOG}" 2>/dev/null; do
    if ! kill -0 "${vllm_pid}" >/dev/null 2>&1; then
        echo "[${my_node}][B] vLLM exited before healthy" >&2
        tail -20 "${VLLM_LOG}" >&2 || true
        wait "${vllm_pid}" || true
        exit 1
    fi
    if (( waited >= SERVER_WAIT_TIMEOUT )); then
        echo "[${my_node}][B] timed out waiting for vLLM" >&2
        tail -20 "${VLLM_LOG}" >&2 || true
        exit 1
    fi
    sleep 15
    waited=$((waited + 15))
    echo "[${my_node}][B] still waiting for vLLM startup... ${waited}s"
done
echo "[${my_node}][B] vLLM is running"

echo "[${my_node}][B] running small benchmark (${SECONDARY_NUM_PROMPTS} prompts)"
NUM_PROMPTS="${SECONDARY_NUM_PROMPTS}" \
MAX_CONCURRENCY="${SECONDARY_NUM_PROMPTS}" \
BENCH_HOST="localhost" \
PORT="${PORT_B}" \
WAIT_FOR_SERVER="false" \
BENCH_LOG_FILE="${RUN_DIR}/benchmark_secondary_${SECONDARY_DP_SIZE}gpu.log" \
    bash "${SCRIPT_DIR}/bench.sh" || true

echo "[${my_node}][B] secondary benchmark done, waiting for scale-up signal"
waited=0
while [[ ! -f "${SIGNAL_FILE}" ]]; do
    if (( waited >= SERVER_WAIT_TIMEOUT )); then
        echo "[${my_node}][B] timed out waiting for scale-up signal" >&2
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done

echo "[${my_node}][B] signal received, tearing down secondary vLLM + Ray"
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

echo "[${my_node}][B] joining primary Ray cluster at ${HEAD_NODE_IP}:${RAY_PORT}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
join_ray_with_retry "${HEAD_NODE_IP}:${RAY_PORT}" "${GPUS_PER_NODE}"
