#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"

export VLLM_HOST_IP="$(get_routable_ip)"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

my_node="$(hostname -s)"
vllm_pid=""

cleanup() {
    if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" >/dev/null 2>&1; then
        echo "[${my_node}] stopping vLLM pid=${vllm_pid}"
        kill "${vllm_pid}" >/dev/null 2>&1 || true
        sleep 5
        kill -9 "${vllm_pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${SIGNAL_FILE}" >/dev/null 2>&1 || true
    ray stop -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "${VLLM_WORKDIR}"

ray stop -f >/dev/null 2>&1 || true
ray start --head \
    --port="${RAY_PORT}" \
    --node-ip-address="${HEAD_NODE_IP}" \
    --num-gpus="${GPUS_PER_NODE}"

echo "[${my_node}] waiting for ${INITIAL_NODES} initial Ray node(s)"
wait_for_ray_nodes "${INITIAL_NODES}" "${RAY_WAIT_TIMEOUT}" || true
ray status || true

echo "[${my_node}] launching vLLM, log=${RUN_DIR}/vllm_server.log"
bash "${SCRIPT_DIR}/serve.sh" >"${RUN_DIR}/vllm_server.log" 2>&1 &
vllm_pid="$!"

echo "[${my_node}] waiting for vLLM health"
waited=0
until check_health "localhost" "${PORT}"; do
    if ! kill -0 "${vllm_pid}" >/dev/null 2>&1; then
        echo "[${my_node}] vLLM exited before becoming healthy" >&2
        wait "${vllm_pid}" || true
        exit 1
    fi
    if (( waited >= SERVER_WAIT_TIMEOUT )); then
        echo "[${my_node}] timed out waiting for vLLM health" >&2
        exit 1
    fi
    sleep 30
    waited=$((waited + 30))
    echo "[${my_node}] still waiting for vLLM health... ${waited}s"
done
echo "[${my_node}] vLLM is healthy"

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
        --new-dp-size "${TARGET_DP_SIZE}"

    echo "[${my_node}] waiting for health after scale-up"
    if ! wait_for_health "localhost" "${PORT}" "${SERVER_WAIT_TIMEOUT}"; then
        echo "[${my_node}] server unhealthy after scale-up" >&2
        exit 1
    fi
fi

echo "[${my_node}] running final benchmark at ${TARGET_DP_SIZE} GPUs"
BENCH_LOG_FILE="${RUN_DIR}/benchmark_final_${TARGET_DP_SIZE}gpu.log" \
    bash "${SCRIPT_DIR}/bench.sh"
echo "[${my_node}] benchmark finished"
