#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"

export VLLM_HOST_IP="$(get_routable_ip)"

my_node="$(hostname -s)"
ROLE="${ROLE:-primary_worker}"
DONE_FILE="${RUN_DIR}/job_done.signal"
trap "ray stop -f 2>&1 || true; pkill -9 -f 'raylet|runtime_env_agent' 2>&1 || true" EXIT

cd "${VLLM_WORKDIR}"
ray stop -f 2>&1 || true

warm_lustre_cache "${MODEL_NAME}"

if [[ "${ROLE}" == "primary_worker" ]]; then
    export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

    while true; do
        if [[ -f "${DONE_FILE}" ]]; then
            echo "[${my_node}] job done, exiting"
            break
        fi

        echo "[${my_node}] joining primary Ray cluster"
        join_ray_with_retry "${HEAD_NODE_IP}:${RAY_PORT}" "${GPUS_PER_NODE}"

        if [[ -f "${DONE_FILE}" ]]; then
            echo "[${my_node}] job done, exiting"
            break
        fi

        echo "[${my_node}] Ray exited, cleaning up for next phase"
        ray stop -f 2>&1 || true
        pkill -9 -f "raylet|runtime_env_agent" 2>&1 || true
        rm -rf /data/tmp/ray/session_* 2>/dev/null || true
        sleep 5
    done
else
    echo "[${my_node}][B] joining secondary Ray cluster (non-blocking)"
    sleep 15
    ray start --address="${SECONDARY_HEAD_IP}:${RAY_PORT_B}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --min-worker-port=20000 --max-worker-port=29999 \
        --metrics-export-port=9091 \
        --dashboard-agent-grpc-port=9094 \
        --runtime-env-agent-port=9095

    echo "[${my_node}][B] joined secondary Ray, waiting for scale-up signal"
    while [[ ! -f "${SIGNAL_FILE}" ]]; do
        sleep 5
    done

    echo "[${my_node}][B] signal received, killing all Ray processes"
    ray stop -f 2>&1 || true
    sleep 3
    pkill -9 -f "raylet" 2>&1 || true
    pkill -9 -f "runtime_env_agent" 2>&1 || true
    sleep 10

    echo "[${my_node}][B] joining primary Ray cluster at ${HEAD_NODE_IP}:${RAY_PORT}"
    export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
    join_ray_with_retry "${HEAD_NODE_IP}:${RAY_PORT}" "${GPUS_PER_NODE}"
fi
