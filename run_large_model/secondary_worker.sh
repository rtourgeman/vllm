#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"

export VLLM_HOST_IP="$(get_routable_ip)"

my_node="$(hostname -s)"
trap "ray stop -f 2>&1 || true; pkill -9 -f 'raylet|runtime_env_agent|gcs_server' 2>&1 || true" EXIT

cd "${VLLM_WORKDIR}"

echo "[${my_node}][B] joining secondary Ray cluster (non-blocking)"
sleep 15
ray stop -f 2>&1 || true
sleep 2
ray start --address="${SECONDARY_HEAD_IP}:${RAY_PORT_B}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --metrics-export-port=9091

echo "[${my_node}][B] joined secondary Ray, waiting for scale-up signal"
while [[ ! -f "${SIGNAL_FILE}" ]]; do
    sleep 5
done

echo "[${my_node}][B] signal received, killing all Ray processes"
ray stop -f 2>&1 || true
sleep 3
pkill -9 -f "raylet" 2>&1 || true
pkill -9 -f "runtime_env_agent" 2>&1 || true
pkill -9 -f "gcs_client" 2>&1 || true
sleep 10

echo "[${my_node}][B] joining primary Ray cluster at ${HEAD_NODE_IP}:${RAY_PORT}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
join_ray_with_retry "${HEAD_NODE_IP}:${RAY_PORT}" "${GPUS_PER_NODE}"
