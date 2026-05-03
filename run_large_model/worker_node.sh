#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"

export VLLM_HOST_IP="$(get_routable_ip)"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

my_node="$(hostname -s)"
trap "ray stop -f 2>&1 || true" EXIT

cd "${VLLM_WORKDIR}"
ray stop -f 2>&1 || true

echo "[${my_node}] joining primary Ray cluster"
join_ray_with_retry "${HEAD_NODE_IP}:${RAY_PORT}" "${GPUS_PER_NODE}"
