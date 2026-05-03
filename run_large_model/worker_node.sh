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

read -ra all_nodes <<< "${SLURM_NODES_STR}"
my_node_index=-1
for i in "${!all_nodes[@]}"; do
    if [[ "${all_nodes[$i]}" == "${my_node}" ]]; then
        my_node_index="${i}"
        break
    fi
done

if (( my_node_index < 0 )); then
    echo "[${my_node}] could not find self in Slurm node list" >&2
    exit 1
fi

if [[ "${RUN_ELASTIC_SCALE}" == "true" ]] && (( my_node_index >= INITIAL_NODES )); then
    echo "[${my_node}] waiting for elastic scale-up signal"
    waited=0
    while [[ ! -f "${SIGNAL_FILE}" ]]; do
        if (( waited >= SERVER_WAIT_TIMEOUT )); then
            echo "[${my_node}] timed out waiting for scale-up signal" >&2
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
        if (( waited % 100 == 0 )); then
            echo "[${my_node}] still waiting for scale-up signal... ${waited}s"
        fi
   
    done
fi

echo "[${my_node}] joining Ray cluster"
join_ray_with_retry "${HEAD_NODE_IP}:${RAY_PORT}" "${GPUS_PER_NODE}"
