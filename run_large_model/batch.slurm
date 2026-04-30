#!/usr/bin/env bash
#SBATCH --job-name=deepseek_v3_bench
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --account=network_research_advdev
#SBATCH --time=04:00:00
#SBATCH --output=deepseek_v3_%j.out
#SBATCH --error=deepseek_v3_%j.err

set -Eeuo pipefail

if [[ -z "${SCRIPT_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/config.sh" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    else
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
fi

source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/helpers.sh"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${IMAGE}}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${MOUNT_SRC}:${MOUNT_DST},${SCRIPT_DIR}:${SCRIPT_DIR}}"

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/logs}"
RUN_DIR="${RUN_ROOT}/${SLURM_JOB_ID}"

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
TARGET_DP_SIZE="${TARGET_DP_SIZE:-${TOTAL_GPUS}}"
INITIAL_DP_SIZE="${INITIAL_DP_SIZE:-${TARGET_DP_SIZE}}"

RUN_ELASTIC_SCALE="${RUN_ELASTIC_SCALE:-auto}"
if [[ "${RUN_ELASTIC_SCALE}" == "auto" ]]; then
    if (( INITIAL_DP_SIZE < TARGET_DP_SIZE )); then
        RUN_ELASTIC_SCALE="true"
    else
        RUN_ELASTIC_SCALE="false"
    fi
fi

DATA_PARALLEL_SIZE="${INITIAL_DP_SIZE}"
DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-${GPUS_PER_NODE}}"
VLLM_NIXL_EP_MAX_NUM_RANKS="${VLLM_NIXL_EP_MAX_NUM_RANKS:-${TARGET_DP_SIZE}}"
INITIAL_NODES=$(((INITIAL_DP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
TARGET_NODES=$(((TARGET_DP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))

RUN_BASELINE_BENCH="${RUN_BASELINE_BENCH:-${RUN_ELASTIC_SCALE}}"

RAY_PORT="${RAY_PORT:-$((6379 + SLURM_JOB_ID % 1000))}"
DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT:-$((9876 + SLURM_JOB_ID % 1000))}"

NUM_ROUTED_EXPERTS=256
TOTAL_EXPERTS=$((NUM_ROUTED_EXPERTS + NUM_REDUNDANT_EXPERTS))

if (( TARGET_DP_SIZE > TOTAL_GPUS )); then
    echo "ERROR: TARGET_DP_SIZE (${TARGET_DP_SIZE}) > allocated GPUs (${TOTAL_GPUS})" >&2
    exit 1
fi
if (( INITIAL_DP_SIZE > TARGET_DP_SIZE )); then
    echo "ERROR: INITIAL_DP_SIZE (${INITIAL_DP_SIZE}) > TARGET_DP_SIZE (${TARGET_DP_SIZE})" >&2
    exit 1
fi
if (( INITIAL_DP_SIZE % GPUS_PER_NODE != 0 || TARGET_DP_SIZE % GPUS_PER_NODE != 0 )); then
    echo "ERROR: DP sizes must use whole ${GPUS_PER_NODE}-GPU nodes (initial=${INITIAL_DP_SIZE}, target=${TARGET_DP_SIZE})" >&2
    exit 1
fi
if (( TOTAL_EXPERTS % INITIAL_DP_SIZE != 0 )); then
    echo "ERROR: experts (${TOTAL_EXPERTS}) not divisible by initial DP (${INITIAL_DP_SIZE})" >&2
    exit 1
fi
if (( TOTAL_EXPERTS % TARGET_DP_SIZE != 0 )); then
    echo "ERROR: experts (${TOTAL_EXPERTS}) not divisible by target DP (${TARGET_DP_SIZE})" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}"
mapfile -t SLURM_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${SLURM_NODES[0]}"
SIGNAL_FILE="${RUN_DIR}/join_additional_nodes.signal"
rm -f "${SIGNAL_FILE}"

HEAD_NODE_IP="$(
    srun --nodes=1 --ntasks=1 --overlap -w "${HEAD_NODE}" \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        bash -lc "hostname -I | tr ' ' '\n' | grep -v '^169\\.254\\.' | head -1"
)"

export SCRIPT_DIR VLLM_WORKDIR RUN_DIR HEAD_NODE HEAD_NODE_IP RAY_PORT
export GPUS_PER_NODE DATA_PARALLEL_SIZE DATA_PARALLEL_SIZE_LOCAL
export DATA_PARALLEL_ADDRESS="${HEAD_NODE_IP}"
export DATA_PARALLEL_RPC_PORT VLLM_NIXL_EP_MAX_NUM_RANKS
export INITIAL_DP_SIZE TARGET_DP_SIZE INITIAL_NODES TARGET_NODES
export RUN_ELASTIC_SCALE RUN_BASELINE_BENCH SIGNAL_FILE
export SLURM_NODES_STR="${SLURM_NODES[*]}"
export MODEL_NAME HOST PORT MAX_MODEL_LEN GPU_MEMORY_UTILIZATION ALL2ALL_BACKEND
export NUM_REDUNDANT_EXPERTS EPLB_WINDOW_SIZE EPLB_STEP_INTERVAL
export NUM_PROMPTS MAX_CONCURRENCY RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN
export SERVER_WAIT_TIMEOUT RAY_WAIT_TIMEOUT
export BENCH_HOST="localhost" WAIT_FOR_SERVER="false"
export ENABLE_ELASTIC_EP ENABLE_EPLB TENSOR_PARALLEL_SIZE

cat <<EOF
============================================================
DeepSeek V3 vLLM batch benchmark
============================================================
job_id:            ${SLURM_JOB_ID}
nodes:             ${SLURM_JOB_NUM_NODES} (${TOTAL_GPUS} GPUs)
head_node:         ${HEAD_NODE} (${HEAD_NODE_IP})
run_dir:           ${RUN_DIR}
container:         ${CONTAINER_IMAGE}
model:             ${MODEL_NAME}
max_model_len:     ${MAX_MODEL_LEN}
gpu_memory_util:   ${GPU_MEMORY_UTILIZATION}
all2all:           ${ALL2ALL_BACKEND}
elastic:           ${RUN_ELASTIC_SCALE}
dp:                ${INITIAL_DP_SIZE} -> ${TARGET_DP_SIZE}
experts_per_gpu:   $((TOTAL_EXPERTS / INITIAL_DP_SIZE)) -> $((TOTAL_EXPERTS / TARGET_DP_SIZE))
benchmark:         ${NUM_PROMPTS} prompts, c=${MAX_CONCURRENCY}, i=${RANDOM_INPUT_LEN}/o=${RANDOM_OUTPUT_LEN}
============================================================
EOF

srun --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc '
        if [[ "$(hostname -s)" == "${HEAD_NODE}" ]]; then
            bash "${SCRIPT_DIR}/head_node.sh"
        else
            bash "${SCRIPT_DIR}/worker_node.sh"
        fi
    '

echo "Job completed. Logs in ${RUN_DIR}"
