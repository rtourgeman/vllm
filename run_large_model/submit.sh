#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

NODES="${NODES:-4}"
NODES_SET="false"
PARTITION="${BATCH_PARTITION}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${IMAGE}}"
INITIAL_DP_SIZE=""
TARGET_DP_SIZE=""
RUN_BASELINE_BENCH="auto"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] 

Submit a Slurm batch job that serves DeepSeek V3 and runs one benchmark.

Slurm:
  -N NODES        Number of 8xH100 nodes (default: ${NODES})
  -p PARTITION    Slurm partition (default: ${PARTITION})
  -A ACCOUNT      Slurm account (default: ${ACCOUNT})
  -t TIME         Time limit (default: ${TIME})

Runtime:
  -C IMAGE        Container image (default: ${CONTAINER_IMAGE})
  -m MODEL        Model path or HF name (default: ${MODEL_NAME})
  -P PORT         vLLM HTTP port (default: ${PORT})
  -l MAX_LEN      Max model length (default: ${MAX_MODEL_LEN})
  -g GPU_UTIL     GPU memory utilization (default: ${GPU_MEMORY_UTILIZATION})
  -r REDUNDANT    Redundant experts for EPLB (default: ${NUM_REDUNDANT_EXPERTS})

Elastic scale-up:
  -x GPUS         Initial DP size (default: all allocated GPUs)
  -y GPUS         Target DP size after scale-up (default: all allocated GPUs)

Benchmark:
  -n PROMPTS      Number of prompts (default: ${NUM_PROMPTS})
  -c CONCURRENCY  Max concurrency (default: ${MAX_CONCURRENCY})
  -i INPUT_LEN    Random input tokens (default: ${RANDOM_INPUT_LEN})
  -o OUTPUT_LEN   Random output tokens (default: ${RANDOM_OUTPUT_LEN})
  -B              Skip baseline benchmark before scale-up

Examples:
  $0 -N 4
  $0 -x 32 -y 64
  $0 -N 4 -n 1024 -c 256 -i 1024 -o 1024
EOF
}

while getopts ":N:p:A:t:C:m:P:l:g:r:x:y:n:c:i:o:Bh" opt; do
    case "${opt}" in
        N) NODES="${OPTARG}"; NODES_SET="true" ;;
        p) PARTITION="${OPTARG}" ;;
        A) ACCOUNT="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        C) CONTAINER_IMAGE="${OPTARG}" ;;
        m) MODEL_NAME="${OPTARG}" ;;
        P) PORT="${OPTARG}" ;;
        l) MAX_MODEL_LEN="${OPTARG}" ;;
        g) GPU_MEMORY_UTILIZATION="${OPTARG}" ;;
        r) NUM_REDUNDANT_EXPERTS="${OPTARG}" ;;
        x) INITIAL_DP_SIZE="${OPTARG}" ;;
        y) TARGET_DP_SIZE="${OPTARG}" ;;
        n) NUM_PROMPTS="${OPTARG}" ;;
        c) MAX_CONCURRENCY="${OPTARG}" ;;
        i) RANDOM_INPUT_LEN="${OPTARG}" ;;
        o) RANDOM_OUTPUT_LEN="${OPTARG}" ;;
        B) RUN_BASELINE_BENCH="false" ;;
        h) usage; exit 0 ;;
        :) echo "ERROR: -${OPTARG} requires an argument" >&2; exit 2 ;;
        \?) echo "ERROR: invalid option -${OPTARG}" >&2; exit 2 ;;
    esac
done

if [[ -z "${TARGET_DP_SIZE}" ]]; then
    TARGET_DP_SIZE=$((NODES * GPUS_PER_NODE))
elif [[ "${NODES_SET}" == "false" ]]; then
    NODES=$(((TARGET_DP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
fi
if [[ -z "${INITIAL_DP_SIZE}" ]]; then
    INITIAL_DP_SIZE="${TARGET_DP_SIZE}"
fi

for var in NODES PORT MAX_MODEL_LEN NUM_REDUNDANT_EXPERTS NUM_PROMPTS \
           MAX_CONCURRENCY RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN \
           INITIAL_DP_SIZE TARGET_DP_SIZE; do
    if ! [[ "${!var}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${var} must be a positive integer (got '${!var}')" >&2
        exit 1
    fi
done

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
NUM_ROUTED_EXPERTS=256
TOTAL_EXPERTS=$((NUM_ROUTED_EXPERTS + NUM_REDUNDANT_EXPERTS))
RUN_ELASTIC_SCALE="false"
if (( INITIAL_DP_SIZE < TARGET_DP_SIZE )); then
    RUN_ELASTIC_SCALE="true"
fi
if [[ "${RUN_BASELINE_BENCH}" == "auto" ]]; then
    RUN_BASELINE_BENCH="${RUN_ELASTIC_SCALE}"
fi

(( TARGET_DP_SIZE <= TOTAL_GPUS )) || { echo "ERROR: target DP (${TARGET_DP_SIZE}) > allocated GPUs (${TOTAL_GPUS})" >&2; exit 1; }
(( INITIAL_DP_SIZE <= TARGET_DP_SIZE )) || { echo "ERROR: initial DP > target DP" >&2; exit 1; }

LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y-%m-%d)"
mkdir -p "${LOG_DIR}"

if [[ "${RUN_ELASTIC_SCALE}" == "true" ]]; then
    JOB_NAME="dsv3_${INITIAL_DP_SIZE}to${TARGET_DP_SIZE}_c${MAX_CONCURRENCY}"
else
    JOB_NAME="dsv3_${TARGET_DP_SIZE}gpu_c${MAX_CONCURRENCY}"
fi
LOG_PREFIX="${JOB_NAME}_np${NUM_PROMPTS}_i${RANDOM_INPUT_LEN}_o${RANDOM_OUTPUT_LEN}"

SBATCH_ARGS=(
    --nodes="${NODES}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --ntasks-per-node=1
    --exclusive
    --partition="${PARTITION}"
    --account="${ACCOUNT}"
    --time="${TIME}"
    --job-name="${JOB_NAME}"
    --output="${LOG_DIR}/${LOG_PREFIX}_%j.out"
    --error="${LOG_DIR}/${LOG_PREFIX}_%j.err"
    --parsable
    --export=ALL,SCRIPT_DIR="${SCRIPT_DIR}",CONTAINER_IMAGE="${CONTAINER_IMAGE}",MODEL_NAME="${MODEL_NAME}",PORT="${PORT}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",NUM_REDUNDANT_EXPERTS="${NUM_REDUNDANT_EXPERTS}",INITIAL_DP_SIZE="${INITIAL_DP_SIZE}",TARGET_DP_SIZE="${TARGET_DP_SIZE}",RUN_ELASTIC_SCALE="${RUN_ELASTIC_SCALE}",RUN_BASELINE_BENCH="${RUN_BASELINE_BENCH}",NUM_PROMPTS="${NUM_PROMPTS}",MAX_CONCURRENCY="${MAX_CONCURRENCY}",RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN}",RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN}"
)

cat <<EOF
Submitting DeepSeek V3 benchmark:
  nodes:        ${NODES} (${TOTAL_GPUS} GPUs)
  dp:           ${INITIAL_DP_SIZE} -> ${TARGET_DP_SIZE}
  elastic:      ${RUN_ELASTIC_SCALE}
  partition:    ${PARTITION}
  account:      ${ACCOUNT}
  time:         ${TIME}
  image:        ${CONTAINER_IMAGE}
  model:        ${MODEL_NAME}
  prompts:      ${NUM_PROMPTS}
  concurrency:  ${MAX_CONCURRENCY}
  input/output: ${RANDOM_INPUT_LEN}/${RANDOM_OUTPUT_LEN}
  slurm logs:   ${LOG_DIR}
EOF

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/batch.slurm")"

cat <<EOF

Submitted job ${JOB_ID}
Monitor:
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/${LOG_PREFIX}_${JOB_ID}.out

Run logs:
  ${SCRIPT_DIR}/logs/${JOB_ID}/
EOF
