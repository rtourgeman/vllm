#!/usr/bin/env bash
# Scale-up benchmark: serve DeepSeek V3, scale from X GPUs to Y GPUs, then benchmark.
#
# Usage:
#   ./run_bench_scaleup.sh -x 32 -y 40 -R 24 -n 8192 -c 512
#   ./run_bench_scaleup.sh -x 32 -y 40 -r 0 -R 24 -n 1024 -c 256
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ── CLI ──
INITIAL_GPUS=32
TARGET_GPUS=40
REDUNDANT=0
SCALE_REDUNDANT=24
PROMPTS=8192
CONCURRENCY=256

usage() {
    cat <<EOF
Usage: $0 -x INITIAL_GPUS -y TARGET_GPUS [-r REDUNDANT] [-R SCALE_REDUNDANT] [-n PROMPTS] [-c CONCURRENCY] [-t TIME]

Required:
  -x INITIAL_GPUS     GPU count at startup
  -y TARGET_GPUS      GPU count after scale-up

Optional:
  -r REDUNDANT        Redundant experts at initial serve (default: ${REDUNDANT})
  -R SCALE_REDUNDANT  Redundant experts in scale-up request (default: ${SCALE_REDUNDANT})
  -n PROMPTS          Number of prompts (default: ${PROMPTS})
  -c CONCURRENCY      Max concurrency (default: ${CONCURRENCY})
  -t TIME             Slurm time limit (default: ${TIME})
  -h                  Show this help
EOF
}

while getopts ":x:y:r:R:n:c:t:h" opt; do
    case "${opt}" in
        x) INITIAL_GPUS="${OPTARG}" ;;
        y) TARGET_GPUS="${OPTARG}" ;;
        r) REDUNDANT="${OPTARG}" ;;
        R) SCALE_REDUNDANT="${OPTARG}" ;;
        n) PROMPTS="${OPTARG}" ;;
        c) CONCURRENCY="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        h) usage; exit 0 ;;
        :) echo "ERROR: -${OPTARG} requires an argument" >&2; exit 2 ;;
        \?) echo "ERROR: invalid option -${OPTARG}" >&2; exit 2 ;;
    esac
done

INITIAL_NODES=$(( (INITIAL_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
NODES=$(( (TARGET_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))

(( INITIAL_GPUS < TARGET_GPUS )) || { echo "ERROR: initial (${INITIAL_GPUS}) must be < target (${TARGET_GPUS})" >&2; exit 1; }
(( TARGET_GPUS % GPUS_PER_NODE == 0 )) || { echo "ERROR: TARGET_GPUS (${TARGET_GPUS}) must be divisible by ${GPUS_PER_NODE}" >&2; exit 1; }
(( INITIAL_GPUS % GPUS_PER_NODE == 0 )) || { echo "ERROR: INITIAL_GPUS (${INITIAL_GPUS}) must be divisible by ${GPUS_PER_NODE}" >&2; exit 1; }

LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y-%m-%d)"
mkdir -p "${LOG_DIR}"
JOB_TAG="${INITIAL_GPUS}to${TARGET_GPUS}gpu_r${SCALE_REDUNDANT}_c${CONCURRENCY}_np${PROMPTS}"

cat <<EOF
Submitting scale-up benchmark:
  nodes:            ${NODES} ($((NODES * GPUS_PER_NODE)) GPUs)
  initial:          ${INITIAL_GPUS} GPUs (${INITIAL_NODES} nodes)
  target:           ${TARGET_GPUS} GPUs
  redundant:        ${REDUNDANT} (initial) -> ${SCALE_REDUNDANT} (after scale)
  prompts:          ${PROMPTS}
  concurrency:      ${CONCURRENCY}
  time:             ${TIME}
EOF

JOB_ID="$(sbatch --parsable \
    --nodes="${NODES}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --ntasks-per-node=1 \
    --exclusive \
    --partition="${BATCH_PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --job-name="scale_${JOB_TAG}" \
    --output="${LOG_DIR}/scale_${JOB_TAG}_%j.out" \
    --error="${LOG_DIR}/scale_${JOB_TAG}_%j.err" \
    --export=ALL,SCRIPT_DIR="${SCRIPT_DIR}",INITIAL_GPUS="${INITIAL_GPUS}",INITIAL_NODES="${INITIAL_NODES}",TARGET_GPUS="${TARGET_GPUS}",REDUNDANT="${REDUNDANT}",SCALE_REDUNDANT="${SCALE_REDUNDANT}",PROMPTS="${PROMPTS}",CONCURRENCY="${CONCURRENCY}" \
    "${SCRIPT_DIR}/slurm_scaleup.sh"
)"

cat <<EOF

Submitted job ${JOB_ID}
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/scale_${JOB_TAG}_${JOB_ID}.out
EOF
