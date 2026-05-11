#!/usr/bin/env bash
# Static benchmark: serve DeepSeek V3 on N nodes and run a benchmark.
# No elastic scale-up.
#
# Usage:
#   ./run_bench.sh -N 4                           # 32 GPUs, defaults
#   ./run_bench.sh -N 5 -r 24 -n 8192 -c 512     # 40 GPUs, 24 redundant experts
#   ./run_bench.sh -N 4 -n 1024 -c 128 -t 01:30:00
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ── CLI ──
NODES=4
REDUNDANT=0
PROMPTS=8192
CONCURRENCY=256

usage() {
    cat <<EOF
Usage: $0 -N NODES [-r REDUNDANT] [-n PROMPTS] [-c CONCURRENCY] [-t TIME]

Required:
  -N NODES        Number of 8xH100 nodes

Optional:
  -r REDUNDANT    Redundant experts (default: ${REDUNDANT})
  -n PROMPTS      Number of prompts (default: ${PROMPTS})
  -c CONCURRENCY  Max concurrency (default: ${CONCURRENCY})
  -t TIME         Slurm time limit (default: ${TIME})
  -h              Show this help
EOF
}

while getopts ":N:r:n:c:t:h" opt; do
    case "${opt}" in
        N) NODES="${OPTARG}" ;;
        r) REDUNDANT="${OPTARG}" ;;
        n) PROMPTS="${OPTARG}" ;;
        c) CONCURRENCY="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        h) usage; exit 0 ;;
        :) echo "ERROR: -${OPTARG} requires an argument" >&2; exit 2 ;;
        \?) echo "ERROR: invalid option -${OPTARG}" >&2; exit 2 ;;
    esac
done

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y-%m-%d)"
mkdir -p "${LOG_DIR}"
JOB_TAG="${TOTAL_GPUS}gpu_r${REDUNDANT}_c${CONCURRENCY}_np${PROMPTS}"

cat <<EOF
Submitting static benchmark:
  nodes:       ${NODES} (${TOTAL_GPUS} GPUs)
  redundant:   ${REDUNDANT}
  prompts:     ${PROMPTS}
  concurrency: ${CONCURRENCY}
  time:        ${TIME}
EOF

JOB_ID="$(sbatch --parsable \
    --nodes="${NODES}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --ntasks-per-node=1 \
    --exclusive \
    --partition="${BATCH_PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --job-name="bench_${JOB_TAG}" \
    --output="${LOG_DIR}/bench_${JOB_TAG}_%j.out" \
    --error="${LOG_DIR}/bench_${JOB_TAG}_%j.err" \
    --export=ALL,SCRIPT_DIR="${SCRIPT_DIR}",TOTAL_GPUS="${TOTAL_GPUS}",REDUNDANT="${REDUNDANT}",PROMPTS="${PROMPTS}",CONCURRENCY="${CONCURRENCY}" \
    "${SCRIPT_DIR}/slurm_static.sh"
)"

cat <<EOF

Submitted job ${JOB_ID}
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/bench_${JOB_TAG}_${JOB_ID}.out
EOF
