#!/usr/bin/env bash
# Static benchmark: serve DeepSeek V3 on N nodes and run the full benchmark suite.
# No elastic scale-up.
#
# Usage:
#   ./run_bench.sh -N 4              # 32 GPUs
#   ./run_bench.sh -N 5 -r 24       # 40 GPUs, 24 redundant experts
#   ./run_bench.sh -N 4 -t 02:00:00
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

NODES=4
REDUNDANT=0

usage() {
    cat <<EOF
Usage: $0 -N NODES [-r REDUNDANT] [-t TIME]

Required:
  -N NODES        Number of 8xH100 nodes

Optional:
  -r REDUNDANT    Redundant experts (default: ${REDUNDANT})
  -t TIME         Slurm time limit (default: ${TIME})
  -h              Show this help

Benchmark suite (fixed):
  warmup:  8192 prompts, unlimited concurrency
  bench1:  8192 prompts, concurrency 1024
  bench2:  4096 prompts, concurrency 512
  bench3:  2048 prompts, concurrency 256
  bench4:  1024 prompts, concurrency 128
  bench5:   512 prompts, concurrency 64
EOF
}

while getopts ":N:r:t:h" opt; do
    case "${opt}" in
        N) NODES="${OPTARG}" ;;
        r) REDUNDANT="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        h) usage; exit 0 ;;
        :) echo "ERROR: -${OPTARG} requires an argument" >&2; exit 2 ;;
        \?) echo "ERROR: invalid option -${OPTARG}" >&2; exit 2 ;;
    esac
done

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y-%m-%d)"
mkdir -p "${LOG_DIR}"
JOB_TAG="${TOTAL_GPUS}gpu_r${REDUNDANT}"

cat <<EOF
Submitting static benchmark:
  nodes:      ${NODES} (${TOTAL_GPUS} GPUs)
  redundant:  ${REDUNDANT}
  time:       ${TIME}
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
    --export=ALL,SCRIPT_DIR="${SCRIPT_DIR}",TOTAL_GPUS="${TOTAL_GPUS}",REDUNDANT="${REDUNDANT}" \
    "${SCRIPT_DIR}/slurm_static.sh"
)"

cat <<EOF

Submitted job ${JOB_ID}
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/bench_${JOB_TAG}_${JOB_ID}.out
EOF
