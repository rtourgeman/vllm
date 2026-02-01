#!/usr/bin/env bash
#===============================================================================
# submit_vllm_deepseek.sh
# Submit vLLM DeepSeek V3 multi-node batch job
#===============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------- Defaults --------
DEFAULT_NODES=3
DEFAULT_PARTITION="batch"
DEFAULT_ACCOUNT="coreai_comparch_trtllm"
DEFAULT_TIME="04:00:00"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Submit vLLM DeepSeek V3 multi-node batch job (expert parallel mode)

Options:
    -N NODES       Number of nodes (default: ${DEFAULT_NODES})
    -p PARTITION   Slurm partition (default: ${DEFAULT_PARTITION})
    -A ACCOUNT     Slurm account (default: ${DEFAULT_ACCOUNT})
    -t TIME        Time limit (default: ${DEFAULT_TIME})
    -h             Show this help message

Examples:
    # Submit with defaults (3 nodes, 4 hours)
    $0
    
    # Submit with 6 hours time limit
    $0 -t 06:00:00

Note: For 3 nodes, DATA_PARALLEL_SIZE=24 is used automatically.
      Edit vllm_deepseek_v3.slurm to change model path or other settings.
EOF
}

NODES="${DEFAULT_NODES}"
PARTITION="${DEFAULT_PARTITION}"
ACCOUNT="${DEFAULT_ACCOUNT}"
TIME="${DEFAULT_TIME}"

while getopts ":N:p:A:t:h" opt; do
    case "${opt}" in
        N) NODES="${OPTARG}" ;;
        p) PARTITION="${OPTARG}" ;;
        A) ACCOUNT="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        h) usage; exit 0 ;;
        :) echo "Option -${OPTARG} requires an argument." >&2; usage; exit 2 ;;
        \?) echo "Invalid option: -${OPTARG}" >&2; usage; exit 2 ;;
    esac
done

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

echo "============================================================"
echo "Submitting vLLM DeepSeek V3 Job"
echo "============================================================"
echo "Nodes:          ${NODES}"
echo "Total GPUs:     $((NODES * 8))"
echo "Partition:      ${PARTITION}"
echo "Account:        ${ACCOUNT}"
echo "Time:           ${TIME}"
echo "============================================================"

# Submit the job
JOB_ID=$(sbatch \
    --nodes="${NODES}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --parsable \
    "${SCRIPT_DIR}/vllm_deepseek_v3.slurm")

echo ""
echo "Job submitted successfully!"
echo "Job ID: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${SCRIPT_DIR}/logs/vllm_server_${JOB_ID}.out"
echo ""
echo "View Ray logs:"
echo "  tail -f ${SCRIPT_DIR}/logs/ray_head_${JOB_ID}.out"
echo ""
echo "Cancel with:"
echo "  scancel ${JOB_ID}"
