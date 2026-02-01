#!/usr/bin/env bash
#===============================================================================
# submit_vllm_deepseek.sh
# Convenience wrapper to submit vLLM DeepSeek V3 batch job
#===============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------- Defaults --------
DEFAULT_NODES=3
DEFAULT_PARTITION="batch"
DEFAULT_ACCOUNT="coreai_comparch_trtllm"
DEFAULT_TIME="04:00:00"
DEFAULT_IMAGE="/lustre/fsw/coreai_comparch_trtllm/rtourgeman/elastic-ep/vllm/27_1_26_deep_seek_perf_nixl_weight_transfer_v4.sqsh"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Submit vLLM DeepSeek V3 multi-node batch job

Options:
    -N NODES       Number of nodes (default: ${DEFAULT_NODES})
    -p PARTITION   Slurm partition (default: ${DEFAULT_PARTITION})
    -A ACCOUNT     Slurm account (default: ${DEFAULT_ACCOUNT})
    -t TIME        Time limit (default: ${DEFAULT_TIME})
    -i IMAGE       Container image (default: uses script default)
    -h             Show this help message

Examples:
    # Submit with defaults (3 nodes, 4 hours)
    $0
    
    # Submit with 6 hours time limit
    $0 -t 06:00:00
    
    # Submit to a different partition
    $0 -p interactive_long
EOF
}

NODES="${DEFAULT_NODES}"
PARTITION="${DEFAULT_PARTITION}"
ACCOUNT="${DEFAULT_ACCOUNT}"
TIME="${DEFAULT_TIME}"
IMAGE="${DEFAULT_IMAGE}"

while getopts ":N:p:A:t:i:h" opt; do
    case "${opt}" in
        N) NODES="${OPTARG}" ;;
        p) PARTITION="${OPTARG}" ;;
        A) ACCOUNT="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        i) IMAGE="${OPTARG}" ;;
        h) usage; exit 0 ;;
        :) echo "Option -${OPTARG} requires an argument." >&2; usage; exit 2 ;;
        \?) echo "Invalid option: -${OPTARG}" >&2; usage; exit 2 ;;
    esac
done

# Create logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/vllm_deepseek_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Submitting vLLM DeepSeek V3 Job"
echo "============================================================"
echo "Nodes:      ${NODES}"
echo "Partition:  ${PARTITION}"
echo "Account:    ${ACCOUNT}"
echo "Time:       ${TIME}"
echo "Image:      ${IMAGE}"
echo "Log Dir:    ${LOG_DIR}"
echo "============================================================"

# Submit the job
JOB_ID=$(sbatch \
    --nodes="${NODES}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --output="${LOG_DIR}/vllm_%j.out" \
    --error="${LOG_DIR}/vllm_%j.err" \
    --parsable \
    "${SCRIPT_DIR}/vllm_deepseek_v3.slurm")

echo ""
echo "Job submitted successfully!"
echo "Job ID: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/vllm_${JOB_ID}.out"
echo ""
echo "Cancel with:"
echo "  scancel ${JOB_ID}"
