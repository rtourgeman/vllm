#!/usr/bin/env bash
#===============================================================================
# vllm_deepseek_v3.slurm
# Slurm batch script for running vLLM with DeepSeek V3 on 3 nodes × 8 GPUs
#===============================================================================

#-------------------------------------------------------------------------------
# SBATCH Directives - Resource Allocation
#-------------------------------------------------------------------------------

# Job identification
#SBATCH --job-name=vllm-deepseek-v3
#SBATCH --comment="vLLM DeepSeek V3 multi-node inference"

# Resource allocation
#SBATCH --nodes=3                      # 3 nodes for full DeepSeek V3
#SBATCH --ntasks-per-node=1            # 1 task per node (vLLM handles GPU distribution internally)
#SBATCH --gpus-per-node=8              # 8 H100 GPUs per node (24 total)
#SBATCH --exclusive                    # Exclusive node access for consistent performance

# Time and queue
#SBATCH --time=04:00:00                # 4 hours wall time (adjust as needed)
#SBATCH --partition=batch              # Use batch partition (interactive only allows ≤2 nodes)
#SBATCH --account=coreai_comparch_trtllm

# Output logs
#SBATCH --output=logs/vllm_%j.out      # stdout (%j = job ID)
#SBATCH --error=logs/vllm_%j.err       # stderr

#-------------------------------------------------------------------------------
# Configuration Variables - EDIT THESE
#-------------------------------------------------------------------------------

# Container image (your existing image)
CONTAINER_IMAGE="/lustre/fsw/coreai_comparch_trtllm/rtourgeman/elastic-ep/vllm/27_1_26_deep_seek_perf_nixl_weight_transfer_v4.sqsh"

# Container mounts - ADD YOUR MOUNTS HERE
# Format: "host_path:container_path,host_path2:container_path2,..."
CONTAINER_MOUNTS="/lustre/fsw/coreai_comparch_trtllm:/lustre/fsw/coreai_comparch_trtllm,/swgwork/rtourgeman:/swgwork/rtourgeman"

# Model configuration
MODEL_PATH="/lustre/fsw/coreai_comparch_trtllm/models/DeepSeek-V3"  # Adjust to your model path
TENSOR_PARALLEL_SIZE=8     # TP=8: parallelize across 8 GPUs within each node
PIPELINE_PARALLEL_SIZE=3   # PP=3: pipeline across 3 nodes

# vLLM server settings
HOST="0.0.0.0"
PORT=8000
MAX_MODEL_LEN=8192         # Adjust based on your memory/needs

#-------------------------------------------------------------------------------
# Script Setup
#-------------------------------------------------------------------------------

set -Eeuo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

echo "============================================================"
echo "vLLM DeepSeek V3 Multi-Node Job"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job Name:      ${SLURM_JOB_NAME}"
echo "Nodes:         ${SLURM_JOB_NUM_NODES}"
echo "GPUs/Node:     8"
echo "Total GPUs:    $((SLURM_JOB_NUM_NODES * 8))"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Node List:     ${SLURM_JOB_NODELIST}"
echo "============================================================"

#-------------------------------------------------------------------------------
# Distributed Environment Setup
#-------------------------------------------------------------------------------

# Get the first node as the head/master node
# SLURM_NODELIST can be in formats like "node[001-003]" or "node001,node002,node003"
HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -i)

# Ray configuration for vLLM distributed inference
RAY_PORT=6379              # Ray cluster port
DASHBOARD_PORT=8265        # Ray dashboard port

echo ""
echo "Distributed Configuration:"
echo "  Head Node:       ${HEAD_NODE}"
echo "  Head Node IP:    ${HEAD_NODE_IP}"
echo "  Ray Port:        ${RAY_PORT}"
echo "  TP Size:         ${TENSOR_PARALLEL_SIZE}"
echo "  PP Size:         ${PIPELINE_PARALLEL_SIZE}"
echo "============================================================"

#-------------------------------------------------------------------------------
# Launch vLLM with Ray Distributed Backend
#-------------------------------------------------------------------------------

# vLLM uses Ray for multi-node distributed inference.
# We launch the head node first, then worker nodes join the Ray cluster.

echo ""
echo "Starting Ray cluster and vLLM..."
echo ""

# Use srun to launch across all nodes
# Each node will:
#   1. Start Ray (head on first node, workers on others)
#   2. The head node runs the vLLM server

srun \
    --nodes=${SLURM_JOB_NUM_NODES} \
    --ntasks-per-node=1 \
    --gpus-per-node=8 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --mpi=pmix \
    bash -c '
        # Determine if this is the head node
        HEAD_NODE_IP='"${HEAD_NODE_IP}"'
        MY_IP=$(hostname -i)
        
        # Export environment for vLLM/Ray
        export RAY_ADDRESS="${HEAD_NODE_IP}:'"${RAY_PORT}"'"
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export NCCL_DEBUG=WARN
        
        # Source environment setup if available
        if [[ -f /workspace/set_env.sh ]]; then
            source /workspace/set_env.sh
        fi
        
        echo "[Node $(hostname)] IP=${MY_IP}, HEAD_NODE_IP=${HEAD_NODE_IP}"
        
        if [[ "${MY_IP}" == "${HEAD_NODE_IP}" ]]; then
            echo "[HEAD NODE] Starting Ray head and vLLM server..."
            
            # Start Ray head node
            ray start --head --port='"${RAY_PORT}"' \
                --dashboard-host=0.0.0.0 \
                --dashboard-port='"${DASHBOARD_PORT}"' \
                --num-gpus=8 \
                --block &
            
            # Wait for Ray to initialize
            sleep 10
            
            # Wait for all worker nodes to join
            EXPECTED_GPUS=$(('"${SLURM_JOB_NUM_NODES}"' * 8))
            echo "Waiting for ${EXPECTED_GPUS} GPUs to be available in Ray cluster..."
            
            for i in {1..60}; do
                AVAILABLE_GPUS=$(ray status 2>/dev/null | grep -oP "GPU:\s+\K[0-9]+" | head -1 || echo "0")
                echo "  Attempt ${i}: ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs available"
                if [[ "${AVAILABLE_GPUS}" -ge "${EXPECTED_GPUS}" ]]; then
                    echo "All GPUs available!"
                    break
                fi
                sleep 5
            done
            
            # Launch vLLM API server
            echo "Launching vLLM server..."
            python -m vllm.entrypoints.openai.api_server \
                --model '"${MODEL_PATH}"' \
                --tensor-parallel-size '"${TENSOR_PARALLEL_SIZE}"' \
                --pipeline-parallel-size '"${PIPELINE_PARALLEL_SIZE}"' \
                --host '"${HOST}"' \
                --port '"${PORT}"' \
                --max-model-len '"${MAX_MODEL_LEN}"' \
                --trust-remote-code \
                --distributed-executor-backend ray
        else
            echo "[WORKER NODE] Joining Ray cluster at ${HEAD_NODE_IP}:'"${RAY_PORT}"'..."
            
            # Give head node time to start
            sleep 15
            
            # Join the Ray cluster
            ray start --address="${HEAD_NODE_IP}:'"${RAY_PORT}"'" \
                --num-gpus=8 \
                --block
        fi
    '

echo ""
echo "Job completed."
