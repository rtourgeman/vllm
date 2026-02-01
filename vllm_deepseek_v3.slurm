#!/usr/bin/env bash
#===============================================================================
# vllm_deepseek_v3.slurm
# Slurm batch script for running vLLM with DeepSeek V3 on 3 nodes × 8 GPUs
# Uses expert parallelism with NIXL all2all backend
#===============================================================================

#-------------------------------------------------------------------------------
# SBATCH Directives
#-------------------------------------------------------------------------------
#SBATCH --job-name=vllm-deepseek-v3
#SBATCH --comment="vLLM DeepSeek V3 multi-node inference"
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

#-------------------------------------------------------------------------------
# Configuration - EDIT THESE AS NEEDED
#-------------------------------------------------------------------------------

# Container
CONTAINER_IMAGE="/lustre/fsw/coreai_comparch_trtllm/rtourgeman/elastic-ep/vllm/27_1_26_deep_seek_perf_nixl_weight_transfer_v4.sqsh"

CONTAINER_MOUNTS="/etc/slurm,\
/usr/bin/srun,\
/usr/bin/sinfo,\
/usr/bin/scontrol,\
/usr/bin/squeue,\
/usr/lib/x86_64-linux-gnu/slurm,\
/lib/x86_64-linux-gnu/libmunge.so.2,\
/run/munge,\
/lustre/fsw/network_research_advdev/${USER}:/workspace/external,\
/lustre/fsw/network_research_advdev/bblack/gdaki/linux-headers-5.15.0-88-generic:/lib/modules/5.15.0-88-generic/build,\
/lustre/fsw/network_research_advdev/bblack/gdaki/linux-headers-5.15.0-88-generic:/usr/src/linux-headers-5.15.0-88-generic,\
/lustre/fsw/coreai_comparch_trtllm/llm_data/llm-models/:/workspace/llm_models"

# Model - UPDATE THIS PATH for full DeepSeek V3
MODEL_NAME="/workspace/llm_models/DeepSeek-V3/fp8/"  # <-- ADJUST PATH

# vLLM server settings
HOST="0.0.0.0"
PORT=8006

# Parallelism settings for 3 nodes × 8 GPUs = 24 GPUs
DATA_PARALLEL_SIZE=24          # Total GPUs across all nodes
DATA_PARALLEL_SIZE_LOCAL=8     # GPUs per node

# NIXL/Expert parallelism settings
VLLM_NIXL_EP_MAX_NUM_RANKS=24  # Must match DATA_PARALLEL_SIZE

# EPLB (Expert Parallelism Load Balancing) settings
NUM_REDUNDANT_EXPERTS=8
EPLB_WINDOW_SIZE=1000
EPLB_STEP_INTERVAL=3000

# Model settings - may need adjustment for full V3
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.9

# Ray settings
RAY_PORT=6379

#-------------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------------
set -Eeuo pipefail
mkdir -p logs

echo "============================================================"
echo "vLLM DeepSeek V3 - Multi-Node Expert Parallel"
echo "============================================================"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Nodes:            ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs:       $((SLURM_JOB_NUM_NODES * 8))"
echo "Node List:        ${SLURM_JOB_NODELIST}"
echo "DATA_PARALLEL:    ${DATA_PARALLEL_SIZE}"
echo "============================================================"

#-------------------------------------------------------------------------------
# Get head node info
#-------------------------------------------------------------------------------
HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')

echo "Head Node:        ${HEAD_NODE}"
echo "Head Node IP:     ${HEAD_NODE_IP}"
echo "============================================================"

#-------------------------------------------------------------------------------
# Step 1: Start Ray head on first node
#-------------------------------------------------------------------------------
echo ""
echo "[Step 1] Starting Ray head on ${HEAD_NODE}..."

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --output=logs/ray_head_${SLURM_JOB_ID}.out \
    --error=logs/ray_head_${SLURM_JOB_ID}.err \
    bash -c "
        export LD_LIBRARY_PATH=/workspace/nixl/install/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
        export PYTHONPATH=/workspace/nixl/install/lib/python3/dist-packages
        export NIXL_PLUGIN_DIR=/workspace/nixl/install/lib/x86_64-linux-gnu/plugins
        
        cd /workspace/vllm
        ray start --head --port=${RAY_PORT} --node-ip-address=${HEAD_NODE_IP} --block
    " &

RAY_HEAD_PID=$!
echo "Ray head starting in background (PID: ${RAY_HEAD_PID})"

# Wait for Ray head to initialize
sleep 15

#-------------------------------------------------------------------------------
# Step 2: Start Ray workers on remaining nodes
#-------------------------------------------------------------------------------
WORKER_NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tail -n +2)

if [[ -n "${WORKER_NODES}" ]]; then
    echo ""
    echo "[Step 2] Starting Ray workers on: ${WORKER_NODES}"
    
    for WORKER in ${WORKER_NODES}; do
        echo "  Starting worker on ${WORKER}..."
        srun --nodes=1 --ntasks=1 -w "${WORKER}" \
            --container-image="${CONTAINER_IMAGE}" \
            --container-mounts="${CONTAINER_MOUNTS}" \
            --output=logs/ray_worker_${WORKER}_${SLURM_JOB_ID}.out \
            --error=logs/ray_worker_${WORKER}_${SLURM_JOB_ID}.err \
            bash -c "
                export LD_LIBRARY_PATH=/workspace/nixl/install/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
                export PYTHONPATH=/workspace/nixl/install/lib/python3/dist-packages
                export NIXL_PLUGIN_DIR=/workspace/nixl/install/lib/x86_64-linux-gnu/plugins
                
                cd /workspace/vllm
                ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --block
            " &
    done
    
    # Wait for workers to join
    echo "Waiting for workers to join Ray cluster..."
    sleep 20
fi

#-------------------------------------------------------------------------------
# Step 3: Launch vLLM server on head node
#-------------------------------------------------------------------------------
echo ""
echo "[Step 3] Launching vLLM server on ${HEAD_NODE}..."
echo "  Model: ${MODEL_NAME}"
echo "  Data Parallel Size: ${DATA_PARALLEL_SIZE}"
echo "  Data Parallel Size Local: ${DATA_PARALLEL_SIZE_LOCAL}"
echo ""

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --output=logs/vllm_server_${SLURM_JOB_ID}.out \
    --error=logs/vllm_server_${SLURM_JOB_ID}.err \
    bash -c "
        # NIXL environment
        export LD_LIBRARY_PATH=/workspace/nixl/install/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
        export PYTHONPATH=/workspace/nixl/install/lib/python3/dist-packages
        export NIXL_PLUGIN_DIR=/workspace/nixl/install/lib/x86_64-linux-gnu/plugins
        
        # DeepGemm/CUDA
        export DG_JIT_NVCC_COMPILER=/usr/local/cuda-12.9/bin/nvcc
        export CUDA_HOME='/usr/local/cuda-12.9'
        
        # vLLM settings
        export VLLM_USE_V1=1
        export VLLM_USE_DEEP_GEMM=1
        export VLLM_NIXL_EP_MAX_NUM_RANKS=${VLLM_NIXL_EP_MAX_NUM_RANKS}
        export VLLM_NIXL_EP_UCX_IB_DEVICES='mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11'
        export VLLM_NIXL_EP_UCX_TCP_DEVICES='ibp154s0,ibp192s0,ibp206s0,ibp220s0,ibp94s0'
        export UCX_CUDA_COPY_DMABUF=n
        
        cd /workspace/vllm
        
        LEADER_ADDRESS=\$(hostname -I | awk '{print \$1}')
        
        echo 'Starting vLLM serve...'
        echo \"LEADER_ADDRESS: \${LEADER_ADDRESS}\"
        
        vllm serve ${MODEL_NAME} --trust-remote-code \\
            --disable-log-requests \\
            --host ${HOST} \\
            --port ${PORT} \\
            --tensor-parallel-size 1 \\
            --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\
            --max-model-len ${MAX_MODEL_LEN} \\
            --no-enable-prefix-caching \\
            --enable-expert-parallel \\
            --enable-elastic-ep \\
            --enable-eplb \\
            --eplb-config.num_redundant_experts ${NUM_REDUNDANT_EXPERTS} \\
            --eplb-config.window_size ${EPLB_WINDOW_SIZE} \\
            --eplb-config.step_interval ${EPLB_STEP_INTERVAL} \\
            --data-parallel-backend ray \\
            --data-parallel-size ${DATA_PARALLEL_SIZE} \\
            --data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL} \\
            --data-parallel-address \${LEADER_ADDRESS} \\
            --data-parallel-rpc-port 9876 \\
            --data-parallel-start-rank 0 \\
            --all2all-backend nixl_ep
    "

echo ""
echo "Job completed."
