#!/usr/bin/env bash
#===============================================================================
# vllm_deepseek_v3.slurm
# Slurm batch script for running vLLM with DeepSeek V3 on 3 nodes × 8 GPUs
# Uses expert parallelism with NIXL all2all backend
#===============================================================================

#-------------------------------------------------------------------------------
# SBATCH Directives
#-------------------------------------------------------------------------------
#SBATCH --job-name=coreai_comparch_trtllm-vllm.deepseek_v3
#SBATCH --comment="vLLM DeepSeek V3 multi-node inference"
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
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
CONTAINER_IMAGE="/lustre/fsw/coreai_comparch_trtllm/rtourgeman/elastic-ep/vllm/27_1_26_deep_seek_perf_nixl_weight_transfer_v5.sqsh"

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

# MODEL_NAME="/workspace/llm_models/DeepSeek-V3/"
# Model - Full DeepSeek V3 (BF16)
MODEL_NAME="/workspace/llm_models/DeepSeek-V3/"

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

# Get head node IP by running hostname on that specific node
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 --overlap -w "${HEAD_NODE}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    hostname -I | awk '{print $1}')

echo "Head Node:        ${HEAD_NODE}"
echo "Head Node IP:     ${HEAD_NODE_IP}"
echo "============================================================"

# Export for use in the srun script
export HEAD_NODE
export HEAD_NODE_IP
export RAY_PORT
export MODEL_NAME
export HOST
export PORT
export DATA_PARALLEL_SIZE
export DATA_PARALLEL_SIZE_LOCAL
export VLLM_NIXL_EP_MAX_NUM_RANKS
export NUM_REDUNDANT_EXPERTS
export EPLB_WINDOW_SIZE
export EPLB_STEP_INTERVAL
export MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION

#-------------------------------------------------------------------------------
# Launch all nodes with a single srun
# Each node will determine if it's head or worker based on hostname
#-------------------------------------------------------------------------------
echo ""
echo "Launching Ray cluster and vLLM across all nodes..."
echo ""

srun --nodes=${SLURM_JOB_NUM_NODES} \
    --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -c '
        set -e
        
        # Patch NIXL EP timeout from 30s to 300s for large model loading
        NIXL_BUFFER_PY="/workspace/nixl/install/lib/python3/dist-packages/nixl_ep/buffer.py"
        if grep -q "timedelta(seconds=30)" "$NIXL_BUFFER_PY" 2>/dev/null; then
            echo "[PATCH] Increasing NIXL EP timeout from 30s to 300s..."
            sed -i "s/timedelta(seconds=30)/timedelta(seconds=300)/g" "$NIXL_BUFFER_PY"
        fi
        
        # NIXL environment
        export LD_LIBRARY_PATH=/workspace/nixl/install/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
        export PYTHONPATH=/workspace/nixl/install/lib/python3/dist-packages
        export NIXL_PLUGIN_DIR=/workspace/nixl/install/lib/x86_64-linux-gnu/plugins
        
        # DeepGemm/CUDA
        export DG_JIT_NVCC_COMPILER=/usr/local/cuda-12.9/bin/nvcc
        export CUDA_HOME="/usr/local/cuda-12.9"
        
        # vLLM settings
        export VLLM_USE_V1=1
        export VLLM_USE_DEEP_GEMM=1
        export VLLM_NIXL_EP_MAX_NUM_RANKS='"${VLLM_NIXL_EP_MAX_NUM_RANKS}"'
        export VLLM_NIXL_EP_UCX_IB_DEVICES="mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11"
        export VLLM_NIXL_EP_UCX_TCP_DEVICES="ibp154s0,ibp192s0,ibp206s0,ibp220s0,ibp94s0"
        export UCX_CUDA_COPY_DMABUF=n
        
        # Use job-unique port to avoid conflicts between jobs
        export MASTER_PORT=$((29500 + ('"${SLURM_JOB_ID}"' % 1000)))
        
        cd /workspace/vllm
        
        MY_HOSTNAME=$(hostname -s)
        MY_IP=$(hostname -I | awk "{print \$1}")
        
        echo "[Node ${MY_HOSTNAME}] IP=${MY_IP}, HEAD_NODE='"${HEAD_NODE}"', HEAD_IP='"${HEAD_NODE_IP}"'"
        
        if [[ "${MY_HOSTNAME}" == "'"${HEAD_NODE}"'" ]]; then
            echo "[HEAD NODE ${MY_HOSTNAME}] Starting Ray head..."
            
            # Start Ray head
            ray start --head --port='"${RAY_PORT}"' --node-ip-address='"${HEAD_NODE_IP}"' --num-gpus=8
            
            # Wait for workers to join (give them time to start)
            echo "[HEAD NODE] Waiting for workers to join..."
            sleep 45
            
            # Check Ray cluster status
            echo "[HEAD NODE] Ray cluster status:"
            ray status || true
            
            # Launch vLLM server
            echo "[HEAD NODE] Starting vLLM server..."
            LEADER_ADDRESS=$(hostname -I | awk "{print \$1}")

            RESTART_AFTER_SECONDS=900  # 10 minutes

            start_vllm_serve() {
                vllm serve '"${MODEL_NAME}"' --trust-remote-code \
                    --disable-log-requests \
                    --host '"${HOST}"' \
                    --port '"${PORT}"' \
                    --tensor-parallel-size 1 \
                    --gpu-memory-utilization '"${GPU_MEMORY_UTILIZATION}"' \
                    --max-model-len '"${MAX_MODEL_LEN}"' \
                    --no-enable-prefix-caching \
                    --enable-expert-parallel \
                    --enable-elastic-ep \
                    --enable-eplb \
                    --eplb-config.num_redundant_experts '"${NUM_REDUNDANT_EXPERTS}"' \
                    --eplb-config.window_size '"${EPLB_WINDOW_SIZE}"' \
                    --eplb-config.step_interval '"${EPLB_STEP_INTERVAL}"' \
                    --data-parallel-backend ray \
                    --data-parallel-size '"${DATA_PARALLEL_SIZE}"' \
                    --data-parallel-size-local '"${DATA_PARALLEL_SIZE_LOCAL}"' \
                    --data-parallel-address ${LEADER_ADDRESS} \
                    --data-parallel-rpc-port 9876 \
                    --data-parallel-start-rank 0 \
                    --all2all-backend nixl_ep
            }

            echo "[HEAD NODE] vLLM will be restarted after ${RESTART_AFTER_SECONDS}s..."
            start_vllm_serve &
            VLLM_PID=$!

            # Wait 10 minutes, then restart once.
            sleep "${RESTART_AFTER_SECONDS}" || true
            echo "[HEAD NODE] Stopping vLLM (pid=${VLLM_PID}) for planned restart..."
            kill -INT "${VLLM_PID}" 2>/dev/null || true
            wait "${VLLM_PID}" || true
            echo "[HEAD NODE] vLLM stopped. Restarting..."
            sleep 5

            # Replace this bash with the restarted vLLM process (keeps job alive).
            exec start_vllm_serve
        else
            echo "[WORKER NODE ${MY_HOSTNAME}] Waiting for Ray head to start..."
            sleep 15
            
            echo "[WORKER NODE ${MY_HOSTNAME}] Joining Ray cluster at '"${HEAD_NODE_IP}"':'"${RAY_PORT}"'..."
            ray start --address='"${HEAD_NODE_IP}"':'"${RAY_PORT}"' --num-gpus=8 --block
        fi
    '

echo ""
echo "Job completed."
