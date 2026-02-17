#!/bin/bash

MODEL_NAME="/workspace/external/llm_models/DeepSeek-V3-Lite/fp8/"
MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite-Chat"
HOST="0.0.0.0"
PORT=8006

# Default value
DATA_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE_LOCAL=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dp-size)
            DATA_PARALLEL_SIZE="$2"
            DATA_PARALLEL_SIZE_LOCAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

LEADER_ADDRESS=$(hostname -I | awk '{print $1}')

NUM_REDUNDANT_EXPERTS=0
EPLB_WINDOW_SIZE=1000
EPLB_STEP_INTERVAL=3000
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.8

export DG_JIT_NVCC_COMPILER=/usr/local/cuda-12.9/bin/nvcc
export CUDA_HOME='/usr/local/cuda-12.9'

export VLLM_USE_V1=1
VLLM_ALL2ALL_BACKEND="pplx"
VLLM_ALL2ALL_BACKEND="deepep_low_latency"
VLLM_ALL2ALL_BACKEND="nixl_ep"
export VLLM_USE_DEEP_GEMM=1

export UCX_CUDA_COPY_DMABUF=n
export VLLM_NIXL_EP_MAX_NUM_RANKS=8

# Launch the vLLM server
vllm serve $MODEL_NAME --trust-remote-code \
    --disable-log-requests \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --enable-eplb \
    --eplb-config.num_redundant_experts $NUM_REDUNDANT_EXPERTS \
    --eplb-config.window_size $EPLB_WINDOW_SIZE \
    --eplb-config.step_interval $EPLB_STEP_INTERVAL \
    --data-parallel-backend ray \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --data-parallel-size-local $DATA_PARALLEL_SIZE_LOCAL \
    --data-parallel-address $LEADER_ADDRESS \
    --data-parallel-rpc-port 9876 \
    --data-parallel-start-rank 0 --all2all-backend $VLLM_ALL2ALL_BACKEND --api-server-count 1
