#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/config.sh"

DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-32}"
DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-${GPUS_PER_NODE}}"
DATA_PARALLEL_START_RANK="${DATA_PARALLEL_START_RANK:-0}"
DATA_PARALLEL_ADDRESS="${DATA_PARALLEL_ADDRESS:-$(hostname -I | tr ' ' '\n' | grep -v '^169\.254\.' | head -1)}"
DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT:-9876}"
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export DG_JIT_NVCC_COMPILER="${DG_JIT_NVCC_COMPILER:-${CUDA_HOME}/bin/nvcc}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
export VLLM_NIXL_EP_MAX_NUM_RANKS="${VLLM_NIXL_EP_MAX_NUM_RANKS:-${DATA_PARALLEL_SIZE}}"
export UCX_CUDA_COPY_DMABUF="${UCX_CUDA_COPY_DMABUF:-n}"

cmd=(
    vllm serve "${MODEL_NAME}" --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --no-enable-prefix-caching
    --enable-expert-parallel
    --data-parallel-backend ray
    --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}"
    --data-parallel-address "${DATA_PARALLEL_ADDRESS}"
    --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}"
    --data-parallel-start-rank "${DATA_PARALLEL_START_RANK}"
    --api-server-count "${API_SERVER_COUNT}"
    --all2all-backend "${ALL2ALL_BACKEND}"
    --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"}'
    --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
)

if [[ "${ENABLE_ELASTIC_EP}" == "true" ]]; then
    cmd+=(--enable-elastic-ep)
fi

if [[ "${ENABLE_EPLB}" == "true" ]]; then
    cmd+=(
        --enable-eplb
        --eplb-config.num_redundant_experts "${NUM_REDUNDANT_EXPERTS}"
        --eplb-config.window_size "${EPLB_WINDOW_SIZE}"
        --eplb-config.step_interval "${EPLB_STEP_INTERVAL}"
    )
fi

printf 'Starting vLLM DeepSeek V3 server:\n'
printf '  model=%s  dp=%s  all2all=%s\n' "${MODEL_NAME}" "${DATA_PARALLEL_SIZE}" "${ALL2ALL_BACKEND}"

exec "${cmd[@]}"
