#!/usr/bin/env bash
# Shared defaults for all run_large_model scripts.
# Every variable uses ${VAR:-default} so env or CLI overrides take precedence.

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Container
#IMAGE="${IMAGE:-/lustre/fsw/portfolios/network/users/rtourgeman/latest_rebase_20_4_26_v9.sqsh}"
IMAGE="${IMAGE:-/lustre/fsw/portfolios/network/users/rtourgeman/large_model.sqsh}"
MOUNT_SRC="${MOUNT_SRC:-/lustre/fsw/portfolios/network/users/rtourgeman}"
MOUNT_DST="${MOUNT_DST:-/rtourgeman}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Slurm
ACCOUNT="${ACCOUNT:-network_research_advdev}"
BATCH_PARTITION="${BATCH_PARTITION:-batch}"
TIME="${TIME:-01:00:00}"

# Model / vLLM
MODEL_NAME="${MODEL_NAME:-/rtourgeman/models/DeepSeek-V3}"
SECONDARY_MODEL_NAME="${SECONDARY_MODEL_NAME:-/rtourgeman/models/DeepSeek-V2-Lite-Chat}"
PORT="${PORT:-8006}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-nixl_ep}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ENABLE_ELASTIC_EP="${ENABLE_ELASTIC_EP:-true}"
ENABLE_EPLB="${ENABLE_EPLB:-true}"
VLLM_WORKDIR="${VLLM_WORKDIR:-/vllm}"

# EPLB
NUM_REDUNDANT_EXPERTS="${NUM_REDUNDANT_EXPERTS:-0}"
EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-1000}"
EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-3000}"

# Benchmark
NUM_PROMPTS="${NUM_PROMPTS:-512}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1024}"

# Timeouts (seconds)
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-1200}"
RAY_WAIT_TIMEOUT="${RAY_WAIT_TIMEOUT:-300}"
