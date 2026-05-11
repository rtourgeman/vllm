#!/usr/bin/env bash
# Slurm job: serve DeepSeek V3 on all allocated nodes, run warmup + benchmark.
# Launched by run_bench.sh -- do not run directly.
set -Eeuo pipefail

source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/helpers.sh"

CONTAINER_IMAGE="${IMAGE}"
CONTAINER_MOUNTS="${MOUNT_SRC}:${MOUNT_DST},${SCRIPT_DIR}:${SCRIPT_DIR}"

DP_SIZE="${TOTAL_GPUS}"
NUM_NODES="${SLURM_JOB_NUM_NODES}"
RAY_PORT=$((6379 + SLURM_JOB_ID % 1000))
RPC_PORT=$((9876 + SLURM_JOB_ID % 1000))
RUN_DIR="${SCRIPT_DIR}/logs/${SLURM_JOB_ID}_${DP_SIZE}gpu_r${REDUNDANT}_c${CONCURRENCY}"
mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD="${NODES[0]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 --overlap -w "${HEAD}" \
    --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc "hostname -I | tr ' ' '\n' | grep -v '^169\\.254\\.' | head -1")"

cat <<EOF
============================================================
Static benchmark  |  job=${SLURM_JOB_ID}  dp=${DP_SIZE}  redundant=${REDUNDANT}
head=${HEAD} (${HEAD_IP})  prompts=${PROMPTS}  concurrency=${CONCURRENCY}
run_dir=${RUN_DIR}
============================================================
EOF

export SCRIPT_DIR HEAD_IP RAY_PORT RPC_PORT DP_SIZE RUN_DIR REDUNDANT PROMPTS CONCURRENCY MODEL_NAME

srun --nodes="${NUM_NODES}" --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc '
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"
export VLLM_HOST_IP="$(get_routable_ip)"
cd /vllm

my_node="$(hostname -s)"

if [[ "${my_node}" == "'"${HEAD}"'" ]]; then
    # ── HEAD NODE ──
    ray stop -f 2>&1 || true
    rm -rf /data/tmp/ray/session_* || true

    ray start --head --port="${RAY_PORT}" \
        --node-ip-address="${HEAD_IP}" \
        --num-gpus=8 \
        --metrics-export-port=9090 \
        --dashboard-agent-grpc-port=9094 \
        --runtime-env-agent-port=9095 \
        --min-worker-port=20000 --max-worker-port=29999

    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    wait_for_ray_nodes '"${NUM_NODES}"' 300 || true
    ray status || true

    warm_lustre_cache "${MODEL_NAME}"

    # Launch vLLM
    NUM_REDUNDANT_EXPERTS="${REDUNDANT}" \
    DATA_PARALLEL_SIZE="${DP_SIZE}" \
    DATA_PARALLEL_ADDRESS="${HEAD_IP}" \
    DATA_PARALLEL_RPC_PORT="${RPC_PORT}" \
    VLLM_NIXL_EP_MAX_NUM_RANKS="${DP_SIZE}" \
        bash "${SCRIPT_DIR}/serve.sh" >"${RUN_DIR}/vllm_server.log" 2>&1 &
    vllm_pid=$!

    # Wait for startup
    waited=0
    while ! grep -q "Application startup complete" "${RUN_DIR}/vllm_server.log" 2>/dev/null; do
        if ! kill -0 "${vllm_pid}" 2>/dev/null; then
            echo "[${my_node}] vLLM crashed; see ${RUN_DIR}/vllm_server.log"
            exit 1
        fi
        (( waited >= 1200 )) && { echo "[${my_node}] vLLM startup timeout"; exit 1; }
        sleep 15; waited=$((waited + 15))
        echo "[${my_node}] waiting for vLLM... ${waited}s"
    done
    echo "[${my_node}] vLLM is ready"

    # Warmup (1000 prompts, concurrency=256)
    echo "[${my_node}] running warmup benchmark"
    NUM_PROMPTS=1000 MAX_CONCURRENCY=256 \
    BENCH_HOST=localhost WAIT_FOR_SERVER=false \
    BENCH_LOG_FILE="${RUN_DIR}/bench_warmup_np1000_c256_i1024_o1024.log" \
        bash "${SCRIPT_DIR}/bench.sh"

    # Real benchmark
    echo "[${my_node}] running real benchmark"
    NUM_PROMPTS="${PROMPTS}" MAX_CONCURRENCY="${CONCURRENCY}" \
    BENCH_HOST=localhost WAIT_FOR_SERVER=false \
    BENCH_LOG_FILE="${RUN_DIR}/bench_${DP_SIZE}gpu_np${PROMPTS}_c${CONCURRENCY}_i1024_o1024.log" \
        bash "${SCRIPT_DIR}/bench.sh"

    echo "[${my_node}] done, shutting down"
    kill "${vllm_pid}" 2>&1 || true; sleep 3; kill -9 "${vllm_pid}" 2>&1 || true
    ray stop -f 2>&1 || true
else
    # ── WORKER NODE ──
    ray stop -f 2>&1 || true
    rm -rf /data/tmp/ray/session_* || true
    warm_lustre_cache "${MODEL_NAME}"
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    join_ray_with_retry "${HEAD_IP}:${RAY_PORT}" 8
fi
'

echo "Job ${SLURM_JOB_ID} completed. Logs in ${RUN_DIR}"
