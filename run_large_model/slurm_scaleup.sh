#!/usr/bin/env bash
# Slurm job: serve DeepSeek V3 on initial nodes, scale up, run warmup + benchmark.
# Launched by run_bench_scaleup.sh -- do not run directly.
set -Eeuo pipefail

source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/helpers.sh"

CONTAINER_IMAGE="${IMAGE}"
CONTAINER_MOUNTS="${MOUNT_SRC}:${MOUNT_DST},${SCRIPT_DIR}:${SCRIPT_DIR}"

NUM_NODES="${SLURM_JOB_NUM_NODES}"
RAY_PORT=$((6379 + SLURM_JOB_ID % 1000))
RPC_PORT=$((9876 + SLURM_JOB_ID % 1000))
RUN_DIR="${SCRIPT_DIR}/logs/c${CONCURRENCY}_${INITIAL_GPUS}to${TARGET_GPUS}gpu_r${SCALE_REDUNDANT}_id${SLURM_JOB_ID}"
SIGNAL_FILE="${RUN_DIR}/join_now.signal"
mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD="${NODES[0]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 --overlap -w "${HEAD}" \
    --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc "hostname -I | tr ' ' '\n' | grep -v '^169\\.254\\.' | head -1")"

cat <<EOF
============================================================
Scale-up benchmark  |  job=${SLURM_JOB_ID}  dp=${INITIAL_GPUS}->${TARGET_GPUS}
head=${HEAD} (${HEAD_IP})  redundant=${REDUNDANT}->${SCALE_REDUNDANT}
prompts=${PROMPTS}  concurrency=${CONCURRENCY}
run_dir=${RUN_DIR}
============================================================
EOF

export SCRIPT_DIR HEAD_IP RAY_PORT RPC_PORT RUN_DIR SIGNAL_FILE
export INITIAL_GPUS INITIAL_NODES TARGET_GPUS REDUNDANT SCALE_REDUNDANT PROMPTS CONCURRENCY

srun --nodes="${NUM_NODES}" --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc '
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/helpers.sh"
export VLLM_HOST_IP="$(get_routable_ip)"
cd /vllm

my_node="$(hostname -s)"
all_nodes=('"$(printf "'%s' " "${NODES[@]}")"')

# Determine this node index
my_idx=-1
for i in "${!all_nodes[@]}"; do
    [[ "${all_nodes[$i]}" == "${my_node}" ]] && { my_idx=$i; break; }
done

if [[ "${my_node}" == "'"${HEAD}"'" ]]; then
    # ── HEAD NODE ──
    ray stop -f 2>&1 || true
    rm -rf /data/tmp/ray/session_* 2>/dev/null || true

    ray start --head --port="${RAY_PORT}" \
        --node-ip-address="${HEAD_IP}" \
        --num-gpus=8 \
        --metrics-export-port=9090 \
        --dashboard-agent-grpc-port=9094 \
        --runtime-env-agent-port=9095 \
        --min-worker-port=20000 --max-worker-port=29999

    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

    # Wait for initial nodes only (not the extra node)
    wait_for_ray_nodes "${INITIAL_NODES}" 300 || true
    ray status || true

    warm_lustre_cache /rtourgeman/models/DeepSeek-V3

    # Launch vLLM with initial GPU count
    NUM_REDUNDANT_EXPERTS="${REDUNDANT}" \
    DATA_PARALLEL_SIZE="${INITIAL_GPUS}" \
    DATA_PARALLEL_ADDRESS="${HEAD_IP}" \
    DATA_PARALLEL_RPC_PORT="${RPC_PORT}" \
    VLLM_NIXL_EP_MAX_NUM_RANKS="${TARGET_GPUS}" \
        bash "${SCRIPT_DIR}/serve.sh" >"${RUN_DIR}/vllm_server.log" 2>&1 &
    vllm_pid=$!

    # Wait for startup
    waited=0
    while ! grep -q "Application startup complete" "${RUN_DIR}/vllm_server.log" 2>/dev/null; do
        if ! kill -0 "${vllm_pid}" 2>/dev/null; then
            echo "[${my_node}] vLLM crashed"; tail -20 "${RUN_DIR}/vllm_server.log" 2>/dev/null; exit 1
        fi
        (( waited >= 1200 )) && { echo "[${my_node}] vLLM startup timeout"; exit 1; }
        sleep 15; waited=$((waited + 15))
        echo "[${my_node}] waiting for vLLM... ${waited}s"
    done
    echo "[${my_node}] vLLM is ready (dp=${INITIAL_GPUS})"

    # Signal extra node to join Ray
    echo "[${my_node}] signaling extra node(s) to join Ray"
    touch "${SIGNAL_FILE}"

    # Wait for all nodes
    wait_for_ray_nodes '"${NUM_NODES}"' 300 || { echo "Not enough nodes for scale-up"; exit 1; }

    # Scale up
    echo "[${my_node}] scaling from ${INITIAL_GPUS} to ${TARGET_GPUS} GPUs (redundant=${SCALE_REDUNDANT})"
    scale_start=$(date +%s)
    python3 examples/online_serving/elastic_ep/scale.py \
        --host localhost --port 8006 \
        --new-dp-size "${TARGET_GPUS}" \
        --num-redundant-experts "${SCALE_REDUNDANT}"
    scale_end=$(date +%s)
    echo "[${my_node}] scale-up completed in $((scale_end - scale_start))s"
    sleep 30

    TAG="np${PROMPTS}_c${CONCURRENCY}_i1024_o1024"

    # Warmup
    echo "[${my_node}] running warmup benchmark"
    NUM_PROMPTS=1000 MAX_CONCURRENCY=256 \
    BENCH_HOST=localhost WAIT_FOR_SERVER=false \
    BENCH_LOG_FILE="${RUN_DIR}/bench_warmup_${TAG}.log" \
        bash "${SCRIPT_DIR}/bench.sh"

    # Real benchmark
    echo "[${my_node}] running real benchmark (${TARGET_GPUS} GPUs)"
    NUM_PROMPTS="${PROMPTS}" MAX_CONCURRENCY="${CONCURRENCY}" \
    BENCH_HOST=localhost WAIT_FOR_SERVER=false \
    BENCH_LOG_FILE="${RUN_DIR}/bench_${INITIAL_GPUS}to${TARGET_GPUS}gpu_${TAG}.log" \
        bash "${SCRIPT_DIR}/bench.sh"

    echo "[${my_node}] done, shutting down"
    kill "${vllm_pid}" 2>&1 || true; sleep 3; kill -9 "${vllm_pid}" 2>&1 || true
    ray stop -f 2>&1 || true

elif (( my_idx > 0 && my_idx < INITIAL_NODES )); then
    # ── INITIAL WORKER (joins Ray immediately) ──
    ray stop -f 2>&1 || true
    rm -rf /data/tmp/ray/session_* 2>/dev/null || true
    warm_lustre_cache /rtourgeman/models/DeepSeek-V3
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    join_ray_with_retry "${HEAD_IP}:${RAY_PORT}" 8

else
    # ── EXTRA WORKER (waits for signal, then joins Ray) ──
    ray stop -f 2>&1 || true
    rm -rf /data/tmp/ray/session_* 2>/dev/null || true
    warm_lustre_cache /rtourgeman/models/DeepSeek-V3

    echo "[${my_node}] waiting for signal to join Ray..."
    while [[ ! -f "${SIGNAL_FILE}" ]]; do sleep 5; done

    echo "[${my_node}] signal received, joining Ray"
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    join_ray_with_retry "${HEAD_IP}:${RAY_PORT}" 8
fi
'

echo "Job ${SLURM_JOB_ID} completed. Logs in ${RUN_DIR}"
