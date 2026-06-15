#!/usr/bin/env bash
# Slurm job: bring up a Ray cluster once, then cycle through a schedule of
# vLLM serves. For each entry in SCHEDULE we: start vLLM, wait until healthy,
# run a warmup, then close vLLM. The time for every "close + upload" transition
# (from the moment we start killing serve i until serve i+1 is healthy) is
# measured and reported in a summary at the end.
#
# Only the first serve's nodes join Ray up front; the extra node(s) join after
# the first vLLM serve has run (signalled via SIGNAL_FILE), like the scale-up
# run. This is required because vLLM claims every GPU in the Ray cluster for DP
# placement groups -- a dp=32 serve on a 40-GPU cluster fails with
# "Created 40 DP placement groups, expected 32".
#
# Launched by run_restart_bench.sh -- do not run directly.
set -Eeuo pipefail

source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/helpers.sh"

CONTAINER_IMAGE="${IMAGE}"
CONTAINER_MOUNTS="${MOUNT_SRC}:${MOUNT_DST},${SCRIPT_DIR}:${SCRIPT_DIR}"

NUM_NODES="${SLURM_JOB_NUM_NODES}"
RAY_PORT=$((6379 + SLURM_JOB_ID % 1000))
RPC_PORT=$((9876 + SLURM_JOB_ID % 1000))
RUN_DIR="${SCRIPT_DIR}/logs/${SLURM_JOB_ID}_restart_$(echo "${SCHEDULE}" | tr ' ' '-')gpu"
SIGNAL_FILE="${RUN_DIR}/join_now.signal"
mkdir -p "${RUN_DIR}"
rm -f "${SIGNAL_FILE}" 2>/dev/null || true

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD="${NODES[0]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 --overlap -w "${HEAD}" \
    --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc "hostname -I | tr ' ' '\n' | grep -v '^169\\.254\\.' | head -1")"

cat <<EOF
============================================================
Restart benchmark  |  job=${SLURM_JOB_ID}
schedule=${SCHEDULE}  redundant=${REDUNDANT_SCHEDULE}  max_gpu=${MAX_GPU}
head=${HEAD} (${HEAD_IP})
run_dir=${RUN_DIR}
============================================================
EOF

export SCRIPT_DIR HEAD_IP RAY_PORT RPC_PORT RUN_DIR SIGNAL_FILE REDUNDANT_SCHEDULE MODEL_NAME SCHEDULE MAX_GPU GPUS_PER_NODE

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

# Index of this node in the allocation (0 == head).
my_idx=-1
for i in "${!all_nodes[@]}"; do
    [[ "${all_nodes[$i]}" == "${my_node}" ]] && { my_idx=$i; break; }
done

schedule=(${SCHEDULE})
# Nodes the FIRST serve needs; the extra node(s) join only after it has run.
first_serve_nodes=$(( schedule[0] / GPUS_PER_NODE ))
max_nodes=$(( MAX_GPU / GPUS_PER_NODE ))

if [[ "${my_node}" == "'"${HEAD}"'" ]]; then
    # ?? HEAD NODE ??
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
    # Only wait for the first serve's nodes; the extra node joins later.
    wait_for_ray_nodes "${first_serve_nodes}" 300 || true
    ray status || true

    warm_lustre_cache "${MODEL_NAME}"

    redundant=(${REDUNDANT_SCHEDULE})
    summary_file="${RUN_DIR}/restart_summary.txt"

    # Per-serve timing arrays (indexed by serve number).
    declare -a serve_gpu serve_red up_secs close_secs healthy_ts close_start_ts

    for idx in "${!schedule[@]}"; do
        gpu="${schedule[$idx]}"
        red="${redundant[$idx]}"
        n=$(printf "%02d" "${idx}")
        log="${RUN_DIR}/vllm_server_${n}_${gpu}gpu.log"
        echo "[${my_node}] === serve #${idx}: dp=${gpu} redundant=${red} (log=${log}) ==="

        # ?? UPLOAD ??
        up_start=$(date +%s)
        NUM_REDUNDANT_EXPERTS="${red}" \
        DATA_PARALLEL_SIZE="${gpu}" \
        DATA_PARALLEL_ADDRESS="${HEAD_IP}" \
        DATA_PARALLEL_RPC_PORT="${RPC_PORT}" \
        VLLM_NIXL_EP_MAX_NUM_RANKS="${MAX_GPU}" \
            bash "${SCRIPT_DIR}/serve.sh" >"${log}" 2>&1 &
        vllm_pid=$!

        waited=0
        while ! grep -q "Application startup complete" "${log}" 2>/dev/null; do
            if ! kill -0 "${vllm_pid}" 2>/dev/null; then
                echo "[${my_node}] vLLM crashed on serve #${idx}; see ${log}"
                exit 1
            fi
            (( waited >= 1800 )) && { echo "[${my_node}] vLLM startup timeout on serve #${idx}"; exit 1; }
            sleep 5; waited=$((waited + 5))
        done
        healthy=$(date +%s)

        serve_gpu[$idx]="${gpu}"
        serve_red[$idx]="${red}"
        up_secs[$idx]=$(( healthy - up_start ))
        healthy_ts[$idx]="${healthy}"
        echo "[${my_node}] serve #${idx} healthy in ${up_secs[$idx]}s"

        # Report the close+upload transition into this serve (skip the first).
        if (( idx > 0 )); then
            prev=$(( idx - 1 ))
            cycle=$(( healthy - close_start_ts[$prev] ))
            echo "[${my_node}] >>> close(${serve_gpu[$prev]}gpu)+upload(${gpu}gpu) = ${cycle}s"
        fi

        # ?? WARMUP ??
        echo "[${my_node}] warmup: 8192 prompts, unlimited concurrency"
        NUM_PROMPTS=8192 MAX_CONCURRENCY=0 BENCH_HOST=localhost WAIT_FOR_SERVER=false \
        BENCH_LOG_FILE="${RUN_DIR}/bench_${n}_${gpu}gpu_warmup_np8192.log" \
            bash "${SCRIPT_DIR}/bench.sh"

        # ?? CLOSE ??
        echo "[${my_node}] closing serve #${idx} (dp=${gpu})"
        close_start=$(date +%s)
        close_start_ts[$idx]="${close_start}"
        kill "${vllm_pid}" 2>/dev/null || true
        cwait=0
        while kill -0 "${vllm_pid}" 2>/dev/null; do
            if (( cwait >= 120 )); then
                kill -9 "${vllm_pid}" 2>/dev/null || true
                sleep 2
                break
            fi
            sleep 2; cwait=$((cwait + 2))
        done
        close_end=$(date +%s)
        close_secs[$idx]=$(( close_end - close_start ))
        echo "[${my_node}] serve #${idx} closed in ${close_secs[$idx]}s"

        # After the first serve has run, bring in the extra node(s) so the
        # remaining (larger) serves have the full cluster.
        if (( idx == 0 && max_nodes > first_serve_nodes )); then
            echo "[${my_node}] signalling extra node(s) to join Ray"
            touch "${SIGNAL_FILE}"
            wait_for_ray_nodes "${max_nodes}" 300 || true
            ray status || true
        fi
    done

    # ?? SUMMARY ??
    {
        echo "============================================================"
        echo "CLOSE+UPLOAD SUMMARY  (job=${SLURM_JOB_ID})"
        echo "schedule:  ${SCHEDULE}"
        echo "redundant: ${REDUNDANT_SCHEDULE}"
        echo "============================================================"
        printf "%-52s %10s %10s %12s\n" "transition" "close(s)" "upload(s)" "close+up(s)"
        echo "------------------------------------------------------------------------------------"
        total=0
        count=0
        for idx in "${!schedule[@]}"; do
            (( idx == 0 )) && continue
            prev=$(( idx - 1 ))
            cycle=$(( healthy_ts[$idx] - close_start_ts[$prev] ))
            label="#${prev}(${serve_gpu[$prev]}gpu/r${serve_red[$prev]}) -> #${idx}(${serve_gpu[$idx]}gpu/r${serve_red[$idx]})"
            printf "%-52s %10s %10s %12s\n" \
                "${label}" "${close_secs[$prev]}" "${up_secs[$idx]}" "${cycle}"
            total=$(( total + cycle ))
            count=$(( count + 1 ))
        done
        echo "------------------------------------------------------------------------------------"
        if (( count > 0 )); then
            printf "%-52s %10s %10s %12s\n" "average (n=${count})" "" "" "$(( total / count ))"
        fi
        echo "============================================================"
    } | tee "${summary_file}"
    echo "[${my_node}] summary written to ${summary_file}"

    echo "[${my_node}] all serves done, shutting down Ray"
    ray stop -f 2>&1 || true
elif (( my_idx < first_serve_nodes )); then
    # ?? INITIAL WORKER (needed for the first serve, joins immediately) ??
    ray stop -f 2>&1 || true
    rm -rf /data/tmp/ray/session_* || true
    warm_lustre_cache "${MODEL_NAME}"
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    join_ray_with_retry "${HEAD_IP}:${RAY_PORT}" 8

else
    # ?? EXTRA WORKER (joins only after the first vLLM serve has run) ??
    ray stop -f 2>&1 || true
    rm -rf /data/tmp/ray/session_* || true
    warm_lustre_cache "${MODEL_NAME}"

    echo "[${my_node}] node idx ${my_idx}: waiting for signal to join Ray after first serve..."
    while [[ ! -f "${SIGNAL_FILE}" ]]; do sleep 5; done

    echo "[${my_node}] signal received, joining Ray"
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    join_ray_with_retry "${HEAD_IP}:${RAY_PORT}" 8
fi
'

echo "Job ${SLURM_JOB_ID} completed. Logs in ${RUN_DIR}"
