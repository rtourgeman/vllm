#!/usr/bin/env bash
# Restart benchmark: bring up a Ray cluster once, then cycle vLLM through a
# schedule of GPU counts. Each cycle does: serve -> warmup -> close. The time
# for every "close + upload" transition is measured and summarized at the end.
#
# Default schedule "32 40 40 40" gives 3 close+upload measurements (the three
# transitions into the 40-GPU serves) -- i.e. the restart tested 3 times.
#
# Usage:
#   ./run_restart_bench.sh                          # schedule "32 40 40 40"
#   ./run_restart_bench.sh -s "32 40 40 40"         # explicit schedule
#   ./run_restart_bench.sh -s "40 40 40" -r 24      # 3x 40-GPU restarts
#   ./run_restart_bench.sh -s "32 40 40 40" -t 03:00:00
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

SCHEDULE="32 40 40 40"
REDUNDANT_SCHEDULE="0 24 24 24"
REDUNDANT_SINGLE=""

usage() {
    cat <<EOF
Usage: $0 [-s "GPU GPU ..."] [-R "RED RED ..."] [-r REDUNDANT] [-t TIME]

Optional:
  -s SCHEDULE     Space-separated GPU counts, one per serve cycle
                  (default: "${SCHEDULE}")
  -R REDUNDANT    Space-separated redundant-expert counts, one per serve,
                  aligned with -s (default: "${REDUNDANT_SCHEDULE}")
  -r REDUNDANT    Single redundant-expert count applied to every serve
                  (shortcut; overrides -R)
  -t TIME         Slurm time limit (default: ${TIME})
  -h              Show this help

Flow (per entry in SCHEDULE):
  upload vLLM serve -> run warmup -> close vLLM serve

Defaults give 32-GPU serves 0 redundant experts and 40-GPU serves 24.

Each serve writes its own log:  vllm_server_<idx>_<gpu>gpu.log
The "close + upload" time of every transition is timed and a summary is
printed at the end (and saved to restart_summary.txt).
EOF
}

while getopts ":s:R:r:t:h" opt; do
    case "${opt}" in
        s) SCHEDULE="${OPTARG}" ;;
        R) REDUNDANT_SCHEDULE="${OPTARG}" ;;
        r) REDUNDANT_SINGLE="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        h) usage; exit 0 ;;
        :) echo "ERROR: -${OPTARG} requires an argument" >&2; exit 2 ;;
        \?) echo "ERROR: invalid option -${OPTARG}" >&2; exit 2 ;;
    esac
done

# Validate schedule and find the max GPU count -> number of nodes to allocate.
read -ra _sched <<< "${SCHEDULE}"
(( ${#_sched[@]} >= 1 )) || { echo "ERROR: empty schedule" >&2; exit 1; }
MAX_GPU=0
for g in "${_sched[@]}"; do
    [[ "${g}" =~ ^[0-9]+$ ]] || { echo "ERROR: schedule entry '${g}' is not a number" >&2; exit 1; }
    (( g > 0 )) || { echo "ERROR: schedule entry '${g}' must be > 0" >&2; exit 1; }
    (( g % GPUS_PER_NODE == 0 )) || { echo "ERROR: schedule entry '${g}' must be divisible by ${GPUS_PER_NODE}" >&2; exit 1; }
    (( g > MAX_GPU )) && MAX_GPU="${g}"
done

# -r (single value) overrides -R by replicating across every serve.
if [[ -n "${REDUNDANT_SINGLE}" ]]; then
    REDUNDANT_SCHEDULE=""
    for _ in "${_sched[@]}"; do
        REDUNDANT_SCHEDULE+="${REDUNDANT_SINGLE} "
    done
    REDUNDANT_SCHEDULE="${REDUNDANT_SCHEDULE% }"
fi

# Validate redundant schedule: numeric and same length as the GPU schedule.
read -ra _red <<< "${REDUNDANT_SCHEDULE}"
(( ${#_red[@]} == ${#_sched[@]} )) || {
    echo "ERROR: redundant schedule '${REDUNDANT_SCHEDULE}' (${#_red[@]} entries) must match GPU schedule '${SCHEDULE}' (${#_sched[@]} entries)" >&2
    echo "       use -R \"...\" with one value per serve, or -r N for a single value" >&2
    exit 1
}
for r in "${_red[@]}"; do
    [[ "${r}" =~ ^[0-9]+$ ]] || { echo "ERROR: redundant entry '${r}' is not a number" >&2; exit 1; }
done

NODES=$(( MAX_GPU / GPUS_PER_NODE ))
LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y-%m-%d)"
mkdir -p "${LOG_DIR}"
JOB_TAG="restart_$(echo "${SCHEDULE}" | tr ' ' '-')gpu"

cat <<EOF
Submitting restart benchmark:
  schedule:   ${SCHEDULE}
  redundant:  ${REDUNDANT_SCHEDULE}
  serves:     ${#_sched[@]}  (${#_sched[@]} - 1 = $(( ${#_sched[@]} - 1 )) close+upload measurements)
  max GPUs:   ${MAX_GPU}
  nodes:      ${NODES} (${GPUS_PER_NODE} GPUs each)
  time:       ${TIME}
EOF

JOB_ID="$(sbatch --parsable \
    --nodes="${NODES}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --ntasks-per-node=1 \
    --exclusive \
    --partition="${BATCH_PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --job-name="${JOB_TAG}" \
    --output="${LOG_DIR}/${JOB_TAG}_%j.out" \
    --error="${LOG_DIR}/${JOB_TAG}_%j.err" \
    --export=ALL,SCRIPT_DIR="${SCRIPT_DIR}",SCHEDULE="${SCHEDULE}",MAX_GPU="${MAX_GPU}",REDUNDANT_SCHEDULE="${REDUNDANT_SCHEDULE}" \
    "${SCRIPT_DIR}/slurm_restart.sh"
)"

cat <<EOF

Submitted job ${JOB_ID}
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/${JOB_TAG}_${JOB_ID}.out
EOF
