#!/usr/bin/env bash
set -Eeuo pipefail

source "${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/config.sh"
source "${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/helpers.sh"

HOST="${BENCH_HOST:-${HOST:-localhost}}"
WAIT_FOR_SERVER="${WAIT_FOR_SERVER:-true}"
BENCH_LOG_FILE="${BENCH_LOG_FILE:-}"

cmd=(
    vllm bench serve
    --model "${MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --dataset-name random
    --random-input-len "${RANDOM_INPUT_LEN}"
    --random-output-len "${RANDOM_OUTPUT_LEN}"
    --num-prompts "${NUM_PROMPTS}"
    --max-concurrency "${MAX_CONCURRENCY}"
)

if [[ "${WAIT_FOR_SERVER}" == "true" ]]; then
    wait_for_health "${HOST}" "${PORT}" "${SERVER_WAIT_TIMEOUT}"
fi

printf 'Running benchmark: prompts=%s concurrency=%s input=%s output=%s\n' \
    "${NUM_PROMPTS}" "${MAX_CONCURRENCY}" "${RANDOM_INPUT_LEN}" "${RANDOM_OUTPUT_LEN}"

if [[ -n "${BENCH_LOG_FILE}" ]]; then
    mkdir -p "$(dirname "${BENCH_LOG_FILE}")"
    {
        printf 'Benchmark started at %s\n' "$(date --iso-8601=seconds)"
        "${cmd[@]}"
        printf '\nBenchmark completed at %s\n' "$(date --iso-8601=seconds)"
    } 2>&1 | tee "${BENCH_LOG_FILE}"
else
    exec "${cmd[@]}"
fi

