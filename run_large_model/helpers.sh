#!/usr/bin/env bash
# Shared helper functions for run_large_model scripts.

get_routable_ip() {
    hostname -I | tr ' ' '\n' | grep -v '^169\.254\.' | head -1
}

check_health() {
    local host="${1:-localhost}"
    local port="${2:-${PORT:-8006}}"
    python3 -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://${host}:${port}/health', timeout=5)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

wait_for_health() {
    local host="${1:-localhost}"
    local port="${2:-${PORT:-8006}}"
    local timeout="${3:-${SERVER_WAIT_TIMEOUT:-1200}}"
    local interval="${4:-30}"
    local waited=0

    echo "Waiting for vLLM health at http://${host}:${port}/health (timeout=${timeout}s)"
    until check_health "${host}" "${port}"; do
        if (( waited >= timeout )); then
            echo "ERROR: server not healthy after ${timeout}s" >&2
            return 1
        fi
        sleep "${interval}"
        waited=$((waited + interval))
        echo "  still waiting... ${waited}s"
    done
    echo "Server is healthy"
}

wait_for_ray_nodes() {
    local expected="${1}"
    local timeout="${2:-${RAY_WAIT_TIMEOUT:-300}}"
    local waited=0
    local node_count=0

    echo "Waiting for ${expected} Ray node(s) (timeout=${timeout}s)"
    while (( waited < timeout )); do
        node_count="$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
print(sum(1 for n in ray.nodes() if n.get('Alive')))
" 2>/dev/null || echo 0)"
        if (( node_count >= expected )); then
            echo "Ray cluster has ${node_count} node(s)"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo "  Ray nodes: ${node_count}/${expected} (${waited}s)"
    done

    echo "WARNING: only ${node_count}/${expected} Ray nodes after ${timeout}s" >&2
    return 1
}

join_ray_with_retry() {
    local address="${1}"
    local gpus="${2:-${GPUS_PER_NODE:-8}}"
    local max_attempts="${3:-30}"

    echo "Joining Ray at ${address}"
    sleep 15
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        ray stop -f 2>&1 || true
        sleep 2
        if ray start --address="${address}" --num-gpus="${gpus}" --block; then
            return 0
        fi
        echo "  Ray join attempt ${attempt}/${max_attempts} failed, retrying"
        sleep 10
    done

    echo "ERROR: failed to join Ray after ${max_attempts} attempts" >&2
    return 1
}
