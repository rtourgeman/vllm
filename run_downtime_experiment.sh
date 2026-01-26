#!/bin/bash
#
# Serve Downtime Experiment Runner
# ================================
# This script helps run reproducible downtime measurements.
#
# Usage:
#   ./run_downtime_experiment.sh <scenario> <serve_url> [options]
#
# Scenarios:
#   scale_up  - Measure downtime during scale-up
#   restart   - Measure downtime during stop/start
#   compare   - Compare results from both scenarios
#
# Examples:
#   # Run scale-up experiment (probe runs for 60s or until Ctrl+C)
#   ./run_downtime_experiment.sh scale_up http://localhost:8000/health
#
#   # Run restart experiment
#   ./run_downtime_experiment.sh restart http://localhost:8000/health
#
#   # Compare both scenarios after running experiments
#   ./run_downtime_experiment.sh compare
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_SCRIPT="${SCRIPT_DIR}/serve_downtime_measure.py"
RESULTS_DIR="${SCRIPT_DIR}/downtime_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default settings
PROBE_INTERVAL_MS=50       # 50ms between probes (20 probes/sec)
PROBE_TIMEOUT_MS=5000      # 5 second timeout
SERVE_URL=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}  Serve Downtime Measurement Tool${NC}"
    echo -e "${BLUE}=====================================${NC}"
}

print_usage() {
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  scale_up <url>  - Start probing for scale-up scenario"
    echo "  restart <url>   - Start probing for restart scenario"
    echo "  analyze <log>   - Analyze a specific log file"
    echo "  compare         - Compare scale_up vs restart results"
    echo ""
    echo "Options:"
    echo "  --interval <ms>  Probe interval in ms (default: 50)"
    echo "  --timeout <ms>   Request timeout in ms (default: 5000)"
    echo ""
    echo "Examples:"
    echo "  $0 scale_up http://localhost:8000/health"
    echo "  $0 restart http://localhost:8000/v1/health"
    echo "  $0 compare"
    echo ""
    echo "Workflow:"
    echo "  1. Terminal 1: $0 scale_up http://localhost:8000/health"
    echo "  2. Terminal 2: Trigger scale-up operation"
    echo "  3. Terminal 1: Wait for serve to recover, then Ctrl+C"
    echo "  4. Terminal 1: $0 restart http://localhost:8000/health"
    echo "  5. Terminal 2: Stop and restart the serve"
    echo "  6. Terminal 1: Wait for serve to recover, then Ctrl+C"
    echo "  7. $0 compare"
    echo ""
}

ensure_results_dir() {
    mkdir -p "${RESULTS_DIR}"
}

run_probe() {
    local scenario=$1
    local url=$2
    local log_file="${RESULTS_DIR}/${scenario}_${TIMESTAMP}.log"
    
    ensure_results_dir
    
    echo -e "${YELLOW}Starting probe for scenario: ${scenario}${NC}"
    echo -e "URL:      ${url}"
    echo -e "Interval: ${PROBE_INTERVAL_MS}ms"
    echo -e "Timeout:  ${PROBE_TIMEOUT_MS}ms"
    echo -e "Log file: ${log_file}"
    echo ""
    echo -e "${GREEN}Press Ctrl+C to stop after the serve recovers${NC}"
    echo ""
    
    # Create a symlink to the latest log for easy access
    ln -sf "${log_file}" "${RESULTS_DIR}/${scenario}_latest.log"
    
    python3 "${PROBE_SCRIPT}" probe \
        --url "${url}" \
        --interval "${PROBE_INTERVAL_MS}" \
        --timeout "${PROBE_TIMEOUT_MS}" \
        --log "${log_file}"
    
    echo ""
    echo -e "${GREEN}Probe complete. Analyzing results...${NC}"
    echo ""
    
    python3 "${PROBE_SCRIPT}" analyze --log "${log_file}"
}

analyze_log() {
    local log_file=$1
    
    if [[ ! -f "${log_file}" ]]; then
        echo -e "${RED}Error: Log file not found: ${log_file}${NC}"
        exit 1
    fi
    
    python3 "${PROBE_SCRIPT}" analyze --log "${log_file}"
}

compare_scenarios() {
    local scale_up_log="${RESULTS_DIR}/scale_up_latest.log"
    local restart_log="${RESULTS_DIR}/restart_latest.log"
    
    # Check if both logs exist
    if [[ ! -f "${scale_up_log}" ]]; then
        echo -e "${RED}Error: Scale-up log not found: ${scale_up_log}${NC}"
        echo "Run the scale_up scenario first."
        exit 1
    fi
    
    if [[ ! -f "${restart_log}" ]]; then
        echo -e "${RED}Error: Restart log not found: ${restart_log}${NC}"
        echo "Run the restart scenario first."
        exit 1
    fi
    
    echo -e "${YELLOW}Comparing scenarios...${NC}"
    echo ""
    
    python3 "${PROBE_SCRIPT}" compare \
        --log1 "${scale_up_log}" \
        --log2 "${restart_log}" \
        --label1 "Scale-up" \
        --label2 "Restart"
}

# Parse arguments
COMMAND=${1:-}

case "${COMMAND}" in
    scale_up|restart)
        shift
        SERVE_URL=${1:-}
        shift || true
        
        if [[ -z "${SERVE_URL}" ]]; then
            echo -e "${RED}Error: URL is required${NC}"
            print_usage
            exit 1
        fi
        
        # Parse optional arguments
        while [[ $# -gt 0 ]]; do
            case $1 in
                --interval)
                    PROBE_INTERVAL_MS=$2
                    shift 2
                    ;;
                --timeout)
                    PROBE_TIMEOUT_MS=$2
                    shift 2
                    ;;
                *)
                    echo -e "${RED}Unknown option: $1${NC}"
                    print_usage
                    exit 1
                    ;;
            esac
        done
        
        print_header
        run_probe "${COMMAND}" "${SERVE_URL}"
        ;;
    
    analyze)
        shift
        LOG_FILE=${1:-}
        if [[ -z "${LOG_FILE}" ]]; then
            echo -e "${RED}Error: Log file is required${NC}"
            print_usage
            exit 1
        fi
        print_header
        analyze_log "${LOG_FILE}"
        ;;
    
    compare)
        print_header
        compare_scenarios
        ;;
    
    -h|--help|help)
        print_header
        print_usage
        ;;
    
    *)
        print_header
        print_usage
        exit 1
        ;;
esac
