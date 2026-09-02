#!/usr/bin/env bash
# ==============================================================================
# start-agent-server.sh - Launcher for 8-Slot 128k Parallel Agent Stack (ROCm/HIP)
# ==============================================================================

set -euo pipefail

RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
BLUE='[0;34m'
CYAN='[0;36m'
BOLD='[1m'
NC='[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/build-amd/bin"
SERVER_BIN="${BIN_DIR}/llama-server"

DEFAULT_MODEL="${SCRIPT_DIR}/models/gemma-4-12b-it-UD-Q4_K_XL.gguf"
MODEL_PATH="${DEFAULT_MODEL}"
HOST="0.0.0.0"
PORT=8080
SLOTS=8
CTX_PER_SLOT=131072

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server optimized for AMD Radeon RX 9060 XT & Intel Core Ultra 7 265K.
Configured for 8 parallel agent slots with 128k context per slot, zero-freeze chunked prefill,
and DRY / Repeat Penalty anti-loop protection.

Options:
  -m, --model PATH        Path to GGUF model (default: ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf)
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --slots N               Number of parallel slots (default: 8)
  -h, --help              Show this help message

Examples:
  ./start-agent-server.sh
  ./start-agent-server.sh -p 8081
  ./start-agent-server.sh -m ./models/Qwen3.8-27B-UD-Q3_K_XL.gguf
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --slots)
            SLOTS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  8-Slot 128k Parallel Agent Server (ROCm / HIP)      ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

# 1. Check binaries
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${SERVER_BIN}${NC}"
    echo -e "Please build the project first by running: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

# 2. Check model file
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo -e "${YELLOW}[WARN] Model file not found at: ${MODEL_PATH}${NC}"
    
    # Check if other models exist in models/
    FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -name "*.gguf" 2>/dev/null || true))
    if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
        echo -e "
Found existing models in ./models/:\ introduce el número para usarlo:"
        select opt in "${FOUND_MODELS[@]}" "Descargar Gemma 4" "Salir"; do
            if [[ -n "${opt}" && -f "${opt}" ]]; then
                MODEL_PATH="${opt}"
                break
            elif [[ "${opt}" == "Descargar Gemma 4" ]]; then
                ./scripts/download-gemma.sh
                exit 0
            else
                exit 1
            fi
        done
    else
        echo -e "No GGUF models found in ./models/. Run ${CYAN}./scripts/download-gemma.sh${NC} first."
        exit 1
    fi
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens (128k)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens (1M tokens)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub 512, -b 2048)${NC}"
echo -e "${BOLD}KV Cache Quant:${NC}      ${GREEN}Q4_0 (-ctk q4_0 -ctv q4_0)${NC}"
echo -e "${BOLD}Anti-Loop Samplers:${NC}  ${GREEN}DRY (mult 0.8, base 1.75, len 2) + Repeat Penalty 1.1 + Temp 0.7${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}100% on AMD Radeon RX 9060 XT (-ngl 99 -fa auto)${NC}"
echo -e "${BOLD}Server Endpoint:${NC}     ${CYAN}http://${HOST}:${PORT}${NC}"
echo -e "------------------------------------------------------
"

exec "${SERVER_BIN}"     -m "${MODEL_PATH}"     --host "${HOST}"     --port "${PORT}"     -c "${TOTAL_CTX}"     -np "${SLOTS}"     -b 2048     -ub 512     -cb     -ctk q4_0     -ctv q4_0     -ngl 99     -fa auto     -t 8     --temp 0.7     --repeat-penalty 1.1     --dry-multiplier 0.8     --dry-base 1.75     --dry-allowed-length 2     --dry-penalty-last-n 256
