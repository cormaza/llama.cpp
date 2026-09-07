#!/usr/bin/env bash
# ==============================================================================
# start-spark.sh - Spark-X2.5-4B Parallel Agent Server (ROCm / HIP)
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Spark-X2.5-4B-Q4_K_M.gguf"
ALT_MODEL="${SCRIPT_DIR}/models/Spark-X2.5-4B-Q5_K_M.gguf"
MODEL_PATH=""
HOST="0.0.0.0"
PORT=8080
SLOTS=4
CUSTOM_CTX=""
ALIAS="spark-x2.5-4b,spark-4b,spark,gpt-4o,qwen"
THREADS=8
ENABLE_CTX_SHIFT=1
TEMPERATURE=0.2

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Spark-X2.5-4B optimized for High-Speed Agentic Workflows,
featuring native 1M hybrid attention, continuous parallel slots, chunked prefill,
and automatic context-shifting for agent tools (e.g. OpenCode, OMP).

Options:
  -a, --alias NAMES       Model alias for API clients (default: spark-x2.5-4b,spark-4b,spark,gpt-4o,qwen)
  -m, --model PATH        Path to GGUF model (default: ./models/Spark-X2.5-4B-Q4_K_M.gguf)
  -c, --ctx-slot N        Context per slot (default: 131072 for <=4 slots, 65536 for 8 slots)
  --slots N               Number of parallel agent slots (default: 4; use 1 for single-agent deep context)
  --temp N                Sampling temperature (default: 0.2, low/precise for coding)
  -t, --threads N         Number of CPU threads (default: 8)
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --context-shift         Enable context shifting for continuous agent operations (default: enabled)
  --no-context-shift      Disable context shifting
  -h, --help              Show this help message

Examples:
  ./start-spark.sh                     # 4 agent slots x 128k context (512k pool total in 16GB VRAM)
  ./start-spark.sh --slots 8           # 8 parallel agent slots x 64k context
  ./start-spark.sh --slots 1 -c 262144 # 1 slot x 256k single-agent maximized context
  ./start-spark.sh --temp 0.5          # Custom temperature
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--alias)
            ALIAS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--ctx-slot)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        --slots)
            SLOTS="$2"
            shift 2
            ;;
        --temp|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
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
        --context-shift)
            ENABLE_CTX_SHIFT=1
            shift
            ;;
        --no-context-shift)
            ENABLE_CTX_SHIFT=0
            shift
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
echo -e "${BOLD}${CYAN}  Spark-X2.5-4B Agent Server (ROCm / HIP Accelerated) ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

# 1. Check binaries
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${SERVER_BIN}${NC}"
    echo -e "Please build first: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

# 2. Select Model
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -f "${DEFAULT_MODEL}" ]]; then
        MODEL_PATH="${DEFAULT_MODEL}"
    elif [[ -f "${ALT_MODEL}" ]]; then
        MODEL_PATH="${ALT_MODEL}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*spark*4b*.gguf" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            echo -e "
Select a model from ./models/:"
            select opt in "${FOUND_MODELS[@]}" "Descargar Spark-X2.5-4B" "Salir"; do
                if [[ -n "${opt}" && -f "${opt}" ]]; then
                    MODEL_PATH="${opt}"
                    break
                elif [[ "${opt}" == "Descargar Spark-X2.5-4B" ]]; then
                    ./scripts/download-spark.sh
                    exit 0
                else
                    exit 1
                fi
            done
        else
            echo -e "${YELLOW}[WARN] No Spark-X2.5-4B model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-spark.sh${NC} to download the model."
            exit 1
        fi
    fi
fi

# 3. Context Calculation
if [[ -n "${CUSTOM_CTX}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX}"
elif [[ "${SLOTS}" -le 4 ]]; then
    CTX_PER_SLOT=131072 # 128k context per slot (512k total across 4 slots)
else
    CTX_PER_SLOT=65536  # 64k context per slot for 8 slots
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

CTX_SHIFT_ARGS=()
if [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Continuous agent support / auto context shift)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Architecture:${NC}        ${GREEN}Spark2_5 (Hybrid 512-token SWA + Full Attn, 1M Context)${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}${THREADS} threads (-t ${THREADS})${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub 512, -b 2048)${NC}"
echo -e "${BOLD}KV Cache Quant:${NC}      ${GREEN}Q4_0 (-ctk q4_0 -ctv q4_0)${NC}"
echo -e "${BOLD}Temperature:${NC}         ${GREEN}${TEMPERATURE} (low/precise for coding)${NC}"
echo -e "${BOLD}Tool Calling:${NC}        ${GREEN}Native Jinja Template (--jinja enabled)${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}100% on AMD Radeon RX 9060 XT (-ngl 99 -fa auto)${NC}"
echo -e "
${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------
"

# Enable prompt and token stream exposure in /slots for monitor drill-down
export LLAMA_SERVER_SLOTS_DEBUG=1

exec "${SERVER_BIN}"     -m "${MODEL_PATH}"     --alias "${ALIAS}"     --host "${HOST}"     --port "${PORT}"     -c "${TOTAL_CTX}"     -np "${SLOTS}"     -b 2048     -ub 512     -cb     -ctk q4_0     -ctv q4_0     -ngl 99     -fa auto     --jinja     -t "${THREADS}"     --temp "${TEMPERATURE}"     "${CTX_SHIFT_ARGS[@]}"
