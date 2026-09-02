#!/usr/bin/env bash
# ==============================================================================
# start-gemma-max-context.sh - Single-Agent High-Speed & 128k/256k Context Server
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/build-amd/bin"
SERVER_BIN="${BIN_DIR}/llama-server"

DEFAULT_MODEL="${SCRIPT_DIR}/models/gemma-4-12b-it-UD-Q4_K_XL.gguf"
DEFAULT_MTP="${SCRIPT_DIR}/models/mtp-gemma-4-12b-it-Q8_0.gguf"
MODEL_PATH="${DEFAULT_MODEL}"
MTP_PATH="${DEFAULT_MTP}"
ENABLE_MTP=1
HOST="0.0.0.0"
PORT=8080
CTX_SIZE=131072
KV_QUANT="q4_0"
THREADS=4
ENABLE_CTX_SHIFT=1
ALIAS="gemma-4-12b,gemma-4,gemma,gpt-4o"

LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Gemma 4 12B optimized for Single-Agent High-Speed Execution (OMP),
with maximized context window (128k default, up to 256k), 4 CPU threads, automatic context
shifting to prevent agent halts, and MTP speculative acceleration (~42 t/s).

Options:
  -a, --alias NAMES       Model alias for API clients (default: gemma-4-12b,gemma-4,gemma,gpt-4o)
  -m, --model PATH        Path to GGUF model (default: ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf)
  --mtp PATH              Path to MTP draft model (default: ./models/mtp-gemma-4-12b-it-Q8_0.gguf)
  --no-mtp                Disable MTP speculative decoding
  -c, --context N         Context window size (default: 131072 / 128k; supports 262144 / 256k)
  -t, --threads N         Number of CPU threads (default: 4)
  --no-context-shift      Disable automatic context shifting
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  -h, --help              Show this help message

Examples:
  ./start-gemma-max-context.sh               # 1 slot x 128k context with MTP (~42 t/s)
  ./start-gemma-max-context.sh -c 262144     # 1 slot x 256k deep context with MTP
  ./start-gemma-max-context.sh -t 2          # 1 slot x 128k with only 2 CPU threads
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
        --mtp)
            MTP_PATH="$2"
            shift 2
            ;;
        --no-mtp)
            ENABLE_MTP=0
            shift
            ;;
        -c|--context)
            CTX_SIZE="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        --no-context-shift)
            ENABLE_CTX_SHIFT=0
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
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
echo -e "${BOLD}${CYAN}  Gemma 4 Max-Context Agent Server (ROCm / HIP)       ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${SERVER_BIN}${NC}"
    echo -e "Please build first: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo -e "${RED}[ERROR] Model not found at: ${MODEL_PATH}${NC}"
    echo -e "Run ${CYAN}./scripts/download-gemma.sh${NC} first."
    exit 1
fi

MTP_ARGS=()
MTP_STATUS="Disabled"
if [[ "${ENABLE_MTP}" -eq 1 && -f "${MTP_PATH}" ]]; then
    MTP_STATUS="Active (Multi-Token Prediction: $(basename "${MTP_PATH}"))"
    MTP_ARGS+=("--spec-type" "draft-mtp" "-md" "${MTP_PATH}" "-ngld" "99")
fi

CTX_SHIFT_ARGS=()
if [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite generation / continuous agent support)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Mode:${NC}                ${GREEN}Single Slot (1 Dedicated Agent / Max Context)${NC}"
echo -e "${BOLD}Context Size:${NC}        ${GREEN}${CTX_SIZE} tokens ($(( CTX_SIZE / 1024 ))k tokens)${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}${THREADS} threads (-t ${THREADS})${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}100% on AMD Radeon RX 9060 XT (-ngl 99 -fa auto)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${MTP_STATUS}${NC}"
echo -e "${BOLD}Anti-Loop Samplers:${NC}  ${GREEN}DRY (mult 0.8, base 1.75, len 2) + Repeat Penalty 1.1 + Temp 0.7${NC}"
echo -e ""
echo -e "${BOLD}${YELLOW}=== Connection Info (for OMP / OpenCode / Cursor) ===${NC}"
echo -e "  Endpoint:          ${CYAN}http://127.0.0.1:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://127.0.0.1:${PORT}/v1${NC}"
echo -e "  Network URL:       ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "------------------------------------------------------\n"

exec "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${CTX_SIZE}" \
    -np 1 \
    -b 2048 \
    -ub 512 \
    -cb \
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl 99 \
    -fa auto \
    -t "${THREADS}" \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    --dry-multiplier 0.8 \
    --dry-base 1.75 \
    --dry-allowed-length 2 \
    --dry-penalty-last-n 256 \
    "${CTX_SHIFT_ARGS[@]}" \
    "${MTP_ARGS[@]}"
