#!/usr/bin/env bash
# ==============================================================================
# start-agent-server.sh - Launcher for 8-Slot Parallel Agent Stack (ROCm/HIP)
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
DEFAULT_MTP="${SCRIPT_DIR}/models/mtp-gemma-4-12b-it-Q8_0.gguf"
MODEL_PATH="${DEFAULT_MODEL}"
MTP_PATH=""
ENABLE_MTP=1
HOST="0.0.0.0"
PORT=8080
SLOTS=8
CUSTOM_CTX=""
ALIAS="gemma-4-12b,gemma-4,gemma,gpt-4o"
THREADS=4
ENABLE_CTX_SHIFT=1

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server optimized for AMD Radeon RX 9060 XT (16GB VRAM) & Intel Core Ultra 7 265K.
Configured with zero-freeze chunked prefill, DRY anti-loop protection, MTP speculative decoding (~42 t/s),
and automatic context-shifting for continuous agent operation (e.g. OMP).

Options:
  -a, --alias NAMES       Model alias for API clients (default: gemma-4-12b,gemma-4,gemma,gpt-4o)
  -m, --model PATH        Path to GGUF model (default: ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf)
  --mtp PATH              Path to MTP draft model (default: ./models/mtp-gemma-4-12b-it-Q8_0.gguf)
  --no-mtp                Disable MTP (allows 128k context per slot without VRAM overflow)
  -c, --ctx-slot N        Context per slot (default: 131072 for <=4 slots, 65536 for 8 slots)
  -t, --threads N         Number of CPU threads (default: 4, reduced to avoid CPU contention)
  --context-shift         Enable context shifting for infinite text generation (default: enabled)
  --no-context-shift      Disable context shifting
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --slots N               Number of parallel slots (default: 8; use 1 for single-agent max context)
  -h, --help              Show this help message

Examples:
  ./start-agent-server.sh                     # 8 slots x 64k with MTP (~42 t/s in 16GB VRAM)
  ./start-agent-server.sh --slots 1           # 1 slot x 128k maximized context for OMP agent
  ./start-agent-server.sh --slots 1 -c 262144 # 1 slot x 256k ultra-deep context
  ./start-agent-server.sh --no-mtp            # 8 slots x 128k without MTP
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
        -c|--ctx-slot)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
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
echo -e "${BOLD}${CYAN}  8-Slot Parallel Agent Server (ROCm / HIP)           ${NC}"
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
    
    FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -name "*.gguf" ! -name "mtp-*" 2>/dev/null || true))
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

# 3. Check MTP draft model & Context calculation
MTP_ARGS=()
MTP_STATUS="Disabled"

if [[ "${ENABLE_MTP}" -eq 1 ]]; then
    if [[ -z "${MTP_PATH}" && -f "${DEFAULT_MTP}" ]]; then
        MTP_PATH="${DEFAULT_MTP}"
    fi

    if [[ -n "${MTP_PATH}" && -f "${MTP_PATH}" ]]; then
        MTP_STATUS="Active (Multi-Token Prediction: $(basename "${MTP_PATH}"))"
        MTP_ARGS+=("--spec-type" "draft-mtp" "-md" "${MTP_PATH}" "-ngld" "99")
        if [[ -n "${CUSTOM_CTX}" ]]; then
            CTX_PER_SLOT="${CUSTOM_CTX}"
        elif [[ "${SLOTS}" -le 4 ]]; then
            # Maximize context when using 1-4 slots (e.g. 128k context for single/quad agent)
            CTX_PER_SLOT=131072
        else
            # 8 slots: 64k per slot (512k total) to fit base + MTP compute buffer in 16GB VRAM
            CTX_PER_SLOT=65536
        fi
    else
        MTP_STATUS="Not found (Download via ./scripts/download-gemma.sh option 2)"
        CTX_PER_SLOT=${CUSTOM_CTX:-131072}
    fi
else
    CTX_PER_SLOT=${CUSTOM_CTX:-131072}
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

CTX_SHIFT_ARGS=()
if [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite generation / continuous agent support)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}${THREADS} threads (-t ${THREADS})${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub 512, -b 2048)${NC}"
echo -e "${BOLD}KV Cache Quant:${NC}      ${GREEN}Q4_0 (-ctk q4_0 -ctv q4_0)${NC}"
echo -e "${BOLD}Anti-Loop Samplers:${NC}  ${GREEN}DRY (mult 0.8, base 1.75, len 2) + Repeat Penalty 1.1 + Temp 0.7${NC}"
echo -e "${BOLD}MTP Speculative:${NC}     ${GREEN}${MTP_STATUS}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}100% on AMD Radeon RX 9060 XT (-ngl 99 -fa auto)${NC}"
echo -e "${BOLD}Server Endpoint:${NC}     ${CYAN}http://${HOST}:${PORT}${NC}"
echo -e "------------------------------------------------------\n"

exec "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${TOTAL_CTX}" \
    -np "${SLOTS}" \
    -b 2048 \
    -ub 512 \
    -cb \
    -ctk q4_0 \
    -ctv q4_0 \
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
