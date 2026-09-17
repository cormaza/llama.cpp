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
SLOTS=1
CUSTOM_CTX=""
CUSTOM_CTX_SLOT=""
CUSTOM_TEMP=""
CUSTOM_TOP_P=""
CUSTOM_TOP_K=""
CUSTOM_PRESENCE=""
KV_QUANT="q4_0"
THREADS=4
ENABLE_CTX_SHIFT=1
ENABLE_KV_UNIFIED=0
ALIAS="gemma-4-12b"

# Helper function to parse human-readable token notation (e.g., 32k, 64k, 128k, 256k)
parse_tokens() {
    local val="${1,,}"
    val="${val//[[:space:]]/}"
    if [[ "${val}" =~ ^([0-9]+)k$ ]]; then
        echo $(( ${BASH_REMATCH[1]} * 1024 ))
    elif [[ "${val}" =~ ^([0-9]+)m$ ]]; then
        echo $(( ${BASH_REMATCH[1]} * 1024 * 1024 ))
    elif [[ "${val}" =~ ^[0-9]+$ ]]; then
        echo "${val}"
    else
        echo "${val}"
    fi
}

LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Gemma 4 12B optimized for High-Speed Agentic Workflows,
featuring MTP speculative acceleration (~42 t/s), continuous batching,
and automatic context-shifting for continuous agent operations (e.g. OMP, OpenCode).

Options:
  -a, --alias NAMES             Model alias for API clients (default: gemma-4-12b,gemma-4,gemma,gpt-4o)
  -m, --model PATH              Path to GGUF model (default: ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf)
  --mtp PATH                    Path to MTP draft model (default: ./models/mtp-gemma-4-12b-it-Q8_0.gguf)
  --no-mtp                      Disable MTP speculative decoding
  -c, --context, --total-ctx N  Total context pool across all slots (default: 131072; supports 64k, 128k, 256k)
  --ctx-slot N                  Context per slot (e.g. 32k, 64k; total = slots * ctx_slot)
  --slots, -np N                Number of parallel agent slots (default: 1; use 2, 4, 8 for multi-agent)
  -kvu, --kv-unified            Enable dynamic unified KV cache pool shared across all slots
  --temp N                      Sampling temperature (default: 0.2, low/precise for coding)
  --top-p N                     Top-p sampling (default: 0.95)
  --top-k N                     Top-k sampling (default: 40)
  --presence-penalty N          Presence penalty (default: 0.0, avoids distorted paths/commands)
  -t, --threads N               Number of CPU threads (default: 4)
  --no-context-shift            Disable automatic context shifting
  -p, --port PORT               HTTP server port (default: 8080)
  --host HOST                   Host address to bind (default: 0.0.0.0)
  -h, --help                    Show this help message

Examples:
  ./start-gemma-max-context.sh                            # 1 slot x 128k context with MTP (~42 t/s)
  ./start-gemma-max-context.sh --slots 4                  # 4 slots x 32k context with MTP (128k total)
  ./start-gemma-max-context.sh --slots 4 --ctx-slot 64k   # 4 slots x 64k context (256k total pool)
  ./start-gemma-max-context.sh --slots 8 --ctx-slot 16k   # 8 slots x 16k context for agent swarms
  ./start-gemma-max-context.sh -c 256k                    # 1 slot x 256k deep context
  ./start-gemma-max-context.sh -kvu -c 128k --slots 4     # Unified 128k KV pool shared dynamically
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
        -c|--context|--ctx|--total-ctx|--total-context)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        --ctx-slot|--slot-ctx|--context-slot)
            CUSTOM_CTX_SLOT="$2"
            shift 2
            ;;
        --slots|-np|--parallel)
            SLOTS="$2"
            shift 2
            ;;
        -kvu|--kv-unified)
            ENABLE_KV_UNIFIED=1
            shift
            ;;
        --temp|--temperature)
            CUSTOM_TEMP="$2"
            shift 2
            ;;
        --top-p)
            CUSTOM_TOP_P="$2"
            shift 2
            ;;
        --top-k)
            CUSTOM_TOP_K="$2"
            shift 2
            ;;
        --presence-penalty)
            CUSTOM_PRESENCE="$2"
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

TEMPERATURE="${CUSTOM_TEMP:-0.2}"
TOP_P="${CUSTOM_TOP_P:-0.95}"
TOP_K="${CUSTOM_TOP_K:-40}"
PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"

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

# 3. Context calculation per slot & MTP Setup
if [[ -n "${CUSTOM_CTX}" ]]; then
    CUSTOM_CTX="$(parse_tokens "${CUSTOM_CTX}")"
fi
if [[ -n "${CUSTOM_CTX_SLOT}" ]]; then
    CUSTOM_CTX_SLOT="$(parse_tokens "${CUSTOM_CTX_SLOT}")"
fi

if [[ -n "${CUSTOM_CTX_SLOT}" && -n "${CUSTOM_CTX}" ]]; then
    # Both explicitly specified: user pinned slot context and total pool
    CTX_PER_SLOT="${CUSTOM_CTX_SLOT}"
    TOTAL_CTX="${CUSTOM_CTX}"
elif [[ -n "${CUSTOM_CTX_SLOT}" ]]; then
    # Per-slot specified: total pool is slots * per_slot
    CTX_PER_SLOT="${CUSTOM_CTX_SLOT}"
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
elif [[ -n "${CUSTOM_CTX}" ]]; then
    # Total context pool specified: per-slot is total / slots
    TOTAL_CTX="${CUSTOM_CTX}"
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
else
    # Default: 131072 total context pool
    TOTAL_CTX=131072
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# Ensure minimum viable context per slot
if [[ "${CTX_PER_SLOT}" -lt 1024 ]]; then
    CTX_PER_SLOT=1024
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
fi

# Dynamic batch sizes: clamp batch size to total context if context is small
BATCH_SIZE=2048
UBATCH_SIZE=512
if [[ "${TOTAL_CTX}" -lt "${BATCH_SIZE}" ]]; then
    BATCH_SIZE="${TOTAL_CTX}"
fi
if [[ "${BATCH_SIZE}" -lt "${UBATCH_SIZE}" ]]; then
    UBATCH_SIZE="${BATCH_SIZE}"
fi

# Unified KV Cache Configuration
KV_UNIFIED_ARGS=()
KV_UNIFIED_STATUS="Dedicated per slot"
if [[ "${ENABLE_KV_UNIFIED}" -eq 1 ]]; then
    KV_UNIFIED_ARGS+=("-kvu")
    if [[ -n "${CUSTOM_CTX_SLOT}" ]]; then
        KV_UNIFIED_ARGS+=("--kv-unified-per-slot" "${CTX_PER_SLOT}")
        KV_UNIFIED_STATUS="Unified shared pool (max ${CTX_PER_SLOT} per slot)"
    else
        KV_UNIFIED_STATUS="Unified shared pool (dynamic)"
    fi
fi

# MTP Setup
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
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}Native Google Gemma 4 (--jinja enabled)${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}${THREADS} threads (-t ${THREADS})${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}KV Cache Allocation:${NC} ${GREEN}${KV_UNIFIED_STATUS}${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub ${UBATCH_SIZE}, -b ${BATCH_SIZE})${NC}"
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}100% on AMD Radeon RX 9060 XT (-ngl 99 -fa auto)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${MTP_STATUS}${NC}"
echo -e ""
echo -e "${BOLD}${YELLOW}=== Connection Info (for OMP / OpenCode / Cursor) ===${NC}"
echo -e "  Endpoint:          ${CYAN}http://127.0.0.1:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://127.0.0.1:${PORT}/v1${NC}"
echo -e "  Network URL:       ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "------------------------------------------------------\n"

# Enable prompt and token stream exposure in /slots for monitor drill-down
export LLAMA_SERVER_SLOTS_DEBUG=1

exec "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --alias "${ALIAS}" \
    --jinja \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${TOTAL_CTX}" \
    -np "${SLOTS}" \
    -b "${BATCH_SIZE}" \
    -ub "${UBATCH_SIZE}" \
    -cb \
    "${KV_UNIFIED_ARGS[@]}" \
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl 99 \
    -fa auto \
    -t "${THREADS}" \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${CTX_SHIFT_ARGS[@]}" \
    "${MTP_ARGS[@]}"
