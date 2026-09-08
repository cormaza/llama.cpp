#!/usr/bin/env bash
# ==============================================================================
# start-qwen-hf.sh - Run Unsloth Qwen3.8-27B-GGUF:Q4_0 with stack optimizations
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/build-amd/bin"
SERVER_BIN="${BIN_DIR}/llama-server"
CLI_BIN="${BIN_DIR}/llama-cli"

DEFAULT_HF_MODEL="unsloth/Qwen3.8-27B-GGUF:Q4_0"
HF_MODEL="${DEFAULT_HF_MODEL}"
HOST="0.0.0.0"
PORT=8080
SLOTS=1
CUSTOM_CTX=""
CUSTOM_TEMP=""
CUSTOM_TOP_P=""
CUSTOM_TOP_K=""
CUSTOM_PRESENCE=""
ENABLE_THINKING=1
KV_QUANT="q4_0"
GPU_LAYERS=50
ENABLE_SPEC=1
RUN_CLI=0

# Detect local LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Runs unsloth/Qwen3.8-27B-GGUF:Q4_0 with full AMD ROCm and Intel CPU optimizations,
with Froggeric Fixed Chat Templates and official sampling parameters for Qwen 3.8.
Automatically downloads and caches the model from Hugging Face Hub if needed.

Options:
  --hf REPO:QUANT        Hugging Face repo and quant (default: ${DEFAULT_HF_MODEL})
  -c, --context, --ctx-slot N  Context per slot (default: 131072 for 1 slot, 65536 for 2 slots, 32768 for 4 slots)
  --slots N              Number of parallel agent slots (default: 1; use 2, 4 for multi-agent)
  --thinking             Enable reasoning mode (default; uses temp 1.0, top_p 0.95, top_k 20)
  --no-thinking          Disable reasoning mode (direct agent output; uses temp 0.7, top_p 0.80, top_k 20, presence 1.5)
  --temp N               Sampling temperature override (default: 1.0 with thinking, 0.7 without thinking)
  --top-p N              Top-p sampling override (default: 0.95 with thinking, 0.80 without thinking)
  --top-k N              Top-k sampling override (default: 20)
  --presence-penalty N   Presence penalty override (default: 0.0 with thinking, 1.5 without thinking)
  -p, --port PORT        Server port (default: 8080)
  --host HOST            Bind address (default: 0.0.0.0)
  --kv-quant TYPE        KV cache precision: q4_0 (default) | q8_0 | f16
  --ngl N                Layers offloaded to GPU (default: 50)
  --no-spec              Disable N-Gram speculative decoding
  --cli                  Run interactive CLI chat instead of HTTP server
  -h, --help             Show this help message

Examples:
  ./start-qwen-hf.sh               # Default thinking mode (temp 1.0, top_p 0.95, deepseek reasoning)
  ./start-qwen-hf.sh --no-thinking # Direct fast coding (temp 0.7, top_p 0.80, presence 1.5)
  ./start-qwen-hf.sh --slots 2
  ./start-qwen-hf.sh --slots 4 -c 32768
  ./start-qwen-hf.sh --temp 0.6
  ./start-qwen-hf.sh --cli
  ./start-qwen-hf.sh --hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf)
            HF_MODEL="$2"
            shift 2
            ;;
        -c|--context|--ctx-slot)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        --slots)
            SLOTS="$2"
            shift 2
            ;;
        --thinking)
            ENABLE_THINKING=1
            shift
            ;;
        --no-thinking)
            ENABLE_THINKING=0
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
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --kv-quant)
            KV_QUANT="$2"
            shift 2
            ;;
        --ngl)
            GPU_LAYERS="$2"
            shift 2
            ;;
        --no-spec)
            ENABLE_SPEC=0
            shift
            ;;
        --cli)
            RUN_CLI=1
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
echo -e "${BOLD}${CYAN}  Qwen3.8-27B HF Runner (ROCm / HIP Accelerated)      ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

# Check binaries
TARGET_BIN="${SERVER_BIN}"
if [[ "${RUN_CLI}" -eq 1 ]]; then
    TARGET_BIN="${CLI_BIN}"
fi

if [[ ! -x "${TARGET_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${TARGET_BIN}${NC}"
    echo -e "Please build first: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

SPEC_ARGS=()
SPEC_STATUS="Disabled"
if [[ "${ENABLE_SPEC}" -eq 1 ]]; then
    SPEC_STATUS="Active (N-Gram Lookup, m=48)"
    SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
fi

# Context calculation per slot
if [[ -n "${CUSTOM_CTX}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX}"
elif [[ "${SLOTS}" -le 1 ]]; then
    CTX_PER_SLOT=131072 # 128k context for single slot
elif [[ "${SLOTS}" -le 2 ]]; then
    CTX_PER_SLOT=65536  # 64k context per slot for 2 slots
elif [[ "${SLOTS}" -le 4 ]]; then
    CTX_PER_SLOT=32768  # 32k context per slot for 4 slots
else
    CTX_PER_SLOT=16384  # 16k context per slot for 8 slots
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

# Template & Sampling Configuration (Froggeric Qwen-Fixed Recommendations)
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-1.0}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Froggeric Fixed Jinja, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct agent mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "none")
fi

echo -e "${BOLD}HF Model:${NC}            ${CYAN}${HF_MODEL}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}${GPU_LAYERS} layers -> AMD Radeon RX 9060 XT (-fa auto)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${SPEC_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}Froggeric Qwen-Fixed v22.5 (--jinja enabled)${NC}"
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}8 P-cores (Intel Core Ultra 7 265K)${NC}"

if [[ "${RUN_CLI}" -eq 1 ]]; then
    echo -e "${BOLD}Mode:${NC}                ${YELLOW}Interactive CLI${NC}"
    echo -e "------------------------------------------------------\n"
    exec "${CLI_BIN}" \
        -hf "${HF_MODEL}" \
        -c "${CTX_PER_SLOT}" \
        -b 2048 \
        -ub 512 \
        -ctk "${KV_QUANT}" \
        -ctv "${KV_QUANT}" \
        -ngl "${GPU_LAYERS}" \
        -fa auto \
        -t 8 \
        --temp "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --presence-penalty "${PRESENCE_PENALTY}" \
        "${SPEC_ARGS[@]}" \
        -co -cnv
else
    echo -e "${BOLD}Mode:${NC}                ${GREEN}Server (OpenAI API / Web UI)${NC}"
    echo -e "\n${BOLD}${YELLOW}=== Connection Endpoints ===${NC}"
    echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
    echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
    echo -e "  Localhost Base:    ${CYAN}http://127.0.0.1:${PORT}/v1${NC}"
    echo -e "------------------------------------------------------\n"

    # Enable prompt and token stream exposure in /slots for monitor drill-down
    export LLAMA_SERVER_SLOTS_DEBUG=1

    exec "${SERVER_BIN}" \
        -hf "${HF_MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        -c "${TOTAL_CTX}" \
        -np "${SLOTS}" \
        -b 2048 \
        -ub 512 \
        -cb \
        -ctk "${KV_QUANT}" \
        -ctv "${KV_QUANT}" \
        -ngl "${GPU_LAYERS}" \
        -fa auto \
        -t 8 \
        --temp "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --presence-penalty "${PRESENCE_PENALTY}" \
        "${JINJA_ARGS[@]}" \
        "${REASONING_ARGS[@]}" \
        "${SPEC_ARGS[@]}"
fi
