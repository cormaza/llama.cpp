#!/usr/bin/env bash
# ==============================================================================
# start-ornith.sh - Launcher for Ornith-1.5-9B-MTP (High-Speed Single-Agent Stack)
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Ornith-1.5-9B-MTP-Q5_K_M.gguf"
ALT_MODEL="${SCRIPT_DIR}/models/Ornith-1.5-9B-MTP-Q6_K.gguf"
Q8_MODEL="${SCRIPT_DIR}/models/Ornith-1.5-9B-MTP-Q8_0.gguf"
MODEL_PATH=""
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
ENABLE_MTP=1
DRAFT_N_MAX=3     # Optimal draft depth for Ornith MTP
ALIAS="ornith-1.5-9b"

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Ornith-1.5-9B-MTP optimized for Single/Multi-Agent Deep Reasoning & Fast Coding,
with Froggeric Fixed Chat Templates, 100% GPU offload on AMD Radeon RX 9060 XT (16GB VRAM),
built-in MTP speculative decoding, and official Qwen 3.5 coding sampling parameters.

Options:
  -m, --model PATH        Path to Ornith GGUF model
  -a, --alias NAMES       Model alias for API clients (default: ornith-1.5-9b,ornith,gpt-4o,qwen)
  -c, --context, --ctx-slot N  Context per slot (default: 131072 for <=2 slots, 65536 for 4 slots, 32768 for 8 slots)
  --slots N               Number of parallel agent slots (default: 1; use 2, 4, 8 for multi-agent)
  --thinking              Enable reasoning mode (default; uses temp 0.6, top_p 0.95, top_k 20)
  --no-thinking           Disable reasoning mode (direct agent output; uses temp 0.7, top_p 0.80, top_k 20, presence 1.5)
  --temp N                Sampling temperature override (default: 0.6 with thinking, 0.7 without thinking)
  --top-p N               Top-p sampling override (default: 0.95 with thinking, 0.80 without thinking)
  --top-k N               Top-k sampling override (default: 20)
  --presence-penalty N    Presence penalty override (default: 0.0 with thinking, 1.5 without thinking)
  -p, --port PORT         HTTP server port (default: 8080)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --draft-n N             MTP draft depth (default: 3, optimal for throughput)
  --no-mtp                Disable MTP speculative decoding
  -h, --help              Show this help message

Examples:
  ./start-ornith.sh               # Default thinking mode (temp 0.6, top_p 0.95, deepseek reasoning)
  ./start-ornith.sh --no-thinking # Direct fast coding (temp 0.7, top_p 0.80, presence 1.5)
  ./start-ornith.sh --slots 4
  ./start-ornith.sh --slots 8 -c 32768
  ./start-ornith.sh --temp 0.7
  ./start-ornith.sh -c 262144
  ./start-ornith.sh -m ./models/Ornith-1.5-9B-MTP-Q6_K.gguf
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -a|--alias)
            ALIAS="$2"
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
        --kv-quant)
            KV_QUANT="$2"
            shift 2
            ;;
        --draft-n)
            DRAFT_N_MAX="$2"
            shift 2
            ;;
        --no-mtp)
            ENABLE_MTP=0
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
echo -e "${BOLD}${CYAN}  Ornith 1.5 9B High-Speed Coding Server (MTP / ROCm)  ${NC}"
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
    elif [[ -f "${Q8_MODEL}" ]]; then
        MODEL_PATH="${Q8_MODEL}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*ornith*.gguf" ! -iname "mmproj*" ! -iname "*head*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            echo -e "
Found existing Ornith models in ./models/:"
            select opt in "${FOUND_MODELS[@]}" "Descargar Ornith 1.5 9B" "Salir"; do
                if [[ -n "${opt}" && -f "${opt}" ]]; then
                    MODEL_PATH="${opt}"
                    break
                elif [[ "${opt}" == "Descargar Ornith 1.5 9B" ]]; then
                    ./scripts/download-ornith.sh
                    exit 0
                else
                    exit 1
                fi
            done
        else
            echo -e "${YELLOW}[WARN] No Ornith GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-ornith.sh${NC} to download Ornith-1.5-9B-MTP."
            exit 1
        fi
    fi
fi

# 3. Context calculation per slot & MTP Setup
if [[ -n "${CUSTOM_CTX}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX}"
elif [[ "${SLOTS}" -le 2 ]]; then
    CTX_PER_SLOT=131072 # 128k context per slot for 1-2 slots
elif [[ "${SLOTS}" -le 4 ]]; then
    CTX_PER_SLOT=65536  # 64k context per slot for 3-4 slots
else
    CTX_PER_SLOT=32768  # 32k context per slot for 8 slots (fits in 16GB VRAM)
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

MTP_STATUS="Disabled"
MTP_ARGS=()
if [[ "${ENABLE_MTP}" -eq 1 ]]; then
    MTP_STATUS="Active (Built-in MTP Head, draft_n_max=${DRAFT_N_MAX})"
    MTP_ARGS+=("--spec-type" "draft-mtp" "--spec-draft-n-max" "${DRAFT_N_MAX}")
fi

# 4. Template & Sampling Configuration (Froggeric Qwen 3.5 Recommendations)
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-0.6}" # Official Qwen 3.5 coding default
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Froggeric Fixed Jinja, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}" # Official Qwen non-thinking default
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct agent mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "none")
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}100% on AMD Radeon RX 9060 XT (All 34 layers offloaded)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${MTP_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}Froggeric Qwen-Fixed v22.5 (--jinja enabled)${NC}"
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"
echo -e "\n${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------\n"

# Enable prompt and token stream exposure in /slots for monitor drill-down
export LLAMA_SERVER_SLOTS_DEBUG=1

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
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl 99 \
    -fit off \
    -fa auto \
    -t 8 \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    "${MTP_ARGS[@]}"
