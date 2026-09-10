#!/usr/bin/env bash
# ==============================================================================
# start-k2-horizon-max-context.sh - Single-Agent Ultra-Deep Context Server
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/K2-Horizon-7B-Q4_K_M.gguf"
ALT_MODEL_IQ3="${SCRIPT_DIR}/models/K2-Horizon-7B-IQ3_XXS.gguf"
ALT_MODEL_Q8="${SCRIPT_DIR}/models/K2-Horizon-7B-Q8_0.gguf"
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
CUSTOM_NGL=""
ALIAS="k2-horizon-7b"
THREADS=8

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for K2-Horizon-7B configured for Ultra-Deep Context
(native support up to 524,288 tokens / 512k context), optimized for AMD ROCm/HIP.

Options:
  -a, --alias NAMES       Model alias for API clients (default: k2-horizon-7b,k2-horizon,gpt-4o)
  -m, --model PATH        Path to GGUF model (default: ./models/K2-Horizon-7B-Q4_K_M.gguf)
  -c, --context N         Context size (default: 131072 for 100% GPU, up to 524288 native)
  --thinking              Enable reasoning mode (default; uses temp 1.0, top_p 0.95)
  --no-thinking           Disable reasoning mode (direct response; uses temp 0.7, top_p 0.80)
  --temp N                Sampling temperature override (default: 1.0 with thinking, 0.7 without)
  --top-p N               Top-p sampling override (default: 0.95 with thinking, 0.80 without)
  --top-k N               Top-k sampling override (default: 40)
  --presence-penalty N    Presence penalty override (default: 0.0 with thinking, 1.5 without)
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                 Number of layers to offload to GPU (default: auto based on context)
  -t, --threads N         Number of CPU threads (default: 8)
  -h, --help              Show this help message

Examples:
  ./start-k2-horizon-max-context.sh               # 128k context (100% in 16GB VRAM)
  ./start-k2-horizon-max-context.sh -c 262144     # 256k context (ultra-deep document analysis)
  ./start-k2-horizon-max-context.sh -c 524288     # 512k native maximum context (hybrid GPU/CPU)
  ./start-k2-horizon-max-context.sh --no-thinking # Direct deep extraction without reasoning
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
        -c|--context)
            CUSTOM_CTX="$2"
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
        --kv-quant)
            KV_QUANT="$2"
            shift 2
            ;;
        --ngl)
            CUSTOM_NGL="$2"
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
echo -e "${BOLD}${CYAN}  K2-Horizon-7B Ultra-Deep Context Server (ROCm / HIP) ${NC}"
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
    elif [[ -f "${ALT_MODEL_IQ3}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ3}"
    elif [[ -f "${ALT_MODEL_Q8}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q8}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*horizon*.gguf" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            echo -e "\nSelect a model from ./models/:"
            select opt in "${FOUND_MODELS[@]}" "Descargar K2-Horizon" "Salir"; do
                if [[ -n "${opt}" && -f "${opt}" ]]; then
                    MODEL_PATH="${opt}"
                    break
                elif [[ "${opt}" == "Descargar K2-Horizon" ]]; then
                    ./scripts/download-k2-horizon.sh
                    exit 0
                else
                    exit 1
                fi
            done
        else
            echo -e "${YELLOW}[WARN] No K2-Horizon GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-k2-horizon.sh${NC} to download K2-Horizon-7B."
            exit 1
        fi
    fi
fi

# 3. Context & GPU Offload calculation
CONTEXT="${CUSTOM_CTX:-131072}"

if [[ -n "${CUSTOM_NGL}" ]]; then
    GPU_LAYERS="${CUSTOM_NGL}"
elif [[ "${CONTEXT}" -le 131072 ]]; then
    # 128k fits 100% in 16GB VRAM with Q4_0 KV
    GPU_LAYERS=99
    OFFLOAD_DESC="All 36 layers offloaded to GPU (100% VRAM)"
elif [[ "${CONTEXT}" -le 262144 ]]; then
    # 256k context (~11.4GB KV cache): offload 28 layers or 100% if compact quant
    GPU_LAYERS=30
    OFFLOAD_DESC="30 layers to GPU (Hybrid mode: 6 layers on CPU to prevent VRAM overflow)"
else
    # 512k context (~22.8GB KV cache): hybrid mode with system RAM
    GPU_LAYERS=20
    OFFLOAD_DESC="20 layers to GPU (Hybrid mode: remaining in RAM via Intel Ultra 7 265K)"
fi

# 4. Template & Sampling Configuration
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-1.0}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-40}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Native reasoning tags, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    if [[ -f "${SCRIPT_DIR}/models/templates/K2-Horizon.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/K2-Horizon.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-40}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct response mode, temp ${TEMPERATURE}, top_p ${TOP_P}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/K2-Horizon-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/K2-Horizon-no-thinking.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "none")
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Context Window:${NC}      ${GREEN}${CONTEXT} tokens ($(( CONTEXT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}1 slot (Dedicated deep ingestion)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}${OFFLOAD_DESC} (-ngl ${GPU_LAYERS} -fa auto)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub 512, -b 2048)${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}K2-Horizon Native Jinja (--jinja enabled)${NC}"
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
    -c "${CONTEXT}" \
    -np 1 \
    -b 2048 \
    -ub 512 \
    -cb \
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl "${GPU_LAYERS}" \
    -fa auto \
    -t "${THREADS}" \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}"
