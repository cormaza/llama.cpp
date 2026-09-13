#!/usr/bin/env bash
# ==============================================================================
# start-oxcoder-9b-agent.sh - Launcher for OxCoder-9B Parallel Agent Stack
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/OxCoder-9B.Q4_K_M.gguf"
ALT_MODEL_Q5="${SCRIPT_DIR}/models/OxCoder-9B.Q5_K_M.gguf"
ALT_MODEL_Q8="${SCRIPT_DIR}/models/OxCoder-9B.Q8_0.gguf"
ALT_MODEL_IQ3="${SCRIPT_DIR}/models/OxCoder-9B.i1-IQ3_XXS.gguf"
MODEL_PATH=""
HOST="0.0.0.0"
PORT=8080
SLOTS=4
CUSTOM_CTX=""
CUSTOM_TEMP=""
CUSTOM_TOP_P=""
CUSTOM_TOP_K=""
CUSTOM_PRESENCE=""
ENABLE_THINKING=1
KV_QUANT="q4_0"
CUSTOM_NGL=""
ALIAS="oxcoder-9b"
THREADS=6
ENABLE_CTX_SHIFT=1
ENABLE_SPEC=1

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for OxCoder-9B (OrionLLM / Qwen 3.5 Hybrid) optimized for
Multi-Agent Coding Workflows, with Froggeric v21.3 Jinja templates, N-Gram speculative
decoding, and 100% GPU offload on AMD Radeon RX 9060 XT (16GB VRAM).

Options:
  -a, --alias NAMES       Model alias for API clients (default: oxcoder-9b,oxcoder,gpt-4o)
  -m, --model PATH        Path to GGUF model (default: ./models/OxCoder-9B.Q4_K_M.gguf)
  -c, --context, --ctx-slot N  Context per slot (default: 65536 for 4 slots, 32768 for 8 slots)
  --slots N               Number of parallel agent slots (default: 4; use 8 for large agent swarms)
  --thinking              Enable reasoning mode (default; uses temp 1.0, top_p 0.95, reasoning tags)
  --no-thinking           Disable reasoning mode (direct agent mode; uses temp 0.6, top_p 0.80)
  --temp N                Sampling temperature override (default: 1.0 with thinking, 0.6 without)
  --top-p N               Top-p sampling override (default: 0.95 with thinking, 0.80 without)
  --top-k N               Top-k sampling override (default: 40)
  --presence-penalty N    Presence penalty override (default: 0.0 with thinking, 1.5 without)
  -t, --threads N         Number of CPU threads (default: 6)
  --context-shift         Enable context shifting for continuous agent operation (default: enabled)
  --no-context-shift      Disable context shifting
  --no-spec               Disable N-Gram speculative decoding
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                 Number of layers to offload to GPU (default: 99, 100% GPU)
  -h, --help              Show this help message

Examples:
  ./start-oxcoder-9b-agent.sh                     # 4 slots x 64k with thinking (100% VRAM)
  ./start-oxcoder-9b-agent.sh --no-thinking       # Fast direct tool use / ClawEval mode
  ./start-oxcoder-9b-agent.sh --slots 8           # 8 slots x 32k for multi-agent teams
  ./start-oxcoder-9b-agent.sh --slots 2 -c 131072 # 2 slots x 128k for dual deep agents
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
        -c|--context|--ctx-slot)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        --slots)
            SLOTS="$2"
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
        --no-spec)
            ENABLE_SPEC=0
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
echo -e "${BOLD}${CYAN}  OxCoder-9B Multi-Slot Agent Server (ROCm / HIP)     ${NC}"
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
    elif [[ -f "${ALT_MODEL_Q5}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q5}"
    elif [[ -f "${ALT_MODEL_Q8}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q8}"
    elif [[ -f "${ALT_MODEL_IQ3}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ3}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*oxcoder*.gguf" ! -iname "*mmproj*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            echo -e "\nSelect a model from ./models/:"
            select opt in "${FOUND_MODELS[@]}" "Descargar OxCoder-9B" "Salir"; do
                if [[ -n "${opt}" && -f "${opt}" ]]; then
                    MODEL_PATH="${opt}"
                    break
                elif [[ "${opt}" == "Descargar OxCoder-9B" ]]; then
                    ./scripts/download-oxcoder-9b.sh
                    exit 0
                else
                    exit 1
                fi
            done
        else
            echo -e "${YELLOW}[WARN] No OxCoder-9B GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-oxcoder-9b.sh${NC} to download OxCoder-9B."
            exit 1
        fi
    fi
fi

# 3. Context calculation per slot (Qwen 3.5 hybrid KV cache is 4x smaller than standard transformers)
if [[ -n "${CUSTOM_CTX}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX}"
elif [[ "${SLOTS}" -le 2 ]]; then
    CTX_PER_SLOT=131072 # 128k context per slot for <=2 slots
elif [[ "${SLOTS}" -le 4 ]]; then
    CTX_PER_SLOT=65536  # 64k context per slot for 4 slots (256k total pool)
else
    CTX_PER_SLOT=32768  # 32k context per slot for 8 slots (256k total pool)
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

# 4. Context shift
CTX_SHIFT_ARGS=()
if [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite generation / continuous agent operation)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

# 5. Speculative Decoding (N-Gram Prompt Lookup)
SPEC_STATUS="Disabled"
SPEC_ARGS=()
if [[ "${ENABLE_SPEC}" -eq 1 ]]; then
    SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
    SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
fi

# 6. Template & Sampling Configuration
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-1.0}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-40}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Froggeric v21.3 reasoning tags, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    if [[ -f "${SCRIPT_DIR}/models/templates/OxCoder-9B.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/OxCoder-9B.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.6}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-40}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct ClawEval agent mode, temp ${TEMPERATURE}, top_p ${TOP_P}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/OxCoder-9B-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/OxCoder-9B-no-thinking.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "none")
fi

GPU_LAYERS="${CUSTOM_NGL:-99}"

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}All 32 layers offloaded to GPU (-ngl ${GPU_LAYERS} -fa auto)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub 512, -b 2048)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${SPEC_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}Froggeric v21.3 Qwen-Fixed (--jinja enabled)${NC}"
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
    -ngl "${GPU_LAYERS}" \
    -fa auto \
    -t "${THREADS}" \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${CTX_SHIFT_ARGS[@]}" \
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    "${SPEC_ARGS[@]}"
