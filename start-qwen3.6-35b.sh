#!/usr/bin/env bash
# ==============================================================================
# start-qwen3.6-35b.sh - Launcher for Qwen3.6-35B-A3B-MTP (ROCm / HIP)
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"
ALT_MODEL_IQ2M="${SCRIPT_DIR}/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
ALT_MODEL_Q2K="${SCRIPT_DIR}/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"
ALT_MODEL_IQ3="${SCRIPT_DIR}/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
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
TEMPLATE_CHOICE="froggeric"
KV_QUANT="q4_0"
ENABLE_MTP=1
DRAFT_N_MAX=2     # Optimal draft depth for Qwen3.6 MTP
ENABLE_CTX_SHIFT=1
THREADS=8
ALIAS="qwen3.6-35b"

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Qwen3.6-35B-A3B-MTP optimized for AMD Radeon RX 9060 XT (16GB VRAM),
featuring 35B MoE with 3B active parameters (A3B), built-in MTP speculative decoding,
Froggeric/Unsloth Jinja chat templates, and official Qwen 3.6 coding sampling parameters.

Options:
  -m, --model PATH        Path to Qwen3.6 GGUF model (default: ./models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf)
  -a, --alias NAMES       Model alias for API clients (default: qwen3.6-35b,qwen3.6,qwen,gpt-4o)
  -c, --context, --ctx-slot N  Context per slot (default: 131072 for <=2 slots, 65536 for 4 slots, 32768 for 8 slots)
  --slots N               Number of parallel agent slots (default: 1; use 2, 4, 8 for multi-agent)
  --thinking              Enable reasoning mode (default; uses temp 0.6, top_p 0.95, top_k 20)
  --no-thinking           Disable reasoning mode (direct agent output; uses temp 0.7, top_p 0.80, top_k 20, presence 1.5)
  --template TYPE         Chat template: froggeric (default) | unsloth | native
  --temp N                Sampling temperature override (default: 0.6 with thinking, 0.7 without thinking)
  --top-p N               Top-p sampling override (default: 0.95 with thinking, 0.80 without thinking)
  --top-k N               Top-k sampling override (default: 20)
  --presence-penalty N    Presence penalty override (default: 0.0 with thinking, 1.5 without thinking)
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --draft-n N             MTP draft depth (default: 2, optimal for Qwen3.6 throughput)
  --no-mtp                Disable MTP speculative decoding (enables multi-slot mode)
  --context-shift         Enable automatic context shifting (default: enabled)
  --no-context-shift      Disable automatic context shifting
  -t, --threads N         Number of CPU threads (default: 8)
  -h, --help              Show this help message

Examples:
  ./start-qwen3.6-35b.sh                      # 1 slot x 128k context with MTP & Froggeric template
  ./start-qwen3.6-35b.sh -c 262144            # 1 slot x 256k native deep context
  ./start-qwen3.6-35b.sh --no-thinking        # Fast direct coding without <think> tags
  ./start-qwen3.6-35b.sh --template unsloth   # Use official Unsloth Qwen3.6 template
  ./start-qwen3.6-35b.sh --slots 4 --no-mtp   # 4 parallel agent slots x 64k context
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
        --template)
            TEMPLATE_CHOICE="$2"
            shift 2
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
        --draft-n|--spec-draft-n-max)
            DRAFT_N_MAX="$2"
            shift 2
            ;;
        --no-mtp)
            ENABLE_MTP=0
            shift
            ;;
        --context-shift)
            ENABLE_CTX_SHIFT=1
            shift
            ;;
        --no-context-shift)
            ENABLE_CTX_SHIFT=0
            shift
            ;;
        -t|--threads)
            THREADS="$2"
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
echo -e "${BOLD}${CYAN}  Qwen3.6-35B-A3B Server (MoE A3B / MTP / ROCm)       ${NC}"
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
    elif [[ -f "${ALT_MODEL_IQ2M}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ2M}"
    elif [[ -f "${ALT_MODEL_Q2K}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q2K}"
    elif [[ -f "${ALT_MODEL_IQ3}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ3}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*qwen3.6*.gguf" ! -iname "mmproj*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            echo -e "
Found existing Qwen3.6 models in ./models/:"
            select opt in "${FOUND_MODELS[@]}" "Descargar Qwen3.6-35B-A3B (IQ2_XXS)" "Salir"; do
                if [[ -n "${opt}" && -f "${opt}" ]]; then
                    MODEL_PATH="${opt}"
                    break
                elif [[ "${opt}" == "Descargar Qwen3.6-35B-A3B (IQ2_XXS)" ]]; then
                    ./scripts/download-qwen3.6-35b.sh 1
                    exit 0
                else
                    exit 1
                fi
            done
        else
            echo -e "${YELLOW}[WARN] No Qwen3.6 GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-qwen3.6-35b.sh${NC} to download Qwen3.6-35B-A3B-UD-IQ2_XXS."
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
    CTX_PER_SLOT=32768  # 32k context per slot for 8 slots
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

MTP_STATUS="Disabled"
MTP_ARGS=()
if [[ "${ENABLE_MTP}" -eq 1 ]]; then
    MTP_STATUS="Active (Built-in MTP Head, draft_n_max=${DRAFT_N_MAX})"
    MTP_ARGS+=("--spec-type" "draft-mtp" "--spec-draft-n-max" "${DRAFT_N_MAX}")
fi

CTX_SHIFT_ARGS=()
if [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite generation / continuous agent support)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

# 4. Template & Sampling Configuration (Official Qwen 3.6 Recommendations)
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-0.6}" # Official Qwen 3.6 coding default
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (<think> enabled, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    REASONING_ARGS=("--reasoning-format" "deepseek")

    if [[ "${TEMPLATE_CHOICE}" == "froggeric" && -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
        TEMPLATE_DESC="Froggeric Qwen-Fixed v22.5"
    elif [[ "${TEMPLATE_CHOICE}" == "unsloth" && -f "${SCRIPT_DIR}/models/templates/Qwen3.6-Unsloth.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen3.6-Unsloth.jinja")
        TEMPLATE_DESC="Unsloth Qwen3.6 Official"
    else
        TEMPLATE_DESC="Embedded GGUF Template (--jinja)"
    fi
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}" # Official Qwen non-thinking default
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct response mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    REASONING_ARGS=("--reasoning-format" "none")

    if [[ "${TEMPLATE_CHOICE}" == "froggeric" && -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
        TEMPLATE_DESC="Froggeric Qwen-Fixed-no-thinking v22.5"
    elif [[ "${TEMPLATE_CHOICE}" == "unsloth" && -f "${SCRIPT_DIR}/models/templates/Qwen3.6-Unsloth-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen3.6-Unsloth-no-thinking.jinja")
        TEMPLATE_DESC="Unsloth Qwen3.6-no-thinking"
    else
        TEMPLATE_DESC="Embedded GGUF Template (--jinja)"
    fi
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}${THREADS} threads (-t ${THREADS})${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}100% on AMD Radeon RX 9060 XT (All 40 layers offloaded)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${MTP_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}${TEMPLATE_DESC} (--jinja enabled)${NC}"
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"
echo -e "
${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------
"

# Enable prompt and token stream exposure in /slots for monitor drill-down
export LLAMA_SERVER_SLOTS_DEBUG=1

exec "${SERVER_BIN}"     -m "${MODEL_PATH}"     --alias "${ALIAS}"     --host "${HOST}"     --port "${PORT}"     -c "${TOTAL_CTX}"     -np "${SLOTS}"     -b 2048     -ub 512     -cb     -ctk "${KV_QUANT}"     -ctv "${KV_QUANT}"     -ngl 99     -fit off     -fa auto     -t "${THREADS}"     --temp "${TEMPERATURE}"     --top-p "${TOP_P}"     --top-k "${TOP_K}"     --presence-penalty "${PRESENCE_PENALTY}"     "${CTX_SHIFT_ARGS[@]}"     "${JINJA_ARGS[@]}"     "${REASONING_ARGS[@]}"     "${MTP_ARGS[@]}"
