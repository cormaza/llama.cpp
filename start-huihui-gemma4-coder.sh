#!/usr/bin/env bash
# ==============================================================================
# start-huihui-gemma4-coder.sh - Launcher for Huihui-Gemma4-12B-Coder-Abliterated
# Model: https://huggingface.co/mradermacher/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated-i1-GGUF
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200)
# Features: Google Gemma 4 12B + Composer 2.5 & Fable 5 CoT + Abliterated + imatrix
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
SERVER_BIN="${SERVER_BIN:-${BIN_DIR}/llama-server}"

DEFAULT_MODEL="${SCRIPT_DIR}/models/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q4_K_M.gguf"
ALT_MODEL_Q5="${SCRIPT_DIR}/models/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q5_K_M.gguf"
ALT_MODEL_Q6="${SCRIPT_DIR}/models/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q6_K.gguf"
ALT_MODEL_IQ4="${SCRIPT_DIR}/models/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-IQ4_XS.gguf"
ALT_MODEL_Q3="${SCRIPT_DIR}/models/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q3_K_M.gguf"
ALT_MODEL_IQ3="${SCRIPT_DIR}/models/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-IQ3_M.gguf"

MODEL_PATH=""
ENABLE_THINKING=1
TEMPLATE_CHOICE="gemma4"
ENABLE_CTX_SHIFT=1
ENABLE_KV_UNIFIED=0

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
CUSTOM_NGL=""
ALIAS="huihui-gemma4-coder,gemma4-coder-abliterated,gemma4-coder,gemma4-12b-coder,gemma"
THREADS=8

# Helper function to parse human-readable token notation (e.g., 32k, 64k, 128k, 256k)
parse_tokens() {
    local val="${1,,}"
    val="${val//[[:space:]]/}"
    case "${val}" in
        0)
            echo 262144
            return
            ;;
        256k|262144)
            echo 262144
            return
            ;;
        128k|131072)
            echo 131072
            return
            ;;
        64k|65536)
            echo 65536
            return
            ;;
        32k|32768)
            echo 32768
            return
            ;;
        16k|16384)
            echo 16384
            return
            ;;
    esac
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

format_tokens_k() {
    local n="$1"
    if [[ "$n" -eq 262144 ]]; then
        echo "256k"
    elif [[ "$n" -eq 131072 ]]; then
        echo "128k"
    elif [[ "$n" -eq 65536 ]]; then
        echo "64k"
    elif [[ "$n" -eq 32768 ]]; then
        echo "32k"
    elif [[ "$n" -eq 16384 ]]; then
        echo "16k"
    elif [[ $(( n % 1024 )) -eq 0 ]]; then
        echo "$(( n / 1024 ))k"
    else
        echo "${n}"
    fi
}

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Launcher for Huihui-Gemma4-12B-Coder-Abliterated (imatrix i1 quants)
Optimized for AMD Radeon RX 9060 XT (16GB VRAM, ROCm/HIP gfx1200)

Options:
  -m, --model PATH              Path to GGUF model (default: auto-detected in models/)
  -a, --alias NAMES             Comma-separated model aliases for API clients
  -c, --context, --total-ctx N  Total context pool across all slots (default: 64k; supports 32k, 64k, 128k, 256k)
  --ctx-slot N                  Context per slot (e.g. 32k, 64k; total = slots * ctx_slot)
  --slots, -np N                Number of parallel agent slots (default: 1; use 2 or 4 for multi-agent)
  -kvu, --kv-unified            Enable dynamic unified KV cache pool shared across all slots
  --thinking                    Enable native Gemma 4 thinking channel (default: active, temp 1.0, top-k 64)
  --no-thinking                 Disable thinking mode (fast direct code generation, temp 0.2)
  --template NAME|PATH          Chat template: 'gemma4' (default), 'native', or custom .jinja file
  --temp N                      Sampling temperature (default: 1.0 with thinking, 0.2 without)
  --top-p N                     Top-p sampling (default: 0.95)
  --top-k N                     Top-k sampling (default: 64 with thinking, 20 without)
  --presence-penalty N          Presence penalty (default: 0.0)
  --kv-quant QUANT              KV cache quantization (default: q4_0; options: q8_0, f16)
  --ngl N                       Number of GPU offloaded layers (default: 99 for full offload)
  -t, --threads N               Number of CPU threads (default: 8)
  --no-context-shift            Disable continuous context shifting
  -p, --port PORT               HTTP server port (default: 8080)
  --host HOST                   Host address to bind (default: 0.0.0.0)
  -h, --help                    Show this help message

Examples:
  ./start-huihui-gemma4-coder.sh                           # 1 slot x 64k context with thinking (temp 1.0)
  ./start-huihui-gemma4-coder.sh -c 128k                   # 1 slot x 128k deep context
  ./start-huihui-gemma4-coder.sh -c 256k                   # 1 slot x 256k maximum native context
  ./start-huihui-gemma4-coder.sh --no-thinking             # Fast direct code execution mode (temp 0.2)
  ./start-huihui-gemma4-coder.sh --slots 2 --ctx-slot 32k  # Dual-agent serving (32k context per slot)
  ./start-huihui-gemma4-coder.sh -kvu -c 128k --slots 4    # Dynamic shared unified KV pool
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -m=*|--model=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        -a|--alias)
            ALIAS="$2"
            shift 2
            ;;
        -a=*|--alias=*)
            ALIAS="${1#*=}"
            shift
            ;;
        -c|--context|--ctx|--ctx-size|--context-size|--total-ctx|--total-context)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        -c=*|--context=*|--ctx=*|--ctx-size=*|--context-size=*|--total-ctx=*|--total-context=*)
            CUSTOM_CTX="${1#*=}"
            shift
            ;;
        --ctx-slot|--slot-ctx|--context-slot|--ctx-per-slot|--slot-context)
            CUSTOM_CTX_SLOT="$2"
            shift 2
            ;;
        --ctx-slot=*|--slot-ctx=*|--context-slot=*|--ctx-per-slot=*|--slot-context=*)
            CUSTOM_CTX_SLOT="${1#*=}"
            shift
            ;;
        -kvu|--kv-unified)
            ENABLE_KV_UNIFIED=1
            shift
            ;;
        --slots|-np|--parallel)
            SLOTS="$2"
            shift 2
            ;;
        --slots=*|-np=*|--parallel=*)
            SLOTS="${1#*=}"
            shift
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
        --template=*)
            TEMPLATE_CHOICE="${1#*=}"
            shift
            ;;
        --temp|--temperature)
            CUSTOM_TEMP="$2"
            shift 2
            ;;
        --temp=*|--temperature=*)
            CUSTOM_TEMP="${1#*=}"
            shift
            ;;
        --top-p)
            CUSTOM_TOP_P="$2"
            shift 2
            ;;
        --top-p=*)
            CUSTOM_TOP_P="${1#*=}"
            shift
            ;;
        --top-k)
            CUSTOM_TOP_K="$2"
            shift 2
            ;;
        --top-k=*)
            CUSTOM_TOP_K="${1#*=}"
            shift
            ;;
        --presence-penalty)
            CUSTOM_PRESENCE="$2"
            shift 2
            ;;
        --presence-penalty=*)
            CUSTOM_PRESENCE="${1#*=}"
            shift
            ;;
        --kv-quant|--ctk|--ctv)
            KV_QUANT="$2"
            shift 2
            ;;
        --kv-quant=*|--ctk=*|--ctv=*)
            KV_QUANT="${1#*=}"
            shift
            ;;
        --ngl|--n-gpu-layers)
            CUSTOM_NGL="$2"
            shift 2
            ;;
        --ngl=*|--n-gpu-layers=*)
            CUSTOM_NGL="${1#*=}"
            shift
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -t=*|--threads=*)
            THREADS="${1#*=}"
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
        -p=*|--port=*)
            PORT="${1#*=}"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown argument: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BOLD}${CYAN}===================================================================${NC}"
echo -e "${BOLD}${CYAN}  Huihui-Gemma4-12B-Coder Abliterated Server (ROCm / HIP gfx1200)   ${NC}"
echo -e "${BOLD}${CYAN}  Google Gemma 4 12B + Composer 2.5 & Fable 5 CoT + imatrix        ${NC}"
echo -e "${BOLD}${CYAN}===================================================================${NC}"

# 1. Binary check
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${SERVER_BIN}${NC}"
    echo -e "Please build first: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

# 2. Model Auto-Detection
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -f "${DEFAULT_MODEL}" ]]; then
        MODEL_PATH="${DEFAULT_MODEL}"
    elif [[ -f "${ALT_MODEL_Q5}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q5}"
    elif [[ -f "${ALT_MODEL_Q6}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q6}"
    elif [[ -f "${ALT_MODEL_IQ4}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ4}"
    elif [[ -f "${ALT_MODEL_Q3}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q3}"
    elif [[ -f "${ALT_MODEL_IQ3}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ3}"
    else
        echo -e "${YELLOW}[WARN] No Huihui-Gemma4-12B-Coder Abliterated model found in ./models/${NC}"
        echo -e "You can download it with:"
        echo -e "  ${CYAN}./scripts/download-huihui-gemma4-coder.sh${NC}\n"
        read -rp "Would you like to download it now? [y/N]: " RUN_DL
        if [[ "${RUN_DL}" =~ ^[Yy]$ ]]; then
            "${SCRIPT_DIR}/scripts/download-huihui-gemma4-coder.sh"
            MODEL_PATH="${DEFAULT_MODEL}"
        else
            echo -e "${RED}[ERROR] Model file required to proceed.${NC}"
            exit 1
        fi
    fi
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo -e "${RED}[ERROR] Specified model file does not exist: ${MODEL_PATH}${NC}"
    exit 1
fi

# 3. Context Calculation
if [[ -n "${CUSTOM_CTX}" ]]; then
    CUSTOM_CTX="$(parse_tokens "${CUSTOM_CTX}")"
fi
if [[ -n "${CUSTOM_CTX_SLOT}" ]]; then
    CUSTOM_CTX_SLOT="$(parse_tokens "${CUSTOM_CTX_SLOT}")"
fi

if [[ -n "${CUSTOM_CTX_SLOT}" && -n "${CUSTOM_CTX}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX_SLOT}"
    TOTAL_CTX="${CUSTOM_CTX}"
elif [[ -n "${CUSTOM_CTX_SLOT}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX_SLOT}"
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
elif [[ -n "${CUSTOM_CTX}" ]]; then
    TOTAL_CTX="${CUSTOM_CTX}"
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
else
    # Default: 65536 (64k) tokens distributed across slots
    TOTAL_CTX=65536
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# Ensure minimum viable context per slot
if [[ "${CTX_PER_SLOT}" -lt 2048 ]]; then
    CTX_PER_SLOT=2048
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
fi

MAX_NATIVE_CTX=262144
if [[ "${TOTAL_CTX}" -gt "${MAX_NATIVE_CTX}" ]]; then
    echo -e "${YELLOW}[WARN] Total context (${TOTAL_CTX} tokens) exceeds 256k native context limit.${NC}"
fi

# Dynamic batch sizes
BATCH_SIZE=2048
UBATCH_SIZE=1024
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

# 4. Context Shift
CTX_SHIFT_ARGS=()
if [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite continuous operation)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

# 5. Sampling & Template Configuration
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-1.0}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-64}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Gemma 4 CoT thinking mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    case "${TEMPLATE_CHOICE}" in
        gemma4|gemma)
            if [[ -f "${SCRIPT_DIR}/models/templates/google-gemma-4-12B-it.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/google-gemma-4-12B-it.jinja")
            fi
            TEMPLATE_STATUS="Google Gemma 4 12B (with native <|channel>thought & tool calling)"
            ;;
        native)
            TEMPLATE_STATUS="Embedded GGUF template"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE})"
            else
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/google-gemma-4-12B-it.jinja")
                TEMPLATE_STATUS="Google Gemma 4 12B"
            fi
            ;;
    esac
else
    TEMPERATURE="${CUSTOM_TEMP:-0.2}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Disabled (Direct fast execution, temp ${TEMPERATURE}, top_p ${TOP_P})"
    case "${TEMPLATE_CHOICE}" in
        gemma4|gemma)
            if [[ -f "${SCRIPT_DIR}/models/templates/google-gemma-4-12B-it.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/google-gemma-4-12B-it.jinja")
            fi
            TEMPLATE_STATUS="Google Gemma 4 12B (Direct mode)"
            ;;
        native)
            TEMPLATE_STATUS="Embedded GGUF template"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE})"
            fi
            ;;
    esac
fi

# 6. GPU Offload Configuration
GPU_LAYERS="${CUSTOM_NGL:-99}"

# 7. Print System Status
echo -e "\n${BOLD}Model:${NC}               ${GREEN}$(basename "${MODEL_PATH}")${NC}"
echo -e "${BOLD}Architecture:${NC}        ${GREEN}Google Gemma 4 12B (Abliterated, imatrix i1)${NC}"
echo -e "${BOLD}Template Engine:${NC}     ${GREEN}${TEMPLATE_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}$(format_tokens_k "${TOTAL_CTX}") tokens (-c ${TOTAL_CTX})${NC}"
echo -e "${BOLD}Slots / Agents:${NC}      ${GREEN}${SLOTS} parallel slot(s) (-np ${SLOTS})${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}$(format_tokens_k "${CTX_PER_SLOT}") tokens per slot${NC}"
echo -e "${BOLD}KV Cache Allocation:${NC} ${GREEN}${KV_UNIFIED_STATUS}${NC}"
echo -e "${BOLD}KV Precision:${NC}        ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}FlashAttention:${NC}      ${GREEN}Active (-fa on)${NC}"
echo -e "${BOLD}Continuous Batch:${NC}   ${GREEN}Active (-cb -b ${BATCH_SIZE} -ub ${UBATCH_SIZE})${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}${THREADS} threads (-t ${THREADS} --cpu-range 0-7)${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}${GPU_LAYERS} layers on AMD Radeon RX 9060 XT (gfx1200)${NC}"
echo -e "${BOLD}Model Aliases:${NC}       ${GREEN}${ALIAS}${NC}"

echo -e "\n${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------\n"

export LLAMA_SERVER_SLOTS_DEBUG=1

exec "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${TOTAL_CTX}" \
    -np "${SLOTS}" \
    -b "${BATCH_SIZE}" \
    -ub "${UBATCH_SIZE}" \
    -cb \
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl "${GPU_LAYERS}" \
    -fa on \
    -t "${THREADS}" \
    --cpu-range 0-7 \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${KV_UNIFIED_ARGS[@]}" \
    "${CTX_SHIFT_ARGS[@]}" \
    "${JINJA_ARGS[@]}"
