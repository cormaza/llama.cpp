#!/usr/bin/env bash
# ==============================================================================
# start-qwen3.8-9b-heretic.sh - Launcher for Qwen3.8-9B-Distill-Uncensored-Heretic
# Model: https://huggingface.co/ge525/Qwen3.8-9B-Distill-uncensored-heretic-Q4_K_M-GGUF
# Base: petruhonk/Qwen3.8-9B-Distill-uncensored-heretic (from empero-ai/Qwen3.8-9B)
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200)
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/qwen3.8-9b-distill-uncensored-heretic-q4_k_m.gguf"
ALT_MODEL_Q8="${SCRIPT_DIR}/models/Qwen3.8-9B-Distill-Heretic-Uncensored-Q8_0.gguf"

MODEL_PATH=""
ENABLE_THINKING=1
TEMPLATE_CHOICE="qwen"
ENABLE_CTX_SHIFT=1
ENABLE_KV_UNIFIED=0
ENABLE_MTP=0
DRAFT_N_MAX=3

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
ALIAS="qwen3.8-9b-distill-uncensored-heretic,qwen3.8-9b-heretic,qwen3.8-heretic,qwen3.8-9b,qwen3.8,qwen"
THREADS=8

# Helper function to parse human-readable token notation (e.g., 32k, 64k, 128k, 256k, 262k)
parse_tokens() {
    local val="${1,,}"
    val="${val//[[:space:]]/}"
    case "${val}" in
        0)
            echo 262144
            return
            ;;
        262k|262144)
            echo 262144
            return
            ;;
        256k|256000)
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
        echo "262k"
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

Launcher for Qwen3.8-9B-Distill-Uncensored-Heretic
Optimized for AMD Radeon RX 9060 XT (16GB VRAM, ROCm/HIP gfx1200)

Features:
  - Distilled reasoning from Qwen3.8 2.4T A95B traces (<think> blocks)
  - Uncensored via Heretic 3-stage sweep (minimal KL divergence 0.0306)
  - 100% GPU offload on RX 9060 XT with Flash Attention & Q4_0 KV Cache
  - Native 262K context window support
  - Native XML tool calling format (<tool_call>\n<function=...>)
  - Continuous context shifting for multi-turn agent workflows

Options:
  -m, --model PATH              Path to GGUF model (default: auto-detected in models/)
  -a, --alias NAMES             Comma-separated model aliases for API clients
  -c, --context, --total-ctx N  Total context pool across all slots (default: 128k; supports 64k, 128k, 262k)
  --ctx-slot N                  Context per slot (e.g. 32k, 64k; total = slots * ctx_slot)
  --slots, -np N                Number of parallel agent slots (default: 1; use 2 or 4 for multi-agent)
  -kvu, --kv-unified            Enable dynamic unified KV cache pool shared across all slots
  --thinking                    Enable reasoning mode (default: active, temp 0.6, top_p 0.95, top_k 20)
  --no-thinking                 Disable reasoning mode (direct agent mode, temp 0.7, top_p 0.80, presence 1.5)
  --template NAME|PATH          Chat template: 'qwen' (default Froggeric Fixed), 'native', or custom .jinja
  --mtp                         Enable speculative decoding with built-in MTP head
  --no-mtp                      Disable speculative decoding (default)
  --temp N                      Sampling temperature override
  --top-p N                     Top-p sampling override
  --top-k N                     Top-k sampling override
  --presence-penalty N          Presence penalty override
  --kv-quant QUANT              KV cache quantization (default: q4_0; options: q8_0, f16)
  --ngl N                       Number of GPU offloaded layers (default: 99 for full offload)
  -t, --threads N               Number of CPU threads (default: 8)
  --context-shift               Enable continuous context shifting (default: enabled)
  --no-context-shift            Disable continuous context shifting
  -p, --port PORT               HTTP server port (default: 8080)
  --host HOST                   Host address to bind (default: 0.0.0.0)
  -h, --help                    Show this help message

Examples:
  ./start-qwen3.8-9b-heretic.sh                           # 1 slot x 128k context with reasoning (temp 0.6)
  ./start-qwen3.8-9b-heretic.sh -c 262k                   # Full 262k native context window
  ./start-qwen3.8-9b-heretic.sh --no-thinking             # Fast direct agent mode (temp 0.7)
  ./start-qwen3.8-9b-heretic.sh --slots 2 --ctx-slot 64k  # Dual-agent serving (64k context per slot)
  ./start-qwen3.8-9b-heretic.sh -kvu -c 128k --slots 4    # Dynamic shared unified KV pool
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
        --mtp)
            ENABLE_MTP=1
            shift
            ;;
        --no-mtp)
            ENABLE_MTP=0
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
echo -e "${BOLD}${CYAN}  Qwen3.8-9B-Distill Uncensored Heretic Server (ROCm / HIP gfx1200)${NC}"
echo -e "${BOLD}${CYAN}  Distilled Reasoning from Qwen3.8 2.4T A95B | 262k Native Context ${NC}"
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
    elif [[ -f "${ALT_MODEL_Q8}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q8}"
    else
        DETECTED_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 \( -iname "*qwen3.8*heretic*.gguf" -o -iname "*distill*heretic*.gguf" \) 2>/dev/null || true))
        if [[ ${#DETECTED_MODELS[@]} -gt 0 && -f "${DETECTED_MODELS[0]}" ]]; then
            MODEL_PATH="${DETECTED_MODELS[0]}"
        else
            echo -e "${YELLOW}[WARN] No Qwen3.8-9B-Distill Heretic model found in ./models/${NC}"
            echo -e "You can download it with:"
            echo -e "  ${CYAN}./scripts/download-qwen3.8-9b-heretic.sh${NC}\n"
            read -rp "Would you like to download it now? [y/N]: " RUN_DL
            if [[ "${RUN_DL}" =~ ^[Yy]$ ]]; then
                "${SCRIPT_DIR}/scripts/download-qwen3.8-9b-heretic.sh"
                MODEL_PATH="${DEFAULT_MODEL}"
            else
                echo -e "${RED}[ERROR] Model file required to proceed.${NC}"
                exit 1
            fi
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
    # Default: 131072 (128k) tokens distributed across slots
    TOTAL_CTX=131072
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# Ensure minimum viable context per slot
if [[ "${CTX_PER_SLOT}" -lt 2048 ]]; then
    CTX_PER_SLOT=2048
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
fi

MAX_NATIVE_CTX=262144
if [[ "${TOTAL_CTX}" -gt "${MAX_NATIVE_CTX}" ]]; then
    echo -e "${YELLOW}[WARN] Total context (${TOTAL_CTX} tokens) exceeds 262k native context limit.${NC}"
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

# 4. Context Shift Configuration
CTX_SHIFT_ARGS=()
if [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite continuous operation)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

# 5. Speculative Decoding Configuration
MTP_ARGS=()
MTP_STATUS="Disabled"
if [[ "${ENABLE_MTP}" -eq 1 ]]; then
    MTP_ARGS+=("--spec-type" "draft-mtp" "--spec-draft-n-max" "${DRAFT_N_MAX}")
    MTP_STATUS="Active (Built-in MTP Head, draft_n_max=${DRAFT_N_MAX})"
fi

# 6. Sampling & Template Configuration
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-0.6}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Qwen3.8 distilled <think> traces, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    case "${TEMPLATE_CHOICE}" in
        qwen)
            if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
            fi
            TEMPLATE_STATUS="Froggeric Qwen-Fixed (--jinja)"
            ;;
        native)
            TEMPLATE_STATUS="Embedded GGUF template"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE})"
            else
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
                TEMPLATE_STATUS="Froggeric Qwen-Fixed"
            fi
            ;;
    esac
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct agent mode, temp ${TEMPERATURE}, top_p ${TOP_P}, presence ${PRESENCE_PENALTY})"
    case "${TEMPLATE_CHOICE}" in
        qwen)
            if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
            fi
            TEMPLATE_STATUS="Froggeric Qwen-Fixed-no-thinking (--jinja)"
            ;;
        native)
            TEMPLATE_STATUS="Embedded GGUF template"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE})"
            else
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
                TEMPLATE_STATUS="Froggeric Qwen-Fixed-no-thinking"
            fi
            ;;
    esac
    REASONING_ARGS=("--reasoning-format" "none")
fi

NGL="${CUSTOM_NGL:-99}"

echo -e "\n${BOLD}Server Configuration:${NC}"
echo -e "  ${BOLD}Model:${NC}             ${CYAN}${MODEL_PATH}${NC}"
echo -e "  ${BOLD}API Model Alias:${NC}   ${GREEN}${ALIAS}${NC}"
echo -e "  ${BOLD}Parallel Slots:${NC}    ${GREEN}${SLOTS} slot(s)${NC}"
echo -e "  ${BOLD}Context per Slot:${NC}  ${GREEN}${CTX_PER_SLOT} tokens ($(format_tokens_k "${CTX_PER_SLOT}"))${NC}"
echo -e "  ${BOLD}Total Context Pool:${NC}${GREEN}${TOTAL_CTX} tokens ($(format_tokens_k "${TOTAL_CTX}"))${NC}"
echo -e "  ${BOLD}Context Shift:${NC}     ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "  ${BOLD}KV Cache Alloc:${NC}    ${GREEN}${KV_UNIFIED_STATUS}${NC}"
echo -e "  ${BOLD}KV Cache Type:${NC}     ${GREEN}${KV_QUANT}${NC}"
echo -e "  ${BOLD}GPU Offload:${NC}       ${GREEN}100% on AMD Radeon RX 9060 XT (ngl ${NGL}, FA auto)${NC}"
echo -e "  ${BOLD}Batching:${NC}          ${GREEN}Continuous (-cb) | Chunked Prefill (-ub ${UBATCH_SIZE}, -b ${BATCH_SIZE})${NC}"
echo -e "  ${BOLD}Thinking Mode:${NC}     ${GREEN}${THINKING_STATUS}${NC}"
echo -e "  ${BOLD}Chat Template:${NC}     ${GREEN}${TEMPLATE_STATUS}${NC}"
echo -e "  ${BOLD}Speculative Dec:${NC}   ${GREEN}${MTP_STATUS}${NC}"
echo -e "  ${BOLD}Sampling Params:${NC}   ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"

echo -e "\n${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------\n"

export LLAMA_SERVER_SLOTS_DEBUG=1
export LLAMA_ARG_ENDPOINT_METRICS=1

exec "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --metrics \
    -c "${TOTAL_CTX}" \
    -np "${SLOTS}" \
    -b "${BATCH_SIZE}" \
    -ub "${UBATCH_SIZE}" \
    -cb \
    "${KV_UNIFIED_ARGS[@]}" \
    "${CTX_SHIFT_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl "${NGL}" \
    -fit off \
    -fa auto \
    -t "${THREADS}" \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}"
