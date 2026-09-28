#!/usr/bin/env bash
# ==============================================================================
# start-qwen3.5-c3sm-9b.sh - Launcher for Qwen3.5-9B-C3SM-SDM-Agentic-Coder
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200)
# Features: Qwen3.5 9B Hybrid Linear Attention + C3SM-SDM Consensus Agentic Merging
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q4_K_M.gguf"
ALT_MODEL_Q5="${SCRIPT_DIR}/models/Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q5_K_M.gguf"
ALT_MODEL_Q6="${SCRIPT_DIR}/models/Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q6_K.gguf"
ALT_MODEL_Q8="${SCRIPT_DIR}/models/Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q8_0.gguf"
ALT_MODEL_Q3="${SCRIPT_DIR}/models/Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q3_K_L.gguf"

MODEL_PATH=""
ENABLE_THINKING=1
TEMPLATE_CHOICE="c3sm"
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
ALIAS="qwen3.5-c3sm-9b,qwen3.5-agentic-coder,c3sm-9b,qwen-agent,qwen"
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
        131k|131072)
            echo 131072
            return
            ;;
        65k|65536)
            echo 65536
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

Starts llama-server for Qwen3.5-9B-C3SM-SDM-Agentic-Coder (Spectral Consensus Merged 9B Agent)
optimized for AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200).

Features:
  - Curvature-Calibrated Spectral Consensus Merging with Sinusoidal Depth Modulation
  - Donors: MiMo, Ornith, Qwopus, OxCoder, Qwen3.8-Distill
  - Hybrid linear attention (Gated DeltaNet) with 262K native context window
  - Dedicated agentic template with multi-turn failure circuit breaker & JSON tools
  - Infinite context shifting enabled by default (pure text architecture)
  - 100% GPU Offload with Flash Attention & Q4_0 KV cache

Options:
  -m, --model PATH              Path to C3SM-Agentic-Coder GGUF (default: auto-detect Q4/Q5/Q6/Q8)
  -a, --alias NAMES             Model alias for API clients (default: ${ALIAS})
  -c, --context, --ctx-size, --total-ctx N  Total context pool across all slots (supports 64k, 128k, 256k, 262k; default: 262k)
  --ctx-slot N                  Explicit context per slot override (e.g. 32k, 64k, 128k)
  --slots, -np, --parallel N    Number of parallel agent slots (default: 1; use 2 or 4 for multi-agent)
  -kvu, --kv-unified            Enable dynamic unified KV cache pool shared across all slots
  --thinking                    Enable reasoning mode (default; temp 0.6, top_p 0.95, top_k 20)
  --no-thinking                 Disable reasoning mode (direct agent mode; temp 0.2, top_p 0.95)
  --template TYPE               Chat template: c3sm (default) | qwen (Froggeric Fixed) | native
  --temp N                      Sampling temperature override
  --top-p N                     Top-p sampling override
  --top-k N                     Top-k sampling override
  --presence-penalty N          Presence penalty override
  -t, --threads N               Number of CPU threads (default: 8)
  --context-shift               Enable context shifting (default: enabled)
  --no-context-shift            Disable context shifting
  -p, --port PORT               HTTP server port (default: 8080)
  --host HOST                   Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE               KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                       GPU layers offloaded (default: 99, 100% GPU)
  -h, --help                    Show this help message

Examples:
  ./start-qwen3.5-c3sm-9b.sh                            # Full stack with reasoning (262k native context)
  ./start-qwen3.5-c3sm-9b.sh -c 128k                    # Conservative 128k context pool
  ./start-qwen3.5-c3sm-9b.sh --no-thinking              # Fast direct agent mode (temp 0.2)
  ./start-qwen3.5-c3sm-9b.sh --slots 2 --ctx-slot 64k   # Dual-agent serving (64k context per slot)
  ./start-qwen3.5-c3sm-9b.sh -kvu -c 256k --slots 4     # Dynamic shared unified KV pool
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
        --kv-quant)
            KV_QUANT="$2"
            shift 2
            ;;
        --kv-quant=*)
            KV_QUANT="${1#*=}"
            shift
            ;;
        --ngl)
            CUSTOM_NGL="$2"
            shift 2
            ;;
        --ngl=*)
            CUSTOM_NGL="${1#*=}"
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
echo -e "${BOLD}${CYAN}  Qwen3.5-9B-C3SM-SDM Server (ROCm / HIP gfx1200)     ${NC}"
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
    elif [[ -f "${ALT_MODEL_Q6}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q6}"
    elif [[ -f "${ALT_MODEL_Q8}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q8}"
    elif [[ -f "${ALT_MODEL_Q3}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q3}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 \( -iname "*qwen3.5*c3sm*.gguf" -o -iname "*c3sm*agentic*.gguf" -o -iname "*qwen*c3sm*.gguf" \) 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            MODEL_PATH="${FOUND_MODELS[0]}"
        else
            echo -e "${YELLOW}[WARN] No Qwen3.5-9B-C3SM-SDM-Agentic-Coder GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-qwen3.5-c3sm-9b.sh${NC} to download the model."
            exit 1
        fi
    fi
fi

# 3. Context Calculation & VRAM Bounds
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
    # Default: 262144 native context pool distributed across slots
    TOTAL_CTX=262144
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# Ensure minimum viable context per slot
if [[ "${CTX_PER_SLOT}" -lt 1024 ]]; then
    CTX_PER_SLOT=1024
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
fi

# Qwen3.5 9B Hybrid Linear Attention context window is 262144 tokens
MAX_SAFE_CTX=262144
if [[ "${TOTAL_CTX}" -gt "${MAX_SAFE_CTX}" ]]; then
    echo -e "${YELLOW}[WARN] Total context (${TOTAL_CTX} tokens) exceeds the 262k safe limit for 16GB VRAM.${NC}"
    echo -e "${YELLOW}[WARN] Ensure you have sufficient RAM or adjust offload layers if running near limits.${NC}"
fi

# Dynamic batch sizes: clamp batch size to total context if context is small
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

# 4. Context Shift (Pure text model allows infinite continuous generation)
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
    TEMPERATURE="${CUSTOM_TEMP:-0.6}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Thinking mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    case "${TEMPLATE_CHOICE}" in
        c3sm)
            if [[ -f "${SCRIPT_DIR}/models/templates/Qwen3.5-C3SM-Agentic-Coder.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen3.5-C3SM-Agentic-Coder.jinja")
            fi
            TEMPLATE_STATUS="Official Olivia Rossi C3SM-SDM (with <thought>/<think> & circuit breaker)"
            ;;
        qwen|froggeric)
            if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
            fi
            TEMPLATE_STATUS="Froggeric Qwen-Fixed"
            ;;
        native)
            TEMPLATE_STATUS="Embedded GGUF template"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE})"
            else
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen3.5-C3SM-Agentic-Coder.jinja")
                TEMPLATE_STATUS="Official C3SM-SDM"
            fi
            ;;
    esac
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.2}" # Official strict coding agent recommended default
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Disabled (Direct agent mode, temp ${TEMPERATURE}, top_p ${TOP_P})"
    case "${TEMPLATE_CHOICE}" in
        c3sm)
            if [[ -f "${SCRIPT_DIR}/models/templates/Qwen3.5-C3SM-Agentic-Coder-no-thinking.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen3.5-C3SM-Agentic-Coder-no-thinking.jinja")
            else
                JINJA_ARGS+=("--chat-template-kwargs" '{"enable_thinking": false}')
            fi
            TEMPLATE_STATUS="C3SM-SDM (no-thinking direct output)"
            ;;
        qwen|froggeric)
            if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
            fi
            TEMPLATE_STATUS="Froggeric Qwen-Fixed-no-thinking"
            ;;
        native)
            JINJA_ARGS+=("--chat-template-kwargs" '{"enable_thinking": false}')
            TEMPLATE_STATUS="Embedded GGUF template (enable_thinking=false)"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE})"
            else
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen3.5-C3SM-Agentic-Coder-no-thinking.jinja")
                TEMPLATE_STATUS="C3SM-SDM (no-thinking)"
            fi
            ;;
    esac
    REASONING_ARGS=("--reasoning-format" "none")
fi

GPU_LAYERS="${CUSTOM_NGL:-99}"

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}Architecture:${NC}        ${GREEN}Qwen3.5 9B Hybrid Linear Attention (C3SM-SDM Consensus Merged)${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slot(s)${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(format_tokens_k "${CTX_PER_SLOT}") tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(format_tokens_k "${TOTAL_CTX}") tokens)${NC}"
echo -e "${BOLD}KV Cache Pool:${NC}       ${GREEN}${KV_UNIFIED_STATUS}${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}All layers offloaded to GPU (-ngl ${GPU_LAYERS} -fa on)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub ${UBATCH_SIZE}, -b ${BATCH_SIZE})${NC}"
echo -e "${BOLD}CPU Affinity:${NC}        ${GREEN}Pinned to 8 P-cores (--cpu-range 0-7, -t ${THREADS})${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}${TEMPLATE_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"
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
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}"
