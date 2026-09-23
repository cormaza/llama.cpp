#!/usr/bin/env bash
# ==============================================================================
# start-bonsai-2-crack.sh - Launcher for Bonsai 2 27B 1-bit CRACK (Uncensored)
# Model: https://huggingface.co/dealignai/Bonsai-2-27B-1bit-CRACK-GGUF
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200)
# Features: Uncensored PTQ1_0 Ternary 1.75 bpw + 262K Hybrid Attention + Vision
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Bonsai-2-27B-PTQ1_0-CRACK.gguf"
DEFAULT_MMPROJ="${SCRIPT_DIR}/models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
ALT_MMPROJ_BF16="${SCRIPT_DIR}/models/Ternary-Bonsai-2-27B-mmproj-BF16.gguf"

MODEL_PATH=""
MMPROJ_PATH=""
ENABLE_MMPROJ=1
ENABLE_THINKING=1
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
ALIAS="bonsai-2-crack,bonsai-2-27b-crack,bonsai-crack,bonsai2-crack"
THREADS=8

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

# Detect Primary LAN IP
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Bonsai 2 27B 1-bit CRACK (dealignai / Uncensored PTQ1_0)
optimized for AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200).

Features:
  - Uncensored / Dealigned model: refusal circuits removed (100% compliance on HarmBench)
  - True 1.75 bits/weight ternary weights (PTQ1_0, 5.95 GB)
  - 262K native context window with hybrid attention (~75% linear attention)
  - Multimodal Vision Projector support (mmproj Q8_0 0.63 GB)
  - 100% GPU Offload with Flash Attention & Q4_0 KV cache

Options:
  -m, --model PATH              Path to GGUF model (default: ./models/Bonsai-2-27B-PTQ1_0-CRACK.gguf)
  -a, --alias NAMES             Model alias for API clients (default: ${ALIAS})
  --mmproj PATH                 Path to multimodal vision projector (default: auto-detect)
  --no-mmproj                   Disable multimodal vision projector (enables context shifting)
  -c, --context, --total-ctx N  Total context size pool (supports 64k, 128k, 256k; up to 262144)
  --ctx-slot N                  Explicit context per slot override (e.g. 32k, 64k)
  --slots, -np N                Number of parallel slots (default: 1; use 2 or 4 for multi-agent)
  -kvu, --kv-unified            Enable dynamic unified KV cache pool shared across all slots
  --thinking                    Enable reasoning mode (default; temp 1.0, top_p 0.95, top_k 20)
  --no-thinking                 Disable reasoning mode (direct response mode; temp 0.7, top_p 0.80, presence 1.5)
  --temp N                      Sampling temperature override
  --top-p N                     Top-p sampling override
  --top-k N                     Top-k sampling override
  --presence-penalty N          Presence penalty override
  -t, --threads N               Number of CPU threads (default: 8)
  --no-context-shift            Disable context shifting
  -p, --port PORT               HTTP server port (default: 8080)
  --host HOST                   Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE               KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                       GPU layers offloaded (default: 99, 100% GPU)
  -h, --help                    Show this help message

Examples:
  ./start-bonsai-2-crack.sh                            # Full stack (CRACK + Vision, 128k context)
  ./start-bonsai-2-crack.sh -c 262144                  # Max 262k native context window
  ./start-bonsai-2-crack.sh --no-thinking              # Fast direct response mode without thinking
  ./start-bonsai-2-crack.sh --no-mmproj                # Pure text mode with infinite context shifting
  ./start-bonsai-2-crack.sh --slots 2 --ctx-slot 64k   # Dual-agent serving (64k context per slot)
  ./start-bonsai-2-crack.sh -kvu -c 256k --slots 4     # Dynamic shared unified KV pool
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
        --mmproj)
            MMPROJ_PATH="$2"
            shift 2
            ;;
        --no-mmproj)
            ENABLE_MMPROJ=0
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
        -kvu|--kv-unified)
            ENABLE_KV_UNIFIED=1
            shift
            ;;
        --slots|-np)
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
echo -e "${BOLD}${CYAN}  Bonsai 2 27B CRACK Server (ROCm / HIP Accelerated)  ${NC}"
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
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 \( -iname "*bonsai*crack*.gguf" -o -iname "*crack*.gguf" \) ! -iname "*mmproj*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            MODEL_PATH="${FOUND_MODELS[0]}"
        else
            echo -e "${YELLOW}[WARN] No Bonsai 2 CRACK GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-bonsai-2-crack.sh${NC} to download Bonsai 2 27B CRACK."
            exit 1
        fi
    fi
fi

# 3. Vision Projector Configuration
MMPROJ_ARGS=()
MMPROJ_ACTIVE=0
if [[ "${ENABLE_MMPROJ}" -eq 1 ]]; then
    if [[ -z "${MMPROJ_PATH}" ]]; then
        if [[ -f "${DEFAULT_MMPROJ}" ]]; then
            MMPROJ_PATH="${DEFAULT_MMPROJ}"
        elif [[ -f "${ALT_MMPROJ_BF16}" ]]; then
            MMPROJ_PATH="${ALT_MMPROJ_BF16}"
        else
            DETECTED_MMPROJ=($(find "${SCRIPT_DIR}/models" -maxdepth 1 \( -iname "*bonsai-2*mmproj*.gguf" -o -iname "*bonsai*2*mmproj*.gguf" \) 2>/dev/null || true))
            if [[ ${#DETECTED_MMPROJ[@]} -gt 0 && -f "${DETECTED_MMPROJ[0]}" ]]; then
                MMPROJ_PATH="${DETECTED_MMPROJ[0]}"
            fi
        fi
    fi

    if [[ -n "${MMPROJ_PATH}" && -f "${MMPROJ_PATH}" ]]; then
        MMPROJ_ARGS=("--mmproj" "${MMPROJ_PATH}" "--image-min-tokens" "1024")
        MMPROJ_STATUS="Active ($(basename "${MMPROJ_PATH}"))"
        MMPROJ_ACTIVE=1
    else
        MMPROJ_STATUS="Disabled (no projector found; run ./scripts/download-bonsai-2-crack.sh mmproj)"
    fi
else
    MMPROJ_ARGS=("--no-mmproj")
    MMPROJ_STATUS="Disabled (--no-mmproj)"
fi

# 4. Context Calculation & VRAM Bounds
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
elif [[ "${SLOTS}" -le 2 ]]; then
    CTX_PER_SLOT=131072
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
else
    TOTAL_CTX=262144
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# Ensure minimum viable context per slot
if [[ "${CTX_PER_SLOT}" -lt 1024 ]]; then
    CTX_PER_SLOT=1024
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
fi

# Bonsai 2 27B native max context is 262144. In 16GB VRAM, 262k is safe due to hybrid linear attention.
MAX_SAFE_CTX=262144
if [[ "${TOTAL_CTX}" -gt "${MAX_SAFE_CTX}" ]]; then
    echo -e "${YELLOW}[WARN] Total context (${TOTAL_CTX}) exceeds the 262k safe limit for 16GB VRAM.${NC}"
    echo -e "${YELLOW}[WARN] Capping total context to ${MAX_SAFE_CTX} ($(( MAX_SAFE_CTX / SLOTS )) per slot) to avoid Out-Of-Memory crashes.${NC}"
    TOTAL_CTX="${MAX_SAFE_CTX}"
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
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

# 5. Context Shift
CTX_SHIFT_ARGS=()
if [[ "${MMPROJ_ACTIVE}" -eq 1 ]]; then
    CTX_SHIFT_STATUS="Disabled (Multimodal active; pass --no-mmproj for infinite context shifting)"
elif [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite continuous operation)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

# 6. Sampling & Template Configuration (Official Bonsai 2 Recommendations)
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-1.0}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Thinking mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct response mode, temp ${TEMPERATURE}, top_p ${TOP_P})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "none")
fi

GPU_LAYERS="${CUSTOM_NGL:-99}"

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}Variant:${NC}             ${YELLOW}CRACK / Dealigned (Uncensored / Refusal circuits removed)${NC}"
echo -e "${BOLD}Architecture:${NC}        ${GREEN}Qwen3.8 Hybrid Attention (~75% linear / ~25% full attention)${NC}"
echo -e "${BOLD}Quantization:${NC}        ${GREEN}True 1.75 bpw Ternary (PTQ1_0, 5.95 GB)${NC}"
echo -e "${BOLD}Vision Projector:${NC}    ${GREEN}${MMPROJ_STATUS}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slot(s)${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}KV Cache Pool:${NC}       ${GREEN}${KV_UNIFIED_STATUS}${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}All layers offloaded to GPU (-ngl ${GPU_LAYERS} -fa on)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub ${UBATCH_SIZE}, -b ${BATCH_SIZE})${NC}"
echo -e "${BOLD}CPU Affinity:${NC}        ${GREEN}Pinned to 8 P-cores (--cpu-range 0-7, -t ${THREADS})${NC}"
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
    "${REASONING_ARGS[@]}" \
    "${MMPROJ_ARGS[@]}"
