#!/usr/bin/env bash
# ==============================================================================
# start-bonsai-27b.sh - Launcher for Ternary Bonsai 27B (Prism ML / Qwen3.6)
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200)
# Features: True 1.71 bpw Ternary PQ2_0 + DSpark Speculative Drafter + Vision
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Ternary-Bonsai-27B-PQ2_0.gguf"
ALT_MODEL_G64="${SCRIPT_DIR}/models/Ternary-Bonsai-27B-Q2_g64.gguf"
ALT_MODEL_Q2="${SCRIPT_DIR}/models/Ternary-Bonsai-27B-Q2_0.gguf"

DEFAULT_DSPARK="${SCRIPT_DIR}/models/Ternary-Bonsai-27B-dspark-Q4_1.gguf"
DEFAULT_DFLASH="${SCRIPT_DIR}/models/Ternary-Bonsai-27B-dflash-Q4_1.gguf"
DEFAULT_MMPROJ="${SCRIPT_DIR}/models/Ternary-Bonsai-27B-mmproj-Q8_0.gguf"

MODEL_PATH=""
DSPARK_PATH=""
MMPROJ_PATH=""
ENABLE_DSPARK=1
ENABLE_MMPROJ=1
ENABLE_THINKING=1
ENABLE_CTX_SHIFT=1

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
ALIAS="bonsai-27b,ternary-bonsai,qwen-27b,gpt-4o"
THREADS=8

# Detect Primary LAN IP
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Ternary Bonsai 27B (Prism ML / Qwen3.6 Hybrid Attention)
optimized for AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP).

Features:
  - True 1.71 bits/weight ternary weights (PQ2_0, 7.17 GB)
  - DSpark speculative-decoding drafter layer (1.34x decode speedup)
  - Multimodal Vision Projector (mmproj HQQ 4-bit)
  - 100% GPU Offload with Flash Attention & Q4_0 KV cache

Options:
  -m, --model PATH        Path to GGUF model (default: ./models/Ternary-Bonsai-27B-PQ2_0.gguf)
  -a, --alias NAMES       Model alias for API clients (default: ${ALIAS})
  --dspark PATH           Path to DSpark draft model (default: auto-detect)
  --no-dspark             Disable DSpark speculative decoding
  --mmproj PATH           Path to multimodal vision projector (default: auto-detect)
  --no-mmproj             Disable multimodal vision projector (enables context shifting)
  -c, --context N         Total context size pool (default: 131072 / 128k; up to 262144)
  --ctx-slot N            Explicit context per slot override
  --slots N               Number of parallel slots (default: 1; use 2 or 4 for multi-agent)
  --thinking              Enable reasoning mode (default; temp 0.7, top_p 0.95, top_k 20)
  --no-thinking           Disable reasoning mode (direct response mode)
  --temp N                Sampling temperature override (default: 0.7)
  --top-p N               Top-p sampling override (default: 0.95)
  --top-k N               Top-k sampling override (default: 20)
  --presence-penalty N    Presence penalty override (default: 0.0)
  -t, --threads N         Number of CPU threads (default: 8)
  --no-context-shift      Disable context shifting
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                 GPU layers offloaded (default: 99, 100% GPU)
  -h, --help              Show this help message

Examples:
  ./start-bonsai-27b.sh                    # Full stack (Ternary + DSpark + Vision, 128k context)
  ./start-bonsai-27b.sh -c 262144          # Max 262k native context window
  ./start-bonsai-27b.sh --no-mmproj        # Pure text mode with infinite context shifting
  ./start-bonsai-27b.sh --slots 2          # Dual-agent serving (64k context per slot)
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
        --dspark)
            DSPARK_PATH="$2"
            shift 2
            ;;
        --no-dspark)
            ENABLE_DSPARK=0
            shift
            ;;
        --mmproj)
            MMPROJ_PATH="$2"
            shift 2
            ;;
        --no-mmproj)
            ENABLE_MMPROJ=0
            shift
            ;;
        -c|--context)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        --ctx-slot)
            CUSTOM_CTX_SLOT="$2"
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
echo -e "${BOLD}${CYAN}  Ternary Bonsai 27B Server (ROCm / HIP Accelerated)  ${NC}"
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
    elif [[ -f "${ALT_MODEL_G64}" ]]; then
        MODEL_PATH="${ALT_MODEL_G64}"
    elif [[ -f "${ALT_MODEL_Q2}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q2}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*bonsai*.gguf" ! -iname "*mmproj*" ! -iname "*dspark*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            MODEL_PATH="${FOUND_MODELS[0]}"
        else
            echo -e "${YELLOW}[WARN] No Bonsai-27B GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-bonsai-27b.sh${NC} to download Ternary Bonsai 27B."
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
        else
            DETECTED_MMPROJ=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*bonsai*mmproj*.gguf" -o -iname "mmproj*bonsai*.gguf" 2>/dev/null || true))
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
        MMPROJ_STATUS="Disabled (no projector found; run ./scripts/download-bonsai-27b.sh mmproj)"
    fi
else
    MMPROJ_ARGS=("--no-mmproj")
    MMPROJ_STATUS="Disabled (--no-mmproj)"
fi

# 4. Speculative Decoding Configuration (DSpark / DFlash)
DSPARK_ARGS=()
DSPARK_STATUS="Disabled"
if [[ "${ENABLE_DSPARK}" -eq 1 ]]; then
    if [[ -z "${DSPARK_PATH}" ]]; then
        if [[ -f "${DEFAULT_DFLASH}" ]]; then
            DSPARK_PATH="${DEFAULT_DFLASH}"
        elif [[ -f "${DEFAULT_DSPARK}" ]]; then
            echo -e "${YELLOW}Converting legacy DSpark drafter to native dflash format...${NC}"
            python3 "${SCRIPT_DIR}/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py" --drop-shared-tensors "${DEFAULT_DSPARK}" "${MODEL_PATH}" "${DEFAULT_DFLASH}"
            DSPARK_PATH="${DEFAULT_DFLASH}"
        else
            DETECTED_DFLASH=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*bonsai*dflash*.gguf" 2>/dev/null || true))
            if [[ ${#DETECTED_DFLASH[@]} -gt 0 && -f "${DETECTED_DFLASH[0]}" ]]; then
                DSPARK_PATH="${DETECTED_DFLASH[0]}"
            fi
        fi
    fi

    if [[ -n "${DSPARK_PATH}" && -f "${DSPARK_PATH}" ]]; then
        DSPARK_ARGS=("-md" "${DSPARK_PATH}" "-ngld" "99")
        DSPARK_STATUS="Active ($(basename "${DSPARK_PATH}"), up to 1.7x speedup)"
    else
        DSPARK_STATUS="Disabled (no drafter found; run ./scripts/download-bonsai-27b.sh dspark)"
    fi
fi

# 5. Context Calculation & VRAM Bounds
# Bonsai 27B native max context is 262144. In 16GB VRAM, 262k is safe due to hybrid linear attention.
MAX_SAFE_CTX=262144
if [[ -n "${CUSTOM_CTX_SLOT}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX_SLOT}"
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
elif [[ -n "${CUSTOM_CTX}" ]]; then
    if [[ "${CUSTOM_CTX}" -ge 131072 && "${SLOTS}" -gt 1 ]]; then
        TOTAL_CTX="${CUSTOM_CTX}"
        CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
    else
        CTX_PER_SLOT="${CUSTOM_CTX}"
        TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
    fi
elif [[ "${SLOTS}" -eq 1 ]]; then
    TOTAL_CTX=131072 # 128k default for single slot (instant startup and low VRAM)
    CTX_PER_SLOT="${TOTAL_CTX}"
elif [[ "${SLOTS}" -le 2 ]]; then
    TOTAL_CTX=131072 # 64k per slot for 2 slots
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
else
    TOTAL_CTX=262144 # 32k-64k per slot for 4-8 slots
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

if [[ "${TOTAL_CTX}" -gt "${MAX_SAFE_CTX}" ]]; then
    echo -e "${YELLOW}[WARN] Total context (${TOTAL_CTX}) exceeds the 262k safe limit for 16GB VRAM.${NC}"
    echo -e "${YELLOW}[WARN] Capping total context to ${MAX_SAFE_CTX} ($(( MAX_SAFE_CTX / SLOTS )) per slot) to avoid Out-Of-Memory crashes.${NC}"
    TOTAL_CTX="${MAX_SAFE_CTX}"
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# 6. Context Shift
CTX_SHIFT_ARGS=()
if [[ "${MMPROJ_ACTIVE}" -eq 1 ]]; then
    CTX_SHIFT_STATUS="Disabled (Multimodal active; pass --no-mmproj for infinite context shifting)"
elif [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite continuous operation)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

# 7. Sampling & Template Configuration
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-0.7}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Thinking mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.6}"
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
echo -e "${BOLD}Quantization:${NC}        ${GREEN}True 1.71 bpw Ternary (PQ2_0 / Q2_0_g128)${NC}"
echo -e "${BOLD}DSpark Drafter:${NC}      ${GREEN}${DSPARK_STATUS}${NC}"
echo -e "${BOLD}Vision Projector:${NC}    ${GREEN}${MMPROJ_STATUS}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slot(s)${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}All layers offloaded to GPU (-ngl ${GPU_LAYERS} -fa on)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub 1024, -b 2048)${NC}"
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
    -b 2048 \
    -ub 1024 \
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
    "${CTX_SHIFT_ARGS[@]}" \
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    "${DSPARK_ARGS[@]}" \
    "${MMPROJ_ARGS[@]}"
