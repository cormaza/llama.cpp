#!/usr/bin/env bash
# ==============================================================================
# start-glm-4.6v-flash.sh - Launcher for GLM-4.6V-Flash (Z.ai 9.4B VLM Agent)
# Model:    https://huggingface.co/prithivMLmods/GLM-4.6V-Flash-GGUF
# Base:     https://huggingface.co/zai-org/GLM-4.6V-Flash
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200)
# Features: GLM-4V 9.4B (GQA 16:1, 128K Context) + Multimodal Tool Use & CoT
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

DEFAULT_MODEL_I1_Q4="${SCRIPT_DIR}/models/GLM-4.6V-Flash.i1-Q4_K_M.gguf"
ALT_MODEL_Q4="${SCRIPT_DIR}/models/GLM-4.6V-Flash.Q4_K_M.gguf"
ALT_MODEL_I1_Q5="${SCRIPT_DIR}/models/GLM-4.6V-Flash.i1-Q5_K_M.gguf"
ALT_MODEL_Q5="${SCRIPT_DIR}/models/GLM-4.6V-Flash.Q5_K_M.gguf"
ALT_MODEL_I1_Q6="${SCRIPT_DIR}/models/GLM-4.6V-Flash.i1-Q6_K.gguf"
ALT_MODEL_Q6="${SCRIPT_DIR}/models/GLM-4.6V-Flash.Q6_K.gguf"
ALT_MODEL_Q8="${SCRIPT_DIR}/models/GLM-4.6V-Flash.Q8_0.gguf"
ALT_MODEL_I1_IQ4="${SCRIPT_DIR}/models/GLM-4.6V-Flash.i1-IQ4_XS.gguf"
ALT_MODEL_IQ4="${SCRIPT_DIR}/models/GLM-4.6V-Flash.IQ4_XS.gguf"

DEFAULT_MMPROJ_Q8="${SCRIPT_DIR}/models/mmproj-GLM-4.6V-Flash-Q8_0.gguf"
ALT_MMPROJ_BF16="${SCRIPT_DIR}/models/mmproj-zai-org_GLM-4.6V-Flash-bf16.gguf"
ALT_MMPROJ_F16="${SCRIPT_DIR}/models/mmproj-zai-org_GLM-4.6V-Flash-f16.gguf"

MODEL_PATH=""
MMPROJ_PATH=""
ENABLE_MMPROJ=1
ENABLE_THINKING=1
TEMPLATE_CHOICE="native"
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
CUSTOM_REPEAT_PENALTY=""
CUSTOM_PRESENCE=""
KV_QUANT="q4_0"
CUSTOM_NGL=""
ALIAS="glm-4.6v-flash,glm-4.6v,glm4v,glm-4v,glm"
THREADS=8

# Helper function to parse human-readable token notation (e.g., 16k, 32k, 64k, 128k)
parse_tokens() {
    local val="${1,,}"
    val="${val//[[:space:]]/}"
    case "${val}" in
        0)
            echo 131072
            return
            ;;
        128k|131k|131072)
            echo 131072
            return
            ;;
        64k|65k|65536)
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
    if [[ "$n" -eq 131072 ]]; then
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

Starts llama-server for GLM-4.6V-Flash (Z.ai 9.4B Vision-Language Agent)
optimized for AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP gfx1200).

Features:
  - 9.4B GLM-4V architecture with 16:1 GQA (only 2 KV heads across 40 layers)
  - 128K native context window with ultra-low KV cache footprint
  - Native Multimodal Function Calling (<tool_call> + <arg_key>/<arg_value> XML)
  - Vision Projector support (GLM4V dynamic resolution image/UI understanding)
  - Reasoning mode (<think>...</think>) & fast /nothink direct execution mode
  - 100% GPU Offload with Flash Attention & Q4_0 KV cache

Options:
  -m, --model PATH              Path to GLM-4.6V-Flash GGUF (default: auto-detect i1-Q4_K_M / Q4_K_M / Q5_K_M / Q8_0)
  -a, --alias NAMES             Model alias for API clients (default: ${ALIAS})
  --mmproj PATH                 Path to GLM4V multimodal vision projector (default: auto-detect)
  --no-mmproj                   Disable multimodal vision projector (enables infinite context shifting)
  -c, --context, --ctx-size, --total-ctx N  Total context pool across all slots (supports 32k, 64k, 128k; default: 128k)
  --ctx-slot N                  Explicit context per slot override (e.g. 32k, 64k, 128k)
  --slots, -np, --parallel N    Number of parallel agent slots (default: 1; use 2 or 4 for multi-agent)
  -kvu, --kv-unified            Enable dynamic unified KV cache pool shared across all slots
  --thinking                    Enable reasoning mode (default; Z.ai recommended temp 0.8, top_p 0.6, top_k 2, repeat 1.1)
  --no-thinking                 Disable reasoning mode (direct agent mode; temp 0.2, top_p 0.8, top_k 20)
  --template TYPE               Chat template: native (default, multimodal GGUF) | glm4 (models/templates/GLM-4.6.jinja) | PATH
  --temp N                      Sampling temperature override
  --top-p N                     Top-p sampling override
  --top-k N                     Top-k sampling override
  --repeat-penalty N            Repeat penalty override (default: 1.1)
  --presence-penalty N          Presence penalty override (default: 0.0)
  -t, --threads N               Number of CPU threads (default: 8)
  --context-shift               Enable context shifting
  --no-context-shift            Disable context shifting
  -p, --port PORT               HTTP server port (default: 8080)
  --host HOST                   Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE               KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                       GPU layers offloaded (default: 99, 100% GPU)
  -h, --help                    Show this help message

Examples:
  ./start-glm-4.6v-flash.sh                            # Full multimodal stack with reasoning (128k native context)
  ./start-glm-4.6v-flash.sh -c 64k                     # Conservative 64k context pool
  ./start-glm-4.6v-flash.sh --no-thinking              # Fast direct agent/tool mode without thinking
  ./start-glm-4.6v-flash.sh --no-mmproj                # Pure text mode with infinite context shifting
  ./start-glm-4.6v-flash.sh --slots 2 --ctx-slot 64k   # Dual-agent serving (64k context per slot)
  ./start-glm-4.6v-flash.sh -kvu -c 128k --slots 4     # Dynamic shared unified 128k KV pool across 4 slots
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
        --mmproj)
            MMPROJ_PATH="$2"
            shift 2
            ;;
        --mmproj=*)
            MMPROJ_PATH="${1#*=}"
            shift
            ;;
        --no-mmproj)
            ENABLE_MMPROJ=0
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
        --repeat-penalty)
            CUSTOM_REPEAT_PENALTY="$2"
            shift 2
            ;;
        --repeat-penalty=*)
            CUSTOM_REPEAT_PENALTY="${1#*=}"
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
echo -e "${BOLD}${CYAN}  GLM-4.6V-Flash VLM Agent Server (ROCm / HIP gfx1200)${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

# 1. Check binaries
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${SERVER_BIN}${NC}"
    echo -e "Please build first: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

# 2. Select Model
if [[ -z "${MODEL_PATH}" ]]; then
    for candidate in \
        "${DEFAULT_MODEL_I1_Q4}" \
        "${ALT_MODEL_Q4}" \
        "${ALT_MODEL_I1_Q5}" \
        "${ALT_MODEL_Q5}" \
        "${ALT_MODEL_I1_Q6}" \
        "${ALT_MODEL_Q6}" \
        "${ALT_MODEL_Q8}" \
        "${ALT_MODEL_I1_IQ4}" \
        "${ALT_MODEL_IQ4}"; do
        if [[ -f "${candidate}" ]]; then
            MODEL_PATH="${candidate}"
            break
        fi
    done

    if [[ -z "${MODEL_PATH}" ]]; then
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*glm*4.6v*flash*.gguf" ! -iname "*mmproj*" ! -iname "*imatrix*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            MODEL_PATH="${FOUND_MODELS[0]}"
        else
            echo -e "${YELLOW}[WARN] No GLM-4.6V-Flash GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-glm-4.6v-flash.sh${NC} to download GLM-4.6V-Flash."
            exit 1
        fi
    fi
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo -e "${RED}[ERROR] Specified model file does not exist: ${MODEL_PATH}${NC}"
    exit 1
fi

# 3. Vision Projector (Multimodal) Configuration
MMPROJ_ARGS=()
MMPROJ_ACTIVE=0
if [[ "${ENABLE_MMPROJ}" -eq 1 ]]; then
    if [[ -z "${MMPROJ_PATH}" ]]; then
        if [[ -f "${DEFAULT_MMPROJ_Q8}" ]]; then
            MMPROJ_PATH="${DEFAULT_MMPROJ_Q8}"
        elif [[ -f "${ALT_MMPROJ_BF16}" ]]; then
            MMPROJ_PATH="${ALT_MMPROJ_BF16}"
        elif [[ -f "${ALT_MMPROJ_F16}" ]]; then
            MMPROJ_PATH="${ALT_MMPROJ_F16}"
        else
            DETECTED_MMPROJ=($(find "${SCRIPT_DIR}/models" -maxdepth 1 \( -iname "*glm*4.6v*mmproj*.gguf" -o -iname "mmproj*glm*4.6v*.gguf" \) 2>/dev/null || true))
            if [[ ${#DETECTED_MMPROJ[@]} -gt 0 && -f "${DETECTED_MMPROJ[0]}" ]]; then
                MMPROJ_PATH="${DETECTED_MMPROJ[0]}"
            fi
        fi
    fi

    if [[ -n "${MMPROJ_PATH}" && -f "${MMPROJ_PATH}" ]]; then
        MMPROJ_ARGS=("--mmproj" "${MMPROJ_PATH}")
        MMPROJ_STATUS="Active ($(basename "${MMPROJ_PATH}"))"
        MMPROJ_ACTIVE=1
    else
        MMPROJ_STATUS="Disabled (no projector found; run ./scripts/download-glm-4.6v-flash.sh mmproj)"
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
else
    TOTAL_CTX=131072
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# Ensure minimum viable context per slot
if [[ "${CTX_PER_SLOT}" -lt 1024 ]]; then
    CTX_PER_SLOT=1024
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
fi

# GLM-4.6V-Flash native max context is 131072 (128k). With 2 KV heads, 128k easily fits in 16GB VRAM.
MAX_SAFE_CTX=131072
if [[ "${TOTAL_CTX}" -gt "${MAX_SAFE_CTX}" ]]; then
    echo -e "${YELLOW}[WARN] Total context (${TOTAL_CTX} tokens) exceeds the 128k native context limit.${NC}"
    echo -e "${YELLOW}[WARN] Ensure you have sufficient VRAM or adjust context size if running near limits.${NC}"
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

# 6. Sampling & Template Configuration
JINJA_ARGS=("--jinja")
REPEAT_PENALTY="${CUSTOM_REPEAT_PENALTY:-1.1}"
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-0.8}" # Official Z.ai GLM-4.6V-Flash evaluation default
    TOP_P="${CUSTOM_TOP_P:-0.60}"
    TOP_K="${CUSTOM_TOP_K:-2}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (GLM-4.6V Thinking mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, repeat ${REPEAT_PENALTY})"
    case "${TEMPLATE_CHOICE}" in
        native)
            TEMPLATE_STATUS="Embedded GGUF multimodal template (<think> + XML <tool_call>)"
            ;;
        glm4|glm-4.6|glm)
            if [[ -f "${SCRIPT_DIR}/models/templates/GLM-4.6.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/GLM-4.6.jinja")
            fi
            TEMPLATE_STATUS="Official GLM-4.6 template (models/templates/GLM-4.6.jinja)"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE})"
            else
                echo -e "${YELLOW}[WARN] Unknown template '${TEMPLATE_CHOICE}', using embedded GGUF template.${NC}"
                TEMPLATE_STATUS="Embedded GGUF multimodal template"
            fi
            ;;
    esac
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.2}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Disabled (/nothink direct agent mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    case "${TEMPLATE_CHOICE}" in
        native)
            JINJA_ARGS+=("--chat-template-kwargs" '{"enable_thinking": false}')
            TEMPLATE_STATUS="Embedded GGUF multimodal template (enable_thinking=false)"
            ;;
        glm4|glm-4.6|glm)
            if [[ -f "${SCRIPT_DIR}/models/templates/GLM-4.6.jinja" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/GLM-4.6.jinja")
            fi
            JINJA_ARGS+=("--chat-template-kwargs" '{"enable_thinking": false}')
            TEMPLATE_STATUS="Official GLM-4.6 template (enable_thinking=false)"
            ;;
        *)
            if [[ -f "${TEMPLATE_CHOICE}" ]]; then
                JINJA_ARGS+=("--chat-template-file" "${TEMPLATE_CHOICE}")
                JINJA_ARGS+=("--chat-template-kwargs" '{"enable_thinking": false}')
                TEMPLATE_STATUS="Custom (${TEMPLATE_CHOICE}, enable_thinking=false)"
            else
                JINJA_ARGS+=("--chat-template-kwargs" '{"enable_thinking": false}')
                TEMPLATE_STATUS="Embedded GGUF multimodal template (enable_thinking=false)"
            fi
            ;;
    esac
    REASONING_ARGS=("--reasoning-format" "none")
fi

GPU_LAYERS="${CUSTOM_NGL:-99}"

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}Architecture:${NC}        ${GREEN}GLM-4V 9.4B (40 layers, GQA 16:1 / 2 KV heads, M-RoPE)${NC}"
echo -e "${BOLD}Vision Projector:${NC}    ${GREEN}${MMPROJ_STATUS}${NC}"
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
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | repeat ${REPEAT_PENALTY} | presence ${PRESENCE_PENALTY}${NC}"
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
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl "${GPU_LAYERS}" \
    -fa on \
    -t "${THREADS}" \
    --cpu-range 0-7 \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --repeat-penalty "${REPEAT_PENALTY}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${KV_UNIFIED_ARGS[@]}" \
    "${CTX_SHIFT_ARGS[@]}" \
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    "${MMPROJ_ARGS[@]}"
