#!/usr/bin/env bash
# ==============================================================================
# start-qwen3.8-gsq.sh - Launcher for Qwen3.8-27B GSQ-RCO (ROCm / HIP Accelerated)
# Features: Built-in MTP Speculative Decoding + Multimodal Vision (mmproj)
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf"
ALT_MODEL_XS="${SCRIPT_DIR}/models/Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf"
ALT_MODEL_3XXS="${SCRIPT_DIR}/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
ALT_MODEL_3S="${SCRIPT_DIR}/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
ALT_MODEL_IQ2S="${SCRIPT_DIR}/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf"
DEFAULT_MMPROJ="${SCRIPT_DIR}/models/mmproj-Qwen3.8-27B-BF16.gguf"
DEFAULT_EXTERNAL_MTP="${SCRIPT_DIR}/models/mtp-Qwen3.8-27B-Q4_0.gguf"

MODEL_PATH=""
MMPROJ_PATH=""
MTP_PATH=""
ENABLE_MMPROJ=1
ENABLE_MTP=1
DRAFT_N_MAX=2
HOST="0.0.0.0"
PORT=8080
SLOTS=1
CUSTOM_CTX=""
CUSTOM_CTX_SLOT=""
CUSTOM_TEMP=""
CUSTOM_TOP_P=""
CUSTOM_TOP_K=""
CUSTOM_PRESENCE=""
ENABLE_THINKING=1
KV_QUANT="q4_0"
CUSTOM_NGL=""
ALIAS="qwen-3.8-27b,qwen-27b,qwen,gpt-4o"
THREADS=8
ENABLE_CTX_SHIFT=1
ENABLE_SPEC=1
ENABLE_KV_UNIFIED=0

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

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Qwen3.8-27B GSQ-RCO (Riemannian Constrained Optimization)
optimized for AMD Radeon RX 9060 XT (16GB VRAM), featuring:
  - Multimodal Vision Support (mmproj for image, screenshot & UI reasoning)
  - Built-in Multi-Token Prediction (MTP) speculative decoding acceleration
  - Froggeric v22.5 Qwen Fixed Jinja chat templates
  - 100% GPU Offload with quantized KV cache (q4_0) & Flash Attention

Options:
  -m, --model PATH              Path to GGUF model (default: auto-detect Qwen3.8 GSQ-RCO)
  -a, --alias NAMES             Model alias for API clients (default: ${ALIAS})
  --mmproj PATH                 Path to multimodal vision projector (default: auto-detect)
  --no-mmproj                   Disable multimodal vision projector
  --draft-n N                   MTP speculative draft depth (default: 2)
  --mtp PATH                    Path to external MTP draft model (if using non-mtp base)
  --no-mtp                      Disable MTP speculative decoding (falls back to N-Gram lookup)
  -c, --context, --total-ctx N  Total context pool across all slots (supports 64k, 128k, 256k)
  --ctx-slot N                  Explicit context per slot (e.g. 32k, 64k; total = slots * ctx_slot)
  --slots, -np N                Number of parallel slots (default: 1; use 2, 4 for multi-agent)
  -kvu, --kv-unified            Enable dynamic unified KV cache pool shared across all slots
  --thinking                    Enable reasoning mode (default; temp 1.0, top_p 0.95, top_k 20)
  --no-thinking                 Disable reasoning mode (direct response; temp 0.7, top_p 0.80, presence 1.5)
  --temp N                      Sampling temperature override
  --top-p N                     Top-p sampling override
  --top-k N                     Top-k sampling override
  --presence-penalty N          Presence penalty override
  -p, --port PORT               HTTP server port (default: 8080)
  --host HOST                   Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE               KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                       GPU layers offloaded (default: 99 for 100% GPU offload)
  --no-spec                     Disable all speculative decoding
  --context-shift               Enable continuous context shifting (default: enabled)
  --no-context-shift            Disable context shifting
  -t, --threads N               Number of CPU threads (default: 8)
  -h, --help                    Show this help message

Examples:
  ./start-qwen3.8-gsq.sh                            # 1 slot x 128k context, MTP + Vision enabled
  ./start-qwen3.8-gsq.sh --no-thinking              # Fast direct coding without <think> tags
  ./start-qwen3.8-gsq.sh --slots 2                  # 2 parallel agent slots x 128k (256k pool)
  ./start-qwen3.8-gsq.sh --slots 4 --ctx-slot 64k   # 4 slots x 64k (256k total pool)
  ./start-qwen3.8-gsq.sh -c 262144                  # 256k native deep context
  ./start-qwen3.8-gsq.sh -kvu -c 256k --slots 4     # Dynamic shared unified KV pool
  ./start-qwen3.8-gsq.sh --no-mmproj                # Pure text mode (saves ~0.9 GB VRAM)
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
        --draft-n)
            DRAFT_N_MAX="$2"
            shift 2
            ;;
        --mtp)
            MTP_PATH="$2"
            shift 2
            ;;
        --no-mtp)
            ENABLE_MTP=0
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
        --slots|-np|--parallel)
            SLOTS="$2"
            shift 2
            ;;
        -kvu|--kv-unified)
            ENABLE_KV_UNIFIED=1
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
        --ngl)
            CUSTOM_NGL="$2"
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
echo -e "${BOLD}${CYAN}  Qwen3.8-27B GSQ-RCO Server (ROCm / HIP Accelerated) ${NC}"
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
    elif [[ -f "${ALT_MODEL_XS}" ]]; then
        MODEL_PATH="${ALT_MODEL_XS}"
    elif [[ -f "${ALT_MODEL_3XXS}" ]]; then
        MODEL_PATH="${ALT_MODEL_3XXS}"
    elif [[ -f "${ALT_MODEL_3S}" ]]; then
        MODEL_PATH="${ALT_MODEL_3S}"
    elif [[ -f "${ALT_MODEL_IQ2S}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ2S}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*qwen3.8*gsq*.gguf" ! -iname "*mmproj*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            echo -e "\nSelect Qwen3.8 GSQ-RCO model from ./models/:"
            select opt in "${FOUND_MODELS[@]}" "Descargar Qwen3.8 GSQ-RCO" "Salir"; do
                if [[ -n "${opt}" && -f "${opt}" ]]; then
                    MODEL_PATH="${opt}"
                    break
                elif [[ "${opt}" == "Descargar Qwen3.8 GSQ-RCO" ]]; then
                    ./scripts/download-qwen3.8-gsq.sh
                    exit 0
                else
                    exit 1
                fi
            done
        else
            echo -e "${YELLOW}[WARN] No Qwen3.8-27B GSQ-RCO GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-qwen3.8-gsq.sh${NC} to download the model."
            exit 1
        fi
    fi
fi

# 3. Vision Projector (Multimodal) Configuration
MMPROJ_ARGS=()
MMPROJ_ACTIVE=0
if [[ "${ENABLE_MMPROJ}" -eq 1 ]]; then
    if [[ -z "${MMPROJ_PATH}" ]]; then
        if [[ -f "${DEFAULT_MMPROJ}" ]]; then
            MMPROJ_PATH="${DEFAULT_MMPROJ}"
        else
            DETECTED_MMPROJ=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*qwen3.8*mmproj*.gguf" -o -iname "mmproj*qwen3.8*.gguf" 2>/dev/null || true))
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
        MMPROJ_STATUS="Disabled (no projector found; run ./scripts/download-qwen3.8-gsq.sh mmproj)"
    fi
else
    MMPROJ_ARGS=("--no-mmproj")
    MMPROJ_STATUS="Disabled (--no-mmproj)"
fi

# 4. MTP / Speculative Decoding Configuration
SPEC_STATUS="Disabled"
SPEC_ARGS=()

MODEL_FILENAME="$(basename "${MODEL_PATH}")"

if [[ "${ENABLE_SPEC}" -eq 1 ]]; then
    if [[ "${ENABLE_MTP}" -eq 1 ]]; then
        # Check if the model has a built-in MTP head (e.g. *-mtp.gguf)
        if [[ "${MODEL_FILENAME}" == *"-mtp"* || "${MODEL_FILENAME}" == *"_mtp"* ]]; then
            SPEC_STATUS="Active (Built-in MTP Head, draft_n_max=${DRAFT_N_MAX})"
            SPEC_ARGS+=("--spec-type" "draft-mtp" "--spec-draft-n-max" "${DRAFT_N_MAX}")
        else
            # Check for external MTP draft model
            if [[ -z "${MTP_PATH}" && -f "${DEFAULT_EXTERNAL_MTP}" ]]; then
                MTP_PATH="${DEFAULT_EXTERNAL_MTP}"
            fi

            if [[ -n "${MTP_PATH}" && -f "${MTP_PATH}" ]]; then
                SPEC_STATUS="Active (External MTP: $(basename "${MTP_PATH}"), draft_n_max=${DRAFT_N_MAX})"
                SPEC_ARGS+=("--spec-type" "draft-mtp" "-md" "${MTP_PATH}" "-ngld" "99" "--spec-draft-n-max" "${DRAFT_N_MAX}")
            else
                SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
                SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
            fi
        fi
    else
        SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
        SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
    fi
fi

# 5. Context calculation per slot and VRAM safety bounds
MAX_SAFE_CTX=262144
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
    CTX_PER_SLOT=131072 # 128k context per slot for 1-2 slots
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
elif [[ "${SLOTS}" -le 4 ]]; then
    TOTAL_CTX=262144
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
else
    TOTAL_CTX=262144
    CTX_PER_SLOT=$(( TOTAL_CTX / SLOTS ))
fi

# Ensure minimum viable context per slot
if [[ "${CTX_PER_SLOT}" -lt 1024 ]]; then
    CTX_PER_SLOT=1024
    TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))
fi

if [[ "${TOTAL_CTX}" -gt "${MAX_SAFE_CTX}" ]]; then
    echo -e "${YELLOW}[WARN] Total context (${TOTAL_CTX} tokens) exceeds the 262k safe capacity for 16GB VRAM.${NC}"
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

# 6. GPU Layers Allocation (Smart memory budgeting for 16GB VRAM)
MODEL_SIZE_BYTES=0
if [[ -f "${MODEL_PATH}" ]]; then
    MODEL_SIZE_BYTES=$(stat -c%s "${MODEL_PATH}" 2>/dev/null || echo 0)
fi

# For total context > 128k, offload 48 layers to keep model + mmproj + KV within 16GB VRAM
if [[ "${TOTAL_CTX}" -gt 131072 ]]; then
    DEFAULT_GPU_LAYERS=48
elif [[ "${MODEL_SIZE_BYTES}" -gt 0 && "${MODEL_SIZE_BYTES}" -lt 11500000000 ]]; then
    DEFAULT_GPU_LAYERS=99
else
    DEFAULT_GPU_LAYERS=56
fi

GPU_LAYERS="${CUSTOM_NGL:-${DEFAULT_GPU_LAYERS}}"

# 7. Context Shift
# llama-server strictly disables ctx_shift when mmproj is loaded (multimodal chunks cannot be shifted)
CTX_SHIFT_ARGS=()
if [[ "${MMPROJ_ACTIVE}" -eq 1 ]]; then
    CTX_SHIFT_STATUS="Disabled (Multimodal active; pass --no-mmproj for infinite context shifting)"
elif [[ "${ENABLE_CTX_SHIFT}" -eq 1 ]]; then
    CTX_SHIFT_ARGS+=("--context-shift")
    CTX_SHIFT_STATUS="Active (Infinite continuous operation)"
else
    CTX_SHIFT_STATUS="Disabled"
fi

# 8. Chat Template & Sampling (Froggeric Qwen Fixed templates)
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-1.0}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Froggeric Fixed Jinja, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct agent mode, temp ${TEMPERATURE}, top_p ${TOP_P}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "none")
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}Vision Projector:${NC}    ${GREEN}${MMPROJ_STATUS}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Context Shift:${NC}       ${GREEN}${CTX_SHIFT_STATUS}${NC}"
echo -e "${BOLD}KV Cache Allocation:${NC} ${GREEN}${KV_UNIFIED_STATUS}${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}-ngl ${GPU_LAYERS} on AMD Radeon RX 9060 XT (-fa on)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub ${UBATCH_SIZE}, -b ${BATCH_SIZE})${NC}"
echo -e "${BOLD}CPU Affinity:${NC}        ${GREEN}Pinned to 8 P-cores (--cpu-range 0-7, -t ${THREADS})${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${SPEC_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}Froggeric Qwen-Fixed v22.5 (--jinja enabled)${NC}"
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"
echo -e "\n${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------\n"

# Expose slot debug information
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
    "${SPEC_ARGS[@]}" \
    "${MMPROJ_ARGS[@]}"
