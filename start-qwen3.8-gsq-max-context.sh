#!/usr/bin/env bash
# ==============================================================================
# start-qwen3.8-gsq-max-context.sh - Single-Agent Native 262K Context Server
# Model: Qwen3.8-27B GSQ-RCO with MTP and Multimodal Vision
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
CUSTOM_TEMP=""
CUSTOM_TOP_P=""
CUSTOM_TOP_K=""
CUSTOM_PRESENCE=""
ENABLE_THINKING=1
KV_QUANT="q4_0"
CUSTOM_NGL=""
ALIAS="qwen-3.8-27b,qwen-27b,qwen,gpt-4o"
THREADS=8
ENABLE_SPEC=1

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Qwen3.8-27B GSQ-RCO configured for Ultra-Deep Context
(native support for the full 262,144 tokens / 262K context window),
with Multimodal Vision (mmproj), MTP speculative decoding, and GPU offloading
on AMD Radeon RX 9060 XT (16GB VRAM).

Options:
  -m, --model PATH        Path to GGUF model (default: auto-detect Qwen3.8 GSQ-RCO)
  -a, --alias NAMES       Model alias for API clients (default: ${ALIAS})
  --mmproj PATH           Path to multimodal vision projector (default: auto-detect)
  --no-mmproj             Disable multimodal vision projector
  --draft-n N             MTP speculative draft depth (default: 2)
  --mtp PATH              Path to external MTP draft model (if using non-mtp base)
  --no-mtp                Disable MTP speculative decoding (falls back to N-Gram lookup)
  -c, --context N         Context size (default: 262144, full native context)
  --thinking              Enable reasoning mode (default; temp 1.0, top_p 0.95, top_k 20)
  --no-thinking           Disable reasoning mode (direct response; temp 0.7, top_p 0.80, presence 1.5)
  --temp N                Sampling temperature override
  --top-p N               Top-p sampling override
  --top-k N               Top-k sampling override
  --presence-penalty N    Presence penalty override
  -p, --port PORT         HTTP server port (default: 8080)
  --host HOST             Host address to bind (default: 0.0.0.0)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                 GPU layers offloaded (default: 99 for 100% GPU offload)
  --no-spec               Disable all speculative decoding
  -t, --threads N         Number of CPU threads (default: 8)
  -h, --help              Show this help message

Examples:
  ./start-qwen3.8-gsq-max-context.sh               # 262k native context with MTP & Vision
  ./start-qwen3.8-gsq-max-context.sh -c 131072     # 128k context
  ./start-qwen3.8-gsq-max-context.sh --no-thinking # Direct deep repo ingestion without reasoning
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
        -c|--context)
            CUSTOM_CTX="$2"
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
echo -e "${BOLD}${CYAN}  Qwen3.8-27B GSQ Native 262K Server (ROCm / HIP)     ${NC}"
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

# 5. Context & GPU Layers Allocation
CONTEXT="${CUSTOM_CTX:-262144}"

MODEL_SIZE_BYTES=0
if [[ -f "${MODEL_PATH}" ]]; then
    MODEL_SIZE_BYTES=$(stat -c%s "${MODEL_PATH}" 2>/dev/null || echo 0)
fi

# Determine safe GPU layer offload
# For deep contexts (>128k up to 262k), 48 layers offload to GPU and 16 layers run on CPU RAM
# to keep model + mmproj + massive 262k KV cache within 16GB VRAM (14.89 GB total)
if [[ "${CONTEXT}" -gt 131072 ]]; then
    DEFAULT_GPU_LAYERS=48
elif [[ "${MODEL_SIZE_BYTES}" -gt 0 && "${MODEL_SIZE_BYTES}" -lt 11500000000 ]]; then
    DEFAULT_GPU_LAYERS=99
else
    DEFAULT_GPU_LAYERS=50
fi

GPU_LAYERS="${CUSTOM_NGL:-${DEFAULT_GPU_LAYERS}}"

# 6. Chat Template & Sampling
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
echo -e "${BOLD}Context Window:${NC}      ${GREEN}${CONTEXT} tokens ($(( CONTEXT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}1 slot (Dedicated deep ingestion)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT} (-ctk ${KV_QUANT} -ctv ${KV_QUANT})${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}-ngl ${GPU_LAYERS} on AMD Radeon RX 9060 XT (-fa on)${NC}"
echo -e "${BOLD}Batching:${NC}            ${GREEN}Continuous (-cb) | Chunked Prefill (-ub 1024, -b 2048)${NC}"
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

exec "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${CONTEXT}" \
    -np 1 \
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
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    "${MMPROJ_ARGS[@]}"
