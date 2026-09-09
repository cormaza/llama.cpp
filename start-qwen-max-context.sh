#!/usr/bin/env bash
# ==============================================================================
# start-qwen-max-context.sh - Single-Thread High-Speed & 128k Context Server
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q2_K_XL.gguf"
ALT_MODEL_IQ3="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-IQ3_XXS.gguf"
ALT_MODEL_Q3="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"
ALT_MODEL_Q4="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q4_K_M.gguf"
DEFAULT_MTP="${SCRIPT_DIR}/models/mtp-Qwen3.8-27B-Q4_0.gguf"
MODEL_PATH=""
MTP_PATH=""
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
ALIAS="qwen-3.8-27b"
ENABLE_MTP=1
ENABLE_SPEC=1

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Qwen optimized for High-Speed Agentic Workflows,
with Froggeric Fixed Chat Templates, MTP speculative decoding (~6.5+ t/s in hybrid mode),
and official sampling parameters for Qwen 3.8.

Options:
  -a, --alias NAMES       Model alias for API clients (default: qwen-3.8-27b,qwen-27b,qwen,gpt-4o)
  -m, --model PATH        Path to GGUF model (default: ./models/Qwen3.8-27B-UD-Q2_K_XL.gguf or Q3_K_XL)
  --mtp PATH              Path to MTP draft model (default: ./models/mtp-Qwen3.8-27B-Q4_0.gguf)
  --no-mtp                Disable MTP (falls back to N-Gram speculative decoding)
  -c, --context, --ctx-slot N  Context per slot (default: 131072 for 1 slot, 65536 for 2 slots, 32768 for 4 slots)
  --slots N               Number of parallel agent slots (default: 1; use 2, 4 for multi-agent)
  --thinking              Enable reasoning mode (default; uses temp 1.0, top_p 0.95, top_k 20)
  --no-thinking           Disable reasoning mode (direct agent output; uses temp 0.7, top_p 0.80, top_k 20, presence 1.5)
  --temp N                Sampling temperature override (default: 1.0 with thinking, 0.7 without thinking)
  --top-p N               Top-p sampling override (default: 0.95 with thinking, 0.80 without thinking)
  --top-k N               Top-k sampling override (default: 20)
  --presence-penalty N    Presence penalty override (default: 0.0 with thinking, 1.5 without thinking)
  -p, --port PORT         HTTP server port (default: 8080)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                 Number of layers to offload to GPU (default: 99 for <11.5GB models; 42/50 for >12GB)
  --no-spec               Disable all speculative decoding
  -h, --help              Show this help message

Examples:
  ./start-qwen-max-context.sh               # Default thinking mode (temp 1.0, top_p 0.95, deepseek reasoning)
  ./start-qwen-max-context.sh --no-thinking # Direct fast coding (temp 0.7, top_p 0.80, presence 1.5)
  ./start-qwen-max-context.sh --slots 2     # 2 parallel slots
  ./start-qwen-max-context.sh --slots 4 -c 32768
  ./start-qwen-max-context.sh --temp 0.6
  ./start-qwen-max-context.sh -c 262144
  ./start-qwen-max-context.sh --no-mtp
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
        --mtp)
            MTP_PATH="$2"
            shift 2
            ;;
        --no-mtp)
            ENABLE_MTP=0
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
        -c|--context|--ctx-slot)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        --slots)
            SLOTS="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
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
echo -e "${BOLD}${CYAN}  Qwen High-Speed 128k Agent Server (MTP / ROCm)      ${NC}"
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
    elif [[ -f "${ALT_MODEL_IQ3}" ]]; then
        MODEL_PATH="${ALT_MODEL_IQ3}"
    elif [[ -f "${ALT_MODEL_Q3}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q3}"
    elif [[ -f "${ALT_MODEL_Q4}" ]]; then
        MODEL_PATH="${ALT_MODEL_Q4}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -iname "*qwen*27b*.gguf" ! -iname "mtp-*" 2>/dev/null || true))
        if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
            echo -e "
Select a model from ./models/:"
            select opt in "${FOUND_MODELS[@]}" "Descargar Qwen 27B" "Salir"; do
                if [[ -n "${opt}" && -f "${opt}" ]]; then
                    MODEL_PATH="${opt}"
                    break
                elif [[ "${opt}" == "Descargar Qwen 27B" ]]; then
                    ./scripts/download-model.sh
                    exit 0
                else
                    exit 1
                fi
            done
        else
            echo -e "${YELLOW}[WARN] No Qwen GGUF model found in ./models/${NC}"
            echo -e "Run ${CYAN}./scripts/download-model.sh${NC} to download Qwen3.8-27B."
            exit 1
        fi
    fi
fi

# 3. Setup MTP or Speculative Decoding & GPU Offload
SPEC_STATUS="Disabled"
SPEC_ARGS=()

MODEL_SIZE_BYTES=0
if [[ -f "${MODEL_PATH}" ]]; then
    MODEL_SIZE_BYTES=$(stat -c%s "${MODEL_PATH}" 2>/dev/null || echo 0)
fi

# Models under 11.5 GB (Q2_K_XL 9.15GB, IQ3_XXS 10.18GB) fit 100% in 16GB VRAM
if [[ "${MODEL_SIZE_BYTES}" -gt 0 && "${MODEL_SIZE_BYTES}" -lt 11500000000 ]]; then
    DEFAULT_GPU_LAYERS=99
else
    # Heavy models (>=12GB like Q3_K_XL / Q4_K_M) need partial offload to prevent HIP OOM
    DEFAULT_GPU_LAYERS=50
fi

if [[ "${ENABLE_SPEC}" -eq 1 ]]; then
    if [[ "${ENABLE_MTP}" -eq 1 ]]; then
        if [[ -z "${MTP_PATH}" && -f "${DEFAULT_MTP}" ]]; then
            MTP_PATH="${DEFAULT_MTP}"
        fi

        if [[ -n "${MTP_PATH}" && -f "${MTP_PATH}" ]]; then
            SPEC_STATUS="Active (Multi-Token Prediction: $(basename "${MTP_PATH}"))"
            SPEC_ARGS+=("--spec-type" "draft-mtp" "-md" "${MTP_PATH}" "-ngld" "99")
            if [[ "${MODEL_SIZE_BYTES}" -ge 11500000000 ]]; then
                DEFAULT_GPU_LAYERS=42 # Heavy model + MTP: offload 42 layers to avoid HIP OOM
            fi
        else
            SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
            SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
        fi
    else
        SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
        SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
    fi
fi

GPU_LAYERS="${CUSTOM_NGL:-${DEFAULT_GPU_LAYERS}}"

# 4. Context calculation per slot
if [[ -n "${CUSTOM_CTX}" ]]; then
    CTX_PER_SLOT="${CUSTOM_CTX}"
elif [[ "${SLOTS}" -le 1 ]]; then
    CTX_PER_SLOT=131072 # 128k context for single slot
elif [[ "${SLOTS}" -le 2 ]]; then
    CTX_PER_SLOT=65536  # 64k context per slot for 2 slots
elif [[ "${SLOTS}" -le 4 ]]; then
    CTX_PER_SLOT=32768  # 32k context per slot for 4 slots
else
    CTX_PER_SLOT=16384  # 16k context per slot for 8 slots
fi

TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

# 5. Template & Sampling Configuration (Froggeric Qwen-Fixed Recommendations)
JINJA_ARGS=("--jinja")
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
    TEMPERATURE="${CUSTOM_TEMP:-1.0}"
    TOP_P="${CUSTOM_TOP_P:-0.95}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-0.0}"
    THINKING_STATUS="Active (Froggeric Fixed Jinja, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "deepseek")
else
    TEMPERATURE="${CUSTOM_TEMP:-0.7}"
    TOP_P="${CUSTOM_TOP_P:-0.80}"
    TOP_K="${CUSTOM_TOP_K:-20}"
    PRESENCE_PENALTY="${CUSTOM_PRESENCE:-1.5}"
    THINKING_STATUS="Disabled (Direct agent mode, temp ${TEMPERATURE}, top_p ${TOP_P}, top_k ${TOP_K}, presence ${PRESENCE_PENALTY})"
    if [[ -f "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja" ]]; then
        JINJA_ARGS+=("--chat-template-file" "${SCRIPT_DIR}/models/templates/Qwen-Fixed-no-thinking.jinja")
    fi
    REASONING_ARGS=("--reasoning-format" "none")
fi

if [[ "${GPU_LAYERS}" -ge 65 ]]; then
    OFFLOAD_DESC="All 65 layers offloaded (100% in GPU VRAM)"
else
    OFFLOAD_DESC="${GPU_LAYERS} layers to GPU (Hybrid mode: $(( 65 - GPU_LAYERS )) layers on CPU)"
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Parallel Slots:${NC}      ${GREEN}${SLOTS} slots${NC}"
echo -e "${BOLD}Context per Slot:${NC}    ${GREEN}${CTX_PER_SLOT} tokens ($(( CTX_PER_SLOT / 1024 ))k tokens)${NC}"
echo -e "${BOLD}Total Context Pool:${NC}  ${GREEN}${TOTAL_CTX} tokens ($(( TOTAL_CTX / 1024 ))k tokens)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}${OFFLOAD_DESC} (-ngl ${GPU_LAYERS} -fa auto)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${SPEC_STATUS}${NC}"
echo -e "${BOLD}Thinking Mode:${NC}       ${GREEN}${THINKING_STATUS}${NC}"
echo -e "${BOLD}Chat Template:${NC}       ${GREEN}Froggeric Qwen-Fixed v22.5 (--jinja enabled)${NC}"
echo -e "${BOLD}Sampling Params:${NC}     ${GREEN}temp ${TEMPERATURE} | top_p ${TOP_P} | top_k ${TOP_K} | presence ${PRESENCE_PENALTY}${NC}"
echo -e "${BOLD}CPU Acceleration:${NC}    ${GREEN}Intel Core Ultra 7 265K (AVX_VNNI, -t 8)${NC}"
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
    -t 8 \
    --temp "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    "${JINJA_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    "${SPEC_ARGS[@]}"
