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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"
ALT_MODEL="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q4_K_M.gguf"
DEFAULT_MTP="${SCRIPT_DIR}/models/mtp-Qwen3.8-27B-Q4_0.gguf"
MODEL_PATH=""
MTP_PATH=""
HOST="0.0.0.0"
PORT=8080
CTX_SIZE=131072 # 128k context (Optimized for speed & agentic workflows)
KV_QUANT="q4_0"
CUSTOM_NGL=""
ALIAS="qwen-3.8-27b,qwen-27b,qwen,gpt-4o"
ENABLE_MTP=1
ENABLE_SPEC=1

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Qwen optimized for High-Speed Agentic Workflows (128k context),
with MTP (Multi-Token Prediction) speculative decoding (~6.5+ t/s in hybrid mode) and DRY anti-loop protection.

Options:
  -a, --alias NAMES       Model alias for API clients (default: qwen-3.8-27b,qwen-27b,qwen,gpt-4o)
  -m, --model PATH        Path to GGUF model (default: ./models/Qwen3.8-27B-UD-Q3_K_XL.gguf)
  --mtp PATH              Path to MTP draft model (default: ./models/mtp-Qwen3.8-27B-Q4_0.gguf)
  --no-mtp                Disable MTP (falls back to N-Gram speculative decoding)
  -c, --context N         Context window size (default: 131072 / 128k tokens, supports up to 262144)
  -p, --port PORT         HTTP server port (default: 8080)
  --kv-quant TYPE         KV Cache precision: q4_0 (default, fast) | q8_0 | f16
  --ngl N                 Number of layers to offload to GPU (default: 42 with MTP, 50 without MTP)
  --no-spec               Disable all speculative decoding
  -h, --help              Show this help message

Examples:
  ./start-qwen-max-context.sh
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
        -c|--context)
            CTX_SIZE="$2"
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
    elif [[ -f "${ALT_MODEL}" ]]; then
        MODEL_PATH="${ALT_MODEL}"
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

# 3. Setup MTP or Speculative Decoding
SPEC_STATUS="Disabled"
SPEC_ARGS=()
GPU_LAYERS=50

if [[ "${ENABLE_SPEC}" -eq 1 ]]; then
    if [[ "${ENABLE_MTP}" -eq 1 ]]; then
        if [[ -z "${MTP_PATH}" && -f "${DEFAULT_MTP}" ]]; then
            MTP_PATH="${DEFAULT_MTP}"
        fi

        if [[ -n "${MTP_PATH}" && -f "${MTP_PATH}" ]]; then
            SPEC_STATUS="Active (Multi-Token Prediction: $(basename "${MTP_PATH}"))"
            SPEC_ARGS+=("--spec-type" "draft-mtp" "-md" "${MTP_PATH}" "-ngld" "99")
            GPU_LAYERS=42 # Balanced offload: keeps model + MTP + KV cache safely in 16GB VRAM
        else
            SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
            SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
            GPU_LAYERS=50
        fi
    else
        SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
        SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
        GPU_LAYERS=50
    fi
fi

if [[ -n "${CUSTOM_NGL}" ]]; then
    GPU_LAYERS="${CUSTOM_NGL}"
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}API Model Alias:${NC}     ${GREEN}${ALIAS}${NC}"
echo -e "${BOLD}Mode:${NC}                ${GREEN}Single Slot (High-Speed Agent)${NC}"
echo -e "${BOLD}Context Size:${NC}        ${GREEN}${CTX_SIZE} tokens ($(( CTX_SIZE / 1024 ))k tokens)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}${GPU_LAYERS} layers to AMD Radeon RX 9060 XT (-ngl ${GPU_LAYERS} -fa auto)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${SPEC_STATUS}${NC}"
echo -e "${BOLD}Anti-Loop Samplers:${NC}  ${GREEN}DRY (mult 0.8, base 1.75, len 2) + Repeat Penalty 1.1 + Temp 0.7${NC}"
echo -e "${BOLD}CPU Acceleration:${NC}    ${GREEN}Intel Core Ultra 7 265K (AVX_VNNI, -t 8)${NC}"
echo -e "
${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------
"

exec "${SERVER_BIN}"     -m "${MODEL_PATH}"     --alias "${ALIAS}"     --host "${HOST}"     --port "${PORT}"     -c "${CTX_SIZE}"     -np 1     -b 2048     -ub 512     -cb     -ctk "${KV_QUANT}"     -ctv "${KV_QUANT}"     -ngl "${GPU_LAYERS}"     -fa auto     -t 8     --temp 0.7     --repeat-penalty 1.1     --dry-multiplier 0.8     --dry-base 1.75     --dry-allowed-length 2     --dry-penalty-last-n 256     "${SPEC_ARGS[@]}"
