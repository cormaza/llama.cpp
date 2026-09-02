#!/usr/bin/env bash
# ==============================================================================
# start-qwen-max-context.sh - Single-Thread High-Precision Max-Context Server
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

DEFAULT_MODEL="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q4_K_M.gguf"
ALT_MODEL="${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"
MODEL_PATH=""
HOST="0.0.0.0"
PORT=8080
CTX_SIZE=262144 # 256k context
KV_QUANT="q4_0"
GPU_LAYERS=46   # Optimized default: 46 layers in GPU VRAM
ENABLE_SPEC=1   # N-Gram Speculative Decoding enabled by default

# Detect Primary LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Starts llama-server for Qwen optimized for Single-Thread Deep Reasoning with MAXIMUM context,
optimized GPU layer offloading (46 layers) and N-Gram speculative acceleration (0 VRAM cost).

Options:
  -m, --model PATH        Path to GGUF model
  -c, --context N         Maximum context window (default: 262144 / 256k tokens)
  -p, --port PORT         HTTP server port (default: 8080)
  --kv-quant TYPE         KV Cache precision: q8_0 (high precision) | q4_0 (default) | f16
  --ngl N                 Number of layers to offload to GPU (default: 46)
  --no-spec               Disable N-Gram speculative decoding
  -h, --help              Show this help message

Examples:
  ./start-qwen-max-context.sh
  ./start-qwen-max-context.sh -c 131072 --kv-quant q8_0
  ./start-qwen-max-context.sh --ngl 48
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
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
            GPU_LAYERS="$2"
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
echo -e "${BOLD}${CYAN}  Qwen Single-Thread Max-Context & Speed Optimized     ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

# 1. Check binaries
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${SERVER_BIN}${NC}"
    echo -e "Please build first: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

# 2. Select Model
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -f "${ALT_MODEL}" ]]; then
        MODEL_PATH="${ALT_MODEL}"
    elif [[ -f "${DEFAULT_MODEL}" ]]; then
        MODEL_PATH="${DEFAULT_MODEL}"
    else
        FOUND_MODELS=($(find "${SCRIPT_DIR}/models" -maxdepth 1 -name "*.gguf" 2>/dev/null || true))
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

# Adjust KV cache quant if using 128k and user didn't explicitly override
if [[ "${CTX_SIZE}" -le 131072 && "${KV_QUANT}" == "q4_0" ]]; then
    KV_QUANT="q8_0"
fi

SPEC_STATUS="Disabled"
SPEC_ARGS=()
if [[ "${ENABLE_SPEC}" -eq 1 ]]; then
    SPEC_STATUS="Active (N-Gram Prompt Lookup, m=48)"
    SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
fi

echo -e "${BOLD}Model:${NC}               ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}Mode:${NC}                ${GREEN}Single Slot (1 Dedicated Agent / Full Power)${NC}"
echo -e "${BOLD}Context Size:${NC}        ${GREEN}${CTX_SIZE} tokens ($(( CTX_SIZE / 1024 ))k)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}${GPU_LAYERS} layers to AMD Radeon RX 9060 XT (-ngl ${GPU_LAYERS} -fa auto)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${SPEC_STATUS}${NC}"
echo -e "${BOLD}CPU Acceleration:${NC}    ${GREEN}Intel Core Ultra 7 265K (AVX_VNNI, -t 8)${NC}"
echo -e "
${BOLD}${YELLOW}=== Remote Connection Info (From another machine) ===${NC}"
echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
echo -e "  API Key:           ${CYAN}sk-no-key-required${NC}"
echo -e "------------------------------------------------------
"

exec "${SERVER_BIN}"     -m "${MODEL_PATH}"     --host "${HOST}"     --port "${PORT}"     -c "${CTX_SIZE}"     -np 1     -b 2048     -ub 512     -cb     -ctk "${KV_QUANT}"     -ctv "${KV_QUANT}"     -ngl "${GPU_LAYERS}"     -fa auto     -t 8     "${SPEC_ARGS[@]}"
