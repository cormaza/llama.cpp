#!/usr/bin/env bash
# ==============================================================================
# start-inkling-2-small.sh - Launcher for Inkling-2-Small (594B-A17B MoE VLM)
# Model:    https://huggingface.co/unsloth/Inkling-2-Small-GGUF
# Base:     https://huggingface.co/thinkingmachines/Inkling-2-Small
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP) + 30GB RAM
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/build-amd/bin"
SERVER_BIN="${SERVER_BIN:-${BIN_DIR}/llama-server}"
MODELS_DIR="${SCRIPT_DIR}/models/Inkling-2-Small-GGUF"

MODEL_PATH=""
MMPROJ_PATH=""
ENABLE_MMPROJ=1
REASONING_EFFORT="high"

HOST="0.0.0.0"
PORT=8080
SLOTS=1
CTX_SIZE=32768
TEMP="1.0"
TOP_P="1.0"
TOP_K="0"
MIN_P="0.0"
KV_QUANT="f16"
GPU_LAYERS=0
THREADS=16
ALIAS="inkling-2-small,inkling2"

LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Recommended defaults from unsloth/Inkling-2-Small-GGUF:
  --jinja, --temp 1.0, --top-p 1.0, --top-k 0, --min-p 0.0,
  --chat-template-kwargs '{"reasoning_effort":"high"}', -c 32768

Options:
  -m, --model PATH       Path to first GGUF shard (auto-detected if omitted)
  --mmproj PATH          Path to mmproj GGUF (auto-detected if omitted)
  --no-mmproj            Disable vision projector (text-only mode)
  -c, --ctx N            Context size (default: ${CTX_SIZE})
  -ngl, --gpu-layers N   GPU layers to offload (default: ${GPU_LAYERS} for 16GB VRAM; use 99 if it fits)
  --effort LEVEL         reasoning_effort: none|minimal|low|medium|high|xhigh|max (default: ${REASONING_EFFORT})
  --kv-quant TYPE        KV cache quant: f16|q8_0|q4_0 (default: ${KV_QUANT})
  -t, --threads N        CPU threads (default: ${THREADS})
  --port PORT            Server port (default: ${PORT})
  -h, --help             Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL_PATH="$2"; shift 2 ;;
        --mmproj) MMPROJ_PATH="$2"; shift 2 ;;
        --no-mmproj) ENABLE_MMPROJ=0; shift ;;
        -c|--ctx) CTX_SIZE="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --effort) REASONING_EFFORT="$2"; shift 2 ;;
        --kv-quant) KV_QUANT="$2"; shift 2 ;;
        -t|--threads) THREADS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help) show_help ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] llama-server binary not found at ${SERVER_BIN}${NC}"
    exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
    for q in UD-IQ1_S UD-IQ1_M UD-IQ2_XXS UD-Q2_K_XL UD-IQ3_XXS UD-Q3_K_XL UD-Q4_K_XL Q8_0 UD-Q8_K_XL; do
        candidate="$(find "${MODELS_DIR}/${q}" -maxdepth 1 -name '*-00001-of-*.gguf' 2>/dev/null | head -n1 || true)"
        if [[ -n "${candidate}" ]]; then
            MODEL_PATH="${candidate}"
            break
        fi
    done
fi

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
    echo -e "${RED}[ERROR] No Inkling-2-Small GGUF found in ${MODELS_DIR}${NC}"
    echo -e "Download first with: ${YELLOW}./scripts/download-inkling-2-small.sh${NC}"
    exit 1
fi

MMPROJ_ARGS=()
MMPROJ_STATUS="Disabled"
if [[ "${ENABLE_MMPROJ}" -eq 1 ]]; then
    if [[ -z "${MMPROJ_PATH}" ]]; then
        for f in mmproj-F16.gguf mmproj-BF16.gguf mmproj-F32.gguf; do
            if [[ -f "${MODELS_DIR}/${f}" ]]; then
                MMPROJ_PATH="${MODELS_DIR}/${f}"
                break
            fi
        done
    fi
    if [[ -n "${MMPROJ_PATH}" && -f "${MMPROJ_PATH}" ]]; then
        MMPROJ_ARGS=("--mmproj" "${MMPROJ_PATH}")
        MMPROJ_STATUS="${MMPROJ_PATH}"
    else
        echo -e "${YELLOW}[WARN] mmproj not found; running text-only.${NC}"
    fi
fi

echo -e "${BOLD}${CYAN}=== Inkling-2-Small Server ===${NC}"
echo -e "${BOLD}Model:${NC}            ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}Vision Projector:${NC} ${GREEN}${MMPROJ_STATUS}${NC}"
echo -e "${BOLD}Context:${NC}          ${GREEN}${CTX_SIZE}${NC}"
echo -e "${BOLD}GPU Layers (-ngl):${NC}${GREEN}${GPU_LAYERS}${NC}"
echo -e "${BOLD}Reasoning Effort:${NC} ${GREEN}${REASONING_EFFORT}${NC}"
echo -e "${BOLD}Sampling:${NC}         ${GREEN}temp=${TEMP} top_p=${TOP_P} top_k=${TOP_K} min_p=${MIN_P}${NC}"
echo -e "${BOLD}Web UI / API:${NC}     ${CYAN}http://${LOCAL_IP}:${PORT}${NC}\n"

exec "${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${CTX_SIZE}" \
    -np "${SLOTS}" \
    -ctk "${KV_QUANT}" \
    -ctv "${KV_QUANT}" \
    -ngl "${GPU_LAYERS}" \
    -fa on \
    -t "${THREADS}" \
    --jinja \
    --temp "${TEMP}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --min-p "${MIN_P}" \
    --chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}" \
    "${MMPROJ_ARGS[@]}"
