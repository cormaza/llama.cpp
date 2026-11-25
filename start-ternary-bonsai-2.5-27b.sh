#!/usr/bin/env bash
# ==============================================================================
# start-ternary-bonsai-2.5-27b.sh - Launcher for Ternary-Bonsai-2.5-27B (Qwen3.8-27B)
# Model:    https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF
# Hardware: AMD Radeon RX 9060 XT (16GB VRAM, ROCm / HIP)
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
MODELS_DIR="${SCRIPT_DIR}/models/Ternary-Bonsai-2.5-27B-GGUF"

MODEL_PATH=""
MMPROJ_PATH=""
ENABLE_MMPROJ=1
ENABLE_MTP=1
DRAFT_N_MAX=3
REASONING_EFFORT="xhigh"

HOST="0.0.0.0"
PORT=8080
SLOTS=1
CTX_SIZE=32768
TEMP="1.0"
TOP_P="0.95"
TOP_K="20"
MIN_P="0.0"
KV_QUANT="q8_0"
GPU_LAYERS=99
THREADS=8
ALIAS="ternary-bonsai-2.5-27b,bonsai-2.5-27b,bonsai"

LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Recommended defaults (16GB VRAM + prism-ml/Ternary-Bonsai-2.5-27B-GGUF Q2_0):
  --jinja, --temp 1.0, --top-p 0.95, --top-k 20, --min-p 0.0,
  --chat-template-kwargs '{"reasoning_effort":"xhigh"}', -c 32768, -np 1, -ngl 99, -fa on,
  -ctk q8_0 -ctv q8_0, and draft-mtp when an -mtp.gguf model is used.

Options:
  -m, --model PATH       Path to GGUF file (auto-detected if omitted)
  --mmproj PATH          Path to mmproj GGUF (auto-detected if omitted)
  --no-mmproj            Disable vision projector (text-only mode)
  --no-mtp               Disable MTP speculative decoding even on -mtp.gguf
  --draft-n N            MTP max draft tokens (default: ${DRAFT_N_MAX})
  -c, --ctx N            Context size (default: ${CTX_SIZE})
  -np, --slots N         Parallel slots (default: ${SLOTS})
  -ngl, --gpu-layers N   GPU layers to offload (default: ${GPU_LAYERS})
  --effort LEVEL         reasoning_effort: none|minimal|low|medium|high|xhigh (default: ${REASONING_EFFORT})
  --kv-quant TYPE        KV cache quant: f16|q8_0|q4_0 (default: ${KV_QUANT})
  -t, --threads N        CPU threads (default: ${THREADS})
  --port PORT            Server port (default: ${PORT})
  --host HOST            Host to bind (default: ${HOST})
  -h, --help             Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL_PATH="$2"; shift 2 ;;
        --mmproj) MMPROJ_PATH="$2"; shift 2 ;;
        --no-mmproj) ENABLE_MMPROJ=0; shift ;;
        --no-mtp) ENABLE_MTP=0; shift ;;
        --draft-n) DRAFT_N_MAX="$2"; shift 2 ;;
        -c|--ctx) CTX_SIZE="$2"; shift 2 ;;
        -np|--slots) SLOTS="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --effort) REASONING_EFFORT="$2"; shift 2 ;;
        --kv-quant) KV_QUANT="$2"; shift 2 ;;
        -t|--threads) THREADS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        -h|--help) show_help ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

if [[ ! -x "${SERVER_BIN}" ]]; then
    echo -e "${RED}[ERROR] llama-server binary not found at ${SERVER_BIN}${NC}"
    exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
    for candidate_name in \
        Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf \
        Ternary-Bonsai-2.5-27B-Q2_0.gguf; do
        candidate="${MODELS_DIR}/${candidate_name}"
        if [[ -f "${candidate}" ]]; then
            MODEL_PATH="${candidate}"
            break
        fi
    done
fi

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
    echo -e "${RED}[ERROR] No upstream-compatible Ternary-Bonsai-2.5-27B Q2_0 GGUF found in ${MODELS_DIR}${NC}"
    echo -e "Download first with: ${YELLOW}./scripts/download-ternary-bonsai-2.5-27b.sh${NC}"
    exit 1
fi

MTP_ARGS=()
MTP_STATUS="Disabled"
if [[ "${ENABLE_MTP}" -eq 1 && "${MODEL_PATH}" == *"-mtp.gguf" ]]; then
    MTP_ARGS=(--spec-type draft-mtp --spec-draft-n-min 0 --spec-draft-n-max "${DRAFT_N_MAX}" --spec-draft-p-min 0)
    MTP_STATUS="Enabled (draft-mtp, n_max=${DRAFT_N_MAX})"
fi

MMPROJ_ARGS=()
MMPROJ_STATUS="Disabled"
if [[ "${ENABLE_MMPROJ}" -eq 1 ]]; then
    if [[ -z "${MMPROJ_PATH}" ]]; then
        for f in Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf Ternary-Bonsai-2.5-27B-mmproj-BF16.gguf; do
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

echo -e "${BOLD}${CYAN}=== Ternary-Bonsai-2.5-27B (Qwen3.8-27B) Server ===${NC}"
echo -e "${BOLD}Model:${NC}            ${CYAN}${MODEL_PATH}${NC}"
echo -e "${BOLD}MTP Speculative:${NC}  ${GREEN}${MTP_STATUS}${NC}"
echo -e "${BOLD}Vision Projector:${NC} ${GREEN}${MMPROJ_STATUS}${NC}"
echo -e "${BOLD}Context:${NC}          ${GREEN}${CTX_SIZE} (slots=${SLOTS}, KV=${KV_QUANT})${NC}"
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
    "${MTP_ARGS[@]}" \
    "${MMPROJ_ARGS[@]}"
