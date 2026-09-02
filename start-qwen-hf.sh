#!/usr/bin/env bash
# ==============================================================================
# start-qwen-hf.sh - Run Unsloth Qwen3.8-27B-GGUF:Q4_0 with stack optimizations
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
SERVER_BIN="${BIN_DIR}/llama-server"
CLI_BIN="${BIN_DIR}/llama-cli"

DEFAULT_HF_MODEL="unsloth/Qwen3.8-27B-GGUF:Q4_0"
HF_MODEL="${DEFAULT_HF_MODEL}"
HOST="0.0.0.0"
PORT=8080
CTX_SIZE=131072
KV_QUANT="q4_0"
GPU_LAYERS=50
ENABLE_SPEC=1
RUN_CLI=0

# Detect local LAN IP for remote access
LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n1 || echo "127.0.0.1")"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Runs unsloth/Qwen3.8-27B-GGUF:Q4_0 with full AMD ROCm and Intel CPU optimizations.
Automatically downloads and caches the model from Hugging Face Hub if needed.

Options:
  --hf REPO:QUANT        Hugging Face repo and quant (default: ${DEFAULT_HF_MODEL})
  -c, --context N        Context window size in tokens (default: 131072 / 128k)
  -p, --port PORT        Server port (default: 8080)
  --host HOST            Bind address (default: 0.0.0.0)
  --kv-quant TYPE        KV cache precision: q4_0 (default) | q8_0 | f16
  --ngl N                Layers offloaded to GPU (default: 50)
  --no-spec              Disable N-Gram speculative decoding
  --cli                  Run interactive CLI chat instead of HTTP server
  -h, --help             Show this help message

Examples:
  ./start-qwen-hf.sh
  ./start-qwen-hf.sh -c 65536
  ./start-qwen-hf.sh --cli
  ./start-qwen-hf.sh --hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf)
            HF_MODEL="$2"
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
        --host)
            HOST="$2"
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
        --cli)
            RUN_CLI=1
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
echo -e "${BOLD}${CYAN}  Qwen3.8-27B HF Runner (ROCm / HIP Accelerated)      ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

# Check binaries
TARGET_BIN="${SERVER_BIN}"
if [[ "${RUN_CLI}" -eq 1 ]]; then
    TARGET_BIN="${CLI_BIN}"
fi

if [[ ! -x "${TARGET_BIN}" ]]; then
    echo -e "${RED}[ERROR] Binary not found at: ${TARGET_BIN}${NC}"
    echo -e "Please build first: ${CYAN}./scripts/build-amd-rocm.sh${NC}"
    exit 1
fi

SPEC_ARGS=()
SPEC_STATUS="Disabled"
if [[ "${ENABLE_SPEC}" -eq 1 ]]; then
    SPEC_STATUS="Active (N-Gram Lookup, m=48)"
    SPEC_ARGS+=("--spec-type" "ngram-simple" "--spec-ngram-simple-size-m" "48")
fi

echo -e "${BOLD}HF Model:${NC}            ${CYAN}${HF_MODEL}${NC}"
echo -e "${BOLD}Context Size:${NC}        ${GREEN}${CTX_SIZE} tokens ($(( CTX_SIZE / 1024 ))k)${NC}"
echo -e "${BOLD}KV Cache Precision:${NC}  ${GREEN}${KV_QUANT}${NC}"
echo -e "${BOLD}GPU Offload:${NC}         ${GREEN}${GPU_LAYERS} layers -> AMD Radeon RX 9060 XT (-fa auto)${NC}"
echo -e "${BOLD}Speculative Dec:${NC}     ${GREEN}${SPEC_STATUS}${NC}"
echo -e "${BOLD}CPU Threads:${NC}         ${GREEN}8 P-cores (Intel Core Ultra 7 265K)${NC}"

if [[ "${RUN_CLI}" -eq 1 ]]; then
    echo -e "${BOLD}Mode:${NC}                ${YELLOW}Interactive CLI${NC}"
    echo -e "------------------------------------------------------\n"
    exec "${CLI_BIN}" \
        -hf "${HF_MODEL}" \
        -c "${CTX_SIZE}" \
        -b 2048 \
        -ub 512 \
        -ctk "${KV_QUANT}" \
        -ctv "${KV_QUANT}" \
        -ngl "${GPU_LAYERS}" \
        -fa auto \
        -t 8 \
        "${SPEC_ARGS[@]}" \
        -co -cnv
else
    echo -e "${BOLD}Mode:${NC}                ${GREEN}Server (OpenAI API / Web UI)${NC}"
    echo -e "\n${BOLD}${YELLOW}=== Connection Endpoints ===${NC}"
    echo -e "  Web UI:            ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
    echo -e "  OpenAI API Base:   ${CYAN}http://${LOCAL_IP}:${PORT}/v1${NC}"
    echo -e "  Localhost Base:    ${CYAN}http://127.0.0.1:${PORT}/v1${NC}"
    echo -e "------------------------------------------------------\n"
    exec "${SERVER_BIN}" \
        -hf "${HF_MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        -c "${CTX_SIZE}" \
        -np 1 \
        -b 2048 \
        -ub 512 \
        -cb \
        -ctk "${KV_QUANT}" \
        -ctv "${KV_QUANT}" \
        -ngl "${GPU_LAYERS}" \
        -fa auto \
        -t 8 \
        "${SPEC_ARGS[@]}"
fi
