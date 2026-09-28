#!/usr/bin/env bash
# ==============================================================================
# download-lfm2.5-8b.sh - Download LFM2.5-8B-A1B-Hermes-Agentic-Coder GGUF models
# Model: https://huggingface.co/DuoNeural/LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-GGUF
# Architecture: Liquid Foundation Model MoE (8.3B Total / 1.5B Active per Token)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
REPO="DuoNeural/LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  LFM2.5-8B-A1B Hermes Agentic Coder GGUF Downloader  ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect LFM2.5-8B-A1B variant to download:"
    echo -e "  ${GREEN}[1] LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q4_K_M.gguf (4.80 GB)${NC} - ${BOLD}RECOMMENDED${NC} (~11.2 GB free VRAM)"
    echo -e "  ${GREEN}[2] LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q5_K_M.gguf (5.62 GB)${NC} - High precision 5-bit (~10.4 GB free VRAM)"
    echo -e "  ${GREEN}[3] LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q6_K.gguf   (6.48 GB)${NC} - Very high quality 6-bit (~9.5 GB free VRAM)"
    echo -e "  ${GREEN}[4] LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q8_0.gguf   (8.39 GB)${NC} - Near-lossless reference quality (~7.6 GB free VRAM)"
    echo -e "  ${GREEN}[5] LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-BF16.gguf   (15.78 GB)${NC} - Full unquantized 16-bit"
    echo -e "  ${GREEN}[6] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-6] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|q4_k_m|Q4_K_M|q4|Q4)
        FILE="LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q4_K_M.gguf"
        ;;
    2|q5_k_m|Q5_K_M|q5|Q5)
        FILE="LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q5_K_M.gguf"
        ;;
    3|q6_k|Q6_K|q6|Q6)
        FILE="LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q6_K.gguf"
        ;;
    4|q8_0|Q8_0|q8|Q8)
        FILE="LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-Q8_0.gguf"
        ;;
    5|bf16|BF16|f16|F16)
        FILE="LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3-BF16.gguf"
        ;;
    6)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|5|q4|q5|q6|q8|bf16|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Q4_K_M (Default, recommended)"
        echo -e "  $(basename "$0") q5_k_m     # High precision 5-bit"
        echo -e "  $(basename "$0") q8_0       # Near-lossless 8-bit"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

TARGET_PATH="${MODELS_DIR}/${FILE}"
DOWNLOAD_URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

if [[ -f "${TARGET_PATH}" ]]; then
    echo -e "${YELLOW}[SKIP] File already exists:${NC} ${TARGET_PATH}"
    exit 0
fi

echo -e "\n${BOLD}Downloading:${NC} ${FILE}"
echo -e "${BOLD}Source Repo:${NC} ${REPO}"
echo -e "${BOLD}Destination:${NC} ${TARGET_PATH}"
echo -e "${BOLD}URL:${NC}         ${DOWNLOAD_URL}\n"

curl -L -C - "${DOWNLOAD_URL}" -o "${TARGET_PATH}" --progress-bar

echo -e "\n${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "Saved to: ${CYAN}${TARGET_PATH}${NC}"
echo -e "Launch with: ${CYAN}./start-lfm2.5-8b.sh${NC}"
