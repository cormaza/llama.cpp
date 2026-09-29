#!/usr/bin/env bash
# ==============================================================================
# download-gemma4-coder.sh - Download Gemma4-12B-Coder (Composer 2.5 x Fable 5) GGUF
# Model: https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF
# Architecture: Google Gemma 4 12B + Composer 2.5 & Fable 5 CoT Distillation
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
REPO="yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Gemma4-12B-Coder (Composer 2.5 x Fable 5) Downloader ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Gemma4-12B-Coder variant to download:"
    echo -e "  ${GREEN}[1] gemma4-coding-Q4_K_M.gguf (6.87 GB)${NC} - ${BOLD}RECOMMENDED${NC} (~9.1 GB free VRAM, sweet spot)"
    echo -e "  ${GREEN}[2] gemma4-coding-Q6_K.gguf   (9.11 GB)${NC} - Near-lossless precision (~6.9 GB free VRAM)"
    echo -e "  ${GREEN}[3] gemma4-coding-Q8_0.gguf   (11.8 GB)${NC} - Full reference quality (~4.2 GB free VRAM)"
    echo -e "  ${GREEN}[4] gemma4-coding-Q3_K_M.gguf (5.70 GB)${NC} - Compact 3-bit quant (~10.3 GB free VRAM)"
    echo -e "  ${GREEN}[5] gemma4-coding-Q2_K.gguf   (4.50 GB)${NC} - Minimal footprint (~11.5 GB free VRAM)"
    echo -e "  ${GREEN}[6] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-6] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|q4_k_m|Q4_K_M|q4|Q4)
        FILE="gemma4-coding-Q4_K_M.gguf"
        ;;
    2|q6_k|Q6_K|q6|Q6)
        FILE="gemma4-coding-Q6_K.gguf"
        ;;
    3|q8_0|Q8_0|q8|Q8)
        FILE="gemma4-coding-Q8_0.gguf"
        ;;
    4|q3_k_m|Q3_K_M|q3|Q3)
        FILE="gemma4-coding-Q3_K_M.gguf"
        ;;
    5|q2_k|Q2_K|q2|Q2)
        FILE="gemma4-coding-Q2_K.gguf"
        ;;
    6)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|5|q4|q6|q8|q3|q2|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Q4_K_M (Default, recommended)"
        echo -e "  $(basename "$0") q6_k       # Near-lossless 6-bit"
        echo -e "  $(basename "$0") q8_0       # Reference 8-bit"
        echo -e "  $(basename "$0") q3_k_m     # Compact 3-bit"
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
echo -e "Launch with: ${CYAN}./start-gemma4-coder.sh${NC}"
