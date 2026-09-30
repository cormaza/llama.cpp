#!/usr/bin/env bash
# ==============================================================================
# download-qwen3.8-9b-heretic.sh - Download Qwen3.8-9B-Distill-Uncensored-Heretic GGUF
# Repo: https://huggingface.co/ge525/Qwen3.8-9B-Distill-uncensored-heretic-Q4_K_M-GGUF
# Base: petruhonk/Qwen3.8-9B-Distill-uncensored-heretic (from empero-ai/Qwen3.8-9B)
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
mkdir -p "${MODELS_DIR}"

DEFAULT_REPO="ge525/Qwen3.8-9B-Distill-uncensored-heretic-Q4_K_M-GGUF"
ALT_REPO="petruhonk/Qwen3.8-9B-Distill-uncensored-heretic-GGUF"

echo -e "${BOLD}${CYAN}===================================================================${NC}"
echo -e "${BOLD}${CYAN}  Qwen3.8-9B-Distill Uncensored Heretic GGUF Downloader            ${NC}"
echo -e "${BOLD}${CYAN}===================================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Qwen3.8-9B-Distill Uncensored Heretic variant to download:"
    echo -e "  ${GREEN}[1] qwen3.8-9b-distill-uncensored-heretic-q4_k_m.gguf (5.78 GB)${NC} - ${BOLD}RECOMMENDED${NC} (100% GPU, ~10 GB free VRAM)"
    echo -e "  ${GREEN}[2] Qwen3.8-9B-Distill-Heretic-Uncensored-Q8_0.gguf   (9.50 GB)${NC} - Reference 8-bit precision (petruhonk)"
    echo -e "  ${GREEN}[3] Custom GGUF filename from ${DEFAULT_REPO}${NC}"

    read -rp "Enter choice [1-3] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

REPO="${DEFAULT_REPO}"

case "${CHOICE}" in
    1|q4|q4_k_m|Q4|Q4_K_M)
        FILE="qwen3.8-9b-distill-uncensored-heretic-q4_k_m.gguf"
        REPO="${DEFAULT_REPO}"
        ;;
    2|q8|q8_0|Q8|Q8_0)
        FILE="Qwen3.8-9B-Distill-Heretic-Uncensored-Q8_0.gguf"
        REPO="${ALT_REPO}"
        ;;
    3)
        read -rp "Enter exact GGUF filename in ${DEFAULT_REPO}: " FILE
        REPO="${DEFAULT_REPO}"
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|q4|q8|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Q4_K_M (Default, recommended)"
        echo -e "  $(basename "$0") q4         # Download Q4_K_M"
        echo -e "  $(basename "$0") q8         # Download Q8_0 reference precision"
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
echo -e "Launch with: ${CYAN}./start-qwen3.8-9b-heretic.sh${NC}"
