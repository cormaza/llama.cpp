#!/usr/bin/env bash
# ==============================================================================
# download-qwen3.5-c3sm-9b.sh - Download Qwen3.5-9B-C3SM-SDM-Agentic-Coder GGUF
# Model: https://huggingface.co/OliviaRossi/Qwen3.5-9B-C3SM-SDM-Agentic-Coder-GGUF
# Architecture: Qwen 3.5 9B Hybrid Linear Attention + C3SM-SDM Consensus Merging
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
REPO="OliviaRossi/Qwen3.5-9B-C3SM-SDM-Agentic-Coder-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Qwen3.5-9B-C3SM-SDM-Agentic-Coder GGUF Downloader   ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Qwen3.5-9B-C3SM-SDM-Agentic-Coder variant to download:"
    echo -e "  ${GREEN}[1] Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q4_K_M.gguf (5.24 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Best balance, ~10.7 GB free VRAM)"
    echo -e "  ${GREEN}[2] Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q5_K_M.gguf (6.02 GB)${NC} - High precision 5-bit (~9.9 GB free VRAM)"
    echo -e "  ${GREEN}[3] Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q6_K.gguf   (6.85 GB)${NC} - Very high quality 6-bit (~9.1 GB free VRAM)"
    echo -e "  ${GREEN}[4] Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q8_0.gguf   (8.87 GB)${NC} - Near-lossless reference quality (~7.1 GB free VRAM)"
    echo -e "  ${GREEN}[5] Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q3_K_L.gguf (4.59 GB)${NC} - Compact 3-bit quant (~11.4 GB free VRAM)"
    echo -e "  ${GREEN}[6] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-6] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|q4_k_m|Q4_K_M|q4|Q4)
        FILE="Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q4_K_M.gguf"
        ;;
    2|q5_k_m|Q5_K_M|q5|Q5)
        FILE="Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q5_K_M.gguf"
        ;;
    3|q6_k|Q6_K|q6|Q6)
        FILE="Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q6_K.gguf"
        ;;
    4|q8_0|Q8_0|q8|Q8)
        FILE="Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q8_0.gguf"
        ;;
    5|q3_k_l|Q3_K_L|q3|Q3)
        FILE="Qwen3.5-9B-C3SM-SDM-Agentic-Coder-Q3_K_L.gguf"
        ;;
    6)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|5|q4|q5|q6|q8|q3|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Q4_K_M (Default, recommended)"
        echo -e "  $(basename "$0") q5_k_m     # High precision 5-bit"
        echo -e "  $(basename "$0") q8_0       # Near-lossless 8-bit"
        echo -e "  $(basename "$0") q3_k_l     # Compact 3-bit"
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
echo -e "Launch with: ${CYAN}./start-qwen3.5-c3sm-9b.sh${NC}"
