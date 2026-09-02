#!/usr/bin/env bash
# ==============================================================================
# download-ornith.sh - Helper to download Ornith-1.5-9B-MTP GGUF models
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

MODELS_DIR="./models"
mkdir -p "${MODELS_DIR}"

REPO="protoLabsAI/Ornith-1.5-9B-MTP-GGUF"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Ornith-1.5-9B-MTP Downloader (Built-in MTP Head)     ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

echo -e "\nSelect Ornith 1.5 9B variant:"
echo -e "  ${GREEN}[1] Ornith-1.5-9B-MTP-Q5_K_M.gguf (6.33 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Balanced precision, fits 128k context 100% in 16GB VRAM)"
echo -e "  ${GREEN}[2] Ornith-1.5-9B-MTP-Q6_K.gguf   (7.21 GB)${NC} - Near-lossless precision (Highest coding fidelity)"
echo -e "  ${GREEN}[3] Ornith-1.5-9B-MTP-IQ4_XS.gguf (5.20 GB)${NC} - Maximum speed & lowest VRAM"
echo -e "  ${GREEN}[4] Ornith-1.5-9B-MTP-Q8_0.gguf   (9.33 GB)${NC} - Reference quality"
echo -e "  ${GREEN}[5] Vision Projector: mmproj-Ornith-1.5-9B-BF16.gguf (879 MB)${NC} - For image input"

read -rp "Enter choice [1-5] (default 1): " CHOICE
CHOICE="${CHOICE:-1}"

case "${CHOICE}" in
    1)
        FILE="Ornith-1.5-9B-MTP-Q5_K_M.gguf"
        ;;
    2)
        FILE="Ornith-1.5-9B-MTP-Q6_K.gguf"
        ;;
    3)
        FILE="Ornith-1.5-9B-MTP-IQ4_XS.gguf"
        ;;
    4)
        FILE="Ornith-1.5-9B-MTP-Q8_0.gguf"
        ;;
    5)
        FILE="mmproj-Ornith-1.5-9B-BF16.gguf"
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac

TARGET_PATH="${MODELS_DIR}/${FILE}"
DOWNLOAD_URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

echo -e "\n${BOLD}Downloading:${NC} ${FILE}"
echo -e "${BOLD}Destination:${NC} ${TARGET_PATH}"
echo -e "${BOLD}URL:${NC} ${DOWNLOAD_URL}\n"

curl -L -C - "${DOWNLOAD_URL}" -o "${TARGET_PATH}" --progress-bar

echo -e "\n${GREEN}${BOLD}Model downloaded successfully!${NC}"
echo -e "Saved at: ${CYAN}${TARGET_PATH}${NC}"
echo -e "\nTo launch Ornith with 128k context and MTP Speculative Acceleration:"
echo -e "  ${YELLOW}./start-ornith.sh -m ${TARGET_PATH}${NC}"
