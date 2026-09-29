#!/usr/bin/env bash
# ==============================================================================
# download-huihui-ornith-9b.sh - Download Huihui-Ornith-1.5-9B-Abliterated GGUF
# Model: https://huggingface.co/mradermacher/Huihui-Ornith-1.5-9B-abliterated-GGUF
# Base: huihui-ai/Huihui-Ornith-1.5-9B-abliterated (Uncensored Ornith-1.5-9B)
# Quantizer: mradermacher
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
REPO="mradermacher/Huihui-Ornith-1.5-9B-abliterated-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}===================================================================${NC}"
echo -e "${BOLD}${CYAN}  Huihui-Ornith-1.5-9B-Abliterated GGUF Downloader                 ${NC}"
echo -e "${BOLD}${CYAN}===================================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Huihui-Ornith-1.5-9B-Abliterated variant to download:"
    echo -e "  ${GREEN}[1] Huihui-Ornith-1.5-9B-abliterated.Q4_K_M.gguf       (5.7 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Fits 128k context 100% in 16GB VRAM)"
    echo -e "  ${GREEN}[2] Huihui-Ornith-1.5-9B-abliterated.Q5_K_M.gguf       (6.6 GB)${NC} - High precision 5-bit"
    echo -e "  ${GREEN}[3] Huihui-Ornith-1.5-9B-abliterated.Q6_K.gguf         (7.5 GB)${NC} - Near-lossless 6-bit"
    echo -e "  ${GREEN}[4] Huihui-Ornith-1.5-9B-abliterated.Q8_0.gguf         (9.6 GB)${NC} - Reference quality 8-bit"
    echo -e "  ${GREEN}[5] Huihui-Ornith-1.5-9B-abliterated.IQ4_XS.gguf       (5.3 GB)${NC} - Efficient 4-bit / lowest VRAM"
    echo -e "  ${GREEN}[6] Huihui-Ornith-1.5-9B-abliterated.Q3_K_M.gguf       (4.7 GB)${NC} - Compact 3-bit"
    echo -e "  ${GREEN}[7] Huihui-Ornith-1.5-9B-abliterated.mmproj-Q8_0.gguf   (0.7 GB)${NC} - Multimodal Vision Projector"
    echo -e "  ${GREEN}[8] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-8] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|q4_k_m|Q4_K_M|q4|Q4)
        FILE="Huihui-Ornith-1.5-9B-abliterated.Q4_K_M.gguf"
        ;;
    2|q5_k_m|Q5_K_M|q5|Q5)
        FILE="Huihui-Ornith-1.5-9B-abliterated.Q5_K_M.gguf"
        ;;
    3|q6_k|Q6_K|q6|Q6)
        FILE="Huihui-Ornith-1.5-9B-abliterated.Q6_K.gguf"
        ;;
    4|q8_0|Q8_0|q8|Q8)
        FILE="Huihui-Ornith-1.5-9B-abliterated.Q8_0.gguf"
        ;;
    5|iq4_xs|IQ4_XS|iq4|IQ4)
        FILE="Huihui-Ornith-1.5-9B-abliterated.IQ4_XS.gguf"
        ;;
    6|q3_k_m|Q3_K_M|q3|Q3)
        FILE="Huihui-Ornith-1.5-9B-abliterated.Q3_K_M.gguf"
        ;;
    7|mmproj|vision)
        FILE="Huihui-Ornith-1.5-9B-abliterated.mmproj-Q8_0.gguf"
        ;;
    8)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|5|6|7|q4|q5|q6|q8|iq4|q3|mmproj|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Q4_K_M (Default, recommended)"
        echo -e "  $(basename "$0") q5_k_m     # High precision 5-bit"
        echo -e "  $(basename "$0") q6_k       # Near-lossless 6-bit"
        echo -e "  $(basename "$0") q8_0       # Reference quality 8-bit"
        echo -e "  $(basename "$0") mmproj     # Download multimodal vision projector"
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
echo -e "Launch with: ${CYAN}./start-huihui-ornith-9b.sh${NC}"
