#!/usr/bin/env bash
# ==============================================================================
# download-k2-horizon.sh - Download K2-Horizon-7B GGUF models
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
REPO="NANI-Nithin/K2-Horizon-7B-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  K2-Horizon-7B Downloader (IFM / NANI-Nithin)        ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect K2-Horizon-7B quantization variant:"
    echo -e "  ${GREEN}[1] K2-Horizon-7B-Q4_K_M.gguf  (5.21 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Best balance, ~10.8 GB free VRAM)"
    echo -e "  ${GREEN}[2] K2-Horizon-7B-Q8_0.gguf    (8.92 GB)${NC} - Near-lossless reference quality"
    echo -e "  ${GREEN}[3] K2-Horizon-7B-Q5_K_M.gguf  (6.02 GB)${NC} - High precision 5-bit quant"
    echo -e "  ${GREEN}[4] K2-Horizon-7B-Q6_K.gguf    (6.89 GB)${NC} - Near-lossless 6-bit quant"
    echo -e "  ${GREEN}[5] K2-Horizon-7B-IQ3_XXS.gguf (3.56 GB)${NC} - Ultra-compact (Ideal for 256k context in 16GB VRAM)"
    echo -e "  ${GREEN}[6] K2-Horizon-7B-IQ2_XXS.gguf (2.71 GB)${NC} - Extreme compression (~13.3 GB free VRAM)"
    echo -e "  ${GREEN}[7] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-7] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|q4_k_m|Q4_K_M|q4|Q4)
        FILE="K2-Horizon-7B-Q4_K_M.gguf"
        ;;
    2|q8_0|Q8_0|q8|Q8)
        FILE="K2-Horizon-7B-Q8_0.gguf"
        ;;
    3|q5_k_m|Q5_K_M|q5|Q5)
        FILE="K2-Horizon-7B-Q5_K_M.gguf"
        ;;
    4|q6_k|Q6_K|q6|Q6)
        FILE="K2-Horizon-7B-Q6_K.gguf"
        ;;
    5|iq3_xxs|IQ3_XXS|iq3|IQ3)
        FILE="K2-Horizon-7B-IQ3_XXS.gguf"
        ;;
    6|iq2_xxs|IQ2_XXS|iq2|IQ2)
        FILE="K2-Horizon-7B-IQ2_XXS.gguf"
        ;;
    7)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [choice|quantization|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download K2-Horizon-7B-Q4_K_M.gguf (Default)"
        echo -e "  $(basename "$0") q4_k_m     # By quantization alias"
        echo -e "  $(basename "$0") iq3_xxs    # Download 3-bit variant for ultra-deep context"
        echo -e "  $(basename "$0") q8_0       # Download near-lossless 8-bit"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

TARGET_PATH="${MODELS_DIR}/${FILE}"
DOWNLOAD_URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

echo -e "\n${BOLD}Downloading:${NC} ${FILE}"
echo -e "${BOLD}Destination:${NC} ${TARGET_PATH}"
echo -e "${BOLD}URL:${NC}         ${DOWNLOAD_URL}\n"

curl -L -C - "${DOWNLOAD_URL}" -o "${TARGET_PATH}" --progress-bar

echo -e "\n${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "Model saved at: ${CYAN}${TARGET_PATH}${NC}"
echo -e "\nLaunch commands:"
echo -e "  ${YELLOW}./start-k2-horizon-agent.sh -m ${TARGET_PATH}${NC}       # Multi-agent parallel stack (4-8 slots)"
echo -e "  ${YELLOW}./start-k2-horizon-max-context.sh -m ${TARGET_PATH}${NC} # Max context single agent (up to 512k)"
