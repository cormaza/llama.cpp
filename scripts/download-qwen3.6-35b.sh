#!/usr/bin/env bash
# ==============================================================================
# download-qwen3.6-35b.sh - Download Qwen3.6-35B-A3B-MTP GGUF models (Unsloth)
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
REPO="unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Qwen3.6-35B-A3B-MTP Downloader (MoE A3B / Unsloth)  ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Qwen3.6-35B-A3B quantization variant:"
    echo -e "  ${GREEN}[1] Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf (11.01 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Max context & speed in 16GB VRAM)"
    echo -e "  ${GREEN}[2] Qwen3.6-35B-A3B-UD-IQ2_M.gguf   (11.07 GB)${NC} - Balanced 2-bit i-matrix precision"
    echo -e "  ${GREEN}[3] Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf (11.71 GB)${NC} - High 2-bit precision"
    echo -e "  ${GREEN}[4] Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf (13.10 GB)${NC} - 3-bit precision (~2.9 GB free VRAM)"
    echo -e "  ${GREEN}[5] mmproj-BF16.gguf                (0.84 GB)${NC} - Vision Projector (for multimodal tasks)"
    echo -e "  ${GREEN}[6] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-6] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|iq2_xxs|iq2_xss|IQ2_XXS|IQ2_XSS)
        FILE="Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"
        ;;
    2|iq2_m|IQ2_M)
        FILE="Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
        ;;
    3|q2_k_xl|Q2_K_XL)
        FILE="Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"
        ;;
    4|iq3_xxs|IQ3_XXS)
        FILE="Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
        ;;
    5|mmproj|vision)
        FILE="mmproj-BF16.gguf"
        ;;
    6)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [choice|quantization|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf (Default)"
        echo -e "  $(basename "$0") iq2_xxs    # By quantization alias"
        echo -e "  $(basename "$0") mmproj     # Download mmproj-BF16.gguf"
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
echo -e "\nTo launch Qwen3.6-35B-A3B with MTP speculative acceleration:"
echo -e "  ${YELLOW}./start-qwen3.6-35b.sh -m ${TARGET_PATH}${NC}\n"
