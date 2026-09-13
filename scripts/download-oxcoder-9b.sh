#!/usr/bin/env bash
# ==============================================================================
# download-oxcoder-9b.sh - Download OxCoder-9B GGUF models (OrionLLM / Qwen 3.5)
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
REPO="mradermacher/OxCoder-9B-GGUF"
REPO_I1="mradermacher/OxCoder-9B-i1-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  OxCoder-9B Downloader (OrionLLM / Qwen 3.5 Hybrid)  ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect OxCoder-9B quantization variant:"
    echo -e "  ${GREEN}[1] OxCoder-9B.Q4_K_M.gguf      (5.24 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Best balance, ~10.7 GB free VRAM)"
    echo -e "  ${GREEN}[2] OxCoder-9B.Q5_K_M.gguf      (6.02 GB)${NC} - High precision 5-bit (Recommended for strict coding)"
    echo -e "  ${GREEN}[3] OxCoder-9B.Q8_0.gguf        (8.87 GB)${NC} - Near-lossless reference quality (~7.1 GB free VRAM)"
    echo -e "  ${GREEN}[4] OxCoder-9B.IQ4_XS.gguf      (4.87 GB)${NC} - Compact 4-bit quant"
    echo -e "  ${GREEN}[5] OxCoder-9B.i1-IQ3_XXS.gguf  (3.67 GB)${NC} - Ultra-compact 3-bit (i-matrix, ~12.3 GB free VRAM)"
    echo -e "  ${GREEN}[6] OxCoder-9B.mmproj-Q8_0.gguf (0.58 GB)${NC} - Vision Projector (for UI / multimodal image tasks)"
    echo -e "  ${GREEN}[7] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-7] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|q4_k_m|Q4_K_M|q4|Q4)
        FILE="OxCoder-9B.Q4_K_M.gguf"
        SOURCE_REPO="${REPO}"
        ;;
    2|q5_k_m|Q5_K_M|q5|Q5)
        FILE="OxCoder-9B.Q5_K_M.gguf"
        SOURCE_REPO="${REPO}"
        ;;
    3|q8_0|Q8_0|q8|Q8)
        FILE="OxCoder-9B.Q8_0.gguf"
        SOURCE_REPO="${REPO}"
        ;;
    4|iq4_xs|IQ4_XS|iq4|IQ4)
        FILE="OxCoder-9B.IQ4_XS.gguf"
        SOURCE_REPO="${REPO}"
        ;;
    5|iq3_xxs|IQ3_XXS|iq3|IQ3)
        FILE="OxCoder-9B.i1-IQ3_XXS.gguf"
        SOURCE_REPO="${REPO_I1}"
        ;;
    6|mmproj|vision)
        FILE="OxCoder-9B.mmproj-Q8_0.gguf"
        SOURCE_REPO="${REPO}"
        ;;
    7)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        SOURCE_REPO="${REPO}"
        ;;
    *.gguf)
        FILE="${CHOICE}"
        SOURCE_REPO="${REPO}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [choice|quantization|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download OxCoder-9B.Q4_K_M.gguf (Default)"
        echo -e "  $(basename "$0") q5_k_m     # High precision 5-bit"
        echo -e "  $(basename "$0") q8_0       # Near-lossless 8-bit"
        echo -e "  $(basename "$0") iq3_xxs    # Compact 3-bit for maximum context"
        echo -e "  $(basename "$0") mmproj     # Download vision projector"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

TARGET_PATH="${MODELS_DIR}/${FILE}"
DOWNLOAD_URL="https://huggingface.co/${SOURCE_REPO}/resolve/main/${FILE}"

echo -e "\n${BOLD}Downloading:${NC} ${FILE}"
echo -e "${BOLD}Source Repo:${NC} ${SOURCE_REPO}"
echo -e "${BOLD}Destination:${NC} ${TARGET_PATH}"
echo -e "${BOLD}URL:${NC}         ${DOWNLOAD_URL}\n"

curl -L -C - "${DOWNLOAD_URL}" -o "${TARGET_PATH}" --progress-bar

echo -e "\n${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "Model saved at: ${CYAN}${TARGET_PATH}${NC}"
echo -e "\nLaunch commands:"
echo -e "  ${YELLOW}./start-oxcoder-9b-agent.sh -m ${TARGET_PATH}${NC}       # Multi-agent parallel stack (4-8 slots)"
echo -e "  ${YELLOW}./start-oxcoder-9b-max-context.sh -m ${TARGET_PATH}${NC} # Max context single agent (262k native)"
