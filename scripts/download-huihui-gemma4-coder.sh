#!/usr/bin/env bash
# ==============================================================================
# download-huihui-gemma4-coder.sh - Download Huihui-Gemma4-12B-Coder-Abliterated GGUF
# Model: https://huggingface.co/mradermacher/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated-i1-GGUF
# Base: huihui-ai/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated
# Quantizer: mradermacher (imatrix i1 quants)
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
REPO="mradermacher/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated-i1-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}===================================================================${NC}"
echo -e "${BOLD}${CYAN}  Huihui-Gemma4-12B-Coder Abliterated (imatrix i1) Downloader      ${NC}"
echo -e "${BOLD}${CYAN}===================================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Huihui-Gemma4-12B-Coder Abliterated variant to download:"
    echo -e "  ${GREEN}[1] Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q4_K_M.gguf (7.50 GB)${NC} - ${BOLD}RECOMMENDED${NC} (~8.5 GB free VRAM)"
    echo -e "  ${GREEN}[2] Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q5_K_M.gguf (8.60 GB)${NC} - High precision 5-bit (~7.4 GB free VRAM)"
    echo -e "  ${GREEN}[3] Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q6_K.gguf   (9.90 GB)${NC} - Near-lossless 6-bit (~6.1 GB free VRAM)"
    echo -e "  ${GREEN}[4] Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-IQ4_XS.gguf (6.70 GB)${NC} - Efficient 4-bit imatrix (~9.3 GB free VRAM)"
    echo -e "  ${GREEN}[5] Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q3_K_M.gguf (6.20 GB)${NC} - Compact 3-bit (~9.8 GB free VRAM)"
    echo -e "  ${GREEN}[6] Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-IQ3_M.gguf  (5.80 GB)${NC} - Balanced 3-bit imatrix (~10.2 GB free VRAM)"
    echo -e "  ${GREEN}[7] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-7] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

case "${CHOICE}" in
    1|q4_k_m|Q4_K_M|q4|Q4)
        FILE="Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q4_K_M.gguf"
        ;;
    2|q5_k_m|Q5_K_M|q5|Q5)
        FILE="Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q5_K_M.gguf"
        ;;
    3|q6_k|Q6_K|q6|Q6)
        FILE="Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q6_K.gguf"
        ;;
    4|iq4_xs|IQ4_XS|iq4|IQ4)
        FILE="Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-IQ4_XS.gguf"
        ;;
    5|q3_k_m|Q3_K_M|q3|Q3)
        FILE="Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-Q3_K_M.gguf"
        ;;
    6|iq3_m|IQ3_M|iq3|IQ3)
        FILE="Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.i1-IQ3_M.gguf"
        ;;
    7)
        read -rp "Enter exact GGUF filename in ${REPO}: " FILE
        ;;
    *.gguf)
        FILE="${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|5|6|q4|q5|q6|iq4|q3|iq3|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Q4_K_M (Default, recommended)"
        echo -e "  $(basename "$0") q5_k_m     # High precision 5-bit"
        echo -e "  $(basename "$0") q6_k       # Near-lossless 6-bit"
        echo -e "  $(basename "$0") iq4_xs     # Efficient 4-bit imatrix"
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
echo -e "Launch with: ${CYAN}./start-huihui-gemma4-coder.sh${NC}"
