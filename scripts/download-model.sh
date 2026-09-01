#!/usr/bin/env bash
# ==============================================================================
# download-model.sh - Helper to download Unsloth GGUF models with resume support
# ==============================================================================

set -euo pipefail

RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
BLUE='[0;34m'
CYAN='[0;36m'
BOLD='[1m'
NC='[0m'

MODELS_DIR="./models"
DEFAULT_REPO="unsloth/Qwen3.8-27B-GGUF"
DEFAULT_FILE="Qwen3.8-27B-UD-Q3_K_XL.gguf"

mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Qwen3.8-27B-GGUF Downloader (256k Context Ready)    ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

echo -e "
Select quantization for Qwen3.8-27B:"
echo -e "  [1] UD-Q3_K_XL (12.24 GB) - RECOMMENDED for 256k Context (Fits weights + 256k KV cache in 48GB memory)"
echo -e "  [2] UD-IQ4_XS  (13.27 GB) - Excellent balance of quality and size"
echo -e "  [3] UD-Q4_K_M  (15.33 GB) - Higher precision (Best for shorter/medium context ~64k-128k)"
echo -e "  [4] Custom file name"

read -rp "Enter choice [1-4] (default 1): " CHOICE
CHOICE="${CHOICE:-1}"

case "${CHOICE}" in
    1)
        FILE="Qwen3.8-27B-UD-Q3_K_XL.gguf"
        ;;
    2)
        FILE="Qwen3.8-27B-UD-IQ4_XS.gguf"
        ;;
    3)
        FILE="Qwen3.8-27B-UD-Q4_K_M.gguf"
        ;;
    4)
        read -rp "Enter exact GGUF filename: " FILE
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac

TARGET_PATH="${MODELS_DIR}/${FILE}"
DOWNLOAD_URL="https://huggingface.co/${DEFAULT_REPO}/resolve/main/${FILE}"

echo -e "
${BOLD}Downloading:${NC} ${FILE}"
echo -e "${BOLD}Destination:${NC} ${TARGET_PATH}"
echo -e "${BOLD}URL:${NC} ${DOWNLOAD_URL}
"

curl -L -C - "${DOWNLOAD_URL}" -o "${TARGET_PATH}" --progress-bar

echo -e "
${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "Model saved at: ${CYAN}${TARGET_PATH}${NC}"
echo -e "
To run with 256k context using AMD GPU + CPU Hybrid mode:"
echo -e "  ${YELLOW}./run-amd.sh cli -m ${TARGET_PATH} -c 262144 -ctk q4_0 -ctv q4_0 -ngl 38 -fa auto -p "Tu prompt largo aquí"${NC}"
