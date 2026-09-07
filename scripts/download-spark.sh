#!/usr/bin/env bash
# ==============================================================================
# download-spark.sh - Download Spark-X2.5-4B GGUF models for agentic workflows
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
DEFAULT_REPO="sizzlebop/Spark-X2.5-4B-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Spark-X2.5-4B Downloader (Agentic & Fast Reasoning) ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

echo -e "
Select quantization variant:"
echo -e "  [1] Spark-X2.5-4B-Q4_K_M.gguf (2.42 GB) - RECOMMENDED (Fastest, max agent slots & context)"
echo -e "  [2] Spark-X2.5-4B-Q5_K_M.gguf (2.77 GB) - High quality / speed balance"
echo -e "  [3] Spark-X2.5-4B-Q6_K.gguf   (3.15 GB) - Near-lossless precision"
echo -e "  [4] Spark-X2.5-4B-Q8_0.gguf   (4.07 GB) - Maximum 8-bit precision"
echo -e "  [5] Custom GGUF file from Hugging Face"

read -rp "Enter choice [1-5] (default 1): " CHOICE
CHOICE="${CHOICE:-1}"

REPO="${DEFAULT_REPO}"

case "${CHOICE}" in
    1)
        FILE="Spark-X2.5-4B-Q4_K_M.gguf"
        ;;
    2)
        FILE="Spark-X2.5-4B-Q5_K_M.gguf"
        ;;
    3)
        FILE="Spark-X2.5-4B-Q6_K.gguf"
        ;;
    4)
        FILE="Spark-X2.5-4B-Q8_0.gguf"
        ;;
    5)
        read -rp "Enter Hugging Face repo (e.g. sizzlebop/Spark-X2.5-4B-GGUF): " REPO
        read -rp "Enter exact GGUF filename: " FILE
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac

TARGET_PATH="${MODELS_DIR}/${FILE}"
DOWNLOAD_URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

echo -e "
${BOLD}Downloading:${NC} ${FILE}"
echo -e "${BOLD}Destination:${NC} ${TARGET_PATH}"
echo -e "${BOLD}URL:${NC}         ${DOWNLOAD_URL}
"

curl -L -C - "${DOWNLOAD_URL}" -o "${TARGET_PATH}" --progress-bar

echo -e "
${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "Model saved at: ${CYAN}${TARGET_PATH}${NC}"
echo -e "
To serve Spark-X2.5 with parallel agent slots:"
echo -e "  ${YELLOW}./start-spark.sh${NC}"
