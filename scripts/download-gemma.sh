#!/usr/bin/env bash
# ==============================================================================
# download-gemma.sh - Helper to download Gemma 4 GGUF models (Unsloth Dynamic)
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
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Gemma 4 Downloader & 8-Slot Parallel Server Setup   ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

echo -e "
Select Gemma 4 model variant:"
echo -e "  [1] Gemma-4 12B (UD-Q4_K_XL, 6.86 GB)  - RECOMMENDED (Best reasoning, 100% in 16GB VRAM with 8 slots)"
echo -e "  [2] Gemma-4 E4B (UD-Q4_K_XL, 4.77 GB)  - Ultra-fast lightweight agent (Max throughput)"
echo -e "  [3] Gemma-4 26B-A4B MoE (UD-Q3_K_XL, 12.02 GB) - Mixture of Experts (High quality)"
echo -e "  [4] Custom file URL"

read -rp "Enter choice [1-4] (default 1): " CHOICE
CHOICE="${CHOICE:-1}"

case "${CHOICE}" in
    1)
        REPO="unsloth/gemma-4-12b-it-GGUF"
        FILE="gemma-4-12b-it-UD-Q4_K_XL.gguf"
        ;;
    2)
        REPO="unsloth/gemma-4-E4B-it-GGUF"
        FILE="gemma-4-E4B-it-UD-Q4_K_XL.gguf"
        ;;
    3)
        REPO="unsloth/gemma-4-26B-A4B-it-GGUF"
        FILE="gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf"
        ;;
    4)
        read -rp "Enter HF Repo (e.g. unsloth/gemma-4-12b-it-GGUF): " REPO
        read -rp "Enter GGUF Filename: " FILE
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
echo -e "${BOLD}URL:${NC} ${DOWNLOAD_URL}
"

curl -L -C - "${DOWNLOAD_URL}" -o "${TARGET_PATH}" --progress-bar

echo -e "
${GREEN}${BOLD}Gemma 4 model downloaded successfully!${NC}"
echo -e "Model saved at: ${CYAN}${TARGET_PATH}${NC}"
echo -e "
To launch the 8-Slot Parallel Agent Server with Chunked Prefill (Zero-Freeze):"
echo -e "  ${YELLOW}./run-amd.sh server -m ${TARGET_PATH} --host 0.0.0.0 --port 8080 -c 65536 -np 8 -ub 512 -b 2048 -cb -ctk q4_0 -ctv q4_0 -ngl 99 -fa auto${NC}"
