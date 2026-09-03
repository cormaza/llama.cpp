#!/usr/bin/env bash
# ==============================================================================
# download-model.sh - Helper to download Qwen3.8-27B GGUF models and MTP Draft
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
REPO="unsloth/Qwen3.8-27B-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Qwen3.8-27B Downloader & MTP Speculative Setup      ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

echo -e "
Select Qwen3.8-27B download option:"
echo -e "  [1] Base UD-Q3_K_XL (12.24 GB) + MTP Q4_0 Draft (1.27 GB) - RECOMMENDED (Max Speedup 6.5+ t/s)"
echo -e "  [2] Only MTP Draft Module: mtp-Qwen3.8-27B-Q4_0.gguf (1.27 GB) - Download draft if you already have the base"
echo -e "  [3] Base UD-Q3_K_XL Only (12.24 GB) - For 128k/256k context"
echo -e "  [4] Base UD-IQ4_XS  Only (13.27 GB) - Higher precision"
echo -e "  [5] Base UD-Q4_K_M  Only (15.33 GB) - Maximum precision"
echo -e "  [6] Custom GGUF filename"

read -rp "Enter choice [1-6] (default 1): " CHOICE
CHOICE="${CHOICE:-1}"

download_file() {
    local file="$1"
    local target="${MODELS_DIR}/$(basename "${file}")"
    local url="https://huggingface.co/${REPO}/resolve/main/${file}"

    echo -e "
${BOLD}Downloading:${NC} $(basename "${file}")"
    echo -e "${BOLD}Destination:${NC} ${target}"
    echo -e "${BOLD}URL:${NC} ${url}
"

    curl -L -C - "${url}" -o "${target}" --progress-bar
    echo -e "${GREEN}[OK] Saved to: ${target}${NC}"
}

case "${CHOICE}" in
    1)
        download_file "Qwen3.8-27B-UD-Q3_K_XL.gguf"
        download_file "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
        ;;
    2)
        download_file "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
        ;;
    3)
        download_file "Qwen3.8-27B-UD-Q3_K_XL.gguf"
        ;;
    4)
        download_file "Qwen3.8-27B-UD-IQ4_XS.gguf"
        ;;
    5)
        download_file "Qwen3.8-27B-UD-Q4_K_M.gguf"
        ;;
    6)
        read -rp "Enter exact HF path/filename: " FILE
        download_file "${FILE}"
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac

echo -e "
${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "
To launch Qwen with MTP Speculative Acceleration:"
echo -e "  ${YELLOW}./start-qwen-max-context.sh${NC}"
