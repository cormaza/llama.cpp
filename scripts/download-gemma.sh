#!/usr/bin/env bash
# ==============================================================================
# download-gemma.sh - Helper to download Gemma 4 GGUF models and MTP Draft Module
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
echo -e "${BOLD}${CYAN}  Gemma 4 Downloader & MTP Speculative Setup          ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

echo -e "
Select Gemma 4 download option:"
echo -e "  [1] Gemma-4 12B Base + MTP Q8_0 Draft  - RECOMMENDED (Full Model + 444MB MTP for Speculative Speedup)"
echo -e "  [2] Only Gemma-4 12B MTP Draft Module (Q8_0, 444 MB) - Fast download if you already have the base model"
echo -e "  [3] Gemma-4 12B Base Model Only (UD-Q4_K_XL, 6.86 GB)"
echo -e "  [4] Gemma-4 E4B Base Model (UD-Q4_K_XL, 4.77 GB)"
echo -e "  [5] Gemma-4 26B-A4B MoE (UD-Q3_K_XL, 12.02 GB)"
echo -e "  [6] Custom HF URL"

read -rp "Enter choice [1-6] (default 1): " CHOICE
CHOICE="${CHOICE:-1}"

download_file() {
    local repo="$1"
    local file="$2"
    local target="${MODELS_DIR}/$(basename "${file}")"
    local url="https://huggingface.co/${repo}/resolve/main/${file}"

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
        download_file "unsloth/gemma-4-12b-it-GGUF" "gemma-4-12b-it-UD-Q4_K_XL.gguf"
        download_file "unsloth/gemma-4-12b-it-GGUF" "MTP/mtp-gemma-4-12b-it-Q8_0.gguf"
        ;;
    2)
        download_file "unsloth/gemma-4-12b-it-GGUF" "MTP/mtp-gemma-4-12b-it-Q8_0.gguf"
        ;;
    3)
        download_file "unsloth/gemma-4-12b-it-GGUF" "gemma-4-12b-it-UD-Q4_K_XL.gguf"
        ;;
    4)
        download_file "unsloth/gemma-4-E4B-it-GGUF" "gemma-4-E4B-it-UD-Q4_K_XL.gguf"
        ;;
    5)
        download_file "unsloth/gemma-4-26B-A4B-it-GGUF" "gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf"
        ;;
    6)
        read -rp "Enter HF Repo (e.g. unsloth/gemma-4-12b-it-GGUF): " REPO
        read -rp "Enter GGUF Path: " FILE
        download_file "${REPO}" "${FILE}"
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac

echo -e "
${GREEN}${BOLD}All requested files downloaded successfully!${NC}"
echo -e "
To launch the server with Gemma 4 + MTP Speculative Acceleration:"
echo -e "  ${YELLOW}./start-agent-server.sh${NC}"
