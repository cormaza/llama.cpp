#!/usr/bin/env bash
# ==============================================================================
# download-bonsai-2-crack.sh - Downloader for Bonsai 2 27B 1-bit CRACK GGUF
# Model: https://huggingface.co/dealignai/Bonsai-2-27B-1bit-CRACK-GGUF
# Base:  https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
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
CRACK_REPO="dealignai/Bonsai-2-27B-1bit-CRACK-GGUF"
VISION_REPO="prism-ml/Ternary-Bonsai-2-27B-gguf"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Bonsai 2 27B CRACK Downloader (Uncensored PTQ1_0)   ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect component to download:"
    echo -e "  ${GREEN}[1] Bonsai-2-27B-PTQ1_0-CRACK.gguf        (5.95 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Uncensored / Dealigned PTQ1_0 1.75 bpw)"
    echo -e "  ${GREEN}[2] Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf (0.63 GB)${NC} - Vision Projector (from base Prism ML repo for multimodal input)"
    echo -e "  ${GREEN}[3] Complete Stack: CRACK Model + Vision  (6.58 GB)${NC} - Full serving bundle"
    echo -e "  ${GREEN}[4] Ternary-Bonsai-2-27B-mmproj-BF16.gguf (0.93 GB)${NC} - Reference BF16 Vision Projector"
    echo -e "  ${GREEN}[5] Custom filename from ${CRACK_REPO}${NC}"

    read -rp "Enter choice [1-5] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

download_hf_file() {
    local repo="$1"
    local filename="$2"
    local target_path="${MODELS_DIR}/${filename}"
    local download_url="https://huggingface.co/${repo}/resolve/main/${filename}"

    if [[ -f "${target_path}" ]]; then
        echo -e "${YELLOW}[SKIP] File already exists:${NC} ${target_path}"
        return 0
    fi

    echo -e "\n${BOLD}Downloading:${NC} ${filename}"
    echo -e "${BOLD}Source Repo:${NC} ${repo}"
    echo -e "${BOLD}Destination:${NC} ${target_path}"
    echo -e "${BOLD}URL:${NC}         ${download_url}\n"

    curl -L -C - "${download_url}" -o "${target_path}" --progress-bar

    echo -e "${GREEN}[OK] Downloaded: ${filename}${NC}"
}

case "${CHOICE}" in
    1|crack|model|ptq|ptq1_0)
        download_hf_file "${CRACK_REPO}" "Bonsai-2-27B-PTQ1_0-CRACK.gguf"
        ;;
    2|mmproj|vision|q8|Q8)
        download_hf_file "${VISION_REPO}" "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
        ;;
    3|all|full|stack)
        download_hf_file "${CRACK_REPO}" "Bonsai-2-27B-PTQ1_0-CRACK.gguf"
        download_hf_file "${VISION_REPO}" "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
        ;;
    4|bf16|mmproj-bf16)
        download_hf_file "${VISION_REPO}" "Ternary-Bonsai-2-27B-mmproj-BF16.gguf"
        ;;
    5)
        read -rp "Enter exact GGUF filename in ${CRACK_REPO}: " CUSTOM_FILE
        download_hf_file "${CRACK_REPO}" "${CUSTOM_FILE}"
        ;;
    *.gguf)
        download_hf_file "${CRACK_REPO}" "${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|crack|mmproj|all|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Bonsai 2 27B CRACK model (5.95 GB)"
        echo -e "  $(basename "$0") mmproj     # Download Q8_0 vision projector (0.63 GB)"
        echo -e "  $(basename "$0") all        # Download CRACK model + vision projector"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}Download process completed!${NC}"
echo -e "\nLaunch command:"
echo -e "  ${YELLOW}./start-bonsai-2-crack.sh${NC}"
