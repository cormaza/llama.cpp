#!/usr/bin/env bash
# ==============================================================================
# download-bonsai-27b.sh - Downloader for Ternary Bonsai 27B GGUF models
# Model: https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
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
REPO="prism-ml/Ternary-Bonsai-27B-gguf"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Ternary Bonsai 27B Downloader (Prism ML / Qwen3.6)   ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect component to download:"
    echo -e "  ${GREEN}[1] Ternary-Bonsai-27B-PQ2_0.gguf       (7.17 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Ternary 1.71 bpw, native for this branch)"
    echo -e "  ${GREEN}[2] Ternary-Bonsai-27B-dspark-Q4_1.gguf  (1.95 GB)${NC} - DSpark Speculative Drafter (1.34x decode speedup)"
    echo -e "  ${GREEN}[3] Ternary-Bonsai-27B-mmproj-Q8_0.gguf  (0.63 GB)${NC} - Vision Projector (for multimodal image input)"
    echo -e "  ${GREEN}[4] Complete Stack (Model + DSpark + Vision) (9.75 GB)${NC} - Full serving bundle"
    echo -e "  ${GREEN}[5] Ternary-Bonsai-27B-Q2_g64.gguf       (7.59 GB)${NC} - Group-64 pack (upstream-compatible)"
    echo -e "  ${GREEN}[6] Ternary-Bonsai-27B-Q2_0.gguf         (7.17 GB)${NC} - Standard Q2_0 pack"
    echo -e "  ${GREEN}[7] Custom filename from ${REPO}${NC}"

    read -rp "Enter choice [1-7] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

download_file() {
    local filename="$1"
    local target_path="${MODELS_DIR}/${filename}"
    local download_url="https://huggingface.co/${REPO}/resolve/main/${filename}"

    if [[ -f "${target_path}" ]]; then
        echo -e "${YELLOW}[SKIP] File already exists:${NC} ${target_path}"
        return 0
    fi

    echo -e "\n${BOLD}Downloading:${NC} ${filename}"
    echo -e "${BOLD}Source Repo:${NC} ${REPO}"
    echo -e "${BOLD}Destination:${NC} ${target_path}"
    echo -e "${BOLD}URL:${NC}         ${download_url}\n"

    curl -L -C - "${download_url}" -o "${target_path}" --progress-bar

    echo -e "${GREEN}[OK] Downloaded: ${filename}${NC}"
}

case "${CHOICE}" in
    1|pq2|PQ2|pq2_0|PQ2_0)
        download_file "Ternary-Bonsai-27B-PQ2_0.gguf"
        ;;
    2|dspark|drafter|spec)
        download_file "Ternary-Bonsai-27B-dspark-Q4_1.gguf"
        ;;
    3|mmproj|vision)
        download_file "Ternary-Bonsai-27B-mmproj-Q8_0.gguf"
        ;;
    4|all|full|stack)
        download_file "Ternary-Bonsai-27B-PQ2_0.gguf"
        download_file "Ternary-Bonsai-27B-dspark-Q4_1.gguf"
        download_file "Ternary-Bonsai-27B-mmproj-Q8_0.gguf"
        ;;
    5|q2_g64|g64)
        download_file "Ternary-Bonsai-27B-Q2_g64.gguf"
        ;;
    6|q2_0|Q2_0)
        download_file "Ternary-Bonsai-27B-Q2_0.gguf"
        ;;
    7)
        read -rp "Enter exact GGUF filename in ${REPO}: " CUSTOM_FILE
        download_file "${CUSTOM_FILE}"
        ;;
    *.gguf)
        download_file "${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|5|6|all|pq2|dspark|mmproj|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download base PQ2_0 model"
        echo -e "  $(basename "$0") dspark     # Download DSpark speculative drafter"
        echo -e "  $(basename "$0") mmproj     # Download vision projector"
        echo -e "  $(basename "$0") all        # Download model + drafter + vision"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}Download process completed!${NC}"
echo -e "\nLaunch command:"
echo -e "  ${YELLOW}./start-bonsai-27b.sh${NC}"
