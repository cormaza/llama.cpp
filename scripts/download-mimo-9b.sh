#!/usr/bin/env bash
# ==============================================================================
# download-mimo-9b.sh - Download MiMo-V2.6-Distill-Qwen-9B GGUF models
# Model: https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF
# Base:  https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
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
REPO="bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  MiMo-V2.6-Distill-Qwen-9B Downloader (Xiaomi MiMo)   ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect MiMo-V2.6-Distill-Qwen-9B variant to download:"
    echo -e "  ${GREEN}[1] MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf      (5.84 GB)${NC} - ${BOLD}RECOMMENDED${NC} (Best balance, ~10.1 GB free VRAM)"
    echo -e "  ${GREEN}[2] MiMo-V2.6-Distill-Qwen-9B-Q5_K_M.gguf      (6.88 GB)${NC} - High precision 5-bit (~9.1 GB free VRAM)"
    echo -e "  ${GREEN}[3] MiMo-V2.6-Distill-Qwen-9B-Q6_K.gguf        (7.79 GB)${NC} - Very high quality 6-bit (~8.2 GB free VRAM)"
    echo -e "  ${GREEN}[4] MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf        (9.55 GB)${NC} - Near-lossless reference quality (~6.4 GB free VRAM)"
    echo -e "  ${GREEN}[5] MiMo-V2.6-Distill-Qwen-9B-IQ4_XS.gguf      (5.23 GB)${NC} - Compact 4-bit quant (~10.7 GB free VRAM)"
    echo -e "  ${GREEN}[6] mmproj-MiMo-V2.6-Distill-Qwen-9B-bf16.gguf (1.30 GB)${NC} - Multimodal Vision Projector (for image/vision input)"
    echo -e "  ${GREEN}[7] Complete Stack: Q4_K_M + Vision Projector  (7.14 GB)${NC} - Full serving bundle"
    echo -e "  ${GREEN}[8] Complete Stack: Q5_K_M + Vision Projector  (8.18 GB)${NC} - Full high-precision serving bundle"
    echo -e "  ${GREEN}[9] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-9] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

download_hf_file() {
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
    1|q4_k_m|Q4_K_M|q4|Q4)
        download_hf_file "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"
        ;;
    2|q5_k_m|Q5_K_M|q5|Q5)
        download_hf_file "MiMo-V2.6-Distill-Qwen-9B-Q5_K_M.gguf"
        ;;
    3|q6_k|Q6_K|q6|Q6)
        download_hf_file "MiMo-V2.6-Distill-Qwen-9B-Q6_K.gguf"
        ;;
    4|q8_0|Q8_0|q8|Q8)
        download_hf_file "MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf"
        ;;
    5|iq4_xs|IQ4_XS|iq4|IQ4)
        download_hf_file "MiMo-V2.6-Distill-Qwen-9B-IQ4_XS.gguf"
        ;;
    6|mmproj|vision|bf16)
        download_hf_file "mmproj-MiMo-V2.6-Distill-Qwen-9B-bf16.gguf"
        ;;
    7|stack|all|full)
        download_hf_file "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"
        download_hf_file "mmproj-MiMo-V2.6-Distill-Qwen-9B-bf16.gguf"
        ;;
    8|stack-q5|q5-full)
        download_hf_file "MiMo-V2.6-Distill-Qwen-9B-Q5_K_M.gguf"
        download_hf_file "mmproj-MiMo-V2.6-Distill-Qwen-9B-bf16.gguf"
        ;;
    9)
        read -rp "Enter exact GGUF filename in ${REPO}: " CUSTOM_FILE
        download_hf_file "${CUSTOM_FILE}"
        ;;
    *.gguf)
        download_hf_file "${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1|2|3|4|5|6|7|8|q4|q5|q8|mmproj|all|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download MiMo-V2.6 Q4_K_M (Default)"
        echo -e "  $(basename "$0") q5         # Download MiMo-V2.6 Q5_K_M"
        echo -e "  $(basename "$0") mmproj     # Download vision projector"
        echo -e "  $(basename "$0") all        # Download Q4_K_M + vision projector"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "\nLaunch command:"
echo -e "  ${YELLOW}./start-mimo-9b.sh${NC}"
