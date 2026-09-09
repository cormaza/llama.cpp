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

CHOICE="${1:-}"

if [[ "${CHOICE}" == "-h" || "${CHOICE}" == "--help" ]]; then
    echo -e "Usage: $(basename "$0") [choice|quantization|filename]"
    echo -e "Examples:"
    echo -e "  $(basename "$0") 1          # Download Qwen3.8-27B-UD-Q2_K_XL.gguf (9.15 GB)"
    echo -e "  $(basename "$0") q2_k_xl    # By quantization name"
    echo -e "  $(basename "$0") iq3_xxs    # Download 10.18 GB version"
    exit 0
fi

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Qwen3.8-27B download option:"
    echo -e "  [1] Base UD-Q2_K_XL (9.15 GB) - RECOMMENDED (100% GPU Offload in 16GB VRAM, ~28-35 t/s with MTP)"
    echo -e "  [2] Base UD-Q2_K_XL (9.15 GB) + MTP Q4_0 Draft (1.27 GB) - Complete 100% GPU Bundle"
    echo -e "  [3] Base UD-IQ3_XXS (10.18 GB) - Near-lossless 3-bit (100% GPU Offload, max fidelity)"
    echo -e "  [4] Base UD-Q3_K_XL (12.24 GB) - Heavy 3-bit (Hybrid CPU/GPU mode)"
    echo -e "  [5] Only MTP Draft Module: mtp-Qwen3.8-27B-Q4_0.gguf (1.27 GB) - Download draft if you already have the base"
    echo -e "  [6] Base UD-IQ2_XXS (6.77 GB) - Ultra-compact 2-bit (Max context 256k)"
    echo -e "  [7] Base UD-Q4_K_M  (15.33 GB) - Maximum precision"
    echo -e "  [8] Custom GGUF filename"

    read -rp "Enter choice [1-8] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

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
    1|q2_k_xl|Q2_K_XL|9gb|9GB)
        download_file "Qwen3.8-27B-UD-Q2_K_XL.gguf"
        ;;
    2|bundle|BUNDLE)
        download_file "Qwen3.8-27B-UD-Q2_K_XL.gguf"
        download_file "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
        ;;
    3|iq3_xxs|IQ3_XXS|10gb|10GB)
        download_file "Qwen3.8-27B-UD-IQ3_XXS.gguf"
        ;;
    4|q3_k_xl|Q3_K_XL)
        download_file "Qwen3.8-27B-UD-Q3_K_XL.gguf"
        ;;
    5|mtp|MTP)
        download_file "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
        ;;
    6|iq2_xxs|IQ2_XXS|7gb|7GB)
        download_file "Qwen3.8-27B-UD-IQ2_XXS.gguf"
        ;;
    7|q4_k_m|Q4_K_M)
        download_file "Qwen3.8-27B-UD-Q4_K_M.gguf"
        ;;
    8)
        read -rp "Enter exact HF path/filename: " FILE
        download_file "${FILE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [choice|quantization|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") 1          # Download Qwen3.8-27B-UD-Q2_K_XL.gguf (9.15 GB)"
        echo -e "  $(basename "$0") q2_k_xl    # By quantization name"
        echo -e "  $(basename "$0") iq3_xxs    # Download 10.18 GB version"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "
${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "
To launch Qwen with MTP Speculative Acceleration:"
echo -e "  ${YELLOW}./start-qwen-max-context.sh${NC}"
