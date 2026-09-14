#!/usr/bin/env bash
# ==============================================================================
# download-qwen3.8-gsq.sh - Download Qwen3.8-27B GSQ-RCO GGUF models & Vision mmproj
# Repo: ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
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
REPO="ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  Qwen3.8-27B GSQ-RCO Downloader (ISTA-DASLab)        ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

CHOICE="${1:-}"

show_help() {
    cat << EOF
Usage: $(basename "$0") [choice|alias|filename]

Downloads Qwen3.8-27B GSQ-RCO quantized models and multimodal vision projector
from ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF.

Options / Choices:
  1, iq2_s_mtp     Download Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf (9.65 GB) - RECOMMENDED
  2, iq2_xs_mtp    Download Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf (8.77 GB) - Lightest MTP
  3, iq3_xxs_mtp   Download Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf (10.45 GB) - 3-bit MTP
  4, iq3_s_mtp     Download Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf (12.15 GB) - Task-Lossless
  5, mmproj        Download mmproj-Qwen3.8-27B-BF16.gguf (0.90 GB) - Vision Projector
  6, bundle        Download IQ2_S-mtp + mmproj (10.55 GB total) - Complete Vision + MTP
  7, non_mtp       Select non-MTP variant (IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S)
  8, custom        Enter custom filename from the repository
  -h, --help       Show this help message

Examples:
  $(basename "$0") 1          # Download recommended IQ2_S with built-in MTP
  $(basename "$0") mmproj     # Download vision projector
  $(basename "$0") bundle     # Download model and vision projector together
EOF
}

if [[ "${CHOICE}" == "-h" || "${CHOICE}" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect download option:"
    echo -e "  ${GREEN}[1] Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf   (9.65 GB)${NC} - ${BOLD}RECOMMENDED${NC} (100% AIME25 recovery, MTP, 100% GPU)"
    echo -e "  ${GREEN}[2] Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf  (8.77 GB)${NC} - Lightest MTP build (Maximum context head-room)"
    echo -e "  ${GREEN}[3] Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf (10.45 GB)${NC} - 3.0 bpw MTP (High precision, 100% GPU offload)"
    echo -e "  ${GREEN}[4] Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf   (12.15 GB)${NC} - 3.5 bpw Task-Lossless (Matches BF16 on LCB & AIME)"
    echo -e "  ${GREEN}[5] mmproj-Qwen3.8-27B-BF16.gguf         (0.90 GB)${NC} - Multimodal Vision Projector (for image & UI tasks)"
    echo -e "  ${GREEN}[6] Bundle: IQ2_S-mtp + mmproj           (10.55 GB)${NC} - Complete MTP Reasoning + Vision setup"
    echo -e "  ${GREEN}[7] Non-MTP Quantizations (Standard GSQ-RCO)${NC}"
    echo -e "  ${GREEN}[8] Custom GGUF filename from repository${NC}"

    read -rp "Enter choice [1-8] (default 1): " USER_INPUT
    CHOICE="${USER_INPUT:-1}"
fi

download_file() {
    local file="$1"
    local target="${MODELS_DIR}/${file}"
    local url="https://huggingface.co/${REPO}/resolve/main/${file}"

    echo -e "\n${BOLD}Downloading:${NC} ${file}"
    echo -e "${BOLD}Destination:${NC} ${target}"
    echo -e "${BOLD}URL:${NC}         ${url}\n"

    curl -L -C - "${url}" -o "${target}" --progress-bar
    echo -e "${GREEN}[OK] Successfully downloaded: ${target}${NC}"
}

case "${CHOICE}" in
    1|iq2_s_mtp|iq2_s|IQ2_S)
        download_file "Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf"
        ;;
    2|iq2_xs_mtp|iq2_xs|IQ2_XS)
        download_file "Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf"
        ;;
    3|iq3_xxs_mtp|iq3_xxs|IQ3_XXS)
        download_file "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
        ;;
    4|iq3_s_mtp|iq3_s|IQ3_S)
        download_file "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
        ;;
    5|mmproj|vision|VISION)
        download_file "mmproj-Qwen3.8-27B-BF16.gguf"
        ;;
    6|bundle|BUNDLE)
        download_file "Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf"
        download_file "mmproj-Qwen3.8-27B-BF16.gguf"
        ;;
    7|non_mtp|nomtp)
        echo -e "\nSelect Non-MTP Quantization:"
        echo -e "  [1] Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf  (8.40 GB)"
        echo -e "  [2] Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf   (9.30 GB)"
        echo -e "  [3] Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf (10.10 GB)"
        echo -e "  [4] Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf   (11.80 GB)"
        read -rp "Enter sub-choice [1-4] (default 2): " SUB_CHOICE
        case "${SUB_CHOICE:-2}" in
            1) download_file "Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf" ;;
            2) download_file "Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf" ;;
            3) download_file "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf" ;;
            4) download_file "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ;;
            *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
        esac
        ;;
    8|custom|CUSTOM)
        read -rp "Enter exact GGUF filename in ${REPO}: " CUSTOM_FILE
        download_file "${CUSTOM_FILE}"
        ;;
    *.gguf)
        download_file "${CHOICE}"
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        show_help
        exit 1
        ;;
esac

echo -e "\n${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${GREEN}  Download complete! Model ready for execution.       ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "To start the Qwen3.8 GSQ server with MTP speculative decoding and Vision:"
echo -e "  ${YELLOW}./start-qwen3.8-gsq.sh${NC}"
echo -e "To launch with ultra-deep 262k context:"
echo -e "  ${YELLOW}./start-qwen3.8-gsq-max-context.sh${NC}\n"
