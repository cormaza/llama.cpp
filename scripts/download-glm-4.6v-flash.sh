#!/usr/bin/env bash
# ==============================================================================
# download-glm-4.6v-flash.sh - Download GLM-4.6V-Flash GGUF models
# Model:  https://huggingface.co/prithivMLmods/GLM-4.6V-Flash-GGUF
# Base:   https://huggingface.co/zai-org/GLM-4.6V-Flash
# Vision: https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF (mmproj-Q8_0)
#         https://huggingface.co/bartowski/zai-org_GLM-4.6V-Flash-GGUF (mmproj-bf16/f16)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
REPO="prithivMLmods/GLM-4.6V-Flash-GGUF"
MMPROJ_REPO_GGML="ggml-org/GLM-4.6V-Flash-GGUF"
MMPROJ_REPO_BARTOWSKI="bartowski/zai-org_GLM-4.6V-Flash-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}================================================================${NC}"
echo -e "${BOLD}${CYAN}  GLM-4.6V-Flash (9.4B VLM Agent) GGUF Downloader               ${NC}"
echo -e "${BOLD}${CYAN}================================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect GLM-4.6V-Flash variant to download:"
    echo -e "  ${GREEN}[1]  GLM-4.6V-Flash.i1-Q4_K_M.gguf           (5.74 GB)${NC} - ${BOLD}RECOMMENDED${NC} (imatrix 4-bit, fits 128k ctx + vision in 16GB VRAM)"
    echo -e "  ${GREEN}[2]  GLM-4.6V-Flash.Q4_K_M.gguf              (5.74 GB)${NC} - Standard 4-bit Q4_K_M"
    echo -e "  ${GREEN}[3]  GLM-4.6V-Flash.i1-Q5_K_M.gguf           (6.57 GB)${NC} - High precision 5-bit (imatrix)"
    echo -e "  ${GREEN}[4]  GLM-4.6V-Flash.Q5_K_M.gguf              (6.57 GB)${NC} - High precision 5-bit (standard)"
    echo -e "  ${GREEN}[5]  GLM-4.6V-Flash.i1-Q6_K.gguf             (7.70 GB)${NC} - Near-lossless 6-bit (imatrix)"
    echo -e "  ${GREEN}[6]  GLM-4.6V-Flash.Q8_0.gguf                (9.31 GB)${NC} - Reference quality 8-bit"
    echo -e "  ${GREEN}[7]  GLM-4.6V-Flash.i1-IQ4_XS.gguf           (4.90 GB)${NC} - Compact 4-bit imatrix (~11.1 GB free VRAM)"
    echo -e "  ${GREEN}[8]  GLM-4.6V-Flash.i1-Q3_K_M.gguf           (4.63 GB)${NC} - Compact 3-bit imatrix (~11.4 GB free VRAM)"
    echo -e "  ${GREEN}[9]  mmproj-GLM-4.6V-Flash-Q8_0.gguf         (0.91 GB)${NC} - Multimodal Vision Projector Q8_0 (ggml-org)"
    echo -e "  ${GREEN}[10] mmproj-zai-org_GLM-4.6V-Flash-bf16.gguf (1.71 GB)${NC} - Multimodal Vision Projector BF16 (bartowski)"
    echo -e "  ${GREEN}[11] Complete Stack: i1-Q4_K_M + mmproj-Q8_0 (6.65 GB)${NC} - Full multimodal agent bundle (RECOMMENDED)"
    echo -e "  ${GREEN}[12] Complete Stack: i1-Q5_K_M + mmproj-Q8_0 (7.48 GB)${NC} - High-precision multimodal agent bundle"
    echo -e "  ${GREEN}[13] Custom GGUF filename from ${REPO}${NC}"

    read -rp "Enter choice [1-13] (default 11): " USER_INPUT
    CHOICE="${USER_INPUT:-11}"
fi

download_hf_file() {
    local source_repo="$1"
    local filename="$2"
    local target_path="${MODELS_DIR}/${filename}"
    local download_url="https://huggingface.co/${source_repo}/resolve/main/${filename}"

    if [[ -f "${target_path}" ]]; then
        echo -e "${YELLOW}[SKIP] File already exists:${NC} ${target_path}"
        return 0
    fi

    echo -e "\n${BOLD}Downloading:${NC} ${filename}"
    echo -e "${BOLD}Source Repo:${NC} ${source_repo}"
    echo -e "${BOLD}Destination:${NC} ${target_path}"
    echo -e "${BOLD}URL:${NC}         ${download_url}\n"

    curl -L -C - "${download_url}" -o "${target_path}" --progress-bar

    echo -e "${GREEN}[OK] Downloaded: ${filename}${NC}"
}

case "${CHOICE}" in
    1|i1-q4_k_m|i1-Q4_K_M|i1-q4|q4_k_m|Q4_K_M|q4|Q4)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.i1-Q4_K_M.gguf"
        ;;
    2|static-q4|GLM-4.6V-Flash.Q4_K_M.gguf)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.Q4_K_M.gguf"
        ;;
    3|i1-q5_k_m|i1-Q5_K_M|i1-q5|q5_k_m|Q5_K_M|q5|Q5)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.i1-Q5_K_M.gguf"
        ;;
    4|static-q5|GLM-4.6V-Flash.Q5_K_M.gguf)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.Q5_K_M.gguf"
        ;;
    5|i1-q6_k|i1-Q6_K|i1-q6|q6_k|Q6_K|q6|Q6)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.i1-Q6_K.gguf"
        ;;
    6|q8_0|Q8_0|q8|Q8)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.Q8_0.gguf"
        ;;
    7|i1-iq4_xs|i1-IQ4_XS|iq4_xs|IQ4_XS|iq4|IQ4)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.i1-IQ4_XS.gguf"
        ;;
    8|i1-q3_k_m|i1-Q3_K_M|q3_k_m|Q3_K_M|q3|Q3)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.i1-Q3_K_M.gguf"
        ;;
    9|mmproj|mmproj-q8|vision)
        download_hf_file "${MMPROJ_REPO_GGML}" "mmproj-GLM-4.6V-Flash-Q8_0.gguf"
        ;;
    10|mmproj-bf16|bf16)
        download_hf_file "${MMPROJ_REPO_BARTOWSKI}" "mmproj-zai-org_GLM-4.6V-Flash-bf16.gguf"
        ;;
    11|stack|all|full)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.i1-Q4_K_M.gguf"
        download_hf_file "${MMPROJ_REPO_GGML}" "mmproj-GLM-4.6V-Flash-Q8_0.gguf"
        ;;
    12|stack-q5|q5-full)
        download_hf_file "${REPO}" "GLM-4.6V-Flash.i1-Q5_K_M.gguf"
        download_hf_file "${MMPROJ_REPO_GGML}" "mmproj-GLM-4.6V-Flash-Q8_0.gguf"
        ;;
    13)
        read -rp "Enter exact GGUF filename in ${REPO}: " CUSTOM_FILE
        download_hf_file "${REPO}" "${CUSTOM_FILE}"
        ;;
    *.gguf)
        download_hf_file "${REPO}" "${CHOICE}"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1..13|q4|q5|q6|q8|iq4|q3|mmproj|mmproj-bf16|all|stack-q5|filename]"
        echo -e "Examples:"
        echo -e "  $(basename "$0") all         # Download i1-Q4_K_M + mmproj-Q8_0 vision projector (Default)"
        echo -e "  $(basename "$0") 1           # Download GLM-4.6V-Flash.i1-Q4_K_M.gguf (text model only)"
        echo -e "  $(basename "$0") q5          # Download GLM-4.6V-Flash.i1-Q5_K_M.gguf"
        echo -e "  $(basename "$0") q8          # Download GLM-4.6V-Flash.Q8_0.gguf"
        echo -e "  $(basename "$0") mmproj      # Download mmproj-GLM-4.6V-Flash-Q8_0.gguf (ggml-org)"
        echo -e "  $(basename "$0") mmproj-bf16 # Download mmproj-zai-org_GLM-4.6V-Flash-bf16.gguf (bartowski)"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}Download completed successfully!${NC}"
echo -e "\nLaunch command:"
echo -e "  ${YELLOW}./start-glm-4.6v-flash.sh${NC}"
