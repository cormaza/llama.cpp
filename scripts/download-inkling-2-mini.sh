#!/usr/bin/env bash
# ==============================================================================
# download-inkling-2-mini.sh - Download Inkling-2-Mini GGUF models
# Model:  https://huggingface.co/unsloth/Inkling-2-Mini-GGUF
# Base:   https://huggingface.co/thinkingmachines/Inkling-2-Mini
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models/Inkling-2-Mini-GGUF"
REPO="unsloth/Inkling-2-Mini-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}================================================================${NC}"
echo -e "${BOLD}${CYAN}  Inkling-2-Mini (26.2B-A5.4B Looped MoE VLM) GGUF Downloader   ${NC}"
echo -e "${BOLD}${CYAN}================================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Inkling-2-Mini variant to download:"
    echo -e "  ${GREEN}[1]  Inkling-2-Mini-UD-IQ1_S.gguf   ( 7.76 GB)${NC} - Ultra-compact 1-bit"
    echo -e "  ${GREEN}[2]  Inkling-2-Mini-UD-IQ1_M.gguf   ( 8.02 GB)${NC} - Compact 1-bit medium"
    echo -e "  ${GREEN}[3]  Inkling-2-Mini-UD-IQ2_XXS.gguf ( 8.49 GB)${NC} - Dynamic 2-bit XXS"
    echo -e "  ${GREEN}[4]  Inkling-2-Mini-UD-Q2_K_XL.gguf ( 9.14 GB)${NC} - Dynamic 2-bit XL"
    echo -e "  ${GREEN}[5]  Inkling-2-Mini-UD-IQ3_XXS.gguf (11.66 GB)${NC} - ${BOLD}RECOMMENDED for 16GB VRAM${NC} (fits full GPU + 32K ctx + vision)"
    echo -e "  ${GREEN}[6]  Inkling-2-Mini-UD-Q3_K_XL.gguf (13.63 GB)${NC} - High quality 3-bit (fits 16GB VRAM with q8_0/q4_0 KV)"
    echo -e "  ${GREEN}[7]  Inkling-2-Mini-UD-IQ4_XS.gguf  (15.73 GB)${NC} - 4-bit (needs partial CPU offload on 16GB VRAM)"
    echo -e "  ${GREEN}[8]  Inkling-2-Mini-UD-Q4_K_M.gguf  (16.61 GB)${NC} - 4-bit medium (GPU + RAM split)"
    echo -e "  ${GREEN}[9]  Inkling-2-Mini-UD-Q4_K_XL.gguf (17.63 GB)${NC} - 4-bit XL (GPU + RAM split)"
    echo -e "  ${GREEN}[10] Inkling-2-Mini-UD-Q8_K_XL.gguf (22.87 GB)${NC} - Reference 8-bit (RAM + partial GPU)"
    echo -e "  ${GREEN}[11] mmproj-F16.gguf                ( 0.64 GB)${NC} - Vision projector F16 (RECOMMENDED)"
    echo -e "  ${GREEN}[12] mmproj-BF16.gguf               ( 0.64 GB)${NC} - Vision projector BF16"
    echo -e "  ${GREEN}[13] Complete Stack: UD-IQ3_XXS + mmproj-F16.gguf (~12.30 GB)${NC} - ${BOLD}RECOMMENDED for 16GB VRAM${NC}"
    echo -e "  ${GREEN}[14] Complete Stack: UD-Q3_K_XL + mmproj-F16.gguf (~14.27 GB)${NC} - Tight 16GB VRAM stack"

    read -rp "Enter choice [1-14] (default 13): " USER_INPUT
    CHOICE="${USER_INPUT:-13}"
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
    echo -e "${BOLD}Destination:${NC} ${target_path}"
    curl -L -C - "${download_url}" -o "${target_path}" --progress-bar
    echo -e "${GREEN}[OK] Downloaded: ${filename}${NC}"
}

case "${CHOICE}" in
    1|iq1_s|UD-IQ1_S)       download_hf_file "Inkling-2-Mini-UD-IQ1_S.gguf" ;;
    2|iq1_m|UD-IQ1_M)       download_hf_file "Inkling-2-Mini-UD-IQ1_M.gguf" ;;
    3|iq2_xxs|UD-IQ2_XXS)   download_hf_file "Inkling-2-Mini-UD-IQ2_XXS.gguf" ;;
    4|q2_k_xl|UD-Q2_K_XL)   download_hf_file "Inkling-2-Mini-UD-Q2_K_XL.gguf" ;;
    5|iq3_xxs|UD-IQ3_XXS)   download_hf_file "Inkling-2-Mini-UD-IQ3_XXS.gguf" ;;
    6|q3_k_xl|UD-Q3_K_XL)   download_hf_file "Inkling-2-Mini-UD-Q3_K_XL.gguf" ;;
    7|iq4_xs|UD-IQ4_XS)     download_hf_file "Inkling-2-Mini-UD-IQ4_XS.gguf" ;;
    8|q4_k_m|UD-Q4_K_M)     download_hf_file "Inkling-2-Mini-UD-Q4_K_M.gguf" ;;
    9|q4_k_xl|UD-Q4_K_XL)   download_hf_file "Inkling-2-Mini-UD-Q4_K_XL.gguf" ;;
    10|q8_k_xl|UD-Q8_K_XL)  download_hf_file "Inkling-2-Mini-UD-Q8_K_XL.gguf" ;;
    11|mmproj|mmproj-f16)   download_hf_file "mmproj-F16.gguf" ;;
    12|mmproj-bf16)         download_hf_file "mmproj-BF16.gguf" ;;
    13|stack|all|full)
        download_hf_file "Inkling-2-Mini-UD-IQ3_XXS.gguf"
        download_hf_file "mmproj-F16.gguf"
        ;;
    14|stack-q3)
        download_hf_file "Inkling-2-Mini-UD-Q3_K_XL.gguf"
        download_hf_file "mmproj-F16.gguf"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1..14|iq1_s|iq1_m|iq2_xxs|q2_k_xl|iq3_xxs|q3_k_xl|iq4_xs|q4_k_m|q4_k_xl|q8_k_xl|mmproj|all|stack-q3]"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}Download completed!${NC}"
echo -e "Launch with: ${YELLOW}./start-inkling-2-mini.sh${NC}"
