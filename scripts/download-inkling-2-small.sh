#!/usr/bin/env bash
# ==============================================================================
# download-inkling-2-small.sh - Download Inkling-2-Small GGUF models
# Model:  https://huggingface.co/unsloth/Inkling-2-Small-GGUF
# Base:   https://huggingface.co/thinkingmachines/Inkling-2-Small
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models/Inkling-2-Small-GGUF"
REPO="unsloth/Inkling-2-Small-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}================================================================${NC}"
echo -e "${BOLD}${CYAN}  Inkling-2-Small (594B-A17B MoE VLM) GGUF Downloader           ${NC}"
echo -e "${BOLD}${CYAN}================================================================${NC}"
echo -e "${YELLOW}Note: Even the smallest quant (UD-IQ1_S, 171.7 GB) exceeds 16 GB VRAM + 30 GB RAM.${NC}"
echo -e "${YELLOW}      Use -ngl 0 (CPU/mmap) or consider Inkling-2-Mini when available.${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Inkling-2-Small variant to download:"
    echo -e "  ${GREEN}[1]  UD-IQ1_S   (171.7 GB)${NC} - ${BOLD}Smallest${NC} (closest to fitting local RAM+swap)"
    echo -e "  ${GREEN}[2]  UD-IQ1_M   (178.5 GB)${NC} - Dynamic 1-bit medium"
    echo -e "  ${GREEN}[3]  UD-IQ2_XXS (190.6 GB)${NC} - Unsloth README default example"
    echo -e "  ${GREEN}[4]  UD-Q2_K_XL (205.0 GB)${NC} - Dynamic 2-bit XL"
    echo -e "  ${GREEN}[5]  UD-IQ3_XXS (226.0 GB)${NC} - Dynamic 3-bit XXS"
    echo -e "  ${GREEN}[6]  UD-Q3_K_XL (285.1 GB)${NC} - Dynamic 3-bit XL (needs #32657)"
    echo -e "  ${GREEN}[7]  UD-Q4_K_XL (354.2 GB)${NC} - Dynamic 4-bit XL (needs #32657)"
    echo -e "  ${GREEN}[8]  Q8_0       (386.5 GB)${NC} - 8-bit + NVFP4 experts (needs #32657)"
    echo -e "  ${GREEN}[9]  mmproj-F16.gguf  (0.66 GB)${NC} - Vision projector F16 (RECOMMENDED)"
    echo -e "  ${GREEN}[10] mmproj-BF16.gguf (0.66 GB)${NC} - Vision projector BF16"
    echo -e "  ${GREEN}[11] Complete Stack: UD-IQ1_S + mmproj-F16.gguf (~172.4 GB)${NC} - RECOMMENDED"
    echo -e "  ${GREEN}[12] Complete Stack: UD-IQ2_XXS + mmproj-F16.gguf (~191.3 GB)${NC}"

    read -rp "Enter choice [1-12] (default 11): " USER_INPUT
    CHOICE="${USER_INPUT:-11}"
fi

download_hf_file() {
    local rel_path="$1"
    local target_path="${MODELS_DIR}/${rel_path}"
    local download_url="https://huggingface.co/${REPO}/resolve/main/${rel_path}"

    mkdir -p "$(dirname "${target_path}")"
    if [[ -f "${target_path}" ]]; then
        echo -e "${YELLOW}[SKIP] File already exists:${NC} ${target_path}"
        return 0
    fi

    echo -e "\n${BOLD}Downloading:${NC} ${rel_path}"
    echo -e "${BOLD}Destination:${NC} ${target_path}"
    curl -L -C - "${download_url}" -o "${target_path}" --progress-bar
    echo -e "${GREEN}[OK] Downloaded: ${rel_path}${NC}"
}

download_quant_folder() {
    local quant="$1"
    if command -v hf >/dev/null 2>&1; then
        echo -e "\n${BOLD}Downloading ${quant} via hf CLI...${NC}"
        hf download "${REPO}" --include "${quant}/*" --local-dir "${MODELS_DIR}"
        return 0
    fi

    echo -e "\n${BOLD}Fetching shard list for ${quant}...${NC}"
    local shards
    shards="$(curl -fsSL "https://huggingface.co/api/models/${REPO}/tree/main/${quant}" | python3 -c "import sys,json; print('\n'.join(x['path'] for x in json.load(sys.stdin) if x['path'].endswith('.gguf')))")"
    if [[ -z "${shards}" ]]; then
        echo -e "${RED}Failed to list shards for ${quant}.${NC}"
        exit 1
    fi
    while IFS= read -r shard; do
        download_hf_file "${shard}"
    done <<< "${shards}"
}

case "${CHOICE}" in
    1|ud-iq1_s|UD-IQ1_S|iq1_s)
        download_quant_folder "UD-IQ1_S"
        ;;
    2|ud-iq1_m|UD-IQ1_M|iq1_m)
        download_quant_folder "UD-IQ1_M"
        ;;
    3|ud-iq2_xxs|UD-IQ2_XXS|iq2_xxs)
        download_quant_folder "UD-IQ2_XXS"
        ;;
    4|ud-q2_k_xl|UD-Q2_K_XL|q2_k_xl)
        download_quant_folder "UD-Q2_K_XL"
        ;;
    5|ud-iq3_xxs|UD-IQ3_XXS|iq3_xxs)
        download_quant_folder "UD-IQ3_XXS"
        ;;
    6|ud-q3_k_xl|UD-Q3_K_XL|q3_k_xl)
        download_quant_folder "UD-Q3_K_XL"
        ;;
    7|ud-q4_k_xl|UD-Q4_K_XL|q4_k_xl)
        download_quant_folder "UD-Q4_K_XL"
        ;;
    8|q8_0|Q8_0)
        download_quant_folder "Q8_0"
        ;;
    9|mmproj|mmproj-f16|f16)
        download_hf_file "mmproj-F16.gguf"
        ;;
    10|mmproj-bf16|bf16)
        download_hf_file "mmproj-BF16.gguf"
        ;;
    11|stack|all|full)
        download_quant_folder "UD-IQ1_S"
        download_hf_file "mmproj-F16.gguf"
        ;;
    12|stack-iq2)
        download_quant_folder "UD-IQ2_XXS"
        download_hf_file "mmproj-F16.gguf"
        ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1..12|iq1_s|iq1_m|iq2_xxs|q2_k_xl|iq3_xxs|q3_k_xl|q4_k_xl|q8_0|mmproj|all]"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}Download completed!${NC}"
echo -e "Launch with: ${YELLOW}./start-inkling-2-small.sh${NC}"
