#!/usr/bin/env bash
# ==============================================================================
# download-ternary-bonsai-2.5-27b.sh - Download Ternary-Bonsai-2.5-27B GGUF models
# Model:  https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF
# Base:   https://huggingface.co/Qwen/Qwen3.8-27B
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models/Ternary-Bonsai-2.5-27B-GGUF"
REPO="prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
mkdir -p "${MODELS_DIR}"

echo -e "${BOLD}${CYAN}================================================================${NC}"
echo -e "${BOLD}${CYAN}  Ternary-Bonsai-2.5-27B (Qwen3.8-27B) GGUF Downloader          ${NC}"
echo -e "${BOLD}${CYAN}================================================================${NC}"

CHOICE="${1:-}"

if [[ -z "${CHOICE}" ]]; then
    echo -e "\nSelect Ternary-Bonsai-2.5-27B variant to download:"
    echo -e "  ${GREEN}[1]  Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf    (7.91 GB)${NC} - Upstream llama.cpp + MTP draft head"
    echo -e "  ${GREEN}[2]  Ternary-Bonsai-2.5-27B-Q2_0.gguf        (7.67 GB)${NC} - Upstream llama.cpp (no MTP)"
    echo -e "  ${GREEN}[3]  Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf (0.63 GB)${NC} - Vision projector Q8_0"
    echo -e "  ${GREEN}[4]  Ternary-Bonsai-2.5-27B-mmproj-BF16.gguf (0.93 GB)${NC} - Vision projector BF16"
    echo -e "  ${GREEN}[5]  Complete Stack: Q2_0-mtp + mmproj-Q8_0  (~8.54 GB)${NC} - ${BOLD}RECOMMENDED for 16GB VRAM (upstream llama.cpp)${NC}"
    echo -e "  ${GREEN}[6]  Complete Stack: Q2_0-mtp + mmproj-BF16  (~8.84 GB)${NC} - Full precision vision projector"
    echo -e "  ${YELLOW}[7]  Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf  (6.23 GB)${NC} - [FORK ONLY] Needs PrismML-Eng/llama.cpp"
    echo -e "  ${YELLOW}[8]  Ternary-Bonsai-2.5-27B-PTQ1_0.gguf      (5.99 GB)${NC} - [FORK ONLY] Needs PrismML-Eng/llama.cpp"
    echo -e "  ${YELLOW}[9]  Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf   (7.49 GB)${NC} - [FORK ONLY] Needs PrismML-Eng/llama.cpp"
    echo -e "  ${YELLOW}[10] Ternary-Bonsai-2.5-27B-PQ2_0.gguf       (7.25 GB)${NC} - [FORK ONLY] Needs PrismML-Eng/llama.cpp"

    read -rp "Enter choice [1-10] (default 5): " USER_INPUT
    CHOICE="${USER_INPUT:-5}"
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
    1|q2_0-mtp|Q2_0-mtp)      download_hf_file "Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf" ;;
    2|q2_0|Q2_0)              download_hf_file "Ternary-Bonsai-2.5-27B-Q2_0.gguf" ;;
    3|mmproj|mmproj-q8_0)     download_hf_file "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf" ;;
    4|mmproj-bf16)            download_hf_file "Ternary-Bonsai-2.5-27B-mmproj-BF16.gguf" ;;
    5|stack|all|full)
        download_hf_file "Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf"
        download_hf_file "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"
        ;;
    6|stack-bf16)
        download_hf_file "Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf"
        download_hf_file "Ternary-Bonsai-2.5-27B-mmproj-BF16.gguf"
        ;;
    7|ptq1_0-mtp|PTQ1_0-mtp)  download_hf_file "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf" ;;
    8|ptq1_0|PTQ1_0)          download_hf_file "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf" ;;
    9|pq2_0-mtp|PQ2_0-mtp)    download_hf_file "Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf" ;;
    10|pq2_0|PQ2_0)           download_hf_file "Ternary-Bonsai-2.5-27B-PQ2_0.gguf" ;;
    -h|--help)
        echo -e "Usage: $(basename "$0") [1..10|q2_0-mtp|q2_0|mmproj|mmproj-bf16|stack|stack-bf16|ptq1_0-mtp|ptq1_0|pq2_0-mtp|pq2_0]"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: ${CHOICE}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}${BOLD}Download completed!${NC}"
echo -e "Launch with: ${YELLOW}./start-ternary-bonsai-2.5-27b.sh${NC}"
