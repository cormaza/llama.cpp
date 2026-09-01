#!/usr/bin/env bash
# Helper script to run llama.cpp on Tesla K80 (dual-GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/build-k80/bin"

if [[ ! -d "${BIN_DIR}" ]]; then
    echo "Error: Binaries not found in ${BIN_DIR}. Run ./scripts/build-k80.sh first."
    exit 1
fi

show_usage() {
    cat << EOF
Tesla K80 Runner (Optimized for 2x GK210, 24GB total VRAM)
Usage:
  ./run-k80.sh cli -m <path_to_model.gguf> -p "Prompt here" [extra args...]
  ./run-k80.sh server -m <path_to_model.gguf> --port 8080 [extra args...]
  ./run-k80.sh tune-gpu   (Sets GPU persistence mode, maximum boost clocks and power limit)

Default flags applied for Tesla K80:
  -ngl 99                (Offload all layers to GPUs)
  --split-mode layer     (Pipeline parallelism across GPU 0 and GPU 1; minimizes PCIe 3.0 traffic)
  --tensor-split 0.5,0.5 (Equal layer distribution between the two 12GB dies)
  -ctk q8_0 -ctv q8_0    (Quantized KV cache: halves VRAM bandwidth consumption)
  -fa                    (FlashAttention tile kernels)

Acceleration techniques available:
  1. Speculative Decoding (DSpark / DFlash):
     ./run-k80.sh cli -m target.gguf -md draft_dspark.gguf --spec-type draft-dspark -p "Prompt"
  2. N-Gram / Prompt Lookup (0 extra VRAM, great for code/chat):
     ./run-k80.sh cli -m target.gguf --spec-type ngram-simple --spec-ngram-simple-size-m 48 -p "Prompt"
EOF
}

MODE="${1:-}"
if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
    show_usage
    exit 0
fi
shift

if [[ "${MODE}" == "tune-gpu" ]]; then
    echo "Applying Tesla K80 performance tuning (requires sudo)..."
    sudo nvidia-smi -pm 1
    sudo nvidia-smi -ac 2505,875 || true
    sudo nvidia-smi -pl 149 || true
    echo "Tesla K80 clocks and power limit tuned successfully."
    exit 0
fi

# Optimized base flags for Tesla K80
K80_BASE_ARGS=(
    "-ngl" "99"
    "--split-mode" "layer"
    "--tensor-split" "0.5,0.5"
    "-ctk" "q8_0"
    "-ctv" "q8_0"
    "-fa" "auto"
)

case "${MODE}" in
    cli|llama-cli)
        exec "${BIN_DIR}/llama-cli" "${K80_BASE_ARGS[@]}" "$@"
        ;;
    server|llama-server)
        exec "${BIN_DIR}/llama-server" "${K80_BASE_ARGS[@]}" "$@"
        ;;
    *)
        exec "${BIN_DIR}/${MODE}" "$@"
        ;;
esac
