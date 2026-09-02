#!/usr/bin/env bash
# Helper script to run llama.cpp on AMD Radeon GPUs (ROCm / HIP)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/build-amd/bin"

if [[ ! -d "${BIN_DIR}" ]]; then
    echo "Error: Binaries not found in ${BIN_DIR}. Run ./scripts/build-amd-rocm.sh first."
    exit 1
fi

show_usage() {
    cat << EOF
AMD Radeon ROCm Runner
Usage:
  ./run-amd.sh cli -m <path_to_model.gguf> -p "Prompt here" [extra args...]
  ./run-amd.sh server -m <path_to_model.gguf> --port 8080 [extra args...]
  ./run-amd.sh bench -m <path_to_model.gguf> [extra args...]
  ./run-amd.sh tune-gpu   (Sets performance profile if rocm-smi is available)

Default flags applied for AMD Radeon GPU:
  -ngl 99                (Offload all layers to AMD GPU via ROCm)
  -fa auto               (FlashAttention kernel acceleration)
  -t 8                   (Optimal thread count for hybrid CPU P-cores)

Optimization tips:
  1. Quantized KV Cache (saves VRAM, boosts generation speed):
     ./run-amd.sh cli -m model.gguf -ctk q8_0 -ctv q8_0 -p "Prompt"
  2. Speculative Decoding (N-Gram / Prompt Lookup):
     ./run-amd.sh cli -m model.gguf --spec-type ngram-simple --spec-ngram-simple-size-m 48 -p "Prompt"
  3. Server Mode:
     ./run-amd.sh server -m model.gguf --host 0.0.0.0 --port 8080 -ngl 99 -fa auto
EOF
}

MODE="${1:-}"
if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
    show_usage
    exit 0
fi
shift

if [[ "${MODE}" == "tune-gpu" ]]; then
    echo "Checking AMD GPU power profile..."
    if command -v rocm-smi &>/dev/null; then
        rocm-smi --showprofile || true
        echo "To set performance profile: sudo rocm-smi --setperflevel high"
    else
        echo "rocm-smi utility not found."
    fi
    exit 0
fi

# Optimized base flags for AMD Radeon GPU
AMD_BASE_ARGS=(
    "-ngl" "99"
    "-fa" "auto"
    "-t" "8"
    "--temp" "0.7"
    "--repeat-penalty" "1.1"
    "--dry-multiplier" "0.8"
    "--dry-base" "1.75"
    "--dry-allowed-length" "2"
    "--dry-penalty-last-n" "256"
)

case "${MODE}" in
    cli|llama-cli)
        exec "${BIN_DIR}/llama-cli" "${AMD_BASE_ARGS[@]}" "$@"
        ;;
    server|llama-server)
        exec "${BIN_DIR}/llama-server" "${AMD_BASE_ARGS[@]}" "$@"
        ;;
    bench|llama-bench)
        exec "${BIN_DIR}/llama-bench" "${AMD_BASE_ARGS[@]}" "$@"
        ;;
    *)
        exec "${BIN_DIR}/${MODE}" "$@"
        ;;
esac
