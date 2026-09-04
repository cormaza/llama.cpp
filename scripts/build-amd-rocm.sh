#!/usr/bin/env bash
# ==============================================================================
# build-amd-rocm.sh - Build llama.cpp for AMD Radeon GPUs (ROCm / HIP)
# ==============================================================================

set -euo pipefail

RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
BLUE='[0;34m'
CYAN='[0;36m'
BOLD='[1m'
NC='[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-amd"

ROCM_DIR="${ROCM_PATH:-/opt/rocm}"
CUSTOM_TARGET=""
CLEAN_BUILD=0
AUTO_INSTALL=0
JOBS="$(nproc 2>/dev/null || echo 4)"

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()    { echo -e "
${BOLD}${BLUE}==> $*${NC}"; }

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Builds llama.cpp with AMD ROCm/HIP acceleration on Omarchy / Arch / Linux.

Options:
  -c, --clean             Clean previous build directory before building
  -t, --target ARCH       Specify AMD GPU target architecture (e.g. gfx1200, gfx1100)
  -p, --rocm-path PATH    Specify custom ROCm root directory (default: /opt/rocm)
  -i, --install-deps      Automatically install missing ROCm packages via package manager
  -j, --jobs N            Number of parallel compilation jobs (default: $(nproc 2>/dev/null || echo 4))
  -h, --help              Show this help message

Examples:
  ./scripts/build-amd-rocm.sh
  ./scripts/build-amd-rocm.sh --clean
  ./scripts/build-amd-rocm.sh --target gfx1200
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        -t|--target)
            CUSTOM_TARGET="$2"
            shift 2
            ;;
        -p|--rocm-path)
            ROCM_DIR="$2"
            shift 2
            ;;
        -i|--install-deps)
            AUTO_INSTALL=1
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BOLD}${CYAN}======================================================${NC}"
echo -e "${BOLD}${CYAN}  llama.cpp - AMD ROCm / HIP Optimized Build Setup    ${NC}"
echo -e "${BOLD}${CYAN}======================================================${NC}"

log_step "1. Detecting System Hardware & CPU Optimizations"

CPU_MODEL="$(lscpu 2>/dev/null | grep "Model name:" | sed 's/Model name:[ 	]*//' || echo "Unknown CPU")"
log_info "CPU Detected: ${CPU_MODEL}"

CPU_FLAGS="$(lscpu 2>/dev/null | grep "Flags:" || true)"
OPTIMIZATIONS=()
if echo "${CPU_FLAGS}" | grep -qw "avx2"; then
    OPTIMIZATIONS+=("AVX2")
fi
if echo "${CPU_FLAGS}" | grep -qw "fma"; then
    OPTIMIZATIONS+=("FMA")
fi
if echo "${CPU_FLAGS}" | grep -qw "f16c"; then
    OPTIMIZATIONS+=("F16C")
fi
if echo "${CPU_FLAGS}" | grep -qw "avx_vnni"; then
    OPTIMIZATIONS+=("AVX_VNNI")
fi

log_success "CPU Vector Extensions: ${OPTIMIZATIONS[*]:-None} (Native host vectorization enabled)"

log_step "2. Checking ROCm / HIP Environment"

check_rocm_ready() {
    if [[ -d "${ROCM_DIR}" ]] && [[ -x "${ROCM_DIR}/bin/hipcc" || -x "${ROCM_DIR}/lib/llvm/bin/clang++" ]]; then
        return 0
    fi
    return 1
}

if check_rocm_ready; then
    log_success "ROCm toolchain found at: ${ROCM_DIR}"
else
    log_warn "ROCm toolchain not found at: ${ROCM_DIR}"

    if command -v yay &>/dev/null; then
        PKG_CMD="yay -S --needed"
    elif command -v pacman &>/dev/null; then
        PKG_CMD="sudo pacman -S --needed"
    else
        PKG_CMD=""
    fi

    ROCM_PKGS="rocm-hip-runtime rocm-llvm rocblas hipblas rocminfo rocm-cmake"

    if [[ "${AUTO_INSTALL}" -eq 1 && -n "${PKG_CMD}" ]]; then
        INSTALL_CHOICE="1"
    else
        echo -e "
Choose an option to configure ROCm dependencies:"
        echo "  [1] Install ROCm packages via package manager (${PKG_CMD:-pacman} ${ROCM_PKGS})"
        echo "  [2] Specify a custom ROCm directory"
        echo "  [3] Exit"
        read -rp "Enter choice [1-3] (default 1): " INSTALL_CHOICE
        INSTALL_CHOICE="${INSTALL_CHOICE:-1}"
    fi

    case "${INSTALL_CHOICE}" in
        1)
            if [[ -z "${PKG_CMD}" ]]; then
                log_error "No supported package manager found (pacman/yay). Please install ROCm manually."
                exit 1
            fi
            log_info "Installing ROCm packages: ${ROCM_PKGS}..."
            ${PKG_CMD} ${ROCM_PKGS}
            ROCM_DIR="/opt/rocm"
            ;;
        2)
            read -rp "Enter path to ROCm directory: " USER_ROCM_PATH
            ROCM_DIR="${USER_ROCM_PATH}"
            ;;
        *)
            log_info "Aborting setup."
            exit 0
            ;;
    esac
fi

if ! check_rocm_ready; then
    log_error "ROCm toolchain validation failed in ${ROCM_DIR}."
    exit 1
fi

log_step "3. Detecting AMD GPU Target Architecture"

TARGET_ARCH=""
if [[ -n "${CUSTOM_TARGET}" ]]; then
    TARGET_ARCH="${CUSTOM_TARGET}"
    log_info "Using user-specified GPU target architecture: ${TARGET_ARCH}"
elif [[ -x "${ROCM_DIR}/bin/rocminfo" ]] || command -v rocminfo &>/dev/null; then
    ROCMINFO_BIN="${ROCM_DIR}/bin/rocminfo"
    if ! [[ -x "${ROCMINFO_BIN}" ]]; then
        ROCMINFO_BIN="$(command -v rocminfo)"
    fi

    DETECTED_ARCH="$(${ROCMINFO_BIN} 2>/dev/null | grep -E "amdgcn-amd-amdhsa--gfx" | head -n1 | sed -e 's/.*amdgcn-amd-amdhsa--//' | tr -d ' ' || true)"
    GPU_NAME="$(${ROCMINFO_BIN} 2>/dev/null | grep -A2 "amdgcn-amd-amdhsa--" | grep "Marketing Name:" | head -n1 | sed 's/Marketing Name:[ 	]*//' || true)"

    if [[ -n "${DETECTED_ARCH}" ]]; then
        TARGET_ARCH="${DETECTED_ARCH}"
        log_success "Detected AMD GPU: ${GPU_NAME:-AMD GPU} (${TARGET_ARCH})"
    fi
fi

if [[ -z "${TARGET_ARCH}" ]]; then
    log_warn "Could not auto-detect GPU target architecture via rocminfo."
    read -rp "Enter AMD GPU target architecture (e.g. gfx1200, gfx1100, gfx1030): " TARGET_ARCH
    if [[ -z "${TARGET_ARCH}" ]]; then
        log_error "No GPU architecture specified."
        exit 1
    fi
fi

log_step "4. Configuring CMake for ROCm / HIP"

if [[ "${CLEAN_BUILD}" -eq 1 && -d "${BUILD_DIR}" ]]; then
    log_info "Cleaning previous build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

CMAKE_FLAGS=(
    "-B" "${BUILD_DIR}"
    "-S" "${ROOT_DIR}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DGGML_HIP=ON"
    "-DAMDGPU_TARGETS=${TARGET_ARCH}"
    "-DCMAKE_HIP_ARCHITECTURES=${TARGET_ARCH}"
    "-DGGML_HIP_GRAPHS=ON"
    "-DROCM_PATH=${ROCM_DIR}"
)

log_info "Running CMake with flags:"
for flag in "${CMAKE_FLAGS[@]}"; do
    echo "   ${flag}"
done

cmake "${CMAKE_FLAGS[@]}"

log_step "5. Compiling llama.cpp Binaries"
log_info "Building with ${JOBS} parallel threads..."

cmake --build "${BUILD_DIR}" --config Release -j"${JOBS}"

log_step "6. Generating AMD GPU Helper Runner Script"

RUN_SCRIPT="${ROOT_DIR}/run-amd.sh"
cat << 'RUN_EOF' > "${RUN_SCRIPT}"
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
    "--temp" "0.2"
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
RUN_EOF

chmod +x "${RUN_SCRIPT}"

echo -e "
${GREEN}${BOLD}======================================================${NC}"
echo -e "${GREEN}${BOLD}  AMD ROCm build completed successfully!             ${NC}"
echo -e "${GREEN}${BOLD}======================================================${NC}"
echo -e "Target GPU Architecture: ${CYAN}${TARGET_ARCH}${NC}"
echo -e "Binaries located in: ${CYAN}${BUILD_DIR}/bin/${NC}"
echo -e "Helper script created: ${CYAN}${RUN_SCRIPT}${NC}"
echo -e "
Usage examples:"
echo -e "  ${YELLOW}./run-amd.sh cli -m /path/to/model.gguf -p "Hola mundo"${NC}"
echo -e "  ${YELLOW}./run-amd.sh server -m /path/to/model.gguf --host 0.0.0.0 --port 8080${NC}"
echo ""
