#!/usr/bin/env bash
# ==============================================================================
# build-k80.sh - Build llama.cpp for NVIDIA Tesla K80 (sm_37 / Kepler) on Omarchy
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-k80"
CUDA_TARGET_DIR="${HOME}/.local/cuda-11.4"
CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run"
CUDA_RUNFILE_NAME="cuda_11.4.4_470.82.01_linux.run"

CUSTOM_CUDA_PATH=""
CLEAN_BUILD=0
AUTO_DOWNLOAD=0
FORCE_CUBLAS=0
JOBS="$(nproc 2>/dev/null || echo 4)"

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()    { echo -e "\n${BOLD}${BLUE}==> $*${NC}"; }

show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Builds llama.cpp with CUDA support for NVIDIA Tesla K80 (sm_37) on Omarchy / Arch Linux.

Options:
  -c, --clean             Clean previous build directory before building
  -p, --cuda-path PATH    Specify custom CUDA Toolkit directory (containing bin/nvcc)
  -d, --download-cuda     Automatically download and install standalone CUDA 11.4 toolkit to ~/.local/cuda-11.4
  -j, --jobs N            Number of parallel compilation jobs (default: $(nproc 2>/dev/null || echo 4))
  --force-cublas          Force using cuBLAS kernels
  -h, --help              Show this help message

Examples:
  ./scripts/build-k80.sh
  ./scripts/build-k80.sh --download-cuda
  ./scripts/build-k80.sh --cuda-path /opt/cuda-11.4
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        -p|--cuda-path)
            CUSTOM_CUDA_PATH="$2"
            shift 2
            ;;
        -d|--download-cuda)
            AUTO_DOWNLOAD=1
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --force-cublas)
            FORCE_CUBLAS=1
            shift
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

log_step "1. Checking Hardware & GPU Driver"

if command -v nvidia-smi &>/dev/null; then
    DRIVER_INFO=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 || true)
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader || true)
    log_success "NVIDIA driver detected: ${DRIVER_INFO}"
    echo -e "   Detected GPU(s):\n${GPU_NAMES}" | sed 's/^/   /'
    
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "${GPU_COUNT}" -ge 2 ]]; then
        log_info "Detected dual-GPU configuration (${GPU_COUNT} GPUs). Multi-GPU tensor splitting will be available."
    fi
else
    log_warn "nvidia-smi not found in PATH. Ensure NVIDIA driver (nvidia-470xx) is properly installed."
fi

log_step "2. Checking CUDA 11.x Toolkit"

find_cuda_toolkit() {
    local candidate_paths=(
        "${CUSTOM_CUDA_PATH}"
        "${CUDA_PATH:-}"
        "${CUDA_HOME:-}"
        "${CUDA_TARGET_DIR}"
        "/opt/cuda-11.4"
        "/opt/cuda-11.8"
        "/opt/cuda-11.7"
        "/opt/cuda-11.2"
        "/opt/cuda-11.1"
        "/opt/cuda-11.0"
        "/opt/cuda"
        "/usr/local/cuda-11.4"
        "/usr/local/cuda-11.8"
        "/usr/local/cuda-11.7"
        "/usr/local/cuda"
    )

    for path in "${candidate_paths[@]}"; do
        if [[ -n "${path}" && -x "${path}/bin/nvcc" ]]; then
            local ver
            ver=$("${path}/bin/nvcc" --version | grep -o "release [0-9]\+\.[0-9]\+" | cut -d' ' -f2 || true)
            local major
            major=$(echo "${ver}" | cut -d'.' -f1)
            if [[ "${major}" == "11" || "${major}" == "10" ]]; then
                echo "${path}"
                return 0
            fi
        fi
    done

    # Check PATH
    if command -v nvcc &>/dev/null; then
        local nvcc_path
        nvcc_path=$(dirname "$(dirname "$(command -v nvcc)")")
        local ver
        ver=$(nvcc --version | grep -o "release [0-9]\+\.[0-9]\+" | cut -d' ' -f2 || true)
        local major
        major=$(echo "${ver}" | cut -d'.' -f1)
        if [[ "${major}" == "11" || "${major}" == "10" ]]; then
            echo "${nvcc_path}"
            return 0
        fi
    fi

    return 1
}

CUDA_DIR=""
if CUDA_DIR=$(find_cuda_toolkit); then
    NVCC_BIN="${CUDA_DIR}/bin/nvcc"
    CUDA_VER=$("${NVCC_BIN}" --version | grep -o "release [0-9]\+\.[0-9]\+" | cut -d' ' -f2)
    log_success "Found compatible CUDA Toolkit ${CUDA_VER} at: ${CUDA_DIR}"
else
    log_warn "No CUDA 10.x/11.x Toolkit found."
    log_info "Kepler architecture (sm_37 / Tesla K80) requires CUDA 11.x (CUDA 12+ dropped Kepler support)."

    if [[ "${AUTO_DOWNLOAD}" -eq 1 ]]; then
        INSTALL_CHOICE="1"
    else
        echo -e "\nChoose an option to install CUDA 11.4 Toolkit:"
        echo "  [1] Automatically download and install standalone CUDA 11.4 toolkit to ${CUDA_TARGET_DIR} (No root required)"
        echo "  [2] Install CUDA 11.7 from AUR using yay (yay -S cuda-11.7)"
        echo "  [3] Specify an existing CUDA directory"
        echo "  [4] Exit"
        read -rp "Enter choice [1-4] (default 1): " INSTALL_CHOICE
        INSTALL_CHOICE="${INSTALL_CHOICE:-1}"
    fi

    case "${INSTALL_CHOICE}" in
        1)
            log_step "Downloading and installing CUDA 11.4.4 Toolkit..."
            DOWNLOAD_TMP="${HOME}/.cache/cuda_k80_install_tmp"
            mkdir -p "${DOWNLOAD_TMP}"
            mkdir -p "${CUDA_TARGET_DIR}"

            RUNFILE="${DOWNLOAD_TMP}/${CUDA_RUNFILE_NAME}"
            if [[ ! -s "${RUNFILE}" ]]; then
                log_info "Downloading CUDA 11.4.4 runfile from NVIDIA (~3.8GB)..."
                curl -L -C - "${CUDA_RUNFILE_URL}" -o "${RUNFILE}" --progress-bar
            else
                log_info "Found cached runfile at ${RUNFILE}"
            fi

            if [[ ! -s "${RUNFILE}" ]]; then
                log_error "Failed to download CUDA runfile or file is empty."
                exit 1
            fi

            chmod +x "${RUNFILE}"
            log_info "Extracting toolkit components into ${CUDA_TARGET_DIR}..."
            "${RUNFILE}" --silent --toolkit --toolkitpath="${CUDA_TARGET_DIR}" --override

            CUDA_DIR="${CUDA_TARGET_DIR}"
            NVCC_BIN="${CUDA_DIR}/bin/nvcc"
            if [[ ! -x "${NVCC_BIN}" ]]; then
                log_error "Installation failed: nvcc not found at ${NVCC_BIN}"
                exit 1
            fi
            log_success "CUDA 11.4 Toolkit installed successfully at ${CUDA_DIR}"
            ;;
        2)
            log_info "Running: yay -S --needed cuda-11.7 gcc11"
            yay -S --needed cuda-11.7 gcc11
            CUDA_DIR="/opt/cuda-11.7"
            NVCC_BIN="${CUDA_DIR}/bin/nvcc"
            ;;
        3)
            read -rp "Enter path to CUDA directory: " USER_CUDA_PATH
            if [[ -x "${USER_CUDA_PATH}/bin/nvcc" ]]; then
                CUDA_DIR="${USER_CUDA_PATH}"
                NVCC_BIN="${CUDA_DIR}/bin/nvcc"
            else
                log_error "nvcc not found in ${USER_CUDA_PATH}/bin/nvcc"
                exit 1
            fi
            ;;
        *)
            log_info "Aborting setup."
            exit 0
            ;;
    esac
fi

log_step "3. Checking Host C++ Compiler for NVCC"

find_host_compiler() {
    local compilers=("gcc-11" "gcc-10" "gcc11" "gcc10" "/usr/bin/gcc-11" "/usr/bin/gcc-10" "/usr/bin/gcc11" "/usr/bin/gcc10")
    for cmp in "${compilers[@]}"; do
        if command -v "${cmp}" &>/dev/null; then
            echo "$(command -v "${cmp}")"
            return 0
        fi
    done
    return 1
}

HOST_COMPILER=""
HOST_CMAKE_ARGS=()
if HOST_COMPILER=$(find_host_compiler); then
    log_success "Found compatible host compiler for nvcc: ${HOST_COMPILER}"
    HOST_CMAKE_ARGS+=("-DCMAKE_CUDA_HOST_COMPILER=${HOST_COMPILER}")
else
    log_info "No separate gcc-11/gcc-10 found. Will test system default host compiler."
fi

log_step "4. Configuring CMake"

if [[ "${CLEAN_BUILD}" -eq 1 && -d "${BUILD_DIR}" ]]; then
    log_info "Cleaning previous build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

CMAKE_FLAGS=(
    "-B" "${BUILD_DIR}"
    "-S" "${ROOT_DIR}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DGGML_CUDA=ON"
    "-DGGML_CUDA_K80=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=37"
    "-DCMAKE_CUDA_COMPILER=${NVCC_BIN}"
    "-DCUDAToolkit_ROOT=${CUDA_DIR}"
    "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
    "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128"
)

if [[ "${FORCE_CUBLAS}" -eq 1 ]]; then
    CMAKE_FLAGS+=("-DGGML_CUDA_FORCE_CUBLAS=ON")
fi

if [[ ${#HOST_CMAKE_ARGS[@]} -gt 0 ]]; then
    CMAKE_FLAGS+=("${HOST_CMAKE_ARGS[@]}")
fi

log_info "Running CMake with flags:"
for flag in "${CMAKE_FLAGS[@]}"; do
    echo "   ${flag}"
done

cmake "${CMAKE_FLAGS[@]}"

log_step "5. Compiling llama.cpp binaries"
log_info "Building with ${JOBS} parallel threads..."

cmake --build "${BUILD_DIR}" --config Release -j"${JOBS}"

log_step "6. Generating K80 Dual-GPU Helper Script"

RUN_SCRIPT="${ROOT_DIR}/run-k80.sh"
cat << 'RUN_EOF' > "${RUN_SCRIPT}"
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
    "-fa"
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
RUN_EOF

chmod +x "${RUN_SCRIPT}"

echo -e "\n${GREEN}${BOLD}======================================================${NC}"
echo -e "${GREEN}${BOLD}  Tesla K80 build completed successfully!            ${NC}"
echo -e "${GREEN}${BOLD}======================================================${NC}"
echo -e "Binaries located in: ${CYAN}${BUILD_DIR}/bin/${NC}"
echo -e "Helper script created: ${CYAN}${RUN_SCRIPT}${NC}"
echo -e "\nUsage examples for dual Tesla K80 (24GB VRAM total):"
echo -e "  ${YELLOW}./run-k80.sh cli -m /path/to/model.gguf -p \"Hola mundo\"${NC}"
echo -e "  ${YELLOW}./run-k80.sh server -m /path/to/model.gguf --host 0.0.0.0 --port 8080${NC}"
echo ""
