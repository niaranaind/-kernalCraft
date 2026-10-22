#!/usr/bin/env bash
#
# MiniPTX vs Hand-Written CUDA Benchmark Runner
#
# Target: Tesla V100 (sm_70) on DGX Server
# Workload: SAXPY on N = 2^24 elements (128 MiB footprint, 192 MiB traffic)
#
# Usage:
#   bash scripts/benchmark_dgx.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_dgx"
ARCH="sm_70"

echo "================================================================="
echo " MiniPTX Micro-Benchmark Suite: MiniPTX vs. CUDA (Tesla V100)    "
echo "================================================================="
echo

# 1. Check tools
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH." >&2
    exit 1
fi
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found in PATH." >&2
    exit 1
fi

CUDA_BIN_DIR="$(dirname "$(which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")")"
CUDA_DIR="$(dirname "${CUDA_BIN_DIR}")"
mkdir -p "${BUILD_DIR}"

# 2. Build MiniPTX compiler if not already built
if [ ! -f "${BUILD_DIR}/miniptxc" ]; then
    echo "--- Building miniptxc ---"
    g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic \
        -static-libstdc++ -static-libgcc \
        -I"${ROOT_DIR}/include" \
        "${ROOT_DIR}/src/lexer.cpp" \
        "${ROOT_DIR}/src/parser.cpp" \
        "${ROOT_DIR}/src/ast.cpp" \
        "${ROOT_DIR}/src/ir.cpp" \
        "${ROOT_DIR}/src/ptx_emitter.cpp" \
        "${ROOT_DIR}/src/main.cpp" \
        -o "${BUILD_DIR}/miniptxc"
fi

# 3. Build benchmark executable with CUDA Driver API
echo "--- Compiling pure CUDA Driver API benchmark harness ---"
g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic \
    -static-libstdc++ -static-libgcc \
    -I"${ROOT_DIR}/include" \
    -I"${CUDA_DIR}/include" \
    "${ROOT_DIR}/runtime/benchmark.cpp" \
    -L"${CUDA_DIR}/lib64" -L"${CUDA_DIR}/lib64/stubs" -L/usr/lib/x86_64-linux-gnu \
    -lcuda \
    -o "${BUILD_DIR}/benchmark"

# 4. Generate PTX for both kernels
echo "--- Generating PTX for MiniPTX SAXPY ---"
"${BUILD_DIR}/miniptxc" "${ROOT_DIR}/examples/saxpy.mk" > "${BUILD_DIR}/miniptx_saxpy.ptx"

echo "--- Compiling hand-written CUDA SAXPY to PTX (nvcc -O3 -arch=${ARCH}) ---"
nvcc -ptx -arch="${ARCH}" -O3 "${ROOT_DIR}/examples/cuda_saxpy.cu" -o "${BUILD_DIR}/cuda_saxpy.ptx"

# 5. Run Benchmark on GPU 0
echo "--- Executing Benchmark on GPU (CUDA_VISIBLE_DEVICES=0) ---"
export CUDA_VISIBLE_DEVICES=0
"${BUILD_DIR}/benchmark" "${BUILD_DIR}/miniptx_saxpy.ptx" "${BUILD_DIR}/cuda_saxpy.ptx"
