#!/usr/bin/env bash
#
# MiniPTX DGX Execution & Validation Script
#
# Target: NVIDIA Tesla V100 (sm_70) on DGX Server
# Prerequisites on DGX: CUDA Toolkit (nvcc, ptxas, libcuda.so), CMake 3.16+, C++17 compiler
#
# Usage:
#   bash scripts/run_dgx.sh
#

set -e

ARCH="sm_70"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_dgx"

echo "================================================================="
echo " MiniPTX DGX Validation Suite (NVIDIA Tesla V100, sm_70)         "
echo "================================================================="
echo

# -------------------------------------------------------------------
# 1. Environment & CUDA Version Check
# -------------------------------------------------------------------
echo "--- Step 1: Checking CUDA Environment ---"
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH."
    echo "Please ensure CUDA is loaded (e.g. module load cuda or export PATH=/usr/local/cuda/bin:\$PATH)"
    exit 1
fi

if ! command -v ptxas &> /dev/null; then
    echo "ERROR: ptxas not found in PATH."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9\.]*\).*/\1/p')
echo "Detected CUDA Toolkit Version: ${CUDA_VERSION}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Device Information:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true
fi
echo

# -------------------------------------------------------------------
# 2. Build MiniPTX Compiler & run_ptx Driver
# -------------------------------------------------------------------
echo "--- Step 2: Building MiniPTX (Compiler & CUDA Driver Runtime) ---"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 4)"
echo "Build complete."
echo

# -------------------------------------------------------------------
# 3. Compile all 5 kernels to PTX using miniptxc
# -------------------------------------------------------------------
echo "--- Step 3: Compiling Kernels with MiniPTX Compiler ---"
KERNELS=("vecadd" "saxpy" "scale" "ewmax" "fill")

for k in "${KERNELS[@]}"; do
    SRC="${ROOT_DIR}/examples/${k}.mk"
    DST="${BUILD_DIR}/${k}.ptx"
    echo "  Compiling ${k}.mk -> ${k}.ptx..."
    "${BUILD_DIR}/miniptxc" "${SRC}" > "${DST}"
done
echo "All 5 kernels compiled to PTX."
echo

# -------------------------------------------------------------------
# 4. Verify PTX with ptxas
# -------------------------------------------------------------------
echo "--- Step 4: Assembling PTX with ptxas (-arch=${ARCH}) ---"
PTXAS_PASSED=0
for k in "${KERNELS[@]}"; do
    PTX_FILE="${BUILD_DIR}/${k}.ptx"
    echo -n "  ptxas ${k}.ptx: "
    if ptxas -arch="${ARCH}" -c "${PTX_FILE}" -o /dev/null 2>&1; then
        echo "[OK]"
        PTXAS_PASSED=$((PTXAS_PASSED + 1))
    else
        echo "[FAILED]"
        exit 1
    fi
done
echo "All ${PTXAS_PASSED}/${#KERNELS[@]} kernels successfully verified by ptxas."
echo

# -------------------------------------------------------------------
# 5. Run GPU Execution & CPU Reference Check via CUDA Driver API
# -------------------------------------------------------------------
echo "--- Step 5: Executing on GPU via CUDA Driver API (run_ptx) ---"
"${BUILD_DIR}/run_ptx" "${BUILD_DIR}"

echo
echo "================================================================="
echo " DGX Validation Complete: All 5 kernels verified on GPU!        "
echo "================================================================="
