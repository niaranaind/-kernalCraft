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
echo "--- Step 1: Checking Compiler & CUDA Environment ---"
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found in PATH." >&2
    exit 1
fi
echo "Host Compiler: $(which g++ 2>/dev/null)"
echo "Host Compiler Version: $(g++ --version 2>&1 | head -n 1)"

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH." >&2
    echo "Please ensure CUDA is loaded (e.g. module load cuda or export PATH=/usr/local/cuda/bin:\$PATH)" >&2
    exit 1
fi

if ! command -v ptxas &> /dev/null; then
    echo "ERROR: ptxas not found in PATH." >&2
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
mkdir -p "${BUILD_DIR}"

CUDA_BIN_DIR="$(dirname "$(which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")")"
CUDA_DIR="$(dirname "${CUDA_BIN_DIR}")"
echo "  Detected CUDA directory: ${CUDA_DIR}"

echo "  Compiling miniptxc with g++ (-static-libstdc++ -static-libgcc)..."
if ! g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic \
    -static-libstdc++ -static-libgcc \
    -I"${ROOT_DIR}/include" \
    "${ROOT_DIR}/src/lexer.cpp" \
    "${ROOT_DIR}/src/parser.cpp" \
    "${ROOT_DIR}/src/ast.cpp" \
    "${ROOT_DIR}/src/ir.cpp" \
    "${ROOT_DIR}/src/ptx_emitter.cpp" \
    "${ROOT_DIR}/src/main.cpp" \
    -o "${BUILD_DIR}/miniptxc"; then
    echo "ERROR: Compilation of miniptxc failed in Step 2." >&2
    exit 1
fi

echo "  Compiling run_ptx with g++ (-static-libstdc++ -static-libgcc)..."
if ! g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic \
    -static-libstdc++ -static-libgcc \
    -I"${ROOT_DIR}/include" \
    -I"${CUDA_DIR}/include" \
    "${ROOT_DIR}/runtime/run_ptx.cpp" \
    -L"${CUDA_DIR}/lib64" -L"${CUDA_DIR}/lib64/stubs" -L/usr/lib/x86_64-linux-gnu \
    -lcuda \
    -o "${BUILD_DIR}/run_ptx"; then
    echo "ERROR: Compilation of run_ptx failed in Step 2." >&2
    exit 1
fi

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
    if ! "${BUILD_DIR}/miniptxc" "${SRC}" > "${DST}"; then
        echo "ERROR: miniptxc failed to compile ${k}.mk in Step 3." >&2
        exit 1
    fi
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
        echo "ERROR: ptxas verification failed for ${k}.ptx in Step 4." >&2
        exit 1
    fi
done
echo "All ${PTXAS_PASSED}/${#KERNELS[@]} kernels successfully verified by ptxas."
echo

# -------------------------------------------------------------------
# 4b. Verify DWARF Line Info (.loc / .file) with ptxas -lineinfo
# -------------------------------------------------------------------
echo "--- Step 4b: Verifying Debug Line Info (ptxas -lineinfo) ---"
echo "  Assembling saxpy.ptx with -lineinfo -> saxpy.cubin..."
if ! ptxas -lineinfo -arch="${ARCH}" -c "${BUILD_DIR}/saxpy.ptx" -o "${BUILD_DIR}/saxpy.cubin"; then
    echo "ERROR: ptxas -lineinfo assembly failed for saxpy.ptx in Step 4b." >&2
    exit 1
fi
echo "  ELF debug sections generated by ptxas (cuobjdump -elf | grep -i debug):"
cuobjdump -elf "${BUILD_DIR}/saxpy.cubin" | grep -i debug || echo "  (no debug sections matched)"
echo
echo "  Tip: To inspect disassembled SASS with interleaved source lines, run:"
echo "    nvdisasm -c -g ${BUILD_DIR}/saxpy.cubin | head -n 60"
echo

# -------------------------------------------------------------------
# 5. Run GPU Execution & CPU Reference Check via CUDA Driver API
# -------------------------------------------------------------------
echo "--- Step 5: Executing on GPU via CUDA Driver API (run_ptx) ---"
if ! "${BUILD_DIR}/run_ptx" "${BUILD_DIR}"; then
    echo "ERROR: GPU execution verification failed in Step 5." >&2
    exit 1
fi

echo
echo "================================================================="
echo " DGX Validation Complete: All 5 kernels verified on GPU!        "
echo "================================================================="
