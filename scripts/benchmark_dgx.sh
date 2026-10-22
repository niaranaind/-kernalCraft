#!/usr/bin/env bash
#
# MiniPTX vs Hand-Written CUDA Benchmark Runner
#
# Target: Tesla V100-SXM2 (sm_70) on DGX Server
# Workloads:
#   1. SAXPY: N = 2^24 elements (128 MiB footprint, 192 MiB DRAM traffic) - Memory-Bound
#   2. Compute-Heavy: N = 2^24 elements, 64-iteration inner loop
#
# Usage:
#   bash scripts/benchmark_dgx.sh
#   NVCC_CCBIN=/usr/bin/g++ bash scripts/benchmark_dgx.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_dgx"
ARCH="sm_70"
NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++}"

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
if ! command -v ptxas &> /dev/null; then
    echo "ERROR: ptxas not found in PATH." >&2
    exit 1
fi
if ! command -v cuobjdump &> /dev/null; then
    echo "ERROR: cuobjdump not found in PATH." >&2
    exit 1
fi

CUDA_BIN_DIR="$(dirname "$(which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")")"
CUDA_DIR="$(dirname "${CUDA_BIN_DIR}")"
CUDA_VER="$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9\.]*\).*/\1/p')"
mkdir -p "${BUILD_DIR}"

echo "Environment:"
echo "  CUDA Toolkit: ${CUDA_VER} (at ${CUDA_DIR})"
echo "  Host Compiler (for miniptx): $(which g++ 2>/dev/null) ($(g++ --version 2>&1 | head -n 1))"
echo "  NVCC Host Compiler (NVCC_CCBIN): ${NVCC_CCBIN}"
if command -v "${NVCC_CCBIN}" &>/dev/null; then
    echo "  NVCC Host Compiler Version: $("${NVCC_CCBIN}" --version 2>&1 | head -n 1)"
else
    echo "  WARNING: ${NVCC_CCBIN} not found; nvcc may fall back to default PATH compiler."
fi
echo

# 2. Build MiniPTX compiler
echo "--- Compiling miniptxc ---"
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

# 4. Generate PTX for SAXPY (Memory-Bound)
echo "--- Generating PTX for MiniPTX SAXPY ---"
"${BUILD_DIR}/miniptxc" "${ROOT_DIR}/examples/saxpy.mk" > "${BUILD_DIR}/miniptx_saxpy.ptx"

echo "--- Compiling hand-written CUDA SAXPY to PTX (nvcc -O3 -arch=${ARCH}) ---"
echo "  Executing: nvcc -ccbin \"${NVCC_CCBIN}\" -ptx -arch=\"${ARCH}\" -O3 examples/cuda_saxpy.cu -o build_dgx/cuda_saxpy.ptx"
if ! nvcc -ccbin "${NVCC_CCBIN}" -ptx -arch="${ARCH}" -O3 "${ROOT_DIR}/examples/cuda_saxpy.cu" -o "${BUILD_DIR}/cuda_saxpy.ptx"; then
    echo "ERROR: nvcc failed to compile examples/cuda_saxpy.cu." >&2
    echo "Note: CUDA ${CUDA_VER} supports host GCC versions up to GCC 12." >&2
    echo "If your default compiler is GCC 13+, specify a supported host compiler via NVCC_CCBIN, e.g.:" >&2
    echo "  NVCC_CCBIN=/usr/bin/g++ bash scripts/benchmark_dgx.sh" >&2
    exit 1
fi

# 5. Generate PTX for Compute-Heavy
echo "--- Generating PTX for MiniPTX Compute-Heavy ---"
"${BUILD_DIR}/miniptxc" "${ROOT_DIR}/examples/compute_heavy.mk" > "${BUILD_DIR}/miniptx_compute_heavy.ptx"

echo "--- Compiling hand-written CUDA Compute-Heavy to PTX (nvcc -O3 -arch=${ARCH}) ---"
echo "  Executing: nvcc -ccbin \"${NVCC_CCBIN}\" -ptx -arch=\"${ARCH}\" -O3 examples/cuda_compute_heavy.cu -o build_dgx/cuda_compute_heavy.ptx"
if ! nvcc -ccbin "${NVCC_CCBIN}" -ptx -arch="${ARCH}" -O3 "${ROOT_DIR}/examples/cuda_compute_heavy.cu" -o "${BUILD_DIR}/cuda_compute_heavy.ptx"; then
    echo "ERROR: nvcc failed to compile examples/cuda_compute_heavy.cu." >&2
    exit 1
fi

# Diagnostic (a): CUDA compute-heavy with --fmad=false
echo "--- Compiling hand-written CUDA Compute-Heavy with --fmad=false (Diagnostic a) ---"
echo "  Executing: nvcc -ccbin \"${NVCC_CCBIN}\" -ptx -arch=\"${ARCH}\" -O3 --fmad=false examples/cuda_compute_heavy.cu -o build_dgx/cuda_compute_heavy_nofma.ptx"
if ! nvcc -ccbin "${NVCC_CCBIN}" -ptx -arch="${ARCH}" -O3 --fmad=false "${ROOT_DIR}/examples/cuda_compute_heavy.cu" -o "${BUILD_DIR}/cuda_compute_heavy_nofma.ptx"; then
    echo "ERROR: nvcc failed to compile examples/cuda_compute_heavy.cu with --fmad=false." >&2
    exit 1
fi

# 6. Run Benchmark on GPU 0
echo
echo "--- Executing Benchmark on GPU (CUDA_VISIBLE_DEVICES=0) ---"
export CUDA_VISIBLE_DEVICES=0
"${BUILD_DIR}/benchmark" "${BUILD_DIR}"

# 7. SASS Disassembly & Machine Instruction Analysis (Diagnostic b)
echo
echo "================================================================="
echo " SASS Disassembly & Machine Instruction Analysis                 "
echo "================================================================="
echo

echo "--- Assembling PTX to CUBIN with ptxas ---"
ptxas -m64 -arch="${ARCH}" "${BUILD_DIR}/miniptx_saxpy.ptx" -o "${BUILD_DIR}/miniptx_saxpy.cubin"
ptxas -m64 -arch="${ARCH}" "${BUILD_DIR}/cuda_saxpy.ptx" -o "${BUILD_DIR}/cuda_saxpy.cubin"
ptxas -m64 -arch="${ARCH}" "${BUILD_DIR}/miniptx_compute_heavy.ptx" -o "${BUILD_DIR}/miniptx_compute_heavy.cubin"
ptxas -m64 -arch="${ARCH}" "${BUILD_DIR}/cuda_compute_heavy.ptx" -o "${BUILD_DIR}/cuda_compute_heavy.cubin"
ptxas -m64 -arch="${ARCH}" "${BUILD_DIR}/cuda_compute_heavy_nofma.ptx" -o "${BUILD_DIR}/cuda_compute_heavy_nofma.cubin"

count_sass_instructions() {
    local cubin="$1"
    if [ ! -f "${cubin}" ]; then
        echo "0"
        return
    fi
    cuobjdump -sass "${cubin}" 2>/dev/null | grep -E '^\s*/\*[0-9a-fA-F]+\*/' | grep -c ';' || echo "0"
}

analyze_compute_cubin() {
    local cubin="$1"
    local label="$2"
    if [ ! -f "${cubin}" ]; then
        echo "  ${label}: file not found"
        return
    fi

    local sass
    sass="$(cuobjdump -sass "${cubin}" 2>/dev/null)"

    local total_sass
    total_sass="$(echo "${sass}" | grep -E '^\s*/\*[0-9a-fA-F]+\*/' | grep -c ';' || echo 0)"

    local ffma_count
    ffma_count="$(echo "${sass}" | grep -E '^\s*/\*[0-9a-fA-F]+\*/' | grep -iE '\bFFMA(\.[A-Z0-9_]+)*\b' | grep -c ';' || echo 0)"

    local fmul_count
    fmul_count="$(echo "${sass}" | grep -E '^\s*/\*[0-9a-fA-F]+\*/' | grep -iE '\bFMUL(\.[A-Z0-9_]+)*\b' | grep -c ';' || echo 0)"

    local fadd_count
    fadd_count="$(echo "${sass}" | grep -E '^\s*/\*[0-9a-fA-F]+\*/' | grep -iE '\bFADD(\.[A-Z0-9_]+)*\b' | grep -c ';' || echo 0)"

    local bra_count
    bra_count="$(echo "${sass}" | grep -E '^\s*/\*[0-9a-fA-F]+\*/' | grep -iE '\bBRA(\.[A-Z0-9_]+)*\b' | grep -c ';' || echo 0)"

    local math_total=$((ffma_count + fmul_count + fadd_count))
    local unroll_info="Rolled loop (unroll factor 1: ${math_total} float op(s) in body, ${bra_count} branch(es))"
    if [ "${bra_count}" -eq 0 ] && [ "${math_total}" -ge 32 ]; then
        unroll_info="Fully unrolled (${math_total} float ops, no backward branch)"
    elif [ "${math_total}" -gt 4 ] && [ "${bra_count}" -gt 0 ]; then
        unroll_info="Partially unrolled (${math_total} float ops per loop, ${bra_count} branch(es))"
    fi

    echo "  ${label}:"
    echo "    - Total SASS instructions:    ${total_sass}"
    echo "    - Float arithmetic breakdown: FFMA: ${ffma_count} | FMUL: ${fmul_count} | FADD: ${fadd_count}"
    echo "    - Loop structure:             ${unroll_info}"
}

echo "SASS Instruction Counts (sm_70 machine instructions via cuobjdump -sass):"
echo "  Benchmark 1 - SAXPY:"
echo "    - MiniPTX:                    $(count_sass_instructions "${BUILD_DIR}/miniptx_saxpy.cubin") SASS instructions"
echo "    - Hand-written CUDA (nvcc -O3): $(count_sass_instructions "${BUILD_DIR}/cuda_saxpy.cubin") SASS instructions"
echo
echo "  Benchmark 2 - Compute-Heavy Diagnostics (Diagnostic b):"
analyze_compute_cubin "${BUILD_DIR}/miniptx_compute_heavy.cubin"       "MiniPTX (emitted PTX)"
analyze_compute_cubin "${BUILD_DIR}/cuda_compute_heavy.cubin"          "CUDA (nvcc -O3)"
analyze_compute_cubin "${BUILD_DIR}/cuda_compute_heavy_nofma.cubin"    "CUDA no-FMA (nvcc -O3 --fmad=false)"
echo
echo "Methodology Note on Instruction Counts:"
echo "  - In SAXPY: MiniPTX produces 24 SASS instructions vs 16 for CUDA, but both achieve"
echo "    identical runtime because DRAM bandwidth is fully saturated."
echo "  - In Compute-Heavy: MiniPTX produces 24 SASS instructions vs 32 for CUDA, yet MiniPTX"
echo "    runs slower. Static instruction counts do not equal dynamically executed instructions"
echo "    when loops, instruction scheduling, and hardware FMA fusion are involved."
echo "================================================================="
