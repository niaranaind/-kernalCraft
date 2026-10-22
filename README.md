# MiniPTX

A small C++17 compiler from a tiny C-like GPU kernel language to NVIDIA PTX,
with a runtime loader (CUDA Driver API), a simple module linker, and
source-line debug info. Written from scratch (no LLVM/ANTLR).

## Pipeline
source -> lexer -> parser/AST -> type check -> 3-address IR -> optimization passes -> PTX -> ptxas -> run on GPU

## Language Specification

### Types
- `int`     — 32-bit signed integer
- `float`   — 32-bit IEEE 754 float
- `int*`    — pointer to int  (global memory)
- `float*`  — pointer to float (global memory)

**No implicit conversions.** You cannot mix `int` and `float` in the same
expression — for example, `n + 1.0` is a type error if `n` is `int`.
Write `n + 1` instead.  There is no cast syntax.

**Pointer comparison is not allowed.**  Pointers may only be used with
the indexing operator `p[i]`; they cannot appear in arithmetic or comparison
expressions.

### Built-in read-only values
- `tid.x`  — thread index within the block (like `threadIdx.x` in CUDA)
- `bid.x`  — block index within the grid   (like `blockIdx.x`  in CUDA)
- `bdim.x` — number of threads per block   (like `blockDim.x`  in CUDA)

Only the `.x` component is supported.  Only `tid`, `bid`, and `bdim` are
valid builtin bases.

### Operators (in order of precedence, tightest first)
1. `-` (prefix) — unary minus (int and float only, not pointers)
2. `*` `/`      — multiplication, division
3. `+` `-`      — addition, subtraction
4. `<` `>` `<=` `>=` `==` `!=` — comparison (produce int 0 or 1)

Binary operators are left-associative.  Unary minus binds tighter than
any binary operator, so `-a * b` is `(-a) * b`.

### Grammar (EBNF)

```
program     = kernel ;
kernel      = "kernel" IDENT "(" param_list ")" block ;
param_list  = param { "," param } ;
param       = type IDENT ;
type        = "int" [ "*" ] | "float" [ "*" ] ;

block       = "{" { stmt } "}" ;
stmt        = var_decl | assign_stmt | if_stmt | for_stmt ;
var_decl    = type IDENT "=" expr ";" ;
assign_stmt = lvalue "=" expr ";" ;
if_stmt     = "if" "(" expr ")" block [ "else" block ] ;
for_stmt    = "for" "(" var_decl expr ";" assign_stmt_no_semi ")" block ;

assign_stmt_no_semi = lvalue "=" expr ;
lvalue      = IDENT | IDENT "[" expr "]" ;

expr        = cmp_expr ;
cmp_expr    = add_expr { ( "<" | ">" | "<=" | ">=" | "==" | "!=" ) add_expr } ;
add_expr    = mul_expr { ( "+" | "-" ) mul_expr } ;
mul_expr    = unary    { ( "*" | "/" ) unary } ;
unary       = "-" unary | primary ;
primary     = INT_LIT | FLOAT_LIT | IDENT "[" expr "]"
            | IDENT "." IDENT | IDENT | "(" expr ")" ;

INT_LIT     = digit { digit } ;
FLOAT_LIT   = digit { digit } "." digit { digit } ;
IDENT       = letter { letter | digit | "_" } ;
```

### Semantics
- A program is exactly one kernel.
- Parameters are passed by value (scalars) or by pointer (arrays in global GPU memory).
- `tid.x`, `bid.x`, `bdim.x` are int-typed and read-only.
- Array indexing `p[i]` dereferences pointer `p` at offset `i` (element-sized).
- Integer division truncates toward zero.
- Comparisons return `int` 0 or 1.
- The condition in `if` and `for` is true when non-zero.
- Both operands of a binary operator must have the same scalar type (no implicit int↔float).
- Pointers cannot appear in arithmetic or comparison expressions.

## Intermediate Representation (IR)

MiniPTX lowers the AST into a three-address code IR organized into basic blocks.

### Value Representation: Temporaries vs Variable Slots
The IR uses **single-assignment expression temporaries plus mutable variable slots**, not full SSA with phi nodes:
- **Expression Temporaries** (`%t0`, `%t1`, `%t2`, ...): Every temporary is defined exactly once by an instruction. Operands to arithmetic, comparisons, loads, stores, and branches are always temporaries.
- **Mutable Variable Slots** (`%var_i_0`, `%var_x_1`, ...): Each declared variable and kernel parameter has a dedicated virtual register slot that can be assigned multiple times across different basic blocks.
  - A **variable read** emits a `mov` from the variable slot into a fresh temporary at the point of use.
  - A **variable assignment** emits a `mov` from the computed temporary into the variable slot.

### Control Flow and Merging
- **If / Else**: Evaluates the condition in the current block, ending with `condbr %cond, BB_then, BB_else`. Both branches execute straight-line code and terminate with an unconditional jump (`br BB_merge`). At the merge block, variable reads simply load from the variable slots, naturally observing whichever branch executed without needing phi nodes.
- **For Loops**: Lowers into four canonical blocks:
  1. Header / Condition (`BB_for_cond`): Tests loop condition and branches with `condbr` to the body or exit.
  2. Body (`BB_for_body`): Executes loop statements and jumps to the update block.
  3. Update (`BB_for_upd`): Evaluates update expression, stores into the loop variable slot, and jumps back to the condition block (`br BB_for_cond`).
  4. Exit (`BB_for_exit`): Target when loop condition is zero.

### Optimization Passes (Block-Local)
MiniPTX runs an optimization pipeline: `Constant Folding -> DCE -> CSE -> DCE`.
1. **Constant Folding**:
   - Evaluates operations whose operands are known compile-time constants.
   - Evaluates signed integer arithmetic with wraparound semantics using `uint32_t`.
   - **Safety rules**: Never folds division by zero. Never folds `INT_MIN / -1` (prevents UB and hardware faults). Evaluates floats strictly in `float` precision and skips NaN / infinity results.
2. **Dead Code Elimination (DCE)**:
   - Eliminates unused expression temporaries.
   - Cascades iteratively until a fixed point.
   - **Safety rules**: Never deletes `store` instructions (global memory side-effects) and never deletes `mov` instructions into variable slots.
3. **Common Subexpression Elimination (CSE)**:
   - Identifies duplicate computations in the same basic block and replaces subsequent occurrences with a copy of the initial result.
   - Supports commutativity for symmetric operations (`+`, `*`, `==`, `!=`).
   - **Safety rules**: Whenever a `store` instruction occurs, all cached `load` expressions are invalidated to prevent unsafe memory alias caching. Whenever a variable slot is assigned, cached reads from that slot are invalidated.
4. **IR Verifier**:
   - Runs automatically after lowering and after every optimization pass.
   - Asserts that every block has exactly one terminator (`br`, `condbr`, `ret`), no terminator occurs mid-block, every operand is defined before use, branch targets exist, `condbr` conditions are `i32`, and instruction operand/result types strictly match.

## PTX Back End & GPU Execution

### Target Architecture & Version
- Target: `.target sm_70` (NVIDIA Volta / Tesla V100 on DGX).
- Default PTX ISA: `.version 6.0` (defined as a named constant in `ptx_emitter.hpp`).
- Address Size: `.address_size 64` (64-bit pointers).

#### Checking and Changing PTX / CUDA Version on your DGX
1. On your DGX, check your CUDA version:
   ```bash
   nvcc --version
   nvidia-smi
   ```
2. In `include/miniptx/ptx_emitter.hpp`, the constant `DEFAULT_PTX_VERSION` is set to `"6.0"`. Volta (sm_70) requires PTX ISA 6.0 or newer. If you are using a newer CUDA release (e.g. CUDA 11 or 12) and wish to target a higher PTX version, you can change `DEFAULT_PTX_VERSION` or pass `EmitterOptions{ "7.0", "sm_70" }`.

### Emitted PTX Characteristics
- **Registers**: Virtual register classes `.reg .b32 %r<N>`, `.reg .f32 %f<N>`, `.reg .b64 %rd<N>`, and `.reg .pred %p<N>`. Declared bounds strictly encompass all register indices used in the body.
- **Arithmetic**:
  - Integer multiply: `mul.lo.s32`
  - Float arithmetic: uses `.rn` (round-to-nearest-even) explicitly: `add.rn.f32`, `sub.rn.f32`, `mul.rn.f32`, `div.rn.f32` to prevent unintended FMA fusion.
  - Float inequality: uses `setp.neu.f32` (not equal or unordered) to match C NaN behavior.
  - Float literals: emitted in exact IEEE 754 hex format (`0fHHHHHHHH`), preventing decimal round-trip issues.
- **Memory Addressing**:
  - Pointers loaded from `.param` space are converted with `cvta.to.global.u64`.
  - Element addressing `ptr[idx]` uses `mul.wide.s32 %rd0, %r_idx, 4` followed by `add.s64 %rd0, %rd_ptr, %rd0` and `ld.global` / `st.global`.
- **Branch Optimization**:
  - Comparison temporaries used exclusively by `condbr` are fused directly into predicate registers without materializing into integer registers with `selp`.
  - If a branch target is the next block in sequential order, the unconditional jump is omitted and allowed to fall through.

### Running on the DGX
To build the compiler, assemble with `ptxas`, and execute all 5 kernels on the Tesla V100 with the CUDA Driver API:
```bash
bash scripts/run_dgx.sh
```

### Troubleshooting on Shared Linux Clusters (e.g. DGX)

#### `GLIBCXX_3.4.XX not found`
- **Cause**: On shared systems such as NVIDIA DGX servers, a newer `g++` toolchain in `$PATH` may produce binaries referencing newer C++ runtime symbols than the host's default dynamic `/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. When launching the compiled binary (`miniptxc` or `run_ptx`), the dynamic linker throws:
  ```text
  /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
  ```
- **Solution**: Link both binaries with `-static-libstdc++ -static-libgcc`. This statically embeds the required standard library routines into the executable, making it completely independent of the system's dynamic `libstdc++` version. `scripts/run_dgx.sh` automatically links with these flags.

#### `unsupported GNU version! gcc versions later than 12 are not supported!`
- **Cause**: When compiling the hand-written CUDA comparison kernel with `nvcc` (e.g. CUDA 12.0), `nvcc` invokes the default host compiler found in `$PATH` (`/usr/local/bin/g++`, which is GCC 13.1 on some DGX setups). CUDA 12.0 only supports GCC versions up to GCC 12, causing `nvcc` to abort.
- **Solution**: Pass a supported system host compiler (e.g. `/usr/bin/g++`) to `nvcc` using `-ccbin`. In `scripts/benchmark_dgx.sh`, the variable `NVCC_CCBIN` defaults to `/usr/bin/g++` and can be overridden:
  ```bash
  NVCC_CCBIN=/usr/bin/g++ bash scripts/benchmark_dgx.sh
  ```
  Note that `miniptxc` and the benchmark harness continue to use the modern `g++` with `-static-libstdc++`; only `nvcc` uses `NVCC_CCBIN`.

## Source-Line Debug Info & Source Mapping (Phase 4a)

MiniPTX emits standard DWARF-compatible line information using PTX directives:
- **`.file 1 "<filename>"`**: Declares the source file at module scope.
- **`.loc 1 <line> <col>`**: Precedes emitted instructions, mapping each PTX instruction to its exact 1-based source line and column.
- Column numbers are tracked throughout the compilation pipeline (lexer $\to$ AST $\to$ IR $\to$ PTX emitter).

### Interleaved Source-to-PTX Listing (`--map`)
Invoking `miniptxc --map <kernel.mk>` prints an interleaved listing pairing each source line with its generated PTX instructions.

**Real emitter output for `examples/saxpy.mk`**:
```text
$ ./build/miniptxc --map examples/saxpy.mk
; [Line 1] kernel saxpy(float a, float* x, float* y, int n) {
    ld.param.f32    %f0, [param_a];
    ld.param.u64    %rd1, [param_x];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u64    %rd3, [param_y];
    cvta.to.global.u64 %rd4, %rd3;
    ld.param.s32    %r0, [param_n];
BB0:
    mov.f32         %f1, %f0;
    mov.b64         %rd5, %rd2;
    mov.b64         %rd6, %rd4;
    mov.b32         %r1, %r0;

; [Line 2]     int i = tid.x + bid.x * bdim.x;
    mov.u32         %r3, %tid.x;
    mov.u32         %r4, %ctaid.x;
    mov.u32         %r5, %ntid.x;
    mul.lo.s32      %r6, %r4, %r5;
    add.s32         %r7, %r3, %r6;
    mov.b32         %r2, %r7;

; [Line 3]     if (i < n) { y[i] = a * x[i] + y[i]; }
    mov.b32         %r8, %r2;
    mov.b32         %r9, %r1;
    setp.lt.s32    %p0, %r8, %r9;
    @!%p0 bra       BB_merge_1;
BB_then_0:
    mov.b64         %rd7, %rd6;
    mov.b32         %r10, %r2;
    mov.f32         %f2, %f1;
    mov.b64         %rd8, %rd5;
    mul.wide.s32    %rd0, %r10, 4;
    add.s64         %rd0, %rd8, %rd0;
    ld.global.f32   %f3, [%rd0];
    mul.rn.f32      %f4, %f2, %f3;
    mul.wide.s32    %rd0, %r10, 4;
    add.s64         %rd0, %rd7, %rd0;
    ld.global.f32   %f5, [%rd0];
    add.rn.f32      %f6, %f4, %f5;
    mul.wide.s32    %rd0, %r10, 4;
    add.s64         %rd0, %rd7, %rd0;
    st.global.f32   [%rd0], %f6;

; [Line 4] }
BB_merge_1:
    ret;
```

### Inspecting Debug Line Info on the DGX
To verify that `ptxas` translates `.loc` directives into DWARF line information in the compiled ELF (`.cubin`):
```bash
# 1. Compile PTX to cubin with line info
ptxas -lineinfo -m64 -arch=sm_70 build_dgx/saxpy.ptx -o build_dgx/saxpy.cubin

# 2. Inspect debug sections in the ELF binary
cuobjdump -elf build_dgx/saxpy.cubin | grep -i debug

# 3. Disassemble SASS instructions with interleaved source lines
nvdisasm -c -g build_dgx/saxpy.cubin
```
*(Verified on DGX Tesla V100: `cuobjdump -elf` confirms both `.debug_line` and `.nv_debug_line_sass` sections are emitted).*

## Micro-Benchmark: MiniPTX vs. Hand-Written CUDA (Phase 4c)

To evaluate code generation quality, we benchmark MiniPTX against idiomatic hand-written CUDA compiled with `nvcc -ptx -arch=sm_70 -O3`.

### Benchmark 1: SAXPY (Memory-Bound)

- **Hardware Target**: NVIDIA Tesla V100-SXM2-32GB (`sm_70`, 6 MiB L2 cache, shared cluster).
- **Workload**: SAXPY on $N = 2^{24} = 16,777,216$ `float` elements.
  - Footprint in GPU memory: $\approx 128\text{ MiB}$ (vectors $x$ and $y$).
  - Memory traffic per launch: $192\text{ MiB}$ ($3 \times N \times 4\text{ bytes}$ for read $x$, read $y$, write $y$).
  - Exceeds the 6 MiB L2 cache, streaming data directly through HBM2 DRAM.
- **Methodology**:
  - Pure CUDA Driver API execution (`cuEvent*`, `cuLaunchKernel`) through the identical harness and launch configuration (256 threads/block, 65,536 blocks).
  - Both kernels verified against single-precision CPU references on fresh buffers before timing.
  - Alternating A/B timed iterations on shared GPU 0 using high-precision GPU hardware events (`cuEventElapsedTime`).

#### Results on Tesla V100

| Implementation | PTX Instructions | SASS Instructions | Median Time (ms) | Effective BW (GB/s) | Relative Ratio |
|---|---|---|---|---|---|
| **MiniPTX** (emitted PTX) | 36 | 24 | 0.2570 ms | 783.2988 GB/s | 1.000x |
| **CUDA** (`nvcc -O3` PTX) | 20 | 16 | 0.2570 ms | 783.2988 GB/s | 1.000x |

> [!NOTE]
> **Observation**: Both MiniPTX and `nvcc -O3` achieve an identical median runtime of `0.2570 ms` (`783.2988 GB/s`). SAXPY has an arithmetic intensity of $0.167\text{ FLOP/byte}$, which completely saturates the Volta HBM2 memory subsystem. Because memory bandwidth is the bottleneck, the difference in instruction counts (36 vs 20 PTX, 24 vs 16 SASS) does not impact execution time.

### Benchmark 2: Compute-Heavy (N = 2^24 elements, 64-iter inner loop)

- **Workload**: $N = 2^{24}$ elements with an inner loop executing 64 multiply-add operations per element ($v = v \times a + 0.5$).
- **Characterization**: Described as "compute-heavy" (rather than strictly arithmetic-bound) because it also loads and stores $N$ elements across global memory.
- **Fair Comparison**: Identical Driver API loader, launch configuration (256 threads/block, 65,536 blocks), alternating execution, and CPU reference verification.

#### Results on Tesla V100 (Run 3)

| Implementation | PTX Instructions | SASS Instructions | Median Time (ms) | Throughput (GFLOPS) | Relative Ratio |
|---|---|---|---|---|---|
| **MiniPTX** (emitted PTX) | 54 | 24 | 0.4874 ms | 4405.8 GFLOPS | 1.123x (12.26% slower) |
| **CUDA** (`nvcc -O3` PTX) | 28 | 32 | 0.4342 ms | 4946.1 GFLOPS | 1.000x (baseline) |
| **CUDA no-FMA** (`nvcc -O3 --fmad=false`) | 29 | 32 | 0.4884 ms | 4396.5 GFLOPS | 0.998x (0.21% diff) |

#### SASS Opcode & Loop Structure Breakdown (Verified)

| Implementation | Total SASS | FFMA | FMUL | FADD | Loop Structure (Heuristic) |
|---|---|---|---|---|---|
| **MiniPTX** (emitted PTX) | 24 | 0 | 1 | 1 | Rolled loop |
| **CUDA** (`nvcc -O3` PTX) | 32 | 1 | 0 | 0 | Rolled loop |
| **CUDA no-FMA** (`nvcc -O3 --fmad=false`) | 32 | 0 | 1 | 1 | Rolled loop |

> [!NOTE]
> **Key Findings & Performance Analysis**:
> 1. **FMA Contraction Explains the Gap**:
>    - MiniPTX emits explicit `mul.rn.f32` and `add.rn.f32` (two separate, dependent operations per iteration: `FMUL` followed by `FADD`).
>    - Default `nvcc -O3` contracts the multiply and add into a single fused multiply-add instruction (`FFMA`), executing 1 arithmetic operation per loop iteration instead of 2.
>    - When `nvcc` is compiled with `--fmad=false`, it also emits `FMUL` and `FADD` (0 FFMA, 1 FMUL, 1 FADD), and its runtime matches MiniPTX within measurement noise (`0.4874 ms` vs `0.4884 ms`, a `0.21%` difference).
>    - Hardware FMA contraction delivers an `11.11%` speedup to `nvcc` (`0.889x`), accounting for the entire observed performance difference.
> 2. **Loop Unrolling Ruled Out**:
>    - The SASS inspection confirms all three implementations preserve a rolled loop with 1 backward branch and a single arithmetic sequence per loop. This rules out unrolling as a contributing factor. *(Note: the "rolled/unrolled" check is a heuristic based on instruction and branch counts).*
> 3. **SASS Instruction Count Does Not Predict Speed**:
>    - MiniPTX has 24 total SASS instructions while the `--fmad=false` CUDA control has 32, yet both exhibit identical execution time (~0.487 ms - 0.488 ms). Static instruction count does not predict execution speed when loop execution cycles dominate.

#### Multi-Run Measurement Stability (Shared DGX Cluster)

| Run | MiniPTX Time (ms) | CUDA -O3 Time (ms) | CUDA no-FMA Time (ms) | Ratio (MiniPTX / CUDA -O3) | Notes |
|---|---|---|---|---|---|
| **Run 1** | 0.4721 ms | 0.4209 ms | N/A | **1.122x** | Initial benchmark, no FMA control |
| **Run 2** | 0.4424 ms | 0.3948 ms | 0.4424 ms | **1.121x** | Added `--fmad=false` diagnostic |
| **Run 3** | 0.4874 ms | 0.4342 ms | 0.4884 ms | **1.123x** | Full diagnostic with SASS opcode counts |

Absolute execution times varied across runs on the shared DGX cluster (e.g. MiniPTX between `0.4424 ms` and `0.4874 ms`) due to concurrent user workloads and GPU thermal/power states. However, the comparative ratio between MiniPTX and default CUDA remained remarkably stable at **1.121x – 1.123x**.

### Running Benchmarks on the DGX
```bash
bash scripts/benchmark_dgx.sh
# or override host compiler for nvcc if needed:
NVCC_CCBIN=/usr/bin/g++ bash scripts/benchmark_dgx.sh
```

## Build & Test Locally (macOS)
```bash
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

## Usage
```bash
./build/miniptxc examples/saxpy.mk              # compile to PTX (with .loc line info)
./build/miniptxc --no-lineinfo examples/saxpy.mk # compile to PTX without debug directives
./build/miniptxc --map examples/saxpy.mk        # print interleaved source-to-PTX map
./build/miniptxc --dump-ast examples/saxpy.mk    # print the AST
./build/miniptxc --dump-ir examples/saxpy.mk     # print the optimized IR
```

## Roadmap
- [x] Week 1: lexer, parser, AST, type checker, error messages
- [x] Week 2: IR lowering, basic blocks, IR printer & verifier, constant folding / DCE / CSE
- [x] Week 3: PTX emitter, CUDA Driver API runtime, structural golden-file tests, DGX validation script
- [ ] Week 4: module linker, .loc debug info, benchmark vs hand-written CUDA
  - [x] Phase 4a: .loc & .file debug info, column tracking, --map source-to-PTX listing
  - [x] Phase 4c: Micro-benchmarks vs hand-written CUDA (Driver API, CUDA events)
  - [ ] Phase 4b: Helper functions (`func`), call-graph cycle check, module linker

## Verification Results
- **Phase 1 tests (CPU)**: 43/43 passed (lexer, parser, AST shape, precedence, scopes, type checking).
- **Phase 2 tests (CPU)**: 25/25 passed (IR verifier, before/after pass tests, safety invariant tests, full pipeline).
- **Phase 3 tests (CPU)**: 10/10 passed (structural checks, register coverage, branch target checks, instruction format checks, float hex formatting, comparison fusion).
- **Phase 4a tests (CPU)**: 6/6 passed (lexer column tracking, AST column propagation, PTX `.file`/`.loc` emission, `--map` interleaved mapping, for-loop mapping).
- **GPU Execution & ptxas (DGX Tesla V100)**: **5/5 tests passed** (Phase 3 & 4a verified, DWARF `.debug_line` and `.nv_debug_line_sass` verified, SAXPY verified at 0.2570 ms / 783.2988 GB/s, compute-heavy verified at 0.4874 ms / 4405.8 GFLOPS matching CUDA --fmad=false control within 0.21% at 0.998x; sanitized log in `docs/dgx_benchmark.log`).
