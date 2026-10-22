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
nvdisasm -g build_dgx/saxpy.cubin
```
*(Note: Inspecting the cubin sections and SASS disassembly via `cuobjdump` and `nvdisasm` is **unverified until run on the DGX**).*

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
  - [ ] Phase 4c: SAXPY micro-benchmark vs hand-written CUDA (Driver API, CUDA events)
  - [ ] Phase 4b: Helper functions (`func`), call-graph cycle check, module linker

## Verification Results
- **Phase 1 tests (CPU)**: 43/43 passed (lexer, parser, AST shape, precedence, scopes, type checking).
- **Phase 2 tests (CPU)**: 25/25 passed (IR verifier, before/after pass tests, safety invariant tests, full pipeline).
- **Phase 3 tests (CPU)**: 10/10 passed (structural checks, register coverage, branch target checks, instruction format checks, float hex formatting, comparison fusion).
- **Phase 4a tests (CPU)**: 6/6 passed (lexer column tracking, AST column propagation, PTX `.file`/`.loc` emission, `--map` interleaved mapping, for-loop mapping).
- **GPU Execution & ptxas**: **Unverified on macOS** (no NVIDIA GPU or ptxas installed locally). Complete code and automated test script `scripts/run_dgx.sh` provided for execution on your DGX Tesla V100.
