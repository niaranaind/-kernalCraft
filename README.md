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
   - Cascades iteratively until a fixed point (e.g. deleting an unused operation makes its operands unused, removing them on the next pass).
   - **Safety rules**: Never deletes `store` instructions (global memory side-effects) and never deletes `mov` instructions into variable slots.
3. **Common Subexpression Elimination (CSE)**:
   - Identifies duplicate computations in the same basic block and replaces subsequent occurrences with a copy of the initial result.
   - Supports commutativity for symmetric operations (`+`, `*`, `==`, `!=`).
   - **Safety rules**: Whenever a `store` instruction occurs, all cached `load` expressions are invalidated to prevent unsafe memory alias caching. Whenever a variable slot is assigned, cached reads from that slot are invalidated.
4. **IR Verifier**:
   - Runs automatically after lowering and after every optimization pass.
   - Asserts that every block has exactly one terminator (`br`, `condbr`, `ret`), no terminator occurs mid-block, every operand is defined before use, branch targets exist, `condbr` conditions are `i32`, and instruction operand/result types strictly match.

## Build
    cmake -S . -B build && cmake --build build && ctest --test-dir build

## Usage
    ./build/miniptxc examples/saxpy.mk           # compile to PTX (stdout)
    ./build/miniptxc --dump-ast examples/saxpy.mk # print the AST
    ./build/miniptxc --dump-ir examples/saxpy.mk  # print the optimized IR

## Roadmap
- [x] Week 1: lexer, parser, AST, type checker, error messages
- [x] Week 2: IR lowering, basic blocks, IR printer & verifier, constant folding / DCE / CSE
- [ ] Week 3: PTX emitter, run on GPU, CPU-reference tests
- [ ] Week 4: module linker, .loc debug info, benchmark vs hand-written CUDA

## Results
- **Phase 1 tests**: 43/43 passed (lexer, parser, AST shape, precedence, scopes, type checking).
- **Phase 2 tests**: 25/25 passed (IR verifier, before/after pass tests, safety invariant tests, full pipeline).
- **Example kernels**: saxpy, vecadd, scale, ewmax, fill all compile, lower, and optimize cleanly.
