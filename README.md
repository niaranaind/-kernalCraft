# MiniPTX

A small C++17 compiler from a tiny C-like GPU kernel language to NVIDIA PTX,
with a runtime loader (CUDA Driver API), a simple module linker, and
source-line debug info. Written from scratch (no LLVM/ANTLR).

## Pipeline
source -> lexer -> parser/AST -> type check -> SSA IR -> optimization passes -> PTX -> ptxas -> run on GPU

## Language Specification

### Types
- `int`     — 32-bit signed integer
- `float`   — 32-bit IEEE 754 float
- `int*`    — pointer to int  (global memory)
- `float*`  — pointer to float (global memory)

### Built-in read-only values
- `tid.x`  — thread index within the block (like `threadIdx.x` in CUDA)
- `bid.x`  — block index within the grid   (like `blockIdx.x`  in CUDA)
- `bdim.x` — number of threads per block   (like `blockDim.x`  in CUDA)

### Operators (in order of precedence, tightest first)
1. `*` `/`         — multiplication, division
2. `+` `-`         — addition, subtraction
3. `<` `>` `<=` `>=` `==` `!=` — comparison (produce int 0 or 1)

All operators are left-associative and binary.
Unary minus is not supported (use `0 - x`).

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
mul_expr    = unary     { ( "*" | "/" ) unary } ;
unary       = primary ;
primary     = INT_LIT | FLOAT_LIT | IDENT "(" args ")" | IDENT "[" expr "]"
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

## Build
    cmake -S . -B build && cmake --build build && ctest --test-dir build

## Usage
    ./build/miniptxc examples/saxpy.mk           # compile to PTX (stdout)
    ./build/miniptxc --dump-ast examples/saxpy.mk # print the AST
    ./build/miniptxc --dump-ir examples/saxpy.mk  # print the IR (Phase 2)

## Roadmap
- [ ] Week 1: lexer, parser, AST, type checker, error messages
- [ ] Week 2: SSA IR, IR printer, constant folding / DCE / CSE
- [ ] Week 3: PTX emitter, run on GPU, CPU-reference tests
- [ ] Week 4: module linker, .loc debug info, benchmark vs hand-written CUDA

## Results
(Fill in: kernels supported, tests passing, benchmark numbers.)
