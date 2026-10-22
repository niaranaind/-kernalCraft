# MiniPTX

A small C++17 compiler from a tiny C-like GPU kernel language to NVIDIA PTX,
with a runtime loader (CUDA Driver API), a simple module linker, and
source-line debug info. Written from scratch (no LLVM/ANTLR).

## Pipeline
source -> lexer -> parser/AST -> type check -> SSA IR -> optimization passes -> PTX -> ptxas -> run on GPU

## Build
    cmake -S . -B build && cmake --build build && ctest --test-dir build

## Roadmap
- [ ] Week 1: lexer, parser, AST, type checker, error messages
- [ ] Week 2: SSA IR, IR printer, constant folding / DCE / CSE
- [ ] Week 3: PTX emitter, run on GPU, CPU-reference tests
- [ ] Week 4: module linker, .loc debug info, benchmark vs hand-written CUDA

## Results
(Fill in: kernels supported, tests passing, benchmark numbers.)
