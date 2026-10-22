#pragma once
#include "miniptx/ast.hpp"
#include "miniptx/lexer.hpp"

namespace miniptx {

// Parse a token stream into a KernelDecl (AST root).
// Throws std::runtime_error with line numbers on syntax errors.
KernelDecl parse(const std::vector<Token>& tokens);

}  // namespace miniptx
