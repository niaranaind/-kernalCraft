#pragma once
#include "miniptx/lexer.hpp"
namespace miniptx {
// TODO(week 1): recursive-descent parser: tokens -> AST. Report errors with line numbers.
void parse(const std::vector<Token>& tokens);
}  // namespace miniptx
