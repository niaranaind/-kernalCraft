#include <cassert>
#include "miniptx/lexer.hpp"
int main() {
  auto t = miniptx::lex("");
  assert(!t.empty() && t.back().kind == miniptx::Tok::Eof);
  // TODO: add real tests as you implement the lexer (keywords, numbers, line numbers).
  return 0;
}
