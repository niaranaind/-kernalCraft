#include "miniptx/lexer.hpp"

namespace miniptx {

std::vector<Token> lex(const std::string& src) {
  std::vector<Token> out;
  // TODO: scan characters, skip whitespace/comments, emit tokens.
  (void)src;
  out.push_back({Tok::Eof, "", 1});
  return out;
}

}  // namespace miniptx
