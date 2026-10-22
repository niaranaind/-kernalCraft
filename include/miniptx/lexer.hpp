#pragma once
#include <string>
#include <vector>

namespace miniptx {

enum class Tok { Ident, IntLit, FloatLit, Kernel, If, For, Int, Float,
                 LParen, RParen, LBrace, RBrace, LBracket, RBracket,
                 Comma, Semi, Assign, Plus, Minus, Star, Slash, Lt, Gt, Eof };

struct Token {
  Tok kind;
  std::string text;
  int line;
};

// TODO(week 1): hand-written lexer. Track line numbers for error messages and .loc debug info.
std::vector<Token> lex(const std::string& src);

}  // namespace miniptx
