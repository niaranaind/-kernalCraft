#include <fstream>
#include <iostream>
#include <sstream>
#include "miniptx/lexer.hpp"
#include "miniptx/parser.hpp"
#include "miniptx/ir.hpp"
#include "miniptx/ptx_emitter.hpp"

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: miniptxc <kernel.mk>\n"; return 1; }
  std::ifstream in(argv[1]);
  std::stringstream ss; ss << in.rdbuf();
  auto tokens = miniptx::lex(ss.str());
  miniptx::parse(tokens);
  miniptx::lower_and_optimize();
  std::cout << miniptx::emit_ptx();
  return 0;
}
