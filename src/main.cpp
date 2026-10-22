#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include "miniptx/lexer.hpp"
#include "miniptx/parser.hpp"
#include "miniptx/ast.hpp"
#include "miniptx/ir.hpp"
#include "miniptx/ptx_emitter.hpp"

int main(int argc, char** argv) {
    // --- Parse command-line flags ---
    bool dump_ast = false;
    const char* filename = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-ast") == 0) {
            dump_ast = true;
        } else if (argv[i][0] != '-') {
            filename = argv[i];
        } else {
            std::cerr << "unknown flag: " << argv[i] << "\n";
            return 1;
        }
    }

    if (!filename) {
        std::cerr << "usage: miniptxc [--dump-ast] <kernel.mk>\n";
        return 1;
    }

    // --- Read the source file ---
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "error: cannot open '" << filename << "'\n";
        return 1;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string source = ss.str();

    try {
        // Phase 1: Lex + Parse + Type-check
        auto tokens = miniptx::lex(source);
        auto kernel = miniptx::parse(tokens);
        miniptx::type_check(kernel);

        if (dump_ast) {
            miniptx::dump_ast(kernel);
            return 0;
        }

        // Phase 2+: IR lowering and PTX emission (stubs for now)
        miniptx::lower_and_optimize();
        std::cout << miniptx::emit_ptx();

    } catch (const std::runtime_error& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
