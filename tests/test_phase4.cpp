#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "miniptx/lexer.hpp"
#include "miniptx/parser.hpp"
#include "miniptx/ast.hpp"
#include "miniptx/ir.hpp"
#include "miniptx/ptx_emitter.hpp"

using namespace miniptx;

// ---------------------------------------------------------------------------
// Test 1: Lexer column tracking
// ---------------------------------------------------------------------------
bool test_lexer_columns() {
    std::string src = "int x = 42;\n  float y = 3.14;";
    auto tokens = lex(src);

    // Line 1:
    // int at col 1
    // x at col 5
    // = at col 7
    // 42 at col 9
    // ; at col 11
    if (tokens.size() < 10) return false;

    if (tokens[0].kind != Tok::Int || tokens[0].line != 1 || tokens[0].col != 1) return false;
    if (tokens[1].kind != Tok::Ident || tokens[1].text != "x" || tokens[1].line != 1 || tokens[1].col != 5) return false;
    if (tokens[2].kind != Tok::Assign || tokens[2].line != 1 || tokens[2].col != 7) return false;
    if (tokens[3].kind != Tok::IntLit || tokens[3].text != "42" || tokens[3].line != 1 || tokens[3].col != 9) return false;
    if (tokens[4].kind != Tok::Semi || tokens[4].line != 1 || tokens[4].col != 11) return false;

    // Line 2:
    // indent 2 spaces -> float at col 3
    if (tokens[5].kind != Tok::Float || tokens[5].line != 2 || tokens[5].col != 3) return false;
    if (tokens[6].kind != Tok::Ident || tokens[6].text != "y" || tokens[6].line != 2 || tokens[6].col != 9) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Test 2: AST column propagation
// ---------------------------------------------------------------------------
bool test_ast_columns() {
    std::string src = "kernel test() {\n    int a = 10 + 20;\n}";
    auto tokens = lex(src);
    auto kernel = parse(tokens);

    if (kernel.line != 1 || kernel.col != 1) return false;
    if (kernel.end_line != 3) return false;
    if (kernel.body.size() != 1) return false;

    // Check VarDeclStmt
    const auto& s = kernel.body[0];
    const auto* vdecl = std::get_if<VarDeclStmt>(&s->node);
    if (!vdecl) return false;
    if (vdecl->line != 2 || vdecl->col != 5) return false;

    // Check BinaryExpr
    const auto* bin = std::get_if<BinaryExpr>(&vdecl->init->node);
    if (!bin) return false;
    if (bin->line != 2) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Test 3: PTX .file and .loc emission
// ---------------------------------------------------------------------------
bool test_ptx_line_info_emission() {
    std::string src = R"(kernel k(int* a, int n) {
    int i = tid.x;
    if (i < n) {
        a[i] = 42;
    }
})";
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    type_check(kernel);
    auto ir_fn = lower_to_ir(kernel);
    optimize(ir_fn);

    EmitterOptions opts;
    opts.emit_line_info = true;
    opts.source_filename = "custom_kernel.mk";

    std::string ptx = emit_ptx(ir_fn, opts);

    // Verify .file directive
    if (ptx.find(".file 1 \"custom_kernel.mk\"") == std::string::npos) {
        std::cerr << "Expected .file 1 \"custom_kernel.mk\" in PTX output\n";
        return false;
    }

    // Verify .loc directives exist and have file index 1
    if (ptx.find(".loc 1 1 1") == std::string::npos) {
        std::cerr << "Expected entry .loc 1 1 1 in PTX output\n";
        return false;
    }

    if (ptx.find(".loc 1 2") == std::string::npos) {
        std::cerr << "Expected .loc for line 2\n";
        return false;
    }

    if (ptx.find(".loc 1 3") == std::string::npos) {
        std::cerr << "Expected .loc for line 3\n";
        return false;
    }

    if (ptx.find(".loc 1 4") == std::string::npos) {
        std::cerr << "Expected .loc for line 4\n";
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Line info disabled by default in EmitterOptions
// ---------------------------------------------------------------------------
bool test_ptx_line_info_disabled_by_default() {
    std::string src = "kernel k(int a) { int b = a + 1; }";
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    type_check(kernel);
    auto ir_fn = lower_to_ir(kernel);
    optimize(ir_fn);

    // Default EmitterOptions has emit_line_info = false
    std::string ptx = emit_ptx(ir_fn);
    if (ptx.find(".file") != std::string::npos) return false;
    if (ptx.find(".loc") != std::string::npos) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Test 5: --map Source-to-PTX mapping output format
// ---------------------------------------------------------------------------
bool test_ptx_map_emission() {
    std::string src =
        "kernel saxpy(float a, float* x, float* y, int n) {\n"
        "    int i = tid.x + bid.x * bdim.x;\n"
        "    if (i < n) {\n"
        "        y[i] = a * x[i] + y[i];\n"
        "    }\n"
        "}";

    auto tokens = lex(src);
    auto kernel = parse(tokens);
    type_check(kernel);
    auto ir_fn = lower_to_ir(kernel);
    optimize(ir_fn);

    std::string map_out = emit_ptx_map(ir_fn, src);

    // Check that source lines are printed as comments
    if (map_out.find("; [Line 1] kernel saxpy(float a, float* x, float* y, int n) {") == std::string::npos) {
        std::cerr << "Missing Line 1 in --map output\n";
        return false;
    }
    if (map_out.find("; [Line 2]     int i = tid.x + bid.x * bdim.x;") == std::string::npos) {
        std::cerr << "Missing Line 2 in --map output\n";
        return false;
    }
    if (map_out.find("; [Line 3]     if (i < n) {") == std::string::npos) {
        std::cerr << "Missing Line 3 in --map output\n";
        return false;
    }
    if (map_out.find("; [Line 4]         y[i] = a * x[i] + y[i];") == std::string::npos) {
        std::cerr << "Missing Line 4 in --map output\n";
        return false;
    }
    if (map_out.find("; [Line 6] }") == std::string::npos) {
        std::cerr << "Missing Line 6 in --map output\n";
        return false;
    }

    // Verify key instructions are under the proper lines
    size_t pos_line2 = map_out.find("; [Line 2]");
    size_t pos_line3 = map_out.find("; [Line 3]");
    size_t pos_line4 = map_out.find("; [Line 4]");

    size_t pos_mul_lo = map_out.find("mul.lo.s32");
    if (pos_mul_lo < pos_line2 || pos_mul_lo > pos_line3) {
        std::cerr << "mul.lo.s32 should appear under Line 2\n";
        return false;
    }

    size_t pos_setp = map_out.find("setp.lt.s32");
    if (pos_setp < pos_line3 || pos_setp > pos_line4) {
        std::cerr << "setp.lt.s32 should appear under Line 3\n";
        return false;
    }

    size_t pos_st = map_out.find("st.global.f32");
    if (pos_st < pos_line4) {
        std::cerr << "st.global.f32 should appear under Line 4\n";
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Test 6: For-loop source mapping
// ---------------------------------------------------------------------------
bool test_ptx_map_for_loop() {
    std::string src =
        "kernel fill(int* x, int n) {\n"
        "    for (int i = 0; i < n; i = i + 1) {\n"
        "        x[i] = i;\n"
        "    }\n"
        "}";

    auto tokens = lex(src);
    auto kernel = parse(tokens);
    type_check(kernel);
    auto ir_fn = lower_to_ir(kernel);
    optimize(ir_fn);

    std::string map_out = emit_ptx_map(ir_fn, src);

    if (map_out.find("; [Line 2]     for (int i = 0; i < n; i = i + 1) {") == std::string::npos) return false;
    if (map_out.find("; [Line 3]         x[i] = i;") == std::string::npos) return false;
    if (map_out.find("BB_for_cond") == std::string::npos) return false;
    if (map_out.find("st.global.s32") == std::string::npos) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Main test driver
// ---------------------------------------------------------------------------
int main() {
    int passed = 0;
    int total = 0;

    auto run_test = [&](const char* name, bool (*fn)()) {
        total++;
        std::cout << "  " << name << " ... ";
        if (fn()) {
            std::cout << "PASS\n";
            passed++;
        } else {
            std::cout << "FAIL\n";
        }
    };

    std::cout << "=== Phase 4a Test Suite (Debug Info & --map) ===\n";
    run_test("test_lexer_columns", test_lexer_columns);
    run_test("test_ast_columns", test_ast_columns);
    run_test("test_ptx_line_info_emission", test_ptx_line_info_emission);
    run_test("test_ptx_line_info_disabled_by_default", test_ptx_line_info_disabled_by_default);
    run_test("test_ptx_map_emission", test_ptx_map_emission);
    run_test("test_ptx_map_for_loop", test_ptx_map_for_loop);

    std::cout << "\nResults: " << passed << "/" << total << " tests passed.\n";
    return (passed == total) ? 0 : 1;
}
