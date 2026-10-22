//
// Phase 1 tests: lexer, parser, AST, and type checker.
//
// Each test function returns true on success.  We run them all and report
// failures at the end so you can see everything that went wrong at once.
//
// Tests are organized into three groups:
//   1. Lexer tests: tokenization, line numbers, comments, errors
//   2. Parser tests: valid programs parse, syntax errors are caught
//   3. Type checker tests: well-typed programs pass, ill-typed are rejected
//

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "miniptx/lexer.hpp"
#include "miniptx/parser.hpp"
#include "miniptx/ast.hpp"

using namespace miniptx;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int tests_run = 0;
static int tests_passed = 0;
static std::vector<std::string> failures;

// Run a test.  If it returns false or throws, record it as a failure.
static void run_test(const char* name, bool (*fn)()) {
    ++tests_run;
    try {
        if (fn()) {
            ++tests_passed;
        } else {
            failures.push_back(name);
            std::cerr << "FAIL: " << name << "\n";
        }
    } catch (const std::exception& e) {
        failures.push_back(std::string(name) + " (threw: " + e.what() + ")");
        std::cerr << "FAIL: " << name << " — exception: " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Helper: try to lex, parse, and type-check a program.
// Returns true if all three succeed without throwing.
// ---------------------------------------------------------------------------
static bool compiles_ok(const std::string& src) {
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    type_check(kernel);
    return true;
}

// Returns true if lex+parse+typecheck throws an error containing `substr`.
static bool compile_fails_with(const std::string& src, const std::string& substr) {
    try {
        compiles_ok(src);
        return false;  // Should have thrown
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find(substr) != std::string::npos) return true;
        std::cerr << "  expected error containing '" << substr
                  << "', got: " << msg << "\n";
        return false;
    }
}

// ============================================================================
// LEXER TESTS
// ============================================================================

// Test 1: Empty input produces just an Eof token.
bool test_lex_empty() {
    auto t = lex("");
    return t.size() == 1 && t[0].kind == Tok::Eof;
}

// Test 2: All keywords are recognized (not treated as identifiers).
bool test_lex_keywords() {
    auto t = lex("kernel if else for int float");
    return t.size() == 7 &&  // 6 keywords + Eof
           t[0].kind == Tok::Kernel &&
           t[1].kind == Tok::If &&
           t[2].kind == Tok::Else &&
           t[3].kind == Tok::For &&
           t[4].kind == Tok::Int &&
           t[5].kind == Tok::Float &&
           t[6].kind == Tok::Eof;
}

// Test 3: Integer and float literals.
bool test_lex_numbers() {
    auto t = lex("42 3.14");
    return t.size() == 3 &&
           t[0].kind == Tok::IntLit && t[0].text == "42" &&
           t[1].kind == Tok::FloatLit && t[1].text == "3.14" &&
           t[2].kind == Tok::Eof;
}

// Test 4: Two-character operators are not split into two tokens.
bool test_lex_two_char_ops() {
    auto t = lex("<= >= == !=");
    return t.size() == 5 &&
           t[0].kind == Tok::Le &&
           t[1].kind == Tok::Ge &&
           t[2].kind == Tok::Eq &&
           t[3].kind == Tok::Ne;
}

// Test 5: Line numbers are tracked across newlines.
bool test_lex_line_numbers() {
    auto t = lex("int\nfloat\nkernel");
    return t[0].line == 1 && t[0].kind == Tok::Int &&
           t[1].line == 2 && t[1].kind == Tok::Float &&
           t[2].line == 3 && t[2].kind == Tok::Kernel;
}

// Test 6: Comments are skipped entirely.
bool test_lex_comments() {
    auto t = lex("int // this is a comment\nfloat");
    return t.size() == 3 &&
           t[0].kind == Tok::Int && t[0].line == 1 &&
           t[1].kind == Tok::Float && t[1].line == 2;
}

// Test 7: An unexpected character produces an error with a line number.
bool test_lex_bad_char() {
    try {
        lex("int @");
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("line 1") != std::string::npos &&
               msg.find("'@'") != std::string::npos;
    }
}

// Test 8: The dot operator is lexed for builtins like tid.x.
bool test_lex_dot() {
    auto t = lex("tid.x");
    return t.size() == 4 &&
           t[0].kind == Tok::Ident && t[0].text == "tid" &&
           t[1].kind == Tok::Dot &&
           t[2].kind == Tok::Ident && t[2].text == "x";
}

// ============================================================================
// PARSER TESTS (valid programs)
// ============================================================================

// Test 9: The saxpy example from the spec parses successfully.
bool test_parse_saxpy() {
    std::string src = R"(
        kernel saxpy(float a, float* x, float* y, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { y[i] = a * x[i] + y[i]; }
        }
    )";
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    return kernel.name == "saxpy" && kernel.params.size() == 4;
}

// Test 10: A minimal kernel with no body.
bool test_parse_empty_body() {
    auto tokens = lex("kernel empty(int n) {}");
    auto kernel = parse(tokens);
    return kernel.name == "empty" && kernel.body.empty();
}

// Test 11: A for loop parses correctly.
bool test_parse_for_loop() {
    std::string src = R"(
        kernel loop_test(int* x, int n) {
            for (int i = 0; i < n; i = i + 1) {
                x[i] = i;
            }
        }
    )";
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    return kernel.body.size() == 1;  // one for statement
}

// Test 12: Nested if-else.
bool test_parse_if_else() {
    std::string src = R"(
        kernel cond(int* x, int n) {
            int i = tid.x;
            if (i < n) {
                x[i] = 1;
            } else {
                x[i] = 0;
            }
        }
    )";
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    return kernel.body.size() == 2;  // VarDecl + If
}

// ============================================================================
// PARSER TESTS (syntax errors)
// ============================================================================

// Test 13: Missing semicolon after variable declaration.
bool test_parse_missing_semi() {
    return compile_fails_with(
        "kernel bad(int n) { int i = 0 }",
        "expected ';'");
}

// Test 14: Missing closing parenthesis.
bool test_parse_missing_rparen() {
    return compile_fails_with(
        "kernel bad(int n { }",
        "expected ')'");
}

// Test 15: Token after kernel body.
bool test_parse_extra_tokens() {
    return compile_fails_with(
        "kernel ok(int n) {} extra",
        "unexpected token");
}

// ============================================================================
// TYPE CHECKER TESTS (well-typed programs)
// ============================================================================

// Test 16: The saxpy example is well-typed.
bool test_type_saxpy() {
    std::string src = R"(
        kernel saxpy(float a, float* x, float* y, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { y[i] = a * x[i] + y[i]; }
        }
    )";
    return compiles_ok(src);
}

// Test 17: Vector add (int) is well-typed.
bool test_type_vecadd() {
    std::string src = R"(
        kernel vecadd(int* a, int* b, int* c, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { c[i] = a[i] + b[i]; }
        }
    )";
    return compiles_ok(src);
}

// ============================================================================
// TYPE CHECKER TESTS (ill-typed programs)
// ============================================================================

// Test 18: Undeclared variable.
bool test_type_undeclared() {
    return compile_fails_with(
        "kernel bad(int n) { int i = x; }",
        "undeclared variable 'x'");
}

// Test 19: Duplicate declaration in the same scope.
bool test_type_duplicate() {
    return compile_fails_with(
        "kernel bad(int n) { int n = 0; }",
        "duplicate declaration of 'n'");
}

// Test 20: Type mismatch in assignment (int = float).
bool test_type_mismatch_assign() {
    return compile_fails_with(
        "kernel bad(int n) { int i = 1.0; }",
        "cannot initialize int with float");
}

// Test 21: Indexing a non-pointer.
bool test_type_index_non_pointer() {
    return compile_fails_with(
        "kernel bad(int n) { int i = n[0]; }",
        "cannot index non-pointer");
}

// Test 22: Type mismatch in binary expression (int + float).
bool test_type_mismatch_binop() {
    return compile_fails_with(
        R"(kernel bad(int n, float f) { int i = n + f; })",
        "type mismatch");
}

// Test 23: Pointer used in arithmetic.
bool test_type_ptr_arithmetic() {
    return compile_fails_with(
        R"(kernel bad(int* p, int n) { int i = p + n; })",
        "cannot use pointer type");
}

// Test 24: Store type mismatch — assigning int to float* element.
bool test_type_store_mismatch() {
    return compile_fails_with(
        R"(kernel bad(float* x) { int i = 0; x[i] = i; })",
        "cannot assign int to float element");
}

// Test 25: Unknown builtin.
bool test_parse_unknown_builtin() {
    return compile_fails_with(
        R"(kernel bad(int n) { int i = tid.y; })",
        "unknown builtin 'tid.y'");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Lexer tests
    run_test("lex_empty",           test_lex_empty);
    run_test("lex_keywords",        test_lex_keywords);
    run_test("lex_numbers",         test_lex_numbers);
    run_test("lex_two_char_ops",    test_lex_two_char_ops);
    run_test("lex_line_numbers",    test_lex_line_numbers);
    run_test("lex_comments",        test_lex_comments);
    run_test("lex_bad_char",        test_lex_bad_char);
    run_test("lex_dot",             test_lex_dot);

    // Parser tests (valid)
    run_test("parse_saxpy",         test_parse_saxpy);
    run_test("parse_empty_body",    test_parse_empty_body);
    run_test("parse_for_loop",      test_parse_for_loop);
    run_test("parse_if_else",       test_parse_if_else);

    // Parser tests (errors)
    run_test("parse_missing_semi",  test_parse_missing_semi);
    run_test("parse_missing_rparen",test_parse_missing_rparen);
    run_test("parse_extra_tokens",  test_parse_extra_tokens);

    // Type checker tests (valid)
    run_test("type_saxpy",          test_type_saxpy);
    run_test("type_vecadd",         test_type_vecadd);

    // Type checker tests (errors)
    run_test("type_undeclared",     test_type_undeclared);
    run_test("type_duplicate",      test_type_duplicate);
    run_test("type_mismatch_assign",test_type_mismatch_assign);
    run_test("type_index_non_ptr",  test_type_index_non_pointer);
    run_test("type_mismatch_binop", test_type_mismatch_binop);
    run_test("type_ptr_arithmetic", test_type_ptr_arithmetic);
    run_test("type_store_mismatch", test_type_store_mismatch);
    run_test("parse_unknown_builtin", test_parse_unknown_builtin);

    // Summary
    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed.\n";
    if (!failures.empty()) {
        std::cout << "Failures:\n";
        for (const auto& f : failures) std::cout << "  - " << f << "\n";
        return 1;
    }
    return 0;
}
