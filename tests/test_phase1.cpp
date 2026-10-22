//
// Phase 1 tests: lexer, parser, AST, and type checker.
//
// Each test function returns true on success.  We run them all and report
// failures at the end so you can see everything that went wrong at once.
//
// Tests are organized into groups:
//   1. Lexer tests: tokenization, line numbers, comments, errors
//   2. Parser tests: valid programs, syntax errors, AST shape checks
//   3. Type checker tests: well-typed programs, ill-typed programs, scopes
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
// Helpers
// ---------------------------------------------------------------------------

// Try to lex, parse, and type-check a program.
// Returns true if all three succeed without throwing.
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

// Parse a source string and return the kernel AST (no type checking).
static KernelDecl parse_src(const std::string& src) {
    auto tokens = lex(src);
    return parse(tokens);
}

// ============================================================================
// LEXER TESTS
// ============================================================================

// Test: Empty input produces just an Eof token.
bool test_lex_empty() {
    auto t = lex("");
    return t.size() == 1 && t[0].kind == Tok::Eof;
}

// Test: All keywords are recognized (not treated as identifiers).
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

// Test: Integer and float literals.
bool test_lex_numbers() {
    auto t = lex("42 3.14");
    return t.size() == 3 &&
           t[0].kind == Tok::IntLit && t[0].text == "42" &&
           t[1].kind == Tok::FloatLit && t[1].text == "3.14" &&
           t[2].kind == Tok::Eof;
}

// Test: Two-character operators are not split into two tokens.
bool test_lex_two_char_ops() {
    auto t = lex("<= >= == !=");
    return t.size() == 5 &&
           t[0].kind == Tok::Le &&
           t[1].kind == Tok::Ge &&
           t[2].kind == Tok::Eq &&
           t[3].kind == Tok::Ne;
}

// Test: Line numbers are tracked across newlines.
bool test_lex_line_numbers() {
    auto t = lex("int\nfloat\nkernel");
    return t[0].line == 1 && t[0].kind == Tok::Int &&
           t[1].line == 2 && t[1].kind == Tok::Float &&
           t[2].line == 3 && t[2].kind == Tok::Kernel;
}

// Test: Comments are skipped entirely.
bool test_lex_comments() {
    auto t = lex("int // this is a comment\nfloat");
    return t.size() == 3 &&
           t[0].kind == Tok::Int && t[0].line == 1 &&
           t[1].kind == Tok::Float && t[1].line == 2;
}

// Test: An unexpected character produces an error with a line number.
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

// Test: The dot operator is lexed for builtins like tid.x.
bool test_lex_dot() {
    auto t = lex("tid.x");
    return t.size() == 4 &&
           t[0].kind == Tok::Ident && t[0].text == "tid" &&
           t[1].kind == Tok::Dot &&
           t[2].kind == Tok::Ident && t[2].text == "x";
}

// Test: A lone "!" is a lexer error (only "!=" is valid).
bool test_lex_lone_bang() {
    try {
        lex("!");
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("unexpected character") != std::string::npos &&
               msg.find("'!'") != std::string::npos;
    }
}

// ============================================================================
// PARSER TESTS (valid programs)
// ============================================================================

// Test: The saxpy example from the spec parses successfully.
bool test_parse_saxpy() {
    std::string src = R"(
        kernel saxpy(float a, float* x, float* y, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { y[i] = a * x[i] + y[i]; }
        }
    )";
    auto kernel = parse_src(src);
    return kernel.name == "saxpy" && kernel.params.size() == 4;
}

// Test: A minimal kernel with no body.
bool test_parse_empty_body() {
    auto kernel = parse_src("kernel empty(int n) {}");
    return kernel.name == "empty" && kernel.body.empty();
}

// Test: A for loop parses correctly.
bool test_parse_for_loop() {
    std::string src = R"(
        kernel loop_test(int* x, int n) {
            for (int i = 0; i < n; i = i + 1) {
                x[i] = i;
            }
        }
    )";
    auto kernel = parse_src(src);
    return kernel.body.size() == 1;  // one for statement
}

// Test: If-else structure.
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
    auto kernel = parse_src(src);
    return kernel.body.size() == 2;  // VarDecl + If
}

// Test: Unary minus parses.
bool test_parse_unary_minus() {
    std::string src = "kernel neg(int n) { int i = -n; }";
    auto kernel = parse_src(src);
    // body[0] is VarDeclStmt, its init should be a UnaryMinusExpr
    auto& init = std::get<VarDeclStmt>(kernel.body[0]->node).init;
    return std::holds_alternative<UnaryMinusExpr>(init->node);
}

// Test: Double negation "--a" parses as UnaryMinus(UnaryMinus(a)).
bool test_parse_double_negation() {
    std::string src = "kernel neg2(int n) { int i = - -n; }";
    auto kernel = parse_src(src);
    auto& init = std::get<VarDeclStmt>(kernel.body[0]->node).init;
    // Outer should be UnaryMinus
    auto& outer = std::get<UnaryMinusExpr>(init->node);
    // Inner should also be UnaryMinus
    return std::holds_alternative<UnaryMinusExpr>(outer.operand->node);
}

// ============================================================================
// PARSER TESTS — AST shape (precedence and associativity)
// ============================================================================

// Test: "a + b * c" should parse as BinOp(+, Var(a), BinOp(*, Var(b), Var(c)))
// This verifies that * binds tighter than +.
bool test_precedence_mul_over_add() {
    // Wrap in a kernel so the parser is happy.
    std::string src = "kernel p(int a, int b, int c) { int r = a + b * c; }";
    auto kernel = parse_src(src);
    auto& init = std::get<VarDeclStmt>(kernel.body[0]->node).init;

    // Root should be BinOp(+)
    auto& root = std::get<BinaryExpr>(init->node);
    if (root.op != BinOp::Add) return false;

    // LHS of + should be Var(a)
    auto& lhs = std::get<VarExpr>(root.lhs->node);
    if (lhs.name != "a") return false;

    // RHS of + should be BinOp(*)
    auto& rhs = std::get<BinaryExpr>(root.rhs->node);
    if (rhs.op != BinOp::Mul) return false;

    // Children of * should be b and c
    auto& mul_lhs = std::get<VarExpr>(rhs.lhs->node);
    auto& mul_rhs = std::get<VarExpr>(rhs.rhs->node);
    return mul_lhs.name == "b" && mul_rhs.name == "c";
}

// Test: "a - b - c" should parse as BinOp(-, BinOp(-, a, b), c)
// This verifies left-associativity.
bool test_left_associativity_sub() {
    std::string src = "kernel p(int a, int b, int c) { int r = a - b - c; }";
    auto kernel = parse_src(src);
    auto& init = std::get<VarDeclStmt>(kernel.body[0]->node).init;

    // Root should be BinOp(-)
    auto& root = std::get<BinaryExpr>(init->node);
    if (root.op != BinOp::Sub) return false;

    // RHS of the outer - should be Var(c)
    auto& rhs = std::get<VarExpr>(root.rhs->node);
    if (rhs.name != "c") return false;

    // LHS of the outer - should be BinOp(-, a, b) (left-assoc)
    auto& inner = std::get<BinaryExpr>(root.lhs->node);
    if (inner.op != BinOp::Sub) return false;

    auto& inner_lhs = std::get<VarExpr>(inner.lhs->node);
    auto& inner_rhs = std::get<VarExpr>(inner.rhs->node);
    return inner_lhs.name == "a" && inner_rhs.name == "b";
}

// Test: "a * b / c" should parse as BinOp(/, BinOp(*, a, b), c)
// Left-associativity at the multiplicative level.
bool test_left_associativity_mul_div() {
    std::string src = "kernel p(int a, int b, int c) { int r = a * b / c; }";
    auto kernel = parse_src(src);
    auto& init = std::get<VarDeclStmt>(kernel.body[0]->node).init;

    auto& root = std::get<BinaryExpr>(init->node);
    if (root.op != BinOp::Div) return false;

    auto& rhs = std::get<VarExpr>(root.rhs->node);
    if (rhs.name != "c") return false;

    auto& inner = std::get<BinaryExpr>(root.lhs->node);
    if (inner.op != BinOp::Mul) return false;

    auto& inner_lhs = std::get<VarExpr>(inner.lhs->node);
    auto& inner_rhs = std::get<VarExpr>(inner.rhs->node);
    return inner_lhs.name == "a" && inner_rhs.name == "b";
}

// Test: "-a * b" should parse as BinOp(*, UnaryMinus(a), b)
// Unary minus binds tighter than *.
bool test_unary_minus_precedence() {
    std::string src = "kernel p(int a, int b) { int r = -a * b; }";
    auto kernel = parse_src(src);
    auto& init = std::get<VarDeclStmt>(kernel.body[0]->node).init;

    // Root should be BinOp(*)
    auto& root = std::get<BinaryExpr>(init->node);
    if (root.op != BinOp::Mul) return false;

    // LHS of * should be UnaryMinus(Var(a))
    auto& neg = std::get<UnaryMinusExpr>(root.lhs->node);
    auto& neg_operand = std::get<VarExpr>(neg.operand->node);
    if (neg_operand.name != "a") return false;

    // RHS of * should be Var(b)
    auto& rhs = std::get<VarExpr>(root.rhs->node);
    return rhs.name == "b";
}

// ============================================================================
// PARSER TESTS (syntax errors)
// ============================================================================

// Test: Missing semicolon after variable declaration.
bool test_parse_missing_semi() {
    return compile_fails_with(
        "kernel bad(int n) { int i = 0 }",
        "expected ';'");
}

// Test: Missing closing parenthesis.
bool test_parse_missing_rparen() {
    return compile_fails_with(
        "kernel bad(int n { }",
        "expected ')'");
}

// Test: Token after kernel body.
bool test_parse_extra_tokens() {
    return compile_fails_with(
        "kernel ok(int n) {} extra",
        "unexpected token");
}

// Test: Exact error line number for a missing semicolon.
// The error should point to line 3 where the "}" is, since that's where
// the parser finds "}" when it expected ";".
bool test_error_exact_line() {
    // Line 1: kernel bad(int n) {
    // Line 2:     int i = 0
    // Line 3: }
    std::string src = "kernel bad(int n) {\n    int i = 0\n}";
    try {
        auto tokens = lex(src);
        parse(tokens);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        // The "}" is on line 3, so the error must say "line 3".
        return msg.find("line 3") != std::string::npos &&
               msg.find("expected ';'") != std::string::npos;
    }
}

// ============================================================================
// PARSER TESTS — builtins
// ============================================================================

// Test: Unknown builtin (tid.y not supported — only .x).
bool test_parse_unknown_builtin() {
    return compile_fails_with(
        R"(kernel bad(int n) { int i = tid.y; })",
        "only .x is supported");
}

// Test: Unknown builtin base (foo.x is not valid).
bool test_parse_unknown_builtin_base() {
    return compile_fails_with(
        R"(kernel bad(int n) { int i = foo.x; })",
        "only tid, bid, bdim are allowed");
}

// ============================================================================
// TYPE CHECKER TESTS (well-typed programs)
// ============================================================================

// Test: The saxpy example is well-typed.
bool test_type_saxpy() {
    std::string src = R"(
        kernel saxpy(float a, float* x, float* y, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { y[i] = a * x[i] + y[i]; }
        }
    )";
    return compiles_ok(src);
}

// Test: Vector add (int) is well-typed.
bool test_type_vecadd() {
    std::string src = R"(
        kernel vecadd(int* a, int* b, int* c, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { c[i] = a[i] + b[i]; }
        }
    )";
    return compiles_ok(src);
}

// Test: Unary minus on int is well-typed.
bool test_type_unary_minus_int() {
    return compiles_ok("kernel neg(int n) { int i = -n; }");
}

// Test: Unary minus on float is well-typed.
bool test_type_unary_minus_float() {
    return compiles_ok("kernel neg(float f) { float g = -f; }");
}

// Test: If-else where both branches assign the same type.
bool test_type_if_else_ok() {
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
    return compiles_ok(src);
}

// ============================================================================
// TYPE CHECKER TESTS (ill-typed programs)
// ============================================================================

// Test: Undeclared variable.
bool test_type_undeclared() {
    return compile_fails_with(
        "kernel bad(int n) { int i = x; }",
        "undeclared variable 'x'");
}

// Test: Duplicate declaration in the same scope (param + local).
bool test_type_duplicate() {
    return compile_fails_with(
        "kernel bad(int n) { int n = 0; }",
        "duplicate declaration of 'n'");
}

// Test: Duplicate declaration — two locals in the same block.
bool test_type_duplicate_locals() {
    return compile_fails_with(
        "kernel bad(int n) { int x = 0; int x = 1; }",
        "duplicate declaration of 'x'");
}

// Test: Type mismatch in variable init (int = float).
bool test_type_mismatch_assign() {
    return compile_fails_with(
        "kernel bad(int n) { int i = 1.0; }",
        "cannot initialize int with float");
}

// Test: Indexing a non-pointer.
bool test_type_index_non_pointer() {
    return compile_fails_with(
        "kernel bad(int n) { int i = n[0]; }",
        "cannot index non-pointer");
}

// Test: Type mismatch in binary expression (int + float).
bool test_type_mismatch_binop() {
    return compile_fails_with(
        R"(kernel bad(int n, float f) { int i = n + f; })",
        "type mismatch");
}

// Test: Pointer used in arithmetic.
bool test_type_ptr_arithmetic() {
    return compile_fails_with(
        R"(kernel bad(int* p, int n) { int i = p + n; })",
        "cannot use pointer type");
}

// Test: Store type mismatch — assigning int to float* element.
bool test_type_store_mismatch() {
    return compile_fails_with(
        R"(kernel bad(float* x) { int i = 0; x[i] = i; })",
        "cannot assign int to float element");
}

// Test: Unary minus on pointer is a type error.
bool test_type_unary_minus_ptr() {
    return compile_fails_with(
        R"(kernel bad(int* p) { int i = -p; })",
        "cannot negate pointer");
}

// ============================================================================
// SCOPE TESTS
// ============================================================================

// Test: A variable declared inside an if-then block is not visible outside.
bool test_scope_if_leak() {
    // "inner" is declared inside if-then; using it after should fail.
    return compile_fails_with(
        R"(kernel bad(int n) {
            if (n) {
                int inner = 1;
            }
            int x = inner;
        })",
        "undeclared variable 'inner'");
}

// Test: A variable declared inside an else block is not visible outside.
bool test_scope_else_leak() {
    return compile_fails_with(
        R"(kernel bad(int n) {
            if (n) {
                int a = 1;
            } else {
                int b = 2;
            }
            int x = b;
        })",
        "undeclared variable 'b'");
}

// Test: A for-loop init variable is not visible after the loop.
bool test_scope_for_leak() {
    return compile_fails_with(
        R"(kernel bad(int n) {
            for (int i = 0; i < n; i = i + 1) {
                int dummy = i;
            }
            int x = i;
        })",
        "undeclared variable 'i'");
}

// Test: Nested scopes allow shadowing — same name can be redeclared
// in a deeper scope without error.
bool test_scope_shadowing_ok() {
    std::string src = R"(
        kernel ok(int n) {
            int x = 0;
            if (n) {
                int x = 1;
            }
        }
    )";
    return compiles_ok(src);
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
    run_test("lex_lone_bang",       test_lex_lone_bang);

    // Parser tests (valid programs)
    run_test("parse_saxpy",              test_parse_saxpy);
    run_test("parse_empty_body",         test_parse_empty_body);
    run_test("parse_for_loop",           test_parse_for_loop);
    run_test("parse_if_else",            test_parse_if_else);
    run_test("parse_unary_minus",        test_parse_unary_minus);
    run_test("parse_double_negation",    test_parse_double_negation);

    // Parser tests — AST shape (precedence and associativity)
    run_test("precedence_mul_over_add",  test_precedence_mul_over_add);
    run_test("left_assoc_sub",           test_left_associativity_sub);
    run_test("left_assoc_mul_div",       test_left_associativity_mul_div);
    run_test("unary_minus_precedence",   test_unary_minus_precedence);

    // Parser tests (syntax errors)
    run_test("parse_missing_semi",       test_parse_missing_semi);
    run_test("parse_missing_rparen",     test_parse_missing_rparen);
    run_test("parse_extra_tokens",       test_parse_extra_tokens);
    run_test("error_exact_line",         test_error_exact_line);

    // Parser tests — builtins
    run_test("parse_unknown_builtin",      test_parse_unknown_builtin);
    run_test("parse_unknown_builtin_base", test_parse_unknown_builtin_base);

    // Type checker tests (valid)
    run_test("type_saxpy",              test_type_saxpy);
    run_test("type_vecadd",             test_type_vecadd);
    run_test("type_unary_minus_int",    test_type_unary_minus_int);
    run_test("type_unary_minus_float",  test_type_unary_minus_float);
    run_test("type_if_else_ok",         test_type_if_else_ok);

    // Type checker tests (errors)
    run_test("type_undeclared",         test_type_undeclared);
    run_test("type_duplicate",          test_type_duplicate);
    run_test("type_duplicate_locals",   test_type_duplicate_locals);
    run_test("type_mismatch_assign",    test_type_mismatch_assign);
    run_test("type_index_non_ptr",      test_type_index_non_pointer);
    run_test("type_mismatch_binop",     test_type_mismatch_binop);
    run_test("type_ptr_arithmetic",     test_type_ptr_arithmetic);
    run_test("type_store_mismatch",     test_type_store_mismatch);
    run_test("type_unary_minus_ptr",    test_type_unary_minus_ptr);

    // Scope tests
    run_test("scope_if_leak",           test_scope_if_leak);
    run_test("scope_else_leak",         test_scope_else_leak);
    run_test("scope_for_leak",          test_scope_for_leak);
    run_test("scope_shadowing_ok",      test_scope_shadowing_ok);

    // Summary
    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed.\n";
    if (!failures.empty()) {
        std::cout << "Failures:\n";
        for (const auto& f : failures) std::cout << "  - " << f << "\n";
        return 1;
    }
    return 0;
}
