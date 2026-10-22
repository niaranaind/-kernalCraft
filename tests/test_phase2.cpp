//
// Phase 2 tests: Three-address IR, IR Verifier, and Optimization Passes
// (Constant Folding, DCE, CSE).
//
// Each test function returns true on success.
// We test:
//   1. IR Verifier (valid IR passes, malformed IR is caught with specific errors)
//   2. Constant Folding (before vs after IR, integer wraparound, and unsafe cases left untouched)
//   3. Dead Code Elimination (before vs after IR, cascading, keeping stores and variable movs)
//   4. Common Subexpression Elimination (before vs after IR, commutative ops, store invalidating loads)
//   5. Full pipeline on representative kernels
//

#include <cassert>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "miniptx/lexer.hpp"
#include "miniptx/parser.hpp"
#include "miniptx/ast.hpp"
#include "miniptx/ir.hpp"

using namespace miniptx;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int tests_run = 0;
static int tests_passed = 0;
static std::vector<std::string> failures;

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

// Helper: lower source code to IR without optimizations
static IRFunction lower_source(const std::string& src) {
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    type_check(kernel);
    return lower_to_ir(kernel);
}

// ============================================================================
// 1. IR VERIFIER TESTS
// ============================================================================

// Valid IR from lowering passes verification
bool test_verify_valid() {
    auto fn = lower_source("kernel k(int a, int b) { int c = a + b; }");
    verify_ir(fn); // should not throw
    return true;
}

// Empty blocks vector is rejected
bool test_verify_no_blocks() {
    IRFunction fn;
    fn.name = "empty";
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("no basic blocks") != std::string::npos;
    }
}

// Block with no instructions is rejected
bool test_verify_empty_block() {
    IRFunction fn;
    fn.name = "k";
    fn.blocks.push_back({"BB0", {}});
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("no instructions") != std::string::npos;
    }
}

// Block missing a terminator is rejected
bool test_verify_no_terminator() {
    IRFunction fn;
    fn.name = "k";
    Instruction c;
    c.op = IROp::Const;
    c.type = Type{BaseType::Int, false};
    c.dest = Value{ValueKind::Temp, "%t0", c.type};
    c.int_val = 42;
    fn.blocks.push_back({"BB0", {c}});
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("does not end with a terminator") != std::string::npos;
    }
}

// Terminator in the middle of a block is rejected
bool test_verify_terminator_middle() {
    IRFunction fn;
    fn.name = "k";
    Instruction ret;
    ret.op = IROp::Ret;
    Instruction c;
    c.op = IROp::Const;
    c.type = Type{BaseType::Int, false};
    c.dest = Value{ValueKind::Temp, "%t0", c.type};
    c.int_val = 1;
    // Ret placed before const, then another ret
    fn.blocks.push_back({"BB0", {ret, c, ret}});
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("terminator in the middle") != std::string::npos;
    }
}

// Branch to undefined block label is rejected
bool test_verify_undef_branch_target() {
    IRFunction fn;
    fn.name = "k";
    Instruction br;
    br.op = IROp::Br;
    br.target_block = "BB_nonexistent";
    fn.blocks.push_back({"BB0", {br}});
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("undefined block") != std::string::npos;
    }
}

// CondBr with float condition is rejected
bool test_verify_condbr_non_i32() {
    IRFunction fn;
    fn.name = "k";
    Value fcond{ValueKind::Temp, "%t0", Type{BaseType::Float, false}};
    Instruction c;
    c.op = IROp::Const;
    c.type = Type{BaseType::Float, false};
    c.dest = fcond;
    c.float_val = 1.0f;

    Instruction br;
    br.op = IROp::CondBr;
    br.type = Type{BaseType::Float, false};
    br.args = {fcond};
    br.then_block = "BB0";
    br.else_block = "BB0";

    fn.blocks.push_back({"BB0", {c, br}});
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("condition must be i32") != std::string::npos;
    }
}

// Temporary used before definition is rejected
bool test_verify_use_before_def() {
    IRFunction fn;
    fn.name = "k";
    Value undef_temp{ValueKind::Temp, "%t99", Type{BaseType::Int, false}};
    Instruction add;
    add.op = IROp::Add;
    add.type = Type{BaseType::Int, false};
    add.dest = Value{ValueKind::Temp, "%t0", add.type};
    add.args = {undef_temp, undef_temp};

    Instruction ret;
    ret.op = IROp::Ret;

    fn.blocks.push_back({"BB0", {add, ret}});
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("used before definition") != std::string::npos;
    }
}

// Binary op with mismatched operand types is rejected
bool test_verify_type_mismatch_binary() {
    IRFunction fn;
    fn.name = "k";
    Value t0{ValueKind::Temp, "%t0", Type{BaseType::Int, false}};
    Value t1{ValueKind::Temp, "%t1", Type{BaseType::Float, false}};

    Instruction c0{IROp::Const, Type{BaseType::Int, false}, t0, {}, 1, 0.0f, "", "", "", "", 1};
    Instruction c1{IROp::Const, Type{BaseType::Float, false}, t1, {}, 0, 2.0f, "", "", "", "", 1};

    Instruction add;
    add.op = IROp::Add;
    add.type = Type{BaseType::Int, false};
    add.dest = Value{ValueKind::Temp, "%t2", Type{BaseType::Int, false}};
    add.args = {t0, t1}; // type mismatch: int + float

    Instruction ret;
    ret.op = IROp::Ret;

    fn.blocks.push_back({"BB0", {c0, c1, add, ret}});
    try {
        verify_ir(fn);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("operand/result type mismatch") != std::string::npos;
    }
}

// ============================================================================
// 2. CONSTANT FOLDING PASS TESTS
// ============================================================================

// Shows before and after IR for integer constant folding
bool test_const_fold_before_after_int() {
    auto fn = lower_source(R"(
        kernel fold_int(int* out) {
            int a = 10 + 20 * 2;
            out[0] = a;
        }
    )");

    std::string before = dump_ir_str(fn);
    // Before should contain mul.i32 and add.i32
    if (before.find("mul.i32") == std::string::npos || before.find("add.i32") == std::string::npos) {
        return false;
    }

    int folds = constant_folding_pass(fn);
    verify_ir(fn);
    if (folds < 2) return false;

    std::string after = dump_ir_str(fn);
    // After constant folding, 20*2=40 and 10+40=50 should be folded into const 50
    // and neither mul.i32 nor add.i32 should remain!
    if (after.find("mul.i32") != std::string::npos || after.find("add.i32") != std::string::npos) {
        return false;
    }
    if (after.find("const.i32 50") == std::string::npos) {
        return false;
    }
    return true;
}

// Shows float constant folding in float (not double)
bool test_const_fold_float() {
    auto fn = lower_source(R"(
        kernel fold_float(float* out) {
            float x = 2.5 * 4.0;
            out[0] = x;
        }
    )");

    int folds = constant_folding_pass(fn);
    verify_ir(fn);
    if (folds != 1) return false;

    std::string after = dump_ir_str(fn);
    if (after.find("mul.f32") != std::string::npos) return false;
    if (after.find("const.f32 10.000000") == std::string::npos) return false;
    return true;
}

// Unary negation of constants is folded
bool test_const_fold_neg() {
    auto fn = lower_source(R"(
        kernel fold_neg(int* out) {
            int x = -42;
            out[0] = x;
        }
    )");

    int folds = constant_folding_pass(fn);
    verify_ir(fn);
    if (folds != 1) return false;

    std::string after = dump_ir_str(fn);
    if (after.find("neg.i32") != std::string::npos) return false;
    if (after.find("const.i32 -42") == std::string::npos) return false;
    return true;
}

// Integer wraparound semantics for overflow
bool test_const_fold_int_wraparound() {
    // Manually create IR for INT_MAX + 1
    IRFunction fn;
    fn.name = "wrap";
    Value t0{ValueKind::Temp, "%t0", Type{BaseType::Int, false}};
    Value t1{ValueKind::Temp, "%t1", Type{BaseType::Int, false}};
    Value t2{ValueKind::Temp, "%t2", Type{BaseType::Int, false}};

    Instruction c0{IROp::Const, Type{BaseType::Int, false}, t0, {}, INT_MAX, 0.0f, "", "", "", "", 1};
    Instruction c1{IROp::Const, Type{BaseType::Int, false}, t1, {}, 1, 0.0f, "", "", "", "", 1};
    Instruction add{IROp::Add, Type{BaseType::Int, false}, t2, {t0, t1}, 0, 0.0f, "", "", "", "", 1};
    Instruction ret{IROp::Ret, Type{BaseType::Int, false}, std::nullopt, {}, 0, 0.0f, "", "", "", "", 1};

    fn.blocks.push_back({"BB0", {c0, c1, add, ret}});
    verify_ir(fn);

    int folds = constant_folding_pass(fn);
    verify_ir(fn);
    if (folds != 1) return false;

    // INT_MAX + 1 wraps around to INT_MIN
    std::string after = dump_ir_str(fn);
    return after.find("const.i32 " + std::to_string(INT_MIN)) != std::string::npos;
}

// Unsafe case: Division by zero must NOT be folded
bool test_const_fold_unsafe_div_zero() {
    IRFunction fn;
    fn.name = "divzero";
    Value t0{ValueKind::Temp, "%t0", Type{BaseType::Int, false}};
    Value t1{ValueKind::Temp, "%t1", Type{BaseType::Int, false}};
    Value t2{ValueKind::Temp, "%t2", Type{BaseType::Int, false}};

    Instruction c0{IROp::Const, Type{BaseType::Int, false}, t0, {}, 10, 0.0f, "", "", "", "", 1};
    Instruction c1{IROp::Const, Type{BaseType::Int, false}, t1, {}, 0, 0.0f, "", "", "", "", 1};
    Instruction div{IROp::Div, Type{BaseType::Int, false}, t2, {t0, t1}, 0, 0.0f, "", "", "", "", 1};
    Instruction ret{IROp::Ret, Type{BaseType::Int, false}, std::nullopt, {}, 0, 0.0f, "", "", "", "", 1};

    fn.blocks.push_back({"BB0", {c0, c1, div, ret}});
    verify_ir(fn);

    int folds = constant_folding_pass(fn);
    verify_ir(fn);
    // Must NOT fold!
    if (folds != 0) return false;
    std::string after = dump_ir_str(fn);
    return after.find("div.i32") != std::string::npos;
}

// Unsafe case: INT_MIN / -1 must NOT be folded (causes UB/crash)
bool test_const_fold_unsafe_int_min_neg1() {
    IRFunction fn;
    fn.name = "div_int_min";
    Value t0{ValueKind::Temp, "%t0", Type{BaseType::Int, false}};
    Value t1{ValueKind::Temp, "%t1", Type{BaseType::Int, false}};
    Value t2{ValueKind::Temp, "%t2", Type{BaseType::Int, false}};

    Instruction c0{IROp::Const, Type{BaseType::Int, false}, t0, {}, INT_MIN, 0.0f, "", "", "", "", 1};
    Instruction c1{IROp::Const, Type{BaseType::Int, false}, t1, {}, -1, 0.0f, "", "", "", "", 1};
    Instruction div{IROp::Div, Type{BaseType::Int, false}, t2, {t0, t1}, 0, 0.0f, "", "", "", "", 1};
    Instruction ret{IROp::Ret, Type{BaseType::Int, false}, std::nullopt, {}, 0, 0.0f, "", "", "", "", 1};

    fn.blocks.push_back({"BB0", {c0, c1, div, ret}});
    verify_ir(fn);

    int folds = constant_folding_pass(fn);
    verify_ir(fn);
    // Must NOT fold!
    if (folds != 0) return false;
    std::string after = dump_ir_str(fn);
    return after.find("div.i32") != std::string::npos;
}

// Unsafe case: Float division by 0.0f must NOT be folded
bool test_const_fold_unsafe_float_div_zero() {
    IRFunction fn;
    fn.name = "float_div_zero";
    Value t0{ValueKind::Temp, "%t0", Type{BaseType::Float, false}};
    Value t1{ValueKind::Temp, "%t1", Type{BaseType::Float, false}};
    Value t2{ValueKind::Temp, "%t2", Type{BaseType::Float, false}};

    Instruction c0{IROp::Const, Type{BaseType::Float, false}, t0, {}, 0, 5.0f, "", "", "", "", 1};
    Instruction c1{IROp::Const, Type{BaseType::Float, false}, t1, {}, 0, 0.0f, "", "", "", "", 1};
    Instruction div{IROp::Div, Type{BaseType::Float, false}, t2, {t0, t1}, 0, 0.0f, "", "", "", "", 1};
    Instruction ret{IROp::Ret, Type{BaseType::Float, false}, std::nullopt, {}, 0, 0.0f, "", "", "", "", 1};

    fn.blocks.push_back({"BB0", {c0, c1, div, ret}});
    verify_ir(fn);

    int folds = constant_folding_pass(fn);
    verify_ir(fn);
    if (folds != 0) return false;
    std::string after = dump_ir_str(fn);
    return after.find("div.f32") != std::string::npos;
}

// ============================================================================
// 3. DEAD CODE ELIMINATION (DCE) PASS TESTS
// ============================================================================

// Shows before and after IR for cascading DCE
bool test_dce_cascading() {
    IRFunction fn;
    fn.name = "dce_cascade";
    Value t0{ValueKind::Temp, "%t0", Type{BaseType::Int, false}};
    Value t1{ValueKind::Temp, "%t1", Type{BaseType::Int, false}};
    Value t2{ValueKind::Temp, "%t2", Type{BaseType::Int, false}};

    // t0 = 1, t1 = 2, t2 = t0 + t1; but t2 is never used!
    Instruction c0{IROp::Const, Type{BaseType::Int, false}, t0, {}, 1, 0.0f, "", "", "", "", 1};
    Instruction c1{IROp::Const, Type{BaseType::Int, false}, t1, {}, 2, 0.0f, "", "", "", "", 1};
    Instruction add{IROp::Add, Type{BaseType::Int, false}, t2, {t0, t1}, 0, 0.0f, "", "", "", "", 1};
    Instruction ret{IROp::Ret, Type{BaseType::Int, false}, std::nullopt, {}, 0, 0.0f, "", "", "", "", 1};

    fn.blocks.push_back({"BB0", {c0, c1, add, ret}});
    verify_ir(fn);

    std::string before = dump_ir_str(fn);
    if (before.find("add.i32") == std::string::npos) return false;

    // First round of DCE removes t2, then cascades to remove t0 and t1!
    int eliminated = dead_code_elimination_pass(fn);
    verify_ir(fn);
    if (eliminated != 3) return false;

    std::string after = dump_ir_str(fn);
    // Only 'ret' should remain!
    if (after.find("add.i32") != std::string::npos || after.find("const.i32") != std::string::npos) {
        return false;
    }
    if (after.find("ret") == std::string::npos) return false;
    return true;
}

// DCE must NEVER delete store instructions
bool test_dce_keep_stores() {
    auto fn = lower_source(R"(
        kernel keep_stores(int* p) {
            p[0] = 42;
        }
    )");

    int eliminated = dead_code_elimination_pass(fn);
    verify_ir(fn);
    (void)eliminated;

    std::string after = dump_ir_str(fn);
    return after.find("store.i32") != std::string::npos;
}

// DCE must NEVER delete movs into variable slots
bool test_dce_keep_var_movs() {
    auto fn = lower_source(R"(
        kernel keep_vars(int n) {
            int x = n;
        }
    )");

    dead_code_elimination_pass(fn);
    verify_ir(fn);

    std::string after = dump_ir_str(fn);
    // mov into %var_x must be preserved!
    return after.find("%var_x_1 = mov.i32") != std::string::npos;
}

// ============================================================================
// 4. COMMON SUBEXPRESSION ELIMINATION (CSE) PASS TESTS
// ============================================================================

// Shows before and after IR for arithmetic CSE
bool test_cse_arithmetic() {
    auto fn = lower_source(R"(
        kernel cse_test(int* out, int a, int b) {
            int x = a + b;
            int y = a + b;
            out[0] = x;
            out[1] = y;
        }
    )");

    std::string before = dump_ir_str(fn);
    // Count occurrences of add.i32 before CSE
    size_t pos1 = before.find("add.i32");
    size_t pos2 = (pos1 != std::string::npos) ? before.find("add.i32", pos1 + 1) : std::string::npos;
    if (pos1 == std::string::npos || pos2 == std::string::npos) return false;

    int eliminated = common_subexpression_elimination_pass(fn);
    verify_ir(fn);
    if (eliminated < 1) return false;

    // Run DCE to clean up the replaced mov
    dead_code_elimination_pass(fn);
    verify_ir(fn);

    std::string after = dump_ir_str(fn);
    // Only 1 add.i32 should remain after CSE + DCE!
    size_t a_pos1 = after.find("add.i32");
    size_t a_pos2 = (a_pos1 != std::string::npos) ? after.find("add.i32", a_pos1 + 1) : std::string::npos;
    return (a_pos1 != std::string::npos && a_pos2 == std::string::npos);
}

// Commutative CSE: a + b and b + a are recognized as identical
bool test_cse_commutative() {
    auto fn = lower_source(R"(
        kernel cse_comm(int* out, int a, int b) {
            int x = a + b;
            int y = b + a;
            out[0] = x;
            out[1] = y;
        }
    )");

    int eliminated = common_subexpression_elimination_pass(fn);
    verify_ir(fn);
    if (eliminated < 1) return false;

    dead_code_elimination_pass(fn);
    verify_ir(fn);

    std::string after = dump_ir_str(fn);
    size_t p1 = after.find("add.i32");
    size_t p2 = (p1 != std::string::npos) ? after.find("add.i32", p1 + 1) : std::string::npos;
    return (p1 != std::string::npos && p2 == std::string::npos);
}

// CSE on load: identical loads without intervening store are eliminated
bool test_cse_load_elimination() {
    auto fn = lower_source(R"(
        kernel cse_loads(int* out, int* in) {
            int x = in[0];
            int y = in[0];
            out[0] = x + y;
        }
    )");

    int eliminated = common_subexpression_elimination_pass(fn);
    verify_ir(fn);
    if (eliminated < 1) return false;

    dead_code_elimination_pass(fn);
    verify_ir(fn);

    std::string after = dump_ir_str(fn);
    // Only 1 load.i32 from in[0] should remain
    size_t p1 = after.find("load.i32");
    size_t p2 = (p1 != std::string::npos) ? after.find("load.i32", p1 + 1) : std::string::npos;
    return (p1 != std::string::npos && p2 == std::string::npos);
}

// Unsafe case: Store invalidates load CSE table
bool test_cse_unsafe_store_invalidates_load() {
    auto fn = lower_source(R"(
        kernel store_inv(int* arr) {
            int x = arr[0];
            arr[0] = 99;
            int y = arr[0];
            arr[1] = x + y;
        }
    )");

    // Run CSE
    common_subexpression_elimination_pass(fn);
    verify_ir(fn);
    dead_code_elimination_pass(fn);
    verify_ir(fn);

    std::string after = dump_ir_str(fn);
    // Both loads MUST remain because arr[0] = 99 invalidated previous loads!
    size_t p1 = after.find("load.i32");
    size_t p2 = (p1 != std::string::npos) ? after.find("load.i32", p1 + 1) : std::string::npos;
    return (p1 != std::string::npos && p2 != std::string::npos);
}

// ============================================================================
// 5. FULL PIPELINE OPTIMIZATION TESTS
// ============================================================================

bool test_pipeline_saxpy() {
    auto fn = lower_source(R"(
        kernel saxpy(float a, float* x, float* y, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { y[i] = a * x[i] + y[i]; }
        }
    )");

    optimize(fn);
    verify_ir(fn);

    std::string ir = dump_ir_str(fn);
    return ir.find("kernel saxpy") != std::string::npos &&
           ir.find("condbr") != std::string::npos &&
           ir.find("store.f32") != std::string::npos;
}

bool test_pipeline_for_loop() {
    auto fn = lower_source(R"(
        kernel fill(int* x, int n) {
            for (int i = 0; i < n; i = i + 1) {
                x[i] = i * 2;
            }
        }
    )");

    optimize(fn);
    verify_ir(fn);

    std::string ir = dump_ir_str(fn);
    return ir.find("BB_for_cond") != std::string::npos &&
           ir.find("BB_for_body") != std::string::npos &&
           ir.find("BB_for_upd") != std::string::npos &&
           ir.find("store.i32") != std::string::npos;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // 1. Verifier tests
    run_test("verify_valid",                   test_verify_valid);
    run_test("verify_no_blocks",               test_verify_no_blocks);
    run_test("verify_empty_block",             test_verify_empty_block);
    run_test("verify_no_terminator",           test_verify_no_terminator);
    run_test("verify_terminator_middle",       test_verify_terminator_middle);
    run_test("verify_undef_branch_target",     test_verify_undef_branch_target);
    run_test("verify_condbr_non_i32",          test_verify_condbr_non_i32);
    run_test("verify_use_before_def",          test_verify_use_before_def);
    run_test("verify_type_mismatch_binary",    test_verify_type_mismatch_binary);

    // 2. Constant folding tests
    run_test("const_fold_before_after_int",    test_const_fold_before_after_int);
    run_test("const_fold_float",               test_const_fold_float);
    run_test("const_fold_neg",                 test_const_fold_neg);
    run_test("const_fold_int_wraparound",      test_const_fold_int_wraparound);
    run_test("const_fold_unsafe_div_zero",     test_const_fold_unsafe_div_zero);
    run_test("const_fold_unsafe_int_min_neg1", test_const_fold_unsafe_int_min_neg1);
    run_test("const_fold_unsafe_float_div_0",  test_const_fold_unsafe_float_div_zero);

    // 3. Dead code elimination tests
    run_test("dce_cascading",                  test_dce_cascading);
    run_test("dce_keep_stores",                test_dce_keep_stores);
    run_test("dce_keep_var_movs",              test_dce_keep_var_movs);

    // 4. Common subexpression elimination tests
    run_test("cse_arithmetic",                 test_cse_arithmetic);
    run_test("cse_commutative",                test_cse_commutative);
    run_test("cse_load_elimination",           test_cse_load_elimination);
    run_test("cse_unsafe_store_inv_load",      test_cse_unsafe_store_invalidates_load);

    // 5. Full pipeline tests
    run_test("pipeline_saxpy",                 test_pipeline_saxpy);
    run_test("pipeline_for_loop",              test_pipeline_for_loop);

    // Summary
    std::cout << "\n" << tests_passed << "/" << tests_run << " Phase 2 tests passed.\n";
    if (!failures.empty()) {
        std::cout << "Failures:\n";
        for (const auto& f : failures) std::cout << "  - " << f << "\n";
        return 1;
    }
    return 0;
}
