//
// Phase 3 tests: PTX Back End Emitter & Structural Golden Verifications.
//
// These tests run on CPU without requiring an NVIDIA GPU or ptxas.
// They verify:
//   1. PTX Structure: .version, .target, .address_size, .entry declaration
//   2. Register Coverage: every %r, %f, %rd, %p register index used in the
//      body is strictly less than the declared count <N>.
//   3. Branch Target Validity: every bra and condbr target label exists as a
//      defined block label.
//   4. Instruction specifics: mul.lo.s32, div.rn.f32, mul.rn.f32, add.rn.f32,
//      float != using setp.neu.f32, 0f hex float notation, and comparison fusion.
//   5. Full golden check for all 5 target kernels (vecadd, saxpy, scale, ewmax, fill).
//

#include <algorithm>
#include <cassert>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "miniptx/lexer.hpp"
#include "miniptx/parser.hpp"
#include "miniptx/ast.hpp"
#include "miniptx/ir.hpp"
#include "miniptx/ptx_emitter.hpp"

using namespace miniptx;

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

static std::string compile_to_ptx(const std::string& src) {
    auto tokens = lex(src);
    auto kernel = parse(tokens);
    type_check(kernel);
    auto ir_fn = lower_to_ir(kernel);
    optimize(ir_fn);
    return emit_ptx(ir_fn);
}

// ---------------------------------------------------------------------------
// Structural Validator for PTX Text
// ---------------------------------------------------------------------------

static bool verify_ptx_structure(const std::string& ptx, const std::string& expected_entry) {
    // 1. Directives check
    if (ptx.find(".version " + std::string(DEFAULT_PTX_VERSION)) == std::string::npos) {
        std::cerr << "missing .version directive\n";
        return false;
    }
    if (ptx.find(".target " + std::string(DEFAULT_PTX_TARGET)) == std::string::npos) {
        std::cerr << "missing .target directive\n";
        return false;
    }
    if (ptx.find(".address_size 64") == std::string::npos) {
        std::cerr << "missing .address_size 64 directive\n";
        return false;
    }

    // 2. Entry point check
    std::string entry_decl = ".visible .entry " + expected_entry;
    if (ptx.find(entry_decl) == std::string::npos) {
        std::cerr << "missing entry declaration: " << entry_decl << "\n";
        return false;
    }

    // 3. Extract declared register limits: .reg .b32 %r<N>;
    int max_r_decl = 0, max_f_decl = 0, max_rd_decl = 0, max_p_decl = 0;

    std::regex r_decl_regex(R"(\.reg\s+\.b32\s+%r<(\d+)>)");
    std::regex f_decl_regex(R"(\.reg\s+\.f32\s+%f<(\d+)>)");
    std::regex rd_decl_regex(R"(\.reg\s+\.b64\s+%rd<(\d+)>)");
    std::regex p_decl_regex(R"(\.reg\s+\.pred\s+%p<(\d+)>)");

    std::smatch match;
    if (std::regex_search(ptx, match, r_decl_regex))  max_r_decl = std::stoi(match[1]);
    if (std::regex_search(ptx, match, f_decl_regex))  max_f_decl = std::stoi(match[1]);
    if (std::regex_search(ptx, match, rd_decl_regex)) max_rd_decl = std::stoi(match[1]);
    if (std::regex_search(ptx, match, p_decl_regex))  max_p_decl = std::stoi(match[1]);

    // 4. Gather all defined block labels: "Label:"
    std::unordered_set<std::string> defined_labels;
    std::regex label_def_regex(R"(^([A-Za-z0-9_]+):)");

    // 5. Check all register usages and branch targets line by line
    std::stringstream ss(ptx);
    std::string line;
    std::vector<std::string> branch_targets;

    std::regex reg_use_regex(R"(%(r|f|rd|p)(\d+))");
    std::regex bra_regex(R"(bra\s+([A-Za-z0-9_]+);)");

    while (std::getline(ss, line)) {
        // Trim leading whitespace
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        std::string trimmed = line.substr(first);

        if (std::regex_search(trimmed, match, label_def_regex)) {
            defined_labels.insert(match[1]);
        }

        // Search for branch instructions
        if (std::regex_search(trimmed, match, bra_regex)) {
            branch_targets.push_back(match[1]);
        }

        // Skip declaration lines when checking register usage bounds
        if (trimmed.rfind(".reg", 0) == 0) continue;

        auto words_begin = std::sregex_iterator(trimmed.begin(), trimmed.end(), reg_use_regex);
        auto words_end = std::sregex_iterator();
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch m = *i;
            std::string prefix = m[1];
            int idx = std::stoi(m[2]);

            if (prefix == "r" && idx >= max_r_decl) {
                std::cerr << "Register %r" << idx << " exceeds declared count " << max_r_decl << "\n";
                return false;
            }
            if (prefix == "f" && idx >= max_f_decl) {
                std::cerr << "Register %f" << idx << " exceeds declared count " << max_f_decl << "\n";
                return false;
            }
            if (prefix == "rd" && idx >= max_rd_decl) {
                std::cerr << "Register %rd" << idx << " exceeds declared count " << max_rd_decl << "\n";
                return false;
            }
            if (prefix == "p" && idx >= max_p_decl) {
                std::cerr << "Register %p" << idx << " exceeds declared count " << max_p_decl << "\n";
                return false;
            }
        }
    }

    // 6. Check that every branch target label was defined
    for (const auto& target : branch_targets) {
        if (defined_labels.find(target) == defined_labels.end()) {
            std::cerr << "Branch targets undefined label: '" << target << "'\n";
            return false;
        }
    }

    return true;
}

// ============================================================================
// Kernel Tests (All 5 Kernels)
// ============================================================================

bool test_kernel_saxpy_ptx() {
    std::string src = R"(
        kernel saxpy(float a, float* x, float* y, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { y[i] = a * x[i] + y[i]; }
        }
    )";
    std::string ptx = compile_to_ptx(src);
    if (!verify_ptx_structure(ptx, "saxpy")) return false;

    // Checks essential instructions
    if (ptx.find("mul.lo.s32") == std::string::npos) return false;
    if (ptx.find("mul.rn.f32") == std::string::npos) return false;
    if (ptx.find("add.rn.f32") == std::string::npos) return false;
    if (ptx.find("cvta.to.global.u64") == std::string::npos) return false;
    if (ptx.find("ld.global.f32") == std::string::npos) return false;
    if (ptx.find("st.global.f32") == std::string::npos) return false;
    return true;
}

bool test_kernel_vecadd_ptx() {
    std::string src = R"(
        kernel vecadd(int* a, int* b, int* c, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { c[i] = a[i] + b[i]; }
        }
    )";
    std::string ptx = compile_to_ptx(src);
    if (!verify_ptx_structure(ptx, "vecadd")) return false;

    if (ptx.find("add.s32") == std::string::npos) return false;
    if (ptx.find("ld.global.s32") == std::string::npos) return false;
    if (ptx.find("st.global.s32") == std::string::npos) return false;
    return true;
}

bool test_kernel_scale_ptx() {
    std::string src = R"(
        kernel scale(float* x, float s, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) { x[i] = x[i] * s; }
        }
    )";
    std::string ptx = compile_to_ptx(src);
    if (!verify_ptx_structure(ptx, "scale")) return false;

    if (ptx.find(".param .f32 param_s") == std::string::npos) return false;
    if (ptx.find("mul.rn.f32") == std::string::npos) return false;
    return true;
}

bool test_kernel_ewmax_ptx() {
    std::string src = R"(
        kernel ewmax(float* a, float* b, float* c, int n) {
            int i = tid.x + bid.x * bdim.x;
            if (i < n) {
                float va = a[i];
                float vb = b[i];
                if (va > vb) {
                    c[i] = va;
                } else {
                    c[i] = vb;
                }
            }
        }
    )";
    std::string ptx = compile_to_ptx(src);
    if (!verify_ptx_structure(ptx, "ewmax")) return false;

    if (ptx.find("setp.gt.f32") == std::string::npos) return false;
    return true;
}

bool test_kernel_fill_ptx() {
    std::string src = R"(
        kernel fill(int* x, int n) {
            for (int i = 0; i < n; i = i + 1) {
                x[i] = i * 2;
            }
        }
    )";
    std::string ptx = compile_to_ptx(src);
    if (!verify_ptx_structure(ptx, "fill")) return false;

    if (ptx.find("BB_for_cond") == std::string::npos) return false;
    if (ptx.find("BB_for_body") == std::string::npos) return false;
    if (ptx.find("BB_for_upd") == std::string::npos) return false;
    if (ptx.find("mul.lo.s32") == std::string::npos) return false;
    return true;
}

// ============================================================================
// Instruction Specific Tests (User Corrections)
// ============================================================================

// Correction 1: Lower 32-bit integer multiply to mul.lo.s32 (not mul.s32)
bool test_ptx_mul_lo_s32() {
    std::string src = "kernel k(int a, int b) { int c = a * b; }";
    std::string ptx = compile_to_ptx(src);
    return ptx.find("mul.lo.s32") != std::string::npos &&
           ptx.find("mul.s32") == std::string::npos;
}

// Correction 2: Use div.rn.f32, mul.rn.f32, add.rn.f32, sub.rn.f32
bool test_ptx_float_rounding_rn() {
    std::string src = R"(
        kernel k(float a, float b) {
            float c = a / b;
            float d = a * b;
            float e = a + b;
            float f = a - b;
        }
    )";
    std::string ptx = compile_to_ptx(src);
    if (ptx.find("div.rn.f32") == std::string::npos) return false;
    if (ptx.find("mul.rn.f32") == std::string::npos) return false;
    if (ptx.find("add.rn.f32") == std::string::npos) return false;
    if (ptx.find("sub.rn.f32") == std::string::npos) return false;
    return true;
}

// Correction 3: Float != must use setp.neu.f32; other comparisons use ordered forms
bool test_ptx_float_ne_neu() {
    std::string src_ne = "kernel k(float a, float b) { if (a != b) {} }";
    std::string ptx_ne = compile_to_ptx(src_ne);
    if (ptx_ne.find("setp.neu.f32") == std::string::npos) return false;

    std::string src_lt = "kernel k(float a, float b) { if (a < b) {} }";
    std::string ptx_lt = compile_to_ptx(src_lt);
    if (ptx_lt.find("setp.lt.f32") == std::string::npos) return false;

    return true;
}

// Float constants must be formatted as 0fHHHHHHHH hex notation
bool test_ptx_float_hex_format() {
    std::string src = "kernel k(float* out) { out[0] = 2.5; }";
    std::string ptx = compile_to_ptx(src);
    // 2.5f in IEEE 754 is 0x40200000 -> 0f40200000
    return ptx.find("0f40200000") != std::string::npos;
}

// Correction 5: Comparison fusion vs materialization
bool test_ptx_fused_vs_materialized() {
    // 1. Fused: comparison temporary is ONLY used by condbr
    std::string src_fused = "kernel k(int a, int b) { if (a < b) {} }";
    std::string ptx_fused = compile_to_ptx(src_fused);
    // Should NOT contain selp.s32
    if (ptx_fused.find("selp.s32") != std::string::npos) return false;

    // 2. Materialized: comparison result is stored into a variable
    std::string src_mat = "kernel k(int a, int b, int* out) { int flag = (a < b); out[0] = flag; }";
    std::string ptx_mat = compile_to_ptx(src_mat);
    // MUST contain selp.s32 to convert predicate to integer 1 or 0
    if (ptx_mat.find("selp.s32") == std::string::npos) return false;

    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Golden kernel tests
    run_test("kernel_saxpy_ptx",        test_kernel_saxpy_ptx);
    run_test("kernel_vecadd_ptx",       test_kernel_vecadd_ptx);
    run_test("kernel_scale_ptx",        test_kernel_scale_ptx);
    run_test("kernel_ewmax_ptx",        test_kernel_ewmax_ptx);
    run_test("kernel_fill_ptx",         test_kernel_fill_ptx);

    // Specific instruction & emitter corrections
    run_test("ptx_mul_lo_s32",          test_ptx_mul_lo_s32);
    run_test("ptx_float_rounding_rn",   test_ptx_float_rounding_rn);
    run_test("ptx_float_ne_neu",        test_ptx_float_ne_neu);
    run_test("ptx_float_hex_format",    test_ptx_float_hex_format);
    run_test("ptx_fused_vs_mat",        test_ptx_fused_vs_materialized);

    // Summary
    std::cout << "\n" << tests_passed << "/" << tests_run << " Phase 3 tests passed.\n";
    if (!failures.empty()) {
        std::cout << "Failures:\n";
        for (const auto& f : failures) std::cout << "  - " << f << "\n";
        return 1;
    }
    return 0;
}
