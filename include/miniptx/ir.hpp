#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "miniptx/ast.hpp"

namespace miniptx {

// ---------------------------------------------------------------------------
// Values and Types in IR
// ---------------------------------------------------------------------------

// Expression temporaries (%t0, %t1, ...) are assigned exactly once.
// Variable slots (%var_x, %var_i, ...) are mutable virtual registers that can
// be assigned multiple times across different basic blocks.
enum class ValueKind {
    Temp,  // Expression temporary (assigned once)
    Var    // Mutable variable slot (virtual register)
};

struct Value {
    ValueKind kind = ValueKind::Temp;
    std::string name;
    Type type;

    bool is_temp() const { return kind == ValueKind::Temp; }
    bool is_var()  const { return kind == ValueKind::Var; }

    bool operator==(const Value& o) const {
        return kind == o.kind && name == o.name && type == o.type;
    }
    bool operator!=(const Value& o) const { return !(*this == o); }
};

// Returns e.g. "i32", "f32", "i32*", "f32*"
std::string ir_type_name(const Type& t);

// ---------------------------------------------------------------------------
// Instructions and Opcodes
// ---------------------------------------------------------------------------

enum class IROp {
    // Constants
    Const,
    // Arithmetic
    Add, Sub, Mul, Div,
    Neg,
    // Comparisons
    Lt, Gt, Le, Ge, Eq, Ne,
    // Memory
    Load, Store,
    // Copies
    Mov,
    // GPU builtins (tid.x, bid.x, bdim.x)
    Builtin,
    // Control flow terminators
    Br, CondBr, Ret
};

const char* ir_op_name(IROp op);
bool is_terminator(IROp op);

struct Instruction {
    IROp op;
    Type type;                       // Result/element type
    std::optional<Value> dest;       // Destination value (nullopt for Store, Br, CondBr, Ret)
    std::vector<Value> args;         // Operands

    // Constant immediate values (when op == IROp::Const)
    int int_val = 0;
    float float_val = 0.0f;

    // Builtin name (when op == IROp::Builtin)
    std::string builtin_name;

    // Branch targets (when op == IROp::Br or CondBr)
    std::string target_block;        // For Br
    std::string then_block;          // For CondBr
    std::string else_block;          // For CondBr

    int line = 0;                    // Source line number for debug info (.loc)
    int col = 0;                     // Source column number for debug info (.loc)
};

// ---------------------------------------------------------------------------
// Basic Blocks and Functions
// ---------------------------------------------------------------------------

struct BasicBlock {
    std::string label;
    std::vector<Instruction> instrs;
};

struct IRParam {
    std::string name;
    Type type;
};

struct IRFunction {
    std::string name;
    std::vector<IRParam> params;
    std::vector<BasicBlock> blocks;
    std::vector<Value> var_slots;    // All declared mutable variable slots
    int entry_line = 1;
    int entry_col = 1;
    int end_line = 1;
};

// ---------------------------------------------------------------------------
// IR Printer, Verifier, Lowering, and Optimization Passes
// ---------------------------------------------------------------------------

// Print human-readable IR to string or stdout
std::string dump_ir_str(const IRFunction& fn);
void dump_ir(const IRFunction& fn);

// Verify that the IR is well-formed. Throws std::runtime_error on failure.
void verify_ir(const IRFunction& fn);

// Lower AST to three-address IR with basic blocks
IRFunction lower_to_ir(const KernelDecl& ast);

// Optimization passes (block-local)
// Each pass modifies the function in place and returns number of optimizations made.
int constant_folding_pass(IRFunction& fn);
int dead_code_elimination_pass(IRFunction& fn);
int common_subexpression_elimination_pass(IRFunction& fn);

// Run the full optimization pipeline: ConstFold -> DCE -> CSE -> DCE
// Verifies IR after lowering and after each pass.
void optimize(IRFunction& fn);

// Legacy stub
void lower_and_optimize();

}  // namespace miniptx
