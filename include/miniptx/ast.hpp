#pragma once
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace miniptx {

// ============================================================================
// Types
// ============================================================================

// The four types our language supports.
// Base is the scalar type; is_ptr says whether it's a pointer to that scalar.
enum class BaseType { Int, Float };

struct Type {
    BaseType base;
    bool is_ptr = false;  // true for int* and float*

    bool operator==(const Type& o) const {
        return base == o.base && is_ptr == o.is_ptr;
    }
    bool operator!=(const Type& o) const { return !(*this == o); }
};

// Human-readable type name for error messages.
inline std::string type_name(const Type& t) {
    std::string s = (t.base == BaseType::Int) ? "int" : "float";
    if (t.is_ptr) s += "*";
    return s;
}

// ============================================================================
// Expressions
// ============================================================================

// Forward-declare so we can use unique_ptr<Expr> inside variants.
struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

// A binary operator.
enum class BinOp {
    Add, Sub, Mul, Div,                 // arithmetic
    Lt, Gt, Le, Ge, Eq, Ne             // comparison (result is int 0/1)
};

// Returns a printable string for a binary operator ("+", "<", etc.)
const char* binop_str(BinOp op);

// An integer literal like 42.
struct IntLitExpr   { int value; int line; };

// A float literal like 3.14.
struct FloatLitExpr { float value; int line; };

// A variable reference like "n" or "x".
struct VarExpr      { std::string name; int line; };

// A binary operation like "a + b".
struct BinaryExpr   { BinOp op; ExprPtr lhs, rhs; int line; };

// An array index like "x[i]" — dereferences pointer x at offset i.
struct IndexExpr    { std::string array_name; ExprPtr index; int line; };

// Unary minus like "-x" or "-3".  Negates a scalar (int or float).
struct UnaryMinusExpr { ExprPtr operand; int line; };

// A GPU built-in like tid.x, bid.x, bdim.x.
// We store the full name (e.g. "tid.x") rather than parsing it further.
struct BuiltinExpr  { std::string name; int line; };

// An expression is one of the above alternatives.
// We use a variant so each node is a plain struct — no virtual functions.
struct Expr {
    std::variant<IntLitExpr, FloatLitExpr, VarExpr, BinaryExpr,
                 IndexExpr, UnaryMinusExpr, BuiltinExpr> node;

    // Convenience: line number of this expression (delegates to whichever
    // variant is active).
    int line() const;
};

// Helper to create an ExprPtr wrapping a node.
template <typename T>
ExprPtr make_expr(T&& node) {
    auto e = std::make_unique<Expr>();
    e->node = std::forward<T>(node);
    return e;
}

// ============================================================================
// Statements
// ============================================================================

struct Stmt;
using StmtPtr = std::unique_ptr<Stmt>;

// Variable declaration with initializer: "int i = expr;"
struct VarDeclStmt {
    Type type;
    std::string name;
    ExprPtr init;
    int line;
};

// Assignment to a variable: "x = expr;"
struct AssignStmt {
    std::string name;
    ExprPtr value;
    int line;
};

// Indexed assignment: "y[i] = expr;"  (store through a pointer)
struct IndexAssignStmt {
    std::string array_name;
    ExprPtr index;
    ExprPtr value;
    int line;
};

// If statement: "if (cond) { ... }" with optional else block.
struct IfStmt {
    ExprPtr cond;
    std::vector<StmtPtr> then_body;
    std::vector<StmtPtr> else_body;   // empty if no else clause
    int line;
};

// For loop: "for (int i = 0; i < n; i = i + 1) { ... }"
struct ForStmt {
    // The init part is a variable declaration.
    Type init_type;
    std::string init_name;
    ExprPtr init_expr;
    // The condition is an expression.
    ExprPtr cond;
    // The update is an assignment (name = expr).
    std::string update_name;
    ExprPtr update_expr;
    // The loop body.
    std::vector<StmtPtr> body;
    int line;
};

struct Stmt {
    std::variant<VarDeclStmt, AssignStmt, IndexAssignStmt,
                 IfStmt, ForStmt> node;
};

// Helper to create a StmtPtr wrapping a node.
template <typename T>
StmtPtr make_stmt(T&& node) {
    auto s = std::make_unique<Stmt>();
    s->node = std::forward<T>(node);
    return s;
}

// ============================================================================
// Top-level: kernel declaration
// ============================================================================

struct Param {
    Type type;
    std::string name;
};

struct KernelDecl {
    std::string name;
    std::vector<Param> params;
    std::vector<StmtPtr> body;
    int line;   // line of the "kernel" keyword
};

// ============================================================================
// AST printer (--dump-ast)
// ============================================================================

// Print a human-readable representation of the AST to stdout.
void dump_ast(const KernelDecl& kernel);

// ============================================================================
// Type checker
// ============================================================================

// Walk the AST and check types. Throws std::runtime_error on type errors.
// After this returns successfully, the program is well-typed.
void type_check(const KernelDecl& kernel);

}  // namespace miniptx
