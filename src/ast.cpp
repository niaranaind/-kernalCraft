#include "miniptx/ast.hpp"
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace miniptx {

// ============================================================================
// Utility functions
// ============================================================================

const char* binop_str(BinOp op) {
    switch (op) {
        case BinOp::Add: return "+";
        case BinOp::Sub: return "-";
        case BinOp::Mul: return "*";
        case BinOp::Div: return "/";
        case BinOp::Lt:  return "<";
        case BinOp::Gt:  return ">";
        case BinOp::Le:  return "<=";
        case BinOp::Ge:  return ">=";
        case BinOp::Eq:  return "==";
        case BinOp::Ne:  return "!=";
    }
    return "?";
}

// Each expression variant stores its own line number.
// This helper extracts it regardless of which variant is active.
int Expr::line() const {
    return std::visit([](const auto& n) -> int { return n.line; }, node);
}

int Expr::col() const {
    return std::visit([](const auto& n) -> int { return n.col; }, node);
}

// ============================================================================
// AST printer  (--dump-ast)
// ============================================================================

// We indent by two spaces per nesting level to show tree structure.
static void print_indent(int depth) {
    for (int i = 0; i < depth; ++i) std::cout << "  ";
}

static void dump_expr(const Expr& e, int depth);
static void dump_stmts(const std::vector<StmtPtr>& stmts, int depth);

static void dump_expr(const Expr& e, int depth) {
    print_indent(depth);
    std::visit([&](const auto& n) {
        using T = std::decay_t<decltype(n)>;

        if constexpr (std::is_same_v<T, IntLitExpr>) {
            std::cout << "IntLit(" << n.value << ")\n";
        } else if constexpr (std::is_same_v<T, FloatLitExpr>) {
            std::cout << "FloatLit(" << n.value << ")\n";
        } else if constexpr (std::is_same_v<T, VarExpr>) {
            std::cout << "Var(" << n.name << ")\n";
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            std::cout << "BinOp(" << binop_str(n.op) << ")\n";
            dump_expr(*n.lhs, depth + 1);
            dump_expr(*n.rhs, depth + 1);
        } else if constexpr (std::is_same_v<T, IndexExpr>) {
            std::cout << "Index(" << n.array_name << ")\n";
            dump_expr(*n.index, depth + 1);
        } else if constexpr (std::is_same_v<T, UnaryMinusExpr>) {
            std::cout << "UnaryMinus\n";
            dump_expr(*n.operand, depth + 1);
        } else if constexpr (std::is_same_v<T, BuiltinExpr>) {
            std::cout << "Builtin(" << n.name << ")\n";
        }
    }, e.node);
}

static void dump_stmt(const Stmt& s, int depth) {
    print_indent(depth);
    std::visit([&](const auto& n) {
        using T = std::decay_t<decltype(n)>;

        if constexpr (std::is_same_v<T, VarDeclStmt>) {
            std::cout << "VarDecl(" << type_name(n.type) << " " << n.name << ")\n";
            dump_expr(*n.init, depth + 1);
        } else if constexpr (std::is_same_v<T, AssignStmt>) {
            std::cout << "Assign(" << n.name << ")\n";
            dump_expr(*n.value, depth + 1);
        } else if constexpr (std::is_same_v<T, IndexAssignStmt>) {
            std::cout << "IndexAssign(" << n.array_name << ")\n";
            print_indent(depth + 1);
            std::cout << "index:\n";
            dump_expr(*n.index, depth + 2);
            print_indent(depth + 1);
            std::cout << "value:\n";
            dump_expr(*n.value, depth + 2);
        } else if constexpr (std::is_same_v<T, IfStmt>) {
            std::cout << "If\n";
            print_indent(depth + 1);
            std::cout << "cond:\n";
            dump_expr(*n.cond, depth + 2);
            print_indent(depth + 1);
            std::cout << "then:\n";
            dump_stmts(n.then_body, depth + 2);
            if (!n.else_body.empty()) {
                print_indent(depth + 1);
                std::cout << "else:\n";
                dump_stmts(n.else_body, depth + 2);
            }
        } else if constexpr (std::is_same_v<T, ForStmt>) {
            std::cout << "For\n";
            print_indent(depth + 1);
            std::cout << "init: " << type_name({n.init_type.base, n.init_type.is_ptr})
                      << " " << n.init_name << " =\n";
            dump_expr(*n.init_expr, depth + 2);
            print_indent(depth + 1);
            std::cout << "cond:\n";
            dump_expr(*n.cond, depth + 2);
            print_indent(depth + 1);
            std::cout << "update: " << n.update_name << " =\n";
            dump_expr(*n.update_expr, depth + 2);
            print_indent(depth + 1);
            std::cout << "body:\n";
            dump_stmts(n.body, depth + 2);
        }
    }, s.node);
}

static void dump_stmts(const std::vector<StmtPtr>& stmts, int depth) {
    for (const auto& s : stmts) {
        dump_stmt(*s, depth);
    }
}

void dump_ast(const KernelDecl& kernel) {
    std::cout << "Kernel(" << kernel.name << ")\n";
    std::cout << "  params:\n";
    for (const auto& p : kernel.params) {
        std::cout << "    " << type_name(p.type) << " " << p.name << "\n";
    }
    std::cout << "  body:\n";
    dump_stmts(kernel.body, 2);
}

// ============================================================================
// Type checker
// ============================================================================

// The type checker walks the AST after parsing. It uses a symbol table
// (a map from variable name to Type) to track what's in scope. It checks:
//   - Every variable is declared before use.
//   - No variable is declared twice in the same scope.
//   - Operand types match (e.g., you can't add an int to a float*).
//   - Array indexing is only done on pointers.
//   - Comparisons return int.

// A simple scope: we don't have nested functions, just one flat scope per
// kernel plus the for-loop's init variable. We use a vector of maps so that
// for-loop and if-block scopes can shadow outer variables. When we exit the
// scope, we pop the map.
class TypeChecker {
public:
    void check(const KernelDecl& kernel) {
        // Start a fresh scope and add all kernel parameters.
        push_scope();
        for (const auto& p : kernel.params) {
            declare(p.name, p.type, kernel.line);
        }
        check_stmts(kernel.body);
        pop_scope();
    }

private:
    // Stack of scopes. Each scope maps variable names to types.
    std::vector<std::unordered_map<std::string, Type>> scopes_;

    void push_scope() { scopes_.emplace_back(); }
    void pop_scope()  { scopes_.pop_back(); }

    // Declare a variable in the current (innermost) scope.
    void declare(const std::string& name, const Type& t, int line) {
        auto& scope = scopes_.back();
        if (scope.count(name)) {
            throw std::runtime_error(
                "line " + std::to_string(line) + ": duplicate declaration of '" + name + "'");
        }
        scope[name] = t;
    }

    // Look up a variable by searching from innermost scope outward.
    Type lookup(const std::string& name, int line) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
        }
        throw std::runtime_error(
            "line " + std::to_string(line) + ": undeclared variable '" + name + "'");
    }

    // --- Type-check an expression and return its type ---
    Type check_expr(const Expr& e) {
        return std::visit([&](const auto& n) -> Type {
            using T = std::decay_t<decltype(n)>;

            if constexpr (std::is_same_v<T, IntLitExpr>) {
                return {BaseType::Int, false};
            } else if constexpr (std::is_same_v<T, FloatLitExpr>) {
                return {BaseType::Float, false};
            } else if constexpr (std::is_same_v<T, VarExpr>) {
                return lookup(n.name, n.line);
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                Type lt = check_expr(*n.lhs);
                Type rt = check_expr(*n.rhs);

                // Pointers can't be used with arithmetic or comparison operators
                // (we keep the language simple — no pointer arithmetic).
                if (lt.is_ptr || rt.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot use pointer type in arithmetic expression");
                }

                // Both operands must be the same scalar type.
                if (lt != rt) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": type mismatch in binary expression: " +
                        type_name(lt) + " vs " + type_name(rt));
                }

                // Comparison operators always produce int.
                bool is_cmp = (n.op == BinOp::Lt || n.op == BinOp::Gt ||
                               n.op == BinOp::Le || n.op == BinOp::Ge ||
                               n.op == BinOp::Eq || n.op == BinOp::Ne);
                if (is_cmp) return {BaseType::Int, false};

                // Arithmetic operators preserve the operand type.
                return lt;
            } else if constexpr (std::is_same_v<T, IndexExpr>) {
                Type arr_type = lookup(n.array_name, n.line);
                if (!arr_type.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot index non-pointer type '" + type_name(arr_type) + "'");
                }
                Type idx_type = check_expr(*n.index);
                if (idx_type.base != BaseType::Int || idx_type.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": array index must be int, got " + type_name(idx_type));
                }
                // Dereferencing a pointer gives the base scalar type.
                return {arr_type.base, false};
            } else if constexpr (std::is_same_v<T, UnaryMinusExpr>) {
                // Unary minus: operand must be a scalar (int or float), not a pointer.
                Type t = check_expr(*n.operand);
                if (t.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot negate pointer type");
                }
                return t;  // result has the same type as the operand
            } else if constexpr (std::is_same_v<T, BuiltinExpr>) {
                // tid.x, bid.x, bdim.x are all int.
                return {BaseType::Int, false};
            }
        }, e.node);
    }

    // --- Type-check a list of statements ---
    void check_stmts(const std::vector<StmtPtr>& stmts) {
        for (const auto& s : stmts) check_stmt(*s);
    }

    void check_stmt(const Stmt& s) {
        std::visit([&](const auto& n) {
            using T = std::decay_t<decltype(n)>;

            if constexpr (std::is_same_v<T, VarDeclStmt>) {
                Type init_type = check_expr(*n.init);
                // The initializer must match the declared type.
                if (init_type != n.type) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot initialize " + type_name(n.type) +
                        " with " + type_name(init_type));
                }
                declare(n.name, n.type, n.line);
            } else if constexpr (std::is_same_v<T, AssignStmt>) {
                Type var_type = lookup(n.name, n.line);
                Type val_type = check_expr(*n.value);
                if (var_type != val_type) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot assign " + type_name(val_type) +
                        " to " + type_name(var_type));
                }
            } else if constexpr (std::is_same_v<T, IndexAssignStmt>) {
                Type arr_type = lookup(n.array_name, n.line);
                if (!arr_type.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot index non-pointer type '" + type_name(arr_type) + "'");
                }
                Type idx_type = check_expr(*n.index);
                if (idx_type.base != BaseType::Int || idx_type.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": array index must be int, got " + type_name(idx_type));
                }
                Type val_type = check_expr(*n.value);
                Type elem_type = {arr_type.base, false};
                if (val_type != elem_type) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot assign " + type_name(val_type) +
                        " to " + type_name(elem_type) + " element");
                }
            } else if constexpr (std::is_same_v<T, IfStmt>) {
                Type cond_type = check_expr(*n.cond);
                if (cond_type.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": if condition cannot be a pointer type");
                }
                // The then/else bodies get their own scope so local vars
                // declared inside don't leak out.
                push_scope();
                check_stmts(n.then_body);
                pop_scope();
                if (!n.else_body.empty()) {
                    push_scope();
                    check_stmts(n.else_body);
                    pop_scope();
                }
            } else if constexpr (std::is_same_v<T, ForStmt>) {
                // The for-loop init variable is scoped to the loop.
                push_scope();
                Type init_type = check_expr(*n.init_expr);
                if (init_type != n.init_type) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot initialize " + type_name(n.init_type) +
                        " with " + type_name(init_type));
                }
                declare(n.init_name, n.init_type, n.line);

                Type cond_type = check_expr(*n.cond);
                if (cond_type.is_ptr) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": for condition cannot be a pointer type");
                }

                // Check the update assignment.
                Type upd_var_type = lookup(n.update_name, n.line);
                Type upd_val_type = check_expr(*n.update_expr);
                if (upd_var_type != upd_val_type) {
                    throw std::runtime_error(
                        "line " + std::to_string(n.line) +
                        ": cannot assign " + type_name(upd_val_type) +
                        " to " + type_name(upd_var_type) + " in for-update");
                }

                check_stmts(n.body);
                pop_scope();
            }
        }, s.node);
    }
};

void type_check(const KernelDecl& kernel) {
    TypeChecker tc;
    tc.check(kernel);
}

}  // namespace miniptx
