#include "miniptx/ir.hpp"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace miniptx {

// ---------------------------------------------------------------------------
// Helpers and Display Names
// ---------------------------------------------------------------------------

std::string ir_type_name(const Type& t) {
    std::string s = (t.base == BaseType::Int) ? "i32" : "f32";
    if (t.is_ptr) s += "*";
    return s;
}

const char* ir_op_name(IROp op) {
    switch (op) {
        case IROp::Const:   return "const";
        case IROp::Add:     return "add";
        case IROp::Sub:     return "sub";
        case IROp::Mul:     return "mul";
        case IROp::Div:     return "div";
        case IROp::Neg:     return "neg";
        case IROp::Lt:      return "cmp_lt";
        case IROp::Gt:      return "cmp_gt";
        case IROp::Le:      return "cmp_le";
        case IROp::Ge:      return "cmp_ge";
        case IROp::Eq:      return "cmp_eq";
        case IROp::Ne:      return "cmp_ne";
        case IROp::Load:    return "load";
        case IROp::Store:   return "store";
        case IROp::Mov:     return "mov";
        case IROp::Builtin: return "builtin";
        case IROp::Br:      return "br";
        case IROp::CondBr:  return "condbr";
        case IROp::Ret:     return "ret";
    }
    return "?";
}

bool is_terminator(IROp op) {
    return op == IROp::Br || op == IROp::CondBr || op == IROp::Ret;
}

// ---------------------------------------------------------------------------
// IR Printer
// ---------------------------------------------------------------------------

std::string dump_ir_str(const IRFunction& fn) {
    std::ostringstream ss;
    ss << "kernel " << fn.name << "(";
    for (size_t i = 0; i < fn.params.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << ir_type_name(fn.params[i].type) << " %param_" << fn.params[i].name;
    }
    ss << "):\n";

    for (const auto& bb : fn.blocks) {
        ss << bb.label << ":\n";
        for (const auto& inst : bb.instrs) {
            ss << "  ";
            if (inst.dest.has_value()) {
                ss << inst.dest->name << " = ";
            }

            ss << ir_op_name(inst.op);
            if (inst.op != IROp::Br && inst.op != IROp::CondBr && inst.op != IROp::Ret) {
                ss << "." << ir_type_name(inst.type);
            }

            if (inst.op == IROp::Const) {
                if (inst.type.base == BaseType::Int) {
                    ss << " " << inst.int_val;
                } else {
                    ss << " " << std::fixed << std::setprecision(6) << inst.float_val;
                }
            } else if (inst.op == IROp::Builtin) {
                ss << " " << inst.builtin_name;
            } else if (inst.op == IROp::Br) {
                ss << " " << inst.target_block;
            } else if (inst.op == IROp::CondBr) {
                ss << " " << inst.args[0].name << ", " << inst.then_block << ", " << inst.else_block;
            } else {
                for (size_t i = 0; i < inst.args.size(); ++i) {
                    ss << (i == 0 ? " " : ", ") << inst.args[i].name;
                }
            }
            ss << "\n";
        }
    }
    return ss.str();
}

void dump_ir(const IRFunction& fn) {
    std::cout << dump_ir_str(fn);
}

// ---------------------------------------------------------------------------
// IR Verifier
// ---------------------------------------------------------------------------

void verify_ir(const IRFunction& fn) {
    if (fn.blocks.empty()) {
        throw std::runtime_error("IR verification error: function has no basic blocks");
    }

    std::unordered_set<std::string> block_labels;
    for (const auto& bb : fn.blocks) {
        if (bb.label.empty()) {
            throw std::runtime_error("IR verification error: basic block has empty label");
        }
        if (!block_labels.insert(bb.label).second) {
            throw std::runtime_error("IR verification error: duplicate basic block label '" + bb.label + "'");
        }
    }

    // Map of known defined temporary names to their types
    std::unordered_map<std::string, Type> defined_temps;
    // Map of valid mutable variable slots
    std::unordered_map<std::string, Type> var_slot_types;

    for (const auto& slot : fn.var_slots) {
        var_slot_types[slot.name] = slot.type;
    }

    // Function parameter temporaries are pre-defined at entry
    for (const auto& param : fn.params) {
        std::string param_temp = "%param_" + param.name;
        defined_temps[param_temp] = param.type;
    }

    for (const auto& bb : fn.blocks) {
        if (bb.instrs.empty()) {
            throw std::runtime_error("IR verification error: block '" + bb.label + "' has no instructions");
        }

        // Check terminator invariants:
        // 1. Last instruction must be a terminator
        // 2. No non-last instruction can be a terminator
        for (size_t i = 0; i < bb.instrs.size(); ++i) {
            const auto& inst = bb.instrs[i];
            bool is_last = (i + 1 == bb.instrs.size());
            bool term = is_terminator(inst.op);

            if (term && !is_last) {
                throw std::runtime_error("IR verification error: terminator in the middle of block '" + bb.label + "'");
            }
            if (!term && is_last) {
                throw std::runtime_error("IR verification error: block '" + bb.label + "' does not end with a terminator");
            }

            // Check branch targets
            if (inst.op == IROp::Br) {
                if (block_labels.find(inst.target_block) == block_labels.end()) {
                    throw std::runtime_error("IR verification error: branch to undefined block '" + inst.target_block + "'");
                }
            } else if (inst.op == IROp::CondBr) {
                if (inst.args.size() != 1) {
                    throw std::runtime_error("IR verification error: condbr must take exactly 1 condition operand");
                }
                if (inst.args[0].type != Type{BaseType::Int, false}) {
                    throw std::runtime_error("IR verification error: condbr condition must be i32");
                }
                if (block_labels.find(inst.then_block) == block_labels.end()) {
                    throw std::runtime_error("IR verification error: condbr then target '" + inst.then_block + "' undefined");
                }
                if (block_labels.find(inst.else_block) == block_labels.end()) {
                    throw std::runtime_error("IR verification error: condbr else target '" + inst.else_block + "' undefined");
                }
            }

            // Check that every operand temporary has been defined before use
            for (const auto& arg : inst.args) {
                if (arg.is_temp()) {
                    auto it = defined_temps.find(arg.name);
                    if (it == defined_temps.end()) {
                        throw std::runtime_error("IR verification error: operand '" + arg.name + "' used before definition");
                    }
                    if (it->second != arg.type) {
                        throw std::runtime_error("IR verification error: operand '" + arg.name + "' type mismatch with definition");
                    }
                } else if (arg.is_var()) {
                    auto it = var_slot_types.find(arg.name);
                    if (it == var_slot_types.end()) {
                        throw std::runtime_error("IR verification error: variable slot '" + arg.name + "' not registered");
                    }
                    if (it->second != arg.type) {
                        throw std::runtime_error("IR verification error: variable slot '" + arg.name + "' type mismatch");
                    }
                }
            }

            // Check instruction-specific type constraints
            switch (inst.op) {
                case IROp::Const:
                    if (!inst.dest.has_value() || !inst.dest->is_temp()) {
                        throw std::runtime_error("IR verification error: const must have a temporary destination");
                    }
                    if (inst.dest->type != inst.type || inst.type.is_ptr) {
                        throw std::runtime_error("IR verification error: const type mismatch");
                    }
                    break;

                case IROp::Add:
                case IROp::Sub:
                case IROp::Mul:
                case IROp::Div:
                    if (inst.args.size() != 2) {
                        throw std::runtime_error("IR verification error: binary op requires 2 operands");
                    }
                    if (!inst.dest.has_value() || !inst.dest->is_temp()) {
                        throw std::runtime_error("IR verification error: binary op must have a temporary destination");
                    }
                    if (inst.type.is_ptr || inst.dest->type != inst.type ||
                        inst.args[0].type != inst.type || inst.args[1].type != inst.type) {
                        throw std::runtime_error("IR verification error: binary op operand/result type mismatch");
                    }
                    break;

                case IROp::Neg:
                    if (inst.args.size() != 1) {
                        throw std::runtime_error("IR verification error: neg requires 1 operand");
                    }
                    if (!inst.dest.has_value() || !inst.dest->is_temp()) {
                        throw std::runtime_error("IR verification error: neg must have a temporary destination");
                    }
                    if (inst.type.is_ptr || inst.dest->type != inst.type || inst.args[0].type != inst.type) {
                        throw std::runtime_error("IR verification error: neg operand/result type mismatch");
                    }
                    break;

                case IROp::Lt:
                case IROp::Gt:
                case IROp::Le:
                case IROp::Ge:
                case IROp::Eq:
                case IROp::Ne:
                    if (inst.args.size() != 2) {
                        throw std::runtime_error("IR verification error: comparison requires 2 operands");
                    }
                    if (!inst.dest.has_value() || !inst.dest->is_temp()) {
                        throw std::runtime_error("IR verification error: comparison must have a temporary destination");
                    }
                    if (inst.dest->type != Type{BaseType::Int, false}) {
                        throw std::runtime_error("IR verification error: comparison destination must be i32");
                    }
                    if (inst.type.is_ptr || inst.args[0].type != inst.type || inst.args[1].type != inst.type) {
                        throw std::runtime_error("IR verification error: comparison operand types must match");
                    }
                    break;

                case IROp::Load:
                    if (inst.args.size() != 2) {
                        throw std::runtime_error("IR verification error: load requires 2 operands (ptr, index)");
                    }
                    if (!inst.dest.has_value() || !inst.dest->is_temp()) {
                        throw std::runtime_error("IR verification error: load must have a temporary destination");
                    }
                    if (inst.dest->type != Type{inst.type.base, false}) {
                        throw std::runtime_error("IR verification error: load destination type mismatch");
                    }
                    if (inst.args[0].type != Type{inst.type.base, true}) {
                        throw std::runtime_error("IR verification error: load pointer operand must be pointer type");
                    }
                    if (inst.args[1].type != Type{BaseType::Int, false}) {
                        throw std::runtime_error("IR verification error: load index operand must be i32");
                    }
                    break;

                case IROp::Store:
                    if (inst.dest.has_value()) {
                        throw std::runtime_error("IR verification error: store must not have a destination");
                    }
                    if (inst.args.size() != 3) {
                        throw std::runtime_error("IR verification error: store requires 3 operands (ptr, index, val)");
                    }
                    if (inst.args[0].type != Type{inst.type.base, true}) {
                        throw std::runtime_error("IR verification error: store pointer operand must be pointer type");
                    }
                    if (inst.args[1].type != Type{BaseType::Int, false}) {
                        throw std::runtime_error("IR verification error: store index operand must be i32");
                    }
                    if (inst.args[2].type != Type{inst.type.base, false}) {
                        throw std::runtime_error("IR verification error: store value operand type mismatch");
                    }
                    break;

                case IROp::Mov:
                    if (inst.args.size() != 1) {
                        throw std::runtime_error("IR verification error: mov requires exactly 1 operand");
                    }
                    if (!inst.dest.has_value()) {
                        throw std::runtime_error("IR verification error: mov must have a destination");
                    }
                    if (inst.dest->type != inst.type || inst.args[0].type != inst.type) {
                        throw std::runtime_error("IR verification error: mov operand and destination types must match");
                    }
                    break;

                case IROp::Builtin:
                    if (!inst.dest.has_value() || !inst.dest->is_temp()) {
                        throw std::runtime_error("IR verification error: builtin must have a temporary destination");
                    }
                    if (inst.dest->type != Type{BaseType::Int, false}) {
                        throw std::runtime_error("IR verification error: builtin destination must be i32");
                    }
                    if (inst.builtin_name != "tid.x" && inst.builtin_name != "bid.x" && inst.builtin_name != "bdim.x") {
                        throw std::runtime_error("IR verification error: unknown builtin '" + inst.builtin_name + "'");
                    }
                    break;

                case IROp::Br:
                case IROp::CondBr:
                case IROp::Ret:
                    if (inst.dest.has_value()) {
                        throw std::runtime_error("IR verification error: terminator cannot have a destination");
                    }
                    break;
            }

            // Register newly defined temporary
            if (inst.dest.has_value() && inst.dest->is_temp()) {
                defined_temps[inst.dest->name] = inst.dest->type;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lowering: AST -> Three-Address IR with Basic Blocks
// ---------------------------------------------------------------------------

class IRLowerer {
public:
    explicit IRLowerer(const KernelDecl& ast)
        : ast_(ast) {}

    IRFunction lower() {
        fn_.name = ast_.name;

        // Entry block BB0
        create_block("BB0");
        cur_block_idx_ = 0;
        push_scope();

        // Register parameters
        for (const auto& p : ast_.params) {
            fn_.params.push_back({p.name, p.type});

            Value param_temp{ValueKind::Temp, "%param_" + p.name, p.type};
            Value slot = new_var_slot(p.name, p.type);
            var_env_.back()[p.name] = slot;

            Instruction mov;
            mov.op = IROp::Mov;
            mov.type = p.type;
            mov.dest = slot;
            mov.args = {param_temp};
            mov.line = ast_.line;
            emit(mov);
        }

        // Lower body
        lower_stmts(ast_.body);
        pop_scope();

        // Ensure final block ends in ret
        if (cur_block().instrs.empty() || !is_terminator(cur_block().instrs.back().op)) {
            Instruction ret;
            ret.op = IROp::Ret;
            ret.line = ast_.line;
            emit(ret);
        }

        verify_ir(fn_);
        return fn_;
    }

private:
    const KernelDecl& ast_;
    IRFunction fn_;
    int temp_counter_ = 0;
    int block_counter_ = 0;
    int var_counter_ = 0;
    size_t cur_block_idx_ = 0;

    std::vector<std::unordered_map<std::string, Value>> var_env_;

    void push_scope() { var_env_.emplace_back(); }
    void pop_scope()  { var_env_.pop_back(); }

    BasicBlock& cur_block() {
        return fn_.blocks[cur_block_idx_];
    }

    size_t create_block(const std::string& prefix = "BB") {
        std::string label = prefix;
        if (prefix != "BB0") {
            label = prefix + "_" + std::to_string(block_counter_++);
        }
        fn_.blocks.push_back({label, {}});
        return fn_.blocks.size() - 1;
    }

    void emit(Instruction inst) {
        fn_.blocks[cur_block_idx_].instrs.push_back(std::move(inst));
    }

    Value new_temp(Type t) {
        return {ValueKind::Temp, "%t" + std::to_string(temp_counter_++), t};
    }

    Value new_var_slot(const std::string& name, Type t) {
        std::string slot_name = "%var_" + name + "_" + std::to_string(var_counter_++);
        Value slot{ValueKind::Var, slot_name, t};
        fn_.var_slots.push_back(slot);
        return slot;
    }

    Value lookup_var(const std::string& name) {
        for (auto it = var_env_.rbegin(); it != var_env_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
        }
        throw std::runtime_error("internal error: variable '" + name + "' not found in IR environment");
    }

    Value lower_expr(const Expr& expr) {
        return std::visit([&](const auto& n) -> Value {
            using T = std::decay_t<decltype(n)>;

            if constexpr (std::is_same_v<T, IntLitExpr>) {
                Value dest = new_temp(Type{BaseType::Int, false});
                Instruction inst;
                inst.op = IROp::Const;
                inst.type = Type{BaseType::Int, false};
                inst.dest = dest;
                inst.int_val = n.value;
                inst.line = n.line;
                emit(inst);
                return dest;

            } else if constexpr (std::is_same_v<T, FloatLitExpr>) {
                Value dest = new_temp(Type{BaseType::Float, false});
                Instruction inst;
                inst.op = IROp::Const;
                inst.type = Type{BaseType::Float, false};
                inst.dest = dest;
                inst.float_val = n.value;
                inst.line = n.line;
                emit(inst);
                return dest;

            } else if constexpr (std::is_same_v<T, VarExpr>) {
                Value slot = lookup_var(n.name);
                Value dest = new_temp(slot.type);
                Instruction inst;
                inst.op = IROp::Mov;
                inst.type = slot.type;
                inst.dest = dest;
                inst.args = {slot};
                inst.line = n.line;
                emit(inst);
                return dest;

            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                Value lhs = lower_expr(*n.lhs);
                Value rhs = lower_expr(*n.rhs);

                IROp op;
                Type res_type = lhs.type;
                switch (n.op) {
                    case BinOp::Add: op = IROp::Add; break;
                    case BinOp::Sub: op = IROp::Sub; break;
                    case BinOp::Mul: op = IROp::Mul; break;
                    case BinOp::Div: op = IROp::Div; break;
                    case BinOp::Lt:  op = IROp::Lt; res_type = Type{BaseType::Int, false}; break;
                    case BinOp::Gt:  op = IROp::Gt; res_type = Type{BaseType::Int, false}; break;
                    case BinOp::Le:  op = IROp::Le; res_type = Type{BaseType::Int, false}; break;
                    case BinOp::Ge:  op = IROp::Ge; res_type = Type{BaseType::Int, false}; break;
                    case BinOp::Eq:  op = IROp::Eq; res_type = Type{BaseType::Int, false}; break;
                    case BinOp::Ne:  op = IROp::Ne; res_type = Type{BaseType::Int, false}; break;
                }

                Value dest = new_temp(res_type);
                Instruction inst;
                inst.op = op;
                inst.type = lhs.type;  // Operand type
                inst.dest = dest;
                inst.args = {lhs, rhs};
                inst.line = n.line;
                emit(inst);
                return dest;

            } else if constexpr (std::is_same_v<T, UnaryMinusExpr>) {
                Value operand = lower_expr(*n.operand);
                Value dest = new_temp(operand.type);
                Instruction inst;
                inst.op = IROp::Neg;
                inst.type = operand.type;
                inst.dest = dest;
                inst.args = {operand};
                inst.line = n.line;
                emit(inst);
                return dest;

            } else if constexpr (std::is_same_v<T, IndexExpr>) {
                Value slot = lookup_var(n.array_name);
                Value ptr_temp = new_temp(slot.type);
                Instruction mov_ptr;
                mov_ptr.op = IROp::Mov;
                mov_ptr.type = slot.type;
                mov_ptr.dest = ptr_temp;
                mov_ptr.args = {slot};
                mov_ptr.line = n.line;
                emit(mov_ptr);

                Value idx = lower_expr(*n.index);
                Type elem_type{slot.type.base, false};
                Value dest = new_temp(elem_type);

                Instruction load_inst;
                load_inst.op = IROp::Load;
                load_inst.type = elem_type;
                load_inst.dest = dest;
                load_inst.args = {ptr_temp, idx};
                load_inst.line = n.line;
                emit(load_inst);
                return dest;

            } else if constexpr (std::is_same_v<T, BuiltinExpr>) {
                Value dest = new_temp(Type{BaseType::Int, false});
                Instruction inst;
                inst.op = IROp::Builtin;
                inst.type = Type{BaseType::Int, false};
                inst.dest = dest;
                inst.builtin_name = n.name;
                inst.line = n.line;
                emit(inst);
                return dest;
            }
        }, expr.node);
    }

    void lower_stmts(const std::vector<StmtPtr>& stmts) {
        for (const auto& s : stmts) {
            lower_stmt(*s);
        }
    }

    void lower_stmt(const Stmt& stmt) {
        std::visit([&](const auto& n) {
            using T = std::decay_t<decltype(n)>;

            if constexpr (std::is_same_v<T, VarDeclStmt>) {
                Value init_val = lower_expr(*n.init);
                Value slot = new_var_slot(n.name, n.type);
                var_env_.back()[n.name] = slot;

                Instruction mov;
                mov.op = IROp::Mov;
                mov.type = n.type;
                mov.dest = slot;
                mov.args = {init_val};
                mov.line = n.line;
                emit(mov);

            } else if constexpr (std::is_same_v<T, AssignStmt>) {
                Value val = lower_expr(*n.value);
                Value slot = lookup_var(n.name);

                Instruction mov;
                mov.op = IROp::Mov;
                mov.type = slot.type;
                mov.dest = slot;
                mov.args = {val};
                mov.line = n.line;
                emit(mov);

            } else if constexpr (std::is_same_v<T, IndexAssignStmt>) {
                Value slot = lookup_var(n.array_name);
                Value ptr_temp = new_temp(slot.type);
                Instruction mov_ptr;
                mov_ptr.op = IROp::Mov;
                mov_ptr.type = slot.type;
                mov_ptr.dest = ptr_temp;
                mov_ptr.args = {slot};
                mov_ptr.line = n.line;
                emit(mov_ptr);

                Value idx = lower_expr(*n.index);
                Value val = lower_expr(*n.value);

                Instruction store_inst;
                store_inst.op = IROp::Store;
                store_inst.type = Type{slot.type.base, false};
                store_inst.args = {ptr_temp, idx, val};
                store_inst.line = n.line;
                emit(store_inst);

            } else if constexpr (std::is_same_v<T, IfStmt>) {
                Value cond_val = lower_expr(*n.cond);

                size_t then_idx = create_block("BB_then");
                size_t else_idx = n.else_body.empty() ? then_idx : create_block("BB_else");
                size_t merge_idx = create_block("BB_merge");

                Instruction cond_br;
                cond_br.op = IROp::CondBr;
                cond_br.type = Type{BaseType::Int, false};
                cond_br.args = {cond_val};
                cond_br.then_block = fn_.blocks[then_idx].label;
                cond_br.else_block = n.else_body.empty() ? fn_.blocks[merge_idx].label : fn_.blocks[else_idx].label;
                cond_br.line = n.line;
                emit(cond_br);

                // Lower then branch
                cur_block_idx_ = then_idx;
                push_scope();
                lower_stmts(n.then_body);
                pop_scope();
                if (cur_block().instrs.empty() || !is_terminator(cur_block().instrs.back().op)) {
                    Instruction br;
                    br.op = IROp::Br;
                    br.target_block = fn_.blocks[merge_idx].label;
                    br.line = n.line;
                    emit(br);
                }

                // Lower else branch (if any)
                if (!n.else_body.empty()) {
                    cur_block_idx_ = else_idx;
                    push_scope();
                    lower_stmts(n.else_body);
                    pop_scope();
                    if (cur_block().instrs.empty() || !is_terminator(cur_block().instrs.back().op)) {
                        Instruction br;
                        br.op = IROp::Br;
                        br.target_block = fn_.blocks[merge_idx].label;
                        br.line = n.line;
                        emit(br);
                    }
                }

                cur_block_idx_ = merge_idx;

            } else if constexpr (std::is_same_v<T, ForStmt>) {
                push_scope();
                // 1. Loop Init
                Value init_val = lower_expr(*n.init_expr);
                Value slot = new_var_slot(n.init_name, n.init_type);
                var_env_.back()[n.init_name] = slot;

                Instruction init_mov;
                init_mov.op = IROp::Mov;
                init_mov.type = n.init_type;
                init_mov.dest = slot;
                init_mov.args = {init_val};
                init_mov.line = n.line;
                emit(init_mov);

                size_t cond_idx = create_block("BB_for_cond");
                size_t body_idx = create_block("BB_for_body");
                size_t upd_idx  = create_block("BB_for_upd");
                size_t exit_idx = create_block("BB_for_exit");

                Instruction br_cond;
                br_cond.op = IROp::Br;
                br_cond.target_block = fn_.blocks[cond_idx].label;
                br_cond.line = n.line;
                emit(br_cond);

                // 2. Loop Condition
                cur_block_idx_ = cond_idx;
                Value cond_val = lower_expr(*n.cond);
                Instruction cond_br;
                cond_br.op = IROp::CondBr;
                cond_br.type = Type{BaseType::Int, false};
                cond_br.args = {cond_val};
                cond_br.then_block = fn_.blocks[body_idx].label;
                cond_br.else_block = fn_.blocks[exit_idx].label;
                cond_br.line = n.line;
                emit(cond_br);

                // 3. Loop Body
                cur_block_idx_ = body_idx;
                push_scope();
                lower_stmts(n.body);
                pop_scope();
                if (cur_block().instrs.empty() || !is_terminator(cur_block().instrs.back().op)) {
                    Instruction br_upd;
                    br_upd.op = IROp::Br;
                    br_upd.target_block = fn_.blocks[upd_idx].label;
                    br_upd.line = n.line;
                    emit(br_upd);
                }

                // 4. Loop Update
                cur_block_idx_ = upd_idx;
                Value upd_val = lower_expr(*n.update_expr);
                Value upd_slot = lookup_var(n.update_name);
                Instruction upd_mov;
                upd_mov.op = IROp::Mov;
                upd_mov.type = upd_slot.type;
                upd_mov.dest = upd_slot;
                upd_mov.args = {upd_val};
                upd_mov.line = n.line;
                emit(upd_mov);

                Instruction br_back;
                br_back.op = IROp::Br;
                br_back.target_block = fn_.blocks[cond_idx].label;
                br_back.line = n.line;
                emit(br_back);

                // 5. Exit Block
                cur_block_idx_ = exit_idx;
                pop_scope();
            }
        }, stmt.node);
    }
};

IRFunction lower_to_ir(const KernelDecl& ast) {
    IRLowerer lowerer(ast);
    return lowerer.lower();
}

// ---------------------------------------------------------------------------
// Optimization Passes (Block-Local)
// ---------------------------------------------------------------------------

// Constant Value container for Constant Folding
struct ConstVal {
    Type type;
    int int_val = 0;
    float float_val = 0.0f;
};

int constant_folding_pass(IRFunction& fn) {
    int total_folded = 0;

    for (auto& bb : fn.blocks) {
        std::unordered_map<std::string, ConstVal> const_map;

        for (auto& inst : bb.instrs) {
            // Track existing constants
            if (inst.op == IROp::Const && inst.dest.has_value()) {
                const_map[inst.dest->name] = {inst.type, inst.int_val, inst.float_val};
                continue;
            }

            // Unary negation folding
            if (inst.op == IROp::Neg && inst.dest.has_value() && inst.args.size() == 1) {
                auto it = const_map.find(inst.args[0].name);
                if (it != const_map.end()) {
                    if (inst.type.base == BaseType::Int) {
                        int res = static_cast<int>(0u - static_cast<uint32_t>(it->second.int_val));
                        inst.op = IROp::Const;
                        inst.int_val = res;
                        inst.args.clear();
                        const_map[inst.dest->name] = {inst.type, res, 0.0f};
                        ++total_folded;
                    } else if (inst.type.base == BaseType::Float) {
                        float res = -it->second.float_val;
                        if (!std::isnan(res)) {
                            inst.op = IROp::Const;
                            inst.float_val = res;
                            inst.args.clear();
                            const_map[inst.dest->name] = {inst.type, 0, res};
                            ++total_folded;
                        }
                    }
                    continue;
                }
            }

            // Binary arithmetic and comparison folding
            if (inst.args.size() == 2 && inst.dest.has_value()) {
                auto it0 = const_map.find(inst.args[0].name);
                auto it1 = const_map.find(inst.args[1].name);

                if (it0 != const_map.end() && it1 != const_map.end()) {
                    if (inst.type.base == BaseType::Int) {
                        int a = it0->second.int_val;
                        int b = it1->second.int_val;

                        // Integer division safety:
                        // 1. Never fold division by zero!
                        // 2. Never fold INT_MIN / -1 (causes UB/SIGFPE)!
                        if (inst.op == IROp::Div) {
                            if (b == 0) continue;
                            if (a == INT32_MIN && b == -1) continue;
                        }

                        int res = 0;
                        bool folded = true;
                        switch (inst.op) {
                            case IROp::Add: res = static_cast<int>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b)); break;
                            case IROp::Sub: res = static_cast<int>(static_cast<uint32_t>(a) - static_cast<uint32_t>(b)); break;
                            case IROp::Mul: res = static_cast<int>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b)); break;
                            case IROp::Div: res = a / b; break;
                            case IROp::Lt:  res = (a < b)  ? 1 : 0; break;
                            case IROp::Gt:  res = (a > b)  ? 1 : 0; break;
                            case IROp::Le:  res = (a <= b) ? 1 : 0; break;
                            case IROp::Ge:  res = (a >= b) ? 1 : 0; break;
                            case IROp::Eq:  res = (a == b) ? 1 : 0; break;
                            case IROp::Ne:  res = (a != b) ? 1 : 0; break;
                            default: folded = false; break;
                        }

                        if (folded) {
                            inst.op = IROp::Const;
                            inst.type = inst.dest->type; // Result type (always i32 for int arithmetic/cmp)
                            inst.int_val = res;
                            inst.args.clear();
                            const_map[inst.dest->name] = {inst.type, res, 0.0f};
                            ++total_folded;
                        }
                    } else if (inst.type.base == BaseType::Float) {
                        float a = it0->second.float_val;
                        float b = it1->second.float_val;

                        if (std::isnan(a) || std::isnan(b)) continue;

                        if (inst.op == IROp::Div && b == 0.0f) continue;

                        if (inst.op == IROp::Add || inst.op == IROp::Sub ||
                            inst.op == IROp::Mul || inst.op == IROp::Div) {
                            float res = 0.0f;
                            switch (inst.op) {
                                case IROp::Add: res = a + b; break;
                                case IROp::Sub: res = a - b; break;
                                case IROp::Mul: res = a * b; break;
                                case IROp::Div: res = a / b; break;
                                default: break;
                            }
                            // Skip NaN or infinity results
                            if (!std::isnan(res) && !std::isinf(res)) {
                                inst.op = IROp::Const;
                                inst.float_val = res;
                                inst.args.clear();
                                const_map[inst.dest->name] = {inst.type, 0, res};
                                ++total_folded;
                            }
                        } else {
                            // Float comparison returns i32
                            int res = 0;
                            bool folded = true;
                            switch (inst.op) {
                                case IROp::Lt: res = (a < b)  ? 1 : 0; break;
                                case IROp::Gt: res = (a > b)  ? 1 : 0; break;
                                case IROp::Le: res = (a <= b) ? 1 : 0; break;
                                case IROp::Ge: res = (a >= b) ? 1 : 0; break;
                                case IROp::Eq: res = (a == b) ? 1 : 0; break;
                                case IROp::Ne: res = (a != b) ? 1 : 0; break;
                                default: folded = false; break;
                            }
                            if (folded) {
                                inst.op = IROp::Const;
                                inst.type = Type{BaseType::Int, false};
                                inst.int_val = res;
                                inst.args.clear();
                                const_map[inst.dest->name] = {inst.type, res, 0.0f};
                                ++total_folded;
                            }
                        }
                    }
                }
            }
        }
    }

    return total_folded;
}

int dead_code_elimination_pass(IRFunction& fn) {
    int total_eliminated = 0;

    for (auto& bb : fn.blocks) {
        bool changed = true;
        while (changed) {
            changed = false;

            // 1. Gather all temporaries used as operands within this block
            std::unordered_set<std::string> used_temps;
            for (const auto& inst : bb.instrs) {
                for (const auto& arg : inst.args) {
                    if (arg.is_temp()) {
                        used_temps.insert(arg.name);
                    }
                }
            }

            // 2. Identify instructions to keep:
            // Must NEVER delete:
            // - Terminators (Br, CondBr, Ret)
            // - Store instructions (side-effects on global memory)
            // - Mov instructions into variable slots (virtual registers across blocks)
            std::vector<Instruction> surviving;
            for (auto& inst : bb.instrs) {
                bool is_dead = false;

                if (!is_terminator(inst.op) && inst.op != IROp::Store) {
                    if (inst.dest.has_value() && inst.dest->is_temp()) {
                        if (used_temps.find(inst.dest->name) == used_temps.end()) {
                            is_dead = true;
                        }
                    }
                }

                if (is_dead) {
                    changed = true;
                    ++total_eliminated;
                } else {
                    surviving.push_back(std::move(inst));
                }
            }
            bb.instrs = std::move(surviving);
        }
    }

    return total_eliminated;
}

// Key structure for Common Subexpression Elimination
struct CSEKey {
    IROp op;
    Type type;
    std::string arg0;
    std::string arg1;
    int int_val = 0;
    float float_val = 0.0f;
    std::string builtin_name;

    bool operator==(const CSEKey& o) const {
        return op == o.op && type == o.type &&
               arg0 == o.arg0 && arg1 == o.arg1 &&
               int_val == o.int_val && float_val == o.float_val &&
               builtin_name == o.builtin_name;
    }
};

struct CSEKeyHash {
    size_t operator()(const CSEKey& k) const {
        size_t h1 = std::hash<int>()(static_cast<int>(k.op));
        size_t h2 = std::hash<int>()(static_cast<int>(k.type.base)) ^ (k.type.is_ptr ? 0x9e3779b9 : 0);
        size_t h3 = std::hash<std::string>()(k.arg0) ^ (std::hash<std::string>()(k.arg1) << 1);
        size_t h4 = std::hash<int>()(k.int_val) ^ std::hash<std::string>()(k.builtin_name);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

int common_subexpression_elimination_pass(IRFunction& fn) {
    int total_eliminated = 0;

    for (auto& bb : fn.blocks) {
        std::unordered_map<CSEKey, Value, CSEKeyHash> expr_table;

        for (size_t i = 0; i < bb.instrs.size(); ++i) {
            auto& inst = bb.instrs[i];

            // Invalidate ALL Load expressions whenever a Store occurs!
            if (inst.op == IROp::Store) {
                for (auto it = expr_table.begin(); it != expr_table.end(); ) {
                    if (it->first.op == IROp::Load) {
                        it = expr_table.erase(it);
                    } else {
                        ++it;
                    }
                }
                continue;
            }

            // Invalidate Mov entries reading this variable slot when assigned!
            if (inst.op == IROp::Mov && inst.dest.has_value() && inst.dest->is_var()) {
                std::string assigned_slot = inst.dest->name;
                for (auto it = expr_table.begin(); it != expr_table.end(); ) {
                    if (it->first.op == IROp::Mov && it->first.arg0 == assigned_slot) {
                        it = expr_table.erase(it);
                    } else {
                        ++it;
                    }
                }
                continue;
            }

            // Only CSE-eligible operations with a temporary destination
            bool eligible = (inst.op == IROp::Const   ||
                             inst.op == IROp::Add     || inst.op == IROp::Sub  ||
                             inst.op == IROp::Mul     || inst.op == IROp::Div  ||
                             inst.op == IROp::Neg     ||
                             inst.op == IROp::Lt      || inst.op == IROp::Gt   ||
                             inst.op == IROp::Le      || inst.op == IROp::Ge   ||
                             inst.op == IROp::Eq      || inst.op == IROp::Ne   ||
                             inst.op == IROp::Load    ||
                             inst.op == IROp::Builtin ||
                             (inst.op == IROp::Mov && !inst.args.empty() && inst.args[0].is_var()));

            if (!eligible || !inst.dest.has_value() || !inst.dest->is_temp()) {
                continue;
            }

            CSEKey key;
            key.op = inst.op;
            key.type = inst.type;
            if (!inst.args.empty()) key.arg0 = inst.args[0].name;
            if (inst.args.size() > 1) key.arg1 = inst.args[1].name;
            key.int_val = inst.int_val;
            key.float_val = inst.float_val;
            key.builtin_name = inst.builtin_name;

            // Commutative symmetry for Add, Mul, Eq, Ne
            if (inst.op == IROp::Add || inst.op == IROp::Mul ||
                inst.op == IROp::Eq  || inst.op == IROp::Ne) {
                if (key.arg0 > key.arg1) {
                    std::swap(key.arg0, key.arg1);
                }
            }

            auto it = expr_table.find(key);
            if (it != expr_table.end()) {
                Value orig_val = it->second;
                std::string old_dest_name = inst.dest->name;

                // Replace all subsequent uses of old_dest_name in the rest of this block
                for (size_t j = i + 1; j < bb.instrs.size(); ++j) {
                    for (auto& arg : bb.instrs[j].args) {
                        if (arg.name == old_dest_name) {
                            arg = orig_val;
                        }
                    }
                }

                // Turn this redundant instruction into a mov, which DCE will prune
                inst.op = IROp::Mov;
                inst.type = orig_val.type;
                inst.args = {orig_val};
                ++total_eliminated;
            } else {
                expr_table[key] = *inst.dest;
            }
        }
    }

    return total_eliminated;
}

void optimize(IRFunction& fn) {
    verify_ir(fn);

    // Pipeline: ConstFold -> DCE -> CSE -> DCE
    constant_folding_pass(fn);
    verify_ir(fn);

    dead_code_elimination_pass(fn);
    verify_ir(fn);

    common_subexpression_elimination_pass(fn);
    verify_ir(fn);

    dead_code_elimination_pass(fn);
    verify_ir(fn);
}

void lower_and_optimize() {
    // Legacy stub for backward compatibility
}

}  // namespace miniptx
