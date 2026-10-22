#include "miniptx/ptx_emitter.hpp"
#include <cstring>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace miniptx {

namespace {

// Format single-precision IEEE 754 float into PTX exact hex format: 0fHHHHHHHH
std::string format_float_ptx(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    std::ostringstream ss;
    ss << "0f" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << bits;
    return ss.str();
}

bool is_comparison_op(IROp op) {
    return op == IROp::Lt || op == IROp::Gt || op == IROp::Le ||
           op == IROp::Ge || op == IROp::Eq || op == IROp::Ne;
}

const char* ptx_cmp_op(IROp op, bool is_float) {
    if (is_float) {
        // Requirement 3: Float != uses setp.neu.f32 (not equal or unordered)
        // to match C NaN semantics. Other float comparisons use ordered forms.
        switch (op) {
            case IROp::Lt: return "setp.lt.f32";
            case IROp::Gt: return "setp.gt.f32";
            case IROp::Le: return "setp.le.f32";
            case IROp::Ge: return "setp.ge.f32";
            case IROp::Eq: return "setp.eq.f32";
            case IROp::Ne: return "setp.neu.f32";
            default: return "setp.eq.f32";
        }
    } else {
        switch (op) {
            case IROp::Lt: return "setp.lt.s32";
            case IROp::Gt: return "setp.gt.s32";
            case IROp::Le: return "setp.le.s32";
            case IROp::Ge: return "setp.ge.s32";
            case IROp::Eq: return "setp.eq.s32";
            case IROp::Ne: return "setp.ne.s32";
            default: return "setp.eq.s32";
        }
    }
}

class PTXEmitter {
public:
    PTXEmitter(const IRFunction& fn, const EmitterOptions& options)
        : fn_(fn), options_(options) {}

    std::string emit() {
        analyze_uses();
        allocate_registers();

        std::ostringstream ss;
        // Directives
        ss << ".version " << options_.ptx_version << "\n";
        ss << ".target " << options_.target << "\n";
        ss << ".address_size 64\n\n";

        // Kernel Entry Declaration
        ss << ".visible .entry " << fn_.name << "(\n";
        for (size_t i = 0; i < fn_.params.size(); ++i) {
            const auto& p = fn_.params[i];
            ss << "    .param ";
            if (p.type.is_ptr) {
                ss << ".u64";
            } else if (p.type.base == BaseType::Int) {
                ss << ".s32";
            } else {
                ss << ".f32";
            }
            ss << " param_" << p.name;
            if (i + 1 < fn_.params.size()) ss << ",";
            ss << "\n";
        }
        ss << ")\n{\n";

        // Register Declarations
        if (num_pred_ > 0) ss << "    .reg .pred   %p<" << num_pred_ << ">;\n";
        if (num_r_ > 0)    ss << "    .reg .b32    %r<" << num_r_ << ">;\n";
        if (num_f_ > 0)    ss << "    .reg .f32    %f<" << num_f_ << ">;\n";
        if (num_rd_ > 0)   ss << "    .reg .b64    %rd<" << num_rd_ << ">;\n";
        ss << "\n";

        // Load parameters from .param space
        for (const auto& p : fn_.params) {
            std::string param_temp = "%param_" + p.name;
            std::string reg = reg_map_[param_temp];

            if (p.type.is_ptr) {
                std::string raw_reg = param_raw_regs_[p.name];
                ss << "    ld.param.u64    " << raw_reg << ", [param_" << p.name << "];\n";
                ss << "    cvta.to.global.u64 " << reg << ", " << raw_reg << ";\n";
            } else if (p.type.base == BaseType::Int) {
                ss << "    ld.param.s32    " << reg << ", [param_" << p.name << "];\n";
            } else {
                ss << "    ld.param.f32    " << reg << ", [param_" << p.name << "];\n";
            }
        }
        ss << "\n";

        // Emit Basic Blocks
        for (size_t bi = 0; bi < fn_.blocks.size(); ++bi) {
            const auto& bb = fn_.blocks[bi];
            std::string next_label = (bi + 1 < fn_.blocks.size()) ? fn_.blocks[bi + 1].label : "";

            ss << bb.label << ":\n";
            for (const auto& inst : bb.instrs) {
                emit_instruction(ss, inst, next_label);
            }
        }

        ss << "}\n";
        return ss.str();
    }

private:
    const IRFunction& fn_;
    const EmitterOptions& options_;

    // Use count of every temporary name
    std::unordered_map<std::string, int> use_counts_;

    // Set of comparison temporaries that are ONLY used by a condbr
    // (these are fused into setp + branch without materializing with selp)
    std::unordered_set<std::string> fused_cmp_temps_;

    // Map from IR value name to allocated PTX register name
    std::unordered_map<std::string, std::string> reg_map_;

    // Pre-allocated raw generic pointer registers for pointer parameters
    std::unordered_map<std::string, std::string> param_raw_regs_;

    // Map from comparison temporary name to allocated predicate register (%pN)
    std::unordered_map<std::string, std::string> cmp_pred_map_;

    int num_r_ = 0;
    int num_f_ = 0;
    int num_rd_ = 0;
    int num_pred_ = 0;

    void analyze_uses() {
        for (const auto& bb : fn_.blocks) {
            for (const auto& inst : bb.instrs) {
                for (const auto& arg : inst.args) {
                    if (arg.is_temp()) {
                        use_counts_[arg.name]++;
                    }
                }
            }
        }

        // Gather all condition temporary names used by CondBr instructions
        std::unordered_set<std::string> condbr_args;
        for (const auto& bb : fn_.blocks) {
            for (const auto& inst : bb.instrs) {
                if (inst.op == IROp::CondBr && !inst.args.empty()) {
                    condbr_args.insert(inst.args[0].name);
                }
            }
        }

        // Identify comparison temporaries whose ONLY use is in a condbr
        for (const auto& bb : fn_.blocks) {
            for (const auto& inst : bb.instrs) {
                if (is_comparison_op(inst.op) && inst.dest.has_value()) {
                    const std::string& dest_name = inst.dest->name;
                    if (use_counts_[dest_name] == 1 && condbr_args.count(dest_name) == 1) {
                        fused_cmp_temps_.insert(dest_name);
                    }
                }
            }
        }
    }

    void allocate_registers() {
        // Reserve %rd0 as address scratch register for 64-bit array offset computations
        std::string rd_scratch = "%rd" + std::to_string(num_rd_++);
        (void)rd_scratch;

        // Allocate raw generic registers and global registers for parameters
        for (const auto& p : fn_.params) {
            std::string param_temp = "%param_" + p.name;
            if (p.type.is_ptr) {
                std::string raw_reg = "%rd" + std::to_string(num_rd_++);
                param_raw_regs_[p.name] = raw_reg;
                reg_map_[param_temp] = "%rd" + std::to_string(num_rd_++);
            } else if (p.type.base == BaseType::Int) {
                reg_map_[param_temp] = "%r" + std::to_string(num_r_++);
            } else {
                reg_map_[param_temp] = "%f" + std::to_string(num_f_++);
            }
        }

        // Allocate register for all variable slots
        for (const auto& slot : fn_.var_slots) {
            if (slot.type.is_ptr) {
                reg_map_[slot.name] = "%rd" + std::to_string(num_rd_++);
            } else if (slot.type.base == BaseType::Int) {
                reg_map_[slot.name] = "%r" + std::to_string(num_r_++);
            } else {
                reg_map_[slot.name] = "%f" + std::to_string(num_f_++);
            }
        }

        // Allocate register for temporaries defined by instructions
        for (const auto& bb : fn_.blocks) {
            for (const auto& inst : bb.instrs) {
                if (is_comparison_op(inst.op) && inst.dest.has_value()) {
                    // Comparisons always allocate a predicate register
                    std::string pred_reg = "%p" + std::to_string(num_pred_++);
                    cmp_pred_map_[inst.dest->name] = pred_reg;

                    // If comparison is NOT fused (i.e. used by something other than condbr),
                    // it also needs an integer register to hold 0 or 1 materialized via selp
                    if (fused_cmp_temps_.find(inst.dest->name) == fused_cmp_temps_.end()) {
                        reg_map_[inst.dest->name] = "%r" + std::to_string(num_r_++);
                    }
                    continue;
                }

                if (inst.dest.has_value() && inst.dest->is_temp()) {
                    const auto& d = *inst.dest;
                    if (reg_map_.find(d.name) == reg_map_.end()) {
                        if (d.type.is_ptr) {
                            reg_map_[d.name] = "%rd" + std::to_string(num_rd_++);
                        } else if (d.type.base == BaseType::Int) {
                            reg_map_[d.name] = "%r" + std::to_string(num_r_++);
                        } else {
                            reg_map_[d.name] = "%f" + std::to_string(num_f_++);
                        }
                    }
                }
            }
        }

        // Ensure at least 1 predicate register exists for any non-comparison condbr
        if (num_pred_ == 0) {
            num_pred_ = 1;
        }
    }

    void emit_instruction(std::ostringstream& ss, const Instruction& inst, const std::string& next_block) {
        switch (inst.op) {
            case IROp::Const: {
                std::string dest = reg_map_[inst.dest->name];
                if (inst.type.base == BaseType::Int) {
                    ss << "    mov.b32         " << dest << ", " << inst.int_val << ";\n";
                } else {
                    // Float constants use exact 0fHHHHHHHH hex notation
                    ss << "    mov.b32         " << dest << ", " << format_float_ptx(inst.float_val) << ";\n";
                }
                break;
            }

            case IROp::Add: {
                std::string dest = reg_map_[inst.dest->name];
                std::string a = reg_map_[inst.args[0].name];
                std::string b = reg_map_[inst.args[1].name];
                if (inst.type.base == BaseType::Int) {
                    ss << "    add.s32         " << dest << ", " << a << ", " << b << ";\n";
                } else {
                    // Requirement 2: add.rn.f32 prevents unwanted FMA fusion
                    ss << "    add.rn.f32      " << dest << ", " << a << ", " << b << ";\n";
                }
                break;
            }

            case IROp::Sub: {
                std::string dest = reg_map_[inst.dest->name];
                std::string a = reg_map_[inst.args[0].name];
                std::string b = reg_map_[inst.args[1].name];
                if (inst.type.base == BaseType::Int) {
                    ss << "    sub.s32         " << dest << ", " << a << ", " << b << ";\n";
                } else {
                    // Requirement 2: sub.rn.f32 prevents unwanted FMA fusion
                    ss << "    sub.rn.f32      " << dest << ", " << a << ", " << b << ";\n";
                }
                break;
            }

            case IROp::Mul: {
                std::string dest = reg_map_[inst.dest->name];
                std::string a = reg_map_[inst.args[0].name];
                std::string b = reg_map_[inst.args[1].name];
                if (inst.type.base == BaseType::Int) {
                    // Requirement 1: mul.lo.s32 for 32-bit integer multiplication
                    ss << "    mul.lo.s32      " << dest << ", " << a << ", " << b << ";\n";
                } else {
                    // Requirement 2: mul.rn.f32 prevents unwanted FMA fusion
                    ss << "    mul.rn.f32      " << dest << ", " << a << ", " << b << ";\n";
                }
                break;
            }

            case IROp::Div: {
                std::string dest = reg_map_[inst.dest->name];
                std::string a = reg_map_[inst.args[0].name];
                std::string b = reg_map_[inst.args[1].name];
                if (inst.type.base == BaseType::Int) {
                    ss << "    div.s32         " << dest << ", " << a << ", " << b << ";\n";
                } else {
                    // Requirement 2: div.rn.f32 for round-to-nearest float division
                    ss << "    div.rn.f32      " << dest << ", " << a << ", " << b << ";\n";
                }
                break;
            }

            case IROp::Neg: {
                std::string dest = reg_map_[inst.dest->name];
                std::string a = reg_map_[inst.args[0].name];
                if (inst.type.base == BaseType::Int) {
                    ss << "    neg.s32         " << dest << ", " << a << ";\n";
                } else {
                    ss << "    neg.f32         " << dest << ", " << a << ";\n";
                }
                break;
            }

            case IROp::Lt:
            case IROp::Gt:
            case IROp::Le:
            case IROp::Ge:
            case IROp::Eq:
            case IROp::Ne: {
                const std::string& dest_name = inst.dest->name;
                std::string pred = cmp_pred_map_[dest_name];
                std::string a = reg_map_[inst.args[0].name];
                std::string b = reg_map_[inst.args[1].name];
                bool is_float = (inst.args[0].type.base == BaseType::Float);

                // Emit setp condition
                ss << "    " << ptx_cmp_op(inst.op, is_float) << "    " << pred << ", " << a << ", " << b << ";\n";

                // Requirement 5: Comparison Temporary Fusion vs Materialization
                // If this comparison temporary is used beyond a single condbr (e.g. stored or
                // assigned to an int variable), materialize it with selp.s32 into integer 0 or 1.
                // Otherwise, it remains fused directly in predicate register 'pred'.
                if (fused_cmp_temps_.find(dest_name) == fused_cmp_temps_.end()) {
                    std::string int_dest = reg_map_[dest_name];
                    ss << "    selp.s32        " << int_dest << ", 1, 0, " << pred << ";\n";
                }
                break;
            }

            case IROp::Load: {
                std::string dest = reg_map_[inst.dest->name];
                std::string ptr = reg_map_[inst.args[0].name];
                std::string idx = reg_map_[inst.args[1].name];

                // 64-bit array element addressing: addr = ptr + idx * 4
                std::string offset_reg = "%rd0"; // use scratch %rd0
                ss << "    mul.wide.s32    " << offset_reg << ", " << idx << ", 4;\n";
                ss << "    add.s64         " << offset_reg << ", " << ptr << ", " << offset_reg << ";\n";

                if (inst.type.base == BaseType::Int) {
                    ss << "    ld.global.s32   " << dest << ", [" << offset_reg << "];\n";
                } else {
                    ss << "    ld.global.f32   " << dest << ", [" << offset_reg << "];\n";
                }
                break;
            }

            case IROp::Store: {
                std::string ptr = reg_map_[inst.args[0].name];
                std::string idx = reg_map_[inst.args[1].name];
                std::string val = reg_map_[inst.args[2].name];

                std::string offset_reg = "%rd0"; // use scratch %rd0
                ss << "    mul.wide.s32    " << offset_reg << ", " << idx << ", 4;\n";
                ss << "    add.s64         " << offset_reg << ", " << ptr << ", " << offset_reg << ";\n";

                if (inst.type.base == BaseType::Int) {
                    ss << "    st.global.s32   [" << offset_reg << "], " << val << ";\n";
                } else {
                    ss << "    st.global.f32   [" << offset_reg << "], " << val << ";\n";
                }
                break;
            }

            case IROp::Mov: {
                std::string dest = reg_map_[inst.dest->name];
                std::string src = reg_map_[inst.args[0].name];
                if (dest != src) {
                    if (inst.type.is_ptr) {
                        ss << "    mov.b64         " << dest << ", " << src << ";\n";
                    } else if (inst.type.base == BaseType::Float) {
                        ss << "    mov.f32         " << dest << ", " << src << ";\n";
                    } else {
                        ss << "    mov.b32         " << dest << ", " << src << ";\n";
                    }
                }
                break;
            }

            case IROp::Builtin: {
                std::string dest = reg_map_[inst.dest->name];
                if (inst.builtin_name == "tid.x") {
                    ss << "    mov.u32         " << dest << ", %tid.x;\n";
                } else if (inst.builtin_name == "bid.x") {
                    ss << "    mov.u32         " << dest << ", %ctaid.x;\n";
                } else if (inst.builtin_name == "bdim.x") {
                    ss << "    mov.u32         " << dest << ", %ntid.x;\n";
                }
                break;
            }

            case IROp::Br: {
                // Requirement 5: Only emit unconditional bra when target is not the next block
                if (inst.target_block != next_block) {
                    ss << "    bra             " << inst.target_block << ";\n";
                }
                break;
            }

            case IROp::CondBr: {
                std::string cond_name = inst.args[0].name;
                std::string pred_reg;

                auto it = cmp_pred_map_.find(cond_name);
                if (it != cmp_pred_map_.end()) {
                    // Reuses fused comparison predicate directly
                    pred_reg = it->second;
                } else {
                    // Condition is an integer register: test non-zero
                    pred_reg = "%p0";
                    std::string r_cond = reg_map_[cond_name];
                    ss << "    setp.ne.s32     " << pred_reg << ", " << r_cond << ", 0;\n";
                }

                // Requirement 5: Single conditional branch + fall-through optimization
                if (inst.then_block == next_block) {
                    // Fall through to then_block on true, jump to else_block on false
                    ss << "    @!" << pred_reg << " bra       " << inst.else_block << ";\n";
                } else {
                    // Jump to then_block on true
                    ss << "    @" << pred_reg << " bra        " << inst.then_block << ";\n";
                    // Only emit branch to else_block if it does not fall through
                    if (inst.else_block != next_block) {
                        ss << "    bra             " << inst.else_block << ";\n";
                    }
                }
                break;
            }

            case IROp::Ret: {
                ss << "    ret;\n";
                break;
            }
        }
    }
};

}  // namespace

std::string emit_ptx(const IRFunction& fn, const EmitterOptions& options) {
    PTXEmitter emitter(fn, options);
    return emitter.emit();
}

std::string emit_ptx() {
    return ".version 6.0\n.target sm_70\n.address_size 64\n";
}

}  // namespace miniptx
