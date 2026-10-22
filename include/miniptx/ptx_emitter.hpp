#pragma once
#include <string>
namespace miniptx {
// TODO(week 3): IR -> PTX text. Virtual registers (.reg), ld.global/st.global,
// predicates (setp / @p bra) for if, special registers %tid.x %ctaid.x %ntid.x.
// TODO(week 4): emit .loc directives for source-line debug info; module linking.
std::string emit_ptx();
}
