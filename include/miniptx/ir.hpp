#pragma once
// TODO(week 2): three-address IR in SSA form, basic blocks, and passes:
//   constant folding, dead-code elimination, common subexpression elimination.
// Add an IR printer so you can show before/after each pass.
namespace miniptx {
void lower_and_optimize();
}
