#include "miniptx/ptx_emitter.hpp"
namespace miniptx {
std::string emit_ptx() {
  return ".version 7.0\n.target sm_70\n.address_size 64\n// TODO: real output\n";
}
}  // namespace miniptx
