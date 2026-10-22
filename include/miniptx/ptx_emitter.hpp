#pragma once

#include <string>
#include "miniptx/ir.hpp"

namespace miniptx {

// PTX ISA version and target architecture defaults.
// sm_70 (Volta / Tesla V100) requires PTX ISA 6.0 or higher.
// Defaulting to 6.0 ensures compatibility with CUDA 9.0+ toolchains.
inline constexpr const char* DEFAULT_PTX_VERSION = "6.0";
inline constexpr const char* DEFAULT_PTX_TARGET = "sm_70";

struct EmitterOptions {
    std::string ptx_version = DEFAULT_PTX_VERSION;
    std::string target = DEFAULT_PTX_TARGET;
    bool emit_line_info = false;
    std::string source_filename = "";
};

// Emit valid NVIDIA PTX string from an IRFunction.
std::string emit_ptx(const IRFunction& fn, const EmitterOptions& options = {});

// Generate human-readable source-to-PTX mapping
std::string emit_ptx_map(const IRFunction& fn, const std::string& source, const EmitterOptions& options = {});

// Legacy compatibility overload
std::string emit_ptx();

}  // namespace miniptx
