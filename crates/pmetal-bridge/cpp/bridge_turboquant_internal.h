// Internal helpers shared by the split bridge_turboquant_*.cpp files.
// Not part of the public C interface.
#pragma once

#include "bridge_internal.h"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel-getter boilerplate
// ─────────────────────────────────────────────────────────────────────────────
// Each TurboQuant Metal kernel uses the pattern below to lazily register itself
// the first time the getter is called. We intentionally keep this as direct
// boilerplate rather than a macro for two reasons:
//
//   1. The C preprocessor cannot pass brace-enclosed initializer lists for the
//      input/output name vectors as macro arguments without awkward
//      parenthesisation tricks. (`__VA_ARGS__` works around this, but the
//      macro form is no longer obviously readable at the call site.)
//
//   2. Even with the brace-list issue solved, each getter is only ~12 LOC and
//      the most error-prone part is the input/output name-vector binding to
//      the kernel-source identifiers. A macro hides exactly that — wrong
//      ordering or a typo silently produces the wrong tensor binding instead
//      of a compile-time error you can spot in 10 seconds.
//
// If you add a new kernel, copy this template:
//
//   static const char* MY_KERNEL_SOURCE = R"(...)";
//
//   static mlx::core::fast::CustomKernelFunction& get_my_kernel() {
//       static auto kernel = mlx::core::fast::metal_kernel(
//           "my_kernel",
//           {"input_a", "input_b"},      // input tensor names
//           {"output"},                  // output tensor names
//           MY_KERNEL_SOURCE,
//           "",                          // header (none)
//           true,                        // ensure_uniqueness
//           false                        // atomic_outputs
//       );
//       return kernel;
//   }

// Allow PMETAL_TQ_Q8_2PASS_BLOCKS to override the per-family 2-pass block
// count at runtime. Parsed value is clamped to [32, 1024] and rounded down
// to the nearest multiple of 32. Returns `fallback` when the env var is
// unset, empty, or malformed.
static inline uint32_t turboquant_q8_2pass_blocks_override_or(uint32_t fallback) {
    const char* env = std::getenv("PMETAL_TQ_Q8_2PASS_BLOCKS");
    if (!env || !*env) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(env, &end, 10);
    if (end == env || *end != '\0') return fallback;
    if (parsed < 32ul) parsed = 32ul;
    if (parsed > 1024ul) parsed = 1024ul;
    parsed = (parsed / 32ul) * 32ul;
    return parsed ? static_cast<uint32_t>(parsed) : fallback;
}
