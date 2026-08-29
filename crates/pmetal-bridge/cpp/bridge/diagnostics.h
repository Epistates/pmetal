// Graph/node dumps, Metal capture, memory instrumentation, stream
// management, wired-memory limit. Matches inline_array/diagnostics.rs.

#ifndef MLX_INLINE_BRIDGE_DIAGNOSTICS_H
#define MLX_INLINE_BRIDGE_DIAGNOSTICS_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Wired memory limit ───────────────────────────────────────────────────
// Critical for GPU performance.
size_t mlx_inline_set_wired_limit(size_t limit);
size_t mlx_inline_get_max_recommended_size(void);

// ── Stream management ────────────────────────────────────────────────────
// Match Python's mx.new_stream + mx.stream context.
int mlx_inline_new_stream(void);  // Returns stream index
void mlx_inline_set_default_stream(int index);
void mlx_inline_reset_default_stream(void);  // Restore original default stream
void mlx_inline_synchronize(void);

// ── Memory management ────────────────────────────────────────────────────
// Matches Python's mx.metal.clear_cache(), mx.metal.set_cache_limit().
void mlx_inline_clear_cache(void);

// ── Metallib path override ───────────────────────────────────────────────
// Forward to mlx::core::metal::set_metallib_path (upstream API since MLX
// v0.32): the given .metallib is loaded in preference to the automatic
// search when the Metal device initializes. Lazy — safe to call any time
// before the first GPU operation. No-op check is the caller's job.
void mlx_inline_set_metallib_path(const char* path);

// ── Metal capture for profiling ──────────────────────────────────────────
int mlx_inline_metal_start_capture(const char* path);
void mlx_inline_metal_stop_capture(void);

// ── Graph inspection ─────────────────────────────────────────────────────
// Count pending graph nodes for an array (traverses the computation graph).
size_t mlx_inline_graph_node_count(const mlx_inline_array* a);
size_t mlx_inline_graph_desc_count(const mlx_inline_array* a);

// Dump the graph topology: print every node's primitive type, shape, and status.
// This is the key to understanding why Python's graph evaluates 2.8x faster.
void mlx_inline_graph_dump(const mlx_inline_array* a);

// ── Metal memory instrumentation ─────────────────────────────────────────
// Maps directly to MLX's allocator tracking (same values Python sees).
size_t mlx_inline_get_active_memory(void);   // Bytes currently in use by arrays
size_t mlx_inline_get_cache_memory(void);    // Bytes freed but held in buffer cache
size_t mlx_inline_get_peak_memory(void);     // High-water mark of active memory
void   mlx_inline_reset_peak_memory(void);   // Reset peak tracking

#ifdef __cplusplus
}
#endif

#endif
