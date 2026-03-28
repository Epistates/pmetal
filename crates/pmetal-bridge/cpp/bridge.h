// Inline array bridge — zero heap allocation per op.
// mlx::core::array stored directly in a stack buffer managed by Rust.

#ifndef MLX_INLINE_BRIDGE_H
#define MLX_INLINE_BRIDGE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Size/alignment of mlx::core::array — queried at build time via mlx_inline_array_size()
// Conservative upper bound; static_assert in .cpp verifies at compile time.
#define MLX_ARRAY_SIZE 128
#define MLX_ARRAY_ALIGN 8

// Stack-allocated array — NO heap allocation per op.
// Rust creates this on the stack, C++ placement-news into buf.
typedef struct {
    _Alignas(MLX_ARRAY_ALIGN) unsigned char buf[MLX_ARRAY_SIZE];
} mlx_inline_array;

// Lifecycle
void mlx_inline_init_empty(mlx_inline_array* dst);
void mlx_inline_init_copy(mlx_inline_array* dst, const mlx_inline_array* src);
void mlx_inline_init_move(mlx_inline_array* dst, mlx_inline_array* src);
void mlx_inline_destroy(mlx_inline_array* a);

// Interop with legacy mlx_array handles
void mlx_inline_from_handle(mlx_inline_array* dst, void* handle_ctx);
void* mlx_inline_to_handle(const mlx_inline_array* src);

// Binary ops — result written to dst via placement new
void mlx_inline_matmul(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_add(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_multiply(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_subtract(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_divide(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);

// Unary ops
void mlx_inline_negative(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_exp(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_sigmoid(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_silu(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_sqrt(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_transpose(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_reshape(mlx_inline_array* dst, const mlx_inline_array* a, const int* shape, int ndim);
void mlx_inline_sum_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_astype(mlx_inline_array* dst, const mlx_inline_array* a, int dtype);

// Gather MM
void mlx_inline_gather_mm(mlx_inline_array* dst,
    const mlx_inline_array* a, const mlx_inline_array* b,
    const mlx_inline_array* lhs, const mlx_inline_array* rhs, bool sorted);

// Fast ops
void mlx_inline_rms_norm(mlx_inline_array* dst, const mlx_inline_array* x,
    const mlx_inline_array* weight, float eps);
void mlx_inline_rope(mlx_inline_array* dst, const mlx_inline_array* x,
    int dims, bool traditional, float base, float scale, int offset);
void mlx_inline_sdpa(mlx_inline_array* dst,
    const mlx_inline_array* q, const mlx_inline_array* k,
    const mlx_inline_array* v, float scale, const char* mask_mode);

// Split / concatenate
void mlx_inline_split(const mlx_inline_array* input, const int* indices, int num_indices,
    int axis, mlx_inline_array* outputs);
void mlx_inline_concatenate(mlx_inline_array* dst, const mlx_inline_array* arrays,
    int num, int axis);
void mlx_inline_argpartition(mlx_inline_array* dst, const mlx_inline_array* a, int kth, int axis);
void mlx_inline_take_along_axis(mlx_inline_array* dst, const mlx_inline_array* a,
    const mlx_inline_array* indices, int axis);

// Eval
void mlx_inline_eval(mlx_inline_array* a);
void mlx_inline_async_eval(mlx_inline_array* a);

// Factory
void mlx_inline_from_f32(mlx_inline_array* dst, float val);
void mlx_inline_from_i32(mlx_inline_array* dst, int val);

// Query
int mlx_inline_ndim(const mlx_inline_array* a);
int mlx_inline_dim(const mlx_inline_array* a, int axis);
const int* mlx_inline_shape(const mlx_inline_array* a);
int mlx_inline_dtype(const mlx_inline_array* a);

// Item extraction
float mlx_inline_item_f32(mlx_inline_array* a);
uint32_t mlx_inline_item_u32(mlx_inline_array* a);

// Sign: returns -1, 0, or +1 per element
void mlx_inline_sign(mlx_inline_array* dst, const mlx_inline_array* a);

// Create array from float32 data slice
void mlx_inline_from_f32_slice(mlx_inline_array* dst, const float* data, const int* shape, int ndim);

// Copy evaluated f32 data out of an array into a caller-provided buffer.
// Array is cast to float32 and eval'd. n must equal the total element count.
// Returns 0 on success, -1 on size mismatch.
int mlx_inline_to_f32_slice(mlx_inline_array* a, float* out, size_t n);

// Stack arrays along a new axis
void mlx_inline_stack(mlx_inline_array* dst, const mlx_inline_array* arrays, int num, int axis);

// L2 norm along last axis (keepdims=true)
void mlx_inline_norm_l2(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);

// Conv1d
void mlx_inline_conv1d(mlx_inline_array* dst, const mlx_inline_array* input,
    const mlx_inline_array* weight, int stride, int padding, int dilation, int groups);

// Size query (for Rust build-time verification)
size_t mlx_inline_array_size(void);
size_t mlx_inline_array_align(void);

// ── Additional ops for complete model inference ──

// Concatenate exactly two arrays along axis (avoids heap vector)
void mlx_inline_concatenate_2(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b, int axis);

// Softplus: log(1 + exp(x))
void mlx_inline_softplus(mlx_inline_array* dst, const mlx_inline_array* a);

// Where: condition ? a : b
void mlx_inline_where(mlx_inline_array* dst, const mlx_inline_array* condition, const mlx_inline_array* a, const mlx_inline_array* b);

// Maximum element-wise
void mlx_inline_maximum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);

// Factory: zeros/ones with explicit dtype (dtype codes match mlx_inline_astype)
void mlx_inline_zeros(mlx_inline_array* dst, const int* shape, int ndim, int dtype);
void mlx_inline_ones(mlx_inline_array* dst, const int* shape, int ndim, int dtype);

// Slice: a[start:stop] with stride 1 along every axis
void mlx_inline_slice(mlx_inline_array* dst, const mlx_inline_array* a, const int* start, const int* stop, int ndim);
// Slice-set (update): returns copy of a with value written into [start:stop]
void mlx_inline_slice_set(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* value, const int* start, const int* stop, int ndim);

// Repeat along axis
void mlx_inline_repeat(mlx_inline_array* dst, const mlx_inline_array* a, int repeats, int axis);

// Squeeze / expand singleton dimensions
void mlx_inline_squeeze(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_expand_dims(mlx_inline_array* dst, const mlx_inline_array* a, int axis);

// Transpose with explicit axis permutation
void mlx_inline_transpose_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int ndim);

// Cumulative sum along axis
void mlx_inline_cumsum(mlx_inline_array* dst, const mlx_inline_array* a, int axis);

// Natural logarithm
void mlx_inline_log(mlx_inline_array* dst, const mlx_inline_array* a);

// Lower-triangular mask (k=0 includes main diagonal; negative k excludes more)
void mlx_inline_tril(mlx_inline_array* dst, const mlx_inline_array* a, int k);

// Index / embedding lookup: take(a, indices) — flat gather over all elements
void mlx_inline_index(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* indices);

// Softmax with precise=true (fp32 accumulation)
void mlx_inline_softmax_precise(mlx_inline_array* dst, const mlx_inline_array* a, int axis);

// Scaled-dot-product attention with optional array mask (pass NULL for no mask)
void mlx_inline_sdpa_with_mask(mlx_inline_array* dst,
    const mlx_inline_array* q, const mlx_inline_array* k,
    const mlx_inline_array* v, float scale,
    const mlx_inline_array* mask);

// Eval two arrays in one call
void mlx_inline_eval_2(mlx_inline_array* a, mlx_inline_array* b);

// Eval many arrays in one call (single GPU submission, no per-array sync)
void mlx_inline_eval_many(mlx_inline_array** arrays, int count);

// Async eval many arrays in one call
void mlx_inline_async_eval_many(mlx_inline_array** arrays, int count);

// Quantized matmul: x @ dequantize(w, scales, biases)
void mlx_inline_quantized_matmul(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    bool transpose, int group_size, int bits);

// Gather quantized matmul (gathers rows of w before dequantize + matmul)
void mlx_inline_gather_qmm(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    const mlx_inline_array* lhs_indices, const mlx_inline_array* rhs_indices,
    bool transpose, int group_size, int bits, bool sorted);

// GDN recurrence step — calls the Metal kernel for single-dispatch recurrence.
// q, k: [B, T, Hk, Dk], v: [B, T, Hv, Dv], g: [B, T, Hv], beta: [B, T, Hv]
// state_in: [B, Hv, Dv, Dk]
// Returns y → dst_y [B, T, Hv, Dv], new_state → dst_state [B, Hv, Dv, Dk]
// Returns 0 on success, 1 if Metal kernel not available (caller should use ops fallback).
int mlx_inline_gdn_update(
    mlx_inline_array* dst_y,
    mlx_inline_array* dst_state,
    const mlx_inline_array* q,
    const mlx_inline_array* k,
    const mlx_inline_array* v,
    const mlx_inline_array* a,
    const mlx_inline_array* b,
    const mlx_inline_array* a_log,
    const mlx_inline_array* dt_bias,
    const mlx_inline_array* state_in,
    bool training);

// ── Sampling ops ──
void mlx_inline_argmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_argmin(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_logsumexp(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_categorical(mlx_inline_array* dst, const mlx_inline_array* logits);
void mlx_inline_negative(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Element-wise math ──
void mlx_inline_abs(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Fused compiled ops (match Python's @mx.compile) ──
// Each fuses multiple element-wise ops into a single Compiled dispatch node.

// fused_swiglu: silu(gate) * up → 1 dispatch instead of 3 (sigmoid+mul+mul)
void mlx_inline_fused_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* gate, const mlx_inline_array* up);

// fused_silu: x * sigmoid(x) → 1 dispatch instead of 2 (sigmoid+mul)
void mlx_inline_fused_silu(mlx_inline_array* dst, const mlx_inline_array* x);

// Compile mode: shapeless=false (fixed shapes, works with ALL primitives).
// Same as make_compiled but with shapeless=false — ideal for T=1 decode
// where all shapes are known and fixed.
// First call traces; subsequent calls replay the tape with zero graph overhead.
void mlx_inline_compiled_gdn_layer_fixed(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_conv_state,
    mlx_inline_array* dst_ssm_state,
    const mlx_inline_array* normed,
    const mlx_inline_array* qkv_w, const mlx_inline_array* z_w,
    const mlx_inline_array* b_w,   const mlx_inline_array* a_w,
    const mlx_inline_array* conv_w,
    const mlx_inline_array* q_nw, const mlx_inline_array* k_nw,
    const mlx_inline_array* a_log, const mlx_inline_array* dt_bias,
    const mlx_inline_array* norm_w, const mlx_inline_array* out_w,
    const mlx_inline_array* conv_state_in, const mlx_inline_array* ssm_state_in,
    int nv, int nk, int dk, int dv, int cd, int ck, int kd, float norm_eps);

// fused_compute_g: exp(-exp(A_log.f32()) * softplus(a + dt_bias)) → 1 dispatch instead of 6
void mlx_inline_fused_compute_g(mlx_inline_array* dst,
    const mlx_inline_array* a_log, const mlx_inline_array* a, const mlx_inline_array* dt_bias);

// fused_precise_swiglu: (silu(gate.f32()) * x.f32()).as(x.dtype) → 1 dispatch instead of 5
void mlx_inline_fused_precise_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* gate);

// Take rows along axis (for embedding lookup: take(weight, indices, axis=0))
void mlx_inline_take_axis(mlx_inline_array* dst, const mlx_inline_array* a,
    const mlx_inline_array* indices, int axis);

// KV cache: concatenate new K/V with cached K/V along seq axis
// Equivalent to: concatenate([cached, new], axis=2) for [B, H, T, D] format
void mlx_inline_kv_cache_append(mlx_inline_array* dst,
    const mlx_inline_array* cached, const mlx_inline_array* new_kv, int axis);

// Async eval for InlineArray
void mlx_inline_async_eval_arr(const mlx_inline_array* a);

// GDN Metal kernel step with pre-computed g and beta.
// q, k: [B, T, Hk, Dk], v: [B, T, Hv, Dv], g: [B, T, Hv], beta: [B, T, Hv]
// state_in: [B, Hv, Dv, Dk], T: sequence length (usually 1 for decode)
// Returns y → dst_y, new_state → dst_state. Falls back to ops if Metal unavailable.
void mlx_inline_gdn_metal_step(
    mlx_inline_array* dst_y,
    mlx_inline_array* dst_state,
    const mlx_inline_array* q,
    const mlx_inline_array* k,
    const mlx_inline_array* v,
    const mlx_inline_array* g,
    const mlx_inline_array* beta,
    const mlx_inline_array* state_in,
    int T);

// Compile mode control
void mlx_inline_enable_compile(void);
void mlx_inline_disable_compile(void);

// Compile a flat array→array function with shapeless=false.
// inputs: array of InlineArrays to compile over. outputs: caller provides buffer.
// build_fn is called ONCE to trace the graph; subsequent calls replay the tape.
// The function takes N inputs and produces M outputs.
void mlx_inline_compile_fixed(
    const mlx_inline_array* inputs, int num_inputs,
    mlx_inline_array* outputs, int num_outputs,
    int compile_id);

// Wired memory limit — CRITICAL for GPU performance
size_t mlx_inline_set_wired_limit(size_t limit);
size_t mlx_inline_get_max_recommended_size(void);

// Stream management — match Python's mx.new_stream + mx.stream context
int mlx_inline_new_stream(void);  // Returns stream index
void mlx_inline_set_default_stream(int index);
void mlx_inline_synchronize(void);

// Memory management — matches Python's mx.metal.clear_cache(), mx.metal.set_cache_limit()
void mlx_inline_clear_cache(void);
size_t mlx_inline_set_cache_limit(size_t limit);

// Metal capture for profiling
int mlx_inline_metal_start_capture(const char* path);
void mlx_inline_metal_stop_capture(void);

// Count pending graph nodes for an array (traverses the computation graph)
size_t mlx_inline_graph_node_count(const mlx_inline_array* a);
size_t mlx_inline_graph_desc_count(const mlx_inline_array* a);

// Dump the graph topology: print every node's primitive type, shape, and status.
// This is the key to understanding why Python's graph evaluates 2.8x faster.
void mlx_inline_graph_dump(const mlx_inline_array* a);

// Compiled entire GDN layer — fuses element-wise ops, single compilation tape.
// Uses 4 separate projection weights matching Python's in_proj_qkv/z/b/a.
void mlx_inline_compiled_gdn_layer(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_conv_state,
    mlx_inline_array* dst_ssm_state,
    const mlx_inline_array* normed,
    const mlx_inline_array* qkv_w, const mlx_inline_array* z_w,
    const mlx_inline_array* b_w,   const mlx_inline_array* a_w,
    const mlx_inline_array* conv_w,
    const mlx_inline_array* q_nw, const mlx_inline_array* k_nw,
    const mlx_inline_array* a_log, const mlx_inline_array* dt_bias,
    const mlx_inline_array* norm_w, const mlx_inline_array* out_w,
    const mlx_inline_array* conv_state_in, const mlx_inline_array* ssm_state_in,
    int nv, int nk, int dk, int dv, int cd, int ck, int kd, float norm_eps);

// Arange: create [0, 1, 2, ..., n-1] — forces full Metal buffer allocation (no broadcast)
void mlx_inline_arange(mlx_inline_array* dst, int n, int dtype);

// ── Full Qwen3.5 forward pass — single C++ function, zero FFI overhead ──
//
// Implements the entire 24-layer decode step (or prefill for T>1) inside C++,
// eliminating ~1800 per-op FFI round trips per decode step.
//
// Weight pointer layout (weight_ptrs, num_weights = 3 + num_layers * 16):
//   [0]  embed_w          (global)
//   [1]  final_norm_w     (global)
//   [2]  lm_head_w        (global; NULL when tie_word_embeddings=true)
//   Per-layer block at base index 3 + layer_idx * WEIGHTS_PER_LAYER (= 16):
//     [+0]  input_ln_w
//     [+1]  post_ln_w
//     [+2]  mlp_gate_w    (pre-transposed [in, out])
//     [+3]  mlp_up_w
//     [+4]  mlp_down_w
//   GDN-only offsets (+5 .. +15), NULL for attention layers:
//     [+5]  gdn_qkv_w
//     [+6]  gdn_z_w
//     [+7]  gdn_b_w
//     [+8]  gdn_a_w
//     [+9]  gdn_conv_w
//     [+10] gdn_q_nw
//     [+11] gdn_k_nw
//     [+12] gdn_a_log
//     [+13] gdn_dt_bias
//     [+14] gdn_norm_w
//     [+15] gdn_out_w
//   Attention-only offsets (+5 .. +10), NULL for GDN layers:
//     [+5]  attn_q_w
//     [+6]  attn_k_w
//     [+7]  attn_v_w
//     [+8]  attn_o_w
//     [+9]  attn_q_norm_w
//     [+10] attn_k_norm_w
//     (offsets +11..+15 are NULL for attention layers)
//
// Cache pointer layout (cache_ptrs, num_cache = n_gdn*2 + n_attn*4):
//   For each GDN layer gi = 0..n_gdn-1:
//     [gi*2+0]  conv_state  (in/out; may be NULL on first call → zeros allocated inside)
//     [gi*2+1]  ssm_state   (in/out; may be NULL on first call)
//   For each attention layer ai = 0..n_attn-1:
//     [n_gdn*2 + ai*4+0]  kv_keys    (in/out; may be NULL on first call)
//     [n_gdn*2 + ai*4+1]  kv_values  (in/out; may be NULL on first call)
//
// Scalar cache (passed separately, updated in-place):
//   attn_kv_offsets[n_attn]  — per-attention-layer valid-token count (in/out)
//   rope_offset[1]            — global position counter (in/out)
//
// Config integer layout (config_ints, num_config_ints = 18):
//   [0]  num_layers
//   [1]  hidden_size
//   [2]  model_dtype          (11 = bfloat16)
//   [3]  n_gdn                (# GDN layers)
//   [4]  n_attn               (# attention layers)
//   [5]  gdn_nv
//   [6]  gdn_nk
//   [7]  gdn_dk
//   [8]  gdn_dv
//   [9]  gdn_cd
//   [10] gdn_ck
//   [11] gdn_kd
//   [12] attn_n_heads
//   [13] attn_n_kv
//   [14] attn_head_dim
//   [15] attn_rope_dims
//   [16] full_attn_interval   (every Nth layer is attention, e.g. 4)
//   [17] tie_word_embeddings  (1 = tied, 0 = separate lm_head)
//
// Config float layout (config_floats):
//   [0]                                     final_norm_eps
//   [1]                                     attn_scale
//   [2]                                     attn_rope_base
//   [3]                                     attn_rope_scale
//   [4 + i*2]   (i = 0..num_layers-1)      layer_i input_ln_eps
//   [4 + i*2+1]                             layer_i post_ln_eps
//   [4 + num_layers*2 + gi]  (gi=0..n_gdn-1)  gdn_norm_eps
//   [4 + num_layers*2 + n_gdn + ai*2]       attn_q_norm_eps
//   [4 + num_layers*2 + n_gdn + ai*2+1]     attn_k_norm_eps
//
// Returns logits [B, T, vocab] via placement-new into dst_logits.
#define QWEN35_WEIGHTS_PER_LAYER 16
void mlx_inline_qwen35_decode_step(
    mlx_inline_array*              dst_logits,
    const mlx_inline_array*        token_ids,          // [B, T] int32
    const mlx_inline_array* const* weight_ptrs,        // flat weight array
    int                            num_weights,
    mlx_inline_array**             cache_ptrs,          // flat cache array (in/out)
    int                            num_cache,
    int*                           attn_kv_offsets,     // [n_attn] in/out
    int*                           rope_offset,         // [1] in/out
    const int*                     config_ints,         // [18]
    int                            num_config_ints,
    const float*                   config_floats,       // see layout above
    int                            num_config_floats
);

// Load a single array from a safetensors file by key name.
// Returns 0 on success, 1 on error (key not found, file not found, etc.)
int mlx_inline_load_safetensors_key(mlx_inline_array* dst, const char* path, const char* key);

// Load ALL arrays from a safetensors file in one parse.
// key_buf[i] receives a strdup'd key string (caller must free via mlx_inline_free_key_strings).
// arr_buf[i] receives a placement-new'd array (caller must destroy via mlx_inline_destroy).
// Returns number of entries loaded, or -1 on error.
int mlx_inline_load_safetensors_all(
    const char* path,
    char** key_buf,
    mlx_inline_array* arr_buf,
    int max_entries);

// Free key strings returned by mlx_inline_load_safetensors_all.
void mlx_inline_free_key_strings(char** keys, int count);

// Create a 1-D int32 array from a Rust slice.
void mlx_inline_from_i32_slice(mlx_inline_array* dst, const int32_t* data, int len);

// Detach: sever the computation graph, freeing all input references.
// Critical for caches: without this, cache updates chain across steps,
// keeping ALL previous steps' Metal buffers alive (memory leak).
void mlx_inline_detach(mlx_inline_array* a);

// ── Metal memory instrumentation ──
// These map directly to MLX's allocator tracking (same values Python sees).
size_t mlx_inline_get_active_memory(void);   // Bytes currently in use by arrays
size_t mlx_inline_get_cache_memory(void);    // Bytes freed but held in buffer cache
size_t mlx_inline_get_peak_memory(void);     // High-water mark of active memory
void   mlx_inline_reset_peak_memory(void);   // Reset peak tracking

#ifdef __cplusplus
}
#endif

#endif
