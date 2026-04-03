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

// Dequantize: reconstruct float from packed int + scales + biases
void mlx_inline_dequantize(mlx_inline_array* dst, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    int group_size, int bits);

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

// Fixed-shape compiled attention decode layer (shapeless=false).
// Traces per cache-shape bucket on first T=1 call, then replays.
void mlx_inline_compiled_attn_layer_fixed(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_cache_keys,
    mlx_inline_array* dst_cache_vals,
    const mlx_inline_array* normed,
    const mlx_inline_array* q_w,
    const mlx_inline_array* k_w,
    const mlx_inline_array* v_w,
    const mlx_inline_array* o_w,
    const mlx_inline_array* q_nw,
    const mlx_inline_array* k_nw,
    const mlx_inline_array* cache_keys_in,
    const mlx_inline_array* cache_vals_in,
    int kv_offset,
    int rope_offset,
    int n_heads,
    int n_kv,
    int head_dim,
    float scale,
    int rope_dims,
    float rope_base,
    float rope_scale,
    float q_norm_eps,
    float k_norm_eps,
    bool gated);

// Fixed-shape compiled dense MoE decode block (shapeless=false).
// Replays the routed-expert + shared-expert graph for T=1 decode.
void mlx_inline_compiled_moe_layer_fixed(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* router_w,
    const mlx_inline_array* moe_gate_w,
    const mlx_inline_array* moe_up_w,
    const mlx_inline_array* moe_down_w,
    const mlx_inline_array* shared_gate_w,
    const mlx_inline_array* shared_up_w,
    const mlx_inline_array* shared_down_w,
    const mlx_inline_array* shared_expert_gate_w,
    int top_k,
    bool norm_topk_prob);

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
// Weight pointer layout (weight_ptrs, num_weights = 3 + num_layers * WEIGHTS_PER_LAYER):
//   [0]  embed_w          (global)
//   [1]  final_norm_w     (global)
//   [2]  lm_head_w        (global; NULL when tie_word_embeddings=true)
//   Per-layer block at base index 3 + layer_idx * WEIGHTS_PER_LAYER:
//     [+0]  input_ln_w
//     [+1]  post_ln_w
//     [+2]  dense mlp_gate_w / moe_router_w
//     [+3]  dense mlp_up_w   / moe_gate_w
//     [+4]  dense mlp_down_w / moe_up_w
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
//   MoE-only offsets (+16 .. +20), NULL for dense MLP layers:
//     [+16] moe_down_w
//     [+17] shared_gate_w
//     [+18] shared_up_w
//     [+19] shared_down_w
//     [+20] shared_expert_gate_w
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
// Config integer layout (config_ints, num_config_ints = 20 + num_layers):
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
//   [18] moe_top_k
//   [19] moe_norm_topk_prob
//   [20 + i] (i = 0..num_layers-1) layer_i_is_moe
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
#define QWEN35_WEIGHTS_PER_LAYER 21
void mlx_inline_qwen35_decode_step(
    mlx_inline_array*              dst_logits,
    const mlx_inline_array*        token_ids,          // [B, T] int32
    const mlx_inline_array* const* weight_ptrs,        // flat weight array
    int                            num_weights,
    mlx_inline_array**             cache_ptrs,          // flat cache array (in/out)
    int                            num_cache,
    int*                           attn_kv_offsets,     // [n_attn] in/out
    int*                           rope_offset,         // [1] in/out
    const int*                     config_ints,         // [20 + num_layers]
    int                            num_config_ints,
    const float*                   config_floats,       // see layout above
    int                            num_config_floats
);

// ── TurboQuant fused Metal kernels ──────────────────────────────────────────
//
// These replace the expand_dims+subtract+square+argmin chain (which allocates
// a [N, D, C] intermediate tensor) with single-dispatch Metal kernels that
// keep everything in thread registers.
//
// Pipeline split:
//   The Rust caller handles norm computation (keys.norm_l2) and rotation
//   (keys.matmul(rot_t)) — both are standard MLX ops with no intermediates.
//   These kernels handle ONLY the innermost bottleneck:
//
// turboquant_encode:
//   input [N, D] f32  — already normalised + rotated
//   codebook [C] f32  — sorted Lloyd-Max centroids (C = 2^bits, max 16)
//   → indices [N, D] uint32 — nearest centroid index per coordinate
//   (out_norms is reserved for ABI symmetry but not written by the kernel)
//
// turboquant_decode:
//   indices [N, D] uint32   — centroid indices
//   norms [N] f32           — reserved for ABI symmetry (not read by kernel)
//   codebook [C] f32        — same centroids used at encode time
//   → output [N, D] f32     — centroid values in the rotated domain
//   (caller multiply-by-norms and matmul-with-rotation happen outside)
//
// Both return 0 on success, 1 if Metal kernel is unavailable.

// Fused encode: nearest-centroid search over a small codebook.
// input: [N, D] f32 (normalised+rotated).  codebook: [C] f32 (C <= 16).
// out_indices: [N, D] uint32.  out_norms: reserved (may be NULL).
int mlx_inline_turboquant_encode(
    mlx_inline_array*       out_indices,   // [N, D] uint32  (written)
    mlx_inline_array*       out_norms,     // reserved — may be NULL
    const mlx_inline_array* input,         // [N, D] f32
    const mlx_inline_array* codebook,      // [C]    f32
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows);

// Fused decode: codebook lookup → [N, D] f32 centroid values.
// indices: [N, D] uint32.  norms: reserved (may be NULL).  codebook: [C] f32.
// out: [N, D] f32 centroid values in the rotated domain (un-scaled by norm).
int mlx_inline_turboquant_decode(
    mlx_inline_array*       out,           // [N, D] f32      (written)
    const mlx_inline_array* indices,       // [N, D] uint32
    const mlx_inline_array* norms,         // reserved — may be NULL
    const mlx_inline_array* codebook,      // [C]    f32
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows);

// Fused TurboQuant key scoring.
// query_rot/query_proj: [N, D] f32
// indices:              [N, D, S_cap] uint8 transposed for score-friendly seq access
// qjl_signs:            [N, ceil(D/32), S_cap] packed uint32 sign words
// norms/residual_norms: [N, S_cap] f32
// codebook:             [C] f32
// out_scores:           [N, S] f32
int mlx_inline_turboquant_score(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* norms,
    const mlx_inline_array* residual_norms,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                qjl_words,
    uint32_t                n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized q8 key scoring for D=256 on the seq-major transposed cache layout.
int mlx_inline_turboquant_score_q8_d256(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* norms,
    const mlx_inline_array* residual_norms,
    const mlx_inline_array* codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Fused mixed TurboQuant key scoring.
// regular/outlier query tensors: [N, D_reg]/[N, D_out] f32
// regular/outlier indices: [KvRows, S_cap, D_*]
// regular/outlier qjl_signs: [KvRows, S_cap, ceil(D_*/32)] packed uint32 words
// regular/outlier norms/residual_norms: [KvRows, S_cap] f32
// out_scores: [N, S] f32
int mlx_inline_turboquant_mixed_score(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* regular_query_rot,
    const mlx_inline_array* regular_query_proj,
    const mlx_inline_array* regular_indices,
    const mlx_inline_array* regular_qjl_signs,
    const mlx_inline_array* regular_norms,
    const mlx_inline_array* regular_residual_norms,
    const mlx_inline_array* regular_codebook,
    const mlx_inline_array* outlier_query_rot,
    const mlx_inline_array* outlier_query_proj,
    const mlx_inline_array* outlier_indices,
    const mlx_inline_array* outlier_qjl_signs,
    const mlx_inline_array* outlier_norms,
    const mlx_inline_array* outlier_residual_norms,
    const mlx_inline_array* outlier_codebook,
    uint32_t                regular_dim,
    uint32_t                regular_qjl_words,
    uint32_t                regular_n_centroids,
    uint32_t                outlier_dim,
    uint32_t                outlier_qjl_words,
    uint32_t                outlier_n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Pack sign(projected >= 0) along the last dimension into uint32 words.
// projected: [N, D] f32
// out:       [N, ceil(D/32)] uint32
int mlx_inline_turboquant_pack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* projected,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows);

// Pack q8 TurboQuant key bytes from centroid indices plus QJL signs.
// indices:   [N, D, S_cap] uint8   (7-bit centroid indices)
// qjl_signs: [N, ceil(D/32), S_cap] uint32 packed sign words
// out:       [N, D, S_cap] uint8   (low 7 bits index, high bit sign)
int mlx_inline_turboquant_pack_q8_keybytes(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity);

// Pack q8 TurboQuant key bytes into a seq-major shadow layout.
// indices:   [N, D, S_cap] uint8
// qjl_signs: [N, ceil(D/32), S_cap] uint32
// out:       [N, S_cap, D] uint8
int mlx_inline_turboquant_pack_q8_keybytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity);

// Pack q8 TurboQuant key bytes and value indices into a seq-major shadow.
// indices:       [N, D, S_cap] uint8
// qjl_signs:     [N, ceil(D/32), S_cap] uint32
// value_indices: [N, S_cap, D] uint8
// out:           [N, S_cap, D] uint16
//                low byte = key byte, high byte = value centroid index
int mlx_inline_turboquant_pack_q8_kvbytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* value_indices,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity);

// Unpack uint32 sign words back to {-1,+1} f32 signs.
// packed: [N, ceil(D/32)] uint32
// out:    [N, D] f32
int mlx_inline_turboquant_unpack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* packed,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows);

// Fused TurboQuant value aggregation in the rotated domain.
// weights:  [N, S] f32
// indices:  [N, D, S_cap] uint8
// norms:    [N, S_cap] f32
// codebook: [C] f32
// out:      [N, D] f32 aggregated rotated vectors
int mlx_inline_turboquant_weighted_decode(
    mlx_inline_array*       out,
    const mlx_inline_array* weights,
    const mlx_inline_array* indices,
    const mlx_inline_array* norms,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads);

// Specialized long-context q8 decode primitive for D=256/V=256.
// Computes TurboQuant attention directly from compressed K/V in two passes:
// pass 1 emits per-block partial accumulators + log-sum-exp stats,
// pass 2 merges those partials into the final rotated output.
int mlx_inline_turboquant_attention_q8_d256_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_qjl_signs,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=256/V=256 over packed
// key bytes stored as [N, S_cap, D] (low 7 bits centroid index, high bit QJL sign).
int mlx_inline_turboquant_attention_q8_d256_packed_keys_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=256/V=256 over a
// seq-major packed `{key,value}` shadow:
// - `kv_bytes`: [N, S_cap, D] uint16
//   low byte = key byte, high byte = value centroid index
int mlx_inline_turboquant_attention_q8_d256_packed_kv_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* kv_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=256/V=256 over a
// seq-major packed key shadow plus dense rotated values:
// - `kv_bytes`: [N, S_cap, D] uint16
//   low byte = key byte, high byte unused by this path
// - `value_dense`: [N, S_cap, D] bf16/f32 rotated dense values
int mlx_inline_turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* kv_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=128/V=128.
// Computes TurboQuant attention directly from compressed K/V in two passes:
// pass 1 emits per-block partial accumulators + log-sum-exp stats,
// pass 2 merges those partials into the final rotated output.
int mlx_inline_turboquant_attention_q8_d128_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_qjl_signs,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=128/V=128 over packed
// key bytes stored as [N, D, S_cap] (low 7 bits centroid index, high bit QJL sign).
int mlx_inline_turboquant_attention_q8_d128_packed_keys_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Gather/scatter helpers for mixed TurboQuant component layouts.
// input: [N, D] f32, positions: [P] int32, out: [N, P] f32
int mlx_inline_turboquant_gather_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* input,
    const mlx_inline_array* positions,
    uint32_t                full_dim,
    uint32_t                out_dim,
    uint32_t                n_rows);

// regular/outlier: [N, R]/[N, O] f32, positions: [R]/[O] int32, out: [N, D] f32
int mlx_inline_turboquant_scatter_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* regular,
    const mlx_inline_array* outlier,
    const mlx_inline_array* regular_positions,
    const mlx_inline_array* outlier_positions,
    uint32_t                full_dim,
    uint32_t                regular_dim,
    uint32_t                outlier_dim,
    uint32_t                n_rows);

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

// ── Training ops: random ──
void mlx_inline_random_normal(mlx_inline_array* dst, const int* shape, int ndim, int dtype);
void mlx_inline_random_uniform(mlx_inline_array* dst, const int* shape, int ndim, int dtype);
void mlx_inline_random_bernoulli(mlx_inline_array* dst, const mlx_inline_array* p, const int* shape, int ndim);
void mlx_inline_random_seed(uint64_t seed);
void mlx_inline_random_randint(mlx_inline_array* dst, int low, int high, const int* shape, int ndim, int dtype);

// ── Training ops: math ──
void mlx_inline_mean_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_mean_all(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_var_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_pow(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_reciprocal(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_sin(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_cos(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_clip(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* lo, const mlx_inline_array* hi);
void mlx_inline_log_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_cross_entropy(mlx_inline_array* dst, const mlx_inline_array* logits, const mlx_inline_array* targets, int axis);
void mlx_inline_square(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Training ops: creation ──
void mlx_inline_full(mlx_inline_array* dst, const int* shape, int ndim, float val, int dtype);
void mlx_inline_eye(mlx_inline_array* dst, int n, int dtype);
void mlx_inline_tri(mlx_inline_array* dst, int n, int m, int k, int dtype);

// ── Training ops: shape ──
void mlx_inline_broadcast_to(mlx_inline_array* dst, const mlx_inline_array* a, const int* shape, int ndim);
void mlx_inline_flatten(mlx_inline_array* dst, const mlx_inline_array* a, int start_axis, int end_axis);

// ── Training ops: sort/reduction ──
void mlx_inline_argsort(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_sum_all(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_max_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_min_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_minimum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);

// ── Training ops: activation ──
void mlx_inline_relu(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_gelu(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Training ops: comparison ──
void mlx_inline_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_not_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_greater(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_less(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_greater_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_less_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);

// ── Training ops: serialization ──
void mlx_inline_save_safetensors(const char* path, const char** keys, const mlx_inline_array* arrays, int count);

// ── Training ops: quantize ──
void mlx_inline_quantize(mlx_inline_array* dst_w, mlx_inline_array* dst_scales, mlx_inline_array* dst_biases,
    const mlx_inline_array* a, int group_size, int bits);

// ── Training ops: multi-axis sum/mean ──
void mlx_inline_sum_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int num_axes, bool keepdims);
void mlx_inline_mean_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int num_axes, bool keepdims);

// ── Training ops: misc ──
size_t mlx_inline_size(const mlx_inline_array* a);
size_t mlx_inline_nbytes(const mlx_inline_array* a);
int mlx_inline_data_ptr(const mlx_inline_array* a, const void** out_ptr);
void mlx_inline_stop_gradient(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Triangular inverse (CPU stream for WY factorization) ──
// Compute inverse of a triangular square matrix (batched; last 2 dims).
// upper=false -> lower triangular (default).
// Runs on CPU stream matching mlx-lm's tri_inv(StreamOrDevice::cpu()) usage.
void mlx_inline_tri_inv(mlx_inline_array* dst, const mlx_inline_array* a, bool upper, bool use_cpu);

// Singular Value Decomposition (economy/thin SVD).
// Writes U -> dst_u, S -> dst_s, Vt -> dst_vt.
// Always runs on the CPU stream (GPU SVD not yet in MLX).
void mlx_inline_svd(
    mlx_inline_array* dst_u,
    mlx_inline_array* dst_s,
    mlx_inline_array* dst_vt,
    const mlx_inline_array* a);

// ── Autograd: value_and_grad ──
// Callback type for Rust forward function
typedef void (*mlx_rust_forward_fn)(
    const mlx_inline_array* const* all_arrays,
    int n_total,
    mlx_inline_array* loss_out,
    void* ctx
);

// ── FFT ops ──────────────────────────────────────────────────────────────────
// rfft: real-valued FFT along the given axis. n_fft=-1 means use full axis size.
void mlx_inline_rfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis);
// irfft: inverse rfft. n_fft=-1 means infer from input size (n = 2*(freq-1)).
void mlx_inline_irfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis);

// ── leaky_relu ────────────────────────────────────────────────────────────────
void mlx_inline_leaky_relu(mlx_inline_array* dst, const mlx_inline_array* a, float neg_slope);

// ── squeeze all axes (remove all size-1 dims) ─────────────────────────────────
void mlx_inline_squeeze_all(mlx_inline_array* dst, const mlx_inline_array* a);

// ── pad ───────────────────────────────────────────────────────────────────────
// pad_widths: flat array of [before_0, after_0, before_1, after_1, ...] length 2*ndim
void mlx_inline_pad(mlx_inline_array* dst, const mlx_inline_array* a,
                    const int* pad_widths, int ndim, float fill_value);

// Compute loss + gradients via callback-based autograd.
// forward_fn is called ONCE with traced arrays to build the computation graph.
// Gradients are computed w.r.t. the first n_params arrays.
void mlx_inline_value_and_grad(
    mlx_rust_forward_fn forward_fn,
    void* ctx,
    const mlx_inline_array* const* all_arrays,
    int n_params,
    int n_total,
    mlx_inline_array* loss_out,
    mlx_inline_array** grads_out
);

// ── Missing ops for pmetal-models migration ───────────────────────────────────
void mlx_inline_rsqrt(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_zeros_like(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_ones_like(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_tile(mlx_inline_array* dst, const mlx_inline_array* a, const int* reps, int ndim);
void mlx_inline_linspace(mlx_inline_array* dst, float start, float stop, int n, int dtype);
void mlx_inline_split_sections(mlx_inline_array* dst_arr, const mlx_inline_array* a, int sections, int axis, int* out_count);
void mlx_inline_scatter_add(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* indices, const mlx_inline_array* updates, int axis);
void mlx_inline_topk(mlx_inline_array* dst, const mlx_inline_array* a, int k, int axis);
void mlx_inline_put_along_axis(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* indices, const mlx_inline_array* values, int axis);
void mlx_inline_layer_norm(mlx_inline_array* dst, const mlx_inline_array* x, const mlx_inline_array* weight, const mlx_inline_array* bias, float eps);
void mlx_inline_addmm(mlx_inline_array* dst, const mlx_inline_array* c, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_conv2d(mlx_inline_array* dst, const mlx_inline_array* input, const mlx_inline_array* weight, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups);

#ifdef __cplusplus
}
#endif

#endif
