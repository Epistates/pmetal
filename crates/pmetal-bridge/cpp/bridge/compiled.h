// Fused compiled ops (`@mx.compile`): per-primitive element-wise fusions
// (fused_swiglu/geglu/silu/...) and layer-level fixed-shape compiled
// graphs (GDN, attention, MoE, Gemma 4 halves). Also the generic
// `mlx_inline_compile_*` wrapper surface.
//
// Matches inline_array/compiled.rs.

#ifndef MLX_INLINE_BRIDGE_COMPILED_H
#define MLX_INLINE_BRIDGE_COMPILED_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Per-primitive fused activations ──────────────────────────────────────
// fused_swiglu: silu(gate) * up → 1 dispatch instead of 3 (sigmoid+mul+mul)
void mlx_inline_fused_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* gate, const mlx_inline_array* up);

// fused_geglu_tanh: tanh-approx GELU gating used by Gemma 2/3/4 MLP.
//   0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3))) * u
// Scalars are astype'd to gate.dtype() inside the compiled lambda, so
// bf16 inputs stay bf16 (fixes an f32 promotion that doubled MLP
// bandwidth on the Gemma 4 hot path). Shapeless compile.
void mlx_inline_fused_geglu_tanh(mlx_inline_array* dst,
    const mlx_inline_array* gate, const mlx_inline_array* up);

// fused_silu: x * sigmoid(x) → 1 dispatch instead of 2 (sigmoid+mul)
void mlx_inline_fused_silu(mlx_inline_array* dst, const mlx_inline_array* x);

// fused_compute_g: exp(-exp(A_log.f32()) * softplus(a + dt_bias)) → 1 dispatch instead of 6
void mlx_inline_fused_compute_g(mlx_inline_array* dst,
    const mlx_inline_array* a_log, const mlx_inline_array* a, const mlx_inline_array* dt_bias);

// fused_precise_swiglu: (silu(gate.f32()) * x.f32()).as(x.dtype) → 1 dispatch instead of 5
void mlx_inline_fused_precise_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* gate);

// ── Layer-level fixed-shape compiled graphs (shapeless=false) ────────────
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

// Fixed-shape compiled Gemma 4 decoder layer halves. Each side is a
// `make_compiled_fixed` lambda keyed by the layer's per-shape signature
// (batch, seq_len, cache_len, n_heads, n_kv, head_dim, k_eq_v, sliding).
//
// The attention block fuses input_layernorm + q/k/v projections (with
// the optional `attention_k_eq_v` collapse) + q_norm/k_norm/v_norm-no-scale
// + transpose + RoPE (custom freqs OR full base) + KV cache write +
// SDPA + o_proj + post_attention_layernorm into a single Compiled graph.
//
// The MLP block fuses pre_feedforward_layernorm + gate/up projections +
// tanh-approx GELU (matching mlx-lm `nn.gelu_approx`) + multiply +
// down_proj + post_feedforward_layernorm.
//
// Residual adds and the per-layer scalar multiply are intentionally
// kept on the Rust side — they're 3 trivial element-wise ops and the
// FFI surface stays narrow that way.
//
// Small-model-specific helpers:
//   * `mlx_inline_compiled_gemma4_shared_attn_decode`
//       Decode-only q-only attention for KV-shared layers.
//   * `mlx_inline_compiled_gemma4_per_layer_input_block`
//       Decode-time per-layer-input gating/projection block.
void mlx_inline_compiled_gemma4_attn_block(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_cache_keys,
    mlx_inline_array* dst_cache_vals,
    const mlx_inline_array* x,
    const mlx_inline_array* in_norm_w,
    const mlx_inline_array* q_w,
    const mlx_inline_array* k_w,
    const mlx_inline_array* v_w,
    const mlx_inline_array* o_w,
    const mlx_inline_array* q_norm_w,
    const mlx_inline_array* k_norm_w,
    const mlx_inline_array* post_norm_w,
    const mlx_inline_array* rope_freqs,
    const mlx_inline_array* cache_keys_in,
    const mlx_inline_array* cache_vals_in,
    int kv_offset,
    int n_heads,
    int n_kv,
    int head_dim,
    float in_norm_eps,
    float qk_norm_eps,
    float post_norm_eps,
    int sliding_window,
    bool use_k_eq_v,
    float rope_base,
    int rope_dims);

void mlx_inline_compiled_gemma4_shared_attn_decode(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* in_norm_w,
    const mlx_inline_array* q_w,
    const mlx_inline_array* o_w,
    const mlx_inline_array* q_norm_w,
    const mlx_inline_array* post_norm_w,
    const mlx_inline_array* rope_freqs,
    const mlx_inline_array* cache_keys_in,
    const mlx_inline_array* cache_vals_in,
    int valid_kv_len,
    int rope_offset,
    int n_heads,
    int n_kv,
    int head_dim,
    float in_norm_eps,
    float q_norm_eps,
    float post_norm_eps,
    int sliding_window,
    float rope_base,
    int rope_dims);

void mlx_inline_compiled_gemma4_mlp_block(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* pre_norm_w,
    const mlx_inline_array* gate_w,
    const mlx_inline_array* up_w,
    const mlx_inline_array* down_w,
    const mlx_inline_array* post_norm_w,
    float pre_norm_eps,
    float post_norm_eps);

void mlx_inline_compiled_gemma4_per_layer_input_block(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* layer_input,
    const mlx_inline_array* gate_w,
    const mlx_inline_array* projection_w,
    const mlx_inline_array* post_norm_w,
    float post_norm_eps);

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

// ── Compile-mode toggles ─────────────────────────────────────────────────
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

// ── Generic mlx::core::compile() wrapper ─────────────────────────────────
//
// Callback invoked by MLX when the compiled closure is called. Converts
// input mlx::core::array values into `mlx_inline_array` buffers, then
// calls `forward_fn` with them. The Rust side writes output arrays into
// `outputs` (caller-allocated with space for `n_outputs_max` slots) and
// sets `*n_outputs_written` to the actual count.
typedef void (*mlx_rust_compile_forward_fn)(
    const mlx_inline_array* const* inputs,
    int n_inputs,
    mlx_inline_array* outputs,
    int* n_outputs_written,
    void* ctx
);

// Create a compiled closure. `n_outputs_max` bounds the output vector the
// Rust trampoline can produce. `shapeless=true` allows re-use across
// different input shapes; `false` retraces on shape changes.
//
// Returns NULL on failure (check pmetal_bridge_last_error_*).
void* mlx_inline_compile_make(
    mlx_rust_compile_forward_fn forward_fn,
    void* ctx,
    int n_outputs_max,
    bool shapeless
);

// Invoke a compiled closure. `inputs` is a flat array of N input handles;
// on success `outputs[0..*n_outputs_written-1]` hold the result arrays
// (caller must have allocated at least `n_outputs_max` slots).
//
// Returns 0 on success, -1 on any failure (check pmetal_bridge_last_error_*).
int mlx_inline_compile_call(
    void* compiled_handle,
    const mlx_inline_array* const* inputs,
    int n_inputs,
    mlx_inline_array* outputs,
    int n_outputs_max,
    int* n_outputs_written
);

// Destroy a compiled closure handle. Safe to call with NULL. Rust's
// `CompiledFn` Drop impl calls this before dropping its own closure Box.
void mlx_inline_compile_free(void* compiled_handle);

#ifdef __cplusplus
}
#endif

#endif
