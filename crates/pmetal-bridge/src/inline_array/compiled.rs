//! Compiled and fused op entry points on [`InlineArray`].
//!
//! `compiled_*` methods wrap whole transformer sub-layers into a single
//! `mlx::core::compile` graph (fixed shapes) — replacing tens of dispatches
//! with one. `fused_*` methods wrap a handful of elementwise ops that
//! correspond to Python's `@mx.compile`-decorated helpers.

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

impl InlineArray {
    // ── Compiled fixed-shape sub-layers ─────────────────────────────────────

    /// Fixed-shape compiled GDN layer (shapeless=false).
    /// Works with ALL primitives. Traces on first T=1 call, replays tape on subsequent.
    /// Eliminates graph traversal overhead for ~10ms savings per step.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_gdn_layer_fixed(
        normed: &Self,
        qkv_w: &Self,
        z_w: &Self,
        b_w: &Self,
        a_w: &Self,
        conv_w: &Self,
        q_nw: &Self,
        k_nw: &Self,
        a_log: &Self,
        dt_bias: &Self,
        norm_w: &Self,
        out_w: &Self,
        conv_state: &Self,
        ssm_state: &Self,
        nv: i32,
        nk: i32,
        dk: i32,
        dv: i32,
        cd: i32,
        ck: i32,
        kd: i32,
        norm_eps: f32,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut conv = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut ssm = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_gdn_layer_fixed(
                out.as_mut_ptr(),
                conv.as_mut_ptr(),
                ssm.as_mut_ptr(),
                &normed.raw,
                &qkv_w.raw,
                &z_w.raw,
                &b_w.raw,
                &a_w.raw,
                &conv_w.raw,
                &q_nw.raw,
                &k_nw.raw,
                &a_log.raw,
                &dt_bias.raw,
                &norm_w.raw,
                &out_w.raw,
                &conv_state.raw,
                &ssm_state.raw,
                nv,
                nk,
                dk,
                dv,
                cd,
                ck,
                kd,
                norm_eps,
            );
            (
                Self {
                    raw: out.assume_init(),
                },
                Self {
                    raw: conv.assume_init(),
                },
                Self {
                    raw: ssm.assume_init(),
                },
            )
        }
    }

    /// Fixed-shape compiled attention decode layer (shapeless=false).
    /// Traces per cache-capacity bucket on first T=1 call, then replays.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_attn_layer_fixed(
        normed: &Self,
        q_w: &Self,
        k_w: &Self,
        v_w: &Self,
        o_w: &Self,
        q_nw: &Self,
        k_nw: &Self,
        cache_keys_in: &Self,
        cache_vals_in: &Self,
        kv_offset: i32,
        rope_offset: i32,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        scale: f32,
        rope_dims: i32,
        rope_base: f32,
        rope_scale: f32,
        q_norm_eps: f32,
        k_norm_eps: f32,
        gated: bool,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut cache_keys = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut cache_vals = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_attn_layer_fixed(
                out.as_mut_ptr(),
                cache_keys.as_mut_ptr(),
                cache_vals.as_mut_ptr(),
                &normed.raw,
                &q_w.raw,
                &k_w.raw,
                &v_w.raw,
                &o_w.raw,
                &q_nw.raw,
                &k_nw.raw,
                &cache_keys_in.raw,
                &cache_vals_in.raw,
                kv_offset,
                rope_offset,
                n_heads,
                n_kv,
                head_dim,
                scale,
                rope_dims,
                rope_base,
                rope_scale,
                q_norm_eps,
                k_norm_eps,
                gated,
            );
            (
                Self {
                    raw: out.assume_init(),
                },
                Self {
                    raw: cache_keys.assume_init(),
                },
                Self {
                    raw: cache_vals.assume_init(),
                },
            )
        }
    }

    /// Fixed-shape compiled dense MoE decode block (shapeless=false).
    /// Replays the routed-expert + shared-expert graph for T=1 decode.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_moe_layer_fixed(
        x: &Self,
        router_w: &Self,
        moe_gate_w: &Self,
        moe_up_w: &Self,
        moe_down_w: &Self,
        shared_gate_w: &Self,
        shared_up_w: &Self,
        shared_down_w: &Self,
        shared_expert_gate_w: &Self,
        top_k: i32,
        norm_topk_prob: bool,
    ) -> Self {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_moe_layer_fixed(
                out.as_mut_ptr(),
                &x.raw,
                &router_w.raw,
                &moe_gate_w.raw,
                &moe_up_w.raw,
                &moe_down_w.raw,
                &shared_gate_w.raw,
                &shared_up_w.raw,
                &shared_down_w.raw,
                &shared_expert_gate_w.raw,
                top_k,
                norm_topk_prob,
            );
            Self {
                raw: out.assume_init(),
            }
        }
    }

    /// Fixed-shape compiled Gemma 4 attention block. Fuses
    /// input_layernorm → q/k/v projections (with optional
    /// `attention_k_eq_v` collapse) → q_norm / k_norm / v_norm-no-scale
    /// → transpose → RoPE (custom freqs OR full base) → KV cache write
    /// → SDPA → o_proj → post_attention_layernorm into a single
    /// mlx::compile graph. Weights are expected in `[in, out]` form
    /// (pre-transposed by the caller at load time).
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_gemma4_attn_block(
        x: &Self,
        in_norm_w: &Self,
        q_w: &Self,
        k_w: &Self,
        v_w: Option<&Self>,
        o_w: &Self,
        q_norm_w: &Self,
        k_norm_w: &Self,
        post_norm_w: &Self,
        rope_freqs: Option<&Self>,
        cache_keys_in: &Self,
        cache_vals_in: &Self,
        kv_offset: i32,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        in_norm_eps: f32,
        qk_norm_eps: f32,
        post_norm_eps: f32,
        sliding_window: i32,
        rope_base: f32,
        rope_dims: i32,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut ck_out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut cv_out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let use_k_eq_v = v_w.is_none();
        let v_ptr: *const RawBuf = match v_w {
            Some(v) => &v.raw,
            None => &q_w.raw,
        };
        let freqs_ptr: *const RawBuf = match rope_freqs {
            Some(f) => &f.raw,
            None => std::ptr::null(),
        };
        unsafe {
            mlx_inline_compiled_gemma4_attn_block(
                out.as_mut_ptr(),
                ck_out.as_mut_ptr(),
                cv_out.as_mut_ptr(),
                &x.raw,
                &in_norm_w.raw,
                &q_w.raw,
                &k_w.raw,
                v_ptr,
                &o_w.raw,
                &q_norm_w.raw,
                &k_norm_w.raw,
                &post_norm_w.raw,
                freqs_ptr,
                &cache_keys_in.raw,
                &cache_vals_in.raw,
                kv_offset,
                n_heads,
                n_kv,
                head_dim,
                in_norm_eps,
                qk_norm_eps,
                post_norm_eps,
                sliding_window,
                use_k_eq_v,
                rope_base,
                rope_dims,
            );
            (
                Self {
                    raw: out.assume_init(),
                },
                Self {
                    raw: ck_out.assume_init(),
                },
                Self {
                    raw: cv_out.assume_init(),
                },
            )
        }
    }

    /// Decode-only q-only attention path for Gemma 4 KV-shared layers.
    ///
    /// Reuses an already-populated source cache and applies the same
    /// input-layernorm → q_proj → q_norm → RoPE → SDPA → o_proj →
    /// post_attention_layernorm sequence as the full Gemma 4 block, but
    /// without projecting / appending fresh K/V.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_gemma4_shared_attn_decode(
        x: &Self,
        in_norm_w: &Self,
        q_w: &Self,
        o_w: &Self,
        q_norm_w: &Self,
        post_norm_w: &Self,
        rope_freqs: Option<&Self>,
        cache_keys_in: &Self,
        cache_vals_in: &Self,
        valid_kv_len: i32,
        rope_offset: i32,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        in_norm_eps: f32,
        q_norm_eps: f32,
        post_norm_eps: f32,
        sliding_window: i32,
        rope_base: f32,
        rope_dims: i32,
    ) -> Self {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let freqs_ptr: *const RawBuf = match rope_freqs {
            Some(f) => &f.raw,
            None => std::ptr::null(),
        };
        unsafe {
            mlx_inline_compiled_gemma4_shared_attn_decode(
                out.as_mut_ptr(),
                &x.raw,
                &in_norm_w.raw,
                &q_w.raw,
                &o_w.raw,
                &q_norm_w.raw,
                &post_norm_w.raw,
                freqs_ptr,
                &cache_keys_in.raw,
                &cache_vals_in.raw,
                valid_kv_len,
                rope_offset,
                n_heads,
                n_kv,
                head_dim,
                in_norm_eps,
                q_norm_eps,
                post_norm_eps,
                sliding_window,
                rope_base,
                rope_dims,
            );
            Self {
                raw: out.assume_init(),
            }
        }
    }

    /// Fixed-shape compiled Gemma 4 MLP block.
    ///
    /// Fuses `pre_feedforward_layernorm` + gate/up projections + tanh-approx
    /// GELU + element-wise multiply + down_proj + `post_feedforward_layernorm`
    /// into a single mlx::compile graph.
    #[allow(clippy::too_many_arguments)]
    // TODO(gemma4): wire into Gemma4 forward path — staged alongside the
    // Gemma4 compiled-layer roadmap; currently superseded by the per-op path.
    #[allow(dead_code)]
    pub fn compiled_gemma4_mlp_block(
        x: &Self,
        pre_norm_w: &Self,
        gate_w: &Self,
        up_w: &Self,
        down_w: &Self,
        post_norm_w: &Self,
        pre_norm_eps: f32,
        post_norm_eps: f32,
    ) -> Self {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_gemma4_mlp_block(
                out.as_mut_ptr(),
                &x.raw,
                &pre_norm_w.raw,
                &gate_w.raw,
                &up_w.raw,
                &down_w.raw,
                &post_norm_w.raw,
                pre_norm_eps,
                post_norm_eps,
            );
            Self {
                raw: out.assume_init(),
            }
        }
    }

    /// Decode-time compiled per-layer-input gating / projection block used by
    /// Gemma 4 E2B/E4B.
    pub fn compiled_gemma4_per_layer_input_block(
        x: &Self,
        layer_input: &Self,
        gate_w: &Self,
        projection_w: &Self,
        post_norm_w: &Self,
        post_norm_eps: f32,
    ) -> Self {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_gemma4_per_layer_input_block(
                out.as_mut_ptr(),
                &x.raw,
                &layer_input.raw,
                &gate_w.raw,
                &projection_w.raw,
                &post_norm_w.raw,
                post_norm_eps,
            );
            Self {
                raw: out.assume_init(),
            }
        }
    }

    // ── Fused compiled ops (match Python's @mx.compile) ─────────────────

    pub fn fused_swiglu(gate: &Self, up: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_swiglu(dst.as_mut_ptr(), &gate.raw, &up.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Fused tanh-approx GEGLU matching mlx-lm's `nn.gelu_approx(gate) * up`.
    /// All scalar constants are cast to `gate.dtype()` inside the compiled
    /// lambda, so bf16 inputs stay bf16 — avoiding the silent f32 promotion
    /// that otherwise doubled MLP bandwidth on the Gemma 4 hot path.
    #[inline]
    pub fn fused_geglu_tanh(gate: &Self, up: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_geglu_tanh(dst.as_mut_ptr(), &gate.raw, &up.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Fused SiLU: `x * sigmoid(x)` → 1 compiled dispatch instead of 2.
    #[inline]
    pub fn fused_silu(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_silu(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Fused compute_g: `exp(-exp(A_log.f32()) * softplus(a + dt_bias))` → 1 compiled dispatch instead of 6.
    #[inline]
    pub fn fused_compute_g(a_log: &Self, a: &Self, dt_bias: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_compute_g(dst.as_mut_ptr(), &a_log.raw, &a.raw, &dt_bias.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Fused precise SwiGLU: `(silu(gate.f32()) * x.f32()).as(x.dtype)` → 1 compiled dispatch instead of 5.
    #[inline]
    pub fn fused_precise_swiglu(x: &Self, gate: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_precise_swiglu(dst.as_mut_ptr(), &x.raw, &gate.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }
}
