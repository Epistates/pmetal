//! Shared decoder-layer abstractions for dense transformer architectures.
//!
//! ## Why this module exists
//!
//! The April 2026 audit flagged a large pile of copy-paste: most dense
//! transformer architectures in `crates/pmetal-models/src/architectures/`
//! implement the same 6-line pre-norm decoder-layer forward pass
//! (`norm → attn → +residual → norm → mlp → +residual`) with minor
//! per-architecture colouring around the edges. Reading side-by-side
//! across Llama / Mistral / Qwen3 / Phi / Gemma confirmed that 4/5 of
//! those forward bodies are byte-for-byte identical — the variance lives
//! entirely *inside* each `*Attention` module (Qwen3 per-head Q/K norms,
//! Phi partial RoPE, Mistral sliding-window clamp), not in the outer
//! decoder-layer skeleton.
//!
//! This module extracts that shared skeleton exactly once as
//! [`std_pre_norm_forward`], and defines the module-shaped traits
//! ([`AttentionModule`], [`MlpModule`], [`NormModule`], [`DecoderLayer`])
//! that downstream architectures use to plug into it. The traits are
//! *thin* — each exposes a single-method protocol — so wrapping existing
//! arch code is close to mechanical.
//!
//! ## Usage
//!
//! A dense transformer architecture with the standard pre-norm skeleton
//! only needs to:
//!
//! 1. `impl AttentionModule for <ArchName>Attention { … }` — forward the
//!    existing `forward_with_cache` method through the trait.
//! 2. `impl MlpModule for <ArchName>MLP { … }` — same deal for the MLP.
//! 3. `impl DecoderLayer for <ArchName>DecoderLayer` — delegate
//!    `forward_with_cache` to [`std_pre_norm_forward`].
//!
//! Steps 1–2 are one-liners via `Module::forward` delegation. Step 3
//! replaces the per-arch ~60-LOC forward body with a single
//! function call.
//!
//! ### Gemma / Gemma2 deviations
//!
//! * `Gemma` uses a custom `GemmaRmsNorm` type (output = x·(1+w) rather
//!   than x·w). Rather than parameterise the helper on the norm type,
//!   downstream code implements [`NormModule`] for `GemmaRmsNorm` and the
//!   skeleton accepts it through the `N: NormModule` bound on the
//!   generic variants of the helper.
//! * `Gemma2` (and Gemma4) interleave additional post-attention and
//!   post-FFN norms for a 4-norm peri-norm pattern — that variant is
//!   too different to share the standard helper and should keep a
//!   hand-rolled forward (but still implement the [`DecoderLayer`]
//!   trait so generation code can dispatch uniformly).

use pmetal_bridge::compat::{Array, Exception, Module, nn};
use pmetal_mlx::kv_cache::KVCache;

/// A self-attention module that supports an optional KV cache.
///
/// Implemented by every dense-transformer attention block in
/// `crates/pmetal-models/src/architectures/`.
pub trait AttentionModule: std::fmt::Debug {
    /// Forward with an optional KV cache + (layer index, attention mask).
    ///
    /// Signature mirrors the concrete per-arch `forward_with_cache` that
    /// every dense transformer already exposes, so the trait is a
    /// one-line adapter for each arch.
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception>;
}

/// A feed-forward module — dense MLP, SwiGLU, MoE block, etc.
///
/// The decoder skeleton never looks inside; it only calls `forward`.
pub trait MlpModule: std::fmt::Debug {
    /// Forward pass.
    fn forward(&mut self, x: &Array) -> Result<Array, Exception>;
}

/// An RMS-/layer-norm module used by decoder layers.
///
/// Wraps the generic `Module<&Array>` trait in a single-method protocol
/// so the skeleton doesn't need a higher-ranked trait bound. Impls for
/// `nn::RmsNorm` and `nn::LayerNorm` live in this module so every user
/// gets them for free.
pub trait NormModule: std::fmt::Debug {
    /// Apply the norm to `x`.
    fn forward(&mut self, x: &Array) -> Result<Array, Exception>;
}

impl NormModule for nn::RmsNorm {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        <Self as Module<&Array>>::forward(self, x)
    }
}

impl NormModule for nn::LayerNorm {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        <Self as Module<&Array>>::forward(self, x)
    }
}

/// Dynamic-dispatch interface to any architecture's decoder layer.
///
/// Generation / training code that iterates over `Vec<dyn DecoderLayer>`
/// can run every arch through one call without match-on-variant.
pub trait DecoderLayer: std::fmt::Debug {
    /// Forward pass with optional KV cache. This is the main entry.
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception>;

    /// Forward pass without cache (inference-only convenience wrapper).
    ///
    /// Provided as a default so each arch only overrides
    /// `forward_with_cache`.
    fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None)
    }
}

/// The standard pre-norm + residual decoder-layer forward pass.
///
/// Shape:
///
/// ```text
/// let h = x + self_attn(input_norm(x));
/// let y = h + mlp(post_norm(h));
/// ```
///
/// 4/5 dense transformers in the pmetal tree match this exactly
/// (Llama, Mistral, Qwen2, Qwen3, Phi / Phi4, Cohere, Granite,
/// StarCoder2, DFlash draft). Gemma plugs in here with a custom
/// `NormModule` impl for `GemmaRmsNorm`. Gemma2 / Gemma4 keep their
/// own forward body because they add peri-norms that break the shape.
///
/// # Arguments
///
/// * `input_layernorm` — the pre-attention norm.
/// * `self_attn` — the attention module.
/// * `post_attention_layernorm` — the pre-MLP norm.
/// * `mlp` — the feed-forward module.
/// * `x` / `mask` / `cache` — forward-pass inputs, threaded through
///   unchanged from the decoder layer's own forward signature.
pub fn std_pre_norm_forward<A, M, N>(
    input_layernorm: &mut N,
    self_attn: &mut A,
    post_attention_layernorm: &mut N,
    mlp: &mut M,
    x: &Array,
    mask: Option<&Array>,
    cache: Option<(&mut KVCache, usize)>,
) -> Result<Array, Exception>
where
    A: AttentionModule,
    M: MlpModule,
    N: NormModule,
{
    // Pre-norm + attention + residual
    let normed = input_layernorm.forward(x)?;
    let attn_out = self_attn.forward_with_cache(&normed, mask, cache)?;
    let h = x.add(&attn_out);

    // Pre-norm + MLP + residual
    let normed = post_attention_layernorm.forward(&h)?;
    let mlp_out = mlp.forward(&normed)?;
    Ok(h.add(&mlp_out))
}
