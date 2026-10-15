//! Scaled dot-product attention for the model forwards.
//!
//! Plain attention, with any mask, is MLX's fused kernel, which also serves
//! attention under autograd, so one path covers inference and training
//! (`pmetal-runtime`'s `sdpa`). Attention sinks (gpt-oss) are the runtime's
//! `sdpa_with_sinks`. Logit softcapping (Gemma 2) has no fused kernel and is
//! computed from ops, with the scores held in f32.
//!
//! A hand-written Metal attention kernel used to sit in front of MLX's; it was
//! slower than MLX at every prefill length measured (2.4x to 2.6x at 512 to
//! 8192 tokens), wrong for odd lengths, aligned its causal mask to the
//! top-left instead of MLX's bottom-right for a query shorter than its keys,
//! and could not be differentiated, so it was removed.
//!
//! Masks are additive: `0` where a query may attend and `-inf` where it may
//! not; a boolean mask is true where it may. The causal mask and the sliding window are aligned to the bottom
//! right, so a query shorter than its keys (the continuation of a cached
//! prompt, or one decoded token) sees the most recent keys.

use crate::compat::{Array, Dtype, Exception};

/// Attention mask type for fused attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionMaskType {
    /// No mask (for bidirectional attention).
    None,
    /// Causal mask (lower triangular, auto-generated).
    Causal,
    /// Sliding window causal mask with given window size.
    SlidingWindow(i32),
}

/// Configuration for fused attention.
#[derive(Debug, Clone)]
pub struct FusedAttentionConfig {
    /// Number of query heads.
    pub num_heads: i32,
    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: i32,
    /// Head dimension (key/query dimension used for scoring).
    pub head_dim: i32,
    /// Softmax scaling factor (default: 1/sqrt(head_dim)).
    pub scale: f32,
    /// Mask type.
    pub mask_type: AttentionMaskType,
    /// Optional attention logit softcapping (Gemma2 style).
    pub logit_softcapping: Option<f32>,
}

impl FusedAttentionConfig {
    /// Create a new config with standard scaling.
    pub fn new(num_heads: i32, num_kv_heads: i32, head_dim: i32) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: AttentionMaskType::Causal,
            logit_softcapping: None,
        }
    }

    /// Set custom scaling factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set mask type.
    pub fn with_mask_type(mut self, mask_type: AttentionMaskType) -> Self {
        self.mask_type = mask_type;
        self
    }

    /// Set logit softcapping (for Gemma2).
    pub fn with_logit_softcapping(mut self, cap: f32) -> Self {
        self.logit_softcapping = Some(cap);
        self
    }
}

/// How a call tells the kernel which keys each query sees: a mode MLX
/// understands (`"causal"`, or `""` for no mask) and, in place of it, an
/// additive array.
struct Masking {
    mode: &'static str,
    array: Option<Array>,
}

impl Masking {
    /// The custom mask if the caller gave one, else what the configuration's
    /// mask type asks for.
    ///
    /// MLX requires the additive mask to promote to the attention output
    /// dtype, so an f32 mask against a bf16 model is a hard error, not a
    /// silent upcast. Mask builders (`create_causal_mask` and friends) all
    /// produce f32 regardless of the model, so the mask is coerced here, at
    /// the one point every architecture funnels through, rather than in each
    /// of them. A thrown op returns a placeholder that broadcasts through the
    /// layer, so the failure would not even be loud.
    fn resolve(
        config: &FusedAttentionConfig,
        custom_mask: Option<&Array>,
        queries: &Array,
        keys: &Array,
    ) -> Result<Self, Exception> {
        let dtype = queries.dtype().as_i32();
        if let Some(mask) = custom_mask {
            // A boolean mask (true = may attend) is MLX's own and passes
            // through; coerced it would read as the additive values 0 and 1.
            let keep = matches!(mask.dtype(), Dtype::Bool) || mask.dtype() == queries.dtype();
            return Ok(Self {
                mode: "",
                array: Some(if keep {
                    mask.clone()
                } else {
                    mask.as_dtype(dtype)
                }),
            });
        }
        Ok(match config.mask_type {
            AttentionMaskType::None => Self {
                mode: "",
                array: None,
            },
            AttentionMaskType::Causal => Self {
                mode: "causal",
                array: None,
            },
            AttentionMaskType::SlidingWindow(window) => Self {
                mode: "",
                array: Some(
                    create_sliding_window_mask(queries.dim(2), keys.dim(2), window)?
                        .as_dtype(dtype),
                ),
            },
        })
    }
}

/// Fused scaled dot-product attention.
///
/// Computes: softmax(Q @ K.T * scale + mask) @ V
///
/// Uses MLX's optimized Metal kernels for maximum performance. A custom mask
/// replaces the configuration's mask type. With
/// [`logit_softcapping`](FusedAttentionConfig::logit_softcapping) the scores
/// are `cap · tanh(scores / cap)` before the mask and the softmax, which no
/// fused kernel does; that path forms the `[query, key]` scores in f32.
///
/// # Arguments
/// * `queries` - Query tensor [batch, n_heads, seq_len, head_dim]
/// * `keys` - Key tensor [batch, n_kv_heads, seq_len, head_dim] (NOT pre-expanded for GQA)
/// * `values` - Value tensor [batch, n_kv_heads, seq_len, value_dim] (NOT pre-expanded for GQA)
/// * `config` - Attention configuration
/// * `custom_mask` - Optional attention mask [batch?, 1?, seq_len, seq_len]: additive, or boolean with true where a query may attend
///
/// # Returns
/// Attention output [batch, n_heads, seq_len, value_dim], in the queries' dtype
pub fn fused_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
) -> Result<Array, Exception> {
    if let (None, AttentionMaskType::SlidingWindow(window)) = (custom_mask, config.mask_type)
        && window > 0
        && queries.dim(2) == keys.dim(2)
        && queries.dim(2) > 2 * window
    {
        return blocked_sliding_sdpa(queries, keys, values, config, window);
    }
    let masking = Masking::resolve(config, custom_mask, queries, keys)?;
    if let Some(cap) = config.logit_softcapping {
        return softcapped_sdpa(queries, keys, values, config.scale, cap, &masking);
    }
    Ok(match &masking.array {
        Some(mask) => queries.sdpa_with_mask(keys, values, config.scale, Some(mask)),
        None => queries.sdpa(keys, values, config.scale, masking.mode),
    })
}

/// Sliding-window attention over a whole sequence (queries and keys the same
/// length, no cache), a block of `window` queries at a time.
///
/// A query block starting at `start` reaches back to key `start - window + 1`
/// and no further, so it is attended against those keys and its own with the
/// block's band mask, instead of against every key under an `[L, L]` mask. The
/// result is the full masked attention's, row for row (each query's softmax
/// still sees every key it may attend to); what changes is the cost. The scores
/// the kernel computes and the ones backward keeps grow with `L * window`
/// rather than `L * L`. Measured over a forward and backward at 8192 tokens
/// (16 query heads over 8 of width 256, window 1024, bf16): peak memory
/// 6.8 GB to 1.8 GB and 287 ms to 83 ms; at 4096 tokens 2.0 GB to 0.96 GB and
/// 73 ms to 38 ms. Full causal attention, which MLX has fused kernels for,
/// takes 1.0 GB and 183 ms at 8192.
fn blocked_sliding_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    window: i32,
) -> Result<Array, Exception> {
    let len = queries.dim(2);
    let dtype = queries.dtype().as_i32();
    let (batch, heads, kv_heads) = (queries.dim(0), queries.dim(1), keys.dim(1));
    let (width, value_width) = (queries.dim(3), values.dim(3));
    let mut blocks = Vec::with_capacity(((len + window - 1) / window) as usize);
    let mut start = 0;
    while start < len {
        let end = (start + window).min(len);
        let key_start = (start - window + 1).max(0);
        let q = queries.slice(&[0, 0, start, 0], &[batch, heads, end, width]);
        let k = keys.slice(&[0, 0, key_start, 0], &[batch, kv_heads, end, width]);
        let v = values.slice(&[0, 0, key_start, 0], &[batch, kv_heads, end, value_width]);
        let mask =
            create_sliding_window_mask(end - start, end - key_start, window)?.as_dtype(dtype);
        blocks.push(match config.logit_softcapping {
            Some(cap) => softcapped_sdpa(
                &q,
                &k,
                &v,
                config.scale,
                cap,
                &Masking {
                    mode: "",
                    array: Some(mask),
                },
            )?,
            None => q.sdpa_with_mask(&k, &v, config.scale, Some(&mask)),
        });
        start = end;
    }
    Ok(crate::compat::ops::concatenate_axis(
        &blocks.iter().collect::<Vec<_>>(),
        2,
    ))
}

/// Attention with logit softcapping (Gemma 2): the scores `scale · Q Kᵀ` go
/// through `cap · tanh(scores / cap)` before the mask and the softmax.
///
/// MLX's fused kernel has no hook between the scores and the softmax, so this
/// builds them from ops, and holds them in f32: a bf16 score of 40 is only
/// good to 0.1, which the softmax turns into a 10% error in a weight. Grouped
/// heads share their keys and values by broadcasting a group axis in the
/// batched matmul, so neither is copied per query head.
fn softcapped_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    cap: f32,
    masking: &Masking,
) -> Result<Array, Exception> {
    let f32_ = Dtype::Float32.as_i32();
    let (batch, heads, q_len, width) = (
        queries.dim(0),
        queries.dim(1),
        queries.dim(2),
        queries.dim(3),
    );
    let (kv_heads, k_len) = (keys.dim(1), keys.dim(2));
    if kv_heads == 0 || heads % kv_heads != 0 {
        return Err(Exception::custom(format!(
            "softcapped attention: {heads} query heads over {kv_heads} key heads"
        )));
    }
    let group = heads / kv_heads;

    // [B, Hk, G, Lq, D] against [B, Hk, 1, D, Lk]: the group axis broadcasts.
    let q = queries
        .as_dtype(f32_)
        .reshape(&[batch, kv_heads, group, q_len, width]);
    let k_t = keys
        .as_dtype(f32_)
        .reshape(&[batch, kv_heads, 1, k_len, width])
        .transpose_axes(&[0, 1, 2, 4, 3]);
    let mut scores = q.matmul(&k_t).multiply(&Array::from_f32(scale));

    scores = super::softcap(&scores, cap);

    let mask = match (&masking.array, masking.mode) {
        (Some(mask), _) if mask.dtype() == Dtype::Bool => {
            Some(mask.where_cond(&Array::from_f32(0.0), &Array::from_f32(f32::NEG_INFINITY)))
        }
        (Some(mask), _) => Some(mask.as_dtype(f32_)),
        (None, "causal") => Some(create_causal_mask(q_len, k_len)?),
        _ => None,
    };
    if let Some(mask) = mask {
        scores = scores.add(&group_mask(&mask, batch, kv_heads, group, q_len, k_len)?);
    }

    let weights = scores.softmax(-1).as_dtype(values.dtype().as_i32());
    let values = values.reshape(&[batch, kv_heads, 1, k_len, values.dim(3)]);
    let out = weights.matmul(&values); // [B, Hk, G, Lq, Dv]
    Ok(out
        .reshape(&[batch, heads, q_len, values.dim(4)])
        .as_dtype(queries.dtype().as_i32()))
}

/// An additive mask `[Lq, Lk]`, `[1 | B, 1, Lq, Lk]` or `[1 | B, H, Lq, Lk]`
/// laid out against scores of shape `[B, Hk, G, Lq, Lk]`.
fn group_mask(
    mask: &Array,
    batch: i32,
    kv_heads: i32,
    group: i32,
    q_len: i32,
    k_len: i32,
) -> Result<Array, Exception> {
    match mask.ndim() {
        2 => Ok(mask.reshape(&[1, 1, 1, q_len, k_len])),
        4 if mask.dim(1) == 1 => Ok(mask.reshape(&[mask.dim(0), 1, 1, q_len, k_len])),
        4 if mask.dim(1) == kv_heads * group => {
            Ok(mask.reshape(&[mask.dim(0), kv_heads, group, q_len, k_len]))
        }
        _ => Err(Exception::custom(format!(
            "softcapped attention: a mask of shape {:?} does not fit scores of {} x {} x {} x {} x {}",
            mask.shape(),
            batch,
            kv_heads,
            group,
            q_len,
            k_len
        ))),
    }
}

/// Scaled dot-product attention with **attention sinks** (GPT-OSS).
///
/// One learned logit per head joins the softmax denominator as a virtual key
/// that contributes no value: for each (batch, head, query) row the softmax is
/// taken over `[scores, sink_h]` and the sink column is dropped from the
/// weighted sum.
///
/// ```text
///   scores  = scale · (Q · Kᵀ) + mask          # [B, H, L, S]
///   p       = softmax([scores, sink_h], -1)     # [B, H, L, S+1]
///   out     = p[..., :S] @ V
/// ```
///
/// This is the runtime's fused `sdpa_with_sinks`: grouped K/V are not
/// expanded and the `[L, S]` scores are not formed in memory, and its
/// gradient reaches the queries, keys, values and the sinks. `sinks` is a
/// 1-D `[n_heads]` array. Logit softcapping is not supported here (GPT-OSS
/// uses sinks, not softcap).
pub fn sink_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
    sinks: &Array,
) -> Result<Array, Exception> {
    if config.logit_softcapping.is_some() {
        return Err(Exception::custom(
            "sink_sdpa: attention sinks and logit softcapping are not supported together",
        ));
    }
    let masking = Masking::resolve(config, custom_mask, queries, keys)?;
    // MLX requires the sinks to promote to the output dtype.
    let sinks = sinks.as_dtype(queries.dtype().as_i32());
    Ok(queries.sdpa_with_sinks(
        keys,
        values,
        config.scale,
        masking.mode,
        masking.array.as_ref(),
        &sinks,
    ))
}

/// Create causal attention mask.
///
/// Returns mask where positions can only attend to earlier positions.
/// Shape: [1, 1, query_len, key_len] with -inf for masked positions.
fn create_causal_mask(query_len: i32, key_len: i32) -> Result<Array, Exception> {
    // Create lower triangular mask aligned to bottom-right for KV cache support
    // When query_len < key_len (generation), queries attend to all past keys
    let mask = Array::tri(
        query_len,
        key_len,
        key_len - query_len,
        Dtype::Float32.as_i32(),
    );
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);

    // Where mask is 0, put -inf; where mask is 1, put 0
    let mask = mask.equal(&zero).where_cond(&neg_inf, &zero);

    // Add broadcast dimensions [1, 1, query_len, key_len]
    Ok(mask.reshape(&[1, 1, query_len, key_len]))
}

/// A mask the caller built (a block-diagonal causal one for packed sequences,
/// a padding mask) restricted to the sliding window of a local layer.
///
/// A custom mask replaces the configuration's mask type wholesale, so a layer
/// that hands one to [`fused_sdpa`] attends as far back as that mask lets it,
/// which for a causal mask is the whole sequence: a sliding-window model
/// trained under one is a different model from the one it is served as, as
/// soon as a sequence outruns the window. The result is `mask` plus the causal
/// band of `window` keys, in the shapes [`create_sliding_window_mask`] has
/// (queries aligned to the end of the keys). `mask` is additive, or boolean
/// with true where a query may attend; the result is additive.
pub fn restrict_to_sliding_window(
    mask: &Array,
    query_len: i32,
    key_len: i32,
    window: i32,
) -> Result<Array, Exception> {
    let band = create_sliding_window_mask(query_len, key_len, window)?;
    let additive = if mask.dtype() == Dtype::Bool {
        mask.where_cond(&Array::from_f32(0.0), &Array::from_f32(f32::NEG_INFINITY))
    } else {
        mask.as_dtype(Dtype::Float32.as_i32())
    };
    Ok(additive.add(&band))
}

/// Create sliding window causal mask.
///
/// Positions can only attend to positions within `window_size` distance.
/// Shape: [1, 1, query_len, key_len] with -inf for masked positions.
///
/// Public because a training forward pass, which cannot always route through
/// [`fused_sdpa`], still has to see the same geometry the inference path does.
/// Two builders for one window is how a model comes to be fine-tuned under a
/// mask it will never be served under.
pub fn create_sliding_window_mask(
    query_len: i32,
    key_len: i32,
    window_size: i32,
) -> Result<Array, Exception> {
    // Align the causal band to the bottom-right so decode queries attend to
    // the most recent `window_size` cached tokens rather than broadcasting a
    // square [query_len, query_len] mask over the full KV axis.
    let lower = Array::tri(
        query_len,
        key_len,
        key_len - query_len - window_size,
        Dtype::Float32.as_i32(),
    );
    let upper = Array::tri(
        query_len,
        key_len,
        key_len - query_len,
        Dtype::Float32.as_i32(),
    );

    // Valid positions: where upper is 1 AND lower is 0
    let zero = Array::from_f32(0.0);
    let valid = upper.subtract(&lower);

    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let mask = valid.equal(&zero).where_cond(&neg_inf, &zero);

    Ok(mask.reshape(&[1, 1, query_len, key_len]))
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
