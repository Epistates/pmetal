//! Per-layer + full-model weight bundles, plus Hadamard preconditioning that
//! absorbs a random rotation into Q/K/V/O so cached K/V coordinates quantize
//! more uniformly (TurboQuant / PolarQuant / QuIP# insight).

use crate::InlineArray;

use super::QuantizationConfig;

/// A projection weight that is either a dense bf16 tensor or a 4-bit
/// quantized triple (`weight`, `scales`, `biases`).
///
/// - **Dense**: single `InlineArray`, used with `x.matmul(w)`.
/// - **Quantized**: three `InlineArray`s loaded from `{key}.weight`,
///   `{key}.scales`, `{key}.biases`; used with
///   `x.quantized_matmul(w, scales, biases, transpose=true, group_size, bits)`.
///
/// The `transpose=true` flag is standard: MLX stores quantized weights in
/// row-major `[out, in/group_size]` layout and expects the caller to signal
/// that the weight logically needs transposing for the matmul (i.e. the same
/// semantic as storing the dense weight transposed as `[in, out]`).
///
/// Dense weights are pre-transposed at load time (`w.t()`); quantized weights
/// are stored as-is from the checkpoint and the `transpose=true` flag handles
/// the layout internally inside `mx.quantized_matmul`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum LayerWeight {
    Dense(InlineArray),
    Quantized {
        weight: InlineArray, // packed uint32: shape [out, in/(32/bits)]
        scales: InlineArray, // per-group scale: shape [out, in/group_size]
        biases: InlineArray, // per-group bias:  shape [out, in/group_size]
        group_size: i32,
        bits: i32,
    },
}

impl LayerWeight {
    /// Get the underlying weight tensor (for use in copy_fresh or pointer export).
    pub fn weight_arr(&self) -> &InlineArray {
        match self {
            LayerWeight::Dense(w) => w,
            LayerWeight::Quantized { weight, .. } => weight,
        }
    }

    /// `x @ self` — dispatches to `quantized_matmul` or `matmul` as appropriate.
    ///
    /// For dense weights: `x.matmul(w)` — weight is pre-transposed `[in, out]`.
    /// For quantized:    `x.quantized_matmul(w, scales, biases, true, gs, bits)`.
    #[inline(always)]
    pub fn matmul_from(&self, x: &InlineArray) -> InlineArray {
        match self {
            LayerWeight::Dense(w) => x.matmul(w),
            LayerWeight::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => x.quantized_matmul(weight, scales, Some(biases), true, *group_size, *bits),
        }
    }

    /// Gather-matmul: `gather_mm` for dense, `gather_qmm` for quantized.
    ///
    /// Used by MoE expert dispatch.  All expert tensors in a layer must have
    /// the same variant — mixing dense and quantized experts is not supported.
    #[inline(always)]
    pub fn gather_mm_from(
        &self,
        x: &InlineArray,
        lhs_indices: Option<&InlineArray>,
        rhs_indices: Option<&InlineArray>,
        sorted: bool,
    ) -> InlineArray {
        match self {
            LayerWeight::Dense(w) => x.gather_mm(w, lhs_indices, rhs_indices, sorted),
            LayerWeight::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => x.gather_qmm(
                weight,
                scales,
                Some(biases),
                lhs_indices,
                rhs_indices,
                true,
                *group_size,
                *bits,
                sorted,
            ),
        }
    }

    /// Apply `copy_fresh` to all arrays in this weight (add zero + eval + detach).
    pub fn copy_fresh(&self, zero: &InlineArray) -> Self {
        match self {
            LayerWeight::Dense(w) => LayerWeight::Dense(copy_fresh_arr(w, zero)),
            LayerWeight::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                // For quantized weights the zero must be int32 (weight dtype)
                // for the weight tensor and float for scales/biases.
                // Use add-zero on each independently via eval+detach.
                let w2 = copy_fresh_arr(weight, zero);
                let s2 = copy_fresh_arr(scales, zero);
                let b2 = copy_fresh_arr(biases, zero);
                LayerWeight::Quantized {
                    weight: w2,
                    scales: s2,
                    biases: b2,
                    group_size: *group_size,
                    bits: *bits,
                }
            }
        }
    }
}

/// Force a single array into a fresh Metal buffer (add zero + eval + detach).
/// This is the implementation detail shared by `LayerWeight::copy_fresh`.
pub(super) fn copy_fresh_arr(w: &InlineArray, _hint_zero: &InlineArray) -> InlineArray {
    // For quantized weights (uint32 packed), skip copy_fresh — they're already
    // loaded directly through pmetal-bridge's MLX and don't need data duplication.
    let dt = w.dtype_raw();
    if dt == 3
    /* uint32 */
    {
        // Just eval + detach without add
        let mut fresh = w.clone();
        fresh.eval();
        fresh.detach();
        return fresh;
    }
    let own_zero = InlineArray::zeros(&[1], dt);
    let mut fresh = w.add(&own_zero);
    fresh.eval();
    fresh.detach();
    fresh
}

// ============================================================================
// Per-layer weights
// ============================================================================

pub(crate) struct LayerWeights {
    pub(crate) is_linear: bool,

    // Shared: layer norms (never quantized — 1D tensors)
    pub(crate) input_ln_w: InlineArray,
    pub(crate) input_ln_eps: f32,
    pub(crate) post_ln_w: InlineArray,
    pub(crate) post_ln_eps: f32,

    // Dense MLP (when !is_moe_layer)
    pub(crate) mlp_gate_w: Option<LayerWeight>,
    pub(crate) mlp_up_w: Option<LayerWeight>,
    pub(crate) mlp_down_w: Option<LayerWeight>,

    // MoE (when is_moe_layer)
    pub(crate) moe_router_w: Option<InlineArray>,
    pub(crate) moe_gate_w: Option<LayerWeight>,
    pub(crate) moe_up_w: Option<LayerWeight>,
    pub(crate) moe_down_w: Option<LayerWeight>,
    pub(crate) shared_gate_w: Option<LayerWeight>,
    pub(crate) shared_up_w: Option<LayerWeight>,
    pub(crate) shared_down_w: Option<LayerWeight>,
    pub(crate) shared_expert_gate_w: Option<InlineArray>,

    pub(crate) moe_top_k: i32,
    pub(crate) moe_norm_topk_prob: bool,
    pub(crate) is_moe_layer: bool,

    // Attention-specific (only when !is_linear)
    pub(crate) attn_q_w: Option<LayerWeight>,
    pub(crate) attn_k_w: Option<LayerWeight>,
    pub(crate) attn_v_w: Option<LayerWeight>,
    pub(crate) attn_o_w: Option<LayerWeight>,
    pub(crate) attn_q_norm_w: Option<InlineArray>,
    pub(crate) attn_q_norm_eps: f32,
    pub(crate) attn_k_norm_w: Option<InlineArray>,
    pub(crate) attn_k_norm_eps: f32,
    pub(crate) attn_n_heads: i32,
    pub(crate) attn_n_kv_heads: i32,
    pub(crate) attn_head_dim: i32,
    pub(crate) attn_scale: f32,
    pub(crate) attn_rope_dims: i32,
    pub(crate) attn_rope_base: f32,
    pub(crate) attn_rope_scale: f32,
    pub(crate) attn_gated: bool,

    // GDN-specific (only when is_linear)
    pub(crate) gdn_qkv_w: Option<LayerWeight>,
    pub(crate) gdn_z_w: Option<LayerWeight>,
    pub(crate) gdn_b_w: Option<LayerWeight>,
    pub(crate) gdn_a_w: Option<LayerWeight>,
    pub(crate) gdn_conv_w: Option<InlineArray>,
    pub(crate) gdn_q_nw: Option<InlineArray>,
    pub(crate) gdn_k_nw: Option<InlineArray>,
    pub(crate) gdn_a_log: Option<InlineArray>,
    pub(crate) gdn_dt_bias: Option<InlineArray>,
    pub(crate) gdn_norm_w: Option<InlineArray>,
    pub(crate) gdn_norm_eps: f32,
    pub(crate) gdn_out_w: Option<LayerWeight>,
    pub(crate) gdn_nv: i32,
    pub(crate) gdn_nk: i32,
    pub(crate) gdn_dk: i32,
    pub(crate) gdn_dv: i32,
    pub(crate) gdn_kd: i32,
    pub(crate) gdn_cd: i32,
    pub(crate) gdn_ck: i32,
}

// ============================================================================
// Full model weights
// ============================================================================

/// All model weights as InlineArray. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,
    /// Quantized embedding: scales + biases (None for dense models)
    pub embed_scales: Option<InlineArray>,
    pub embed_biases: Option<InlineArray>,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    ///
    /// Untied heads may be dense or quantized, just like the projection
    /// matrices inside the decoder blocks.
    pub lm_head_w: Option<LayerWeight>,
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizationConfig>,
    /// Per-layer weights — opaque to callers; only accessed via [`forward_step`].
    pub(crate) layers: Vec<LayerWeights>,
    /// Model activation dtype (e.g., 11 = bfloat16).
    pub model_dtype: i32,
    /// QJL projection matrix S [head_dim, head_dim] for key residual correction.
    ///
    /// Generated at load time when `apply_qjl_matrix` is called. Used during
    /// attention to compute sign(S · residual) for unbiased inner product
    /// estimation at Q2-Q3 bits. `None` when QJL is disabled.
    pub qjl_matrix: Option<InlineArray>,
}

impl std::fmt::Debug for NativeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeWeights")
            .field("layers", &self.layers.len())
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .field("model_dtype", &self.model_dtype)
            .finish()
    }
}

/// Seed for KV cache preconditioning rotation (distinct from TURBOQUANT_SEED).
const KV_PRECONDITION_SEED: u64 = 0x4b56_5052_4543_4f4e; // "KVPRECON"

/// Apply per-head random orthogonal rotation to attention projection weights.
///
/// Absorbs a random rotation R into Q/K/V/O weights at model load time so that
/// K/V vectors in the cache have more uniform coordinate distributions, improving
/// affine quantization quality at the same bit-width (TurboQuant/PolarQuant/QuIP# insight).
///
/// - Q: `W_q' = W_q @ R_block^T`  (queries in rotated space)
/// - K: `W_k' = W_k @ R_block^T`  (keys in rotated space → better quantization)
/// - V: `W_v' = W_v @ R_block^T`  (values in rotated space → better quantization)
/// - O: `W_o' = R_block @ W_o`    (undo rotation on output)
///
/// Since R is orthogonal (R^T R = I), attention scores Q@K^T are unchanged.
/// Zero runtime cost — rotation is in the weights, not the inference path.
pub fn apply_kv_preconditioning(weights: &mut NativeWeights) {
    use rand::SeedableRng;

    // Find head_dim from first attention layer
    let head_dim = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .map(|lw| lw.attn_head_dim)
        .unwrap_or(0);
    if head_dim == 0 {
        return;
    }

    // Generate deterministic orthogonal rotation R [head_dim, head_dim]
    let mut rng =
        rand::rngs::StdRng::seed_from_u64(KV_PRECONDITION_SEED ^ ((head_dim as u64) << 32));
    let r_data = crate::turboquant::generate_random_orthogonal(head_dim as usize, &mut rng);
    let r_f32 = InlineArray::from_f32_slice(&r_data, &[head_dim, head_dim]);
    // Cast rotation to model dtype to avoid f32 promotion cascade through weights
    let r = r_f32.as_dtype(weights.model_dtype);
    let r_t = r.transpose_axes(&[1, 0]); // R^T [head_dim, head_dim]

    eprintln!(
        "[PRECONDITION] Applying Hadamard preconditioning to attention weights (head_dim={})",
        head_dim
    );

    for lw in &mut weights.layers {
        if lw.is_linear {
            continue; // GDN layers — no Q/K/V/O
        }
        let n_heads = lw.attn_n_heads;
        let n_kv = lw.attn_n_kv_heads;

        if lw.attn_gated {
            // Gated attention (Qwen3.5): element-wise gate*sigmoid doesn't commute
            // with rotation, so we can only rotate Q (queries portion) and K.
            // V stays in original space → gate multiplication is correct.
            // O projection is unchanged (output stays in original space).
            // Key benefit: K preconditioning improves key quantization quality.
            rotate_projection_weight(&mut lw.attn_q_w, &r_t, n_heads, head_dim, true);
            rotate_projection_weight(&mut lw.attn_k_w, &r_t, n_kv, head_dim, false);
            // V and O untouched
        } else {
            // Non-gated attention (Qwen3): full Q/K/V/O rotation is exact.
            rotate_projection_weight(&mut lw.attn_q_w, &r_t, n_heads, head_dim, false);
            rotate_projection_weight(&mut lw.attn_k_w, &r_t, n_kv, head_dim, false);
            rotate_projection_weight(&mut lw.attn_v_w, &r_t, n_kv, head_dim, false);
            rotate_output_weight(&mut lw.attn_o_w, &r, n_heads, head_dim);
        }
    }
}

/// Generate and store the QJL projection matrix S on `weights`.
///
/// S is a [head_dim × head_dim] Gaussian random matrix used to project key
/// quantization residuals before taking their sign. Stored as the model dtype.
/// Must be called after `apply_kv_preconditioning` if both are used.
///
/// Only the uniform Q2-Q3 path uses QJL; mixed-bit paths are unaffected.
pub fn apply_qjl_matrix(weights: &mut NativeWeights) {
    use rand::SeedableRng;

    let head_dim = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .map(|lw| lw.attn_head_dim)
        .unwrap_or(0);
    if head_dim == 0 {
        return;
    }

    // Use a seed distinct from the Hadamard rotation seed so S is independent of R.
    let qjl_seed = KV_PRECONDITION_SEED ^ 0x514a_4c00; // "QJL\0"
    let mut rng = rand::rngs::StdRng::seed_from_u64(qjl_seed ^ ((head_dim as u64) << 32));
    let s_data = crate::turboquant::generate_random_projection(head_dim as usize, &mut rng);
    let s_f32 = InlineArray::from_f32_slice(&s_data, &[head_dim, head_dim]);
    let s = s_f32.as_dtype(weights.model_dtype);
    // Eval and detach into a fresh Metal buffer.
    let zero = InlineArray::scalar_with_dtype(0.0, weights.model_dtype);
    let mut s = s.add(&zero);
    s.eval();
    s.detach();
    weights.qjl_matrix = Some(s);

    eprintln!(
        "[QJL] QJL projection matrix generated (head_dim={})",
        head_dim
    );
}

/// Reorder head dimensions so outlier channels come first, enabling mixed-bit
/// quantization where outlier channels get higher precision.
///
/// Computes per-channel L2 norms from K projection weights, finds the top
/// `outlier_fraction` channels, and builds a permutation that moves them to
/// the front of each head. The permutation is then absorbed into Q/K/V/O
/// projection weights — zero runtime cost.
///
/// Permutation matrices commute with element-wise nonlinearities (sigmoid),
/// so this is safe for gated attention unlike Hadamard rotation.
pub fn apply_outlier_permutation(weights: &mut NativeWeights, outlier_fraction: f32) -> i32 {
    // Find head_dim from first attention layer
    let (head_dim, _) = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .map(|lw| (lw.attn_head_dim, lw.attn_n_kv_heads))
        .unwrap_or((0, 0));
    if head_dim == 0 {
        return 0;
    }

    let outlier_count = (head_dim as f32 * outlier_fraction).round() as i32;
    // Round to group_size boundary for clean quantization
    let outlier_count = (outlier_count / 64) * 64;
    if outlier_count == 0 || outlier_count >= head_dim {
        return 0;
    }

    // Compute per-channel importance from K projection weights (first attn layer).
    // Use L2 norm of each output channel across the hidden dimension.
    let k_w = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .and_then(|lw| match &lw.attn_k_w {
            Some(LayerWeight::Dense(w)) => Some(w.clone()),
            _ => None,
        });
    let Some(k_weight) = k_w else { return 0 };

    // k_weight shape: [hidden, n_kv * head_dim] (pre-transposed)
    // Compute L2 norm per output channel (columns)
    let col_norms = k_weight.square().sum_axis(0, false); // [n_kv * head_dim]
    col_norms.eval();

    // Build per-head permutation: sort channels by norm (descending), outliers first.
    // The permutation is the same for all heads (same relative ordering).
    let first_head_norms = col_norms.slice(&[0], &[head_dim]);
    first_head_norms.eval();
    let norms_slice: &[f32] = first_head_norms.as_slice();
    let mut indices: Vec<usize> = (0..head_dim as usize).collect();
    indices.sort_by(|&a, &b| {
        norms_slice[b]
            .partial_cmp(&norms_slice[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build permutation matrix P [head_dim, head_dim]
    let mut p_data = vec![0.0f32; (head_dim * head_dim) as usize];
    for (new_pos, &old_pos) in indices.iter().enumerate() {
        p_data[new_pos * head_dim as usize + old_pos] = 1.0;
    }
    let p = InlineArray::from_f32_slice(&p_data, &[head_dim, head_dim]);
    let p = p.as_dtype(weights.model_dtype);
    let p_t = p.transpose_axes(&[1, 0]);

    eprintln!(
        "[PRECONDITION] Applying outlier channel permutation (outlier_count={}, head_dim={})",
        outlier_count, head_dim
    );

    // Apply P to all attention projection weights.
    // W' = W @ P^T moves outlier channels to front of each head.
    // Permutation commutes with sigmoid, so safe for gated attention.
    for lw in &mut weights.layers {
        if lw.is_linear {
            continue;
        }
        let n_heads = lw.attn_n_heads;
        let n_kv = lw.attn_n_kv_heads;

        // Q (including gate for Qwen3.5): permute query channels, gate channels
        rotate_projection_weight(&mut lw.attn_q_w, &p_t, n_heads, head_dim, lw.attn_gated);
        rotate_projection_weight(&mut lw.attn_k_w, &p_t, n_kv, head_dim, false);
        rotate_projection_weight(&mut lw.attn_v_w, &p_t, n_kv, head_dim, false);
        rotate_output_weight(&mut lw.attn_o_w, &p, n_heads, head_dim);
    }

    outlier_count
}

/// Rotate a Q/K/V projection weight in-place: `W' = W @ R_block^T`
/// For gated Q (Qwen3.5): treats both query and gate halves as heads to rotate.
fn rotate_projection_weight(
    w_opt: &mut Option<LayerWeight>,
    r_t: &InlineArray,
    n_heads: i32,
    head_dim: i32,
    gated: bool,
) {
    let w = match w_opt {
        Some(LayerWeight::Dense(w)) => w,
        _ => return,
    };
    let shape = w.shape();
    let hidden = shape[0];

    if gated {
        // Gated Q (Qwen3.5): W_q is [hidden, n_heads * head_dim * 2].
        // Per-head layout is interleaved: [head0_q(256), head0_gate(256), head1_q(256), ...].
        // After reshape to [hidden, n_heads, 2*head_dim], split at head_dim:
        //   queries = [:, :, :head_dim]  →  rotate by R^T
        //   gate    = [:, :, head_dim:]  →  leave alone (sigmoid doesn't commute)
        let w_3d = w.reshape(&[hidden, n_heads, head_dim * 2]);
        let w_queries = w_3d.slice(&[0, 0, 0], &[hidden, n_heads, head_dim]);
        let w_gate = w_3d.slice(&[0, 0, head_dim], &[hidden, n_heads, head_dim * 2]);

        let w_q_rot = w_queries.matmul(r_t); // [hidden, n_heads, head_dim] @ [head_dim, head_dim]

        // Concatenate rotated queries + original gate along last dim
        let w_combined = w_q_rot.kv_cache_append(&w_gate, 2); // [hidden, n_heads, 2*head_dim]
        *w_opt = Some(LayerWeight::Dense(
            w_combined.reshape(&[hidden, n_heads * head_dim * 2]),
        ));
    } else {
        // Standard Q/K/V: rotate all heads
        let w_3d = w.reshape(&[hidden, n_heads, head_dim]);
        let w_rot = w_3d.matmul(r_t);
        *w_opt = Some(LayerWeight::Dense(
            w_rot.reshape(&[hidden, n_heads * head_dim]),
        ));
    }
}

/// Rotate an O-projection weight in-place: `W_o' = R_block @ W_o`
fn rotate_output_weight(
    w_opt: &mut Option<LayerWeight>,
    r: &InlineArray,
    n_heads: i32,
    head_dim: i32,
) {
    let w = match w_opt {
        Some(LayerWeight::Dense(w)) => w,
        _ => return,
    };
    let shape = w.shape();
    let hidden = shape[1];
    let w_3d = w.reshape(&[n_heads, head_dim, hidden]);
    // Batched matmul: [head_dim, head_dim] @ [n_heads, head_dim, hidden] → [n_heads, head_dim, hidden]
    let w_rot = r.matmul(&w_3d);
    *w_opt = Some(LayerWeight::Dense(
        w_rot.reshape(&[n_heads * head_dim, hidden]),
    ));
}
