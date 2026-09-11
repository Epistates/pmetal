//! Fused attention kernel using MLX's fast SDPA.
//!
//! This module wraps MLX's `scaled_dot_product_attention` which provides:
//! - Metal-optimized kernels for single-token generation (query_seq_len = 1)
//! - Native support for GQA/MQA without manual K/V head expansion
//! - Memory-efficient attention computation
//! - Automatic float32 softmax precision for numerical stability
//!
//! Performance Benefits:
//! - 30-50% faster than manual SDPA for single-token inference
//! - Reduced memory bandwidth by avoiding intermediate tensor materialization
//! - Native GQA support eliminates expand_kv_heads overhead

use std::{sync::OnceLock, time::Instant};

use crate::ArrayDtypeExt;
use pmetal_bridge::compat::{Array, Dtype, Exception, ops, random};
use pmetal_metal::{
    FlashAttention, FlashAttentionConfig as MetalFlashAttentionConfig, KernelDispatch,
    MetalContext, MppFlashAttention, MppFlashAttentionConfig,
    context::{DeviceProperties, DeviceTier},
};
use serde::{Deserialize, Serialize};

use super::persistent_cache::PersistentChoiceCache;
use super::utils::{array_to_metal_buffer_f16, metal_buffer_into_array_f16};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum AttentionBackendChoice {
    MlxFast,
    MetalFlash,
    MppFlash,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AttentionDispatchKey {
    device_name: String,
    device_tier: &'static str,
    dtype: &'static str,
    batch: i32,
    num_heads: i32,
    num_kv_heads: i32,
    query_seq_len: i32,
    kv_seq_len: i32,
    head_dim: i32,
    value_head_dim: i32,
    mask_type: AttentionMaskType,
    softcap_bits: Option<u32>,
}

static ATTENTION_BACKEND_CACHE: OnceLock<PersistentChoiceCache<AttentionBackendChoice>> =
    OnceLock::new();

fn attention_backend_cache() -> &'static PersistentChoiceCache<AttentionBackendChoice> {
    ATTENTION_BACKEND_CACHE.get_or_init(|| PersistentChoiceCache::new("attention_backends.json"))
}

fn device_tier_key(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Base => "base",
        DeviceTier::Pro => "pro",
        DeviceTier::Max => "max",
        DeviceTier::Ultra => "ultra",
    }
}

fn dtype_key(dtype: Dtype) -> Option<&'static str> {
    match dtype {
        Dtype::Float16 => Some("f16"),
        Dtype::Float32 => Some("f32"),
        Dtype::Bfloat16 => Some("bf16"),
        _ => None,
    }
}

/// Bucket a sequence length so that adjacent decode steps share a cache
/// entry. Without this, every decode step grows the KV cache by 1 and
/// forces a fresh backend benchmark — that's a 5-10x slowdown per step
/// compared to the steady-state cached backend. Bucketing to the next
/// power of two means we re-benchmark only when the cache passes a
/// power-of-two boundary, which doubles the amortisation window each
/// time.
fn bucket_seq_len(seq_len: i32) -> i32 {
    if seq_len <= 1 {
        return 1;
    }
    let mut bucket = 1i32;
    while bucket < seq_len {
        bucket <<= 1;
    }
    bucket
}

impl AttentionDispatchKey {
    fn new(
        props: &DeviceProperties,
        dtype: Dtype,
        q_shape: &[i32],
        k_shape: &[i32],
        config: &FusedAttentionConfig,
    ) -> Option<Self> {
        Some(Self {
            device_name: props.name.clone(),
            device_tier: device_tier_key(props.device_tier),
            dtype: dtype_key(dtype)?,
            batch: q_shape[0],
            num_heads: q_shape[1],
            num_kv_heads: k_shape[1],
            query_seq_len: bucket_seq_len(q_shape[2]),
            kv_seq_len: bucket_seq_len(k_shape[2]),
            head_dim: q_shape[3],
            value_head_dim: config.effective_value_head_dim(),
            mask_type: config.mask_type,
            softcap_bits: config.logit_softcapping.map(f32::to_bits),
        })
    }

    fn cache_key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{:?}:{:?}",
            self.device_name,
            self.device_tier,
            self.dtype,
            self.batch,
            self.num_heads,
            self.num_kv_heads,
            self.query_seq_len,
            self.kv_seq_len,
            self.head_dim,
            self.value_head_dim,
            self.mask_type,
            self.softcap_bits
        )
    }
}

fn cached_attention_backend(key: &AttentionDispatchKey) -> Option<AttentionBackendChoice> {
    attention_backend_cache().get(&key.cache_key())
}

fn cache_attention_backend(key: AttentionDispatchKey, backend: AttentionBackendChoice) {
    attention_backend_cache().insert(key.cache_key(), backend);
}

#[cfg(test)]
fn clear_cached_attention_backends() {
    attention_backend_cache().clear();
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
    /// Value head dimension. When `None`, defaults to `head_dim`.
    /// Set this for architectures with asymmetric K/V dimensions (e.g. DeepSeek MLA).
    pub value_head_dim: Option<i32>,
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
            value_head_dim: None,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: AttentionMaskType::Causal,
            logit_softcapping: None,
        }
    }

    /// Set a distinct value head dimension (for architectures like DeepSeek MLA
    /// where value projections have different width than key/query projections).
    pub fn with_value_head_dim(mut self, value_head_dim: i32) -> Self {
        self.value_head_dim = Some(value_head_dim);
        self
    }

    /// Effective value head dimension (falls back to `head_dim` when unset).
    #[must_use]
    pub fn effective_value_head_dim(&self) -> i32 {
        self.value_head_dim.unwrap_or(self.head_dim)
    }

    /// Whether key and value head dimensions differ.
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        self.value_head_dim.is_some_and(|v| v != self.head_dim)
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

    /// Check if this is grouped-query attention.
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// Get number of query heads per KV head.
    #[must_use]
    pub fn num_groups(&self) -> i32 {
        self.num_heads / self.num_kv_heads
    }
}

/// Fused scaled dot-product attention.
///
/// Computes: softmax(Q @ K.T / sqrt(d_k) + mask) @ V
///
/// Uses MLX's optimized Metal kernels for maximum performance.
///
/// # Arguments
/// * `queries` - Query tensor [batch, n_heads, seq_len, head_dim]
/// * `keys` - Key tensor [batch, n_kv_heads, seq_len, head_dim] (NOT pre-expanded for GQA)
/// * `values` - Value tensor [batch, n_kv_heads, seq_len, head_dim] (NOT pre-expanded for GQA)
/// * `config` - Attention configuration
/// * `custom_mask` - Optional custom attention mask [batch?, 1?, seq_len, seq_len]
///
/// # Returns
/// Attention output [batch, n_heads, seq_len, head_dim]
///
/// # Note
/// For GQA/MQA, pass K/V tensors with their native number of heads.
/// The fused kernel handles head repetition internally, avoiding memory overhead.
pub fn fused_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
) -> Result<Array, Exception> {
    // MLX requires the additive mask to promote to the attention output dtype,
    // so an f32 mask against a bf16 model is a hard error — not a silent
    // upcast. Mask builders (`create_causal_mask` and friends) all produce f32
    // regardless of the model, so coerce here, at the one point every
    // architecture funnels through, rather than in each of them.
    let coerced_mask = custom_mask.and_then(|mask| {
        (mask.dtype() != queries.dtype()).then(|| mask.as_dtype(queries.dtype().as_i32()))
    });
    let custom_mask = coerced_mask.as_ref().or(custom_mask);

    if let Some(output) =
        try_selected_attention_backend(queries, keys, values, config, custom_mask)?
    {
        return Ok(output);
    }

    if let Some(output) = try_metal_flash_attention(queries, keys, values, config, custom_mask)? {
        return Ok(output);
    }

    // Apply logit softcapping if configured (Gemma2 style)
    // This requires pre/post processing around attention
    if let Some(cap) = config.logit_softcapping {
        return manual_sdpa_with_softcapping(queries, keys, values, config, custom_mask, cap);
    }

    fast_fused_sdpa(queries, keys, values, config, custom_mask)
}

fn try_selected_attention_backend(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
) -> Result<Option<Array>, Exception> {
    let metal_ctx = match MetalContext::global() {
        Ok(ctx) => ctx,
        Err(error) => {
            tracing::debug!("Metal attention selection unavailable, falling back: {error}");
            return Ok(None);
        }
    };

    let Some(dispatch_key) = benchmarkable_attention_dispatch_key(
        queries,
        keys,
        values,
        config,
        custom_mask,
        metal_ctx.properties(),
    ) else {
        return Ok(None);
    };

    if let Some(backend) = cached_attention_backend(&dispatch_key) {
        return execute_attention_backend(backend, queries, keys, values, config, &metal_ctx)
            .map(Some)
            .or_else(|error| {
                tracing::debug!(
                    "Cached {:?} attention backend failed, falling back to MLX fast SDPA: {error}",
                    backend
                );
                cache_attention_backend(dispatch_key.clone(), AttentionBackendChoice::MlxFast);
                reference_attention_output(queries, keys, values, config).map(Some)
            });
    }

    let (backend, output) =
        benchmark_attention_backends(queries, keys, values, config, &metal_ctx)?;
    cache_attention_backend(dispatch_key, backend);
    Ok(Some(output))
}

fn benchmarkable_attention_dispatch_key(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
    props: &DeviceProperties,
) -> Option<AttentionDispatchKey> {
    if custom_mask.is_some() {
        return None;
    }

    if !flash_attention_supported(queries, keys, values, custom_mask) {
        return None;
    }

    let q_shape = queries.shape();
    let k_shape = keys.shape();
    AttentionDispatchKey::new(props, queries.dtype(), q_shape, k_shape, config)
}

fn execute_attention_backend(
    backend: AttentionBackendChoice,
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    metal_ctx: &std::sync::Arc<MetalContext>,
) -> Result<Array, Exception> {
    match backend {
        AttentionBackendChoice::MlxFast => {
            reference_attention_output(queries, keys, values, config)
        }
        AttentionBackendChoice::MetalFlash => {
            run_metal_flash_attention(queries, keys, values, config, metal_ctx)
        }
        AttentionBackendChoice::MppFlash => {
            run_mpp_flash_attention(queries, keys, values, config, metal_ctx)
        }
    }
}

fn benchmark_attention_backends(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    metal_ctx: &std::sync::Arc<MetalContext>,
) -> Result<(AttentionBackendChoice, Array), Exception> {
    let reference_start = Instant::now();
    let reference_output = reference_attention_output(queries, keys, values, config)?;
    let ref_eval = reference_output.clone();
    ref_eval.eval();
    let reference_elapsed = reference_start.elapsed();

    let mut best_backend = AttentionBackendChoice::MlxFast;
    let mut best_elapsed = reference_elapsed;
    let mut best_output = reference_output.clone();

    if let Some((elapsed, output)) = benchmark_attention_candidate(
        AttentionBackendChoice::MetalFlash,
        queries,
        keys,
        values,
        config,
        metal_ctx,
        &reference_output,
    )? {
        if elapsed < best_elapsed {
            best_backend = AttentionBackendChoice::MetalFlash;
            best_elapsed = elapsed;
            best_output = output;
        }
    }

    if mpp_flash_attention_supported(queries, keys, values, None, metal_ctx.dispatch()) {
        if let Some((elapsed, output)) = benchmark_attention_candidate(
            AttentionBackendChoice::MppFlash,
            queries,
            keys,
            values,
            config,
            metal_ctx,
            &reference_output,
        )? {
            if elapsed < best_elapsed {
                best_backend = AttentionBackendChoice::MppFlash;
                best_elapsed = elapsed;
                best_output = output;
            }
        }
    }

    tracing::debug!(
        "Selected {:?} attention backend ({:?})",
        best_backend,
        best_elapsed
    );
    Ok((best_backend, best_output))
}

fn benchmark_attention_candidate(
    backend: AttentionBackendChoice,
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    metal_ctx: &std::sync::Arc<MetalContext>,
    reference: &Array,
) -> Result<Option<(std::time::Duration, Array)>, Exception> {
    let start = Instant::now();
    let output = match execute_attention_backend(backend, queries, keys, values, config, metal_ctx)
    {
        Ok(output) => {
            let out_eval = output.clone();
            out_eval.eval();
            output
        }
        Err(error) => {
            tracing::debug!("{backend:?} attention benchmark failed: {error}");
            return Ok(None);
        }
    };
    let elapsed = start.elapsed();
    let max_diff = max_abs_diff(reference, &output)?;
    let reference_scale = max_abs(reference)?;
    if !candidate_matches_reference(max_diff, reference_scale) {
        tracing::debug!(
            "Rejecting {:?} attention backend: max_diff={:.5e} against a reference scale of {:.5e}",
            backend,
            max_diff,
            reference_scale
        );
        return Ok(None);
    }

    Ok(Some((elapsed, output)))
}

/// Whether a candidate backend's output is close enough to the reference's to
/// be worth timing.
///
/// Judged against the reference's own scale, not an absolute constant. The
/// Metal backends stage Q/K/V through f16, whose smallest normal is ~6e-5, and
/// not every model keeps its activations near 1: Nemotron-H's residual stream
/// sits around 1e-17, where the f16 path returns all zeros. Against a fixed
/// 0.1 that scored as a perfect match, so the zero-returning backend won on
/// speed and was cached, and every attention layer in the architecture
/// contributed nothing.
///
/// The ratio is the same 0.1 the absolute rule used, so a unit-scale tensor
/// accepts and rejects exactly as it did before.
fn candidate_matches_reference(max_diff: f32, reference_scale: f32) -> bool {
    max_diff <= 0.1 * reference_scale
}

fn reference_attention_output(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
) -> Result<Array, Exception> {
    if let Some(cap) = config.logit_softcapping {
        manual_sdpa_with_softcapping(queries, keys, values, config, None, cap)
    } else {
        fast_fused_sdpa(queries, keys, values, config, None)
    }
}

fn fast_fused_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
) -> Result<Array, Exception> {
    // Determine mask to use
    match (&config.mask_type, custom_mask) {
        // Custom mask provided - use it directly
        (_, Some(mask)) => Ok(queries.sdpa_with_mask(keys, values, config.scale, Some(mask))),

        // Causal masking - use MLX's built-in causal mask
        (AttentionMaskType::Causal, None) => Ok(queries.sdpa(keys, values, config.scale, "causal")),

        // No mask (bidirectional attention) — MLX accepts "" for no mask, not "none"
        (AttentionMaskType::None, None) => Ok(queries.sdpa(keys, values, config.scale, "")),

        // Sliding window - create custom mask
        (AttentionMaskType::SlidingWindow(window_size), None) => {
            let query_len = queries.dim(2);
            let key_len = keys.dim(2);
            // Build it in the query's dtype. MLX rejects a mask that does not
            // promote to the output type rather than upcasting, so the f32
            // builder threw on every bf16 model — and a thrown bridge op
            // returns a 0-dim array that broadcasts on through the layer
            // instead of failing. `fused_sdpa` coerces the *caller's* mask for
            // exactly this reason; a mask built in here needs the same care.
            let mask = create_sliding_window_mask(query_len, key_len, *window_size)?
                .as_dtype(queries.dtype().as_i32());
            Ok(queries.sdpa_with_mask(keys, values, config.scale, Some(&mask)))
        }
    }
}

fn try_metal_flash_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
) -> Result<Option<Array>, Exception> {
    if !flash_attention_supported(queries, keys, values, custom_mask) {
        return Ok(None);
    }

    let metal_ctx = match MetalContext::global() {
        Ok(ctx) => ctx,
        Err(error) => {
            tracing::debug!("Metal FlashAttention unavailable, falling back to MLX: {error}");
            return Ok(None);
        }
    };

    match run_metal_flash_attention(queries, keys, values, config, &metal_ctx) {
        Ok(output) => Ok(Some(output)),
        Err(error) => {
            tracing::debug!(
                "Metal FlashAttention unavailable, falling back to MLX/manual attention: {}",
                error
            );
            Ok(None)
        }
    }
}

fn flash_attention_supported(
    queries: &Array,
    keys: &Array,
    values: &Array,
    custom_mask: Option<&Array>,
) -> bool {
    if custom_mask.is_some() {
        return false;
    }

    if !matches!(
        queries.dtype(),
        Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16
    ) || queries.dtype() != keys.dtype()
        || queries.dtype() != values.dtype()
    {
        return false;
    }

    let q_shape = queries.shape();
    let k_shape = keys.shape();
    let v_shape = values.shape();

    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return false;
    }

    // Metal flash attention kernels tile over head_dim and assume K_dim == V_dim.
    // Asymmetric value dims (e.g. DeepSeek MLA) must fall through to MLX SDPA.
    if q_shape[3] != v_shape[3] {
        return false;
    }

    matches!(q_shape[3] as usize, 64 | 80 | 96 | 128 | 256)
}

/// Shape-only eligibility check for MPP flash attention (no hardware check).
///
/// Separated from [`mpp_flash_attention_supported`] so that unit tests can
/// exercise shape logic without needing a live [`KernelDispatch`].
fn mpp_flash_attention_shape_ok(
    queries: &Array,
    keys: &Array,
    values: &Array,
    custom_mask: Option<&Array>,
) -> bool {
    flash_attention_supported(queries, keys, values, custom_mask)
        && matches!(queries.shape()[3], 64 | 80 | 96 | 128)
}

/// Full eligibility check for MPP flash attention: hardware capability + shape.
///
/// Uses [`KernelDispatch::preferred_backend`] to determine whether the Metal 4
/// backend is available on this device, rather than calling `has_nax()` directly.
fn mpp_flash_attention_supported(
    queries: &Array,
    keys: &Array,
    values: &Array,
    custom_mask: Option<&Array>,
    dispatch: &KernelDispatch,
) -> bool {
    dispatch.preferred_backend().caps().has_mpp_flash_attention
        && mpp_flash_attention_shape_ok(queries, keys, values, custom_mask)
}

/// Largest magnitude in `array`, as f32.
fn max_abs(array: &Array) -> Result<f32, Exception> {
    let array = if array.dtype() == Dtype::Float32 {
        array.clone()
    } else {
        array.as_dtype(Dtype::Float32.as_i32())
    };
    let max = array.abs_val().max(None);
    max.eval();
    Ok(max.item_f32())
}

fn max_abs_diff(lhs: &Array, rhs: &Array) -> Result<f32, Exception> {
    let lhs = if lhs.dtype() == Dtype::Float32 {
        lhs.clone()
    } else {
        lhs.as_dtype(Dtype::Float32.as_i32())
    };
    let rhs = if rhs.dtype() == Dtype::Float32 {
        rhs.clone()
    } else {
        rhs.as_dtype(Dtype::Float32.as_i32())
    };

    let diff = lhs.subtract(&rhs).abs_val().max(None);
    let diff_owned = diff.clone();
    diff_owned.eval();
    Ok(diff_owned.item_f32())
}

fn run_metal_flash_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    metal_ctx: &std::sync::Arc<MetalContext>,
) -> Result<Array, Exception> {
    let q_shape = queries.shape();
    let k_shape = keys.shape();
    let v_shape = values.shape();
    let head_dim = q_shape[3] as usize;
    // Output uses value dimension (may differ from key dim for asymmetric archs)
    let out_head_dim = v_shape[3];
    let sliding_window = match config.mask_type {
        AttentionMaskType::SlidingWindow(window) => Some(window as usize),
        _ => None,
    };

    let fa_config = MetalFlashAttentionConfig {
        batch_size: q_shape[0] as usize,
        num_heads: q_shape[1] as usize,
        num_kv_heads: k_shape[1] as usize,
        query_seq_len: q_shape[2] as usize,
        kv_seq_len: k_shape[2] as usize,
        head_dim,
        scale: Some(config.scale),
        is_causal: !matches!(config.mask_type, AttentionMaskType::None),
        sliding_window,
        softcap: config.logit_softcapping,
        is_training: false,
    };

    let flash_attn = match FlashAttention::new(metal_ctx.clone(), fa_config) {
        Ok(flash_attn) => flash_attn,
        Err(error) => {
            return Err(Exception::custom(error.to_string()));
        }
    };

    let q_buffer = array_to_metal_buffer_f16(metal_ctx, queries)
        .map_err(|error| Exception::custom(error.to_string()))?;
    let k_buffer = array_to_metal_buffer_f16(metal_ctx, keys)
        .map_err(|error| Exception::custom(error.to_string()))?;
    let v_buffer = array_to_metal_buffer_f16(metal_ctx, values)
        .map_err(|error| Exception::custom(error.to_string()))?;

    let output = match flash_attn.forward(&q_buffer, &k_buffer, &v_buffer) {
        Ok(output) => output,
        Err(error) => {
            return Err(Exception::custom(error.to_string()));
        }
    };

    let out_shape = &[q_shape[0], q_shape[1], q_shape[2], out_head_dim];
    let output = metal_buffer_into_array_f16(output.output, out_shape)
        .map_err(|error| Exception::custom(error.to_string()))?;

    if queries.dtype() == Dtype::Float16 {
        Ok(output)
    } else {
        Ok(output.as_dtype(queries.dtype().as_i32()))
    }
}

fn run_mpp_flash_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    metal_ctx: &std::sync::Arc<MetalContext>,
) -> Result<Array, Exception> {
    if !mpp_flash_attention_supported(queries, keys, values, None, metal_ctx.dispatch()) {
        return Err(Exception::custom(
            "MPP FlashAttention unsupported for current device or shape".to_string(),
        ));
    }

    let q_shape = queries.shape();
    let k_shape = keys.shape();
    let v_shape = values.shape();
    let out_head_dim = v_shape[3];
    let mpp_config = MppFlashAttentionConfig {
        batch_size: q_shape[0] as usize,
        num_heads: q_shape[1] as usize,
        num_kv_heads: k_shape[1] as usize,
        query_seq_len: q_shape[2] as usize,
        kv_seq_len: k_shape[2] as usize,
        head_dim: q_shape[3] as usize,
        scale: Some(config.scale),
        is_causal: !matches!(config.mask_type, AttentionMaskType::None),
        sliding_window: match config.mask_type {
            AttentionMaskType::SlidingWindow(window) => Some(window as usize),
            _ => None,
        },
        softcap: config.logit_softcapping,
    };

    let flash_attn = MppFlashAttention::new(metal_ctx.clone(), mpp_config)
        .map_err(|error| Exception::custom(error.to_string()))?;
    if !flash_attn.is_available() {
        return Err(Exception::custom(
            "MPP FlashAttention unavailable on current device".to_string(),
        ));
    }

    let q_buffer = array_to_metal_buffer_f16(metal_ctx, queries)
        .map_err(|error| Exception::custom(error.to_string()))?;
    let k_buffer = array_to_metal_buffer_f16(metal_ctx, keys)
        .map_err(|error| Exception::custom(error.to_string()))?;
    let v_buffer = array_to_metal_buffer_f16(metal_ctx, values)
        .map_err(|error| Exception::custom(error.to_string()))?;

    let output = flash_attn
        .forward(&q_buffer, &k_buffer, &v_buffer)
        .map_err(|error| Exception::custom(error.to_string()))?;

    let out_shape = &[q_shape[0], q_shape[1], q_shape[2], out_head_dim];
    let output = metal_buffer_into_array_f16(output.output, out_shape)
        .map_err(|error| Exception::custom(error.to_string()))?;

    if queries.dtype() == Dtype::Float16 {
        Ok(output)
    } else {
        Ok(output.as_dtype(queries.dtype().as_i32()))
    }
}

/// Fused SDPA with attention logit softcapping (Gemma2 style).
///
/// Applies: scores = cap * tanh(scores / cap) before softmax
fn manual_sdpa_with_softcapping(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
    cap: f32,
) -> Result<Array, Exception> {
    // Unfortunately, MLX's fused SDPA doesn't support logit softcapping.
    // We need to manually compute attention with softcapping.
    // Still benefit from proper GQA handling.

    let shape = queries.shape();
    let batch = shape[0];
    let n_heads = shape[1];
    let q_seq_len = shape[2];

    let k_shape = keys.shape();
    let n_kv_heads = k_shape[1];
    let kv_seq_len = k_shape[2];

    // Expand K/V for GQA if needed
    let (keys, values) = if n_kv_heads < n_heads {
        let repeats = n_heads / n_kv_heads;
        (
            expand_kv_heads(keys, repeats)?,
            expand_kv_heads(values, repeats)?,
        )
    } else {
        (keys.clone(), values.clone())
    };

    // Q @ K.T
    let keys_t = keys.transpose_axes(&[0, 1, 3, 2]);
    let scores = queries.matmul(&keys_t);

    // Scale
    let scale_arr = Array::from_f32(config.scale);
    let scores = scores.multiply(&scale_arr);

    // Apply softcapping: cap * tanh(scores / cap).
    //
    // Uses the `tanh` primitive rather than expanding it as
    // `(exp(2x) - 1) / (exp(2x) + 1)`, which overflows to `inf/inf = NaN` once
    // `scores / cap` exceeds ~44 — the same defect that made the Gemma GELU
    // produce NaN. Softcapping exists precisely to tame outlier logits, so it
    // must not itself blow up on them.
    let cap_arr = Array::from_f32(cap);
    let scores = pmetal_bridge::compat::ops::tanh(&scores.divide(&cap_arr));
    let scores = scores.multiply(&cap_arr);

    // Apply mask
    let scores = match (&config.mask_type, custom_mask) {
        (_, Some(mask)) => scores.add(mask),
        (AttentionMaskType::Causal, None) => {
            let mask = create_causal_mask(q_seq_len, kv_seq_len)?;
            scores.add(&mask)
        }
        (AttentionMaskType::SlidingWindow(window_size), None) => {
            let mask = create_sliding_window_mask(q_seq_len, kv_seq_len, *window_size)?;
            scores.add(&mask)
        }
        (AttentionMaskType::None, None) => scores,
    };

    // Softmax
    let weights = scores.softmax(-1);

    // Attention output: weights @ V
    let output = weights.matmul(&values);

    // Verify output shape — output uses value dimension, not key dimension
    let v_head_dim = values.dim(3);
    debug_assert_eq!(output.shape(), &[batch, n_heads, q_seq_len, v_head_dim]);

    Ok(output)
}

/// Scaled dot-product attention with **attention sinks** (GPT-OSS).
///
/// MLX's fused kernel (and this crate's bridge SDPA) have no `sinks` parameter,
/// so this mirrors `mx.fast.scaled_dot_product_attention(..., sinks=sinks)`
/// eagerly: one learned per-head logit joins the softmax denominator as a
/// virtual key that contributes nothing to the output. Concretely, for each
/// (batch, head, query) row the softmax is taken over `[scores, sink_h]` and
/// the sink column is then dropped from the weighted sum:
///
/// ```text
///   scores  = scale · (Q · Kᵀ) + mask          # [B, H, L, S]
///   p       = softmax([scores, sink_h], -1)     # [B, H, L, S+1]
///   out     = p[..., :S] @ V
/// ```
///
/// `sinks` is a 1-D `[n_heads]` array. GQA K/V are expanded internally, exactly
/// like [`manual_sdpa_with_softcapping`]. Logit softcapping is intentionally
/// unsupported here — GPT-OSS uses sinks, not softcap.
pub fn sink_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    custom_mask: Option<&Array>,
    sinks: &Array,
) -> Result<Array, Exception> {
    let shape = queries.shape();
    let batch = shape[0];
    let n_heads = shape[1];
    let q_seq_len = shape[2];

    let k_shape = keys.shape();
    let n_kv_heads = k_shape[1];
    let kv_seq_len = k_shape[2];

    // Expand K/V for GQA if needed (sink softmax is per-expanded-head).
    let (keys, values) = if n_kv_heads < n_heads {
        let repeats = n_heads / n_kv_heads;
        (
            expand_kv_heads(keys, repeats)?,
            expand_kv_heads(values, repeats)?,
        )
    } else {
        (keys.clone(), values.clone())
    };

    // scale · (Q · Kᵀ)
    let keys_t = keys.transpose_axes(&[0, 1, 3, 2]);
    let scale_arr = Array::from_f32(config.scale);
    let scores = queries.matmul(&keys_t).multiply(&scale_arr);

    // Additive mask (matches the fused-kernel mask semantics).
    let scores = match (&config.mask_type, custom_mask) {
        (_, Some(mask)) => scores.add(mask),
        (AttentionMaskType::Causal, None) => {
            scores.add(&create_causal_mask(q_seq_len, kv_seq_len)?)
        }
        (AttentionMaskType::SlidingWindow(window_size), None) => scores.add(
            &create_sliding_window_mask(q_seq_len, kv_seq_len, *window_size)?,
        ),
        (AttentionMaskType::None, None) => scores,
    };

    // Sink column: per-head logit broadcast to [B, H, L, 1], concatenated as the
    // (S+1)-th key, softmaxed over the full row, then dropped from the sum.
    let sink_col = sinks
        .reshape(&[1, n_heads, 1, 1])
        .broadcast_to(&[batch, n_heads, q_seq_len, 1]);
    let combined = ops::concatenate_axis(&[&scores, &sink_col], -1);
    let weights = combined.softmax(-1);
    let weights = ops::slice_axis(&weights, -1, 0, kv_seq_len);

    let output = weights.matmul(&values);

    let v_head_dim = values.dim(3);
    debug_assert_eq!(output.shape(), &[batch, n_heads, q_seq_len, v_head_dim]);
    Ok(output)
}

/// Expand K/V heads for grouped query attention.
///
/// [batch, n_kv_heads, seq_len, head_dim] -> [batch, n_heads, seq_len, head_dim]
fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // [B, kv_heads, L, head_dim] -> [B, kv_heads, 1, L, head_dim]
    let x = x.reshape(&[batch, n_kv_heads, 1, seq_len, head_dim]);
    // Broadcast to [B, kv_heads, repeats, L, head_dim]
    let x = x.broadcast_to(&[batch, n_kv_heads, repeats, seq_len, head_dim]);
    // Reshape to [B, n_heads, L, head_dim]
    Ok(x.reshape(&[batch, n_kv_heads * repeats, seq_len, head_dim]))
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

/// Memory-efficient attention for long sequences.
///
/// Uses chunked computation to reduce peak memory usage for very long sequences.
/// Falls back to standard fused SDPA for short sequences.
///
/// # Arguments
/// * `queries` - Query tensor [batch, n_heads, seq_len, head_dim]
/// * `keys` - Key tensor [batch, n_kv_heads, seq_len, head_dim]
/// * `values` - Value tensor [batch, n_kv_heads, seq_len, head_dim]
/// * `config` - Attention configuration
/// * `chunk_size` - Maximum sequence length per chunk (for queries)
///
/// # Note
/// Chunking is applied to queries only. Full K/V context is maintained.
pub fn memory_efficient_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
    chunk_size: i32,
) -> Result<Array, Exception> {
    let q_seq_len = queries.dim(2);

    // Short sequence - use standard fused SDPA
    if q_seq_len <= chunk_size {
        return fused_sdpa(queries, keys, values, config, None);
    }

    // Long sequence - chunk the queries
    let kv_seq_len = keys.dim(2);

    let mut outputs = Vec::new();
    let mut start = 0;

    while start < q_seq_len {
        let end = (start + chunk_size).min(q_seq_len);
        let chunk_len = end - start;

        // Extract query chunk using slice
        // queries shape: [batch, n_heads, seq_len, head_dim]
        let batch = queries.dim(0);
        let n_h = queries.dim(1);
        let hd = queries.dim(3);
        let q_chunk = queries.slice(&[0, 0, start, 0], &[batch, n_h, end, hd]);

        // Create appropriate mask for this chunk
        // Chunk queries can attend to all keys up to their position
        let mask = if config.mask_type == AttentionMaskType::Causal {
            // Causal: can attend to positions [0, start + chunk_pos]
            Some(create_chunk_causal_mask(chunk_len, kv_seq_len, start)?)
        } else {
            None
        };

        // Compute attention for chunk
        let chunk_output = fused_sdpa(&q_chunk, keys, values, config, mask.as_ref())?;
        outputs.push(chunk_output);

        start = end;
    }

    // Concatenate outputs along sequence dimension
    let outputs_refs: Vec<&Array> = outputs.iter().collect();
    Ok(ops::concatenate_axis(&outputs_refs, 2))
}

/// Create causal mask for a query chunk.
///
/// For queries at positions [start, start + chunk_len), create a mask where
/// position i (relative in chunk) can attend to keys at positions [0, start + i].
fn create_chunk_causal_mask(
    chunk_len: i32,
    key_len: i32,
    start_pos: i32,
) -> Result<Array, Exception> {
    // Create base causal mask
    // Each query position can attend to: all keys up to (start_pos + local_pos)

    let mut mask_data = Vec::with_capacity((chunk_len * key_len) as usize);

    for q_pos in 0..chunk_len {
        let global_q_pos = start_pos + q_pos;
        for k_pos in 0..key_len {
            if k_pos <= global_q_pos {
                mask_data.push(0.0f32);
            } else {
                mask_data.push(f32::NEG_INFINITY);
            }
        }
    }

    let mask = Array::from_f32_slice(&mask_data, &[chunk_len, key_len]);
    Ok(mask.reshape(&[1, 1, chunk_len, key_len]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_tensor(shape: &[i32]) -> Array {
        random::normal(shape, Dtype::Float32)
    }

    fn test_device_properties() -> DeviceProperties {
        DeviceProperties {
            name: "Apple M5 Test".to_string(),
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory_length: 32 * 1024,
            has_unified_memory: true,
            recommended_working_set_size: 8 * 1024 * 1024 * 1024,
            max_buffer_length: 256 * 1024 * 1024,
            gpu_family: pmetal_metal::context::AppleGPUFamily::Apple10,
            device_tier: DeviceTier::Max,
            has_dynamic_caching: true,
            has_hardware_ray_tracing: true,
            has_mesh_shaders: true,
            has_nax: true,
            architecture_gen: 17,
            memory_bandwidth_gbps: 546.0,
            memory_bandwidth_source:
                pmetal_metal::context::MemoryBandwidthSource::SpecTableFallback,
            gpu_core_count: 40,
            ane_core_count: 16,
            is_ultra_fusion: false,
            die_count: 1,
        }
    }

    /// An f32 additive mask against a bf16 model must work.
    ///
    /// MLX's SDPA refuses a mask that doesn't promote to the output dtype — it
    /// errors rather than upcasting — and every mask builder in `pmetal-models`
    /// emits f32 regardless of the checkpoint's dtype. Before `fused_sdpa`
    /// coerced the mask, this combination threw on *every* bf16 checkpoint
    /// taking the uncached prefill path (the one `DynamicModel::forward` uses),
    /// and the thrown op returned a placeholder that then broadcast harmlessly
    /// through the residual stream — so the output kept its shape and stayed
    /// finite while attention had contributed nothing.
    #[test]
    fn an_f32_mask_works_against_a_bf16_model() {
        let (heads, seq, head_dim) = (2, 4, 8);
        let shape = [1, heads, seq, head_dim];
        let bf16 = |s: &[i32]| random::normal(s, Dtype::Float32).as_dtype(Dtype::Bfloat16.as_i32());
        let (q, k, v) = (bf16(&shape), bf16(&shape), bf16(&shape));

        // Additive causal mask in f32, exactly as `create_causal_mask` builds it.
        let mask = {
            let lower = ops::tri(seq, seq, 0, Dtype::Float32);
            ops::where_fn(
                &lower.equal(&Array::from_f32(0.0)),
                &Array::from_f32(f32::NEG_INFINITY),
                &Array::from_f32(0.0),
            )
        };
        assert_eq!(mask.dtype(), Dtype::Float32);

        let config = FusedAttentionConfig::new(heads, heads, head_dim)
            .with_mask_type(AttentionMaskType::None);
        let mut out = fused_sdpa(&q, &k, &v, &config, Some(&mask)).expect("fused_sdpa runs");
        out.eval();

        pmetal_bridge::check_last_error().expect("no bridge exception with an f32 mask on bf16");
        assert_eq!(out.shape(), &shape);
        let values = out.to_f32_vec(out.size()).expect("output readable");
        assert!(
            values.iter().all(|x| x.is_finite()),
            "attention output is not finite"
        );
        assert!(
            values.iter().any(|&x| x != 0.0),
            "attention output is all zeros — the masked path produced nothing"
        );
    }

    /// `AttentionMaskType::SlidingWindow` must work against a bf16 model too.
    ///
    /// The sibling test above covers the mask a *caller* supplies, which
    /// `fused_sdpa` coerces. This covers the one the kernel builds for itself,
    /// which it did not: an f32 mask reached `sdpa_with_mask` and threw on every
    /// bf16 checkpoint, so no windowed layer of any Gemma / Phi / Qwen / Mistral
    /// model ever ran its window. Nothing caught it because the throw returns a
    /// placeholder that broadcasts — and because architectures papered over the
    /// window by handing the whole stack one causal mask instead.
    #[test]
    fn a_sliding_window_runs_against_a_bf16_model() {
        let (heads, seq, head_dim) = (2, 8, 8);
        let window = 3;
        let shape = [1, heads, seq, head_dim];
        let f32_qkv: Vec<Array> = (0..3).map(|_| random_tensor(&shape)).collect();
        let to_bf16 = |a: &Array| a.as_dtype(Dtype::Bfloat16.as_i32());

        let config = FusedAttentionConfig::new(heads, heads, head_dim)
            .with_mask_type(AttentionMaskType::SlidingWindow(window));

        let bf16_out = fused_sdpa(
            &to_bf16(&f32_qkv[0]),
            &to_bf16(&f32_qkv[1]),
            &to_bf16(&f32_qkv[2]),
            &config,
            None,
        )
        .expect("fused_sdpa runs");
        bf16_out.eval();
        pmetal_bridge::check_last_error()
            .expect("no bridge exception from a self-built sliding-window mask on bf16");
        assert_eq!(bf16_out.shape(), &shape);

        // The window has to actually be honoured, not just not-throw: compare
        // against the same attention in f32, where the mask dtype already
        // matched and the path was known good.
        let mut f32_out = fused_sdpa(&f32_qkv[0], &f32_qkv[1], &f32_qkv[2], &config, None)
            .expect("f32 reference");
        f32_out.eval();
        let got = bf16_out
            .as_dtype(Dtype::Float32.as_i32())
            .to_f32_vec(bf16_out.size())
            .expect("bf16 output readable");
        let want = f32_out
            .to_f32_vec(f32_out.size())
            .expect("f32 output readable");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 0.05, "bf16 windowed attention diverges by {worst}");

        // ...and the window must differ from plain causal, or the comparison
        // above would pass on an implementation that silently dropped it.
        let causal = config.with_mask_type(AttentionMaskType::Causal);
        let mut causal_out =
            fused_sdpa(&f32_qkv[0], &f32_qkv[1], &f32_qkv[2], &causal, None).expect("causal");
        causal_out.eval();
        let causal_values = causal_out
            .to_f32_vec(causal_out.size())
            .expect("causal output readable");
        assert!(
            want.iter()
                .zip(&causal_values)
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "a {window}-token window over {seq} tokens produced plain causal attention"
        );
    }

    /// Backend selection must judge accuracy against the tensor's own scale.
    ///
    /// The Metal backends stage Q/K/V through f16, whose smallest normal is
    /// ~6e-5. A model whose activations sit far below that gets all zeros back,
    /// and the old absolute `max_diff > 0.1` rule scored those zeros as a
    /// perfect match — so the zero-returning backend won on speed and was
    /// cached. Nemotron-H runs at ~1e-17 and lost every attention layer to it,
    /// which showed up only as a real-weight forward divergence.
    #[test]
    fn a_candidate_is_judged_against_the_reference_scale() {
        // The case that broke: candidate returned all zeros, reference is
        // ~1e-17. `max_diff` is 1e-17, which clears any absolute 0.1.
        assert!(
            !candidate_matches_reference(1e-17, 1e-17),
            "a candidate that zeroed the whole tensor was accepted"
        );

        // Unit-scale behaviour is unchanged: the ratio is the same 0.1 the
        // absolute rule used when the reference happened to be near 1.
        assert!(candidate_matches_reference(0.05, 1.0));
        assert!(!candidate_matches_reference(0.15, 1.0));

        // f16 staging costs ~1e-3 relative; that has to keep passing at any
        // scale, or every model below unit scale loses its fast backends.
        assert!(candidate_matches_reference(1e-20, 1e-17));
        assert!(candidate_matches_reference(1e3, 1e6));

        // A genuinely all-zero reference admits only an all-zero candidate.
        assert!(candidate_matches_reference(0.0, 0.0));
        assert!(!candidate_matches_reference(1e-30, 0.0));
    }

    #[test]
    fn sink_sdpa_matches_hand_computed_softmax() {
        // 1 batch, 1 head, 1 query, 2 keys, head_dim 1, scale 1.0, no mask.
        // scores = q·kᵀ = [2, 3]; sink logit c = 0.5. The sink joins the
        // softmax denominator as a valueless key, so:
        //   Z   = e² + e³ + e^0.5
        //   w   = [e²/Z, e³/Z]              (sink weight dropped)
        //   out = 10·w0 + 20·w1
        let q = Array::from_slice(&[1.0_f32], &[1, 1, 1, 1]);
        let k = Array::from_slice(&[2.0_f32, 3.0], &[1, 1, 2, 1]);
        let v = Array::from_slice(&[10.0_f32, 20.0], &[1, 1, 2, 1]);
        let sinks = Array::from_slice(&[0.5_f32], &[1]);

        let cfg = FusedAttentionConfig::new(1, 1, 1)
            .with_scale(1.0)
            .with_mask_type(AttentionMaskType::None);
        let mut out = sink_sdpa(&q, &k, &v, &cfg, None, &sinks).expect("sink_sdpa");
        let got = out.to_f32_vec(1).expect("vec")[0];

        let (e2, e3, ec) = (2.0_f64.exp(), 3.0_f64.exp(), 0.5_f64.exp());
        let z = e2 + e3 + ec;
        let expected = (10.0 * e2 / z + 20.0 * e3 / z) as f32;
        assert!(
            (got - expected).abs() < 1e-4,
            "sink_sdpa out {got} != hand-computed {expected}"
        );
    }

    #[test]
    fn sink_sdpa_zero_sink_still_shifts_denominator() {
        // A zero sink is NOT a no-op: it adds e^0 = 1 to the denominator, so
        // the output must differ from plain (sink-free) attention.
        let q = Array::from_slice(&[1.0_f32], &[1, 1, 1, 1]);
        let k = Array::from_slice(&[2.0_f32, 3.0], &[1, 1, 2, 1]);
        let v = Array::from_slice(&[10.0_f32, 20.0], &[1, 1, 2, 1]);
        let cfg = FusedAttentionConfig::new(1, 1, 1)
            .with_scale(1.0)
            .with_mask_type(AttentionMaskType::None);

        let zeros = Array::from_slice(&[0.0_f32], &[1]);
        let mut with_sink = sink_sdpa(&q, &k, &v, &cfg, None, &zeros).expect("sink");
        let s = with_sink.to_f32_vec(1).expect("vec")[0];

        // Plain softmax over [2,3]: out = (10·e² + 20·e³)/(e²+e³).
        let (e2, e3) = (2.0_f64.exp(), 3.0_f64.exp());
        let plain = ((10.0 * e2 + 20.0 * e3) / (e2 + e3)) as f32;
        let with_one = ((10.0 * e2 + 20.0 * e3) / (e2 + e3 + 1.0)) as f32;
        assert!(
            (s - with_one).abs() < 1e-4,
            "zero-sink out {s} != {with_one}"
        );
        assert!(
            (s - plain).abs() > 1e-3,
            "zero sink should differ from plain softmax"
        );
    }

    #[test]
    fn test_fused_sdpa_basic() {
        let batch = 2;
        let n_heads = 4;
        let seq_len = 8;
        let head_dim = 32;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_fused_sdpa_gqa() {
        let batch = 2;
        let n_heads = 8;
        let n_kv_heads = 2; // GQA with 4 groups
        let seq_len = 8;
        let head_dim = 32;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_fused_sdpa_mqa() {
        let batch = 2;
        let n_heads = 8;
        let n_kv_heads = 1; // MQA
        let seq_len = 8;
        let head_dim = 32;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_fused_sdpa_no_mask() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 8;
        let head_dim = 32;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim)
            .with_mask_type(AttentionMaskType::None);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_fused_sdpa_sliding_window() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 16;
        let head_dim = 32;
        let window_size = 4;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim)
            .with_mask_type(AttentionMaskType::SlidingWindow(window_size));
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_fused_sdpa_softcapping() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 8;
        let head_dim = 64;
        let softcap = 50.0;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config =
            FusedAttentionConfig::new(n_heads, n_heads, head_dim).with_logit_softcapping(softcap);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_fused_sdpa_metal_matches_fast_reference() {
        let batch = 1;
        let n_heads = 4;
        let n_kv_heads = 2;
        let seq_len = 8;
        let head_dim = 64;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();
        let reference = fast_fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let output = output.as_dtype(Dtype::Float32.as_i32());
        let reference = reference.as_dtype(Dtype::Float32.as_i32());
        let _ = &output; // eval not needed
        let _ = &reference; // eval not needed

        let mut out_m = output.clone();
        out_m.eval();
        let mut ref_m = reference.clone();
        ref_m.eval();
        let out_n = out_m.size();
        let ref_n = ref_m.size();
        let out_data = out_m.to_f32_vec(out_n).unwrap_or_default();
        let ref_data = ref_m.to_f32_vec(ref_n).unwrap_or_default();
        for (actual, expected) in out_data.iter().zip(ref_data.iter()) {
            assert!(
                (actual - expected).abs() < 0.1,
                "Metal attention diverged from MLX fast path: actual={}, expected={}",
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_fused_sdpa_softcapping_matches_manual_reference() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 8;
        let head_dim = 64;
        let softcap = 50.0;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config =
            FusedAttentionConfig::new(n_heads, n_heads, head_dim).with_logit_softcapping(softcap);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();
        let reference =
            manual_sdpa_with_softcapping(&queries, &keys, &values, &config, None, softcap).unwrap();

        let output = output.as_dtype(Dtype::Float32.as_i32());
        let reference = reference.as_dtype(Dtype::Float32.as_i32());
        let _ = &output; // eval not needed
        let _ = &reference; // eval not needed

        let mut out_s = output.clone();
        out_s.eval();
        let mut ref_s = reference.clone();
        ref_s.eval();
        let out_n2 = out_s.size();
        let ref_n2 = ref_s.size();
        let out_data2 = out_s.to_f32_vec(out_n2).unwrap_or_default();
        let ref_data2 = ref_s.to_f32_vec(ref_n2).unwrap_or_default();
        for (actual, expected) in out_data2.iter().zip(ref_data2.iter()) {
            assert!(
                (actual - expected).abs() < 0.1,
                "Metal softcap attention diverged from manual reference: actual={}, expected={}",
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_fused_sdpa_preserves_input_dtype() {
        let batch = 1;
        let n_heads = 2;
        let seq_len = 4;
        let head_dim = 64;

        let queries =
            random_tensor(&[batch, n_heads, seq_len, head_dim]).as_dtype(Dtype::Float32.as_i32());
        let keys =
            random_tensor(&[batch, n_heads, seq_len, head_dim]).as_dtype(Dtype::Float32.as_i32());
        let values =
            random_tensor(&[batch, n_heads, seq_len, head_dim]).as_dtype(Dtype::Float32.as_i32());

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        assert_eq!(output.dtype(), Dtype::Float32);
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_causal_mask_creation() {
        let mask = create_causal_mask(4, 4).unwrap();
        let _ = &mask; // eval not needed

        assert_eq!(mask.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_sliding_window_mask() {
        let mask = create_sliding_window_mask(8, 8, 3).unwrap();
        let _ = &mask; // eval not needed

        assert_eq!(mask.shape(), &[1, 1, 8, 8]);
    }

    #[test]
    fn test_sliding_window_mask_generation_alignment() {
        let mask = create_sliding_window_mask(1, 6, 3).unwrap();
        let _ = &mask; // eval not needed

        assert_eq!(mask.shape(), &[1, 1, 1, 6]);
        let observed = {
            let mut m_own = mask.clone();
            m_own.eval();
            m_own.to_f32_vec(m_own.size()).unwrap_or_default()
        };
        assert_eq!(
            observed.as_slice(),
            &[
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0
            ]
        );
    }

    #[test]
    fn test_memory_efficient_attention_short() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 8;
        let head_dim = 32;
        let chunk_size = 16; // Larger than seq_len, so no chunking

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim);
        let output =
            memory_efficient_attention(&queries, &keys, &values, &config, chunk_size).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_memory_efficient_attention_chunked() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 32;
        let head_dim = 32;
        let chunk_size = 8; // Will create 4 chunks

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim);
        let output =
            memory_efficient_attention(&queries, &keys, &values, &config, chunk_size).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_custom_scale() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 8;
        let head_dim = 64;
        let custom_scale = 0.1; // Different from 1/sqrt(64) = 0.125

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim).with_scale(custom_scale);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_single_token_generation() {
        // This is the optimized path - query_len = 1 triggers Metal kernel
        let batch = 1;
        let n_heads = 4;
        let q_seq_len = 1; // Single token query
        let kv_seq_len = 32; // Cached keys/values
        let head_dim = 64;

        let queries = random_tensor(&[batch, n_heads, q_seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_heads, kv_seq_len, head_dim]);
        let values = random_tensor(&[batch, n_heads, kv_seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, head_dim);
        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();

        let _ = &output; // eval not needed
        assert_eq!(output.shape(), &[batch, n_heads, q_seq_len, head_dim]);
    }

    #[test]
    #[serial_test::serial]
    fn test_attention_backend_cache_roundtrip() {
        clear_cached_attention_backends();

        let key = AttentionDispatchKey {
            device_name: "Apple M5 Max".to_string(),
            device_tier: "max",
            dtype: "f16",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 2,
            query_seq_len: 16,
            kv_seq_len: 16,
            head_dim: 64,
            value_head_dim: 64,
            mask_type: AttentionMaskType::Causal,
            softcap_bits: None,
        };

        assert_eq!(cached_attention_backend(&key), None);
        cache_attention_backend(key.clone(), AttentionBackendChoice::MppFlash);
        assert_eq!(
            cached_attention_backend(&key),
            Some(AttentionBackendChoice::MppFlash)
        );

        clear_cached_attention_backends();
    }

    #[test]
    fn test_benchmarkable_attention_dispatch_key_accepts_softcapping_and_tracks_cap() {
        let props = test_device_properties();
        let queries = random_tensor(&[1, 4, 8, 64]);
        let keys = random_tensor(&[1, 4, 8, 64]);
        let values = random_tensor(&[1, 4, 8, 64]);
        let config = FusedAttentionConfig::new(4, 4, 64).with_logit_softcapping(30.0);
        let other_config = FusedAttentionConfig::new(4, 4, 64).with_logit_softcapping(50.0);

        let key =
            benchmarkable_attention_dispatch_key(&queries, &keys, &values, &config, None, &props)
                .expect("softcapped inference shapes should remain benchmarkable");
        let other_key = benchmarkable_attention_dispatch_key(
            &queries,
            &keys,
            &values,
            &other_config,
            None,
            &props,
        )
        .expect("softcapped inference shapes should remain benchmarkable");

        assert_eq!(key.softcap_bits, Some(30.0f32.to_bits()));
        assert_eq!(other_key.softcap_bits, Some(50.0f32.to_bits()));
        assert_ne!(key.cache_key(), other_key.cache_key());
    }

    #[test]
    fn test_reference_attention_output_matches_manual_softcapping() {
        let queries = random_tensor(&[1, 4, 8, 64]);
        let keys = random_tensor(&[1, 4, 8, 64]);
        let values = random_tensor(&[1, 4, 8, 64]);
        let config = FusedAttentionConfig::new(4, 4, 64).with_logit_softcapping(30.0);

        let reference = reference_attention_output(&queries, &keys, &values, &config).unwrap();
        let manual =
            manual_sdpa_with_softcapping(&queries, &keys, &values, &config, None, 30.0).unwrap();

        let max_diff = max_abs_diff(&reference, &manual).unwrap();
        assert!(max_diff <= 1e-3, "softcap reference drifted: {max_diff}");
    }

    #[test]
    fn test_mpp_flash_attention_shape_ok_accepted_and_rejected_head_dims() {
        // Hardware capability gating (has_nax / preferred_backend) is exercised
        // by integration tests that run on a live MetalContext.  Unit tests here
        // verify only the shape predicate via `mpp_flash_attention_shape_ok`.

        let queries = random_tensor(&[1, 4, 8, 128]);
        let keys = random_tensor(&[1, 4, 8, 128]);
        let values = random_tensor(&[1, 4, 8, 128]);
        assert!(mpp_flash_attention_shape_ok(&queries, &keys, &values, None));

        let queries_d64 = random_tensor(&[1, 4, 8, 64]);
        let keys_d64 = random_tensor(&[1, 4, 8, 64]);
        let values_d64 = random_tensor(&[1, 4, 8, 64]);
        assert!(mpp_flash_attention_shape_ok(
            &queries_d64,
            &keys_d64,
            &values_d64,
            None
        ));

        let queries_d96 = random_tensor(&[1, 4, 8, 96]);
        let keys_d96 = random_tensor(&[1, 4, 8, 96]);
        let values_d96 = random_tensor(&[1, 4, 8, 96]);
        assert!(mpp_flash_attention_shape_ok(
            &queries_d96,
            &keys_d96,
            &values_d96,
            None
        ));

        let queries_d80 = random_tensor(&[1, 4, 8, 80]);
        let keys_d80 = random_tensor(&[1, 4, 8, 80]);
        let values_d80 = random_tensor(&[1, 4, 8, 80]);
        assert!(mpp_flash_attention_shape_ok(
            &queries_d80,
            &keys_d80,
            &values_d80,
            None
        ));

        // head_dim=72 is not in the allowed set; must be rejected.
        let queries_d72 = random_tensor(&[1, 4, 8, 72]);
        let keys_d72 = random_tensor(&[1, 4, 8, 72]);
        let values_d72 = random_tensor(&[1, 4, 8, 72]);
        assert!(!mpp_flash_attention_shape_ok(
            &queries_d72,
            &keys_d72,
            &values_d72,
            None
        ));
    }

    #[test]
    fn test_fused_sdpa_asymmetric_value_head_dim() {
        // DeepSeek-style: key_dim=128, value_dim=64
        let batch = 1;
        let n_heads = 4;
        let n_kv_heads = 2;
        let seq_len = 8;
        let key_dim = 128;
        let value_dim = 64;

        let queries = random_tensor(&[batch, n_heads, seq_len, key_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, key_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, value_dim]);

        let config =
            FusedAttentionConfig::new(n_heads, n_kv_heads, key_dim).with_value_head_dim(value_dim);

        assert!(config.is_asymmetric());
        assert_eq!(config.effective_value_head_dim(), value_dim);

        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();
        let _ = &output; // eval not needed

        // Output should use VALUE dimension, not key dimension
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, value_dim]);
    }

    #[test]
    fn test_fused_sdpa_asymmetric_single_token_decode() {
        // Single-token decode with asymmetric dims (the hot path for inference)
        let batch = 1;
        let n_heads = 8;
        let n_kv_heads = 2;
        let q_seq_len = 1;
        let kv_seq_len = 32;
        let key_dim = 128;
        let value_dim = 64;

        let queries = random_tensor(&[batch, n_heads, q_seq_len, key_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, kv_seq_len, key_dim]);
        let values = random_tensor(&[batch, n_kv_heads, kv_seq_len, value_dim]);

        let config =
            FusedAttentionConfig::new(n_heads, n_kv_heads, key_dim).with_value_head_dim(value_dim);

        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();
        let _ = &output; // eval not needed

        assert_eq!(output.shape(), &[batch, n_heads, q_seq_len, value_dim]);
    }

    #[test]
    fn test_flash_attention_rejects_asymmetric_dims() {
        let key_dim = 128;
        let value_dim = 64;
        let queries = random_tensor(&[1, 4, 8, key_dim]);
        let keys = random_tensor(&[1, 4, 8, key_dim]);
        let values = random_tensor(&[1, 4, 8, value_dim]);

        // Metal flash attention should reject asymmetric dims
        assert!(!flash_attention_supported(&queries, &keys, &values, None));

        // Symmetric dims should pass (assuming head_dim is supported)
        let values_sym = random_tensor(&[1, 4, 8, key_dim]);
        assert!(flash_attention_supported(
            &queries,
            &keys,
            &values_sym,
            None
        ));
    }

    #[test]
    fn test_asymmetric_softcapping_output_shape() {
        let batch = 1;
        let n_heads = 4;
        let seq_len = 8;
        let key_dim = 64;
        let value_dim = 32;

        let queries = random_tensor(&[batch, n_heads, seq_len, key_dim]);
        let keys = random_tensor(&[batch, n_heads, seq_len, key_dim]);
        let values = random_tensor(&[batch, n_heads, seq_len, value_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_heads, key_dim)
            .with_value_head_dim(value_dim)
            .with_logit_softcapping(30.0);

        let output = fused_sdpa(&queries, &keys, &values, &config, None).unwrap();
        let _ = &output; // eval not needed

        assert_eq!(output.shape(), &[batch, n_heads, seq_len, value_dim]);
    }

    #[test]
    fn test_config_symmetric_by_default() {
        let config = FusedAttentionConfig::new(4, 2, 64);
        assert!(!config.is_asymmetric());
        assert_eq!(config.effective_value_head_dim(), 64);
    }
}
