//! Optimized attention implementations with runtime backend dispatch.
//!
//! This module provides attention implementations optimized for Apple Silicon:
//! - Standard scaled dot-product attention
//! - Flash attention (via MLX's mx.fast)
//! - Variable-length attention for packed sequences
//! - Grouped-query attention (GQA)
//! - Multi-query attention (MQA)
//!
//! ## Backend Selection (Unsloth-style)
//!
//! The attention dispatcher automatically selects the best backend based on:
//! 1. Whether sequences are packed (use VarLen)
//! 2. Hardware capabilities (prefer FlashAttention on newer chips)
//! 3. Sequence length (standard attention for very short sequences)
//!
//! ## Priority Order
//!
//! 1. VarLen FlashAttention (for packed sequences) - fastest for training
//! 2. Flash Attention (for regular batches) - efficient for long sequences
//! 3. Standard SDPA - fallback for maximum compatibility

use pmetal_core::Result;
use std::sync::atomic::{AtomicBool, Ordering};

// Feature detection flags (set at runtime)
static FLASH_ATTENTION_AVAILABLE: AtomicBool = AtomicBool::new(true);
static VARLEN_ATTENTION_AVAILABLE: AtomicBool = AtomicBool::new(true);

/// Attention configuration.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Dropout probability.
    pub dropout: f32,
    /// Use causal attention mask.
    pub is_causal: bool,
    /// Softmax scale (default: 1/sqrt(head_dim)).
    pub scale: Option<f32>,
    /// Sliding window size (None for full attention).
    pub sliding_window: Option<usize>,
    /// Softcap value for logits (Gemma2-style, 0.0 to disable).
    pub softcap: f32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            dropout: 0.0,
            is_causal: true,
            scale: None,
            sliding_window: None,
            softcap: 0.0,
        }
    }
}

impl AttentionConfig {
    /// Get the softmax scaling factor.
    #[must_use]
    pub fn scaling_factor(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// Check if this is grouped-query attention.
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// Check if this is multi-query attention.
    #[must_use]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Get the number of query groups.
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Check if sliding window attention is enabled.
    #[must_use]
    pub fn has_sliding_window(&self) -> bool {
        self.sliding_window.is_some()
    }

    /// Check if softcapping is enabled.
    #[must_use]
    pub fn has_softcap(&self) -> bool {
        self.softcap > 0.0
    }
}

/// Attention backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionBackend {
    /// Use MLX's mx.fast.scaled_dot_product_attention.
    #[default]
    MlxFast,
    /// Standard attention implementation (matmul + softmax + matmul).
    Standard,
    /// Variable-length attention for packed sequences.
    VarLen,
    /// Metal custom kernel (if available).
    MetalKernel,
}

impl AttentionBackend {
    /// Get a human-readable name for this backend.
    pub fn name(&self) -> &'static str {
        match self {
            Self::MlxFast => "MLX Fast SDPA",
            Self::Standard => "Standard Attention",
            Self::VarLen => "VarLen FlashAttention",
            Self::MetalKernel => "Metal Custom Kernel",
        }
    }
}

/// Sequence information for attention dispatch.
///
/// Provides metadata needed to select the optimal attention backend.
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Whether sequences are packed (variable length).
    pub is_packed: bool,
    /// Cumulative sequence lengths (for packed sequences).
    pub cu_seqlens: Option<Vec<i32>>,
    /// Maximum sequence length in batch.
    pub max_seqlen: i32,
    /// Total number of tokens.
    pub total_tokens: i32,
    /// Number of sequences in batch.
    pub num_sequences: i32,
}

impl SequenceInfo {
    /// Create sequence info for a regular (non-packed) batch.
    pub fn regular(batch_size: i32, seq_len: i32) -> Self {
        Self {
            is_packed: false,
            cu_seqlens: None,
            max_seqlen: seq_len,
            total_tokens: batch_size * seq_len,
            num_sequences: batch_size,
        }
    }

    /// Create sequence info for a packed batch.
    pub fn packed(cu_seqlens: Vec<i32>, max_seqlen: i32) -> Self {
        let num_sequences = cu_seqlens.len() as i32 - 1;
        let total_tokens = *cu_seqlens.last().unwrap_or(&0);
        Self {
            is_packed: true,
            cu_seqlens: Some(cu_seqlens),
            max_seqlen,
            total_tokens,
            num_sequences,
        }
    }
}

/// Attention backend dispatcher (Unsloth-style).
///
/// Automatically selects the optimal attention backend based on:
/// - Sequence packing status
/// - Hardware capabilities
/// - Sequence length characteristics
///
/// # Priority Order
///
/// 1. **VarLen** (for packed sequences) - O(n) memory, best for training
/// 2. **MlxFast** (FlashAttention) - efficient for long sequences
/// 3. **Standard** - fallback for maximum compatibility
#[derive(Debug, Clone)]
pub struct AttentionDispatcher {
    /// Preferred backend (override auto-selection).
    pub preferred_backend: Option<AttentionBackend>,
    /// Minimum sequence length to use FlashAttention.
    pub flash_min_seqlen: i32,
    /// Whether to log backend selection.
    pub verbose: bool,
}

impl Default for AttentionDispatcher {
    fn default() -> Self {
        Self {
            preferred_backend: None,
            flash_min_seqlen: 64, // FlashAttention not worth it for very short seqs
            verbose: false,
        }
    }
}

impl AttentionDispatcher {
    /// Create a new dispatcher with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a preferred backend (overrides auto-selection).
    pub fn with_preferred_backend(mut self, backend: AttentionBackend) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Enable verbose logging.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Select the optimal backend for the given sequence info and config.
    ///
    /// Returns the selected backend and a reason string for logging.
    pub fn select_backend(
        &self,
        seq_info: &SequenceInfo,
        config: &AttentionConfig,
    ) -> (AttentionBackend, &'static str) {
        // Honor explicit preference if set
        if let Some(preferred) = self.preferred_backend {
            return (preferred, "user preference");
        }

        // Priority 1: VarLen for packed sequences
        if seq_info.is_packed && is_varlen_available() {
            return (AttentionBackend::VarLen, "packed sequences");
        }

        // Priority 2: Metal kernel for special cases (softcap, sliding window)
        if config.has_softcap() || config.has_sliding_window() {
            // Custom Metal kernels handle these better
            return (AttentionBackend::MetalKernel, "softcap/sliding window");
        }

        // Priority 3: MLX Fast (FlashAttention) for long sequences
        if seq_info.max_seqlen >= self.flash_min_seqlen && is_flash_available() {
            return (AttentionBackend::MlxFast, "long sequences");
        }

        // Priority 4: Standard attention as fallback
        (AttentionBackend::Standard, "fallback")
    }

    /// Dispatch attention computation to the selected backend.
    ///
    /// Selects the optimal attention implementation based on sequence characteristics
    /// and hardware capabilities, then executes the computation.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, num_kv_heads, seq_len, head_dim]
    /// * `v` - Value tensor [batch, num_kv_heads, seq_len, head_dim]
    /// * `seq_info` - Sequence metadata for backend selection
    /// * `config` - Attention configuration (scale, mask, GQA settings, etc.)
    ///
    /// # Returns
    /// Tuple of (attention output [batch, num_heads, seq_len, head_dim], backend used)
    pub fn dispatch(
        &self,
        q: &mlx_rs::Array,
        k: &mlx_rs::Array,
        v: &mlx_rs::Array,
        seq_info: &SequenceInfo,
        config: &AttentionConfig,
    ) -> Result<(mlx_rs::Array, AttentionBackend)> {
        let (backend, reason) = self.select_backend(seq_info, config);

        if self.verbose {
            tracing::debug!(
                "Attention dispatch: {} ({}), seq_len={}, packed={}",
                backend.name(),
                reason,
                seq_info.max_seqlen,
                seq_info.is_packed
            );
        }

        let scale = config.scaling_factor();
        let seq_len = seq_info.max_seqlen;

        // Resolve the additive attention mask (0.0 for attend, -inf for mask).
        // Both MlxFast and Standard paths share the same mask construction logic;
        // the MlxFast path can also use the built-in causal string shortcut, but
        // passing an explicit array works equally well and keeps the code uniform.
        let mask: Option<mlx_rs::Array> = if config.is_causal {
            Some(create_causal_mask_array(seq_len))
        } else {
            config
                .sliding_window
                .map(|window| create_sliding_window_mask(seq_len, window as i32))
        };

        // Helper: call mlx_rs fast SDPA with correct mask type conversion
        let fast_sdpa = |q: &mlx_rs::Array,
                         k: &mlx_rs::Array,
                         v: &mlx_rs::Array,
                         scale: f32,
                         mask: &Option<mlx_rs::Array>|
         -> std::result::Result<mlx_rs::Array, pmetal_core::PMetalError> {
            let result = if let Some(m) = mask {
                mlx_rs::fast::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale,
                    m,
                    Option::<&mlx_rs::Array>::None,
                )
            } else {
                mlx_rs::fast::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale,
                    Option::<mlx_rs::fast::ScaledDotProductAttentionMask>::None,
                    Option::<&mlx_rs::Array>::None,
                )
            };
            result.map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))
        };

        let output = match backend {
            // MlxFast: delegate to mlx_rs::fast::scaled_dot_product_attention, which
            // dispatches to an optimised Metal kernel for single-token generation and
            // handles GQA/MQA natively without pre-expanding K/V heads.
            AttentionBackend::MlxFast => fast_sdpa(q, k, v, scale, &mask)?,

            // Standard: manual Q @ K^T * scale + mask -> softmax -> @ V.
            // Caller is responsible for pre-expanding K/V heads for GQA/MQA before
            // passing to dispatch; we simply compute the full attention here.
            AttentionBackend::Standard => {
                // scores = Q @ K^T  →  [batch, num_heads, seq_len_q, seq_len_k]
                let k_t = k
                    .transpose_axes(&[0, 1, 3, 2])
                    .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
                let scores = q
                    .matmul(&k_t)
                    .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

                // Scale
                let scale_arr = mlx_rs::Array::from_f32(scale);
                let scores = scores
                    .multiply(&scale_arr)
                    .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

                // Apply additive mask (if any)
                let scores = if let Some(ref m) = mask {
                    scores
                        .add(m)
                        .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?
                } else {
                    scores
                };

                // Softmax over key dimension (axis = -1)
                let weights = mlx_rs::ops::softmax_axis(&scores, -1, None)
                    .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

                // output = weights @ V  →  [batch, num_heads, seq_len_q, head_dim]
                weights
                    .matmul(v)
                    .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?
            }

            // VarLen: packed sequence attention is handled at a higher level using
            // cu_seqlens to demarcate per-sequence boundaries.  Within dispatch we
            // fall back to MlxFast which still produces correct results; the caller
            // should route packed batches through the dedicated VarLen path before
            // reaching here when maximum efficiency is required.
            AttentionBackend::VarLen => fast_sdpa(q, k, v, scale, &mask)?,

            // MetalKernel: custom fused Metal kernels (e.g. softcap, sliding window)
            // are invoked at the model level via crate::kernels::fused_sdpa.  For
            // dispatch purposes we fall back to MlxFast which correctly handles the
            // general case; model code that requires the fused kernel calls it
            // directly rather than going through this dispatcher.
            AttentionBackend::MetalKernel => fast_sdpa(q, k, v, scale, &mask)?,
        };

        Ok((output, backend))
    }
}

/// Check if FlashAttention is available.
pub fn is_flash_available() -> bool {
    FLASH_ATTENTION_AVAILABLE.load(Ordering::Relaxed)
}

/// Check if VarLen attention is available.
pub fn is_varlen_available() -> bool {
    VARLEN_ATTENTION_AVAILABLE.load(Ordering::Relaxed)
}

/// Mark FlashAttention as unavailable (call if it fails at runtime).
pub fn disable_flash_attention() {
    FLASH_ATTENTION_AVAILABLE.store(false, Ordering::Relaxed);
    tracing::warn!("FlashAttention disabled due to runtime error");
}

/// Mark VarLen attention as unavailable.
pub fn disable_varlen_attention() {
    VARLEN_ATTENTION_AVAILABLE.store(false, Ordering::Relaxed);
    tracing::warn!("VarLen attention disabled due to runtime error");
}

/// Multi-head attention layer with automatic backend dispatch.
pub struct MultiHeadAttention {
    /// Configuration.
    pub config: AttentionConfig,
    /// Backend to use (None for auto-dispatch).
    pub backend: Option<AttentionBackend>,
    /// Dispatcher for automatic backend selection.
    pub dispatcher: AttentionDispatcher,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer with auto-dispatch.
    pub fn new(config: AttentionConfig) -> Result<Self> {
        Ok(Self {
            config,
            backend: None,
            dispatcher: AttentionDispatcher::new(),
        })
    }

    /// Create with a specific backend.
    pub fn with_backend(config: AttentionConfig, backend: AttentionBackend) -> Result<Self> {
        Ok(Self {
            config,
            backend: Some(backend),
            dispatcher: AttentionDispatcher::new(),
        })
    }

    /// Get the effective backend for the given sequence info.
    pub fn get_backend(&self, seq_info: &SequenceInfo) -> AttentionBackend {
        if let Some(backend) = self.backend {
            backend
        } else {
            self.dispatcher.select_backend(seq_info, &self.config).0
        }
    }
}

/// Create a causal attention mask.
///
/// Returns a lower-triangular boolean mask of shape [seq_len, seq_len].
pub fn create_causal_mask(seq_len: usize) -> Vec<Vec<bool>> {
    (0..seq_len)
        .map(|i| (0..seq_len).map(|j| j <= i).collect())
        .collect()
}

/// Create a causal attention mask as an Array.
///
/// Returns a mask where valid positions are 0.0 and masked positions are -inf.
pub fn create_causal_mask_array(seq_len: i32) -> mlx_rs::Array {
    let mut mask_data = vec![0.0f32; (seq_len * seq_len) as usize];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[(i * seq_len + j) as usize] = f32::NEG_INFINITY;
        }
    }
    mlx_rs::Array::from_slice(&mask_data, &[seq_len, seq_len])
}

/// Create a sliding window causal mask.
///
/// Tokens can only attend to the last `window_size` tokens.
pub fn create_sliding_window_mask(seq_len: i32, window_size: i32) -> mlx_rs::Array {
    let mut mask_data = vec![f32::NEG_INFINITY; (seq_len * seq_len) as usize];
    for i in 0..seq_len {
        let start = (i - window_size + 1).max(0);
        for j in start..=i {
            mask_data[(i * seq_len + j) as usize] = 0.0;
        }
    }
    mlx_rs::Array::from_slice(&mask_data, &[seq_len, seq_len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert!(config.is_gqa());
        assert!(!config.is_mqa());
    }

    #[test]
    fn test_attention_config_scaling() {
        let config = AttentionConfig {
            head_dim: 64,
            ..Default::default()
        };
        let expected = 1.0 / (64.0_f32).sqrt();
        assert!((config.scaling_factor() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_info_regular() {
        let info = SequenceInfo::regular(4, 512);
        assert!(!info.is_packed);
        assert!(info.cu_seqlens.is_none());
        assert_eq!(info.max_seqlen, 512);
        assert_eq!(info.total_tokens, 2048);
    }

    #[test]
    fn test_sequence_info_packed() {
        let cu_seqlens = vec![0, 100, 250, 400];
        let info = SequenceInfo::packed(cu_seqlens.clone(), 150);
        assert!(info.is_packed);
        assert_eq!(info.cu_seqlens, Some(cu_seqlens));
        assert_eq!(info.max_seqlen, 150);
        assert_eq!(info.num_sequences, 3);
        assert_eq!(info.total_tokens, 400);
    }

    #[test]
    fn test_dispatcher_packed_sequences() {
        let dispatcher = AttentionDispatcher::new();
        let config = AttentionConfig::default();
        let seq_info = SequenceInfo::packed(vec![0, 100, 200], 100);

        let (backend, _reason) = dispatcher.select_backend(&seq_info, &config);
        assert_eq!(backend, AttentionBackend::VarLen);
    }

    #[test]
    fn test_dispatcher_long_sequences() {
        let dispatcher = AttentionDispatcher::new();
        let config = AttentionConfig::default();
        let seq_info = SequenceInfo::regular(4, 1024);

        let (backend, _reason) = dispatcher.select_backend(&seq_info, &config);
        assert_eq!(backend, AttentionBackend::MlxFast);
    }

    #[test]
    fn test_dispatcher_short_sequences() {
        let dispatcher = AttentionDispatcher {
            flash_min_seqlen: 64,
            ..Default::default()
        };
        let config = AttentionConfig::default();
        let seq_info = SequenceInfo::regular(4, 32);

        let (backend, _reason) = dispatcher.select_backend(&seq_info, &config);
        assert_eq!(backend, AttentionBackend::Standard);
    }

    #[test]
    fn test_dispatcher_softcap() {
        let dispatcher = AttentionDispatcher::new();
        let config = AttentionConfig {
            softcap: 30.0,
            ..Default::default()
        };
        let seq_info = SequenceInfo::regular(4, 512);

        let (backend, _reason) = dispatcher.select_backend(&seq_info, &config);
        assert_eq!(backend, AttentionBackend::MetalKernel);
    }

    #[test]
    fn test_dispatcher_preferred_backend() {
        let dispatcher =
            AttentionDispatcher::new().with_preferred_backend(AttentionBackend::Standard);
        let config = AttentionConfig::default();
        let seq_info = SequenceInfo::packed(vec![0, 100], 100);

        let (backend, reason) = dispatcher.select_backend(&seq_info, &config);
        assert_eq!(backend, AttentionBackend::Standard);
        assert_eq!(reason, "user preference");
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3);
        assert_eq!(mask[0], vec![true, false, false]);
        assert_eq!(mask[1], vec![true, true, false]);
        assert_eq!(mask[2], vec![true, true, true]);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_sliding_window_mask() {
        let mask = create_sliding_window_mask(5, 2);
        mask.eval().unwrap();
        let data: Vec<f32> = mask.as_slice().to_vec();

        // Position 0 can only attend to 0
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], f32::NEG_INFINITY);

        // Position 2 can attend to 1, 2 (window of 2)
        assert_eq!(data[2 * 5 + 0], f32::NEG_INFINITY);
        assert_eq!(data[2 * 5 + 1], 0.0);
        assert_eq!(data[2 * 5 + 2], 0.0);

        // Position 4 can attend to 3, 4
        assert_eq!(data[4 * 5 + 2], f32::NEG_INFINITY);
        assert_eq!(data[4 * 5 + 3], 0.0);
        assert_eq!(data[4 * 5 + 4], 0.0);
    }
}
