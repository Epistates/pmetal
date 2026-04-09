#![allow(unsafe_code)]

//! Metal 4 / MPP FlashAttention dispatch.
//!
//! This is an Apple10/M5-only forward-path wrapper over
//! `metal4/mpp_flash_attention.metal`. The current shader contract supports
//! fp16 attention with `head_dim = 64`, `80`, `96`, or `128` for causal and non-causal inference.

use std::ptr::NonNull;
use std::sync::Arc;

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLComputeCommandEncoder};

use crate::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
    kernels::mpp_dispatch::encode_mpp_kernel,
};

/// Configuration for MPP FlashAttention.
#[derive(Debug, Clone)]
pub struct MppFlashAttentionConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads.
    pub num_kv_heads: usize,
    /// Query sequence length.
    pub query_seq_len: usize,
    /// Key/value sequence length.
    pub kv_seq_len: usize,
    /// Head dimension. The current MPP kernel supports `64`, `80`, `96`, and `128`.
    pub head_dim: usize,
    /// Optional softmax scaling factor. Defaults to `1 / sqrt(head_dim)`.
    pub scale: Option<f32>,
    /// Whether to use causal masking.
    pub is_causal: bool,
    /// Optional sliding window for causal masking.
    pub sliding_window: Option<usize>,
    /// Optional logit softcap.
    pub softcap: Option<f32>,
}

impl MppFlashAttentionConfig {
    /// Return the scaling factor.
    #[inline]
    pub fn scaling_factor(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// Return the GQA ratio.
    #[inline]
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Output buffer size in fp16 elements.
    #[inline]
    pub fn output_size(&self) -> usize {
        self.batch_size * self.num_heads * self.query_seq_len * self.head_dim
    }

    /// Log-sum-exp buffer size in fp32 elements.
    #[inline]
    pub fn logsumexp_size(&self) -> usize {
        self.batch_size * self.num_heads * self.query_seq_len
    }
}

#[repr(C)]
struct FlashAttentionParams {
    batch_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    query_seq_len: u32,
    kv_seq_len: u32,
    head_dim: u32,
    scale: f32,
    block_q: u32,
    block_k: u32,
    gqa_ratio: u32,
    is_causal: u32,
    sliding_window: u32,
    softcap: f32,
}

fn validate_config(config: &MppFlashAttentionConfig) -> Result<()> {
    if config.batch_size == 0
        || config.num_heads == 0
        || config.num_kv_heads == 0
        || config.query_seq_len == 0
        || config.kv_seq_len == 0
    {
        return Err(MetalError::InvalidConfig(
            "MPP FlashAttention dimensions must be non-zero".to_string(),
        ));
    }

    if !matches!(config.head_dim, 64 | 80 | 96 | 128) {
        return Err(MetalError::InvalidConfig(format!(
            "MPP FlashAttention currently supports head_dim=64, 80, 96, or 128, got {}",
            config.head_dim
        )));
    }

    if config.num_heads % config.num_kv_heads != 0 {
        return Err(MetalError::InvalidConfig(format!(
            "MPP FlashAttention requires num_heads ({}) divisible by num_kv_heads ({})",
            config.num_heads, config.num_kv_heads
        )));
    }

    Ok(())
}

fn kernel_name(config: &MppFlashAttentionConfig) -> Result<&'static str> {
    validate_config(config)?;
    match (config.head_dim, config.is_causal) {
        (64, true) => Ok("mpp_flash_attention_fwd_d64_causal"),
        (64, false) => Ok("mpp_flash_attention_fwd_d64"),
        (80, true) => Ok("mpp_flash_attention_fwd_d80_causal"),
        (80, false) => Ok("mpp_flash_attention_fwd_d80"),
        (96, true) => Ok("mpp_flash_attention_fwd_d96_causal"),
        (96, false) => Ok("mpp_flash_attention_fwd_d96"),
        (128, true) => Ok("mpp_flash_attention_fwd_d128_causal"),
        (128, false) => Ok("mpp_flash_attention_fwd_d128"),
        _ => Err(MetalError::InvalidConfig(format!(
            "unsupported MPP FlashAttention head_dim={}",
            config.head_dim
        ))),
    }
}

fn validate_input_sizes(
    config: &MppFlashAttentionConfig,
    queries: &MetalBuffer<f16>,
    keys: &MetalBuffer<f16>,
    values: &MetalBuffer<f16>,
) -> Result<()> {
    let expected_queries =
        config.batch_size * config.num_heads * config.query_seq_len * config.head_dim;
    if queries.len() != expected_queries {
        return Err(MetalError::DimensionMismatch {
            param: "queries",
            expected: expected_queries,
            actual: queries.len(),
        });
    }

    let expected_kv = config.batch_size * config.num_kv_heads * config.kv_seq_len * config.head_dim;
    if keys.len() != expected_kv {
        return Err(MetalError::DimensionMismatch {
            param: "keys",
            expected: expected_kv,
            actual: keys.len(),
        });
    }
    if values.len() != expected_kv {
        return Err(MetalError::DimensionMismatch {
            param: "values",
            expected: expected_kv,
            actual: values.len(),
        });
    }

    Ok(())
}

/// Output from MPP FlashAttention forward.
#[derive(Debug)]
pub struct MppFlashAttentionOutput {
    /// Attention output tensor.
    pub output: MetalBuffer<f16>,
    /// Per-row log-sum-exp values.
    pub logsumexp: MetalBuffer<f32>,
}

/// Apple10/M5-only forward FlashAttention wrapper over the Metal 4 / MPP shader.
pub struct MppFlashAttention {
    ctx: Arc<MetalContext>,
    config: MppFlashAttentionConfig,
}

impl MppFlashAttention {
    /// Create a new MPP FlashAttention wrapper.
    pub fn new(ctx: Arc<MetalContext>, config: MppFlashAttentionConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self { ctx, config })
    }

    /// Whether the current device can execute the MPP FlashAttention kernels.
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Run the forward kernel synchronously.
    pub fn forward(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
    ) -> Result<MppFlashAttentionOutput> {
        validate_input_sizes(&self.config, queries, keys, values)?;

        let output = MetalBuffer::zeros(&self.ctx, self.config.output_size(), BufferUsage::Shared)?;
        let logsumexp =
            MetalBuffer::zeros(&self.ctx, self.config.logsumexp_size(), BufferUsage::Shared)?;

        let command_buffer = self.execute_async(queries, keys, values, &output, &logsumexp)?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(MppFlashAttentionOutput { output, logsumexp })
    }

    /// Encode and submit the forward kernel asynchronously.
    pub fn execute_async(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        output: &MetalBuffer<f16>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP FlashAttention not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let function_name = kernel_name(&self.config)?;

        let params = FlashAttentionParams {
            batch_size: self.config.batch_size as u32,
            num_heads: self.config.num_heads as u32,
            num_kv_heads: self.config.num_kv_heads as u32,
            query_seq_len: self.config.query_seq_len as u32,
            kv_seq_len: self.config.kv_seq_len as u32,
            head_dim: self.config.head_dim as u32,
            scale: self.config.scaling_factor(),
            block_q: 32,
            block_k: 32,
            gqa_ratio: self.config.gqa_ratio() as u32,
            is_causal: self.config.is_causal as u32,
            sliding_window: self.config.sliding_window.unwrap_or(0) as u32,
            softcap: self.config.softcap.unwrap_or(0.0),
        };

        let grid = objc2_metal::MTLSize {
            width: self.config.query_seq_len.div_ceil(32),
            height: self.config.num_heads,
            depth: self.config.batch_size,
        };
        let tg_size = objc2_metal::MTLSize { width: 32, height: 4, depth: 1 };

        let q_buf = queries.metal_buffer();
        let k_buf = keys.metal_buffer();
        let v_buf = values.metal_buffer();
        let out_buf = output.metal_buffer();
        let lse_buf = logsumexp.metal_buffer();

        encode_mpp_kernel(&self.ctx, function_name, grid, tg_size, |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(lse_buf), 0, 4);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MppFlashAttentionConfig {
        MppFlashAttentionConfig {
            batch_size: 1,
            num_heads: 8,
            num_kv_heads: 2,
            query_seq_len: 32,
            kv_seq_len: 64,
            head_dim: 128,
            scale: None,
            is_causal: true,
            sliding_window: None,
            softcap: None,
        }
    }

    #[test]
    fn mpp_flash_attention_config_accepts_supported_shape() {
        assert!(validate_config(&test_config()).is_ok());

        let mut d64 = test_config();
        d64.head_dim = 64;
        assert!(validate_config(&d64).is_ok());

        let mut d96 = test_config();
        d96.head_dim = 96;
        assert!(validate_config(&d96).is_ok());

        let mut d80 = test_config();
        d80.head_dim = 80;
        assert!(validate_config(&d80).is_ok());
    }

    #[test]
    fn mpp_flash_attention_config_rejects_unsupported_head_dim() {
        let mut config = test_config();
        config.head_dim = 72;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("head_dim=64, 80, 96, or 128"));
    }

    #[test]
    fn mpp_flash_attention_config_rejects_invalid_gqa_ratio() {
        let mut config = test_config();
        config.num_heads = 10;
        config.num_kv_heads = 3;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("divisible"));
    }

    #[test]
    fn mpp_flash_attention_kernel_name_tracks_causality() {
        let causal = test_config();
        assert_eq!(
            kernel_name(&causal).unwrap(),
            "mpp_flash_attention_fwd_d128_causal"
        );

        let mut non_causal = test_config();
        non_causal.is_causal = false;
        assert_eq!(
            kernel_name(&non_causal).unwrap(),
            "mpp_flash_attention_fwd_d128"
        );

        let mut d64 = test_config();
        d64.head_dim = 64;
        assert_eq!(
            kernel_name(&d64).unwrap(),
            "mpp_flash_attention_fwd_d64_causal"
        );
        d64.is_causal = false;
        assert_eq!(kernel_name(&d64).unwrap(), "mpp_flash_attention_fwd_d64");

        let mut d80 = test_config();
        d80.head_dim = 80;
        assert_eq!(
            kernel_name(&d80).unwrap(),
            "mpp_flash_attention_fwd_d80_causal"
        );
        d80.is_causal = false;
        assert_eq!(kernel_name(&d80).unwrap(), "mpp_flash_attention_fwd_d80");

        let mut d96 = test_config();
        d96.head_dim = 96;
        assert_eq!(
            kernel_name(&d96).unwrap(),
            "mpp_flash_attention_fwd_d96_causal"
        );
        d96.is_causal = false;
        assert_eq!(kernel_name(&d96).unwrap(), "mpp_flash_attention_fwd_d96");
    }

    #[test]
    fn mpp_flash_attention_sizes_match_shape() {
        let config = test_config();
        assert_eq!(config.output_size(), 8 * 32 * 128);
        assert_eq!(config.logsumexp_size(), 8 * 32);

        let mut d64 = test_config();
        d64.head_dim = 64;
        assert_eq!(d64.output_size(), 8 * 32 * 64);
        assert_eq!(d64.logsumexp_size(), 8 * 32);

        let mut d96 = test_config();
        d96.head_dim = 96;
        assert_eq!(d96.output_size(), 8 * 32 * 96);
        assert_eq!(d96.logsumexp_size(), 8 * 32);

        let mut d80 = test_config();
        d80.head_dim = 80;
        assert_eq!(d80.output_size(), 8 * 32 * 80);
        assert_eq!(d80.logsumexp_size(), 8 * 32);
    }
}
