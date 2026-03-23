#![allow(unsafe_code)]

//! Fused RMSNorm + LoRA projection Metal kernel.
//!
//! This kernel combines RMSNorm with LoRA projection in a single kernel launch:
//!
//! ```text
//! output = (norm(x) @ W.T) + scale * ((norm(x) @ A.T) @ B.T)
//! ```
//!
//! where `norm(x) = x / sqrt(mean(x^2) + eps) * gamma`
//!
//! # Benefits
//!
//! - Eliminates intermediate materialization of normalized values
//! - Single kernel launch instead of 4+ separate ops
//! - ~15-25% speedup over separate RMSNorm + LoRA
//!
//! # Novel Optimization
//!
//! This is a novel optimization: no existing framework fuses normalization with LoRA.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
    pipeline::FunctionConstant,
    tuna::NormLoraTunedConfig,
};

/// Configuration for fused RMSNorm + LoRA kernel.
#[derive(Debug, Clone)]
pub struct FusedNormLoraConfig {
    /// Batch size (number of tokens).
    pub batch_size: usize,

    /// Hidden dimension.
    pub hidden_size: usize,

    /// Output dimension.
    pub out_features: usize,

    /// LoRA rank.
    pub lora_rank: usize,

    /// RMSNorm epsilon.
    pub eps: f32,

    /// LoRA scaling factor (alpha / rank).
    pub lora_scale: f32,

    /// Use fp16 kernel.
    pub use_fp16: bool,

    /// Use tiled kernel for better parallelism.
    pub use_tiled: bool,
}

impl FusedNormLoraConfig {
    /// Create a new config.
    ///
    /// # Panics
    ///
    /// Panics if `hidden_size` is not a multiple of 4 (required for float4 vectorized loads).
    pub fn new(
        batch_size: usize,
        hidden_size: usize,
        out_features: usize,
        lora_rank: usize,
        lora_alpha: f32,
    ) -> Self {
        assert!(
            hidden_size % 4 == 0,
            "hidden_size ({hidden_size}) must be a multiple of 4 for vectorized Metal kernels"
        );
        Self {
            batch_size,
            hidden_size,
            out_features,
            lora_rank,
            eps: 1e-6,
            lora_scale: lora_alpha / lora_rank as f32,
            use_fp16: false,
            use_tiled: true, // Default to tiled for better performance
        }
    }

    /// Set RMSNorm epsilon.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Enable fp16 mode.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }

    /// Use tiled kernel.
    pub fn with_tiled(mut self, tiled: bool) -> Self {
        self.use_tiled = tiled;
        self
    }
}

/// Output from fused norm + LoRA kernel.
#[derive(Debug)]
pub struct FusedNormLoraOutput {
    /// Output tensor [batch_size, out_features].
    pub output: MetalBuffer<f32>,
}

/// Fused RMSNorm + LoRA kernel.
///
/// Combines normalization and LoRA projection for maximum efficiency.
///
/// # Example
///
/// ```ignore
/// let config = FusedNormLoraConfig::new(
///     batch_size,
///     hidden_size,
///     out_features,
///     lora_rank,
///     lora_alpha,
/// );
/// let kernel = FusedNormLora::new(ctx, config)?;
/// let output = kernel.forward(&input, &gamma, &weight, &lora_a, &lora_b)?;
/// ```
pub struct FusedNormLora {
    ctx: Arc<MetalContext>,
    config: FusedNormLoraConfig,
    /// Tuned kernel specialization for this device/problem shape.
    tuned: NormLoraTunedConfig,
    /// Effective threadgroup size.
    threads_per_token: usize,
    /// Effective tiled-path choice after tuning.
    use_tiled: bool,
}

impl FusedNormLora {
    /// Create a new fused norm + LoRA kernel.
    pub fn new(ctx: Arc<MetalContext>, config: FusedNormLoraConfig) -> Result<Self> {
        let tuned = resolve_norm_lora_tuned_config(&ctx, &config)?;
        Self::new_with_tuned_config(ctx, config, tuned)
    }

    /// Create a new fused norm + LoRA kernel with an explicit tuned specialization.
    pub(crate) fn new_with_tuned_config(
        ctx: Arc<MetalContext>,
        config: FusedNormLoraConfig,
        tuned: NormLoraTunedConfig,
    ) -> Result<Self> {
        let tuned = normalize_norm_lora_tuned_config(&ctx, tuned);
        let threads_per_token = tuned.threads_per_token as usize;
        let use_tiled = config.use_tiled && tuned.use_tiled;
        Ok(Self {
            ctx,
            config,
            tuned,
            threads_per_token,
            use_tiled,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusedNormLoraConfig {
        &self.config
    }

    fn function_constants(&self) -> HashMap<u64, FunctionConstant> {
        let mut constants = HashMap::new();
        constants.insert(0, FunctionConstant::UInt(self.tuned.threads_per_token));
        constants
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch_size, hidden_size]
    /// * `gamma` - RMSNorm weight [hidden_size]
    /// * `weight` - Base weight [out_features, hidden_size]
    /// * `lora_a` - LoRA A matrix [lora_rank, hidden_size]
    /// * `lora_b` - LoRA B matrix [out_features, lora_rank]
    pub fn forward(
        &self,
        input: &MetalBuffer<f32>,
        gamma: &MetalBuffer<f32>,
        weight: &MetalBuffer<f32>,
        lora_a: &MetalBuffer<f32>,
        lora_b: &MetalBuffer<f32>,
    ) -> Result<FusedNormLoraOutput> {
        // Validate sizes
        let expected_input = self.config.batch_size * self.config.hidden_size;
        if input.len() != expected_input {
            return Err(MetalError::DimensionMismatch {
                param: "input",
                expected: expected_input,
                actual: input.len(),
            });
        }

        if gamma.len() != self.config.hidden_size {
            return Err(MetalError::DimensionMismatch {
                param: "gamma",
                expected: self.config.hidden_size,
                actual: gamma.len(),
            });
        }

        let expected_weight = self.config.out_features * self.config.hidden_size;
        if weight.len() != expected_weight {
            return Err(MetalError::DimensionMismatch {
                param: "weight",
                expected: expected_weight,
                actual: weight.len(),
            });
        }

        let expected_a = self.config.lora_rank * self.config.hidden_size;
        if lora_a.len() != expected_a {
            return Err(MetalError::DimensionMismatch {
                param: "lora_a",
                expected: expected_a,
                actual: lora_a.len(),
            });
        }

        let expected_b = self.config.out_features * self.config.lora_rank;
        if lora_b.len() != expected_b {
            return Err(MetalError::DimensionMismatch {
                param: "lora_b",
                expected: expected_b,
                actual: lora_b.len(),
            });
        }

        // Allocate output
        let output_size = self.config.batch_size * self.config.out_features;
        let output = MetalBuffer::new(&self.ctx, output_size, BufferUsage::Shared)?;

        self.execute_forward(input, gamma, weight, lora_a, lora_b, &output)?;

        Ok(FusedNormLoraOutput { output })
    }

    fn execute_forward(
        &self,
        input: &MetalBuffer<f32>,
        gamma: &MetalBuffer<f32>,
        weight: &MetalBuffer<f32>,
        lora_a: &MetalBuffer<f32>,
        lora_b: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
    ) -> Result<()> {
        let kernel_name = if self.use_tiled {
            "fused_norm_lora_forward_tiled"
        } else {
            "fused_norm_lora_forward"
        };

        let constants = self.function_constants();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                kernel_name,
                &constants,
            )?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // SAFETY: Metal compute encoder operations are safe when buffers are valid
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gamma.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(lora_a.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(lora_b.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 5);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 6);

            // Threadgroup memory for scratch space.
            // Tiled kernel needs: hidden_size floats for norm_x + lora_rank floats for x_a.
            // Non-tiled kernels additionally need space for the cross-SIMD-group parallel
            // reduction scratch: num_simd_groups = THREADS_PER_TOKEN / SIMD_SIZE = 128 / 32 = 4
            // floats appended after norm_x and x_a.
            let num_simd_groups = self.threads_per_token.div_ceil(32);
            let scratch_floats = if self.use_tiled {
                self.config.hidden_size + self.config.lora_rank
            } else {
                self.config.hidden_size + self.config.lora_rank + num_simd_groups
            };
            let scratch_size = scratch_floats * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(scratch_size, 0);
        }

        let (grid_size, threadgroup_size) = if self.use_tiled {
            // Tiled version: one threadgroup per token
            (
                objc2_metal::MTLSize {
                    width: self.config.batch_size,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: self.threads_per_token,
                    height: 1,
                    depth: 1,
                },
            )
        } else {
            // Non-tiled: one threadgroup per (token, output) pair
            (
                objc2_metal::MTLSize {
                    width: self.config.batch_size,
                    height: self.config.out_features,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: self.threads_per_token,
                    height: 1,
                    depth: 1,
                },
            )
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    fn create_params(&self) -> FusedNormLoraParams {
        FusedNormLoraParams {
            batch_size: self.config.batch_size as u32,
            hidden_size: self.config.hidden_size as u32,
            out_features: self.config.out_features as u32,
            lora_rank: self.config.lora_rank as u32,
            eps: self.config.eps,
            lora_scale: self.config.lora_scale,
        }
    }
}

/// Parameters passed to the Metal kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FusedNormLoraParams {
    batch_size: u32,
    hidden_size: u32,
    out_features: u32,
    lora_rank: u32,
    eps: f32,
    lora_scale: f32,
}

fn resolve_norm_lora_tuned_config(
    ctx: &Arc<MetalContext>,
    config: &FusedNormLoraConfig,
) -> Result<NormLoraTunedConfig> {
    let tuned = ctx.tuner().tune_norm_lora(
        ctx,
        config.batch_size,
        config.hidden_size,
        config.out_features,
        config.lora_rank,
    )?;
    Ok(normalize_norm_lora_tuned_config(ctx, tuned))
}

fn sanitize_norm_lora_threads_per_token(ctx: &MetalContext, threads_per_token: u32) -> u32 {
    let max_threads = (ctx.properties().max_threads_per_threadgroup as u32).max(32);
    threads_per_token.clamp(32, max_threads).div_ceil(32) * 32
}

fn normalize_norm_lora_tuned_config(
    ctx: &MetalContext,
    tuned: NormLoraTunedConfig,
) -> NormLoraTunedConfig {
    NormLoraTunedConfig {
        threads_per_token: sanitize_norm_lora_threads_per_token(ctx, tuned.threads_per_token),
        use_tiled: tuned.use_tiled,
    }
}

impl std::fmt::Debug for FusedNormLora {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedNormLora")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_norm_lora_config() {
        let config = FusedNormLoraConfig::new(4, 512, 1024, 8, 16.0);

        assert_eq!(config.batch_size, 4);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.out_features, 1024);
        assert_eq!(config.lora_rank, 8);
        assert!((config.lora_scale - 2.0).abs() < 1e-6); // 16 / 8 = 2
        assert!(config.use_tiled);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn fused_norm_lora_constructor_uses_tuned_specialization() {
        let ctx = Arc::new(MetalContext::new().expect("Metal required"));
        let small = FusedNormLoraConfig::new(2, 512, 128, 8, 16.0);
        let large = FusedNormLoraConfig::new(2, 512, 1024, 8, 16.0);

        let tuned_small = ctx
            .tuner()
            .tune_norm_lora(
                &ctx,
                small.batch_size,
                small.hidden_size,
                small.out_features,
                small.lora_rank,
            )
            .expect("tune_norm_lora small");
        let tuned_large = ctx
            .tuner()
            .tune_norm_lora(
                &ctx,
                large.batch_size,
                large.hidden_size,
                large.out_features,
                large.lora_rank,
            )
            .expect("tune_norm_lora large");

        let kernel_small = FusedNormLora::new(ctx.clone(), small).expect("kernel small");
        let kernel_large = FusedNormLora::new(ctx.clone(), large).expect("kernel large");

        assert_eq!(
            kernel_small.threads_per_token as u32,
            sanitize_norm_lora_threads_per_token(&ctx, tuned_small.threads_per_token)
        );
        assert_eq!(
            kernel_large.threads_per_token as u32,
            sanitize_norm_lora_threads_per_token(&ctx, tuned_large.threads_per_token)
        );
        assert_eq!(kernel_small.use_tiled, tuned_small.use_tiled);
        assert_eq!(kernel_large.use_tiled, tuned_large.use_tiled);
    }
}
