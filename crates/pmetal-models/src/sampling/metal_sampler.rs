//! High-performance Metal sampler for token generation.
//!
//! This module provides a fused Metal sampling kernel that executes all
//! sampling operations in a single GPU kernel launch, eliminating the
//! CPU overhead of multiple mlx-rs operations.
//!
//! # Performance Benefits
//!
//! - **Single kernel launch** vs 10+ separate launches with mlx-rs path
//! - **Minimal CPU overhead** - critical for battery mode performance
//! - **Zero-copy** from MLX arrays via unified memory
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_models::sampling::MetalSampler;
//!
//! let mut sampler = MetalSampler::new(vocab_size);
//!
//! // Dispatch sampling asynchronously
//! sampler.sample_async(&logits_array, temperature, top_k, top_p, min_p);
//!
//! // ... do other work while GPU computes ...
//!
//! // Get the result
//! let token = sampler.get_token();
//! ```

#![allow(unsafe_code)]

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use pmetal_bridge::compat::{Array, Dtype};
use pmetal_metal::{FusedSampler, MetalContext, MetalError, bridge::metal_buffer_from_ptr};

/// Error type for MetalSampler operations.
#[derive(Debug, thiserror::Error)]
pub enum MetalSamplerError {
    /// Metal operation failed.
    #[error("Metal error: {0}")]
    Metal(#[from] MetalError),

    /// Array is not contiguous.
    #[error("Array must be contiguous for zero-copy Metal access")]
    NonContiguous,

    /// Array has wrong dtype.
    #[error("Array must be float32, got {0:?}")]
    WrongDtype(pmetal_bridge::compat::Dtype),

    /// Array is empty.
    #[error("Cannot sample from empty array")]
    EmptyArray,
}

/// Result type for MetalSampler operations.
pub type Result<T> = std::result::Result<T, MetalSamplerError>;

/// In-flight sampler dispatch that retains the source logits until completion.
struct PendingDispatch {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    logits: Array,
}

/// High-performance Metal sampler using fused kernel.
///
/// This sampler bypasses the mlx-rs sampling path to execute all sampling
/// operations in a single Metal kernel, providing significant speedups
/// especially on battery power where CPU is throttled.
pub struct MetalSampler {
    /// Underlying fused sampler.
    fused: FusedSampler,
    /// Metal context.
    ctx: Arc<MetalContext>,
    /// Pending asynchronous dispatch, including a retained logits array to keep
    /// zero-copy memory alive until the GPU has finished reading from it.
    pending: Option<PendingDispatch>,
}

impl MetalSampler {
    /// Create a new Metal sampler.
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size of the model.
    ///
    /// # Errors
    /// Returns an error if Metal initialization fails.
    pub fn new(vocab_size: usize) -> Result<Self> {
        let ctx = MetalContext::global()?;
        let fused = FusedSampler::with_context(ctx.clone(), vocab_size)?;

        Ok(Self {
            fused,
            ctx,
            pending: None,
        })
    }

    /// Create a new Metal sampler with a specific seed for reproducibility.
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size of the model.
    /// * `seed` - Random seed for reproducible sampling.
    ///
    /// # Errors
    /// Returns an error if Metal initialization fails.
    pub fn with_seed(vocab_size: usize, seed: u64) -> Result<Self> {
        let ctx = MetalContext::global()?;
        let fused = FusedSampler::with_context_and_seed(ctx.clone(), vocab_size, Some(seed))?;

        Ok(Self {
            fused,
            ctx,
            pending: None,
        })
    }

    /// Set the random seed for reproducible sampling.
    ///
    /// # Arguments
    /// * `seed` - Random seed value.
    pub fn set_seed(&mut self, seed: u64) {
        self.fused.set_seed(seed);
    }

    /// Dispatch sampling asynchronously.
    ///
    /// This method returns immediately after dispatching the kernel.
    /// Call `get_token()` to wait for completion and retrieve the result.
    ///
    /// # Arguments
    /// * `logits` - MLX Array of logits [vocab_size] as f32
    /// * `temperature` - Sampling temperature (0 = greedy)
    /// * `top_k` - Top-K filtering (0 = disabled)
    /// * `top_p` - Top-P nucleus sampling threshold
    /// * `min_p` - Min-P threshold relative to max probability
    ///
    /// This method retains the source array until completion so callers may
    /// safely drop their original handle after dispatch.
    pub fn sample_async(
        &mut self,
        logits: &Array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
    ) -> Result<()> {
        self.wait_for_pending()?;
        self.validate_logits(logits)?;

        // Get raw data pointer from MLX array via as_slice()
        // SAFETY: Array is evaluated and we're using unified memory
        let slice = logits.as_slice::<f32>();
        let data_ptr = slice.as_ptr() as *mut f32;

        // Create zero-copy buffer view
        // SAFETY: MLX uses unified memory, pointer is valid for GPU access
        let buffer_view = unsafe { metal_buffer_from_ptr(&self.ctx, data_ptr, logits.size()) }?;

        // Dispatch kernel asynchronously
        let cmd = self
            .fused
            .sample_async(&buffer_view, temperature, top_k, top_p, min_p)?;

        self.pending = Some(PendingDispatch {
            cmd,
            logits: logits.clone(),
        });
        Ok(())
    }

    /// Dispatch greedy argmax asynchronously.
    ///
    /// Faster path for temperature=0 (greedy decoding).
    pub fn argmax_async(&mut self, logits: &Array) -> Result<()> {
        self.wait_for_pending()?;
        self.validate_logits(logits)?;

        // Get raw data pointer via as_slice()
        let slice = logits.as_slice::<f32>();
        let data_ptr = slice.as_ptr() as *mut f32;

        // Create zero-copy buffer view
        // SAFETY: MLX uses unified memory, pointer is valid for GPU access
        let buffer_view = unsafe { metal_buffer_from_ptr(&self.ctx, data_ptr, logits.size()) }?;

        // Dispatch kernel
        let cmd = self.fused.argmax_async(&buffer_view)?;
        self.pending = Some(PendingDispatch {
            cmd,
            logits: logits.clone(),
        });
        Ok(())
    }

    /// Wait for pending operation and get the sampled token.
    ///
    /// If no operation is pending, returns 0.
    pub fn get_token(&mut self) -> Result<u32> {
        self.wait_for_pending()?;
        Ok(self.fused.read_result())
    }

    /// Check if there's a pending operation.
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.fused.vocab_size()
    }

    /// Synchronous sample (dispatches and waits).
    ///
    /// Convenience method that combines `sample_async` and `get_token`.
    pub fn sample(
        &mut self,
        logits: &Array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
    ) -> Result<u32> {
        self.sample_async(logits, temperature, top_k, top_p, min_p)?;
        self.get_token()
    }

    /// Synchronous argmax (dispatches and waits).
    pub fn argmax(&mut self, logits: &Array) -> Result<u32> {
        self.argmax_async(logits)?;
        self.get_token()
    }

    fn validate_logits(&self, logits: &Array) -> Result<()> {
        // Validate input
        if logits.size() == 0 {
            return Err(MetalSamplerError::EmptyArray);
        }

        // Ensure array is evaluated before getting data pointer
        let logits_owned = logits.clone();
        logits_owned.eval();

        // Check dtype - must be f32
        let dtype = logits.dtype();
        if dtype != pmetal_bridge::compat::Dtype::Float32 {
            return Err(MetalSamplerError::WrongDtype(dtype));
        }
        Ok(())
    }

    fn wait_for_pending(&mut self) -> Result<()> {
        if let Some(pending) = self.pending.take() {
            pending.cmd.waitUntilCompleted();

            // Keep the logits alive until the command buffer has fully completed.
            let _retained_logits = pending.logits;

            if let Some(error) = pending.cmd.error() {
                return Err(MetalSamplerError::Metal(MetalError::ExecutionFailed(
                    error.to_string(),
                )));
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for MetalSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSampler")
            .field("vocab_size", &self.vocab_size())
            .field("has_pending", &self.has_pending())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_sampler_creation() {
        let sampler = MetalSampler::new(32000);
        assert!(sampler.is_ok(), "Should create MetalSampler on macOS");
    }

    #[test]
    fn test_metal_sampler_argmax() {
        let mut sampler = MetalSampler::new(100).unwrap();

        // Create logits with token 42 having highest value
        let mut logits_vec = vec![-10.0f32; 100];
        logits_vec[42] = 10.0;
        let logits = Array::from_slice(&logits_vec, &[100]);
        logits.eval().unwrap();

        let token = sampler.argmax(&logits).unwrap();
        assert_eq!(token, 42);
    }

    #[test]
    fn test_metal_sampler_top_k_correctness() {
        // Test that top-K filtering works correctly after the race condition fix
        let mut sampler = MetalSampler::new(1000).unwrap();

        // Create logits where tokens 0-9 have high values, rest are very low
        // This tests that top-K correctly identifies the top candidates
        let mut logits_vec = vec![-100.0f32; 1000];
        for (i, val) in logits_vec.iter_mut().take(10).enumerate() {
            *val = 10.0 - (i as f32); // Token 0 = 10.0, Token 1 = 9.0, etc.
        }
        let logits = Array::from_slice(&logits_vec, &[1000]);
        logits.eval().unwrap();

        // Sample 100 times with top_k=5, temperature=1.0
        // All samples should come from tokens 0-4 (the top 5)
        let mut counts = std::collections::HashMap::new();
        for _ in 0..100 {
            let token = sampler.sample(&logits, 1.0, 5, 1.0, 0.0).unwrap();
            *counts.entry(token).or_insert(0) += 1;

            // Verify sampled token is in top-5
            assert!(
                token < 5,
                "Token {} should be < 5 with top_k=5, but got token outside top-K",
                token
            );
        }

        // Verify we got multiple different tokens (not just argmax)
        assert!(
            counts.len() > 1,
            "Should sample multiple different tokens, got only {:?}",
            counts
        );
    }

    #[test]
    fn test_metal_sampler_distribution() {
        // Test that sampling produces roughly correct distribution
        let mut sampler = MetalSampler::new(10).unwrap();

        // Create logits: token 0 should have ~90% probability after softmax
        // log(0.9) ≈ -0.105, log(0.1/9) ≈ -4.5
        let mut logits_vec = vec![-4.5f32; 10];
        logits_vec[0] = -0.105; // ~90% probability
        let logits = Array::from_slice(&logits_vec, &[10]);
        logits.eval().unwrap();

        // Sample 1000 times with temperature=1.0, no filtering
        let mut count_0 = 0u32;
        for _ in 0..1000 {
            let token = sampler.sample(&logits, 1.0, 0, 1.0, 0.0).unwrap();
            if token == 0 {
                count_0 += 1;
            }
        }

        // Token 0 should appear 80-100% of the time (allowing for randomness)
        let ratio = count_0 as f32 / 1000.0;
        assert!(
            ratio > 0.75 && ratio < 1.0,
            "Token 0 should appear ~90% of time, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_metal_sampler_async_retains_logits_until_completion() {
        let mut sampler = MetalSampler::new(2048).unwrap();

        let mut logits_vec = vec![-100.0f32; 2048];
        logits_vec[777] = 42.0;
        let logits = Array::from_slice(&logits_vec, &[2048]);
        logits.eval().unwrap();

        sampler.argmax_async(&logits).unwrap();
        assert!(sampler.has_pending());

        drop(logits);

        // Churn additional allocations to increase the odds of catching any
        // stale-pointer bug in the zero-copy path.
        let scratch: Vec<Array> = (0..32)
            .map(|i| Array::from_slice(&vec![i as f32; 2048], &[2048]))
            .collect();
        assert_eq!(scratch.len(), 32);

        let token = sampler.get_token().unwrap();
        assert_eq!(token, 777);
        assert!(!sampler.has_pending());
    }

    #[test]
    fn test_metal_sampler_async_serializes_reused_output_buffer() {
        let mut sampler = MetalSampler::new(256).unwrap();

        let mut first_logits = vec![-10.0f32; 256];
        first_logits[11] = 10.0;
        let first = Array::from_slice(&first_logits, &[256]);
        first.eval().unwrap();

        let mut second_logits = vec![-10.0f32; 256];
        second_logits[173] = 10.0;
        let second = Array::from_slice(&second_logits, &[256]);
        second.eval().unwrap();

        sampler.argmax_async(&first).unwrap();
        assert!(sampler.has_pending());

        // A second async dispatch reuses the sampler's shared output buffer.
        // This must implicitly synchronize the first command before launching
        // the second one.
        sampler.argmax_async(&second).unwrap();
        assert!(sampler.has_pending());

        let token = sampler.get_token().unwrap();
        assert_eq!(token, 173);
        assert!(!sampler.has_pending());
    }
}
