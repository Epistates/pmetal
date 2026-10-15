#![allow(unsafe_code)]

//! Fused training kernels for maximum throughput.
//!
//! This module provides Metal kernels that eliminate GPU-CPU synchronization
//! overhead by batching all operations into a single command buffer per training step.
//!
//! # The Problem
//!
//! Standard execution pattern (what we had):
//! ```text
//! for each kernel:
//!     create_command_buffer()
//!     encode_kernel()
//!     commit()
//!     waitUntilCompleted()  // ← GPU-CPU sync, ~0.1ms overhead
//! ```
//!
//! With 100+ kernel dispatches per training step, this adds ~10-15ms overhead.
//!
//! # The Solution
//!
//! Batched execution pattern (what MLX's mx.compile does):
//! ```text
//! create_command_buffer()
//! for each kernel:
//!     encode_kernel()
//! commit()
//! waitUntilCompleted()  // ← Single sync at end
//! ```
//!
//! This module provides:
//! - [`FusedAdamW`]: All-parameter optimizer update in one dispatch
//! - [`BatchedCommandBuffer`]: Helper for batched execution

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::{CommandQueueExt, MetalContext};
use crate::error::{MetalError, Result};

// =============================================================================
// Batched Command Buffer
// =============================================================================

/// A command buffer that batches multiple kernel dispatches.
///
/// Instead of creating a new command buffer for each kernel and waiting
/// after each one, this struct accumulates dispatches and only waits
/// at the end.
///
/// # Usage
///
/// ```ignore
/// let mut batch = BatchedCommandBuffer::new(&ctx)?;
///
/// // Queue multiple operations (no waiting)
/// batch.queue_adamw_update(...)?;
/// batch.queue_gradient_clip(...)?;
///
/// // Execute all at once
/// batch.execute()?;  // Single GPU-CPU sync
/// ```
pub struct BatchedCommandBuffer {
    ctx: Arc<MetalContext>,
    command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    dispatch_count: usize,
}

impl BatchedCommandBuffer {
    /// Create a new batched command buffer.
    pub fn new(ctx: Arc<MetalContext>) -> Result<Self> {
        let command_buffer = ctx.command_queue().new_command_buffer()?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        Ok(Self {
            ctx,
            command_buffer: Some(command_buffer),
            encoder: Some(encoder),
            dispatch_count: 0,
        })
    }

    /// Get the context.
    pub fn context(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Get the current encoder for adding dispatches.
    ///
    /// Returns None if the buffer has already been executed.
    pub fn encoder(&self) -> Option<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder.as_ref().map(|e| e.as_ref())
    }

    /// Get mutable encoder access (crate-internal use for queue_* methods).
    pub(crate) fn encoder_mut(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder
            .as_ref()
            .map(|e| e.as_ref())
            .ok_or(MetalError::CommandBufferCreation)
    }

    /// Get the number of dispatches queued.
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    /// Increment dispatch count.
    pub fn add_dispatch(&mut self) {
        self.dispatch_count += 1;
    }

    /// Execute all queued operations.
    ///
    /// This ends encoding, commits the command buffer, and waits for completion.
    /// After this call, the buffer cannot be reused.
    pub fn execute(mut self) -> Result<()> {
        // End encoding
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }

        // Commit and wait
        if let Some(command_buffer) = self.command_buffer.take() {
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            if let Some(error) = command_buffer.error() {
                return Err(MetalError::ExecutionFailed(error.to_string()));
            }
        }

        tracing::trace!(
            "BatchedCommandBuffer executed {} dispatches",
            self.dispatch_count
        );
        Ok(())
    }

    /// Execute without waiting (for async patterns).
    ///
    /// Returns a handle that can be used to wait later.
    pub fn execute_async(mut self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }

        let command_buffer = self
            .command_buffer
            .take()
            .ok_or(MetalError::CommandBufferCreation)?;

        command_buffer.commit();
        Ok(command_buffer)
    }
}

impl Drop for BatchedCommandBuffer {
    fn drop(&mut self) {
        // Ensure encoder is properly ended even if execute() wasn't called.
        // Metal requires endEncoding() before the encoder is deallocated.
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        // Note: We don't commit or wait - just clean up the encoder.
        // The command buffer will be dropped automatically without execution.
    }
}

// =============================================================================
// AdamW Configuration
// =============================================================================

/// AdamW optimizer hyperparameters for one step.
///
/// Every AdamW in this crate (the Metal kernels and the CPU update in
/// [`crate::accelerate`]) applies them as `torch.optim.AdamW` does: decoupled
/// weight decay on the parameter before the update, and bias correction of
/// both moments.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    /// Learning rate (after scheduling).
    pub learning_rate: f32,
    /// First moment decay (default 0.9).
    pub beta1: f32,
    /// Second moment decay (default 0.999).
    pub beta2: f32,
    /// Numerical stability constant (default 1e-8).
    pub epsilon: f32,
    /// Decoupled weight decay (default 0.01).
    pub weight_decay: f32,
    /// This step's number, from 1 (drives the bias correction).
    pub step: u32,
}

impl AdamWConfig {
    /// The per-step scalars the kernels read, computed as `torch.optim.AdamW`
    /// computes them: in f64, each rounded to f32 once.
    ///
    /// Fails for step 0: the bias correction is undefined before the first
    /// step.
    pub fn scalars(&self) -> Result<AdamWScalars> {
        if self.step == 0 {
            return Err(crate::error::MetalError::InvalidConfig(
                "AdamW steps are 1-based; step 0 has no bias correction".into(),
            ));
        }
        let (lr, wd) = (f64::from(self.learning_rate), f64::from(self.weight_decay));
        let (b1, b2) = (f64::from(self.beta1), f64::from(self.beta2));
        let t = f64::from(self.step);
        Ok(AdamWScalars {
            decay: (1.0 - lr * wd) as f32,
            one_minus_beta1: (1.0 - b1) as f32,
            beta2: self.beta2,
            one_minus_beta2: (1.0 - b2) as f32,
            step_size: (lr / (1.0 - b1.powf(t))) as f32,
            bias_correction2_sqrt: (1.0 - b2.powf(t)).sqrt() as f32,
            epsilon: self.epsilon,
        })
    }
}

/// One AdamW step's scalars, as the kernels read them.
///
/// The update is, per element:
/// ```text
/// p = p·decay
/// m = m + (1−β₁)·(g − m)
/// v = β₂·v + (1−β₂)·g²
/// p = p − step_size · m / (√v / bias_correction2_sqrt + ε)
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct AdamWScalars {
    /// `1 − lr·λ`.
    pub decay: f32,
    /// `1 − β₁`.
    pub one_minus_beta1: f32,
    /// `β₂`.
    pub beta2: f32,
    /// `1 − β₂`.
    pub one_minus_beta2: f32,
    /// `lr / (1 − β₁ᵗ)`.
    pub step_size: f32,
    /// `√(1 − β₂ᵗ)`.
    pub bias_correction2_sqrt: f32,
    /// `ε`.
    pub epsilon: f32,
}

impl AdamWScalars {
    /// Scalars that leave a parameter and both its moments exactly as they
    /// are: `p·1 − 0·(…)`, `m + 0·(g − m)`, `1·v + 0·g²`. A parameter without
    /// a gradient this step takes them, as `torch.optim` skips a parameter
    /// whose `.grad` is `None`.
    pub const UNCHANGED: Self = Self {
        decay: 1.0,
        one_minus_beta1: 0.0,
        beta2: 1.0,
        one_minus_beta2: 0.0,
        step_size: 0.0,
        bias_correction2_sqrt: 1.0,
        epsilon: 1.0,
    };
}

// `AdamWScalars` in `metal/fused_training.metal`.
crate::msl_layout::msl_layout!(AdamWScalars, size = 28, {
    decay: 0,
    one_minus_beta1: 4,
    beta2: 8,
    one_minus_beta2: 12,
    step_size: 16,
    bias_correction2_sqrt: 20,
    epsilon: 24,
});

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step: 1,
        }
    }
}

/// Parameter metadata for batched processing.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ParamInfo {
    /// Offset into the flattened parameter buffer.
    pub offset: u32,
    /// Number of elements in this parameter.
    pub size: u32,
    /// Offset into first moment buffer.
    pub m_offset: u32,
    /// Offset into second moment buffer.
    pub v_offset: u32,
    /// Which of the step's [`AdamWScalars`] this parameter is updated with
    /// (its parameter group); see [`FusedAdamW::queue_update`].
    pub group: u32,
}

// `ParamInfo` in `metal/fused_training.metal`.
crate::msl_layout::msl_layout!(ParamInfo, size = 20, {
    offset: 0,
    size: 4,
    m_offset: 8,
    v_offset: 12,
    group: 16,
});

// =============================================================================
// Fused AdamW Optimizer
// =============================================================================

/// Elements one threadgroup updates (`ADAMW_CHUNK` in the shader).
const ADAMW_CHUNK: usize = 4096;

/// Threads per threadgroup of the update kernel.
const ADAMW_THREADS: usize = 256;

/// A run of up to [`ADAMW_CHUNK`] elements of one parameter: the work of one
/// threadgroup.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
struct AdamWChunk {
    /// Index into the parameter table.
    param: u32,
    /// First element of the run, within the parameter.
    start: u32,
}

// `AdamWChunk` in `metal/fused_training.metal`.
crate::msl_layout::msl_layout!(AdamWChunk, size = 8, {
    param: 0,
    start: 4,
});

/// The runs of elements that cover every parameter exactly once.
fn adamw_chunks(param_sizes: &[usize]) -> Vec<AdamWChunk> {
    param_sizes
        .iter()
        .enumerate()
        .flat_map(|(param, &size)| {
            (0..size.div_ceil(ADAMW_CHUNK)).map(move |run| AdamWChunk {
                param: param as u32,
                start: (run * ADAMW_CHUNK) as u32,
            })
        })
        .collect()
}

/// Fused AdamW optimizer that updates all parameters in a single dispatch.
///
/// Instead of launching N kernel dispatches for N parameters, this updates
/// the whole flattened parameter space in one dispatch: a table of fixed-size
/// runs of elements gives each threadgroup its share, so one huge tensor and
/// hundreds of small ones cost the same per element.
///
/// # Memory Layout
///
/// Parameters, gradients, and optimizer state are stored in flattened buffers:
/// - `params`: All model parameters concatenated
/// - `grads`: All gradients concatenated (same layout as params)
/// - `m`: First moment estimates (same layout)
/// - `v`: Second moment estimates (same layout)
/// - `param_info`: Metadata describing offset and size of each parameter
pub struct FusedAdamW {
    ctx: Arc<MetalContext>,
    /// Size of each parameter, as the table passed to [`Self::queue_update`]
    /// must say.
    sizes: Vec<usize>,
    /// Total elements across all parameters.
    total_elements: usize,
    /// The runs of elements, one per threadgroup.
    chunks: MetalBuffer<AdamWChunk>,
    /// Threadgroups one update launches.
    num_chunks: usize,
}

impl FusedAdamW {
    /// Create a new fused AdamW optimizer.
    ///
    /// # Arguments
    /// * `ctx` - Metal context
    /// * `param_sizes` - Sizes of each parameter tensor
    ///
    /// Fails if the parameters hold more elements than a 32-bit index
    /// reaches.
    pub fn new(ctx: Arc<MetalContext>, param_sizes: &[usize]) -> Result<Self> {
        let total_elements: usize = param_sizes.iter().sum();
        if u32::try_from(total_elements).is_err() {
            return Err(MetalError::InvalidConfig(format!(
                "fused AdamW indexes elements with 32 bits, but the parameters hold {total_elements}"
            )));
        }
        let chunks = adamw_chunks(param_sizes);
        let num_chunks = chunks.len();
        // A zero-length parameter has no chunks; an empty model has no work.
        let chunks = if chunks.is_empty() {
            MetalBuffer::zeros(&ctx, 1, BufferUsage::Shared)?
        } else {
            MetalBuffer::from_slice(&ctx, &chunks, BufferUsage::Shared)?
        };
        Ok(Self {
            ctx,
            sizes: param_sizes.to_vec(),
            total_elements,
            chunks,
            num_chunks,
        })
    }

    /// Build parameter info metadata from sizes, every parameter in group 0.
    pub fn build_param_info(param_sizes: &[usize]) -> Vec<ParamInfo> {
        let mut offset = 0u32;
        param_sizes
            .iter()
            .map(|&size| {
                let info = ParamInfo {
                    offset,
                    size: size as u32,
                    m_offset: offset,
                    v_offset: offset,
                    group: 0,
                };
                offset += size as u32;
                info
            })
            .collect()
    }

    /// The most groups one dispatch takes: their scalars go to the kernel
    /// with `setBytes`, which carries at most 4 KiB.
    pub const MAX_GROUPS: usize = 4096 / std::mem::size_of::<AdamWScalars>();

    /// Check that `param_info` describes the parameters this optimizer was
    /// built for, and that every one lies inside its buffers.
    fn validate(
        &self,
        buffers: [(&'static str, usize); 4],
        param_info: &MetalBuffer<ParamInfo>,
        groups: usize,
    ) -> Result<()> {
        let infos = param_info.as_slice();
        if infos.len() < self.sizes.len() {
            return Err(MetalError::InvalidConfig(format!(
                "fused AdamW over {} parameters given metadata for {}",
                self.sizes.len(),
                infos.len()
            )));
        }
        for (index, (info, &size)) in infos.iter().zip(&self.sizes).enumerate() {
            if info.size as usize != size {
                return Err(MetalError::InvalidConfig(format!(
                    "fused AdamW parameter {index} was built with {size} elements, \
                     but its metadata says {}",
                    info.size
                )));
            }
            if info.group as usize >= groups {
                return Err(MetalError::InvalidConfig(format!(
                    "a parameter at offset {} is in group {}, but the step has {groups} groups",
                    info.offset, info.group
                )));
            }
            let [
                (p_name, p_len),
                (g_name, g_len),
                (m_name, m_len),
                (v_name, v_len),
            ] = buffers;
            let spans = [
                (p_name, p_len, info.offset),
                (g_name, g_len, info.offset),
                (m_name, m_len, info.m_offset),
                (v_name, v_len, info.v_offset),
            ];
            for (name, len, start) in spans {
                let end = start as usize + size;
                if end > len {
                    return Err(MetalError::InvalidConfig(format!(
                        "fused AdamW parameter {index} spans [{start}..{end}) of the {name} \
                         buffer, which holds {len}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Queue an AdamW update into a batched command buffer.
    ///
    /// Parameter `i` is updated with `groups[param_info[i].group]`: one set
    /// of scalars per parameter group, so each group trains at its own
    /// learning rate and weight decay. Fails if a parameter names a group
    /// past the end of `groups`, if there are more than
    /// [`Self::MAX_GROUPS`], or if `param_info` or a buffer does not fit the
    /// parameters this optimizer was built for.
    ///
    /// This does NOT execute immediately - call `batch.execute()` after
    /// queueing all operations.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_update(
        &self,
        batch: &mut BatchedCommandBuffer,
        params: &MetalBuffer<f32>,
        grads: &MetalBuffer<f32>,
        m: &MetalBuffer<f32>,
        v: &MetalBuffer<f32>,
        param_info: &MetalBuffer<ParamInfo>,
        groups: &[AdamWScalars],
    ) -> Result<()> {
        if groups.is_empty() || groups.len() > Self::MAX_GROUPS {
            return Err(MetalError::InvalidConfig(format!(
                "fused AdamW takes 1 to {} parameter groups, got {}",
                Self::MAX_GROUPS,
                groups.len()
            )));
        }
        self.validate(
            [
                ("params", params.len()),
                ("grads", grads.len()),
                ("first moment", m.len()),
                ("second moment", v.len()),
            ],
            param_info,
            groups.len(),
        )?;
        if self.num_chunks == 0 {
            return Ok(());
        }
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "fused_adamw_update", None)?
        };

        let encoder = batch.encoder_mut()?;
        encoder.setComputePipelineState(&pipeline);

        // SAFETY: the buffers outlive the command buffer's execution, every
        // parameter's span was validated against them, and the group scalars
        // are copied by `setBytes`.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(params.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(grads.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(m.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(param_info.metal_buffer()), 0, 4);

            let groups_ptr = NonNull::from(groups).cast();
            encoder.setBytes_length_atIndex(groups_ptr, std::mem::size_of_val(groups), 5);

            encoder.setBuffer_offset_atIndex(Some(self.chunks.metal_buffer()), 0, 6);
        }

        // One threadgroup per run of elements.
        let grid_size = MTLSize {
            width: self.num_chunks,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: ADAMW_THREADS,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Execute AdamW update immediately (for compatibility).
    ///
    /// This creates a dedicated command buffer and waits for completion.
    /// Prefer using `queue_update` with `BatchedCommandBuffer` for better performance.
    pub fn execute_update(
        &self,
        params: &MetalBuffer<f32>,
        grads: &MetalBuffer<f32>,
        m: &MetalBuffer<f32>,
        v: &MetalBuffer<f32>,
        param_info: &MetalBuffer<ParamInfo>,
        groups: &[AdamWScalars],
    ) -> Result<()> {
        let mut batch = BatchedCommandBuffer::new(self.ctx.clone())?;
        self.queue_update(&mut batch, params, grads, m, v, param_info, groups)?;
        batch.execute()
    }

    /// Get total number of elements.
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::BufferUsage;

    fn create_test_context() -> Arc<MetalContext> {
        Arc::new(MetalContext::new().expect("Failed to create Metal context"))
    }

    #[test]
    fn test_adamw_config_default() {
        let config = AdamWConfig::default();
        assert!((config.beta1 - 0.9).abs() < 1e-6);
        assert!((config.beta2 - 0.999).abs() < 1e-6);
        assert!((config.epsilon - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn test_param_info_building() {
        let sizes = vec![100, 200, 50];
        let info = FusedAdamW::build_param_info(&sizes);

        assert_eq!(info.len(), 3);
        assert_eq!(info[0].offset, 0);
        assert_eq!(info[0].size, 100);
        assert_eq!(info[1].offset, 100);
        assert_eq!(info[1].size, 200);
        assert_eq!(info[2].offset, 300);
        assert_eq!(info[2].size, 50);
    }

    #[test]
    fn test_fused_adamw_creation() {
        let ctx = create_test_context();
        let param_sizes = vec![1024, 2048, 512];
        let adamw = FusedAdamW::new(ctx, &param_sizes).unwrap();

        assert_eq!(adamw.sizes, param_sizes);
        assert_eq!(adamw.total_elements, 1024 + 2048 + 512);
        // One threadgroup per parameter: each fits in one run.
        assert_eq!(adamw.num_chunks, 3);
    }

    #[test]
    fn test_batched_command_buffer_creation() {
        let ctx = create_test_context();
        let batch = BatchedCommandBuffer::new(ctx);
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert_eq!(batch.dispatch_count(), 0);
        assert!(batch.encoder().is_some());
    }

    /// Every element of every parameter belongs to exactly one run, however
    /// the sizes fall against the run length.
    #[test]
    fn the_chunk_table_covers_every_element_once() {
        let sizes = [
            0,
            1,
            ADAMW_CHUNK - 1,
            ADAMW_CHUNK,
            ADAMW_CHUNK + 1,
            3 * ADAMW_CHUNK,
            10_000,
        ];
        let chunks = adamw_chunks(&sizes);
        let mut covered: Vec<Vec<u32>> = sizes.iter().map(|&n| vec![0; n]).collect();
        for chunk in &chunks {
            let size = sizes[chunk.param as usize];
            let end = (chunk.start as usize + ADAMW_CHUNK).min(size);
            assert!(
                (chunk.start as usize) < size,
                "a run starts past its parameter"
            );
            for hit in &mut covered[chunk.param as usize][chunk.start as usize..end] {
                *hit += 1;
            }
        }
        assert!(covered.iter().flatten().all(|&hits| hits == 1));
    }

    /// Deterministic values in [-1, 1).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// `torch.optim.AdamW`, in f64: decoupled decay on the old parameter,
    /// bias-corrected moments, `eps` added after the second moment's root.
    #[allow(clippy::too_many_arguments)]
    fn adamw_f64(p: &mut [f64], m: &mut [f64], v: &mut [f64], g: &[f32], config: &AdamWConfig) {
        let (lr, wd) = (
            f64::from(config.learning_rate),
            f64::from(config.weight_decay),
        );
        let (b1, b2) = (f64::from(config.beta1), f64::from(config.beta2));
        let t = f64::from(config.step);
        let (bc1, bc2) = (1.0 - b1.powf(t), 1.0 - b2.powf(t));
        for i in 0..p.len() {
            let g = f64::from(g[i]);
            p[i] *= 1.0 - lr * wd;
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let denom = (v[i] / bc2).sqrt() + f64::from(config.epsilon);
            p[i] -= lr / bc1 * m[i] / denom;
        }
    }

    /// Several steps over parameters whose sizes straddle the run length,
    /// in three groups at their own rates and decay, against the float64
    /// reference.
    #[test]
    fn fused_adamw_matches_the_float64_reference_over_steps() {
        let ctx = create_test_context();
        let sizes = [1usize, 4095, 4096, 4097, 10_000, 100_003, 7];
        let total: usize = sizes.iter().sum();
        let adamw = FusedAdamW::new(ctx.clone(), &sizes).unwrap();
        let usage = BufferUsage::Shared;

        let mut rng = Lcg(11);
        let init: Vec<f32> = (0..total).map(|_| rng.next()).collect();
        let params = MetalBuffer::from_slice(&ctx, &init, usage).unwrap();
        let mut grads = MetalBuffer::<f32>::zeros(&ctx, total, usage).unwrap();
        let m = MetalBuffer::<f32>::zeros(&ctx, total, usage).unwrap();
        let v = MetalBuffer::<f32>::zeros(&ctx, total, usage).unwrap();

        // Groups 0, 1, 2 in rotation.
        let mut infos = FusedAdamW::build_param_info(&sizes);
        for (i, info) in infos.iter_mut().enumerate() {
            info.group = (i % 3) as u32;
        }
        let info_buf = MetalBuffer::from_slice(&ctx, &infos, usage).unwrap();

        let mut ref_p: Vec<f64> = init.iter().map(|&x| f64::from(x)).collect();
        let mut ref_m = vec![0.0f64; total];
        let mut ref_v = vec![0.0f64; total];
        let base = AdamWConfig {
            learning_rate: 3e-3,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.1,
            step: 1,
        };
        for step in 1..=4u32 {
            let configs = [
                AdamWConfig { step, ..base },
                AdamWConfig {
                    step,
                    learning_rate: 1e-2,
                    weight_decay: 0.0,
                    ..base
                },
                AdamWConfig {
                    step,
                    learning_rate: 5e-4,
                    weight_decay: 0.5,
                    ..base
                },
            ];
            let g: Vec<f32> = (0..total).map(|_| rng.next() * 0.3).collect();
            grads.as_mut_slice().copy_from_slice(&g);
            let scalars = configs.map(|c| c.scalars().unwrap());
            adamw
                .execute_update(&params, &grads, &m, &v, &info_buf, &scalars)
                .unwrap();
            for (info, &size) in infos.iter().zip(&sizes) {
                let range = info.offset as usize..info.offset as usize + size;
                adamw_f64(
                    &mut ref_p[range.clone()],
                    &mut ref_m[range.clone()],
                    &mut ref_v[range.clone()],
                    &g[range],
                    &configs[info.group as usize],
                );
            }
        }

        let got = params.as_slice();
        let mut worst = 0.0f64;
        for i in 0..total {
            let err = (f64::from(got[i]) - ref_p[i]).abs();
            worst = worst.max(err);
            assert!(
                err <= 2e-6 * (1.0 + ref_p[i].abs()),
                "element {i}: got {}, want {}, err {err:e}",
                got[i],
                ref_p[i]
            );
        }
        let (gm, gv) = (m.as_slice(), v.as_slice());
        for i in 0..total {
            assert!((f64::from(gm[i]) - ref_m[i]).abs() <= 1e-6 * (1.0 + ref_m[i].abs()));
            assert!((f64::from(gv[i]) - ref_v[i]).abs() <= 1e-6 * (1.0 + ref_v[i].abs()));
        }
        println!("worst parameter error {worst:e}");
    }

    #[test]
    fn a_step_that_does_not_fit_the_optimizer_is_refused() {
        let ctx = create_test_context();
        let usage = BufferUsage::Shared;
        let sizes = [100usize, 5000];
        let total: usize = sizes.iter().sum();
        let adamw = FusedAdamW::new(ctx.clone(), &sizes).unwrap();
        let buf = || MetalBuffer::<f32>::zeros(&ctx, total, usage).unwrap();
        let (params, grads, m, v) = (buf(), buf(), buf(), buf());
        let infos = FusedAdamW::build_param_info(&sizes);
        let info_buf = MetalBuffer::from_slice(&ctx, &infos, usage).unwrap();
        let groups = [AdamWConfig::default().scalars().unwrap()];

        adamw
            .execute_update(&params, &grads, &m, &v, &info_buf, &groups)
            .unwrap();

        // Metadata for other sizes.
        let other = FusedAdamW::build_param_info(&[100, 4000]);
        let other_buf = MetalBuffer::from_slice(&ctx, &other, usage).unwrap();
        assert!(
            adamw
                .execute_update(&params, &grads, &m, &v, &other_buf, &groups)
                .is_err()
        );

        // A buffer shorter than the parameters it must hold.
        let short = MetalBuffer::<f32>::zeros(&ctx, total - 1, usage).unwrap();
        for case in 0..4 {
            let mut bufs = [&params, &grads, &m, &v];
            bufs[case] = &short;
            let err = adamw
                .execute_update(bufs[0], bufs[1], bufs[2], bufs[3], &info_buf, &groups)
                .unwrap_err();
            assert!(err.to_string().contains("buffer, which holds"), "{err}");
        }

        // No groups, and a group the step does not carry.
        assert!(
            adamw
                .execute_update(&params, &grads, &m, &v, &info_buf, &[])
                .is_err()
        );
        let mut grouped = infos.clone();
        grouped[1].group = 1;
        let grouped_buf = MetalBuffer::from_slice(&ctx, &grouped, usage).unwrap();
        assert!(
            adamw
                .execute_update(&params, &grads, &m, &v, &grouped_buf, &groups)
                .is_err()
        );
    }

    /// An optimizer over no parameters has nothing to dispatch.
    #[test]
    fn an_empty_model_is_a_no_op() {
        let ctx = create_test_context();
        let usage = BufferUsage::Shared;
        let adamw = FusedAdamW::new(ctx.clone(), &[]).unwrap();
        let buf = MetalBuffer::<f32>::zeros(&ctx, 1, usage).unwrap();
        let info = MetalBuffer::<ParamInfo>::zeros(&ctx, 1, usage).unwrap();
        let groups = [AdamWConfig::default().scalars().unwrap()];
        adamw
            .execute_update(&buf, &buf, &buf, &buf, &info, &groups)
            .unwrap();
    }

    #[test]
    fn test_fused_adamw_execution() {
        let ctx = create_test_context();

        // Simulate 3 parameters with sizes matching typical LoRA
        let param_sizes = vec![1024, 2048, 512];
        let total_elements: usize = param_sizes.iter().sum();

        let adamw = FusedAdamW::new(ctx.clone(), &param_sizes).unwrap();

        // Create buffers
        // Initialize params to 1.0, grads to 0.1
        let params_data: Vec<f32> = vec![1.0; total_elements];
        let grads_data: Vec<f32> = vec![0.1; total_elements];

        let params = MetalBuffer::from_slice(&ctx, &params_data, BufferUsage::Shared)
            .expect("Failed to create params buffer");
        let grads = MetalBuffer::from_slice(&ctx, &grads_data, BufferUsage::Shared)
            .expect("Failed to create grads buffer");
        let m = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create m buffer");
        let v = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create v buffer");

        // One parameter per group: the second at its own rate without
        // decay, the third left as it is.
        let mut param_info_vec = FusedAdamW::build_param_info(&param_sizes);
        param_info_vec[1].group = 1;
        param_info_vec[2].group = 2;
        let param_info = MetalBuffer::from_slice(&ctx, &param_info_vec, BufferUsage::Shared)
            .expect("Failed to create param_info buffer");

        let config = AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step: 1,
        };
        let second = AdamWConfig {
            learning_rate: 0.004,
            weight_decay: 0.0,
            ..config
        };
        let groups = [
            config.scalars().unwrap(),
            second.scalars().unwrap(),
            AdamWScalars::UNCHANGED,
        ];

        adamw
            .execute_update(&params, &grads, &m, &v, &param_info, &groups)
            .expect("AdamW update failed");

        // The first bias-corrected step moves each weight by lr·sign(g),
        // after its decay: 1·(1 − 0.001·0.01) − 0.001.
        let (p_out, m_out) = (params.as_slice(), m.as_slice());
        let wants = [
            ((1.0 - 0.001 * 0.01) - 0.001, 0.01),
            (1.0 - 0.004, 0.01),
            (1.0, 0.0),
        ];
        let mut offset = 0;
        for (&size, &(want_p, want_m)) in param_sizes.iter().zip(&wants) {
            for i in offset..offset + size {
                let (p, m) = (p_out[i], m_out[i]);
                assert!((p - want_p).abs() < 1e-6, "param {i}: {p}, want {want_p}");
                assert!((m - want_m).abs() < 1e-8, "m {i}: {m}, want {want_m}");
            }
            offset += size;
        }
        let untouched = total_elements - param_sizes[2];
        assert!(v.as_slice()[untouched..].iter().all(|&v| v == 0.0));

        // A group past the end of the step's scalars is refused.
        assert!(
            adamw
                .execute_update(&params, &grads, &m, &v, &param_info, &groups[..2])
                .is_err()
        );
    }

    #[test]
    fn adamw_step_zero_is_refused() {
        let config = AdamWConfig {
            step: 0,
            ..Default::default()
        };
        assert!(config.scalars().is_err());
    }
}
