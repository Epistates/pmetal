//! MLX-Metal fused optimizer.
//!
//! This module provides an optimizer that uses custom Metal kernels for maximum
//! throughput on parameters held as MLX arrays.
//!
//! # Architecture
//!
//! ```text
//! MLX Arrays (unified memory)
//!        │
//!        ▼
//! as_slice::<f32>() → copied into flat Metal buffers (params, grads)
//!        │
//!        ▼
//! Fused Metal Kernel (processes all params in single dispatch)
//!        │
//!        ▼
//! Array::from_f32_slice() → new parameter arrays, assigned back by name
//! ```
//!
//! # Performance
//!
//! By fusing all parameter updates into a single Metal dispatch, we eliminate:
//! - Per-parameter command buffer creation overhead
//! - Per-parameter GPU-CPU synchronization
//! - Multiple kernel launches

#![allow(unsafe_code)]

use std::rc::Rc;
use std::sync::Arc;

use pmetal_bridge::compat::{
    Array, Dtype, Exception,
    module::{FlattenedModuleParam, ModuleParameters, ModuleParametersExt},
    transforms,
};
use thiserror::Error;

use pmetal_metal::buffer::{BufferUsage, MetalBuffer};
use pmetal_metal::context::MetalContext;
use pmetal_metal::error::MetalError;
use pmetal_metal::kernels::{AdamWConfig, BatchedCommandBuffer, FusedAdamW, ParamInfo};

/// Error type for MLX-Metal optimizer operations.
#[derive(Error, Debug)]
pub enum MlxMetalOptimizerError {
    /// Metal error.
    #[error("Metal error: {0}")]
    Metal(#[from] MetalError),
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Parameter mismatch.
    #[error("Parameter count mismatch: expected {expected}, got {actual}")]
    ParamCountMismatch { expected: usize, actual: usize },
    /// Unsupported dtype.
    #[error("Unsupported dtype for parameter {name}: expected Float32")]
    UnsupportedDtype { name: String },
    /// Array not contiguous.
    #[error("Array not contiguous for parameter {name}")]
    NotContiguous { name: String },
}

/// Result type for MLX-Metal optimizer operations.
pub type MlxMetalOptimizerResult<T> = std::result::Result<T, MlxMetalOptimizerError>;

/// Learning rate schedule type.
///
/// Supported schedules:
/// - `Constant`: Fixed learning rate
/// - `CosineDecay`: Cosine annealing from init_lr to 0
/// - `CosineDecayWithWarmup`: Linear warmup followed by cosine decay
#[derive(Debug, Clone, Default)]
pub enum LrSchedule {
    /// Constant learning rate.
    #[default]
    Constant,
    /// Cosine decay: lr * 0.5 * (1 + cos(π * step / total_steps))
    CosineDecay {
        /// Total number of training steps.
        total_steps: u32,
    },
    /// Linear warmup followed by cosine decay.
    CosineDecayWithWarmup {
        /// Number of warmup steps (typically 5-10% of total).
        warmup_steps: u32,
        /// Total number of training steps.
        total_steps: u32,
        /// Initial learning rate during warmup start (default: 0.0).
        warmup_init: f32,
    },
}

impl LrSchedule {
    /// Compute the learning rate multiplier for the given step.
    pub fn get_lr_multiplier(&self, step: u32, base_lr: f32) -> f32 {
        match self {
            Self::Constant => base_lr,
            Self::CosineDecay { total_steps } => {
                if *total_steps == 0 {
                    return base_lr;
                }
                let progress = (step as f32) / (*total_steps as f32);
                let progress = progress.min(1.0);
                base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
            }
            Self::CosineDecayWithWarmup {
                warmup_steps,
                total_steps,
                warmup_init,
            } => {
                if step < *warmup_steps {
                    // Linear warmup: warmup_init -> base_lr
                    let warmup_progress = (step as f32) / (*warmup_steps as f32).max(1.0);
                    warmup_init + (base_lr - warmup_init) * warmup_progress
                } else {
                    // Cosine decay after warmup
                    let decay_steps = total_steps.saturating_sub(*warmup_steps);
                    if decay_steps == 0 {
                        return base_lr;
                    }
                    let progress = ((step - warmup_steps) as f32) / (decay_steps as f32);
                    let progress = progress.min(1.0);
                    base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
                }
            }
        }
    }
}

/// Configuration for the MLX-Metal fused optimizer.
#[derive(Debug, Clone)]
pub struct MlxMetalOptimizerConfig {
    /// Base learning rate.
    pub learning_rate: f32,
    /// AdamW beta1 (first moment decay).
    pub beta1: f32,
    /// AdamW beta2 (second moment decay).
    pub beta2: f32,
    /// Numerical stability constant.
    pub epsilon: f32,
    /// L2 regularization (weight decay).
    pub weight_decay: f32,
    /// Learning rate schedule.
    pub lr_schedule: LrSchedule,
    /// Enable gradient/parameter validation (NaN/Inf detection).
    /// This adds a small overhead (~1-2%) but catches numerical issues early.
    /// Recommended for debugging but can be disabled in production.
    pub validate_numerics: bool,
}

impl Default for MlxMetalOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            lr_schedule: LrSchedule::Constant,
            validate_numerics: false, // Disabled by default for production perf
        }
    }
}

/// Where each parameter sits in the flat buffers.
#[derive(Debug, Default)]
struct ParameterLayout {
    /// Parameter names, sorted, in buffer order.
    names: Vec<Rc<str>>,
    /// Element count of each parameter.
    sizes: Vec<usize>,
    /// Sum of `sizes`.
    total_elements: usize,
}

impl ParameterLayout {
    fn from_params(params: &FlattenedModuleParam) -> MlxMetalOptimizerResult<Self> {
        let mut names: Vec<_> = params.keys().cloned().collect();
        names.sort();
        let sizes: Vec<usize> = names.iter().map(|k| params[k].size()).collect();
        let total_elements: usize = sizes.iter().sum();
        // The kernel's ParamInfo offsets are u32.
        if u32::try_from(total_elements).is_err() {
            return Err(MlxMetalOptimizerError::Mlx(Exception::custom(format!(
                "{total_elements} trainable elements exceed the fused AdamW kernel's u32 offsets"
            ))));
        }
        Ok(Self {
            names,
            sizes,
            total_elements,
        })
    }
}

/// AdamW over every trainable parameter in one Metal dispatch.
///
/// The parameters and gradients are copied into flat f32 Metal buffers,
/// `fused_adamw` updates the parameters and both moments in place, and the
/// parameters are read back as new arrays. Like MLX's `optimizers.AdamW`
/// default, the update has no bias correction.
pub struct MlxMetalOptimizer {
    /// Metal context.
    ctx: Arc<MetalContext>,
    /// Fused AdamW kernel.
    adamw: FusedAdamW,
    /// Parameter info buffer.
    param_info: MetalBuffer<ParamInfo>,
    /// First moment buffer (owned) - for fused_step path.
    m_buffer: MetalBuffer<f32>,
    /// Second moment buffer (owned) - for fused_step path.
    v_buffer: MetalBuffer<f32>,
    /// Flattened parameter buffer for updates.
    flat_params: MetalBuffer<f32>,
    /// Flattened gradient buffer.
    flat_grads: MetalBuffer<f32>,
    /// Parameter layout.
    layout: ParameterLayout,
    /// Configuration.
    config: MlxMetalOptimizerConfig,
    /// Current step.
    step: u32,
    /// Whether the optimizer has been initialized with parameter shapes.
    initialized: bool,
}

impl MlxMetalOptimizer {
    /// Create a new MLX-Metal fused optimizer.
    ///
    /// Note: The optimizer must be initialized with the actual model parameters
    /// before use. Call `initialize()` with the model's trainable parameters.
    pub fn new(config: MlxMetalOptimizerConfig) -> MlxMetalOptimizerResult<Self> {
        let ctx = MetalContext::global()?;

        // Create placeholder buffers - will be resized on first use
        let flat_params = MetalBuffer::zeros(&ctx, 1, BufferUsage::Shared)?;
        let flat_grads = MetalBuffer::zeros(&ctx, 1, BufferUsage::Shared)?;
        let m_buffer = MetalBuffer::zeros(&ctx, 1, BufferUsage::Shared)?;
        let v_buffer = MetalBuffer::zeros(&ctx, 1, BufferUsage::Shared)?;
        let param_info = MetalBuffer::from_slice(
            &ctx,
            &[ParamInfo {
                offset: 0,
                size: 1,
                m_offset: 0,
                v_offset: 0,
            }],
            BufferUsage::Shared,
        )?;

        let adamw = FusedAdamW::new(ctx.clone(), &[1]);

        Ok(Self {
            ctx,
            adamw,
            param_info,
            m_buffer,
            v_buffer,
            flat_params,
            flat_grads,
            layout: ParameterLayout::default(),
            config,
            step: 0,
            initialized: false,
        })
    }

    /// Set total steps for learning rate scheduling.
    /// Call this before training starts if using a decay schedule.
    pub fn set_total_steps(&mut self, total_steps: u32) {
        match &mut self.config.lr_schedule {
            LrSchedule::CosineDecay { total_steps: ts } => *ts = total_steps,
            LrSchedule::CosineDecayWithWarmup {
                total_steps: ts, ..
            } => *ts = total_steps,
            LrSchedule::Constant => {}
        }
    }

    /// Get the current learning rate (after applying schedule).
    pub fn current_lr(&self) -> f32 {
        self.config
            .lr_schedule
            .get_lr_multiplier(self.step, self.config.learning_rate)
    }

    /// Initialize the optimizer with model parameters.
    ///
    /// This must be called once before the first `update()` call.
    /// It allocates optimizer state buffers sized to match the parameters.
    pub fn initialize<M: ModuleParameters>(&mut self, model: &M) -> MlxMetalOptimizerResult<()> {
        let params: FlattenedModuleParam = model.flatten_params();
        self.initialize_from_params(&params)
    }

    /// Initialize from flattened parameters.
    fn initialize_from_params(
        &mut self,
        params: &FlattenedModuleParam,
    ) -> MlxMetalOptimizerResult<()> {
        // Build parameter layout
        self.layout = ParameterLayout::from_params(params)?;

        // Validate all parameters are f32 and contiguous
        for name in &self.layout.names {
            let arr = &params[name];
            if arr.dtype_raw() != Dtype::Float32.as_i32() {
                return Err(MlxMetalOptimizerError::UnsupportedDtype {
                    name: name.to_string(),
                });
            }
        }

        // Allocate buffers for fused_step path
        let total = self.layout.total_elements;

        self.flat_params = MetalBuffer::zeros(&self.ctx, total, BufferUsage::Shared)?;
        self.flat_grads = MetalBuffer::zeros(&self.ctx, total, BufferUsage::Shared)?;
        self.m_buffer = MetalBuffer::zeros(&self.ctx, total, BufferUsage::Shared)?;
        self.v_buffer = MetalBuffer::zeros(&self.ctx, total, BufferUsage::Shared)?;

        // Build param info
        let param_info_vec = FusedAdamW::build_param_info(&self.layout.sizes);
        self.param_info = MetalBuffer::from_slice(&self.ctx, &param_info_vec, BufferUsage::Shared)?;

        // Create fused AdamW with correct sizes
        self.adamw = FusedAdamW::new(self.ctx.clone(), &self.layout.sizes);

        self.initialized = true;

        tracing::info!(
            "MlxMetalOptimizer initialized: {} params, {} total elements",
            self.layout.names.len(),
            total
        );

        Ok(())
    }

    /// Set learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    /// Get current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Get current step count.
    pub fn step_count(&self) -> u32 {
        self.step
    }

    /// Get total number of parameters.
    pub fn total_elements(&self) -> usize {
        self.layout.total_elements
    }

    /// Copy `arrays` into `dst` in layout order, one evaluation for all of
    /// them; a name with no array is zero-filled. Each array must have the
    /// element count the layout recorded for its name.
    fn copy_to_flat(
        layout: &ParameterLayout,
        dst: &mut MetalBuffer<f32>,
        arrays: &FlattenedModuleParam,
        what: &str,
    ) -> MlxMetalOptimizerResult<()> {
        transforms::eval(layout.names.iter().filter_map(|n| arrays.get(n)))?;
        let flat = dst.as_mut_slice();
        let mut offset = 0;
        for (name, &size) in layout.names.iter().zip(&layout.sizes) {
            let dst = &mut flat[offset..offset + size];
            match arrays.get(name) {
                Some(arr) => {
                    if arr.dtype_raw() != Dtype::Float32.as_i32() {
                        return Err(MlxMetalOptimizerError::UnsupportedDtype {
                            name: format!("{what} {name}"),
                        });
                    }
                    if arr.size() != size {
                        return Err(MlxMetalOptimizerError::Mlx(Exception::custom(format!(
                            "{what} `{name}` has {} elements, its parameter {size}",
                            arr.size()
                        ))));
                    }
                    dst.copy_from_slice(arr.as_slice::<f32>());
                }
                None => dst.fill(0.0),
            }
            offset += size;
        }
        Ok(())
    }

    fn copy_params_to_flat(
        &mut self,
        params: &FlattenedModuleParam,
    ) -> MlxMetalOptimizerResult<()> {
        Self::copy_to_flat(&self.layout, &mut self.flat_params, params, "parameter")
    }

    fn copy_grads_to_flat(&mut self, grads: &FlattenedModuleParam) -> MlxMetalOptimizerResult<()> {
        Self::copy_to_flat(&self.layout, &mut self.flat_grads, grads, "gradient")
    }

    /// Copy updated parameters back to MLX arrays.
    fn copy_flat_to_params(
        &self,
        params: &mut FlattenedModuleParam,
    ) -> MlxMetalOptimizerResult<()> {
        let flat_slice = self.flat_params.as_slice();
        let mut offset = 0;

        for name in &self.layout.names {
            let size = params[name].size();

            // Create new MLX array from the updated data
            let updated_data = &flat_slice[offset..offset + size];
            let shape = params[name].shape().to_vec();

            // Create new array with updated values
            let new_arr = Array::from_f32_slice(updated_data, &shape);

            // Replace the parameter
            params.insert(name.clone(), new_arr);

            offset += size;
        }

        Ok(())
    }

    /// One AdamW step over every parameter in one Metal dispatch.
    ///
    /// The parameters and gradients are copied into flat Metal buffers, the
    /// fused kernel updates the parameters and both moments in place, and
    /// the parameters come back as new arrays under the same names.
    pub fn fused_step(
        &mut self,
        params: &mut FlattenedModuleParam,
        grads: &FlattenedModuleParam,
    ) -> MlxMetalOptimizerResult<()> {
        if !self.initialized {
            self.initialize_from_params(params)?;
        }
        if params.len() != self.layout.names.len() {
            return Err(MlxMetalOptimizerError::ParamCountMismatch {
                expected: self.layout.names.len(),
                actual: params.len(),
            });
        }

        self.copy_params_to_flat(params)?;
        self.copy_grads_to_flat(grads)?;
        if self.config.validate_numerics {
            self.validate_gradients()?;
        }

        self.step += 1;
        let adamw_config = AdamWConfig {
            learning_rate: self.current_lr(),
            beta1: self.config.beta1,
            beta2: self.config.beta2,
            epsilon: self.config.epsilon,
            weight_decay: self.config.weight_decay,
            step: self.step,
        };
        let mut batch = BatchedCommandBuffer::new(self.ctx.clone())?;
        self.adamw.queue_update(
            &mut batch,
            &self.flat_params,
            &self.flat_grads,
            &self.m_buffer,
            &self.v_buffer,
            &self.param_info,
            &adamw_config,
        )?;
        batch.execute()?;

        self.copy_flat_to_params(params)?;
        Ok(())
    }

    /// Update a model's parameters with one [`Self::fused_step`].
    ///
    /// Every parameter the optimizer was initialized with must still be in
    /// the model. A parameter without a gradient is left where it is (its
    /// gradient is taken as zero); a step with no gradients at all is an
    /// error, since it means the backward pass never ran.
    pub fn update_model_fused<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: &FlattenedModuleParam,
    ) -> std::result::Result<(), Exception> {
        let mut params: FlattenedModuleParam = model.flatten_params();
        if !self.initialized {
            self.initialize_from_params(&params)
                .map_err(|e| Exception::custom(e.to_string()))?;
        }

        let missing_params: Vec<&str> = self
            .layout
            .names
            .iter()
            .filter(|n| !params.contains_key(*n))
            .map(|n| &**n)
            .collect();
        if !missing_params.is_empty() {
            return Err(Exception::custom(format!(
                "MlxMetalOptimizer: {} parameter(s) missing from the model since initialization: \
                 {missing_params:?}. The model structure changed; re-initialize the optimizer.",
                missing_params.len()
            )));
        }
        let missing_grads: Vec<&str> = self
            .layout
            .names
            .iter()
            .filter(|n| !gradients.contains_key(*n))
            .map(|n| &**n)
            .collect();
        if missing_grads.len() == self.layout.names.len() && !missing_grads.is_empty() {
            return Err(Exception::custom(
                "MlxMetalOptimizer: every gradient is missing; the backward pass was not run \
                 or its gradients were lost",
            ));
        }
        if !missing_grads.is_empty() {
            tracing::warn!(
                "MlxMetalOptimizer: {} gradient(s) missing, taken as zero: {:?}",
                missing_grads.len(),
                &missing_grads[..missing_grads.len().min(5)]
            );
        }

        self.fused_step(&mut params, gradients)
            .map_err(|e| Exception::custom(e.to_string()))?;

        let mut model_params = model.flatten_params_mut();
        for name in &self.layout.names {
            if let (Some(param), Some(updated)) = (model_params.get_mut(&**name), params.get(name))
            {
                **param = updated.clone();
            }
        }
        Ok(())
    }

    /// Fail the step if a gradient is NaN or infinite (`validate_numerics`).
    fn validate_gradients(&self) -> MlxMetalOptimizerResult<()> {
        let grads = &self.flat_grads.as_slice()[..self.layout.total_elements];
        let nan = grads.iter().any(|g| g.is_nan());
        let inf = grads.iter().any(|g| g.is_infinite());
        let issue = match (nan, inf) {
            (false, false) => return Ok(()),
            (true, true) => "NaN and Inf values",
            (true, false) => "NaN values",
            (false, true) => "Inf values",
        };
        Err(MlxMetalOptimizerError::Mlx(Exception::custom(format!(
            "Gradient validation failed: {issue} detected at step {}",
            self.step + 1
        ))))
    }
}

/// Builder for MlxMetalOptimizer.
#[derive(Debug, Clone)]
pub struct MlxMetalOptimizerBuilder {
    config: MlxMetalOptimizerConfig,
}

impl MlxMetalOptimizerBuilder {
    /// Create a new builder with the specified learning rate.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            config: MlxMetalOptimizerConfig {
                learning_rate,
                ..Default::default()
            },
        }
    }

    /// Set beta1 (first moment decay).
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.config.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment decay).
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.config.beta2 = beta2;
        self
    }

    /// Set epsilon.
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set weight decay.
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.config.weight_decay = weight_decay;
        self
    }

    /// Set learning rate schedule to cosine decay.
    pub fn cosine_decay(mut self, total_steps: u32) -> Self {
        self.config.lr_schedule = LrSchedule::CosineDecay { total_steps };
        self
    }

    /// Set learning rate schedule to cosine decay with warmup.
    /// SOTA: This is the recommended schedule; typical warmup is 5-10% of total steps.
    /// Typical warmup is 5-10% of total steps.
    pub fn cosine_decay_with_warmup(
        mut self,
        warmup_steps: u32,
        total_steps: u32,
        warmup_init: Option<f32>,
    ) -> Self {
        self.config.lr_schedule = LrSchedule::CosineDecayWithWarmup {
            warmup_steps,
            total_steps,
            warmup_init: warmup_init.unwrap_or(0.0),
        };
        self
    }

    /// Enable gradient/parameter validation (NaN/Inf detection).
    ///
    /// When enabled, the optimizer checks gradients for NaN/Inf values each step.
    /// This adds ~1-2% overhead but catches numerical issues early.
    /// Recommended for debugging, can be disabled in production.
    pub fn validate_numerics(mut self, enable: bool) -> Self {
        self.config.validate_numerics = enable;
        self
    }

    /// Build the optimizer.
    pub fn build(self) -> MlxMetalOptimizerResult<MlxMetalOptimizer> {
        MlxMetalOptimizer::new(self.config)
    }
}

/// Check if the MLX-Metal optimizer is available.
pub fn is_mlx_metal_optimizer_available() -> bool {
    MetalContext::global().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Weights {
        params: Vec<Array>,
    }
    pmetal_bridge::impl_module_params!(Weights; params);

    #[test]
    fn the_model_update_runs_the_metal_kernel() {
        // `update_model_fused` used to run an AdamW of MLX ops and leave the
        // kernel's moment buffers untouched.
        let (lr, b1, b2, eps, wd) = (0.1f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32);
        let mut model = Weights {
            params: vec![
                Array::from_f32_slice(&[1.0, 2.0], &[2]),
                Array::from_f32_slice(&[3.0, -1.0, 0.5], &[3]),
            ],
        };
        let gs = [vec![0.5f32, -0.25], vec![1.0, 2.0, -4.0]];
        let grads: FlattenedModuleParam = [
            (Rc::from("params.0"), Array::from_f32_slice(&gs[0], &[2])),
            (Rc::from("params.1"), Array::from_f32_slice(&gs[1], &[3])),
        ]
        .into();

        let mut opt = MlxMetalOptimizerBuilder::new(lr)
            .weight_decay(wd)
            .build()
            .unwrap();
        // AdamW by hand over both weights in name order, without bias
        // correction, as the kernel computes it.
        let g = gs.concat();
        let mut p = vec![1.0f32, 2.0, 3.0, -1.0, 0.5];
        let mut m = vec![0.0f32; 5];
        let mut v = vec![0.0f32; 5];
        for _ in 0..2 {
            opt.update_model_fused(&mut model, &grads).unwrap();
            for (((p, m), v), g) in p.iter_mut().zip(&mut m).zip(&mut v).zip(&g) {
                *m = b1 * *m + (1.0 - b1) * g;
                *v = b2 * *v + (1.0 - b2) * g * g;
                *p = *p * (1.0 - lr * wd) - lr * *m / (v.sqrt() + eps);
            }
        }
        let got: Vec<f32> = model
            .params
            .iter()
            .flat_map(|a| a.as_slice::<f32>().to_vec())
            .collect();
        for (a, b) in got.iter().zip(&p) {
            assert!((a - b).abs() < 1e-5, "weights {got:?}, by hand {p:?}");
        }
        // The kernel's own first moments.
        for (a, b) in opt.m_buffer.as_slice()[..5].iter().zip(&m) {
            assert!((a - b).abs() < 1e-6, "m buffer {a} vs {b}");
        }
        pmetal_bridge::check_unobserved_error().expect("no bridge op failed");
    }

    #[test]
    fn a_gradient_of_the_wrong_size_fails_the_step() {
        let mut model = Weights {
            params: vec![Array::from_f32_slice(&[1.0, 2.0], &[2])],
        };
        let grads: FlattenedModuleParam = [(
            Rc::from("params.0"),
            Array::from_f32_slice(&[1.0, 1.0, 1.0], &[3]),
        )]
        .into();
        let mut opt = MlxMetalOptimizerBuilder::new(0.1).build().unwrap();
        let err = opt.update_model_fused(&mut model, &grads).unwrap_err();
        assert!(err.what().contains("params.0"), "{}", err.what());
    }

    /// AdamW over LoRA-sized parameters (392 adapters of 16 x 1600 f32,
    /// about 10M elements). Measures speed: run through the GPU queue.
    #[test]
    #[ignore = "benchmark; run through gpuq"]
    fn bench_fused_adamw_over_lora_sized_params() {
        pmetal_bridge::compat::random::seed(1);
        let mut model = Weights {
            params: (0..392)
                .map(|_| pmetal_bridge::compat::random::normal(&[16, 1600], Dtype::Float32))
                .collect(),
        };
        let grads: FlattenedModuleParam = model
            .flatten_params()
            .into_iter()
            .map(|(k, p)| (k, p.multiply(&Array::from_f32(1e-3))))
            .collect();
        transforms::eval(model.params.iter().chain(grads.values())).unwrap();
        let mut opt = MlxMetalOptimizerBuilder::new(1e-4).build().unwrap();
        for _ in 0..3 {
            opt.update_model_fused(&mut model, &grads).unwrap();
        }
        let steps = 30;
        let t = std::time::Instant::now();
        for _ in 0..steps {
            opt.update_model_fused(&mut model, &grads).unwrap();
        }
        transforms::eval(model.params.iter()).unwrap();
        println!(
            "fused AdamW over {} elements: {:.2} ms/step",
            opt.total_elements(),
            t.elapsed().as_secs_f64() * 1e3 / steps as f64
        );
    }

    #[test]
    fn test_config_default() {
        let config = MlxMetalOptimizerConfig::default();
        assert!((config.learning_rate - 1e-4).abs() < 1e-8);
        assert!((config.beta1 - 0.9).abs() < 1e-6);
        assert!((config.beta2 - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_builder() {
        let opt = MlxMetalOptimizerBuilder::new(2e-4)
            .beta1(0.95)
            .weight_decay(0.1)
            .build();

        assert!(opt.is_ok());
        let opt = opt.unwrap();
        assert!((opt.config.learning_rate - 2e-4).abs() < 1e-8);
        assert!((opt.config.beta1 - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_availability() {
        assert!(is_mlx_metal_optimizer_available());
    }

    #[test]
    fn test_fused_step() {
        let mut opt = MlxMetalOptimizerBuilder::new(0.1)
            .weight_decay(0.0)
            .build()
            .unwrap();

        // Create simple parameters
        let mut params: FlattenedModuleParam = HashMap::new();
        params.insert(
            Rc::from("weight"),
            Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]),
        );

        // Create gradients (constant)
        let grads: FlattenedModuleParam = {
            let mut g = HashMap::new();
            g.insert(
                Rc::from("weight"),
                Array::from_f32_slice(&[0.1f32, 0.1, 0.1, 0.1], &[4]),
            );
            g
        };

        // Run one step
        opt.fused_step(&mut params, &grads).unwrap();

        // Verify params were updated (should decrease with positive gradients)
        let updated = params.get(&Rc::from("weight")).unwrap();
        updated.eval();

        let values: Vec<f32> = updated.as_slice().to_vec();
        assert!(values[0] < 1.0, "Weight should decrease, got {}", values[0]);
    }

    #[test]
    fn test_lr_schedule_constant() {
        let schedule = LrSchedule::Constant;
        let base_lr = 1e-4;

        // Constant schedule should always return base_lr
        assert!((schedule.get_lr_multiplier(0, base_lr) - base_lr).abs() < 1e-8);
        assert!((schedule.get_lr_multiplier(100, base_lr) - base_lr).abs() < 1e-8);
        assert!((schedule.get_lr_multiplier(1000, base_lr) - base_lr).abs() < 1e-8);
    }

    #[test]
    fn test_lr_schedule_cosine_decay() {
        let schedule = LrSchedule::CosineDecay { total_steps: 100 };
        let base_lr = 1e-4;

        // At step 0, should be full LR
        let lr_0 = schedule.get_lr_multiplier(0, base_lr);
        assert!((lr_0 - base_lr).abs() < 1e-8, "Step 0 should be base_lr");

        // At step 50 (halfway), should be ~0.5 * base_lr
        let lr_50 = schedule.get_lr_multiplier(50, base_lr);
        assert!(
            (lr_50 - 0.5 * base_lr).abs() < 1e-6,
            "Step 50 should be ~0.5 * base_lr"
        );

        // At step 100, should be ~0
        let lr_100 = schedule.get_lr_multiplier(100, base_lr);
        assert!(lr_100 < 1e-8, "Step 100 should be ~0");
    }

    #[test]
    fn test_lr_schedule_warmup() {
        let schedule = LrSchedule::CosineDecayWithWarmup {
            warmup_steps: 10,
            total_steps: 100,
            warmup_init: 0.0,
        };
        let base_lr = 1e-4;

        // At step 0, should be warmup_init (0.0)
        let lr_0 = schedule.get_lr_multiplier(0, base_lr);
        assert!(lr_0.abs() < 1e-8, "Step 0 should be warmup_init");

        // At step 5 (halfway through warmup), should be ~0.5 * base_lr
        let lr_5 = schedule.get_lr_multiplier(5, base_lr);
        assert!(
            (lr_5 - 0.5 * base_lr).abs() < 1e-6,
            "Step 5 should be ~0.5 * base_lr"
        );

        // At step 10 (end of warmup), should be full base_lr
        let lr_10 = schedule.get_lr_multiplier(10, base_lr);
        assert!((lr_10 - base_lr).abs() < 1e-6, "Step 10 should be base_lr");

        // After warmup, should decay
        let lr_55 = schedule.get_lr_multiplier(55, base_lr);
        assert!(lr_55 < base_lr, "Step 55 should be less than base_lr");
    }

    #[test]
    fn test_builder_with_schedule() {
        let opt = MlxMetalOptimizerBuilder::new(2e-4)
            .cosine_decay_with_warmup(10, 100, None)
            .build()
            .unwrap();

        // Verify schedule was set
        match &opt.config.lr_schedule {
            LrSchedule::CosineDecayWithWarmup {
                warmup_steps,
                total_steps,
                ..
            } => {
                assert_eq!(*warmup_steps, 10);
                assert_eq!(*total_steps, 100);
            }
            _ => panic!("Expected CosineDecayWithWarmup schedule"),
        }
    }
}
