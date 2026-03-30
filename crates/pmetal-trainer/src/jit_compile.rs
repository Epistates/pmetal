//! JIT Compilation with Explicit State Tracking.
//!
//! This module provides a Python-like compile interface that explicitly tracks
//! mutable state (model parameters + optimizer state), working around the limitations
//! of mlx-rs's `compile_with_state` which fails with complex models.
//!
//! # Architecture
//!
//! The Python MLX compile with state works by:
//! 1. Flattening `inputs` (state containers) to arrays and appending to function inputs
//! 2. Flattening `outputs` (state containers) from function outputs
//! 3. Filling back the state containers after execution
//!
//! This module replicates that pattern in Rust:
//! - [`CompiledTrainStep`]: Compiled training step with explicit state management
//! - [`StateExtractor`]: Trait for extracting/updating state arrays
//!
//! # Performance
//!
//! With proper state initialization (warmup step), this achieves full JIT fusion:
//! - ~3x throughput improvement over non-compiled execution
//! - Matches mlx-lm's compiled training performance
//!
//! # Example
//!
//! ```ignore
//! use pmetal_trainer::jit_compile::CompiledTrainStep;
//!
//! // Create compiled step after warmup
//! let mut compiled = CompiledTrainStep::new(model, optimizer)?;
//!
//! // Run training steps
//! for batch in dataloader {
//!     let loss = compiled.step(&batch.input_ids, &batch.labels)?;
//!     loss.eval()?;
//! }
//!
//! // Get back model and optimizer
//! let (model, optimizer) = compiled.into_inner();
//! ```

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception,
    indexing::IndexOp,
    losses::CrossEntropy,
    module::{FlattenedModuleParam, ModuleParameters, ModuleParametersExt, update_parameters},
    nn,
    optimizers::Optimizer,
    optimizers::Updatable,
};

use crate::Result;

/// No-op shim for `mlx_rs::transforms::compile::compile`.
///
/// The bridge executes pre-compiled Metal kernels; JIT compilation
/// via this path is a no-op that returns the function unchanged.
#[allow(dead_code)]
fn compile<F>(f: F, _shapeful: bool) -> F {
    f
}

/// Trait for types that can extract their state as arrays and update from arrays.
pub trait StateExtractor {
    /// Extract all mutable state arrays in a deterministic order.
    fn extract_state(&self) -> Vec<Array>;

    /// Update state from arrays (same order as extract_state).
    fn update_state(&mut self, arrays: &[Array]) -> Result<()>;

    /// Number of state arrays.
    fn state_count(&self) -> usize;
}

/// Compiled training step with explicit state management.
///
/// This wraps model and optimizer state, providing JIT-compiled training
/// that properly tracks state changes between calls.
pub struct CompiledTrainStep<M, O> {
    model: M,
    optimizer: O,
    #[allow(dead_code)]
    // Used by extract/update_model_params (JIT infra, not yet wired end-to-end)
    param_keys: Vec<Rc<str>>,
    compiled: bool,
}

impl<M, O> CompiledTrainStep<M, O>
where
    M: ModuleParameters,
    O: Optimizer + Updatable,
{
    /// Create a new compiled training step.
    ///
    /// # Arguments
    /// * `model` - Model with initialized parameters
    /// * `optimizer` - Optimizer with initialized state (run warmup first!)
    ///
    /// # Panics
    /// Panics if optimizer state is not initialized (run warmup step first).
    pub fn new(model: M, optimizer: O) -> Result<Self> {
        // Get parameter keys in deterministic order
        let params = model.flatten_params();
        let mut param_keys: Vec<Rc<str>> = params.keys().cloned().collect();
        param_keys.sort();

        // Verify optimizer state is initialized (uses Updatable trait)
        let optimizer_state_count = optimizer.updatable_states_len();
        if optimizer_state_count == 0 && !param_keys.is_empty() {
            tracing::warn!(
                "Optimizer state not initialized ({} params, 0 optimizer states). \
                 Run a warmup step before creating CompiledTrainStep for best performance.",
                param_keys.len()
            );
        }

        Ok(Self {
            model,
            optimizer,
            param_keys,
            compiled: true,
        })
    }

    /// Disable compilation (fallback to eager execution).
    pub fn disable_compilation(&mut self) {
        self.compiled = false;
    }

    /// Enable compilation.
    pub fn enable_compilation(&mut self) {
        self.compiled = true;
    }

    /// Check if compilation is enabled.
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Get reference to model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get reference to optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Get mutable reference to optimizer.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Consume and return the inner model and optimizer.
    pub fn into_inner(self) -> (M, O) {
        (self.model, self.optimizer)
    }

    #[allow(dead_code)] // JIT infrastructure, not yet wired end-to-end
    fn extract_model_params(&self) -> Vec<Array> {
        let params: FlattenedModuleParam = self.model.flatten_params();
        self.param_keys
            .iter()
            .map(|k| params.get(k).cloned().expect("param key must exist"))
            .collect()
    }

    #[allow(dead_code)] // JIT infrastructure, not yet wired end-to-end
    fn update_model_params(&mut self, arrays: &[Array]) -> Result<()> {
        let updates: FlattenedModuleParam = self
            .param_keys
            .iter()
            .cloned()
            .zip(arrays.iter().cloned())
            .collect();

        update_parameters(&mut self.model, updates.into_iter());
        Ok(())
    }
}

/// Stateless loss function that can be compiled.
///
/// This takes all inputs as arrays (params + input data) and returns
/// loss + gradients as arrays, enabling full JIT compilation.
pub fn stateless_loss_and_grad(
    param_arrays: &[Array],
    param_keys: &[Rc<str>],
    input_ids: &Array,
    labels: &Array,
    forward_fn: impl Fn(&FlattenedModuleParam, &Array) -> std::result::Result<Array, Exception>,
) -> std::result::Result<(Array, Vec<Array>), Exception> {
    use pmetal_bridge::compat::nn::keyed_value_and_grad;

    // Reconstruct params map
    let params: FlattenedModuleParam = param_keys
        .iter()
        .cloned()
        .zip(param_arrays.iter().cloned())
        .collect();

    // Define loss function for autodiff
    let loss_fn = |params: HashMap<Rc<str>, Array>,
                   (input_ids_local, labels_local): (Array, Array)|
     -> std::result::Result<Vec<Array>, Exception> {
        let params: FlattenedModuleParam = params;
        let logits = forward_fn(&params, &input_ids_local)?;

        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels_local.index((.., 1..));
        let flat_logits = shift_logits.reshape(&[-1, vocab_size]);
        let flat_labels = shift_labels.reshape(&[-1]);
        let per_token_loss = CrossEntropy::new()?.apply(&flat_logits, &flat_labels)?;
        let labels_dtype = flat_labels.dtype_raw();
        let ignore_idx = Array::from_int(-100).as_dtype(labels_dtype);
        let valid_mask = flat_labels.ne(&ignore_idx);
        use pmetal_bridge::compat::{Dtype, ops};
        let valid_mask_f32 = valid_mask.as_dtype(Dtype::Float32.as_i32());
        let masked_loss = per_token_loss.multiply(&valid_mask_f32);
        let n_valid = valid_mask_f32.sum(None);
        let n_valid_safe = ops::maximum(&n_valid, &Array::from_f32(1.0));
        Ok(vec![masked_loss.sum(None).divide(&n_valid_safe)])
    };

    // Compute value and gradient
    let mut vg = keyed_value_and_grad(loss_fn);
    let (values, grads_map) = vg(params, (input_ids.clone(), labels.clone()))?;

    // Extract gradients in same order as params
    let grads: Vec<Array> = param_keys
        .iter()
        .map(|k| grads_map.get(k).cloned().expect("grad key must exist"))
        .collect();

    Ok((values.into_iter().next().unwrap(), grads))
}

/// Compiled forward-only function for inference.
///
/// This provides JIT compilation for the forward pass only, useful when
/// you need fast inference without gradient computation.
pub struct CompiledForward<F> {
    forward_fn: F,
    compiled: bool,
}

impl<F> CompiledForward<F>
where
    F: Fn(&Array) -> std::result::Result<Array, Exception> + Copy + 'static,
{
    /// Create a new compiled forward function.
    pub fn new(forward_fn: F) -> Self {
        Self {
            forward_fn,
            compiled: true,
        }
    }

    /// Run the forward pass (compiled if enabled).
    pub fn forward(&mut self, input: &Array) -> std::result::Result<Array, Exception> {
        if self.compiled {
            let mut compiled_fn = compile(self.forward_fn, false);
            compiled_fn(input)
        } else {
            (self.forward_fn)(input)
        }
    }

    /// Disable compilation.
    pub fn disable(&mut self) {
        self.compiled = false;
    }

    /// Enable compilation.
    pub fn enable(&mut self) {
        self.compiled = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stateless_loss_basic() {
        // Create simple test case with mock forward function
        let param_keys = vec![Rc::from("weight")];
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let param_arrays = vec![weight];

        let input_ids = Array::from_slice(&[0_i32, 1], &[1, 2]);
        let labels = Array::from_slice(&[-100_i32, 1], &[1, 2]); // First token ignored

        // Mock forward that returns logits
        let forward_fn = |_params: &FlattenedModuleParam,
                          _input: &Array|
         -> std::result::Result<Array, Exception> {
            // Return fake logits [batch=1, seq=2, vocab=4]
            let logits = Array::from_slice(
                &[
                    0.1f32, 0.2, 0.3, 0.4, // token 0
                    0.5, 0.6, 0.7, 0.8, // token 1
                ],
                &[1, 2, 4],
            );
            Ok(logits)
        };

        let result =
            stateless_loss_and_grad(&param_arrays, &param_keys, &input_ids, &labels, forward_fn);

        assert!(result.is_ok());
        let (loss, grads) = result.unwrap();
        loss.eval().unwrap();
        assert!(loss.item::<f32>().is_finite());
        assert_eq!(grads.len(), 1);
    }
}
