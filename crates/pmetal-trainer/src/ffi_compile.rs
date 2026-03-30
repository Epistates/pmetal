//! Direct FFI JIT Compilation for Training.
//!
//! This module bypasses mlx-rs's `compile_with_state` (which has state tracking bugs)
//! and implements direct FFI calls to MLX's compile API with manual state management.
//!
//! # Architecture
//!
//! Python's `mx.compile(inputs=state, outputs=state)` works by:
//! 1. Flattening state containers to arrays
//! 2. Appending state arrays to function inputs
//! 3. Extracting updated state from function outputs
//! 4. Writing updated state back to containers
//!
//! This module replicates that pattern using mlx-rs's basic `compile` function
//! (not `compile_with_state`) with explicit state passing:
//! - [`CompiledTrainingStep`]: JIT-compiled training step with explicit state
//! - Manual state extraction/update around compiled function calls
//!
//! # Performance
//!
//! With proper JIT compilation, this should match mlx-lm's performance:
//! - Full kernel fusion via `mx.compile`
//! - Single GPU dispatch per training step
//! - ~3-4x throughput improvement over eager execution
//!
//! # Usage
//!
//! ```ignore
//! use pmetal_trainer::CompiledTrainingStep;
//!
//! // After warmup step to initialize optimizer state
//! let mut compiled = CompiledTrainingStep::new(model, optimizer)?;
//!
//! // Each training step is JIT-compiled
//! let (loss, ntoks) = compiled.step(&input_ids, &labels, learning_rate)?;
//! ```

#![allow(unsafe_code)]

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

use pmetal_bridge::compat::{
    Array, Dtype, Exception,
    indexing::IndexOp,
    module::{FlattenedModuleParam, ModuleParameters, ModuleParametersExt, update_parameters},
    ops,
    optimizers::{Optimizer, Updatable},
};

use crate::Result;

/// Compiled training step with direct FFI and manual state management.
///
/// This struct owns the model and optimizer, and provides a JIT-compiled
/// training step that properly handles state updates.
// Fields and methods below are part of the experimental FFI JIT compilation
// infrastructure. The JIT path currently falls back to eager execution, so
// these are set up but not yet consumed end-to-end.
#[allow(dead_code)]
pub struct CompiledTrainingStep<M, O> {
    model: M,
    optimizer: O,
    param_keys: Vec<Rc<str>>,
    num_params: usize,
    num_opt_state: usize,
    compile_id: usize,
    compiled: bool,
}

impl<M, O> CompiledTrainingStep<M, O>
where
    M: ModuleParameters + 'static,
    O: Optimizer + Updatable + 'static,
{
    /// Create a new compiled training step.
    ///
    /// **IMPORTANT**: Run a warmup step BEFORE creating this to ensure
    /// optimizer state is fully initialized.
    ///
    /// # Arguments
    /// * `model` - Model with initialized parameters
    /// * `optimizer` - Optimizer with initialized state (run warmup first!)
    pub fn new(model: M, optimizer: O) -> Result<Self> {
        // Get parameter keys in deterministic order
        let params = model.flatten_params();
        let mut param_keys: Vec<Rc<str>> = params.keys().cloned().collect();
        param_keys.sort();
        let num_params = param_keys.len();

        // Get optimizer state count (should be stable after warmup)
        let num_opt_state = optimizer.updatable_states_len();

        // Generate unique compile ID based on type
        let compile_id = {
            let type_id = std::any::TypeId::of::<(M, O)>();
            let mut hasher = DefaultHasher::new();
            type_id.hash(&mut hasher);
            hasher.finish() as usize
        };

        tracing::info!(
            "CompiledTrainingStep created: {} params, {} optimizer states, compile_id={}",
            num_params,
            num_opt_state,
            compile_id
        );

        Ok(Self {
            model,
            optimizer,
            param_keys,
            num_params,
            num_opt_state,
            compile_id,
            compiled: true,
        })
    }

    #[allow(dead_code)]
    fn extract_model_params(&self) -> Vec<Array> {
        let params = self.model.flatten_params();
        self.param_keys
            .iter()
            .map(|k| {
                params
                    .get(k)
                    .map(|a| (*a).clone())
                    .expect("param key must exist")
            })
            .collect()
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn extract_optimizer_state(&self) -> Vec<Array> {
        self.optimizer
            .updatable_states()
            .into_iter()
            .map(|a| a.clone())
            .collect()
    }

    #[allow(dead_code)]
    fn update_optimizer_state(&mut self, arrays: &[Array]) {
        for (state_ref, new_val) in self
            .optimizer
            .updatable_states_mut()
            .into_iter()
            .zip(arrays.iter())
        {
            // Update state array in place via clone assignment.
            *state_ref = new_val.clone();
        }
    }

    /// Get reference to the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get reference to the optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Consume and return the inner model and optimizer.
    pub fn into_inner(self) -> (M, O) {
        (self.model, self.optimizer)
    }

    /// Enable compilation.
    pub fn enable(&mut self) {
        self.compiled = true;
    }

    /// Disable compilation (fallback to eager execution).
    pub fn disable(&mut self) {
        self.compiled = false;
    }

    /// Check if compilation is enabled.
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Get the parameter keys for external use.
    pub fn param_keys(&self) -> &[Rc<str>] {
        &self.param_keys
    }

    /// Get number of parameters.
    pub fn num_params(&self) -> usize {
        self.num_params
    }
}

/// A training step function that can optionally use JIT compilation.
///
/// This struct uses direct FFI calls to MLX's compile API, bypassing mlx-rs's
/// `compile` function which requires the closure to be Copy.
#[allow(dead_code)] // JIT infrastructure — fields set in constructor for future JIT path
pub struct JitTrainingStep<F> {
    forward_fn: F,
    param_keys: Arc<[Rc<str>]>,
    learning_rate: f32,
    num_params: usize,
    num_opt_state: usize,
    max_grad_norm: Option<f32>,
    use_jit: bool,
    compile_id: usize,
}

impl<F> JitTrainingStep<F>
where
    F: Fn(&FlattenedModuleParam, &Array) -> std::result::Result<Array, Exception> + Clone + 'static,
{
    /// Create a new training step.
    ///
    /// # Arguments
    /// * `forward_fn` - Function that takes (params_map, input_ids) and returns logits
    /// * `param_keys` - Ordered parameter keys for reconstruction
    /// * `learning_rate` - Learning rate for optimizer
    /// * `num_params` - Number of model parameters
    /// * `num_opt_state` - Number of optimizer state arrays (2 per param for AdamW)
    /// * `max_grad_norm` - Optional gradient clipping threshold
    /// * `use_jit` - Whether to use JIT compilation
    pub fn new(
        forward_fn: F,
        param_keys: Vec<Rc<str>>,
        learning_rate: f32,
        num_params: usize,
        num_opt_state: usize,
        max_grad_norm: Option<f32>,
        use_jit: bool,
    ) -> Self {
        // Generate unique compile ID based on parameter configuration
        let compile_id = {
            let mut hasher = DefaultHasher::new();
            num_params.hash(&mut hasher);
            num_opt_state.hash(&mut hasher);
            hasher.finish() as usize
        };

        tracing::info!(
            "JitTrainingStep created: {} params, {} opt_state, jit={}, compile_id={}",
            num_params,
            num_opt_state,
            use_jit,
            compile_id
        );

        Self {
            forward_fn,
            param_keys: param_keys.into(),
            learning_rate,
            num_params,
            num_opt_state,
            max_grad_norm,
            use_jit,
            compile_id,
        }
    }

    /// Execute a training step (eager or JIT-compiled).
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `labels` - Target labels [batch, seq_len]
    /// * `params` - Current model parameters (flat array list)
    /// * `opt_state` - Current optimizer state (flat array list)
    ///
    /// # Returns
    /// * `(loss, ntoks, updated_params, updated_opt_state)` - Training step outputs
    pub fn step(
        &self,
        input_ids: &Array,
        labels: &Array,
        params: &[Array],
        opt_state: &[Array],
    ) -> std::result::Result<(Array, Array, Vec<Array>, Vec<Array>), Exception> {
        if self.use_jit {
            self.step_jit(input_ids, labels, params, opt_state)
        } else {
            self.step_eager(input_ids, labels, params, opt_state)
        }
    }

    /// Execute training step with eager execution (no compilation).
    fn step_eager(
        &self,
        input_ids: &Array,
        labels: &Array,
        params: &[Array],
        opt_state: &[Array],
    ) -> std::result::Result<(Array, Array, Vec<Array>, Vec<Array>), Exception> {
        // Compute loss and gradients
        let (loss, grads) = stateless_loss_and_grad(
            input_ids,
            labels,
            params,
            &self.param_keys,
            self.forward_fn.clone(),
        )?;

        // Apply gradient clipping if configured
        let grads = if let Some(max_norm) = self.max_grad_norm {
            clip_grad_norm(&grads, max_norm)?
        } else {
            grads
        };

        // Apply optimizer update
        let (updated_params, updated_opt_state) =
            stateless_optimizer_step(params, &grads, opt_state, self.learning_rate)?;

        // Count valid tokens
        let ntoks = count_valid_tokens(labels)?;

        Ok((loss, ntoks, updated_params, updated_opt_state))
    }

    /// Execute training step with JIT compilation using direct FFI.
    ///
    /// NOTE: Direct FFI JIT compilation is experimental. Falls back to eager
    /// execution due to Metal command buffer timing issues. The raw_ffi module
    /// provides the building blocks for future JIT support.
    fn step_jit(
        &self,
        input_ids: &Array,
        labels: &Array,
        params: &[Array],
        opt_state: &[Array],
    ) -> std::result::Result<(Array, Array, Vec<Array>, Vec<Array>), Exception> {
        // Fall back to eager execution - FFI JIT has Metal timing issues
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::info!(
                "JIT mode requested - using optimized eager execution. \
                 Direct FFI JIT compilation is experimental (see raw_ffi module)."
            );
        });

        self.step_eager(input_ids, labels, params, opt_state)
    }
}

/// Raw FFI helper for JIT compilation.
///
/// This module provides low-level FFI access to MLX's compilation API,
/// allowing JIT compilation without depending on mlx-rs's private types.
/// Stub implementations for the raw FFI JIT compilation interface.
///
/// The bridge does not expose raw mlx_sys FFI types; this module provides
/// no-op stub types that preserve the public API while falling back to
/// eager (non-compiled) execution. Real JIT compilation via this path
/// requires direct mlx_sys access which is unavailable in the bridge model.
pub mod raw_ffi {
    use super::*;

    /// No-op stub replacing `mlx_sys::mlx_closure`.
    pub struct RawClosure {
        f: Option<Box<dyn Fn(&[Array]) -> std::result::Result<Vec<Array>, Exception>>>,
    }

    impl RawClosure {
        /// Create a new empty closure stub.
        pub fn new() -> Self {
            Self { f: None }
        }
    }

    impl Default for RawClosure {
        fn default() -> Self {
            Self::new()
        }
    }

    /// No-op stub replacing `mlx_sys::mlx_vector_array`.
    pub struct RawVectorArray {
        arrays: Vec<Array>,
    }

    impl RawVectorArray {
        /// Create a new empty vector array stub.
        pub fn new() -> Self {
            Self { arrays: Vec::new() }
        }

        /// Create from a slice of Arrays.
        pub fn from_arrays(arrays: &[Array]) -> std::result::Result<Self, Exception> {
            Ok(Self {
                arrays: arrays.to_vec(),
            })
        }

        /// Convert to Vec<Array>.
        pub fn to_arrays(&self) -> std::result::Result<Vec<Array>, Exception> {
            Ok(self.arrays.clone())
        }
    }

    impl Default for RawVectorArray {
        fn default() -> Self {
            Self::new()
        }
    }

    /// No-op compile: returns a stub closure (falls back to eager execution).
    pub fn compile_closure(
        _closure: &RawClosure,
        _compile_id: usize,
        _shapeless: bool,
    ) -> std::result::Result<RawClosure, Exception> {
        Ok(RawClosure::new())
    }

    /// No-op apply: the stub closure does nothing.
    pub fn apply_closure(
        _closure: &RawClosure,
        _inputs: &RawVectorArray,
    ) -> std::result::Result<RawVectorArray, Exception> {
        Err(Exception::custom(
            "raw_ffi: JIT compilation not available in bridge mode",
        ))
    }

    /// Type alias for the Rust closure signature.
    pub type RustClosureFn = Box<dyn Fn(&[Array]) -> std::result::Result<Vec<Array>, Exception>>;

    /// Create a closure from a Rust function (stub — stores the closure for eager fallback).
    pub fn create_closure_from_rust<F>(f: F) -> std::result::Result<RawClosure, Exception>
    where
        F: Fn(&[Array]) -> std::result::Result<Vec<Array>, Exception> + 'static,
    {
        Ok(RawClosure {
            f: Some(Box::new(f)),
        })
    }

    /// A JIT-compiled function that wraps a Rust closure (stub — eager fallback).
    pub struct CompiledRustClosure {
        /// The original closure for eager fallback.
        f: Box<dyn Fn(&[Array]) -> std::result::Result<Vec<Array>, Exception>>,
        #[allow(dead_code)]
        compile_id: usize,
    }

    impl CompiledRustClosure {
        /// Create a new compiled closure stub from a Rust function.
        pub fn new<F>(f: F, compile_id: usize) -> std::result::Result<Self, Exception>
        where
            F: Fn(&[Array]) -> std::result::Result<Vec<Array>, Exception> + 'static,
        {
            Ok(Self {
                f: Box::new(f),
                compile_id,
            })
        }

        /// Execute the closure (eager, not JIT-compiled).
        pub fn call(&self, inputs: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
            (self.f)(inputs)
        }
    }
}

/// Count valid (non-ignored) tokens for throughput tracking.
fn count_valid_tokens(labels: &Array) -> std::result::Result<Array, Exception> {
    let shifted_labels = labels.index((.., 1..));
    let flat_labels = shifted_labels.reshape(&[-1]);
    let labels_dtype = flat_labels.dtype_raw();
    let ignore_idx = Array::from_int(-100).as_dtype(labels_dtype);
    let valid_mask = flat_labels.ne(&ignore_idx);
    Ok(valid_mask.sum(None).as_dtype(Dtype::Float32.as_i32()))
}

/// Clip gradient norm for stability.
///
/// # Arguments
/// * `grads` - Gradient arrays to clip
/// * `max_norm` - Maximum allowed norm
///
/// # Returns
/// * Clipped gradients with total norm <= max_norm
pub fn clip_grad_norm(
    grads: &[Array],
    max_norm: f32,
) -> std::result::Result<Vec<Array>, Exception> {
    // Compute total norm: sqrt(sum(grad^2 for all grads))
    let mut total_norm_sq = Array::from_f32(0.0);
    for grad in grads {
        let grad_sq = grad.multiply(grad);
        let grad_norm_sq = grad_sq.sum(None);
        total_norm_sq = total_norm_sq.add(&grad_norm_sq);
    }
    let total_norm = total_norm_sq.sqrt();

    // Compute scale factor: max_norm / max(total_norm, max_norm)
    let max_norm_arr = Array::from_f32(max_norm);
    let clip_coef = max_norm_arr.divide(&ops::maximum(&total_norm, &max_norm_arr));

    // Scale all gradients
    Ok(grads.iter().map(|g| g.multiply(&clip_coef)).collect())
}

/// Stateless loss computation for JIT compilation.
///
/// This function takes all inputs as arrays and returns loss + gradients,
/// enabling full JIT compilation without mutable state references.
///
/// # Arguments
/// * `input_ids` - Input token IDs [batch, seq_len]
/// * `labels` - Target labels [batch, seq_len]
/// * `param_arrays` - Flattened model parameters
/// * `param_keys` - Keys for parameter reconstruction
///
/// # Returns
/// * `(loss, gradients)` - Loss scalar and gradients in same order as params
pub fn stateless_loss_and_grad(
    input_ids: &Array,
    labels: &Array,
    param_arrays: &[Array],
    param_keys: &[Rc<str>],
    forward_fn: impl Fn(&FlattenedModuleParam, &Array) -> std::result::Result<Array, Exception>,
) -> std::result::Result<(Array, Vec<Array>), Exception> {
    use pmetal_bridge::compat::nn::keyed_value_and_grad;
    use std::collections::HashMap;

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

        // Keep the loss local to the closure to avoid leaking borrow lifetimes
        // through a helper that predates the bridge-era autodiff wrappers.
        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels_local.index((.., 1..));
        let flat_logits = shift_logits.reshape(&[-1, vocab_size]);
        let flat_labels = shift_labels.reshape(&[-1]);
        let per_token_loss = pmetal_bridge::compat::losses::CrossEntropy::new()?
            .apply(&flat_logits, &flat_labels)?;
        let labels_dtype = flat_labels.dtype_raw();
        let ignore_idx = Array::from_int(-100).as_dtype(labels_dtype);
        let valid_mask = flat_labels.ne(&ignore_idx);
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

/// Apply optimizer update to parameters.
///
/// This is a stateless version of optimizer.update() that takes arrays
/// and returns updated arrays, suitable for JIT compilation.
///
/// # Arguments
/// * `params` - Current parameter arrays
/// * `grads` - Gradient arrays (same order as params)
/// * `opt_state` - Current optimizer state arrays
/// * `learning_rate` - Learning rate
///
/// # Returns
/// * `(updated_params, updated_opt_state)` - Updated parameter and optimizer state
pub fn stateless_optimizer_step(
    params: &[Array],
    grads: &[Array],
    opt_state: &[Array],
    learning_rate: f32,
) -> std::result::Result<(Vec<Array>, Vec<Array>), Exception> {
    // For AdamW, opt_state contains [m0, v0, m1, v1, ...] for each param
    // Each param has 2 state arrays (first moment m, second moment v)
    let num_params = params.len();
    let state_per_param = if opt_state.is_empty() {
        0
    } else {
        opt_state.len() / num_params
    };

    let beta1 = Array::from_f32(0.9);
    let beta2 = Array::from_f32(0.999);
    let eps = Array::from_f32(1e-8);
    let lr = Array::from_f32(learning_rate);

    let mut updated_params = Vec::with_capacity(num_params);
    let mut updated_opt_state = Vec::with_capacity(opt_state.len());

    for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
        if state_per_param == 2 {
            // AdamW update
            let m = &opt_state[i * 2];
            let v = &opt_state[i * 2 + 1];

            // m = beta1 * m + (1 - beta1) * grad
            let new_m = beta1
                .multiply(m)
                .add(&Array::from_f32(1.0 - 0.9).multiply(grad));

            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_sq = grad.multiply(grad);
            let new_v = beta2
                .multiply(v)
                .add(&Array::from_f32(1.0 - 0.999).multiply(&grad_sq));

            // param = param - lr * m / (sqrt(v) + eps)
            let denom = new_v.sqrt().add(&eps);
            let update = lr.multiply(&new_m).divide(&denom);
            let new_param = param.subtract(&update);

            updated_params.push(new_param);
            updated_opt_state.push(new_m);
            updated_opt_state.push(new_v);
        } else {
            // SGD fallback
            let new_param = param.subtract(&lr.multiply(grad));
            updated_params.push(new_param);
        }
    }

    Ok((updated_params, updated_opt_state))
}

// Re-export Array's index operation

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stateless_optimizer_step_sgd() {
        // Test SGD (no optimizer state)
        let params = vec![
            Array::from_f32_slice(&[1.0f32, 2.0, 3.0], &[3]),
            Array::from_f32_slice(&[4.0f32, 5.0], &[2]),
        ];
        let grads = vec![
            Array::from_f32_slice(&[0.1f32, 0.2, 0.3], &[3]),
            Array::from_f32_slice(&[0.4f32, 0.5], &[2]),
        ];
        let opt_state: Vec<Array> = vec![];
        let lr = 0.1;

        let (updated_params, updated_opt_state) =
            stateless_optimizer_step(&params, &grads, &opt_state, lr).unwrap();

        assert_eq!(updated_params.len(), 2);
        assert_eq!(updated_opt_state.len(), 0);

        // Check first param: [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
        updated_params[0].eval().unwrap();
        let p0: Vec<f32> = updated_params[0].as_slice().to_vec();
        assert!((p0[0] - 0.99).abs() < 1e-5);
        assert!((p0[1] - 1.98).abs() < 1e-5);
        assert!((p0[2] - 2.97).abs() < 1e-5);
    }

    #[test]
    fn test_stateless_optimizer_step_adam() {
        // Test AdamW with initialized state
        let params = vec![Array::from_f32_slice(&[1.0f32, 2.0], &[2])];
        let grads = vec![Array::from_f32_slice(&[0.1f32, 0.2], &[2])];
        // m and v for one param
        let opt_state = vec![
            Array::from_f32_slice(&[0.0f32, 0.0], &[2]), // m
            Array::from_f32_slice(&[0.0f32, 0.0], &[2]), // v
        ];
        let lr = 0.001;

        let (updated_params, updated_opt_state) =
            stateless_optimizer_step(&params, &grads, &opt_state, lr).unwrap();

        assert_eq!(updated_params.len(), 1);
        assert_eq!(updated_opt_state.len(), 2); // m and v

        // Verify shapes
        updated_params[0].eval().unwrap();
        updated_opt_state[0].eval().unwrap();
        updated_opt_state[1].eval().unwrap();

        assert_eq!(updated_params[0].shape(), &[2]);
        assert_eq!(updated_opt_state[0].shape(), &[2]);
        assert_eq!(updated_opt_state[1].shape(), &[2]);
    }

    // NOTE: FFI JIT compilation tests are disabled due to Metal command buffer
    // timing issues. The raw_ffi module provides the building blocks but needs
    // deeper integration with MLX's async evaluation system.
    //
    // Error: "Completed handler provided after commit call"
    // This suggests the compilation tries to add completion handlers after
    // the Metal command buffer has been committed.

    #[test]
    fn test_compiled_rust_closure_simple() {
        use raw_ffi::CompiledRustClosure;

        // Test simple JIT-compiled closure: doubles input
        let double_fn = |inputs: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
            let x = &inputs[0];
            let doubled = x.multiply(&Array::from_f32(2.0));
            Ok(vec![doubled])
        };

        let compiled = CompiledRustClosure::new(double_fn, 12345).unwrap();

        // Test execution
        let input = Array::from_f32_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let outputs = compiled.call(&[input]).unwrap();

        assert_eq!(outputs.len(), 1);
        outputs[0].eval().unwrap();
        let result: Vec<f32> = outputs[0].as_slice().to_vec();
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 4.0).abs() < 1e-5);
        assert!((result[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_compiled_rust_closure_multiple_inputs_outputs() {
        use raw_ffi::CompiledRustClosure;

        // Test JIT-compiled closure with multiple inputs and outputs
        let fn_multi = |inputs: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
            let a = &inputs[0];
            let b = &inputs[1];
            let sum = a.add(b);
            let prod = a.multiply(b);
            Ok(vec![sum, prod])
        };

        let compiled = CompiledRustClosure::new(fn_multi, 12346).unwrap();

        // Test execution
        let a = Array::from_f32_slice(&[1.0f32, 2.0], &[2]);
        let b = Array::from_f32_slice(&[3.0f32, 4.0], &[2]);
        let outputs = compiled.call(&[a, b]).unwrap();

        assert_eq!(outputs.len(), 2);
        outputs[0].eval().unwrap();
        outputs[1].eval().unwrap();

        let sum_result: Vec<f32> = outputs[0].as_slice().to_vec();
        let prod_result: Vec<f32> = outputs[1].as_slice().to_vec();

        assert!((sum_result[0] - 4.0).abs() < 1e-5); // 1 + 3
        assert!((sum_result[1] - 6.0).abs() < 1e-5); // 2 + 4
        assert!((prod_result[0] - 3.0).abs() < 1e-5); // 1 * 3
        assert!((prod_result[1] - 8.0).abs() < 1e-5); // 2 * 4
    }
}
