//! Schedule-Free Optimizer (SFO) implementation.
//!
//! Based on "The Road Less Traveled: Schedule-Free Optimization" (Defazio et al., 2024).
//! This optimizer removes the need for a learning rate schedule by replacing the
//! standard momentum update with a specific interpolation between the current point
//! and a "conservative" estimate.
//!
//! # Key Benefits
//!
//! - No learning rate schedule tuning required (just set a constant LR)
//! - Faster convergence than AdamW + Schedule in many cases
//! - State-of-the-art performance on various benchmarks
//!
//! # Algorithm
//!
//! The update rule for Schedule-Free AdamW:
//!
//! ```text
//! y_{t} = (1 - β₁) z_{t} + β₁ y_{t-1}
//! z_{t+1} = z_{t} - η * ∇f(y_t) / (√v_t + ε) - η * λ * z_{t}
//! v_{t+1} = β₂ v_t + (1 - β₂) (∇f(y_t))²
//! ```
//!
//! Where:
//! - `z` is the "conservative" sequence (primal weights)
//! - `y` is the "optimistic" sequence (evaluation weights)
//! - `v` is the variance estimate (second moment)
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_trainer::{ScheduleFreeOptimizer, ScheduleFreeConfig};
//!
//! let config = ScheduleFreeConfig::default()
//!     .with_lr(0.0025)
//!     .with_warmup_steps(100);
//!
//! let mut optimizer = ScheduleFreeOptimizer::new(config);
//!
//! // Training loop
//! for step in 0..num_steps {
//!     let grads = compute_gradients(&model, &batch);
//!     optimizer.step(&mut model.parameters(), &grads)?;
//! }
//!
//! // Before saving/inference, finalize to get the conservative estimate
//! optimizer.finalize(&mut model.parameters())?;
//! ```
//!
//! # References
//!
//! - [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)
//! - [Facebook Research Implementation](https://github.com/facebookresearch/schedule_free)

use mlx_rs::{array, Array};

/// Error type for Schedule-Free optimizer.
#[derive(Debug, thiserror::Error)]
pub enum ScheduleFreeError {
    /// MLX computation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// Parameter mismatch error.
    #[error("Parameter count mismatch: expected {expected}, got {actual}")]
    ParameterMismatch { expected: usize, actual: usize },
    /// State not initialized.
    #[error("Optimizer state not initialized")]
    StateNotInitialized,
}

/// Result type for Schedule-Free operations.
pub type ScheduleFreeResult<T> = std::result::Result<T, ScheduleFreeError>;

/// Configuration for the Schedule-Free optimizer.
#[derive(Debug, Clone)]
pub struct ScheduleFreeConfig {
    /// Learning rate (base step size).
    /// Schedule-Free typically needs larger LR than scheduled optimizers.
    /// Default: 0.0025
    pub lr: f32,

    /// Beta1 (momentum decay), typically 0.9 or 0.95.
    /// Controls the interpolation between z and y.
    /// Default: 0.9
    pub beta1: f32,

    /// Beta2 (variance decay), typically 0.95 or 0.98.
    /// Lower than Adam's 0.999 is recommended for Schedule-Free.
    /// Default: 0.95
    pub beta2: f32,

    /// Epsilon for numerical stability.
    /// Default: 1e-8
    pub eps: f32,

    /// Weight decay coefficient.
    /// Applied to the conservative estimate z (decoupled like AdamW).
    /// Default: 0.0
    pub weight_decay: f32,

    /// Warmup steps during which we ramp up the interpolation.
    /// During warmup, we linearly increase the effective beta1 from 0 to beta1.
    /// Default: 0 (no warmup)
    pub warmup_steps: usize,

    /// Whether to use the "warmup" variant from the paper.
    /// When true, uses c_k = min(1, (k+1) / warmup_steps) * beta1.
    /// Default: true
    pub use_warmup_interpolation: bool,
}

impl Default for ScheduleFreeConfig {
    fn default() -> Self {
        Self {
            lr: 0.0025,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.0,
            warmup_steps: 0,
            use_warmup_interpolation: true,
        }
    }
}

impl ScheduleFreeConfig {
    /// Create a new config with the given learning rate.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set the learning rate.
    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set beta1 (momentum).
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (variance decay).
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set weight decay.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set warmup steps.
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> ScheduleFreeResult<()> {
        if self.lr <= 0.0 {
            return Err(ScheduleFreeError::Mlx(mlx_rs::error::Exception::from(
                "learning rate must be positive",
            )));
        }
        if self.beta1 < 0.0 || self.beta1 >= 1.0 {
            return Err(ScheduleFreeError::Mlx(mlx_rs::error::Exception::from(
                "beta1 must be in [0, 1)",
            )));
        }
        if self.beta2 < 0.0 || self.beta2 >= 1.0 {
            return Err(ScheduleFreeError::Mlx(mlx_rs::error::Exception::from(
                "beta2 must be in [0, 1)",
            )));
        }
        Ok(())
    }
}

/// Optimizer state for a single parameter.
#[derive(Debug, Clone)]
struct ParameterState {
    /// Conservative estimate (z in the paper).
    z: Array,
    /// Second moment estimate (v in the paper).
    v: Array,
    /// Saved evaluation iterate (y before eval_mode clobbered params).
    ///
    /// Populated by `eval_mode`, cleared by `train_mode`.
    y_saved: Option<Array>,
}

/// Schedule-Free AdamW Optimizer.
///
/// This optimizer maintains two sequences:
/// - `z`: The "conservative" sequence (primal weights)
/// - `y`: The "optimistic" sequence (evaluation weights, stored in params)
///
/// During training, the model uses `y` for forward/backward passes.
/// For final inference/saving, call `finalize()` to copy `z` to params.
pub struct ScheduleFreeOptimizer {
    /// Configuration.
    config: ScheduleFreeConfig,
    /// Step counter (t).
    step: usize,
    /// Optimizer state for each parameter.
    state: Vec<ParameterState>,
    /// Whether state has been initialized.
    initialized: bool,
}

impl ScheduleFreeOptimizer {
    /// Create a new Schedule-Free optimizer.
    pub fn new(config: ScheduleFreeConfig) -> Self {
        Self {
            config,
            step: 0,
            state: Vec::new(),
            initialized: false,
        }
    }

    /// Create with default config.
    pub fn default_config() -> Self {
        Self::new(ScheduleFreeConfig::default())
    }

    /// Get the current step count.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get the current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.config.lr
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Initialize optimizer state for parameters.
    fn initialize_state(&mut self, params: &[Array]) -> ScheduleFreeResult<()> {
        self.state.clear();
        self.state.reserve(params.len());

        for p in params {
            // Initialize z = params (conservative = current)
            // Initialize v = zeros_like(params) (no variance estimate yet)
            let z = p.clone();
            let v = mlx_rs::ops::zeros::<f32>(p.shape())?;

            self.state.push(ParameterState {
                z,
                v,
                y_saved: None,
            });
        }

        self.initialized = true;
        Ok(())
    }

    /// Compute the interpolation coefficient c_k for the current step.
    fn compute_ck(&self) -> f32 {
        if !self.config.use_warmup_interpolation || self.config.warmup_steps == 0 {
            return self.config.beta1;
        }

        // Linear warmup: c_k = min(1, (k+1) / warmup_steps) * beta1
        let warmup_factor = ((self.step + 1) as f32 / self.config.warmup_steps as f32).min(1.0);
        warmup_factor * self.config.beta1
    }

    /// Perform a single optimization step.
    ///
    /// # Arguments
    ///
    /// * `params` - Mutable reference to model parameters.
    ///   These hold `y` (evaluation weights) during training.
    /// * `grads` - Gradients for the parameters.
    ///
    /// # Algorithm
    ///
    /// 1. Update variance estimate: v = β₂ * v + (1 - β₂) * grad²
    /// 2. Update conservative estimate: z = z - lr * (grad / √v + ε) - lr * wd * z
    /// 3. Update evaluation point: y = (1 - c_k) * z + c_k * y
    pub fn step(&mut self, params: &mut [Array], grads: &[Array]) -> ScheduleFreeResult<()> {
        // Initialize state on first call
        if !self.initialized {
            self.initialize_state(params)?;
        }

        // Verify parameter count matches
        if params.len() != self.state.len() {
            return Err(ScheduleFreeError::ParameterMismatch {
                expected: self.state.len(),
                actual: params.len(),
            });
        }

        if grads.len() != params.len() {
            return Err(ScheduleFreeError::ParameterMismatch {
                expected: params.len(),
                actual: grads.len(),
            });
        }

        // Compute c_k before incrementing the step counter so that step k uses
        // the warmup factor for step k (not k+1).  See Defazio & Mishchenko 2024
        // Algorithm 1: c_k = min(1, (k+1)/warmup) * beta1, where k is 0-indexed.
        let c_k = self.compute_ck();

        // Advance step counter after c_k is computed to avoid the off-by-one.
        self.step += 1;

        // Precompute scalars as arrays for broadcasting
        let lr = array!(self.config.lr);
        let beta2 = array!(self.config.beta2);
        let one_minus_beta2 = array!(1.0 - self.config.beta2);
        let eps = array!(self.config.eps);
        let wd = array!(self.config.weight_decay);
        let ck_arr = array!(c_k);
        let one_minus_ck = array!(1.0 - c_k);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let state = &mut self.state[i];

            // 1. Update variance estimate (v)
            // v_{t} = β₂ * v_{t-1} + (1 - β₂) * grad²
            let grad_sq = grad.square()?;
            state.v = state
                .v
                .multiply(&beta2)?
                .add(&grad_sq.multiply(&one_minus_beta2)?)?;

            // 2. Compute adaptive step size denominator
            // denom = √v + ε
            let denom = state.v.sqrt()?.add(&eps)?;

            // 3. Update conservative estimate (z)
            // z_{t+1} = z_{t} - lr * grad / denom - lr * wd * z_{t}
            let grad_term = grad.divide(&denom)?;
            let update = if self.config.weight_decay > 0.0 {
                let decay_term = state.z.multiply(&wd)?;
                grad_term.add(&decay_term)?
            } else {
                grad_term
            };
            state.z = state.z.subtract(&update.multiply(&lr)?)?;

            // 4. Update evaluation point (y) for next step
            // y_{t+1} = (1 - c_k) * z_{t+1} + c_k * y_{t}
            let z_part = state.z.multiply(&one_minus_ck)?;
            let y_part = param.multiply(&ck_arr)?;
            *param = z_part.add(&y_part)?;
        }

        Ok(())
    }

    /// Switch parameters to evaluation mode per the Schedule-Free paper.
    ///
    /// Computes the evaluation point via weighted interpolation between
    /// the primal (z) and the current iterate (x = params):
    ///
    /// ```text
    /// y = (1 - 1/c_k) * z + (1/c_k) * x
    /// ```
    ///
    /// where `c_k` is the interpolation coefficient at the current step.
    /// The original `y` (iterate) is saved internally so `train_mode` can
    /// restore it.
    ///
    /// Call this before evaluation or checkpointing. Call `train_mode` before
    /// resuming gradient steps.
    ///
    /// Reference: Defazio & Mishchenko (2024), Algorithm 1.
    pub fn eval_mode(&mut self, params: &mut [Array]) -> ScheduleFreeResult<()> {
        if !self.initialized {
            return Err(ScheduleFreeError::StateNotInitialized);
        }
        if params.len() != self.state.len() {
            return Err(ScheduleFreeError::ParameterMismatch {
                expected: self.state.len(),
                actual: params.len(),
            });
        }

        let c_k = self.compute_ck();
        // Guard against c_k == 0 at the very first step (before any updates).
        let inv_ck = if c_k > 0.0 { 1.0 / c_k } else { 1.0 };
        let w_z = array!(1.0 - inv_ck);
        let w_x = array!(inv_ck);

        for (i, param) in params.iter_mut().enumerate() {
            let state = &mut self.state[i];
            // Save the current iterate y so train_mode can restore it.
            state.y_saved = Some(param.clone());
            // Compute evaluation point: y = (1 - 1/c_k) * z + (1/c_k) * x
            let z_part = state.z.multiply(&w_z)?;
            let x_part = param.multiply(&w_x)?;
            *param = z_part.add(&x_part)?;
        }

        Ok(())
    }

    /// Restore parameters to train mode after an `eval_mode` call.
    ///
    /// Copies the saved iterate `y` back into `params`, reversing the
    /// interpolation performed by `eval_mode`.
    ///
    /// Reference: Defazio & Mishchenko (2024), Algorithm 1.
    pub fn train_mode(&mut self, params: &mut [Array]) -> ScheduleFreeResult<()> {
        if !self.initialized {
            return Err(ScheduleFreeError::StateNotInitialized);
        }
        if params.len() != self.state.len() {
            return Err(ScheduleFreeError::ParameterMismatch {
                expected: self.state.len(),
                actual: params.len(),
            });
        }

        for (i, param) in params.iter_mut().enumerate() {
            let state = &mut self.state[i];
            if let Some(y) = state.y_saved.take() {
                *param = y;
            }
            // If y_saved is None we were already in train mode; leave param as-is.
        }

        Ok(())
    }

    /// Finalize the optimizer by copying the conservative estimate to params.
    ///
    /// The Schedule-Free paper recommends using `z` (conservative estimate)
    /// or a weighted average for final inference/saving, as it tends to be
    /// more stable than `y`.
    ///
    /// Call this method before saving the model or running final evaluation.
    pub fn finalize(&self, params: &mut [Array]) -> ScheduleFreeResult<()> {
        if !self.initialized {
            return Err(ScheduleFreeError::StateNotInitialized);
        }

        if params.len() != self.state.len() {
            return Err(ScheduleFreeError::ParameterMismatch {
                expected: self.state.len(),
                actual: params.len(),
            });
        }

        for (i, param) in params.iter_mut().enumerate() {
            *param = self.state[i].z.clone();
        }

        Ok(())
    }

    /// Reset optimizer state.
    ///
    /// This clears all accumulated state (z, v) and resets the step counter.
    /// Useful when starting a new training run with the same optimizer.
    pub fn reset(&mut self) {
        self.state.clear();
        self.step = 0;
        self.initialized = false;
    }

    /// Get the state for a specific parameter (for debugging/inspection).
    pub fn get_state(&self, index: usize) -> Option<(&Array, &Array)> {
        self.state.get(index).map(|s| (&s.z, &s.v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ScheduleFreeConfig::default();
        assert!((config.lr - 0.0025).abs() < 1e-6);
        assert!((config.beta1 - 0.9).abs() < 1e-6);
        assert!((config.beta2 - 0.95).abs() < 1e-6);
        assert_eq!(config.warmup_steps, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = ScheduleFreeConfig::new(0.001)
            .with_beta1(0.95)
            .with_weight_decay(0.01)
            .with_warmup_steps(100);

        assert!((config.lr - 0.001).abs() < 1e-6);
        assert!((config.beta1 - 0.95).abs() < 1e-6);
        assert!((config.weight_decay - 0.01).abs() < 1e-6);
        assert_eq!(config.warmup_steps, 100);
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = ScheduleFreeOptimizer::new(ScheduleFreeConfig::default());
        assert_eq!(optimizer.current_step(), 0);
        assert!((optimizer.learning_rate() - 0.0025).abs() < 1e-6);
    }

    #[test]
    fn test_single_step() {
        let config = ScheduleFreeConfig::default();
        let mut optimizer = ScheduleFreeOptimizer::new(config);

        // Create simple test parameters and gradients
        let mut params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3])];

        // Perform one step
        optimizer.step(&mut params, &grads).unwrap();

        assert_eq!(optimizer.current_step(), 1);

        // Params should have changed
        params[0].eval().unwrap();
        let vals: Vec<f32> = params[0].as_slice().to_vec();
        assert!(vals[0] != 1.0); // Value should have changed
    }

    #[test]
    fn test_multiple_steps() {
        let config = ScheduleFreeConfig::new(0.01).with_warmup_steps(10);
        let mut optimizer = ScheduleFreeOptimizer::new(config);

        let mut params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])];

        // Run multiple steps
        for _ in 0..20 {
            let grads = vec![Array::from_slice(&[0.1f32, 0.1, 0.1], &[3])];
            optimizer.step(&mut params, &grads).unwrap();
        }

        assert_eq!(optimizer.current_step(), 20);
    }

    #[test]
    fn test_finalize() {
        let config = ScheduleFreeConfig::default();
        let mut optimizer = ScheduleFreeOptimizer::new(config);

        let mut params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3])];

        // Run some steps
        for _ in 0..5 {
            optimizer.step(&mut params, &grads.clone()).unwrap();
        }

        // Finalize to get conservative estimate
        optimizer.finalize(&mut params).unwrap();

        // The finalized params should be the conservative estimate z
        params[0].eval().unwrap();
        let (z, _) = optimizer.get_state(0).unwrap();
        z.eval().unwrap();

        let param_vals: Vec<f32> = params[0].as_slice().to_vec();
        let z_vals: Vec<f32> = z.as_slice().to_vec();

        for (p, z) in param_vals.iter().zip(z_vals.iter()) {
            assert!((p - z).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parameter_mismatch() {
        let config = ScheduleFreeConfig::default();
        let mut optimizer = ScheduleFreeOptimizer::new(config);

        let mut params = vec![Array::from_slice(&[1.0f32, 2.0], &[2])];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2], &[2])];

        // Initialize with 1 parameter
        optimizer.step(&mut params, &grads).unwrap();

        // Try with 2 parameters - should fail
        let mut params2 = vec![
            Array::from_slice(&[1.0f32, 2.0], &[2]),
            Array::from_slice(&[3.0f32, 4.0], &[2]),
        ];
        let grads2 = vec![
            Array::from_slice(&[0.1f32, 0.2], &[2]),
            Array::from_slice(&[0.3f32, 0.4], &[2]),
        ];

        let result = optimizer.step(&mut params2, &grads2);
        assert!(result.is_err());
    }

    #[test]
    fn test_warmup_interpolation() {
        let config = ScheduleFreeConfig::new(0.01)
            .with_beta1(0.9)
            .with_warmup_steps(10)
            .with_lr(0.01);

        let optimizer = ScheduleFreeOptimizer::new(config);

        // At step 0, c_k should be 0.1 * 0.9 = 0.09
        assert!((optimizer.compute_ck() - 0.09).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let config = ScheduleFreeConfig::default();
        let mut optimizer = ScheduleFreeOptimizer::new(config);

        let mut params = vec![Array::from_slice(&[1.0f32, 2.0], &[2])];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2], &[2])];

        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.current_step(), 1);

        optimizer.reset();
        assert_eq!(optimizer.current_step(), 0);
        assert!(!optimizer.initialized);
    }
}
