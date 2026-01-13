//! GaLore (Gradient Low-Rank Projection) implementation.
//!
//! GaLore allows full-parameter learning in a memory-efficient manner by
//! projecting gradients into a low-rank subspace before the optimizer step.
//! The optimizer states are maintained only for the projected (low-rank) gradients,
//! significantly reducing memory usage.
//!
//! # Algorithm
//!
//! 1. Compute gradient G ∈ ℝ^(m×n)
//! 2. Every T steps, compute SVD: G = UΣV^T and extract projector P
//!    - If m >= n: P = V[:, :r] (right projection)
//!    - If m < n: P = U[:, :r] (left projection)
//! 3. Project gradient: G_low = G @ P (right) or P^T @ G (left)
//! 4. Apply optimizer to G_low (low-rank, much smaller memory footprint)
//! 5. Reconstruct: G_full = G_low @ P^T (right) or P @ G_low (left)
//! 6. Update weights: W = W - lr * G_full
//!
//! # Memory Savings
//!
//! For a weight matrix W ∈ ℝ^(m×n), traditional Adam stores:
//! - First moment m: m×n floats
//! - Second moment v: m×n floats
//!
//! With GaLore (rank r), Adam stores:
//! - First moment m: min(m,n) × r floats
//! - Second moment v: min(m,n) × r floats
//!
//! For large models, this can reduce optimizer memory by 50-80%.
//!
//! # References
//!
//! - "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
//!   (Zhao et al., 2024) <https://arxiv.org/abs/2403.03507>

use crate::LoraError;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, StreamOrDevice};

/// Projection type for GaLore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GaloreProjectionType {
    /// Standard: right projection if m >= n, left otherwise.
    #[default]
    Standard,
    /// Reverse standard: left projection if m >= n, right otherwise.
    ReverseStandard,
    /// Always use right projection (project with V).
    Right,
    /// Always use left projection (project with U).
    Left,
}

/// Configuration for GaLore projector.
#[derive(Debug, Clone)]
pub struct GaloreConfig {
    /// Low-rank dimension for gradient projection.
    pub rank: usize,
    /// Number of steps between projector updates (SVD recomputation).
    pub update_proj_gap: usize,
    /// Projection type (determines left vs right projection).
    pub proj_type: GaloreProjectionType,
    /// Scale factor for unprojected gradients.
    pub scale: f32,
}

impl Default for GaloreConfig {
    fn default() -> Self {
        Self {
            rank: 128,
            update_proj_gap: 200,
            proj_type: GaloreProjectionType::Standard,
            scale: 1.0,
        }
    }
}

impl GaloreConfig {
    /// Create a new GaLore config with the given rank.
    pub fn with_rank(rank: usize) -> Self {
        Self {
            rank,
            ..Default::default()
        }
    }

    /// Set the update projection gap.
    pub fn update_gap(mut self, gap: usize) -> Self {
        self.update_proj_gap = gap;
        self
    }

    /// Set the projection type.
    pub fn projection_type(mut self, proj_type: GaloreProjectionType) -> Self {
        self.proj_type = proj_type;
        self
    }

    /// Set the scale factor.
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

/// GaLore gradient projector.
///
/// Manages the low-rank projection matrix and provides methods to project
/// and unproject gradients for memory-efficient training.
#[derive(Debug)]
pub struct GaloreProjector {
    /// Configuration.
    config: GaloreConfig,
    /// Current orthogonal projection matrix.
    /// Shape depends on projection direction:
    /// - Right: [n, rank] (columns of V)
    /// - Left: [m, rank] (columns of U)
    ortho_matrix: Option<Array>,
    /// Whether using right projection (true) or left projection (false).
    use_right_projection: bool,
    /// Current iteration count for determining when to update projector.
    iteration: usize,
    /// Original gradient shape [m, n].
    grad_shape: Option<(i32, i32)>,
}

impl GaloreProjector {
    /// Create a new GaLore projector with the given configuration.
    pub fn new(config: GaloreConfig) -> Self {
        Self {
            config,
            ortho_matrix: None,
            use_right_projection: true,
            iteration: 0,
            grad_shape: None,
        }
    }

    /// Create a new projector with default configuration and given rank.
    pub fn with_rank(rank: usize) -> Self {
        Self::new(GaloreConfig::with_rank(rank))
    }

    /// Get the configuration.
    pub fn config(&self) -> &GaloreConfig {
        &self.config
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Check if the projector needs to be updated this iteration.
    fn needs_update(&self) -> bool {
        self.ortho_matrix.is_none() || self.iteration % self.config.update_proj_gap == 0
    }

    /// Determine projection direction based on gradient shape and config.
    fn determine_projection_direction(&self, m: i32, n: i32) -> bool {
        match self.config.proj_type {
            GaloreProjectionType::Standard => m >= n,
            GaloreProjectionType::ReverseStandard => m < n,
            GaloreProjectionType::Right => true,
            GaloreProjectionType::Left => false,
        }
    }

    /// Update the orthogonal projection matrix using SVD.
    ///
    /// Computes SVD of the gradient and extracts the appropriate singular vectors
    /// based on the projection direction.
    fn update_ortho_matrix(&mut self, grad: &Array) -> Result<(), LoraError> {
        let shape = grad.shape();
        if shape.len() != 2 {
            return Err(LoraError::ShapeMismatch(format!(
                "GaLore requires 2D gradients, got shape {:?}",
                shape
            )));
        }

        let m = shape[0];
        let n = shape[1];
        self.grad_shape = Some((m, n));

        // Determine projection direction
        self.use_right_projection = self.determine_projection_direction(m, n);

        // Compute SVD on CPU (GPU SVD not yet implemented in MLX)
        // G = U @ diag(S) @ Vt
        let (u, _s, vt) = mlx_rs::linalg::svd_device(grad, StreamOrDevice::cpu())?;

        let rank = self.config.rank as i32;

        if self.use_right_projection {
            // Right projection: use columns of V (rows of Vt)
            // P = V[:, :rank] = Vt[:rank, :].T
            // Shape: [n, rank]
            let vt_truncated = vt.index((..rank, ..));
            self.ortho_matrix = Some(vt_truncated.t());
        } else {
            // Left projection: use columns of U
            // P = U[:, :rank]
            // Shape: [m, rank]
            self.ortho_matrix = Some(u.index((.., ..rank)));
        }

        Ok(())
    }

    /// Project a gradient to low-rank space.
    ///
    /// # Arguments
    /// * `grad` - Full gradient tensor of shape [m, n]
    ///
    /// # Returns
    /// Low-rank gradient:
    /// - Right projection: [m, rank] = grad @ P
    /// - Left projection: [rank, n] = P^T @ grad
    pub fn project(&mut self, grad: &Array) -> Result<Array, LoraError> {
        // Update projector if needed
        if self.needs_update() {
            self.update_ortho_matrix(grad)?;
        }

        let ortho = self
            .ortho_matrix
            .as_ref()
            .ok_or_else(|| LoraError::InvalidState("Projector not initialized".into()))?;

        let low_rank = if self.use_right_projection {
            // Right: low_rank = grad @ P, shape [m, rank]
            grad.matmul(ortho)?
        } else {
            // Left: low_rank = P^T @ grad, shape [rank, n]
            ortho.t().matmul(grad)?
        };

        self.iteration += 1;
        Ok(low_rank)
    }

    /// Unproject a low-rank gradient back to full space.
    ///
    /// # Arguments
    /// * `low_rank_grad` - Low-rank gradient from optimizer
    ///
    /// # Returns
    /// Full gradient of shape [m, n]:
    /// - Right projection: low_rank @ P^T
    /// - Left projection: P @ low_rank
    pub fn unproject(&self, low_rank_grad: &Array) -> Result<Array, LoraError> {
        let ortho = self
            .ortho_matrix
            .as_ref()
            .ok_or_else(|| LoraError::InvalidState("Projector not initialized".into()))?;

        let full_grad = if self.use_right_projection {
            // Right: full = low_rank @ P^T, shape [m, n]
            low_rank_grad.matmul(&ortho.t())?
        } else {
            // Left: full = P @ low_rank, shape [m, n]
            ortho.matmul(low_rank_grad)?
        };

        // Apply scale factor
        if (self.config.scale - 1.0).abs() > 1e-6 {
            let scale = Array::from_f32(self.config.scale);
            Ok(full_grad.multiply(&scale)?)
        } else {
            Ok(full_grad)
        }
    }

    /// Project and return both the low-rank gradient and projector for external use.
    ///
    /// Useful when the optimizer needs direct access to the projection matrix.
    pub fn project_with_state(&mut self, grad: &Array) -> Result<GaloreProjectionState, LoraError> {
        if self.needs_update() {
            self.update_ortho_matrix(grad)?;
        }

        let ortho = self
            .ortho_matrix
            .as_ref()
            .ok_or_else(|| LoraError::InvalidState("Projector not initialized".into()))?;

        let low_rank = if self.use_right_projection {
            grad.matmul(ortho)?
        } else {
            ortho.t().matmul(grad)?
        };

        self.iteration += 1;

        Ok(GaloreProjectionState {
            low_rank_grad: low_rank,
            projector: ortho.clone(),
            use_right_projection: self.use_right_projection,
            scale: self.config.scale,
        })
    }

    /// Get the projected gradient shape for a given full gradient shape.
    pub fn projected_shape(&self, m: i32, n: i32) -> (i32, i32) {
        let rank = self.config.rank as i32;
        let use_right = self.determine_projection_direction(m, n);
        if use_right {
            (m, rank) // [m, rank]
        } else {
            (rank, n) // [rank, n]
        }
    }

    /// Calculate memory savings ratio compared to full gradient.
    pub fn memory_savings_ratio(&self, m: usize, n: usize) -> f64 {
        let full_size = m * n;
        let (pm, pn) = self.projected_shape(m as i32, n as i32);
        let projected_size = (pm * pn) as usize;
        1.0 - (projected_size as f64 / full_size as f64)
    }
}

/// State from a GaLore projection operation.
#[derive(Debug)]
pub struct GaloreProjectionState {
    /// Low-rank projected gradient.
    pub low_rank_grad: Array,
    /// Orthogonal projection matrix used.
    pub projector: Array,
    /// Whether right projection was used.
    pub use_right_projection: bool,
    /// Scale factor for unprojection.
    pub scale: f32,
}

impl GaloreProjectionState {
    /// Unproject the low-rank gradient back to full space.
    pub fn unproject(&self) -> Result<Array, LoraError> {
        let full_grad = if self.use_right_projection {
            self.low_rank_grad.matmul(&self.projector.t())?
        } else {
            self.projector.matmul(&self.low_rank_grad)?
        };

        if (self.scale - 1.0).abs() > 1e-6 {
            let scale = Array::from_f32(self.scale);
            Ok(full_grad.multiply(&scale)?)
        } else {
            Ok(full_grad)
        }
    }
}

/// GaLore-wrapped optimizer state for a single parameter.
///
/// This wraps an optimizer's state to work in the projected space.
#[derive(Debug)]
pub struct GaloreParamState {
    /// The projector for this parameter.
    pub projector: GaloreProjector,
    /// First moment (mean) in projected space.
    pub m: Option<Array>,
    /// Second moment (variance) in projected space.
    pub v: Option<Array>,
}

impl GaloreParamState {
    /// Create a new GaLore parameter state.
    pub fn new(config: GaloreConfig) -> Self {
        Self {
            projector: GaloreProjector::new(config),
            m: None,
            v: None,
        }
    }

    /// Apply GaLore-Adam update to a parameter.
    ///
    /// # Arguments
    /// * `param` - Current parameter value
    /// * `grad` - Full gradient for the parameter
    /// * `lr` - Learning rate
    /// * `beta1` - First moment decay (default: 0.9)
    /// * `beta2` - Second moment decay (default: 0.999)
    /// * `eps` - Numerical stability epsilon (default: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (default: 0.0)
    ///
    /// # Returns
    /// Updated parameter value
    pub fn adam_step(
        &mut self,
        param: &Array,
        grad: &Array,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<Array, LoraError> {
        // Project gradient to low-rank space
        let low_rank_grad = self.projector.project(grad)?;

        // Get or initialize moments in projected space
        let step = self.projector.iteration() as f32;

        // Initialize moments if needed
        if self.m.is_none() {
            self.m = Some(mlx_rs::ops::zeros_like(&low_rank_grad)?);
            self.v = Some(mlx_rs::ops::zeros_like(&low_rank_grad)?);
        }

        let m = self.m.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();

        // Adam update in projected space
        // m = beta1 * m + (1 - beta1) * g
        let beta1_arr = Array::from_f32(beta1);
        let one_minus_beta1 = Array::from_f32(1.0 - beta1);
        let new_m = m
            .multiply(&beta1_arr)?
            .add(&low_rank_grad.multiply(&one_minus_beta1)?)?;

        // v = beta2 * v + (1 - beta2) * g^2
        let beta2_arr = Array::from_f32(beta2);
        let one_minus_beta2 = Array::from_f32(1.0 - beta2);
        let grad_sq = low_rank_grad.multiply(&low_rank_grad)?;
        let new_v = v
            .multiply(&beta2_arr)?
            .add(&grad_sq.multiply(&one_minus_beta2)?)?;

        // Bias correction
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);
        let bc1_arr = Array::from_f32(bias_correction1);
        let bc2_arr = Array::from_f32(bias_correction2);

        let m_hat = new_m.divide(&bc1_arr)?;
        let v_hat = new_v.divide(&bc2_arr)?;

        // Compute update in projected space
        let eps_arr = Array::from_f32(eps);
        let v_sqrt = v_hat.sqrt()?;
        let denom = v_sqrt.add(&eps_arr)?;
        let low_rank_update = m_hat.divide(&denom)?;

        // Unproject to full space
        let full_update = self.projector.unproject(&low_rank_update)?;

        // Apply learning rate
        let lr_arr = Array::from_f32(lr);
        let scaled_update = full_update.multiply(&lr_arr)?;

        // Apply weight decay (decoupled)
        let new_param = if weight_decay > 0.0 {
            let wd = Array::from_f32(1.0 - lr * weight_decay);
            param.multiply(&wd)?.subtract(&scaled_update)?
        } else {
            param.subtract(&scaled_update)?
        };

        // Store updated moments
        self.m = Some(new_m);
        self.v = Some(new_v);

        Ok(new_param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galore_config_default() {
        let config = GaloreConfig::default();
        assert_eq!(config.rank, 128);
        assert_eq!(config.update_proj_gap, 200);
        assert_eq!(config.proj_type, GaloreProjectionType::Standard);
        assert!((config.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_galore_config_builder() {
        let config = GaloreConfig::with_rank(64)
            .update_gap(100)
            .projection_type(GaloreProjectionType::Right)
            .scale(0.5);

        assert_eq!(config.rank, 64);
        assert_eq!(config.update_proj_gap, 100);
        assert_eq!(config.proj_type, GaloreProjectionType::Right);
        assert!((config.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_projector_right_projection() {
        // For m >= n, should use right projection
        let config = GaloreConfig::with_rank(4);
        let mut projector = GaloreProjector::new(config);

        // Create a gradient where m >= n (10 x 8)
        let grad = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();

        let low_rank = projector.project(&grad).unwrap();

        // Right projection: [m, rank] = [10, 4]
        assert_eq!(low_rank.shape(), &[10, 4]);
        assert!(projector.use_right_projection);

        // Unproject back
        let reconstructed = projector.unproject(&low_rank).unwrap();
        assert_eq!(reconstructed.shape(), &[10, 8]);
    }

    #[test]
    fn test_projector_left_projection() {
        // For m < n, should use left projection (standard mode)
        let config = GaloreConfig::with_rank(4);
        let mut projector = GaloreProjector::new(config);

        // Create a gradient where m < n (8 x 10)
        let grad = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[8, 10], None).unwrap();

        let low_rank = projector.project(&grad).unwrap();

        // Left projection: [rank, n] = [4, 10]
        assert_eq!(low_rank.shape(), &[4, 10]);
        assert!(!projector.use_right_projection);

        // Unproject back
        let reconstructed = projector.unproject(&low_rank).unwrap();
        assert_eq!(reconstructed.shape(), &[8, 10]);
    }

    #[test]
    fn test_projector_forced_direction() {
        // Force right projection even when m < n
        let config = GaloreConfig::with_rank(4).projection_type(GaloreProjectionType::Right);
        let mut projector = GaloreProjector::new(config);

        let grad = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[8, 10], None).unwrap();

        let low_rank = projector.project(&grad).unwrap();

        // Forced right: [m, rank] = [8, 4]
        assert_eq!(low_rank.shape(), &[8, 4]);
        assert!(projector.use_right_projection);
    }

    #[test]
    fn test_projector_update_interval() {
        let config = GaloreConfig::with_rank(4).update_gap(3);
        let mut projector = GaloreProjector::new(config);

        let grad1 = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();
        let grad2 = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();
        let grad3 = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();
        let grad4 = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();

        // First projection - initializes projector
        projector.project(&grad1).unwrap();
        assert_eq!(projector.iteration(), 1);
        let ortho1 = projector.ortho_matrix.clone();

        // Second - no update
        projector.project(&grad2).unwrap();
        assert_eq!(projector.iteration(), 2);

        // Third - no update
        projector.project(&grad3).unwrap();
        assert_eq!(projector.iteration(), 3);

        // Fourth (iteration 3, 3 % 3 == 0) - should update
        projector.project(&grad4).unwrap();
        assert_eq!(projector.iteration(), 4);
        // Projector should have been updated (different random gradients)
    }

    #[test]
    fn test_memory_savings_ratio() {
        let config = GaloreConfig::with_rank(128);
        let projector = GaloreProjector::new(config);

        // For a 4096 x 4096 weight matrix with rank 128
        // Full: 4096 * 4096 = 16,777,216
        // Projected (right): 4096 * 128 = 524,288
        // Savings: 1 - 524288/16777216 = 96.9%
        let savings = projector.memory_savings_ratio(4096, 4096);
        assert!(savings > 0.96);
        assert!(savings < 0.98);
    }

    #[test]
    fn test_projection_state() {
        let config = GaloreConfig::with_rank(4);
        let mut projector = GaloreProjector::new(config);

        let grad = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();

        let state = projector.project_with_state(&grad).unwrap();

        assert_eq!(state.low_rank_grad.shape(), &[10, 4]);
        assert_eq!(state.projector.shape(), &[8, 4]); // [n, rank] for right projection
        assert!(state.use_right_projection);

        // Unproject via state
        let reconstructed = state.unproject().unwrap();
        assert_eq!(reconstructed.shape(), &[10, 8]);
    }

    #[test]
    fn test_galore_param_state_init() {
        let config = GaloreConfig::with_rank(4);
        let state = GaloreParamState::new(config);

        assert!(state.m.is_none());
        assert!(state.v.is_none());
        assert_eq!(state.projector.iteration(), 0);
    }

    #[test]
    fn test_galore_adam_step() {
        let config = GaloreConfig::with_rank(4).update_gap(10);
        let mut state = GaloreParamState::new(config);

        // Create parameter and gradient
        let param = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();
        let grad = mlx_rs::random::uniform::<_, f32>(-0.1, 0.1, &[10, 8], None).unwrap();

        // Take a step
        let new_param = state
            .adam_step(
                &param, &grad, 0.001, // lr
                0.9,   // beta1
                0.999, // beta2
                1e-8,  // eps
                0.0,   // weight_decay
            )
            .unwrap();

        // Check shapes
        assert_eq!(new_param.shape(), param.shape());

        // Moments should now be initialized
        assert!(state.m.is_some());
        assert!(state.v.is_some());

        // Moments should be in projected space
        assert_eq!(state.m.as_ref().unwrap().shape(), &[10, 4]);
        assert_eq!(state.v.as_ref().unwrap().shape(), &[10, 4]);
    }

    #[test]
    fn test_galore_multiple_steps() {
        let config = GaloreConfig::with_rank(4).update_gap(5);
        let mut state = GaloreParamState::new(config);

        let mut param = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[10, 8], None).unwrap();

        // Take multiple steps
        for _ in 0..10 {
            let grad = mlx_rs::random::uniform::<_, f32>(-0.1, 0.1, &[10, 8], None).unwrap();

            param = state
                .adam_step(
                    &param, &grad, 0.001, 0.9, 0.999, 1e-8, 0.01, // with weight decay
                )
                .unwrap();
        }

        assert_eq!(state.projector.iteration(), 10);
        assert_eq!(param.shape(), &[10, 8]);
    }
}
