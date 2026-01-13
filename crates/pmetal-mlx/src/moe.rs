//! Mixture of Experts (MoE) implementation for MLX.
//!
//! MoE architectures use multiple "expert" networks and a router to select
//! which experts process each token. This enables scaling model capacity
//! without proportionally increasing compute.
//!
//! ## Supported Models
//!
//! - Mixtral (8x7B, 8x22B)
//! - Qwen2-MoE, Qwen3-MoE
//! - DeepSeek-MoE
//! - DBRX
//!
//! ## How It Works
//!
//! 1. **Router**: Computes scores for each expert per token
//! 2. **Top-K Selection**: Selects top-k experts (typically k=2)
//! 3. **Expert Computation**: Selected experts process tokens
//! 4. **Weighted Combination**: Outputs weighted by router scores
//!
//! ## Memory Efficiency
//!
//! MoE models have more parameters but similar activation memory to dense models
//! since only k experts are active per token.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::Module,
    nn::{self, Linear},
    ops::indexing::IndexOp,
    Array, Dtype,
};

/// Configuration for Mixture of Experts.
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate (FFN) dimension per expert.
    pub intermediate_size: i32,
    /// Number of experts.
    pub num_experts: usize,
    /// Number of experts to route to per token.
    pub num_experts_per_tok: usize,
    /// Whether to use aux loss for load balancing.
    pub use_aux_loss: bool,
    /// Aux loss coefficient.
    pub aux_loss_coef: f32,
    /// Router jitter noise (for training).
    pub router_jitter: f32,
    /// Normalize router weights.
    pub normalize_router_weights: bool,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_experts: 8,
            num_experts_per_tok: 2,
            use_aux_loss: true,
            aux_loss_coef: 0.01,
            router_jitter: 0.0,
            normalize_router_weights: true,
        }
    }
}

impl MoEConfig {
    /// Create a new MoE config.
    pub fn new(hidden_size: i32, intermediate_size: i32, num_experts: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_experts,
            ..Default::default()
        }
    }

    /// Set number of experts per token.
    pub fn with_num_experts_per_tok(mut self, k: usize) -> Self {
        self.num_experts_per_tok = k;
        self
    }

    /// Enable/disable auxiliary load balancing loss.
    pub fn with_aux_loss(mut self, use_aux: bool, coef: f32) -> Self {
        self.use_aux_loss = use_aux;
        self.aux_loss_coef = coef;
        self
    }

    /// Set router jitter for training.
    pub fn with_router_jitter(mut self, jitter: f32) -> Self {
        self.router_jitter = jitter;
        self
    }
}

/// Router for selecting experts.
///
/// Computes routing scores and selects top-k experts per token.
#[derive(Debug)]
pub struct MoERouter {
    /// Linear projection for routing scores.
    gate: Linear,
    /// Number of experts.
    num_experts: usize,
    /// Number of experts per token.
    num_experts_per_tok: usize,
    /// Router jitter noise.
    jitter: f32,
    /// Whether in training mode.
    training: bool,
}

impl MoERouter {
    /// Create a new router.
    pub fn new(hidden_size: i32, num_experts: usize, num_experts_per_tok: usize) -> Self {
        let gate = nn::LinearBuilder::new(hidden_size, num_experts as i32)
            .bias(false)
            .build()
            .unwrap();
        Self {
            gate,
            num_experts,
            num_experts_per_tok,
            jitter: 0.0,
            training: true,
        }
    }

    /// Set jitter noise.
    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter;
        self
    }

    /// Set training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Compute routing weights and expert indices.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq, hidden]
    ///
    /// # Returns
    /// (routing_weights, selected_experts, router_logits)
    /// - routing_weights: [batch * seq, num_experts_per_tok]
    /// - selected_experts: [batch * seq, num_experts_per_tok]
    /// - router_logits: [batch * seq, num_experts] (for aux loss)
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Array, Array), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];

        // Flatten to [batch * seq, hidden]
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size])?;

        // Compute router logits
        let mut router_logits = self.gate.forward(&hidden_flat)?;

        // Add jitter during training
        if self.training && self.jitter > 0.0 {
            let noise = mlx_rs::random::uniform::<_, f32>(
                -self.jitter,
                self.jitter,
                router_logits.shape(),
                None,
            )?;
            router_logits = router_logits.add(&noise)?;
        }

        // Softmax over experts (axis -1)
        let routing_weights = mlx_rs::ops::softmax_axis(&router_logits, -1, None)?;

        // Select top-k experts using argpartition approach
        let k = self.num_experts_per_tok as i32;

        // For simplicity, we'll use a custom top-k implementation
        // since mlx-rs topk might not be available
        let (top_weights, top_indices) = self.custom_topk(&routing_weights, k)?;

        // Normalize weights to sum to 1
        let weight_sum = top_weights.sum_axis(-1, Some(true))?;
        let normalized_weights = top_weights.divide(&weight_sum)?;

        Ok((normalized_weights, top_indices, router_logits))
    }

    /// Custom top-k implementation.
    fn custom_topk(&self, probs: &Array, k: i32) -> Result<(Array, Array), Exception> {
        probs.eval()?;
        let shape = probs.shape();
        let batch = shape[0] as usize;
        let n = shape[1] as usize;
        let k = k as usize;

        let probs_data: Vec<f32> = probs.as_slice().to_vec();

        let mut top_values = Vec::with_capacity(batch * k);
        let mut top_indices = Vec::with_capacity(batch * k);

        for b in 0..batch {
            let row = &probs_data[b * n..(b + 1) * n];

            // Get indices sorted by value (descending)
            let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for i in 0..k {
                top_values.push(indexed[i].1);
                top_indices.push(indexed[i].0 as i32);
            }
        }

        let values = Array::from_slice(&top_values, &[batch as i32, k as i32]);
        let indices = Array::from_slice(&top_indices, &[batch as i32, k as i32]);

        Ok((values, indices))
    }
}

/// Single expert MLP (SwiGLU).
#[derive(Debug)]
pub struct Expert {
    /// Gate projection.
    pub w1: Linear,
    /// Up projection.
    pub w3: Linear,
    /// Down projection.
    pub w2: Linear,
}

impl Expert {
    /// Create a new expert.
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Self {
        let w1 = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()
            .unwrap();
        let w3 = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()
            .unwrap();
        let w2 = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(false)
            .build()
            .unwrap();
        Self { w1, w3, w2 }
    }

    /// Forward pass with SwiGLU activation.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.w1.forward(x)?;
        let gate_activated = mlx_rs::ops::sigmoid(&gate)?.multiply(&gate)?;
        let up = self.w3.forward(x)?;
        let hidden = gate_activated.multiply(&up)?;
        self.w2.forward(&hidden)
    }
}

/// Mixture of Experts layer.
///
/// Routes tokens to top-k experts and combines their outputs.
#[derive(Debug)]
pub struct MoELayer {
    /// Router for expert selection.
    pub router: MoERouter,
    /// Expert MLPs.
    pub experts: Vec<Expert>,
    /// Configuration.
    pub config: MoEConfig,
}

impl MoELayer {
    /// Create a new MoE layer.
    pub fn new(config: MoEConfig) -> Self {
        let router = MoERouter::new(
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_tok,
        )
        .with_jitter(config.router_jitter);

        let experts = (0..config.num_experts)
            .map(|_| Expert::new(config.hidden_size, config.intermediate_size))
            .collect();

        Self {
            router,
            experts,
            config,
        }
    }

    /// Set training mode.
    pub fn train(&mut self) {
        self.router.train();
    }

    /// Set evaluation mode.
    pub fn eval(&mut self) {
        self.router.eval();
    }

    /// Forward pass through MoE layer.
    ///
    /// # Arguments
    /// * `hidden_states` - Input [batch, seq, hidden]
    ///
    /// # Returns
    /// (output, aux_loss) where aux_loss is for load balancing
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Option<Array>), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];

        // Get routing weights and expert indices
        let (routing_weights, selected_experts, router_logits) =
            self.router.forward(hidden_states)?;

        // Flatten input for expert processing
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size])?;

        // Initialize output accumulator
        let mut final_output = Array::zeros::<f32>(&[batch_seq, hidden_size])?;

        // Process each expert
        selected_experts.eval()?;
        routing_weights.eval()?;

        // For MLX efficiency, we process all tokens for each expert
        for expert_idx in 0..self.config.num_experts {
            // Find tokens routed to this expert
            let expert_mask = selected_experts.eq(Array::from_int(expert_idx as i32))?;

            // Check if any tokens go to this expert
            let any_tokens = expert_mask.any(None)?;
            any_tokens.eval()?;

            if any_tokens.item::<bool>() {
                let expert = &mut self.experts[expert_idx];

                // Process tokens through expert
                for k in 0..self.config.num_experts_per_tok {
                    let k_mask = expert_mask.index((.., k as i32));
                    let weight = routing_weights.index((.., k as i32));

                    // Compute expert output for all tokens
                    let expert_out = expert.forward(&hidden_flat)?;

                    // Weight the output
                    let weighted_out = expert_out.multiply(&weight.reshape(&[-1, 1])?)?;

                    // Mask to only include tokens routed to this expert
                    let mask_f32 = k_mask.reshape(&[-1, 1])?.as_dtype(Dtype::Float32)?;
                    let masked_out = weighted_out.multiply(&mask_f32)?;

                    final_output = final_output.add(&masked_out)?;
                }
            }
        }

        // Reshape back to original shape
        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        let output = final_output.reshape(&output_shape)?;

        // Compute auxiliary load balancing loss if enabled
        let aux_loss = if self.config.use_aux_loss {
            Some(self.compute_aux_loss(&router_logits)?)
        } else {
            None
        };

        Ok((output, aux_loss))
    }

    /// Compute auxiliary load balancing loss.
    fn compute_aux_loss(&self, router_logits: &Array) -> Result<Array, Exception> {
        let num_experts = self.config.num_experts as i32;

        // Compute routing probabilities
        let routing_probs = mlx_rs::ops::softmax_axis(router_logits, -1, None)?;

        // Mean routing probability per expert
        let mean_routing_prob = routing_probs.mean_axis(0, None)?;

        // Simplified aux loss: encourages uniform distribution
        // Loss = num_experts * variance(mean_probs)
        let target = Array::from_f32(1.0 / num_experts as f32);
        let diff = mean_routing_prob.subtract(&target)?;
        let variance = diff.square()?.mean(None)?;
        let aux_loss = variance.multiply(Array::from_f32(num_experts as f32))?;

        // Scale by coefficient
        aux_loss.multiply(Array::from_f32(self.config.aux_loss_coef))
    }
}

/// Sparse MoE with shared experts (DeepSeek style).
///
/// Combines routed experts with always-active shared experts.
#[derive(Debug)]
pub struct SparseMoEWithShared {
    /// Standard MoE layer.
    pub moe: MoELayer,
    /// Shared expert (always active).
    pub shared_expert: Option<Expert>,
    /// Weight for shared expert output.
    pub shared_weight: f32,
}

impl SparseMoEWithShared {
    /// Create sparse MoE with optional shared expert.
    pub fn new(config: MoEConfig, use_shared: bool) -> Self {
        let moe = MoELayer::new(config.clone());
        let shared_expert = if use_shared {
            Some(Expert::new(config.hidden_size, config.intermediate_size))
        } else {
            None
        };

        Self {
            moe,
            shared_expert,
            shared_weight: 1.0,
        }
    }

    /// Set shared expert weight.
    pub fn with_shared_weight(mut self, weight: f32) -> Self {
        self.shared_weight = weight;
        self
    }

    /// Forward pass.
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Option<Array>), Exception> {
        let (moe_out, aux_loss) = self.moe.forward(hidden_states)?;

        if let Some(ref mut shared) = self.shared_expert {
            let shared_out = shared.forward(hidden_states)?;
            let weighted_shared = shared_out.multiply(Array::from_f32(self.shared_weight))?;
            let combined = moe_out.add(&weighted_shared)?;
            Ok((combined, aux_loss))
        } else {
            Ok((moe_out, aux_loss))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_config_default() {
        let config = MoEConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.num_experts_per_tok, 2);
        assert!(config.use_aux_loss);
    }

    #[test]
    fn test_moe_config_builder() {
        let config = MoEConfig::new(2048, 8192, 16)
            .with_num_experts_per_tok(4)
            .with_aux_loss(false, 0.0);

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_experts, 16);
        assert_eq!(config.num_experts_per_tok, 4);
        assert!(!config.use_aux_loss);
    }

    #[test]
    fn test_router_forward() {
        let mut router = MoERouter::new(64, 8, 2);
        router.eval();

        let hidden = Array::zeros::<f32>(&[2, 4, 64]).unwrap();
        let (weights, indices, logits) = router.forward(&hidden).unwrap();

        weights.eval().unwrap();
        indices.eval().unwrap();
        logits.eval().unwrap();

        // weights: [batch*seq, k] = [8, 2]
        assert_eq!(weights.shape(), &[8, 2]);
        // indices: [batch*seq, k] = [8, 2]
        assert_eq!(indices.shape(), &[8, 2]);
        // logits: [batch*seq, num_experts] = [8, 8]
        assert_eq!(logits.shape(), &[8, 8]);
    }

    #[test]
    fn test_expert_forward() {
        let mut expert = Expert::new(64, 256);
        let x = Array::zeros::<f32>(&[4, 64]).unwrap();

        let out = expert.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[4, 64]);
    }

    #[test]
    fn test_moe_layer_forward() {
        let config = MoEConfig::new(64, 128, 4)
            .with_num_experts_per_tok(2)
            .with_aux_loss(false, 0.0);

        let mut moe = MoELayer::new(config);
        moe.eval();

        let hidden = Array::zeros::<f32>(&[2, 4, 64]).unwrap();
        let (output, aux_loss) = moe.forward(&hidden).unwrap();

        output.eval().unwrap();

        assert_eq!(output.shape(), &[2, 4, 64]);
        assert!(aux_loss.is_none());
    }

    #[test]
    fn test_moe_layer_with_aux_loss() {
        let config = MoEConfig::new(64, 128, 4)
            .with_num_experts_per_tok(2)
            .with_aux_loss(true, 0.01);

        let mut moe = MoELayer::new(config);
        moe.eval();

        let hidden = Array::zeros::<f32>(&[2, 4, 64]).unwrap();
        let (output, aux_loss) = moe.forward(&hidden).unwrap();

        output.eval().unwrap();
        assert!(aux_loss.is_some());

        let loss = aux_loss.unwrap();
        loss.eval().unwrap();
        assert_eq!(loss.ndim(), 0); // Scalar
    }

    #[test]
    fn test_sparse_moe_with_shared() {
        let config = MoEConfig::new(64, 128, 4).with_num_experts_per_tok(2);

        let mut moe = SparseMoEWithShared::new(config, true);
        moe.moe.eval();

        let hidden = Array::zeros::<f32>(&[2, 4, 64]).unwrap();
        let (output, _) = moe.forward(&hidden).unwrap();

        output.eval().unwrap();
        assert_eq!(output.shape(), &[2, 4, 64]);
    }

    #[test]
    fn test_router_weights_normalized() {
        let mut router = MoERouter::new(64, 8, 2);
        router.eval();

        let hidden = mlx_rs::random::normal::<f32>(&[4, 64], None, None, None).unwrap();
        let (weights, _, _) = router.forward(&hidden).unwrap();

        weights.eval().unwrap();

        // Sum of weights per token should be ~1.0
        let weight_sums = weights.sum_axis(-1, None).unwrap();
        weight_sums.eval().unwrap();

        let sums: Vec<f32> = weight_sums.as_slice().to_vec();
        for sum in sums {
            assert!((sum - 1.0).abs() < 0.01);
        }
    }
}
