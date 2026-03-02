//! Mixture of Experts (MoE) implementation for MLX.
//!
//! MoE architectures use multiple "expert" networks and a router to select
//! which experts process each token. This enables scaling model capacity
//! without proportionally increasing compute.
//!
//! ## Key design decisions
//! - Router logits are cast to float32 before softmax for numerical stability.
//! - Top-k uses GPU-native argsort + slice (no CPU round-trip).
//! - Per-expert dispatch: only assigned tokens are fed to each expert.
//! - Aux loss follows the Switch Transformer formula: `N * sum(f_i * P_i)`.

#![allow(missing_docs)]

use mlx_rs::{
    Array, Dtype,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn::{self, Linear},
    ops::indexing::IndexOp,
};

/// Configuration for Mixture of Experts.
#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub use_aux_loss: bool,
    pub aux_loss_coef: f32,
    pub router_jitter: f32,
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
    pub fn new(hidden_size: i32, intermediate_size: i32, num_experts: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_experts,
            ..Default::default()
        }
    }
    pub fn with_num_experts_per_tok(mut self, k: usize) -> Self {
        self.num_experts_per_tok = k;
        self
    }
    pub fn with_aux_loss(mut self, use_aux: bool, coef: f32) -> Self {
        self.use_aux_loss = use_aux;
        self.aux_loss_coef = coef;
        self
    }
    pub fn with_router_jitter(mut self, jitter: f32) -> Self {
        self.router_jitter = jitter;
        self
    }
}

/// Router for selecting experts.
///
/// Returns normalized top-k routing weights, top-k expert indices, and raw logits.
#[derive(Debug, ModuleParameters)]
pub struct MoERouter {
    #[param]
    pub gate: Linear,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub jitter: f32,
    pub training: bool,
}

impl MoERouter {
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

    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter;
        self
    }
    pub fn train(&mut self) {
        self.training = true;
    }
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Forward: compute routing weights and expert assignments.
    ///
    /// Returns `(normalized_weights [N, k], top_indices [N, k], router_logits [N, E])`.
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Array, Array), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size])?;

        let mut router_logits = self.gate.forward(&hidden_flat)?;

        // Add jitter noise during training for load balancing
        if self.training && self.jitter > 0.0 {
            let noise = mlx_rs::random::uniform::<_, f32>(
                -self.jitter,
                self.jitter,
                router_logits.shape(),
                None,
            )?;
            router_logits = router_logits.add(&noise)?;
        }

        // M6: Cast to float32 before softmax for numerical stability
        let router_logits_f32 = if router_logits.dtype() != Dtype::Float32 {
            router_logits.as_type::<f32>()?
        } else {
            router_logits.clone()
        };
        let routing_weights = mlx_rs::ops::softmax_axis(&router_logits_f32, -1, None)?;

        // C10: GPU-native top-k using argsort + slice (no CPU round-trip, no NaN panic)
        let k = self.num_experts_per_tok;
        let (top_weights, top_indices) = gpu_topk(&routing_weights, k)?;

        // Normalize selected weights to sum to 1
        let weight_sum = top_weights.sum_axis(-1, Some(true))?;
        let safe_sum = mlx_rs::ops::maximum(&weight_sum, &Array::from_f32(1e-8))?;
        let normalized_weights = top_weights.divide(&safe_sum)?;

        Ok((normalized_weights, top_indices, router_logits))
    }
}

/// GPU-native top-k selection using argsort + slice.
///
/// Returns the top-k values and indices from `probs` along the last axis.
/// This avoids the CPU round-trip and NaN panic of the previous `custom_topk`.
fn gpu_topk(probs: &Array, k: usize) -> Result<(Array, Array), Exception> {
    // Negate for descending sort, then argsort on GPU
    let neg_probs = probs.negative()?;
    let sorted_indices = mlx_rs::ops::argsort_axis(&neg_probs, -1)?;

    // Slice first k (the largest values)
    let top_indices = sorted_indices.index((.., ..k as i32));

    // Gather top-k values
    let top_values = probs.take_along_axis(&top_indices, -1)?;

    Ok((top_values, top_indices))
}

/// Single expert MLP (SwiGLU).
#[derive(Debug, ModuleParameters)]
pub struct Expert {
    #[param]
    pub w1: Linear,
    #[param]
    pub w3: Linear,
    #[param]
    pub w2: Linear,
}

impl Expert {
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

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.w1.forward(x)?;
        // SwiGLU: silu(gate) * up
        let gate_activated = mlx_rs::nn::silu(&gate)?;
        let up = self.w3.forward(x)?;
        let hidden = gate_activated.multiply(&up)?;
        self.w2.forward(&hidden)
    }
}

/// Mixture of Experts layer.
///
/// Routes tokens to top-k experts, runs only assigned tokens per expert,
/// and accumulates weighted outputs.
#[derive(Debug, ModuleParameters)]
pub struct MoELayer {
    #[param]
    pub router: MoERouter,
    #[param]
    pub experts: Vec<Expert>,
    pub config: MoEConfig,
}

impl MoELayer {
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

    pub fn train(&mut self) {
        self.router.train();
    }
    pub fn eval(&mut self) {
        self.router.eval();
    }

    /// Forward pass: route tokens to experts and accumulate outputs.
    ///
    /// C11: Per-expert dispatch — each expert only processes its assigned tokens.
    /// Uses eval + CPU index extraction to gather token indices per expert,
    /// matching the mlx-examples pattern (small routing tensor round-trip is
    /// negligible compared to the expert MLP compute savings).
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Option<Array>), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];

        let (routing_weights, selected_experts, router_logits) =
            self.router.forward(hidden_states)?;
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size])?;

        // Eval routing tensors to CPU for index extraction
        // (small tensors: [batch_seq, k] — negligible transfer cost)
        selected_experts.eval()?;
        routing_weights.eval()?;

        let n_tokens = batch_seq as usize;
        let k = self.config.num_experts_per_tok;
        let expert_indices: Vec<i32> = selected_experts.as_slice().to_vec();
        let expert_weights: Vec<f32> = routing_weights.as_slice().to_vec();

        // Build per-expert token lists: (token_idx, routing_weight)
        let mut expert_assignments: Vec<Vec<(usize, f32)>> =
            vec![Vec::new(); self.config.num_experts];
        for token_idx in 0..n_tokens {
            for slot in 0..k {
                let flat_idx = token_idx * k + slot;
                let expert_id = expert_indices[flat_idx] as usize;
                let weight = expert_weights[flat_idx];
                if expert_id < self.config.num_experts {
                    expert_assignments[expert_id].push((token_idx, weight));
                }
            }
        }

        // C11: Run each expert only on its assigned tokens
        let mut final_output = Array::zeros::<f32>(&[batch_seq, hidden_size])?;

        for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }

            // Gather assigned token indices and weights
            let token_indices: Vec<i32> = assignments.iter().map(|&(idx, _)| idx as i32).collect();
            let weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();

            let idx_array = Array::from_slice(&token_indices, &[token_indices.len() as i32]);
            let weight_array = Array::from_slice(&weights, &[weights.len() as i32, 1]);

            // Gather only the assigned tokens
            let expert_input = hidden_flat.take_axis(&idx_array, 0)?;

            // Run expert only on assigned tokens
            let expert_out = self.experts[expert_idx].forward(&expert_input)?;

            // Weight the output
            let weighted_out = expert_out.multiply(&weight_array)?;

            // Scatter back using scatter_add
            let idx_2d = idx_array.reshape(&[-1, 1])?;
            let idx_broadcast = mlx_rs::ops::broadcast_to(&idx_2d, weighted_out.shape())?;
            final_output = mlx_rs::ops::indexing::scatter_add_single(
                &final_output,
                &idx_broadcast,
                &weighted_out,
                0,
            )?;
        }

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        let output = final_output.reshape(&output_shape)?;

        // Compute auxiliary loss if enabled
        let aux_loss = if self.config.use_aux_loss {
            Some(self.compute_aux_loss(&router_logits, &selected_experts, n_tokens)?)
        } else {
            None
        };

        Ok((output, aux_loss))
    }

    /// M7: Compute auxiliary load-balancing loss using Switch Transformer formula.
    ///
    /// `aux_loss = num_experts * sum(f_i * P_i)`
    ///
    /// Where:
    /// - `f_i` = fraction of tokens dispatched to expert i
    /// - `P_i` = mean routing probability for expert i
    fn compute_aux_loss(
        &self,
        router_logits: &Array,
        selected_experts: &Array,
        num_tokens: usize,
    ) -> Result<Array, Exception> {
        let num_experts = self.config.num_experts;
        let num_experts_i32 = num_experts as i32;

        // P_i: mean routing probability per expert across all tokens
        let routing_probs = mlx_rs::ops::softmax_axis(router_logits, -1, None)?;
        let mean_routing_prob = routing_probs.mean_axis(0, None)?; // [num_experts]

        // f_i: fraction of tokens dispatched to each expert
        // Count how many times each expert appears in selected_experts
        let mut dispatch_fractions = Vec::with_capacity(num_experts);
        let total_dispatches = (num_tokens * self.config.num_experts_per_tok) as f32;
        selected_experts.eval()?;
        let se_data: Vec<i32> = selected_experts.as_slice().to_vec();
        for e in 0..num_experts {
            let count = se_data.iter().filter(|&&x| x == e as i32).count();
            dispatch_fractions.push(count as f32 / total_dispatches.max(1.0));
        }
        let f_array = Array::from_slice(&dispatch_fractions, &[num_experts_i32]);

        // Switch Transformer aux loss: N * sum(f_i * P_i)
        let f_times_p = f_array.multiply(&mean_routing_prob)?;
        let aux_loss = f_times_p
            .sum(None)?
            .multiply(&Array::from_f32(num_experts as f32))?;

        aux_loss.multiply(&Array::from_f32(self.config.aux_loss_coef))
    }
}
