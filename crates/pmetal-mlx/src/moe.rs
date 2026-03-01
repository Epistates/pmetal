//! Mixture of Experts (MoE) implementation for MLX.
//!
//! MoE architectures use multiple "expert" networks and a router to select
//! which experts process each token. This enables scaling model capacity
//! without proportionally increasing compute.

use mlx_rs::{
    Array, Dtype,
    builder::Builder,
    error::Exception,
    module::Module,
    nn::{self, Linear},
    ops::indexing::IndexOp,
    macros::ModuleParameters,
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
        Self { hidden_size, intermediate_size, num_experts, ..Default::default() }
    }
    pub fn with_num_experts_per_tok(mut self, k: usize) -> Self { self.num_experts_per_tok = k; self }
    pub fn with_aux_loss(mut self, use_aux: bool, coef: f32) -> Self { self.use_aux_loss = use_aux; self.aux_loss_coef = coef; self }
    pub fn with_router_jitter(mut self, jitter: f32) -> Self { self.router_jitter = jitter; self }
}

/// Router for selecting experts.
#[derive(Debug, ModuleParameters)]
pub struct MoERouter {
    #[param] pub gate: Linear,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub jitter: f32,
    pub training: bool,
}

impl MoERouter {
    pub fn new(hidden_size: i32, num_experts: usize, num_experts_per_tok: usize) -> Self {
        let gate = nn::LinearBuilder::new(hidden_size, num_experts as i32).bias(false).build().unwrap();
        Self { gate, num_experts, num_experts_per_tok, jitter: 0.0, training: true }
    }
    pub fn with_jitter(mut self, jitter: f32) -> Self { self.jitter = jitter; self }
    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self) { self.training = false; }
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Array, Array), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size])?;
        let mut router_logits = self.gate.forward(&hidden_flat)?;
        if self.training && self.jitter > 0.0 {
            let noise = mlx_rs::random::uniform::<_, f32>(-self.jitter, self.jitter, router_logits.shape(), None)?;
            router_logits = router_logits.add(&noise)?;
        }
        let routing_weights = mlx_rs::ops::softmax_axis(&router_logits, -1, None)?;
        let k = self.num_experts_per_tok as i32;
        let (top_weights, top_indices) = self.custom_topk(&routing_weights, k)?;
        let weight_sum = top_weights.sum_axis(-1, Some(true))?;
        let normalized_weights = top_weights.divide(&weight_sum)?;
        Ok((normalized_weights, top_indices, router_logits))
    }
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
            let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for i in 0..k { top_values.push(indexed[i].1); top_indices.push(indexed[i].0 as i32); }
        }
        let values = Array::from_slice(&top_values, &[batch as i32, k as i32]);
        let indices = Array::from_slice(&top_indices, &[batch as i32, k as i32]);
        Ok((values, indices))
    }
}

/// Single expert MLP (SwiGLU).
#[derive(Debug, ModuleParameters)]
pub struct Expert {
    #[param] pub w1: Linear,
    #[param] pub w3: Linear,
    #[param] pub w2: Linear,
}

impl Expert {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Self {
        let w1 = nn::LinearBuilder::new(hidden_size, intermediate_size).bias(false).build().unwrap();
        let w3 = nn::LinearBuilder::new(hidden_size, intermediate_size).bias(false).build().unwrap();
        let w2 = nn::LinearBuilder::new(intermediate_size, hidden_size).bias(false).build().unwrap();
        Self { w1, w3, w2 }
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.w1.forward(x)?;
        let gate_activated = mlx_rs::ops::sigmoid(&gate)?.multiply(&gate)?;
        let up = self.w3.forward(x)?;
        let hidden = gate_activated.multiply(&up)?;
        self.w2.forward(&hidden)
    }
}

/// Mixture of Experts layer.
#[derive(Debug, ModuleParameters)]
pub struct MoELayer {
    #[param] pub router: MoERouter,
    #[param] pub experts: Vec<Expert>,
    pub config: MoEConfig,
}

impl MoELayer {
    pub fn new(config: MoEConfig) -> Self {
        let router = MoERouter::new(config.hidden_size, config.num_experts, config.num_experts_per_tok).with_jitter(config.router_jitter);
        let experts = (0..config.num_experts).map(|_| Expert::new(config.hidden_size, config.intermediate_size)).collect();
        Self { router, experts, config }
    }
    pub fn train(&mut self) { self.router.train(); }
    pub fn eval(&mut self) { self.router.eval(); }
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Option<Array>), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];
        let (routing_weights, selected_experts, router_logits) = self.router.forward(hidden_states)?;
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size])?;
        let mut final_output = Array::zeros::<f32>(&[batch_seq, hidden_size])?;
        selected_experts.eval()?;
        routing_weights.eval()?;
        for expert_idx in 0..self.config.num_experts {
            let expert_mask = selected_experts.eq(Array::from_int(expert_idx as i32))?;
            let any_tokens = expert_mask.any(None)?;
            any_tokens.eval()?;
            if any_tokens.item::<bool>() {
                let expert = &mut self.experts[expert_idx];
                for k in 0..self.config.num_experts_per_tok {
                    let k_mask = expert_mask.index((.., k as i32));
                    let weight = routing_weights.index((.., k as i32));
                    let expert_out = expert.forward(&hidden_flat)?;
                    let weighted_out = expert_out.multiply(&weight.reshape(&[-1, 1])?)?;
                    let mask_f32 = k_mask.reshape(&[-1, 1])?.as_dtype(Dtype::Float32)?;
                    let masked_out = weighted_out.multiply(&mask_f32)?;
                    final_output = final_output.add(&masked_out)?;
                }
            }
        }
        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        let output = final_output.reshape(&output_shape)?;
        let aux_loss = if self.config.use_aux_loss { Some(self.compute_aux_loss(&router_logits)?) } else { None };
        Ok((output, aux_loss))
    }
    fn compute_aux_loss(&self, router_logits: &Array) -> Result<Array, Exception> {
        let num_experts = self.config.num_experts as i32;
        let routing_probs = mlx_rs::ops::softmax_axis(router_logits, -1, None)?;
        let mean_routing_prob = routing_probs.mean_axis(0, None)?;
        let target = Array::from_f32(1.0 / num_experts as f32);
        let diff = mean_routing_prob.subtract(&target)?;
        let variance = diff.square()?.mean(None)?;
        let aux_loss = variance.multiply(Array::from_f32(num_experts as f32))?;
        aux_loss.multiply(Array::from_f32(self.config.aux_loss_coef))
    }
}
