//! Complete Custom Training Step
//!
//! This module provides a full custom training step implementation that
//! bypasses MLX autodiff for significant memory savings.
//!
//! # Usage
//!
//! ```ignore
//! use pmetal_lora::custom_training_step::Qwen3CustomTrainer;
//!
//! let mut trainer = Qwen3CustomTrainer::new(&model_config, &lora_config, learning_rate);
//!
//! for batch in dataloader {
//!     let loss = trainer.training_step(&mut model, &batch.input_ids, &batch.labels)?;
//!     println!("Loss: {}", loss);
//! }
//! ```

use std::collections::HashMap;

use mlx_rs::{Array, nn};
use mlx_rs::module::Module;

use crate::autograd::{
    AccumulatedLoraGrads, LoraForwardSaved, LoraGradContext, LoraGrads,
    lora_forward_with_grad, lora_backward,
};
use crate::custom_backward::{
    AttentionSaved, RmsNormSaved, SiluSaved,
    attention_backward, attention_forward_with_grad,
    rmsnorm_backward, rmsnorm_forward_with_grad,
    silu_backward, silu_forward_with_grad,
};
use crate::custom_training::CustomLoraTrainer;
use crate::{LoraError, LoraLinear, Qwen3LoraForCausalLM};

/// Saved state for a single decoder layer during custom forward.
#[derive(Debug)]
pub struct Qwen3LayerSaved {
    /// Input to the layer (for residual backward).
    pub x: Array,
    /// Input norm saved state.
    pub input_norm: RmsNormSaved,
    /// Normalized input.
    pub x_normed: Array,
    /// Q projection saved.
    pub q_saved: LoraForwardSaved,
    /// K projection saved.
    pub k_saved: LoraForwardSaved,
    /// V projection saved.
    pub v_saved: LoraForwardSaved,
    /// Attention saved (Q, K, V after reshape/RoPE, weights).
    pub attn_saved: AttentionSaved,
    /// O projection saved.
    pub o_saved: LoraForwardSaved,
    /// After attention + residual.
    pub h: Array,
    /// Post-attention norm saved.
    pub post_attn_norm: RmsNormSaved,
    /// Normalized h.
    pub h_normed: Array,
    /// Gate projection saved.
    pub gate_saved: LoraForwardSaved,
    /// SiLU saved.
    pub silu_saved: SiluSaved,
    /// Up projection saved.
    pub up_saved: LoraForwardSaved,
    /// Gate activated (for multiply backward).
    pub gate_activated: Array,
    /// Up output (for multiply backward).
    pub up_out: Array,
    /// Down projection saved.
    pub down_saved: LoraForwardSaved,
}

/// Saved state for complete model forward.
#[derive(Debug)]
pub struct Qwen3ModelSaved {
    /// Per-layer saved states.
    pub layers: Vec<Qwen3LayerSaved>,
    /// Final hidden before norm.
    pub final_hidden_pre_norm: Array,
    /// Final norm saved.
    pub final_norm: RmsNormSaved,
}

/// Custom trainer for Qwen3 models using explicit gradient computation.
pub struct Qwen3CustomTrainer {
    /// Gradient context.
    ctx: LoraGradContext,
    /// Learning rate.
    learning_rate: f32,
    /// Number of attention heads.
    n_heads: i32,
    /// Number of KV heads.
    n_kv_heads: i32,
    /// Head dimension.
    head_dim: i32,
    /// Attention scale.
    scale: f32,
    /// Ignore index for loss.
    ignore_index: i64,
}

impl Qwen3CustomTrainer {
    /// Create a new custom trainer.
    pub fn new(
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        learning_rate: f32,
    ) -> Self {
        Self {
            ctx: LoraGradContext::new(),
            learning_rate,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            ignore_index: -100,
        }
    }

    /// Set ignore index for loss computation.
    pub fn with_ignore_index(mut self, ignore_index: i64) -> Self {
        self.ignore_index = ignore_index;
        self
    }

    /// Forward through a single LoRA linear with gradient tracking.
    fn lora_forward(
        &self,
        layer: &LoraLinear,
        x: &Array,
    ) -> Result<(Array, LoraForwardSaved), LoraError> {
        lora_forward_with_grad(
            x,
            &layer.weight,
            &layer.lora_a,
            &layer.lora_b,
            layer.scale,
            &self.ctx,
        ).map_err(LoraError::from)
    }

    /// Forward through attention with custom state saving.
    fn attention_forward(
        &self,
        q_proj: &LoraLinear,
        k_proj: &LoraLinear,
        v_proj: &LoraLinear,
        o_proj: &LoraLinear,
        rope: &mut nn::Rope,
        x: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, LoraForwardSaved, LoraForwardSaved, LoraForwardSaved, AttentionSaved, LoraForwardSaved), LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V with LoRA state saving
        let (q, q_saved) = self.lora_forward(q_proj, x)?;
        let (k, k_saved) = self.lora_forward(k_proj, x)?;
        let (v, v_saved) = self.lora_forward(v_proj, x)?;

        // Reshape for multi-head attention
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;

        // Attention with state saving
        let num_heads_per_kv = self.n_heads / self.n_kv_heads;
        let (attn_out, attn_saved) = attention_forward_with_grad(
            &q, &k, &v, mask, self.scale, num_heads_per_kv,
        )?;

        // Reshape back
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // O projection with LoRA state saving
        let (output, o_saved) = self.lora_forward(o_proj, &attn_out)?;

        Ok((output, q_saved, k_saved, v_saved, attn_saved, o_saved))
    }

    /// Forward through MLP with custom state saving.
    fn mlp_forward(
        &self,
        gate_proj: &LoraLinear,
        up_proj: &LoraLinear,
        down_proj: &LoraLinear,
        x: &Array,
    ) -> Result<(Array, LoraForwardSaved, SiluSaved, LoraForwardSaved, Array, Array, LoraForwardSaved), LoraError> {
        // Gate projection
        let (gate, gate_saved) = self.lora_forward(gate_proj, x)?;

        // SiLU activation
        let (gate_activated, silu_saved) = silu_forward_with_grad(&gate)?;

        // Up projection
        let (up_out, up_saved) = self.lora_forward(up_proj, x)?;

        // Element-wise multiply
        let hidden = gate_activated.multiply(&up_out)?;

        // Down projection
        let (output, down_saved) = self.lora_forward(down_proj, &hidden)?;

        Ok((output, gate_saved, silu_saved, up_saved, gate_activated, up_out, down_saved))
    }

    /// Forward through a single decoder layer with state saving.
    fn layer_forward(
        &self,
        layer: &mut crate::qwen3_lora::Qwen3LoraDecoderLayer,
        x: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Qwen3LayerSaved), LoraError> {
        // Input norm
        let input_norm_weight = layer.input_layernorm.weight.value.as_ref().clone();
        let (x_normed, input_norm_saved) = rmsnorm_forward_with_grad(
            x, &input_norm_weight, 1e-6,
        )?;

        // Attention
        let (attn_out, q_saved, k_saved, v_saved, attn_saved, o_saved) = self.attention_forward(
            &layer.self_attn.q_proj,
            &layer.self_attn.k_proj,
            &layer.self_attn.v_proj,
            &layer.self_attn.o_proj,
            &mut layer.self_attn.rope,
            &x_normed,
            mask,
        )?;

        // Residual
        let h = x.add(&attn_out)?;

        // Post-attention norm
        let post_attn_norm_weight = layer.post_attention_layernorm.weight.value.as_ref().clone();
        let (h_normed, post_attn_norm_saved) = rmsnorm_forward_with_grad(
            &h, &post_attn_norm_weight, 1e-6,
        )?;

        // MLP
        let (mlp_out, gate_saved, silu_saved, up_saved, gate_activated, up_out, down_saved) = self.mlp_forward(
            &layer.mlp.gate_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
            &h_normed,
        )?;

        // Residual
        let output = h.add(&mlp_out)?;

        let saved = Qwen3LayerSaved {
            x: x.clone(),
            input_norm: input_norm_saved,
            x_normed,
            q_saved,
            k_saved,
            v_saved,
            attn_saved,
            o_saved,
            h: h.clone(),
            post_attn_norm: post_attn_norm_saved,
            h_normed,
            gate_saved,
            silu_saved,
            up_saved,
            gate_activated,
            up_out,
            down_saved,
        };

        Ok((output, saved))
    }

    /// Backward through MLP.
    fn mlp_backward(
        &self,
        d_output: &Array,
        saved: &Qwen3LayerSaved,
    ) -> Result<(Array, LoraGrads, LoraGrads, LoraGrads), LoraError> {
        // Backward through down projection
        let down_grads = lora_backward(d_output, &saved.down_saved)?;
        let d_hidden = down_grads.d_x.as_ref()
            .ok_or_else(|| LoraError::Mlx(mlx_rs::error::Exception::custom("Expected d_x from down")))?;

        // Backward through multiply: d_gate_act = d_hidden * up_out, d_up_out = d_hidden * gate_act
        let d_gate_activated = d_hidden.multiply(&saved.up_out)?;
        let d_up_out = d_hidden.multiply(&saved.gate_activated)?;

        // Backward through SiLU
        let d_gate = silu_backward(&d_gate_activated, &saved.silu_saved)?;

        // Backward through gate and up projections
        let gate_grads = lora_backward(&d_gate, &saved.gate_saved)?;
        let up_grads = lora_backward(&d_up_out, &saved.up_saved)?;

        // Sum gradients to h_normed
        let d_h_normed = gate_grads.d_x.as_ref()
            .ok_or_else(|| LoraError::Mlx(mlx_rs::error::Exception::custom("Expected d_x from gate")))?
            .add(up_grads.d_x.as_ref()
                .ok_or_else(|| LoraError::Mlx(mlx_rs::error::Exception::custom("Expected d_x from up")))?)?;

        Ok((d_h_normed, gate_grads, up_grads, down_grads))
    }

    /// Backward through a single decoder layer.
    fn layer_backward(
        &self,
        d_output: &Array,
        saved: &Qwen3LayerSaved,
    ) -> Result<(Array, AccumulatedLoraGrads), LoraError> {
        let mut grads = AccumulatedLoraGrads::new();

        // d_output comes from next layer or loss
        // Layer has two residuals: output = h + mlp_out, h = x + attn_out

        // Backward through MLP residual
        let d_mlp_out = d_output;
        let d_h = d_output; // Residual passes gradient through

        // Backward through MLP
        let (d_h_normed, gate_grads, up_grads, down_grads) = self.mlp_backward(d_mlp_out, saved)?;

        // Backward through post-attention norm
        let d_h_from_norm = rmsnorm_backward(&d_h_normed, &saved.post_attn_norm)?;

        // Combine residual gradients
        let d_h_total = d_h.add(&d_h_from_norm)?;

        // Backward through attention residual
        let d_attn_out = &d_h_total;
        let d_x_from_residual = &d_h_total;

        // Backward through O projection
        let o_grads = lora_backward(d_attn_out, &saved.o_saved)?;
        let d_attn_out_reshaped = o_grads.d_x.as_ref()
            .ok_or_else(|| LoraError::Mlx(mlx_rs::error::Exception::custom("Expected d_x from o_proj")))?;

        // Backward through attention
        // Note: We need to reshape d_attn_out for attention backward
        let shape = d_attn_out_reshaped.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let d_attn_reshaped = d_attn_out_reshaped
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let (d_q, d_k, d_v) = attention_backward(&d_attn_reshaped, &saved.attn_saved)?;

        // Reshape gradients back for projection backward
        let d_q_flat = d_q.transpose_axes(&[0, 2, 1, 3])?.reshape(&[batch, seq_len, -1])?;
        let d_k_flat = d_k.transpose_axes(&[0, 2, 1, 3])?.reshape(&[batch, seq_len, -1])?;
        let d_v_flat = d_v.transpose_axes(&[0, 2, 1, 3])?.reshape(&[batch, seq_len, -1])?;

        // Backward through Q, K, V projections
        let q_grads = lora_backward(&d_q_flat, &saved.q_saved)?;
        let k_grads = lora_backward(&d_k_flat, &saved.k_saved)?;
        let v_grads = lora_backward(&d_v_flat, &saved.v_saved)?;

        // Sum gradients to x_normed
        let d_x_normed = q_grads.d_x.as_ref()
            .ok_or_else(|| LoraError::Mlx(mlx_rs::error::Exception::custom("Expected d_x from q")))?
            .add(k_grads.d_x.as_ref()
                .ok_or_else(|| LoraError::Mlx(mlx_rs::error::Exception::custom("Expected d_x from k")))?)?
            .add(v_grads.d_x.as_ref()
                .ok_or_else(|| LoraError::Mlx(mlx_rs::error::Exception::custom("Expected d_x from v")))?)?;

        // Backward through input norm
        let d_x_from_norm = rmsnorm_backward(&d_x_normed, &saved.input_norm)?;

        // Combine with residual
        let d_x = d_x_from_residual.add(&d_x_from_norm)?;

        // Accumulate all gradients
        grads.add_layer_grads("self_attn.q_proj", &q_grads);
        grads.add_layer_grads("self_attn.k_proj", &k_grads);
        grads.add_layer_grads("self_attn.v_proj", &v_grads);
        grads.add_layer_grads("self_attn.o_proj", &o_grads);
        grads.add_layer_grads("mlp.gate_proj", &gate_grads);
        grads.add_layer_grads("mlp.up_proj", &up_grads);
        grads.add_layer_grads("mlp.down_proj", &down_grads);

        Ok((d_x, grads))
    }

    /// Complete training step with custom autograd.
    ///
    /// Returns the loss value.
    pub fn training_step(
        &self,
        model: &mut Qwen3LoraForCausalLM,
        input_ids: &Array,
        labels: &Array,
    ) -> Result<(f32, HashMap<String, Array>), LoraError> {
        // ========== FORWARD PASS ==========

        // Embed tokens (no LoRA here)
        let mut hidden = model.model.embed_tokens.forward(input_ids)?;

        // Create causal mask
        let seq_len = input_ids.dim(1);
        let mask = Some(create_causal_mask(seq_len)?);

        // Forward through all layers with state saving
        let mut layer_saved_states = Vec::with_capacity(model.model.layers.len());

        for layer in &mut model.model.layers {
            let (output, saved) = self.layer_forward(layer, &hidden, mask.as_ref())?;
            layer_saved_states.push(saved);
            hidden = output;
        }

        // Final norm
        let final_norm_weight = model.model.norm.weight.value.as_ref().clone();
        let (hidden_normed, final_norm_saved) = rmsnorm_forward_with_grad(
            &hidden, &final_norm_weight, 1e-6,
        )?;

        // LM head (no LoRA)
        let logits = if let Some(ref mut lm_head) = model.lm_head {
            lm_head.forward(&hidden_normed)?
        } else {
            model.model.embed_tokens.as_linear(&hidden_normed)?
        };

        // ========== LOSS ==========
        let (loss, d_logits) = CustomLoraTrainer::cross_entropy_with_grad(
            &logits, labels, self.ignore_index,
        )?;

        // ========== BACKWARD PASS ==========

        // Backward through LM head
        let d_hidden_normed = if let Some(ref lm_head) = model.lm_head {
            // d_hidden = d_logits @ lm_head.weight
            d_logits.matmul(&lm_head.weight.value.as_ref())?
        } else {
            // Tied weights: d_hidden = d_logits @ embed.weight
            d_logits.matmul(&model.model.embed_tokens.weight.value.as_ref())?
        };

        // Backward through final norm
        let mut d_hidden = rmsnorm_backward(&d_hidden_normed, &final_norm_saved)?;

        // Backward through all layers in reverse
        let mut all_grads = AccumulatedLoraGrads::new();

        for (layer_idx, saved) in layer_saved_states.iter().enumerate().rev() {
            let (d_x, layer_grads) = self.layer_backward(&d_hidden, saved)?;

            // Prefix gradients with layer index
            for (name, grad) in layer_grads.grads {
                all_grads.grads.insert(
                    format!("layers.{}.{}", layer_idx, name),
                    grad,
                );
            }

            d_hidden = d_x;
        }

        // Convert to HashMap for optimizer
        let grads: HashMap<String, Array> = all_grads.grads.into_iter().collect();

        Ok((loss, grads))
    }

    /// Apply gradients to model using SGD.
    pub fn apply_gradients(
        &self,
        model: &mut Qwen3LoraForCausalLM,
        grads: &HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        let lr = Array::from_f32(self.learning_rate);

        for (name, grad) in grads {
            // Parse the gradient name to find the parameter
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() < 4 {
                continue;
            }

            let layer_idx: usize = parts[1].parse().unwrap_or(0);
            if layer_idx >= model.model.layers.len() {
                continue;
            }

            let layer = &mut model.model.layers[layer_idx];
            let update = grad.multiply(&lr)?;

            match (parts[2], parts[3], parts.get(4).copied()) {
                ("self_attn", "q_proj", Some("lora_a")) => {
                    layer.self_attn.q_proj.lora_a = layer.self_attn.q_proj.lora_a.subtract(&update)?;
                }
                ("self_attn", "q_proj", Some("lora_b")) => {
                    layer.self_attn.q_proj.lora_b = layer.self_attn.q_proj.lora_b.subtract(&update)?;
                }
                ("self_attn", "k_proj", Some("lora_a")) => {
                    layer.self_attn.k_proj.lora_a = layer.self_attn.k_proj.lora_a.subtract(&update)?;
                }
                ("self_attn", "k_proj", Some("lora_b")) => {
                    layer.self_attn.k_proj.lora_b = layer.self_attn.k_proj.lora_b.subtract(&update)?;
                }
                ("self_attn", "v_proj", Some("lora_a")) => {
                    layer.self_attn.v_proj.lora_a = layer.self_attn.v_proj.lora_a.subtract(&update)?;
                }
                ("self_attn", "v_proj", Some("lora_b")) => {
                    layer.self_attn.v_proj.lora_b = layer.self_attn.v_proj.lora_b.subtract(&update)?;
                }
                ("self_attn", "o_proj", Some("lora_a")) => {
                    layer.self_attn.o_proj.lora_a = layer.self_attn.o_proj.lora_a.subtract(&update)?;
                }
                ("self_attn", "o_proj", Some("lora_b")) => {
                    layer.self_attn.o_proj.lora_b = layer.self_attn.o_proj.lora_b.subtract(&update)?;
                }
                ("mlp", "gate_proj", Some("lora_a")) => {
                    layer.mlp.gate_proj.lora_a = layer.mlp.gate_proj.lora_a.subtract(&update)?;
                }
                ("mlp", "gate_proj", Some("lora_b")) => {
                    layer.mlp.gate_proj.lora_b = layer.mlp.gate_proj.lora_b.subtract(&update)?;
                }
                ("mlp", "up_proj", Some("lora_a")) => {
                    layer.mlp.up_proj.lora_a = layer.mlp.up_proj.lora_a.subtract(&update)?;
                }
                ("mlp", "up_proj", Some("lora_b")) => {
                    layer.mlp.up_proj.lora_b = layer.mlp.up_proj.lora_b.subtract(&update)?;
                }
                ("mlp", "down_proj", Some("lora_a")) => {
                    layer.mlp.down_proj.lora_a = layer.mlp.down_proj.lora_a.subtract(&update)?;
                }
                ("mlp", "down_proj", Some("lora_b")) => {
                    layer.mlp.down_proj.lora_b = layer.mlp.down_proj.lora_b.subtract(&update)?;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Create a causal attention mask.
fn create_causal_mask(seq_len: i32) -> Result<Array, mlx_rs::error::Exception> {
    let mask = mlx_rs::ops::tri::<f32>(seq_len, None, None)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::LoraConfig;
    use pmetal_models::architectures::qwen3::Qwen3Config;

    fn small_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: 16, // 64 / 4 = 16
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    fn small_lora_config() -> LoraConfig {
        LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            use_rslora: false,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
            ],
            bias: "none".to_string(),
            init_lora_weights: true,
            use_dora: false,
        }
    }

    #[test]
    fn test_custom_training_step() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = Qwen3LoraForCausalLM::new(config.clone(), lora_config).unwrap();

        let trainer = Qwen3CustomTrainer::new(
            config.num_attention_heads,
            config.num_kv_heads(),
            config.get_head_dim(),
            1e-4,
        );

        // Create dummy batch
        let batch_size = 2;
        let seq_len = 8;
        let input_ids = mlx_rs::Array::from_slice(&vec![1_i32; batch_size * seq_len], &[batch_size as i32, seq_len as i32]);
        let labels = mlx_rs::Array::from_slice(&vec![2_i32; batch_size * seq_len], &[batch_size as i32, seq_len as i32]);

        // Run training step
        let (loss, grads) = trainer.training_step(&mut model, &input_ids, &labels).unwrap();

        // Loss should be positive
        assert!(loss > 0.0, "Loss should be positive: {}", loss);

        // Should have gradients
        assert!(!grads.is_empty(), "Should have gradients");

        // Apply gradients
        trainer.apply_gradients(&mut model, &grads).unwrap();
    }
}
