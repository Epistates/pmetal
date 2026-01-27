//! Generic LoRA implementations for transformer architectures.
//!
//! This module provides generic LoRA attention and MLP layers that work with
//! any model configuration implementing `LoraArchitectureConfig`. This eliminates
//! the need for duplicate implementations across architectures.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_lora::{GenericLoraAttention, LoraArchitectureConfig};
//! use pmetal_models::architectures::llama::LlamaConfig;
//!
//! let config = LlamaConfig::default();
//! let lora_config = LoraConfig::default();
//! let attention = GenericLoraAttention::new(&config, &lora_config)?;
//! ```

use mlx_rs::{builder::Builder, error::Exception, nn, Array};
use pmetal_core::LoraConfig;

use crate::{arch_config::LoraArchitectureConfig, LoraError, LoraLinear};

/// Generic LoRA-enabled attention layer.
///
/// This struct provides a unified attention implementation that works with
/// any model architecture implementing `LoraArchitectureConfig`.
#[derive(Debug)]
pub struct GenericLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads (for GQA).
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// Sliding window size (None for full attention).
    pub sliding_window: Option<i32>,

    /// Query projection with LoRA.
    pub q_proj: LoraLinear,
    /// Key projection with LoRA.
    pub k_proj: LoraLinear,
    /// Value projection with LoRA.
    pub v_proj: LoraLinear,
    /// Output projection with LoRA.
    pub o_proj: LoraLinear,
    /// RoPE layer.
    pub rope: nn::Rope,
}

impl GenericLoraAttention {
    /// Create a new generic LoRA attention layer from any architecture config.
    pub fn new<C: LoraArchitectureConfig>(
        config: &C,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads();
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();
        let hidden_size = config.hidden_size();
        let scale = (head_dim as f32).sqrt().recip();

        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        // Create LoRA linear layers for projections
        let q_proj = LoraLinear::new(
            hidden_size,
            n_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let k_proj = LoraLinear::new(
            hidden_size,
            n_kv_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let v_proj = LoraLinear::new(
            hidden_size,
            n_kv_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let o_proj = LoraLinear::new(
            n_heads * head_dim,
            hidden_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;

        // Initialize RoPE
        let rope = nn::RopeBuilder::new(head_dim)
            .base(config.rope_theta())
            .traditional(false)
            .build()
            .unwrap();

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            sliding_window: config.sliding_window(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V using LoRA layers
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [B, L, heads, head_dim]
        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B, heads, L, head_dim]
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let queries = mlx_rs::module::Module::forward(&mut self.rope, &queries)?;
        let keys = mlx_rs::module::Module::forward(&mut self.rope, &keys)?;

        // Expand KV heads for GQA if needed
        let keys = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&keys, repeats)?
        } else {
            keys
        };
        let values = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&values, repeats)?
        } else {
            values
        };

        // Compute attention scores
        let scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = scores.multiply(&Array::from_slice(&[self.scale], &[]))?;

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            scores.add(m)?
        } else {
            scores
        };

        // Softmax and apply to values
        let weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
        let output = weights.matmul(&values)?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, heads * head_dim]
        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        // Output projection
        self.o_proj.forward(&output)
    }

    /// Get LoRA parameters for this attention layer.
    pub fn lora_parameters(&self) -> Vec<(&str, &Array)> {
        vec![
            ("q_proj.lora_a", &self.q_proj.lora_a),
            ("q_proj.lora_b", &self.q_proj.lora_b),
            ("k_proj.lora_a", &self.k_proj.lora_a),
            ("k_proj.lora_b", &self.k_proj.lora_b),
            ("v_proj.lora_a", &self.v_proj.lora_a),
            ("v_proj.lora_b", &self.v_proj.lora_b),
            ("o_proj.lora_a", &self.o_proj.lora_a),
            ("o_proj.lora_b", &self.o_proj.lora_b),
        ]
    }
}

/// Generic LoRA-enabled MLP layer.
///
/// Supports both gated (SwiGLU) and non-gated MLP variants.
#[derive(Debug)]
pub struct GenericLoraMLP {
    /// Gate projection with LoRA.
    pub gate_proj: LoraLinear,
    /// Up projection with LoRA.
    pub up_proj: LoraLinear,
    /// Down projection with LoRA.
    pub down_proj: LoraLinear,
    /// Whether to use gated activation.
    pub use_gated: bool,
}

impl GenericLoraMLP {
    /// Create a new generic LoRA MLP layer from any architecture config.
    pub fn new<C: LoraArchitectureConfig>(
        config: &C,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let hidden_size = config.hidden_size();
        let intermediate_size = config.intermediate_size();

        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let gate_proj = LoraLinear::new(
            hidden_size,
            intermediate_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let up_proj = LoraLinear::new(
            hidden_size,
            intermediate_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let down_proj = LoraLinear::new(
            intermediate_size,
            hidden_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            use_gated: config.use_gated_mlp(),
        })
    }

    /// Forward pass through MLP.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        if self.use_gated {
            // SwiGLU: down(silu(gate(x)) * up(x))
            let gate = self.gate_proj.forward(x)?;
            let up = self.up_proj.forward(x)?;
            let activated = mlx_rs::nn::silu(&gate)?;
            let hidden = activated.multiply(&up)?;
            self.down_proj.forward(&hidden)
        } else {
            // Standard: down(gelu(up(x)))
            let hidden = self.up_proj.forward(x)?;
            let activated = mlx_rs::nn::gelu(&hidden)?;
            self.down_proj.forward(&activated)
        }
    }

    /// Get LoRA parameters for this MLP layer.
    pub fn lora_parameters(&self) -> Vec<(&str, &Array)> {
        vec![
            ("gate_proj.lora_a", &self.gate_proj.lora_a),
            ("gate_proj.lora_b", &self.gate_proj.lora_b),
            ("up_proj.lora_a", &self.up_proj.lora_a),
            ("up_proj.lora_b", &self.up_proj.lora_b),
            ("down_proj.lora_a", &self.down_proj.lora_a),
            ("down_proj.lora_b", &self.down_proj.lora_b),
        ]
    }
}

/// Expand KV heads for grouped query attention (GQA).
fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // [B, kv_heads, L, head_dim] -> [B, kv_heads, 1, L, head_dim]
    let expanded = x.reshape(&[batch, kv_heads, 1, seq_len, head_dim])?;
    // Broadcast to [B, kv_heads, repeats, L, head_dim]
    let tiled =
        mlx_rs::ops::broadcast_to(&expanded, &[batch, kv_heads, repeats, seq_len, head_dim])?;
    // Reshape to [B, kv_heads * repeats, L, head_dim]
    tiled.reshape(&[batch, kv_heads * repeats, seq_len, head_dim])
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::llama::LlamaConfig;

    #[test]
    fn test_generic_attention_creation() {
        let config = LlamaConfig::default();
        let lora_config = LoraConfig::default();
        let attention = GenericLoraAttention::new(&config, &lora_config);
        assert!(attention.is_ok());
    }

    #[test]
    fn test_generic_mlp_creation() {
        let config = LlamaConfig::default();
        let lora_config = LoraConfig::default();
        let mlp = GenericLoraMLP::new(&config, &lora_config);
        assert!(mlp.is_ok());
    }
}
