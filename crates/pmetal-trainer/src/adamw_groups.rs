//! AdamW optimizer with parameter group support.
//!
//! This module provides an AdamW optimizer that supports separate learning rates
//! for different parameter groups, commonly used for:
//!
//! - Lower learning rate for embeddings (Unsloth uses 5e-5 vs 2e-4 for LoRA)
//! - Different weight decay for different layers
//! - Layer-wise learning rate decay
//!
//! # Example
//!
//! ```ignore
//! use pmetal_trainer::{AdamWGroups, AdamWGroupsBuilder};
//!
//! // Create optimizer with separate embedding learning rate
//! let optimizer = AdamWGroupsBuilder::new(2e-4)
//!     .with_embedding_lr(5e-5)
//!     .with_weight_decay(0.01)
//!     .build()?;
//! ```

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    array,
    builder::Builder,
    error::{Exception, Result},
    optimizers::{AdamW, AdamWBuilder, Optimizer, State},
    utils::Updatable,
    Array,
};

use crate::ParameterGroupConfig;

/// AdamW optimizer with support for parameter groups.
///
/// This optimizer routes parameters to different AdamW instances based on
/// pattern matching, allowing separate learning rates and weight decay for
/// embeddings vs other parameters.
#[derive(Debug)]
pub struct AdamWGroups {
    /// Optimizer for LoRA and other non-embedding parameters.
    lora_optimizer: AdamW,
    /// Optimizer for embedding parameters (lower learning rate).
    embedding_optimizer: AdamW,
    /// Configuration for identifying embedding parameters.
    config: ParameterGroupConfig,
    /// Cached parameter classifications.
    param_cache: HashMap<Rc<str>, bool>, // true = embedding
}

impl AdamWGroups {
    /// Create a new AdamWGroups optimizer.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Learning rate for non-embedding parameters
    /// * `embedding_lr` - Learning rate for embedding parameters (None = same as base)
    /// * `weight_decay` - Weight decay for all parameters
    pub fn new(base_lr: f32, embedding_lr: Option<f32>, weight_decay: f32) -> Result<Self> {
        let embedding_lr = embedding_lr.unwrap_or(base_lr);

        let lora_optimizer = AdamWBuilder::new(base_lr)
            .weight_decay(weight_decay)
            .build()
            .map_err(|_| Exception::custom("Failed to build LoRA optimizer"))?;

        let embedding_optimizer = AdamWBuilder::new(embedding_lr)
            .weight_decay(weight_decay)
            .build()
            .map_err(|_| Exception::custom("Failed to build embedding optimizer"))?;

        let config = ParameterGroupConfig {
            base_lr: base_lr as f64,
            embedding_lr: if (embedding_lr - base_lr).abs() > 1e-10 {
                Some(embedding_lr as f64)
            } else {
                None
            },
            weight_decay: weight_decay as f64,
            ..Default::default()
        };

        Ok(Self {
            lora_optimizer,
            embedding_optimizer,
            config,
            param_cache: HashMap::new(),
        })
    }

    /// Check if a parameter name matches embedding patterns.
    fn is_embedding_param(&mut self, name: &Rc<str>) -> bool {
        // Check cache first
        if let Some(&is_embedding) = self.param_cache.get(name) {
            return is_embedding;
        }

        // Classify and cache
        let name_lower = name.to_lowercase();
        let is_embedding = self
            .config
            .embedding_patterns
            .iter()
            .any(|pattern| name_lower.contains(&pattern.to_lowercase()));

        self.param_cache.insert(name.clone(), is_embedding);
        is_embedding
    }

    /// Get the learning rates used by this optimizer.
    pub fn learning_rates(&self) -> (f32, f32) {
        (
            self.lora_optimizer.lr.item(),
            self.embedding_optimizer.lr.item(),
        )
    }

    /// Set the base learning rate, maintaining the embedding/base ratio.
    ///
    /// This is used by learning rate schedulers to dynamically adjust the
    /// learning rate during training (e.g., warmup, cosine decay).
    ///
    /// # Arguments
    ///
    /// * `base_lr` - New base learning rate for non-embedding parameters
    ///
    /// The embedding learning rate is scaled proportionally to maintain the
    /// original embedding/base ratio. For example, if embedding_lr was 5e-5
    /// when base_lr was 2e-4 (ratio 0.25), setting base_lr to 1e-4 will set
    /// embedding_lr to 2.5e-5.
    pub fn set_learning_rate(&mut self, base_lr: f32) {
        // Calculate the original ratio (embedding_lr / base_lr)
        let current_base: f32 = self.lora_optimizer.lr.item();
        let current_embedding: f32 = self.embedding_optimizer.lr.item();

        // Maintain the ratio if there was a custom embedding LR, otherwise same as base
        let ratio = if (current_base - current_embedding).abs() < 1e-10 {
            1.0  // Same LR for both
        } else if current_base > 1e-10 {
            current_embedding / current_base
        } else {
            1.0  // Fallback to same LR
        };

        let new_embedding_lr = base_lr * ratio;

        // Update both optimizers
        self.lora_optimizer.lr = array!(base_lr);
        self.embedding_optimizer.lr = array!(new_embedding_lr);
    }

    /// Get summary of parameter grouping.
    pub fn summary(&self) -> String {
        let (base_lr, emb_lr) = self.learning_rates();
        let lora_count = self.lora_optimizer.state.len();
        let emb_count = self.embedding_optimizer.state.len();

        format!(
            "AdamWGroups: {} LoRA params (lr={:.2e}), {} embedding params (lr={:.2e})",
            lora_count, base_lr, emb_count, emb_lr
        )
    }
}

impl Optimizer for AdamWGroups {
    type State = State<(Array, Array)>;

    fn state(&self) -> &Self::State {
        // Return the combined state (we'll need to handle this carefully)
        // For now, just return the LoRA optimizer state
        &self.lora_optimizer.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.lora_optimizer.state
    }

    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        if self.is_embedding_param(key) {
            self.embedding_optimizer.update_single(key, gradient, parameter)
        } else {
            self.lora_optimizer.update_single(key, gradient, parameter)
        }
    }
}

impl Updatable for AdamWGroups {
    fn updatable_states_len(&self) -> usize {
        self.lora_optimizer.updatable_states_len() + self.embedding_optimizer.updatable_states_len()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        self.lora_optimizer
            .updatable_states()
            .into_iter()
            .chain(self.embedding_optimizer.updatable_states())
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        self.lora_optimizer
            .updatable_states_mut()
            .into_iter()
            .chain(self.embedding_optimizer.updatable_states_mut())
    }
}

/// Builder for AdamWGroups optimizer.
#[derive(Debug, Clone)]
pub struct AdamWGroupsBuilder {
    base_lr: f32,
    embedding_lr: Option<f32>,
    weight_decay: f32,
    betas: (f32, f32),
    eps: f32,
    embedding_patterns: Vec<String>,
}

impl AdamWGroupsBuilder {
    /// Create a new builder with the given base learning rate.
    pub fn new(base_lr: f32) -> Self {
        Self {
            base_lr,
            embedding_lr: None,
            weight_decay: 0.01,
            betas: (0.9, 0.999),
            eps: 1e-8,
            embedding_patterns: vec![
                "embed_tokens".to_string(),
                "lm_head".to_string(),
                "wte".to_string(),
                "wpe".to_string(),
                "token_embd".to_string(),
                "output".to_string(),
            ],
        }
    }

    /// Set the embedding learning rate.
    ///
    /// Unsloth recommends 5e-5 for embeddings (vs 2e-4 for LoRA).
    pub fn with_embedding_lr(mut self, lr: f32) -> Self {
        self.embedding_lr = Some(lr);
        self
    }

    /// Set the weight decay.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set the betas for momentum.
    pub fn with_betas(mut self, betas: (f32, f32)) -> Self {
        self.betas = betas;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Add an additional embedding pattern.
    pub fn add_embedding_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.embedding_patterns.push(pattern.into());
        self
    }

    /// Build the optimizer.
    pub fn build(self) -> Result<AdamWGroups> {
        let embedding_lr = self.embedding_lr.unwrap_or(self.base_lr);

        let lora_optimizer = AdamWBuilder::new(self.base_lr)
            .betas(self.betas)
            .eps(self.eps)
            .weight_decay(self.weight_decay)
            .build()
            .map_err(|_| Exception::custom("Failed to build LoRA optimizer"))?;

        let embedding_optimizer = AdamWBuilder::new(embedding_lr)
            .betas(self.betas)
            .eps(self.eps)
            .weight_decay(self.weight_decay)
            .build()
            .map_err(|_| Exception::custom("Failed to build embedding optimizer"))?;

        let config = ParameterGroupConfig {
            base_lr: self.base_lr as f64,
            embedding_lr: if self.embedding_lr.is_some() {
                self.embedding_lr.map(|lr| lr as f64)
            } else {
                None
            },
            weight_decay: self.weight_decay as f64,
            embedding_patterns: self.embedding_patterns,
        };

        Ok(AdamWGroups {
            lora_optimizer,
            embedding_optimizer,
            config,
            param_cache: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    #[test]
    fn test_adamw_groups_creation() {
        let optimizer = AdamWGroupsBuilder::new(2e-4)
            .with_embedding_lr(5e-5)
            .with_weight_decay(0.01)
            .build()
            .unwrap();

        let (base_lr, emb_lr) = optimizer.learning_rates();
        assert!((base_lr - 2e-4).abs() < 1e-8);
        assert!((emb_lr - 5e-5).abs() < 1e-8);
    }

    #[test]
    fn test_embedding_pattern_matching() {
        let mut optimizer = AdamWGroupsBuilder::new(2e-4)
            .with_embedding_lr(5e-5)
            .build()
            .unwrap();

        // Test embedding patterns
        let embed_key: Rc<str> = Rc::from("model.embed_tokens.weight");
        assert!(optimizer.is_embedding_param(&embed_key));

        let lm_head_key: Rc<str> = Rc::from("lm_head.weight");
        assert!(optimizer.is_embedding_param(&lm_head_key));

        // Test LoRA patterns
        let lora_key: Rc<str> = Rc::from("model.layers.0.self_attn.q_proj.lora_A.weight");
        assert!(!optimizer.is_embedding_param(&lora_key));
    }

    #[test]
    fn test_update_single_routing() {
        let mut optimizer = AdamWGroupsBuilder::new(2e-4)
            .with_embedding_lr(5e-5)
            .build()
            .unwrap();

        // Create test parameter and gradient
        let mut param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]);

        // Test LoRA parameter update
        let lora_key: Rc<str> = Rc::from("model.layers.0.q_proj.lora_A.weight");
        optimizer
            .update_single(&lora_key, &grad, &mut param)
            .unwrap();

        // State should be in lora_optimizer
        assert!(optimizer.lora_optimizer.state.contains_key(&lora_key));
        assert!(!optimizer.embedding_optimizer.state.contains_key(&lora_key));

        // Test embedding parameter update
        let mut embed_param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let embed_key: Rc<str> = Rc::from("model.embed_tokens.weight");
        optimizer
            .update_single(&embed_key, &grad, &mut embed_param)
            .unwrap();

        // State should be in embedding_optimizer
        assert!(optimizer.embedding_optimizer.state.contains_key(&embed_key));
        assert!(!optimizer.lora_optimizer.state.contains_key(&embed_key));
    }
}
