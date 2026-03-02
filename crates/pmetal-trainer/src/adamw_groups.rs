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
    Array, array,
    builder::Builder,
    error::{Exception, Result},
    optimizers::{AdamW, AdamWBuilder, Optimizer, State},
    utils::Updatable,
};

use crate::ParameterGroupConfig;

/// Parameter classification for routing to the correct optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamClass {
    /// Regular parameters (LoRA weights, projections) — full weight decay.
    Regular,
    /// Embedding parameters — separate learning rate, full weight decay.
    Embedding,
    /// No-decay parameters (bias, norm, scale) — zero weight decay.
    NoDecay,
}

/// AdamW optimizer with support for parameter groups.
///
/// This optimizer routes parameters to different AdamW instances based on
/// pattern matching, allowing separate learning rates and weight decay for
/// embeddings vs other parameters. Bias, norm, and scale parameters are
/// automatically routed to a no-decay optimizer (weight_decay=0).
#[derive(Debug)]
pub struct AdamWGroups {
    /// Optimizer for LoRA and other non-embedding parameters.
    lora_optimizer: AdamW,
    /// Optimizer for embedding parameters (lower learning rate).
    embedding_optimizer: AdamW,
    /// Optimizer for bias/norm/scale parameters (zero weight decay).
    no_decay_optimizer: AdamW,
    /// Configuration for identifying embedding parameters.
    config: ParameterGroupConfig,
    /// Cached parameter classifications.
    param_cache: HashMap<Rc<str>, ParamClass>,
}

impl AdamWGroups {
    /// Create a new AdamWGroups optimizer.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Learning rate for non-embedding parameters
    /// * `embedding_lr` - Learning rate for embedding parameters (None = same as base)
    /// * `weight_decay` - Weight decay for regular parameters (bias/norm always get 0)
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

        let no_decay_optimizer = AdamWBuilder::new(base_lr)
            .weight_decay(0.0)
            .build()
            .map_err(|_| Exception::custom("Failed to build no-decay optimizer"))?;

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
            no_decay_optimizer,
            config,
            param_cache: HashMap::new(),
        })
    }

    /// Classify a parameter by name into Regular, Embedding, or NoDecay.
    ///
    /// No-decay check takes priority: bias/norm/scale parameters always get
    /// zero weight decay regardless of whether they also match embedding patterns.
    fn classify_param(&mut self, name: &Rc<str>) -> ParamClass {
        // Check cache first
        if let Some(&class) = self.param_cache.get(name) {
            return class;
        }

        // Classify: no-decay check first, then embedding check
        let class = if self.config.is_no_decay(name) {
            ParamClass::NoDecay
        } else {
            let name_lower = name.to_lowercase();
            let is_embedding = self
                .config
                .embedding_patterns
                .iter()
                .any(|pattern| name_lower.contains(&pattern.to_lowercase()));
            if is_embedding {
                ParamClass::Embedding
            } else {
                ParamClass::Regular
            }
        };

        self.param_cache.insert(name.clone(), class);
        class
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
            1.0 // Same LR for both
        } else if current_base > 1e-10 {
            current_embedding / current_base
        } else {
            1.0 // Fallback to same LR
        };

        let new_embedding_lr = base_lr * ratio;

        // Update all three optimizers
        self.lora_optimizer.lr = array!(base_lr);
        self.embedding_optimizer.lr = array!(new_embedding_lr);
        self.no_decay_optimizer.lr = array!(base_lr);
    }

    /// Get summary of parameter grouping.
    pub fn summary(&self) -> String {
        let (base_lr, emb_lr) = self.learning_rates();
        let lora_count = self.lora_optimizer.state.len();
        let emb_count = self.embedding_optimizer.state.len();
        let no_decay_count = self.no_decay_optimizer.state.len();

        format!(
            "AdamWGroups: {} regular params (lr={:.2e}), {} embedding params (lr={:.2e}), {} no-decay params (wd=0)",
            lora_count, base_lr, emb_count, emb_lr, no_decay_count
        )
    }
}

impl Optimizer for AdamWGroups {
    type State = State<(Array, Array)>;

    /// Returns the LoRA optimizer state.
    ///
    /// # Note on checkpoint completeness
    ///
    /// The `Optimizer` trait requires a single `&Self::State` reference, so
    /// this method can only expose one of the two internal optimizers.  The
    /// embedding optimizer momentum/velocity is intentionally included through
    /// the `Updatable` implementation instead: `updatable_states()` chains
    /// both `lora_optimizer` and `embedding_optimizer` states, ensuring that
    /// mlx-rs checkpoint serialization captures all optimizer state.
    ///
    /// If you are checkpointing manually via `Optimizer::state()`, also call
    /// `self.embedding_optimizer.state()` to persist embedding optimizer state.
    fn state(&self) -> &Self::State {
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
        match self.classify_param(key) {
            ParamClass::NoDecay => self
                .no_decay_optimizer
                .update_single(key, gradient, parameter),
            ParamClass::Embedding => self
                .embedding_optimizer
                .update_single(key, gradient, parameter),
            ParamClass::Regular => self.lora_optimizer.update_single(key, gradient, parameter),
        }
    }
}

impl Updatable for AdamWGroups {
    fn updatable_states_len(&self) -> usize {
        self.lora_optimizer.updatable_states_len()
            + self.embedding_optimizer.updatable_states_len()
            + self.no_decay_optimizer.updatable_states_len()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        self.lora_optimizer
            .updatable_states()
            .into_iter()
            .chain(self.embedding_optimizer.updatable_states())
            .chain(self.no_decay_optimizer.updatable_states())
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        self.lora_optimizer
            .updatable_states_mut()
            .into_iter()
            .chain(self.embedding_optimizer.updatable_states_mut())
            .chain(self.no_decay_optimizer.updatable_states_mut())
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

        let no_decay_optimizer = AdamWBuilder::new(self.base_lr)
            .betas(self.betas)
            .eps(self.eps)
            .weight_decay(0.0)
            .build()
            .map_err(|_| Exception::custom("Failed to build no-decay optimizer"))?;

        let config = ParameterGroupConfig {
            base_lr: self.base_lr as f64,
            embedding_lr: if self.embedding_lr.is_some() {
                self.embedding_lr.map(|lr| lr as f64)
            } else {
                None
            },
            weight_decay: self.weight_decay as f64,
            embedding_patterns: self.embedding_patterns,
            ..Default::default()
        };

        Ok(AdamWGroups {
            lora_optimizer,
            embedding_optimizer,
            no_decay_optimizer,
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
    fn test_parameter_classification() {
        let mut optimizer = AdamWGroupsBuilder::new(2e-4)
            .with_embedding_lr(5e-5)
            .build()
            .unwrap();

        // Test embedding patterns
        let embed_key: Rc<str> = Rc::from("model.embed_tokens.weight");
        assert_eq!(optimizer.classify_param(&embed_key), ParamClass::Embedding);

        let lm_head_key: Rc<str> = Rc::from("lm_head.weight");
        assert_eq!(
            optimizer.classify_param(&lm_head_key),
            ParamClass::Embedding
        );

        // Test LoRA patterns → Regular
        let lora_key: Rc<str> = Rc::from("model.layers.0.self_attn.q_proj.lora_A.weight");
        assert_eq!(optimizer.classify_param(&lora_key), ParamClass::Regular);

        // Test no-decay patterns (bias, norm)
        let bias_key: Rc<str> = Rc::from("model.layers.0.self_attn.q_proj.bias");
        assert_eq!(optimizer.classify_param(&bias_key), ParamClass::NoDecay);

        let norm_key: Rc<str> = Rc::from("model.layers.0.input_layernorm.weight");
        assert_eq!(optimizer.classify_param(&norm_key), ParamClass::NoDecay);
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

        // Test LoRA parameter update → lora_optimizer
        let lora_key: Rc<str> = Rc::from("model.layers.0.q_proj.lora_A.weight");
        optimizer
            .update_single(&lora_key, &grad, &mut param)
            .unwrap();
        assert!(optimizer.lora_optimizer.state.contains_key(&lora_key));
        assert!(!optimizer.embedding_optimizer.state.contains_key(&lora_key));
        assert!(!optimizer.no_decay_optimizer.state.contains_key(&lora_key));

        // Test embedding parameter update → embedding_optimizer
        let mut embed_param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let embed_key: Rc<str> = Rc::from("model.embed_tokens.weight");
        optimizer
            .update_single(&embed_key, &grad, &mut embed_param)
            .unwrap();
        assert!(optimizer.embedding_optimizer.state.contains_key(&embed_key));
        assert!(!optimizer.lora_optimizer.state.contains_key(&embed_key));

        // Test bias parameter update → no_decay_optimizer
        let mut bias_param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let bias_key: Rc<str> = Rc::from("model.layers.0.self_attn.q_proj.bias");
        optimizer
            .update_single(&bias_key, &grad, &mut bias_param)
            .unwrap();
        assert!(optimizer.no_decay_optimizer.state.contains_key(&bias_key));
        assert!(!optimizer.lora_optimizer.state.contains_key(&bias_key));
        assert!(!optimizer.embedding_optimizer.state.contains_key(&bias_key));
    }
}
