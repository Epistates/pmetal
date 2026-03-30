//! AdamW optimizer with parameter group support.
//!
//! This module provides an AdamW optimizer that supports separate learning rates
//! for different parameter groups, commonly used for:
//!
//! - Lower learning rate for embeddings (recommended 5e-5 vs 2e-4 for LoRA)
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

use pmetal_bridge::array;
use pmetal_bridge::compat::{
    Array, Exception,
    optimizers::{AdamW, AdamWBuilder, Optimizer, State, Updatable},
};
type Result<T> = std::result::Result<T, Exception>;

use crate::ParameterGroupConfig;

/// Parameter classification for routing to the correct optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamClass {
    /// Regular parameters (LoRA A weights, projections) — full weight decay.
    Regular,
    /// Embedding parameters — separate learning rate, full weight decay.
    Embedding,
    /// No-decay parameters (bias, norm, scale) — zero weight decay.
    NoDecay,
    /// LoRA B matrices — higher learning rate when LoRA+ is enabled (Hayou et al., ICML 2024).
    /// Falls back to Regular when no loraplus_lr_ratio is configured.
    LoraB,
}

/// AdamW optimizer with support for parameter groups.
///
/// This optimizer routes parameters to different AdamW instances based on
/// pattern matching, allowing separate learning rates and weight decay for
/// embeddings vs other parameters. Bias, norm, and scale parameters are
/// automatically routed to a no-decay optimizer (weight_decay=0).
///
/// When `loraplus_lr_ratio` is configured (LoRA+, Hayou et al. ICML 2024),
/// LoRA B matrices use `base_lr * ratio` while A matrices use `base_lr`.
#[derive(Debug)]
pub struct AdamWGroups {
    /// Optimizer for LoRA A and other non-embedding parameters.
    lora_optimizer: AdamW,
    /// Optimizer for LoRA B matrices when LoRA+ is enabled (`base_lr * ratio`).
    /// When LoRA+ is disabled this is `None` and B matrices use `lora_optimizer`.
    loraplus_b_optimizer: Option<AdamW>,
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
            loraplus_b_optimizer: None,
            embedding_optimizer,
            no_decay_optimizer,
            config,
            param_cache: HashMap::new(),
        })
    }

    /// Classify a parameter by name into Regular, LoraB, Embedding, or NoDecay.
    ///
    /// Priority order:
    /// 1. NoDecay — bias/norm/scale always get zero weight decay.
    /// 2. Embedding — embed_tokens, lm_head, etc. get the embedding learning rate.
    /// 3. LoraB — "lora_b" parameters get the LoRA+ elevated learning rate (if configured).
    /// 4. Regular — everything else (LoRA A, frozen weights, etc.) gets the base learning rate.
    fn classify_param(&mut self, name: &Rc<str>) -> ParamClass {
        // Check cache first
        if let Some(&class) = self.param_cache.get(name) {
            return class;
        }

        let name_lower = name.to_lowercase();

        let class = if self.config.is_no_decay(name) {
            // Bias, norm, and scale parameters — zero weight decay, base LR
            ParamClass::NoDecay
        } else if self
            .config
            .embedding_patterns
            .iter()
            .any(|pattern| name_lower.contains(&pattern.to_lowercase()))
        {
            // Embedding / LM head parameters — embedding LR
            ParamClass::Embedding
        } else if self.loraplus_b_optimizer.is_some()
            && (name_lower.contains("lora_b") || name_lower.contains("lorab"))
        {
            // LoRA B matrices — elevated LR when LoRA+ is configured
            ParamClass::LoraB
        } else {
            // Regular parameters (LoRA A, projections, etc.)
            ParamClass::Regular
        };

        self.param_cache.insert(name.clone(), class);
        class
    }

    /// Get the learning rates used by this optimizer.
    ///
    /// Returns `(base_lr, embedding_lr)`.
    pub fn learning_rates(&mut self) -> (f32, f32) {
        (
            self.lora_optimizer.lr.item_f32(),
            self.embedding_optimizer.lr.item_f32(),
        )
    }

    /// Set the base learning rate, maintaining all group ratios proportionally.
    ///
    /// This is used by learning rate schedulers to dynamically adjust the
    /// learning rate during training (e.g., warmup, cosine decay).
    ///
    /// All group LRs are scaled proportionally to maintain their original ratios
    /// relative to the base LR:
    /// - Embedding: `base_lr * (embedding_lr / original_base_lr)`
    /// - LoRA B (LoRA+): `base_lr * loraplus_ratio`
    /// - No-decay: `base_lr` (no ratio)
    pub fn set_learning_rate(&mut self, base_lr: f32) {
        let current_base: f32 = self.lora_optimizer.lr.item_f32();
        let current_embedding: f32 = self.embedding_optimizer.lr.item_f32();

        // Maintain the embedding ratio relative to the base LR
        let emb_ratio = if (current_base - current_embedding).abs() < 1e-10 {
            1.0
        } else if current_base > 1e-10 {
            current_embedding / current_base
        } else {
            1.0
        };

        let new_embedding_lr = base_lr * emb_ratio;

        self.lora_optimizer.lr = array!(base_lr);
        self.embedding_optimizer.lr = array!(new_embedding_lr);
        self.no_decay_optimizer.lr = array!(base_lr);

        // LoRA B optimizer maintains its ratio relative to the base LR
        if let Some(ref mut b_opt) = self.loraplus_b_optimizer {
            let current_b: f32 = b_opt.lr.item_f32();
            let b_ratio = if current_base > 1e-10 {
                current_b / current_base
            } else {
                1.0
            };
            b_opt.lr = array!(base_lr * b_ratio);
        }
    }

    /// Get summary of parameter grouping.
    pub fn summary(&mut self) -> String {
        let (base_lr, emb_lr) = self.learning_rates();
        let lora_count = self.lora_optimizer.state.len();
        let emb_count = self.embedding_optimizer.state.len();
        let no_decay_count = self.no_decay_optimizer.state.len();

        if let Some(ref mut b_opt) = self.loraplus_b_optimizer {
            let b_lr: f32 = b_opt.lr.item_f32();
            let b_count = b_opt.state.len();
            format!(
                "AdamWGroups(LoRA+): {} A params (lr={:.2e}), {} B params (lr={:.2e}), \
                 {} embed params (lr={:.2e}), {} no-decay params (wd=0)",
                lora_count, base_lr, b_count, b_lr, emb_count, emb_lr, no_decay_count
            )
        } else {
            format!(
                "AdamWGroups: {} regular params (lr={:.2e}), {} embedding params (lr={:.2e}), \
                 {} no-decay params (wd=0)",
                lora_count, base_lr, emb_count, emb_lr, no_decay_count
            )
        }
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
            ParamClass::LoraB => {
                // When LoRA+ is active, route B matrices to the elevated-LR optimizer.
                // If the optimizer is somehow None (shouldn't happen once wired), fall back
                // to the regular optimizer so training never silently stalls.
                if let Some(ref mut b_opt) = self.loraplus_b_optimizer {
                    b_opt.update_single(key, gradient, parameter)
                } else {
                    self.lora_optimizer.update_single(key, gradient, parameter)
                }
            }
            ParamClass::Regular => self.lora_optimizer.update_single(key, gradient, parameter),
        }
    }
}

impl Updatable for AdamWGroups {
    fn updatable_states_len(&self) -> usize {
        let b_len = self
            .loraplus_b_optimizer
            .as_ref()
            .map(|o| o.updatable_states_len())
            .unwrap_or(0);
        self.lora_optimizer.updatable_states_len()
            + b_len
            + self.embedding_optimizer.updatable_states_len()
            + self.no_decay_optimizer.updatable_states_len()
    }

    fn updatable_states(&self) -> Vec<&Array> {
        let b_states: Box<dyn Iterator<Item = &Array>> =
            if let Some(ref b_opt) = self.loraplus_b_optimizer {
                Box::new(b_opt.updatable_states().into_iter())
            } else {
                Box::new(std::iter::empty())
            };

        self.lora_optimizer
            .updatable_states()
            .into_iter()
            .chain(b_states)
            .chain(self.embedding_optimizer.updatable_states())
            .chain(self.no_decay_optimizer.updatable_states())
            .collect()
    }

    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        let b_states: Box<dyn Iterator<Item = &mut Array>> =
            if let Some(ref mut b_opt) = self.loraplus_b_optimizer {
                Box::new(b_opt.updatable_states_mut().into_iter())
            } else {
                Box::new(std::iter::empty())
            };

        self.lora_optimizer
            .updatable_states_mut()
            .into_iter()
            .chain(b_states)
            .chain(self.embedding_optimizer.updatable_states_mut())
            .chain(self.no_decay_optimizer.updatable_states_mut())
            .collect()
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
    /// LoRA+ B matrix LR ratio (Hayou et al., ICML 2024).
    /// When `Some(ratio)`, LoRA B matrices use `base_lr * ratio`.
    loraplus_lr_ratio: Option<f32>,
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
            loraplus_lr_ratio: None,
        }
    }

    /// Set the embedding learning rate.
    ///
    /// Recommended default is 5e-5 for embeddings (vs 2e-4 for LoRA).
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

    /// Enable LoRA+ with the given B/A learning rate ratio.
    ///
    /// When enabled, LoRA B matrices use `base_lr * ratio` and A matrices use `base_lr`.
    /// The paper recommends `ratio = 16.0`.
    pub fn with_loraplus_lr_ratio(mut self, ratio: f32) -> Self {
        self.loraplus_lr_ratio = Some(ratio);
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

        // LoRA+ B matrix optimizer: base_lr * ratio
        let loraplus_b_optimizer = if let Some(ratio) = self.loraplus_lr_ratio {
            let b_lr = self.base_lr * ratio;
            let opt = AdamWBuilder::new(b_lr)
                .betas(self.betas)
                .eps(self.eps)
                .weight_decay(self.weight_decay)
                .build()
                .map_err(|_| Exception::custom("Failed to build LoRA+ B optimizer"))?;
            tracing::info!(
                "LoRA+ enabled: A matrices lr={:.2e}, B matrices lr={:.2e} (ratio={})",
                self.base_lr,
                b_lr,
                ratio
            );
            Some(opt)
        } else {
            None
        };

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
            loraplus_b_optimizer,
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

    #[test]
    fn test_adamw_groups_creation() {
        let mut optimizer = AdamWGroupsBuilder::new(2e-4)
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
