//! AdamW optimizer for InlineArray parameters — zero mlx-rs dependency.
//!
//! Operates purely on [`InlineArray`] values. All arithmetic builds MLX
//! lazy computation graph nodes; no `eval()` is called here. The caller
//! decides when to materialise the updated parameters (typically once per
//! gradient accumulation cycle, or after the full optimizer step).
//!
//! # Parameter groups
//!
//! [`AdamW`] supports differential learning rates via [`ParamClass`]:
//!
//! | Class       | Learning rate          | Weight decay |
//! |-------------|------------------------|--------------|
//! | `Regular`   | `base_lr`              | `weight_decay` |
//! | `NoDecay`   | `base_lr`              | 0.0          |
//! | `Embedding` | `embedding_lr`         | `weight_decay` |
//! | `LoraB`     | `base_lr * loraplus_ratio` | `weight_decay` |
//!
//! LoRA+ (Hayou et al., 2024) sets `loraplus_ratio` to scale up LoRA B
//! matrices relative to LoRA A matrices, improving fine-tuning stability.

use crate::InlineArray;
use std::collections::HashMap;

/// Named parameter set — maps parameter names to their [`InlineArray`] values.
///
/// This is the bridge-native replacement for mlx-rs's
/// `FlattenedModuleParam` (`HashMap<Rc<str>, Array>`).
pub type ParamSet = HashMap<String, InlineArray>;

/// Parameter classification for differential learning rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamClass {
    /// Bias, norm, or scale parameter — weight decay suppressed.
    NoDecay,
    /// Embedding or output projection weights — separate (usually lower) LR.
    Embedding,
    /// LoRA B matrices — scaled up by `loraplus_ratio` (LoRA+ paper).
    LoraB,
    /// Regular trainable parameters: LoRA A, dense projections, etc.
    Regular,
}

/// Per-parameter first and second Adam moment arrays.
struct AdamState {
    /// First moment (exponential moving average of gradients).
    m: InlineArray,
    /// Second moment (exponential moving average of squared gradients).
    v: InlineArray,
}

/// AdamW optimizer with support for parameter groups and LoRA+.
///
/// # Example
/// ```ignore
/// let mut opt = AdamW::new(1e-4, 0.01)
///     .with_loraplus_ratio(16.0)
///     .with_embedding_lr(5e-5);
///
/// // training loop
/// opt.step(&mut params, &grads);
/// mx::eval_many(&mut params.values_mut().collect::<Vec<_>>());
/// ```
pub struct AdamW {
    base_lr: f32,
    embedding_lr: f32,
    loraplus_ratio: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step_count_inner: u64,
    states: HashMap<String, AdamState>,
    /// Maps a parameter name to its [`ParamClass`].
    classifier: Box<dyn Fn(&str) -> ParamClass + Send>,
}

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW")
            .field("base_lr", &self.base_lr)
            .field("embedding_lr", &self.embedding_lr)
            .field("loraplus_ratio", &self.loraplus_ratio)
            .field("weight_decay", &self.weight_decay)
            .field("beta1", &self.beta1)
            .field("beta2", &self.beta2)
            .field("eps", &self.eps)
            .field("step", &self.step_count_inner)
            .field("num_states", &self.states.len())
            .finish_non_exhaustive()
    }
}

impl AdamW {
    /// Create a new AdamW optimizer with the default betas (0.9, 0.999) and
    /// eps 1e-8.
    ///
    /// `base_lr` is the learning rate for `Regular` and `NoDecay` params.
    /// `weight_decay` is applied to all classes except `NoDecay`.
    pub fn new(base_lr: f32, weight_decay: f32) -> Self {
        Self {
            base_lr,
            embedding_lr: 5e-5,
            loraplus_ratio: 1.0,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step_count_inner: 0,
            states: HashMap::new(),
            classifier: Box::new(default_classifier),
        }
    }

    /// Override Adam beta coefficients (default: 0.9, 0.999).
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Override the learning rate applied to `Embedding`-class parameters
    /// (default: 5e-5).
    pub fn with_embedding_lr(mut self, lr: f32) -> Self {
        self.embedding_lr = lr;
        self
    }

    /// Set the LoRA+ scaling factor for `LoraB`-class parameters.
    ///
    /// The effective LR for LoRA B matrices is `base_lr * loraplus_ratio`.
    /// Hayou et al. recommend values in the range 4–16 for most setups.
    /// Default is 1.0 (no scaling).
    pub fn with_loraplus_ratio(mut self, ratio: f32) -> Self {
        self.loraplus_ratio = ratio;
        self
    }

    /// Replace the default parameter classifier with a custom function.
    ///
    /// The classifier receives the full parameter name (e.g.
    /// `"layers.0.self_attn.q_proj.lora_b"`) and returns a [`ParamClass`].
    pub fn with_classifier(mut self, f: impl Fn(&str) -> ParamClass + Send + 'static) -> Self {
        self.classifier = Box::new(f);
        self
    }

    /// Update `base_lr` for LR scheduling. Embedding and LoraB rates are
    /// derived from `base_lr`, so they update automatically.
    pub fn set_lr(&mut self, lr: f32) {
        self.base_lr = lr;
    }

    /// Return the current base learning rate.
    pub fn lr(&self) -> f32 {
        self.base_lr
    }

    /// Return the number of optimizer steps that have been taken.
    pub fn step_count(&self) -> u64 {
        self.step_count_inner
    }

    /// Remove optimizer state for parameters that are no longer present in
    /// the model (e.g. after a LoRA rank change). Only needed in advanced
    /// setups; safe to call at any time.
    pub fn discard_state_for<'a>(&mut self, names: impl Iterator<Item = &'a str>) {
        for n in names {
            self.states.remove(n);
        }
    }

    /// Return the effective learning rate for a given [`ParamClass`].
    #[inline]
    fn lr_for(&self, class: ParamClass) -> f32 {
        match class {
            ParamClass::NoDecay => self.base_lr,
            ParamClass::Embedding => self.embedding_lr,
            ParamClass::LoraB => self.base_lr * self.loraplus_ratio,
            ParamClass::Regular => self.base_lr,
        }
    }

    /// Return the effective weight decay for a given [`ParamClass`].
    #[inline]
    fn wd_for(&self, class: ParamClass) -> f32 {
        match class {
            ParamClass::NoDecay => 0.0,
            _ => self.weight_decay,
        }
    }

    /// Apply one AdamW step.
    ///
    /// Updates every parameter in `params` that also has a corresponding
    /// entry in `grads`. Parameters without gradients are left untouched.
    ///
    /// The update is:
    /// ```text
    /// m  = β₁·m + (1−β₁)·g
    /// v  = β₂·v + (1−β₂)·g²
    /// m̂  = m / (1 − β₁ᵗ)
    /// v̂  = v / (1 − β₂ᵗ)
    /// p  = p − lr · m̂/(√v̂ + ε)  −  wd·lr·p
    /// ```
    ///
    /// No `eval()` calls are made. The caller is responsible for materialising
    /// the updated parameter tensors (and optionally the moment arrays) at the
    /// appropriate point in the training loop.
    pub fn step(&mut self, params: &mut ParamSet, grads: &ParamSet) {
        self.step_count_inner += 1;
        let t = self.step_count_inner as f32;

        // Bias-correction denominators (scalar InlineArrays).
        let bc1 = InlineArray::from_f32(1.0 - self.beta1.powf(t));
        let bc2 = InlineArray::from_f32(1.0 - self.beta2.powf(t));

        // Pre-box scalar constants shared across all parameters.
        let b1 = InlineArray::from_f32(self.beta1);
        let one_minus_b1 = InlineArray::from_f32(1.0 - self.beta1);
        let b2 = InlineArray::from_f32(self.beta2);
        let one_minus_b2 = InlineArray::from_f32(1.0 - self.beta2);
        let eps_arr = InlineArray::from_f32(self.eps);

        for (name, param) in params.iter_mut() {
            let grad = match grads.get(name) {
                Some(g) => g,
                None => continue,
            };

            let class = (self.classifier)(name);
            let lr = self.lr_for(class);
            let wd = self.wd_for(class);

            // Lazily initialise moment arrays to zeros on first encounter.
            let state = self.states.entry(name.clone()).or_insert_with(|| {
                let shape = param.shape().to_vec();
                let dtype = param.dtype_raw();
                AdamState {
                    m: InlineArray::zeros(&shape, dtype),
                    v: InlineArray::zeros(&shape, dtype),
                }
            });

            // m = β₁·m + (1−β₁)·g
            state.m = state.m.multiply(&b1).add(&grad.multiply(&one_minus_b1));

            // v = β₂·v + (1−β₂)·g²
            state.v = state.v.multiply(&b2).add(&grad.square().multiply(&one_minus_b2));

            // Bias-corrected estimates: m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)
            let m_hat = state.m.divide(&bc1);
            let v_hat = state.v.divide(&bc2);

            // Adaptive step: Δ = m̂ / (√v̂ + ε)
            let adaptive_step = m_hat.divide(&v_hat.sqrt().add(&eps_arr));

            // Parameter update: p = p − lr·Δ
            let lr_arr = InlineArray::from_f32(lr);
            *param = param.subtract(&adaptive_step.multiply(&lr_arr));

            // Decoupled weight decay: p = p − wd·lr·p
            // Applied separately from the adaptive step, as in Loshchilov &
            // Hutter (2019). Skipped entirely when wd == 0 (e.g. NoDecay).
            if wd > 0.0 {
                let wd_lr_arr = InlineArray::from_f32(wd * lr);
                *param = param.subtract(&param.clone().multiply(&wd_lr_arr));
            }
        }
    }
}

/// Default parameter classifier based on common naming conventions.
///
/// Rules (first match wins):
/// 1. Name contains `"bias"`, `"norm"`, or `"scale"` → [`ParamClass::NoDecay`]
/// 2. Name contains `"embed"` or `"lm_head"` → [`ParamClass::Embedding`]
/// 3. Name contains `"lora_b"` → [`ParamClass::LoraB`]
/// 4. All other names → [`ParamClass::Regular`]
pub fn default_classifier(name: &str) -> ParamClass {
    if name.contains("bias") || name.contains("norm") || name.contains("scale") {
        ParamClass::NoDecay
    } else if name.contains("embed") || name.contains("lm_head") {
        ParamClass::Embedding
    } else if name.contains("lora_b") {
        ParamClass::LoraB
    } else {
        ParamClass::Regular
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_classifier() {
        assert_eq!(default_classifier("layers.0.norm.weight"), ParamClass::NoDecay);
        assert_eq!(default_classifier("layers.0.self_attn.q_proj.bias"), ParamClass::NoDecay);
        assert_eq!(default_classifier("embed_tokens.weight"), ParamClass::Embedding);
        assert_eq!(default_classifier("lm_head.weight"), ParamClass::Embedding);
        assert_eq!(default_classifier("layers.0.q_proj.lora_b"), ParamClass::LoraB);
        assert_eq!(default_classifier("layers.0.q_proj.lora_a"), ParamClass::Regular);
        assert_eq!(default_classifier("layers.0.mlp.gate_proj.weight"), ParamClass::Regular);
    }

    #[test]
    fn test_adamw_lr_for() {
        let opt = AdamW::new(1e-4, 0.01).with_loraplus_ratio(16.0).with_embedding_lr(5e-5);
        assert!((opt.lr_for(ParamClass::Regular) - 1e-4).abs() < 1e-9);
        assert!((opt.lr_for(ParamClass::NoDecay) - 1e-4).abs() < 1e-9);
        assert!((opt.lr_for(ParamClass::Embedding) - 5e-5).abs() < 1e-9);
        assert!((opt.lr_for(ParamClass::LoraB) - 16.0 * 1e-4).abs() < 1e-9);
    }

    #[test]
    fn test_adamw_wd_for() {
        let opt = AdamW::new(1e-4, 0.01);
        assert_eq!(opt.wd_for(ParamClass::NoDecay), 0.0);
        assert!((opt.wd_for(ParamClass::Regular) - 0.01).abs() < 1e-9);
        assert!((opt.wd_for(ParamClass::Embedding) - 0.01).abs() < 1e-9);
        assert!((opt.wd_for(ParamClass::LoraB) - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_step_count_increments() {
        let mut opt = AdamW::new(1e-4, 0.0);
        assert_eq!(opt.step_count(), 0);

        let mut params = ParamSet::new();
        let grads = ParamSet::new();

        opt.step(&mut params, &grads);
        assert_eq!(opt.step_count(), 1);

        opt.step(&mut params, &grads);
        assert_eq!(opt.step_count(), 2);
    }

    #[test]
    fn test_set_lr() {
        let mut opt = AdamW::new(1e-4, 0.0);
        opt.set_lr(1e-3);
        assert!((opt.lr() - 1e-3).abs() < 1e-9);
        assert!((opt.lr_for(ParamClass::Regular) - 1e-3).abs() < 1e-9);
    }
}
