//! AdamW optimizer for InlineArray parameters, matching `torch.optim.AdamW`.
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
/// The `String`-keyed counterpart of `compat::FlattenedModuleParam`.
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
pub struct AdamState {
    /// First moment (exponential moving average of gradients).
    pub m: InlineArray,
    /// Second moment (exponential moving average of squared gradients).
    pub v: InlineArray,
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
    pub states: HashMap<String, AdamState>,
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

    /// Override the denominator epsilon (default: 1e-8).
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
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

    /// Set the step counter (used when restoring from checkpoint).
    pub fn set_step_count(&mut self, count: u64) {
        self.step_count_inner = count;
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

    /// Apply one AdamW step: advance the step counter, then
    /// [`Self::step_single`] every parameter in `params` that has a gradient
    /// in `grads`. Parameters without gradients are left untouched.
    ///
    /// No `eval()` calls are made. The caller is responsible for materialising
    /// the updated parameter tensors (and optionally the moment arrays) at the
    /// appropriate point in the training loop.
    pub fn step(&mut self, params: &mut ParamSet, grads: &ParamSet) {
        self.advance_step();
        for (name, param) in params.iter_mut() {
            if let Some(grad) = grads.get(name) {
                self.step_single(name, grad, param);
            }
        }
    }

    /// Advance the global step counter by one.
    ///
    /// Must be called **once per training step** before calling `step_single`
    /// for each parameter.  Callers that use `step_single` in a loop must call
    /// `advance_step` once before the loop.
    pub fn advance_step(&mut self) {
        self.step_count_inner += 1;
    }

    /// Update a single parameter in-place given its gradient.
    ///
    /// **Important**: call `advance_step` once before iterating over
    /// parameters.  This method does *not* increment the step counter —
    /// doing so would corrupt the Adam bias-correction terms when multiple
    /// parameters are updated in one training step.
    ///
    /// The update is `torch.optim.AdamW`'s, in its order:
    /// ```text
    /// p = p·(1 − lr·λ)                      decoupled weight decay, on the old p
    /// m = m + (1−β₁)·(g − m)
    /// v = β₂·v + (1−β₂)·g²
    /// p = p − lr/(1−β₁ᵗ) · m / (√v/√(1−β₂ᵗ) + ε)
    /// ```
    /// The step's scalars are computed in f64 and rounded once, and the
    /// moments keep the parameter's dtype.
    pub fn step_single(&mut self, name: &str, gradient: &InlineArray, parameter: &mut InlineArray) {
        let class = (self.classifier)(name);
        let s = AdamWStepScalars::new(
            self.lr_for(class),
            self.wd_for(class),
            self.beta1,
            self.beta2,
            self.eps,
            self.step_count_inner,
        );

        let state = self.states.entry(name.to_string()).or_insert_with(|| {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype_raw();
            AdamState {
                m: InlineArray::zeros(&shape, dtype),
                v: InlineArray::zeros(&shape, dtype),
            }
        });

        if s.decay != 1.0 {
            *parameter = parameter.mul_scalar(s.decay);
        }
        state.m = state
            .m
            .add(&gradient.subtract(&state.m).mul_scalar(s.one_minus_beta1));
        state.v = state
            .v
            .mul_scalar(s.beta2)
            .add(&gradient.square().mul_scalar(s.one_minus_beta2));
        let denom = state
            .v
            .sqrt()
            .div_scalar(s.bias_correction2_sqrt)
            .add_scalar(s.eps);
        *parameter = parameter.subtract(&state.m.divide(&denom).mul_scalar(s.step_size));
    }
}

/// The scalars of AdamW step `t`, computed as `torch.optim.AdamW` computes
/// them: in f64, each rounded to f32 once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWStepScalars {
    /// `1 − lr·λ`: the decoupled weight decay, applied before the update.
    pub decay: f32,
    /// `1 − β₁`, the weight the first moment moves toward the gradient by.
    pub one_minus_beta1: f32,
    /// `β₂`.
    pub beta2: f32,
    /// `1 − β₂`.
    pub one_minus_beta2: f32,
    /// `lr / (1 − β₁ᵗ)`.
    pub step_size: f32,
    /// `√(1 − β₂ᵗ)`.
    pub bias_correction2_sqrt: f32,
    /// `ε`, added after the second moment's bias correction.
    pub eps: f32,
}

impl AdamWStepScalars {
    /// The scalars of 1-based step `t`.
    ///
    /// # Panics
    /// If `t` is 0: Adam's bias correction is undefined before the first step.
    pub fn new(lr: f32, weight_decay: f32, beta1: f32, beta2: f32, eps: f32, t: u64) -> Self {
        assert!(t >= 1, "AdamW steps are 1-based; advance the step first");
        let (lr, wd, b1, b2) = (
            f64::from(lr),
            f64::from(weight_decay),
            f64::from(beta1),
            f64::from(beta2),
        );
        // `powf` rather than `powi`: a step count past i32::MAX must not wrap.
        let t = t as f64;
        Self {
            decay: (1.0 - lr * wd) as f32,
            one_minus_beta1: (1.0 - b1) as f32,
            beta2,
            one_minus_beta2: (1.0 - b2) as f32,
            step_size: (lr / (1.0 - b1.powf(t))) as f32,
            bias_correction2_sqrt: (1.0 - b2.powf(t)).sqrt() as f32,
            eps,
        }
    }
}

/// Classifier that only separates out the parameters weight decay must skip.
///
/// Bias, norm and scale parameters are [`ParamClass::NoDecay`]; everything
/// else, embeddings and LoRA B matrices included, is [`ParamClass::Regular`]
/// and trains at the base learning rate. This is what an optimizer without
/// parameter groups should do: a separate embedding or LoRA+ rate is a group
/// the caller configures, not something the optimizer infers from a name.
pub fn decay_only_classifier(name: &str) -> ParamClass {
    if name.contains("bias") || name.contains("norm") || name.contains("scale") {
        ParamClass::NoDecay
    } else {
        ParamClass::Regular
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
        assert_eq!(
            default_classifier("layers.0.norm.weight"),
            ParamClass::NoDecay
        );
        assert_eq!(
            default_classifier("layers.0.self_attn.q_proj.bias"),
            ParamClass::NoDecay
        );
        assert_eq!(
            default_classifier("embed_tokens.weight"),
            ParamClass::Embedding
        );
        assert_eq!(default_classifier("lm_head.weight"), ParamClass::Embedding);
        assert_eq!(
            default_classifier("layers.0.q_proj.lora_b"),
            ParamClass::LoraB
        );
        assert_eq!(
            default_classifier("layers.0.q_proj.lora_a"),
            ParamClass::Regular
        );
        assert_eq!(
            default_classifier("layers.0.mlp.gate_proj.weight"),
            ParamClass::Regular
        );
    }

    #[test]
    fn test_adamw_lr_for() {
        let opt = AdamW::new(1e-4, 0.01)
            .with_loraplus_ratio(16.0)
            .with_embedding_lr(5e-5);
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
