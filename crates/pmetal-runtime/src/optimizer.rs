//! AdamW optimizer for InlineArray parameters, matching `torch.optim.AdamW`.
//!
//! Operates purely on [`InlineArray`] values. All arithmetic builds MLX
//! lazy computation graph nodes; no `eval()` is called here. The caller
//! decides when to materialise the updated parameters (typically once per
//! gradient accumulation cycle, or after the full optimizer step).
//!
//! # Parameter groups
//!
//! [`AdamW`] trains each [`ParamClass`] at the rate [`GroupRates`] gives it,
//! a multiple of the base learning rate the scheduler sets:
//!
//! | Class       | Learning rate               | Weight decay   |
//! |-------------|-----------------------------|----------------|
//! | `Regular`   | `base_lr`                   | `weight_decay` |
//! | `NoDecay`   | `base_lr`                   | 0.0            |
//! | `Embedding` | `base_lr * embedding_ratio` | `weight_decay` |
//! | `LoraB`     | `base_lr * loraplus_ratio`  | `weight_decay` |
//!
//! LoRA+ (Hayou, Ghosh and Yu, ICML 2024) trains each LoRA B matrix at
//! `loraplus_ratio` times its A matrix's rate. Which parameters skip weight
//! decay is [`excluded_from_weight_decay`].

use crate::InlineArray;
use crate::compat::Dtype;
use std::collections::HashMap;

/// Named parameter set — maps parameter names to their [`InlineArray`] values.
///
/// The `String`-keyed counterpart of `compat::FlattenedModuleParam`.
pub type ParamSet = HashMap<String, InlineArray>;

/// Whether the parameter called `name` trains without weight decay: biases
/// and the weights of normalization layers.
///
/// This is the split `transformers`' `Trainer.get_decay_parameter_names`
/// makes. On the lower-cased dotted name it leaves out anything containing
/// `bias`, `layernorm` or `rmsnorm` and anything under a module named `norm`
/// or `*_norm`; it also leaves out every parameter of an `nn.LayerNorm`,
/// which it finds by type. With only the name to go on, a path component
/// that contains `norm` (`norm1`, `hnorm`, `norm_q_a`) or is GPT-2's `ln_*`
/// stands in for that type check, so every normalization weight is left
/// out whatever its module is called.
pub fn excluded_from_weight_decay(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("bias")
        || name
            .split('.')
            .any(|part| part.contains("norm") || part == "ln" || part.starts_with("ln_"))
}

/// What `transformers` names a model's token embedding and its output head,
/// the modules its `get_input_embeddings` and `get_output_embeddings` return,
/// and what GGUF calls the token embedding. The first two are PEFT's
/// `EMBEDDING_LAYER_NAMES`.
pub const EMBEDDING_MODULE_NAMES: &[&str] = &[
    // Token embeddings.
    "embed_tokens",    // Llama, Mistral, Qwen, Gemma, Phi-3, OPT
    "wte",             // GPT-2, GPT-J
    "word_embeddings", // BERT, BLOOM, Falcon
    "embed_in",        // GPT-NeoX
    "tok_embeddings",  // InternLM2
    "token_embd",      // GGUF
    // Output heads.
    "lm_head",
    "embed_out",    // GPT-NeoX
    "output_layer", // ChatGLM
];

/// Whether the parameter called `name` sits under a module named one of
/// `modules`: one of its dotted path components is, ignoring case.
///
/// Whole components, not substrings, so the `output` of BERT's
/// `attention.output.dense` is not an output head and `embed` does not reach
/// a vision tower's `patch_embed`.
pub fn under_module<S: AsRef<str>>(name: &str, modules: &[S]) -> bool {
    name.split('.').any(|part| {
        modules
            .iter()
            .any(|module| part.eq_ignore_ascii_case(module.as_ref()))
    })
}

/// Whether the parameter called `name` belongs to the token embedding or the
/// output head, by the names in [`EMBEDDING_MODULE_NAMES`].
pub fn is_embedding_parameter(name: &str) -> bool {
    under_module(name, EMBEDDING_MODULE_NAMES)
}

/// The learning rate and weight decay of each [`ParamClass`], the rates as
/// multiples of the base learning rate.
///
/// The scheduler moves the base rate and every class keeps its ratio to it,
/// as a PyTorch scheduler scales each parameter group's initial rate by the
/// same factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupRates {
    /// Weight decay of every class but [`ParamClass::NoDecay`].
    pub weight_decay: f32,
    /// The embedding learning rate over the base rate.
    pub embedding_ratio: f32,
    /// The LoRA B learning rate over the base rate (LoRA+); `None` trains
    /// LoRA B matrices at the base rate.
    pub loraplus_ratio: Option<f32>,
}

impl GroupRates {
    /// Every class at the base rate, decayed by `weight_decay` unless it is
    /// [`ParamClass::NoDecay`].
    pub fn new(weight_decay: f32) -> Self {
        Self {
            weight_decay,
            embedding_ratio: 1.0,
            loraplus_ratio: None,
        }
    }

    /// Train embeddings at `embedding_lr` while the base rate is `base_lr`,
    /// and at the same multiple of it as it moves. With a base rate of 0
    /// there is no multiple, and embeddings train at the base rate.
    pub fn with_embedding_lr(mut self, embedding_lr: f32, base_lr: f32) -> Self {
        self.embedding_ratio = if base_lr > 0.0 {
            embedding_lr / base_lr
        } else {
            1.0
        };
        self
    }

    /// Train LoRA B matrices at `ratio` times the base rate (LoRA+).
    pub fn with_loraplus_ratio(mut self, ratio: f32) -> Self {
        self.loraplus_ratio = Some(ratio);
        self
    }

    /// The learning rate of `class` while the base rate is `base_lr`.
    #[inline]
    pub fn learning_rate(&self, class: ParamClass, base_lr: f32) -> f32 {
        match class {
            ParamClass::Regular | ParamClass::NoDecay => base_lr,
            ParamClass::Embedding => base_lr * self.embedding_ratio,
            ParamClass::LoraB => base_lr * self.loraplus_ratio.unwrap_or(1.0),
        }
    }

    /// The weight decay of `class`.
    #[inline]
    pub fn weight_decay(&self, class: ParamClass) -> f32 {
        match class {
            ParamClass::NoDecay => 0.0,
            _ => self.weight_decay,
        }
    }
}

/// Parameter classification for differential learning rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamClass {
    /// A bias or a normalization weight: no weight decay
    /// ([`excluded_from_weight_decay`]).
    NoDecay,
    /// Embedding or output projection weights — separate (usually lower) LR.
    Embedding,
    /// LoRA B matrices — scaled up by `loraplus_ratio` (LoRA+ paper).
    LoraB,
    /// Regular trainable parameters: LoRA A, dense projections, etc.
    Regular,
}

/// An f32 master copy of a half-precision parameter, which every optimizer
/// updates through.
///
/// A bf16 weight near 1 moves in steps of 2⁻⁸, so a learning-rate-sized update
/// rounds away every time it is added to the weight itself; the master copy
/// accumulates it and the weight is what the master rounds to. A parameter
/// that is already f32 has no master copy and is updated in place.
///
/// An optimizer calls [`weight`](Self::weight) for the f32 weight to update,
/// and [`commit`](Self::commit) with the updated one.
#[derive(Default, Debug)]
pub struct MasterWeight {
    value: Option<InlineArray>,
    /// The parameter array the last commit produced, to tell when something
    /// else has replaced it (a checkpoint load, an adapter merge) and the
    /// master copy is stale. Held, not just identified, so that no new array
    /// can take its identity while it is the one compared against.
    produced: Option<InlineArray>,
    /// Set by [`restore`](Self::restore): the master copy came from a
    /// checkpoint, and the parameter it is read against was loaded from the
    /// same checkpoint, so it is the one to continue from although no commit
    /// of this process produced it.
    restored: bool,
}

impl MasterWeight {
    /// Whether `dtype` is one a master copy is kept for.
    fn is_half(dtype: i32) -> bool {
        matches!(Dtype::from_raw(dtype), Dtype::Float16 | Dtype::Bfloat16)
    }

    /// The f32 weight an update of `parameter` starts from: the master copy,
    /// unless the parameter was replaced since the commit that produced it.
    pub fn weight(&self, parameter: &InlineArray) -> InlineArray {
        let current = self.restored
            || self
                .produced
                .as_ref()
                .is_some_and(|produced| produced.id() == parameter.id());
        match &self.value {
            Some(master) if current && Self::is_half(parameter.dtype_raw()) => master.clone(),
            _ => parameter.as_dtype(Dtype::Float32.as_i32()),
        }
    }

    /// Store `weight`, the updated f32 weight, into `parameter`: rounded to
    /// its dtype with a master copy kept for a half-precision one, as it is
    /// for any other.
    pub fn commit(&mut self, parameter: &mut InlineArray, weight: InlineArray) {
        let dtype = parameter.dtype_raw();
        if Self::is_half(dtype) {
            *parameter = weight.as_dtype(dtype);
            self.value = Some(weight);
            self.produced = Some(parameter.clone());
        } else {
            *parameter = weight.as_dtype(dtype);
            self.value = None;
            self.produced = None;
        }
        self.restored = false;
    }

    /// A master copy read back from a checkpoint. The parameter it belongs to
    /// must be the checkpoint's own weight (the master rounded to its dtype),
    /// which is what a resumed run loads; the next update continues from the
    /// master rather than from that rounded weight.
    pub fn restore(value: InlineArray) -> Self {
        Self {
            value: Some(value),
            produced: None,
            restored: true,
        }
    }

    /// The f32 master copy, for a half-precision parameter that has taken a
    /// step.
    pub fn value(&self) -> Option<&InlineArray> {
        self.value.as_ref()
    }

    /// The master copy, mutably.
    pub fn value_mut(&mut self) -> Option<&mut InlineArray> {
        self.value.as_mut()
    }
}

/// Per-parameter Adam state: the two moments, always f32 (`1 − lr·λ` and
/// β₂ = 0.999 are 1.0 in bf16, which is why the arithmetic is f32), and the
/// [master copy](MasterWeight) of a half-precision parameter.
#[non_exhaustive]
pub struct AdamState {
    /// First moment (exponential moving average of gradients), f32.
    pub m: InlineArray,
    /// Second moment (exponential moving average of squared gradients), f32.
    pub v: InlineArray,
    /// The f32 master copy of an f16 or bf16 parameter.
    pub master: MasterWeight,
}

impl AdamState {
    /// A state from restored moments; any master copy is rebuilt from the
    /// parameter at its next step.
    pub fn new(m: InlineArray, v: InlineArray) -> Self {
        Self {
            m,
            v,
            master: MasterWeight::default(),
        }
    }

    /// A state from a checkpoint: the moments and, for a half-precision
    /// parameter, its f32 master copy, from which the next step continues.
    pub fn restored(m: InlineArray, v: InlineArray, master: Option<InlineArray>) -> Self {
        Self {
            m,
            v,
            master: master.map_or_else(MasterWeight::default, MasterWeight::restore),
        }
    }

    /// Every array the state holds, for evaluating it between steps.
    pub fn arrays(&self) -> impl Iterator<Item = &InlineArray> {
        [&self.m, &self.v].into_iter().chain(self.master.value())
    }

    /// [`arrays`](Self::arrays), mutably.
    pub fn arrays_mut(&mut self) -> impl Iterator<Item = &mut InlineArray> {
        [&mut self.m, &mut self.v]
            .into_iter()
            .chain(self.master.value_mut())
    }
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
    rates: GroupRates,
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
            .field("rates", &self.rates)
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
    /// Every class trains at `base_lr` until [`Self::with_embedding_lr`] or
    /// [`Self::with_loraplus_ratio`] says otherwise. `weight_decay` is
    /// applied to all classes except `NoDecay`.
    pub fn new(base_lr: f32, weight_decay: f32) -> Self {
        Self {
            base_lr,
            rates: GroupRates::new(weight_decay),
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

    /// Train `Embedding`-class parameters at `lr` now, and at the same
    /// multiple of the base rate as [`Self::set_lr`] moves it. Without this
    /// they train at the base rate, as every class does.
    pub fn with_embedding_lr(mut self, lr: f32) -> Self {
        self.rates = self.rates.with_embedding_lr(lr, self.base_lr);
        self
    }

    /// Set the LoRA+ scaling factor for `LoraB`-class parameters.
    ///
    /// The effective LR for LoRA B matrices is `base_lr * loraplus_ratio`.
    /// Hayou et al. recommend values in the range 4–16 for most setups.
    /// Without it, LoRA B matrices train at the base rate.
    pub fn with_loraplus_ratio(mut self, ratio: f32) -> Self {
        self.rates = self.rates.with_loraplus_ratio(ratio);
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

    /// Update `base_lr` for LR scheduling. The embedding and LoRA B rates
    /// are multiples of it ([`GroupRates`]), so they move with it.
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
        self.rates.learning_rate(class, self.base_lr)
    }

    /// Return the effective weight decay for a given [`ParamClass`].
    #[inline]
    fn wd_for(&self, class: ParamClass) -> f32 {
        self.rates.weight_decay(class)
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
    /// The step's scalars are computed in f64 and rounded once. Everything
    /// else is f32 whatever the parameter's and gradient's dtype: the moments
    /// are f32, and an f16 or bf16 parameter is updated through an f32 master
    /// copy ([`AdamState`]) and rounded back, keeping its dtype.
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
        let f32_ = Dtype::Float32.as_i32();

        let state = self.states.entry(name.to_string()).or_insert_with(|| {
            let shape = parameter.shape().to_vec();
            AdamState::new(
                InlineArray::zeros(&shape, f32_),
                InlineArray::zeros(&shape, f32_),
            )
        });
        // Restored moments may be of any dtype; the arithmetic is f32.
        for moment in [&mut state.m, &mut state.v] {
            if moment.dtype_raw() != f32_ {
                *moment = moment.as_dtype(f32_);
            }
        }

        let weight = state.master.weight(parameter);
        let gradient = gradient.as_dtype(f32_);

        let weight = if s.decay != 1.0 {
            weight.mul_scalar(s.decay)
        } else {
            weight
        };
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
        let weight = weight.subtract(&state.m.divide(&denom).mul_scalar(s.step_size));

        state.master.commit(parameter, weight);
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
/// Biases and normalization weights ([`excluded_from_weight_decay`]) are
/// [`ParamClass::NoDecay`]; everything else, embeddings and LoRA B matrices
/// included, is [`ParamClass::Regular`] and trains at the base learning
/// rate. This is what an optimizer without parameter groups should do: a
/// separate embedding or LoRA+ rate is a group the caller configures, not
/// something the optimizer infers from a name.
pub fn decay_only_classifier(name: &str) -> ParamClass {
    if excluded_from_weight_decay(name) {
        ParamClass::NoDecay
    } else {
        ParamClass::Regular
    }
}

/// Default parameter classifier based on common naming conventions.
///
/// Rules (first match wins):
/// 1. A bias or normalization weight ([`excluded_from_weight_decay`]) →
///    [`ParamClass::NoDecay`]
/// 2. The token embedding or the output head ([`is_embedding_parameter`]) →
///    [`ParamClass::Embedding`]
/// 3. Name contains `"lora_b"` → [`ParamClass::LoraB`]
/// 4. All other names → [`ParamClass::Regular`]
pub fn default_classifier(name: &str) -> ParamClass {
    if excluded_from_weight_decay(name) {
        ParamClass::NoDecay
    } else if is_embedding_parameter(name) {
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

    /// The token embedding and the output head are embeddings, under the
    /// names `transformers` gives them; nothing else is, however much of
    /// one of those names it contains.
    #[test]
    fn only_the_token_embedding_and_output_head_are_embeddings() {
        for name in [
            // Llama
            "model.embed_tokens.weight",
            "lm_head.weight",
            "lm_head.lora_a",
            // GPT-2
            "transformer.wte.weight",
            // GPT-NeoX
            "gpt_neox.embed_in.weight",
            "embed_out.weight",
            "embed_out.lora_b",
            // BERT
            "bert.embeddings.word_embeddings.weight",
            // GGUF, and a name in another case
            "token_embd.weight",
            "model.Embed_Tokens.weight",
        ] {
            assert!(is_embedding_parameter(name), "{name} is an embedding");
            assert_eq!(default_classifier(name), ParamClass::Embedding, "{name}");
        }
        for name in [
            // BERT: an attention or MLP output projection is not the head.
            "bert.encoder.layer.0.attention.output.dense.weight",
            "bert.encoder.layer.0.attention.output.dense.lora_a",
            "bert.encoder.layer.0.output.dense.lora_b",
            // GPT-2's position embedding is not the token embedding.
            "transformer.wpe.weight",
            "transformer.h.0.attn.c_attn.lora_a",
            "transformer.h.0.mlp.c_proj.weight",
            // GPT-NeoX
            "gpt_neox.layers.0.attention.dense.weight",
            "gpt_neox.layers.0.mlp.dense_4h_to_h.lora_a",
            // Llama
            "model.layers.0.self_attn.o_proj.lora_a",
            "model.layers.0.mlp.down_proj.weight",
            // Names that contain an embedding's name without being one.
            "vision_tower.patch_embed.proj.weight",
            "model.layers.0.output_proj.weight",
            "output.weight",
            "model.lm_head_norm_scale",
        ] {
            assert!(!is_embedding_parameter(name), "{name} is not an embedding");
        }
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

    /// Biases and normalization weights skip weight decay, as
    /// `Trainer.get_decay_parameter_names` decides; everything else decays.
    #[test]
    fn biases_and_normalization_weights_skip_weight_decay() {
        for name in [
            // The name patterns `get_decay_parameter_names` matches.
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "transformer.h.0.mlp.rmsnorm.weight",
            "model.norm.weight",
            "norm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "encoder.layer.0.attention.output.LayerNorm.weight",
            "model.decoder.layers.0.final_layer_norm.weight",
            // `nn.LayerNorm` modules it finds by type.
            "transformer.h.0.ln_1.weight",
            "transformer.ln_f.bias",
            "transformer.ln_f.weight",
            "vision_model.encoder.layers.0.layer_norm1.weight",
            // Normalization layers named neither way.
            "model.layers.3.enorm.weight",
            "model.layers.3.hnorm.weight",
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            "model.layers.0.mixer.norm_q_a.weight",
        ] {
            assert!(excluded_from_weight_decay(name), "{name} should not decay");
            assert_eq!(decay_only_classifier(name), ParamClass::NoDecay, "{name}");
        }
        for name in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.lora_a",
            "model.layers.0.self_attn.q_proj.lora_b",
            "model.layers.0.self_attn.q_proj.lora_magnitude",
            "model.layers.0.mlp.down_proj.weight",
            // A learned scalar is a weight like any other.
            "model.layers.0.layer_scalar",
            "model.layers.0.self_attn.per_dim_scale",
            "transformer.wte.weight",
            "transformer.h.0.attn.c_attn.weight",
        ] {
            assert!(!excluded_from_weight_decay(name), "{name} should decay");
            assert_eq!(decay_only_classifier(name), ParamClass::Regular, "{name}");
        }
    }

    /// The embedding rate is a multiple of the base rate, so the scheduler
    /// moves it as it moves every other class's.
    #[test]
    fn the_embedding_rate_follows_the_base_rate() {
        let mut opt = AdamW::new(2e-4, 0.0).with_embedding_lr(5e-5);
        assert!((opt.lr_for(ParamClass::Embedding) - 5e-5).abs() < 1e-12);
        opt.set_lr(1e-4);
        assert!((opt.lr_for(ParamClass::Embedding) - 2.5e-5).abs() < 1e-12);
        // Without an embedding rate, embeddings train at the base rate.
        let opt = AdamW::new(3e-4, 0.0);
        assert_eq!(opt.lr_for(ParamClass::Embedding), 3e-4);
    }

    #[test]
    fn test_adamw_wd_for() {
        let opt = AdamW::new(1e-4, 0.01);
        assert_eq!(opt.wd_for(ParamClass::NoDecay), 0.0);
        assert!((opt.wd_for(ParamClass::Regular) - 0.01).abs() < 1e-9);
        assert!((opt.wd_for(ParamClass::Embedding) - 0.01).abs() < 1e-9);
        assert!((opt.wd_for(ParamClass::LoraB) - 0.01).abs() < 1e-9);
    }

    fn read(a: &InlineArray) -> Vec<f32> {
        let mut a = a.as_dtype(crate::compat::Dtype::Float32.as_i32());
        a.eval();
        let n = a.size();
        let v = a.to_f32_vec(n).expect("readable");
        crate::check_unobserved_error().expect("no bridge error");
        v
    }

    /// `torch.optim.AdamW` in f64 for one scalar parameter under a constant
    /// gradient.
    fn adamw_reference(
        mut p: f64,
        g: f64,
        steps: u64,
        (lr, wd, b1, b2, eps): (f64, f64, f64, f64, f64),
    ) -> f64 {
        let (mut m, mut v) = (0.0, 0.0);
        for t in 1..=steps {
            p *= 1.0 - lr * wd;
            m = b1 * m + (1.0 - b1) * g;
            v = b2 * v + (1.0 - b2) * g * g;
            let step = lr / (1.0 - b1.powi(t as i32));
            p -= step * m / ((v.sqrt() / (1.0 - b2.powi(t as i32)).sqrt()) + eps);
        }
        p
    }

    /// A bf16 parameter trains: its moments and its update are f32, and the
    /// parameter is rounded from an f32 master copy. Held in bf16 alone, an
    /// update of 1e-4 to a weight near 1 (bf16 steps are 4e-3 wide there)
    /// rounds away every time, `1 - lr·λ` is 1.0, and β₂ = 0.999 is 1.0.
    #[test]
    fn a_bf16_parameter_keeps_training_through_an_f32_master_copy() {
        use crate::compat::Dtype;
        let (lr, wd, b1, b2, eps) = (1e-4_f64, 0.1, 0.9, 0.999, 1e-8);
        let steps = 300;
        let bf16 = Dtype::Bfloat16.as_i32();
        let mut opt = AdamW::new(lr as f32, wd as f32);
        let mut param = InlineArray::from_f32_slice(&[1.0, -2.0], &[2]).as_dtype(bf16);
        let grad = InlineArray::from_f32_slice(&[0.5, -0.25], &[2]).as_dtype(bf16);
        for _ in 0..steps {
            opt.advance_step();
            opt.step_single("w", &grad, &mut param);
            param.eval();
        }
        let want = [
            adamw_reference(1.0, 0.5, steps, (lr, wd, b1, b2, eps)),
            adamw_reference(-2.0, -0.25, steps, (lr, wd, b1, b2, eps)),
        ];
        assert_eq!(
            param.dtype(),
            Dtype::Bfloat16,
            "the parameter keeps its dtype"
        );
        let state = &opt.states["w"];
        assert_eq!(state.m.dtype(), Dtype::Float32);
        assert_eq!(state.v.dtype(), Dtype::Float32);
        let master = read(state.master.value().expect("a bf16 parameter has a master"));
        for i in 0..2 {
            assert!(
                (master[i] as f64 - want[i]).abs() < 1e-4,
                "master {i}: {} vs {}",
                master[i],
                want[i]
            );
        }
        // The parameter is the master rounded to bf16 (a step is 2^-8 wide).
        let got = read(&param);
        for i in 0..2 {
            assert!(
                (got[i] as f64 - want[i]).abs() <= want[i].abs() / 256.0,
                "param {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    /// A parameter replaced between steps (a checkpoint loaded, an adapter
    /// merged) is the optimizer's new starting point, not overwritten by the
    /// master copy of the old one.
    #[test]
    fn a_replaced_parameter_reseeds_its_master_copy() {
        use crate::compat::Dtype;
        let bf16 = Dtype::Bfloat16.as_i32();
        let mut opt = AdamW::new(1e-3, 0.0);
        let grad = InlineArray::from_f32_slice(&[1.0], &[1]).as_dtype(bf16);
        let mut param = InlineArray::from_f32_slice(&[1.0], &[1]).as_dtype(bf16);
        opt.advance_step();
        opt.step_single("w", &grad, &mut param);
        param.eval();

        param = InlineArray::from_f32_slice(&[8.0], &[1]).as_dtype(bf16);
        opt.advance_step();
        opt.step_single("w", &grad, &mut param);
        let got = read(&param)[0];
        assert!(
            (got - 8.0).abs() < 0.05,
            "stepped from the old master: {got}"
        );
    }

    /// f32 parameters are updated in place, with no master copy.
    #[test]
    fn an_f32_parameter_needs_no_master_copy() {
        let mut opt = AdamW::new(1e-3, 0.0);
        let mut param = InlineArray::from_f32_slice(&[1.0], &[1]);
        opt.advance_step();
        opt.step_single("w", &InlineArray::from_f32_slice(&[1.0], &[1]), &mut param);
        assert!(opt.states["w"].master.value().is_none());
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
