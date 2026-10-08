//! The optimizers `TrainingConfig::optimizer` selects, and the parameter
//! groups every training loop wraps them in.
//!
//! [`TrainOptimizer`] is one optimizer of the kind a [`OptimizerType`] names.
//! [`ParamGroupOptimizer`] holds one per parameter group: embeddings at their
//! own learning rate, LoRA B matrices at the LoRA+ rate, and bias, norm and
//! scale parameters without weight decay. Every group follows the learning
//! rate the scheduler sets through [`ParamGroupOptimizer::set_learning_rate`],
//! keeping its ratio to the base rate.
//!
//! Each update rule follows the paper that introduced it, with that paper's
//! default hyperparameters:
//!
//! | Kind | Rule | Defaults |
//! |------|------|----------|
//! | AdamW | Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (arXiv 1711.05101) | β = (0.9, 0.999), ε = 1e-8 |
//! | SGD | heavy-ball momentum, Sutskever et al., *On the importance of initialization and momentum in deep learning* (ICML 2013); L2 weight decay added to the gradient | momentum 0.9 |
//! | Lion | Chen et al., *Symbolic Discovery of Optimization Algorithms* (arXiv 2302.06675) | β = (0.9, 0.99), decoupled weight decay |
//! | Adafactor | Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* (arXiv 1804.04235) | ε₁ = 1e-30, update clip d = 1, β̂₂ₜ = 1 − t^-0.8, no first moment, external learning rate |
//!
//! The learning rate and weight decay are the caller's. They are not
//! interchangeable between kinds: the Lion paper recommends a learning rate
//! 3-10x smaller and a weight decay 3-10x larger than AdamW's, and Adafactor
//! with an external learning rate is usually run around 1e-3.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, FlattenedModuleParam,
    module::{ModuleParameters, ModuleParametersExt},
    optimizers::{AdamW, AdamWBuilder, Optimizer, State, Updatable},
};
use pmetal_core::{OptimizerType, TrainingConfig};

use crate::ParameterGroupConfig;

type Result<T> = std::result::Result<T, Exception>;

/// SGD with heavy-ball momentum and L2 weight decay.
///
/// `buf = μ·buf + (g + λ·p)`, `p -= lr·buf`, with the buffer starting at the
/// first gradient (no dampening, no Nesterov).
#[derive(Debug)]
pub struct MomentumSgd {
    lr: f32,
    weight_decay: f32,
    momentum: f32,
    /// Momentum buffer per parameter.
    pub state: State<Array>,
}

impl MomentumSgd {
    /// Momentum coefficient (Sutskever et al., 2013).
    pub const MOMENTUM: f32 = 0.9;

    /// SGD with the default momentum.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            momentum: Self::MOMENTUM,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for MomentumSgd {
    type State = State<Array>;
    fn state(&self) -> &Self::State {
        &self.state
    }
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        let mut d = gradient.clone();
        if self.weight_decay != 0.0 {
            d = d.add(&parameter.mul_scalar(self.weight_decay));
        }
        let buf = match self.state.remove(key) {
            Some(buf) => buf.mul_scalar(self.momentum).add(&d),
            None => d,
        };
        *parameter = parameter.subtract(&buf.mul_scalar(self.lr));
        self.state.insert(key.clone(), buf);
        Ok(())
    }
}

/// Lion: the sign of an interpolated momentum, with decoupled weight decay.
///
/// ```text
/// c = β₁·m + (1−β₁)·g
/// p = p − lr·(sign(c) + λ·p)
/// m = β₂·m + (1−β₂)·g
/// ```
#[derive(Debug)]
pub struct Lion {
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    /// Momentum per parameter.
    pub state: State<Array>,
}

impl Lion {
    /// β₁ and β₂ from the Lion paper.
    pub const BETAS: (f32, f32) = (0.9, 0.99);

    /// Lion with the paper's betas.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            beta1: Self::BETAS.0,
            beta2: Self::BETAS.1,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for Lion {
    type State = State<Array>;
    fn state(&self) -> &Self::State {
        &self.state
    }
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        let m = self
            .state
            .remove(key)
            .unwrap_or_else(|| gradient.zeros_like());
        let c = m
            .mul_scalar(self.beta1)
            .add(&gradient.mul_scalar(1.0 - self.beta1));
        if self.weight_decay != 0.0 {
            *parameter = parameter.subtract(&parameter.mul_scalar(self.lr * self.weight_decay));
        }
        *parameter = parameter.subtract(&c.sign().mul_scalar(self.lr));
        let m = m
            .mul_scalar(self.beta2)
            .add(&gradient.mul_scalar(1.0 - self.beta2));
        self.state.insert(key.clone(), m);
        Ok(())
    }
}

/// Adafactor's second-moment estimate for one parameter.
#[derive(Debug)]
pub enum SecondMoment {
    /// Row and column means of the squared gradient, for matrices.
    Factored { row: Array, col: Array },
    /// The full squared-gradient average, for vectors and scalars.
    Full(Array),
}

/// Adafactor with an external learning rate.
///
/// Matrices keep only the row and column means of the squared gradient; the
/// update is that estimate's inverse square root times the gradient, scaled
/// down so its RMS is at most `clip_threshold`. The decay of the estimate
/// grows with the step, `β̂₂ₜ = 1 − t^decay_rate`. The learning rate is the
/// scheduler's: relative step sizes and parameter-scale multiplication, the
/// paper's other two options, are off, as for fine-tuning with a schedule.
/// Statistics are kept in f32 whatever the parameter's dtype.
#[derive(Debug)]
pub struct Adafactor {
    lr: f32,
    weight_decay: f32,
    eps1: f32,
    clip_threshold: f32,
    decay_rate: f32,
    step: u64,
    /// Second-moment estimate per parameter.
    pub state: State<SecondMoment>,
}

impl Adafactor {
    /// ε₁, added to the squared gradient.
    pub const EPS1: f32 = 1e-30;
    /// d, the largest RMS an update may have.
    pub const CLIP_THRESHOLD: f32 = 1.0;
    /// The exponent in `β̂₂ₜ = 1 − t^decay_rate`.
    pub const DECAY_RATE: f32 = -0.8;

    /// Adafactor with the paper's defaults.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            eps1: Self::EPS1,
            clip_threshold: Self::CLIP_THRESHOLD,
            decay_rate: Self::DECAY_RATE,
            step: 0,
            state: HashMap::new(),
        }
    }

    /// Advance the step that `β̂₂ₜ` decays with. Once per optimizer update.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }
}

impl Optimizer for Adafactor {
    type State = State<SecondMoment>;
    fn state(&self) -> &Self::State {
        &self.state
    }
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        let f32_dtype = pmetal_bridge::dtype::F32;
        let param_dtype = parameter.dtype_raw();
        let t = self.step.max(1) as f32;
        let beta2t = 1.0 - t.powf(self.decay_rate);
        let g = gradient.as_dtype(f32_dtype);
        let sq = g.square().add_scalar(self.eps1);

        let (update, moment) = if g.ndim() >= 2 {
            let (row, col) = match self.state.remove(key) {
                Some(SecondMoment::Factored { row, col }) => (row, col),
                _ => (
                    sq.mean_axis(-1, false).zeros_like(),
                    sq.mean_axis(-2, false).zeros_like(),
                ),
            };
            let row = row
                .mul_scalar(beta2t)
                .add(&sq.mean_axis(-1, false).mul_scalar(1.0 - beta2t));
            let col = col
                .mul_scalar(beta2t)
                .add(&sq.mean_axis(-2, false).mul_scalar(1.0 - beta2t));
            let r_factor = row.divide(&row.mean_axis(-1, true)).rsqrt().expand_dims(-1);
            let c_factor = col.rsqrt().expand_dims(-2);
            (
                r_factor.multiply(&c_factor).multiply(&g),
                SecondMoment::Factored { row, col },
            )
        } else {
            let v = match self.state.remove(key) {
                Some(SecondMoment::Full(v)) => v,
                _ => sq.zeros_like(),
            };
            let v = v.mul_scalar(beta2t).add(&sq.mul_scalar(1.0 - beta2t));
            (v.rsqrt().multiply(&g), SecondMoment::Full(v))
        };

        // Update clipping: divide by max(1, RMS(update) / d).
        let rms = update.square().mean_all().sqrt();
        let denom = rms
            .div_scalar(self.clip_threshold)
            .maximum(&Array::from_f32(1.0));
        let update = update.divide(&denom);

        let mut p = parameter.as_dtype(f32_dtype);
        if self.weight_decay != 0.0 {
            p = p.subtract(&p.mul_scalar(self.weight_decay * self.lr));
        }
        p = p.subtract(&update.mul_scalar(self.lr));
        *parameter = p.as_dtype(param_dtype);
        self.state.insert(key.clone(), moment);
        Ok(())
    }
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<()> {
        self.advance_step();
        apply_to_model(self, model, gradients)
    }
}

/// Run `update_single` over every parameter that has a gradient.
fn apply_to_model<O: Optimizer, M: ModuleParameters>(
    optimizer: &mut O,
    model: &mut M,
    gradients: FlattenedModuleParam,
) -> Result<()> {
    let mut flat = model.flatten_params_mut();
    for (key, grad) in &gradients {
        if let Some(arr) = flat.get_mut(key.as_ref()) {
            optimizer.update_single(key, grad, arr)?;
        }
    }
    Ok(())
}

/// One optimizer of the kind an [`OptimizerType`] names.
#[derive(Debug)]
pub enum TrainOptimizer {
    /// AdamW.
    AdamW(AdamW),
    /// SGD with momentum.
    Sgd(MomentumSgd),
    /// Lion.
    Lion(Lion),
    /// Adafactor.
    Adafactor(Adafactor),
}

impl TrainOptimizer {
    /// An optimizer of `kind` at learning rate `lr` and weight decay
    /// `weight_decay`, with the kind's paper defaults for everything else.
    pub fn new(kind: OptimizerType, lr: f32, weight_decay: f32) -> Self {
        match kind {
            OptimizerType::AdamW => Self::AdamW(
                AdamWBuilder::new(lr)
                    .weight_decay(weight_decay)
                    .build()
                    .expect("AdamWBuilder::build is infallible"),
            ),
            OptimizerType::Sgd => Self::Sgd(MomentumSgd::new(lr, weight_decay)),
            OptimizerType::Lion => Self::Lion(Lion::new(lr, weight_decay)),
            OptimizerType::Adafactor => Self::Adafactor(Adafactor::new(lr, weight_decay)),
        }
    }

    /// The optimizer `config` names, at its learning rate and weight decay.
    pub fn from_config(config: &TrainingConfig) -> Self {
        Self::new(
            config.optimizer,
            config.learning_rate as f32,
            config.weight_decay as f32,
        )
    }

    /// Which kind this is.
    pub fn kind(&self) -> OptimizerType {
        match self {
            Self::AdamW(_) => OptimizerType::AdamW,
            Self::Sgd(_) => OptimizerType::Sgd,
            Self::Lion(_) => OptimizerType::Lion,
            Self::Adafactor(_) => OptimizerType::Adafactor,
        }
    }

    /// The learning rate the next update uses.
    pub fn lr(&self) -> f32 {
        match self {
            Self::AdamW(o) => o.lr(),
            Self::Sgd(o) => o.lr,
            Self::Lion(o) => o.lr,
            Self::Adafactor(o) => o.lr,
        }
    }

    /// Set the learning rate the next update uses.
    pub fn set_lr(&mut self, lr: f32) {
        match self {
            Self::AdamW(o) => o.set_lr(lr),
            Self::Sgd(o) => o.lr = lr,
            Self::Lion(o) => o.lr = lr,
            Self::Adafactor(o) => o.lr = lr,
        }
    }

    /// Advance the optimizer's step counter. Once per update, before the
    /// per-parameter calls; [`Optimizer::update`] does this itself.
    pub fn advance_step(&mut self) {
        match self {
            Self::AdamW(o) => o.advance_step(),
            Self::Adafactor(o) => o.advance_step(),
            Self::Sgd(_) | Self::Lion(_) => {}
        }
    }

    /// Number of parameters this optimizer holds state for.
    pub fn num_params(&self) -> usize {
        match self {
            Self::AdamW(o) => o.state.len(),
            Self::Sgd(o) => o.state.len(),
            Self::Lion(o) => o.state.len(),
            Self::Adafactor(o) => o.state.len(),
        }
    }
}

impl Optimizer for TrainOptimizer {
    /// The optimizer is its own state; each kind keeps a different one.
    type State = Self;
    fn state(&self) -> &Self::State {
        self
    }
    fn state_mut(&mut self) -> &mut Self::State {
        self
    }
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        match self {
            Self::AdamW(o) => o.update_single(key, gradient, parameter),
            Self::Sgd(o) => o.update_single(key, gradient, parameter),
            Self::Lion(o) => o.update_single(key, gradient, parameter),
            Self::Adafactor(o) => o.update_single(key, gradient, parameter),
        }
    }
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<()> {
        self.advance_step();
        apply_to_model(self, model, gradients)
    }
}

impl Updatable for TrainOptimizer {
    fn updatable_states_len(&self) -> usize {
        self.updatable_states().len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        match self {
            Self::AdamW(o) => o.updatable_states(),
            Self::Sgd(o) => o.state.values().collect(),
            Self::Lion(o) => o.state.values().collect(),
            Self::Adafactor(o) => o
                .state
                .values()
                .flat_map(|m| match m {
                    SecondMoment::Factored { row, col } => vec![row, col],
                    SecondMoment::Full(v) => vec![v],
                })
                .collect(),
        }
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        match self {
            Self::AdamW(o) => o.updatable_states_mut(),
            Self::Sgd(o) => o.state.values_mut().collect(),
            Self::Lion(o) => o.state.values_mut().collect(),
            Self::Adafactor(o) => o
                .state
                .values_mut()
                .flat_map(|m| match m {
                    SecondMoment::Factored { row, col } => vec![row, col],
                    SecondMoment::Full(v) => vec![v],
                })
                .collect(),
        }
    }
}

/// Which group a parameter trains in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamClass {
    /// Base learning rate, full weight decay: LoRA A, projections.
    Regular,
    /// The embedding learning rate, full weight decay.
    Embedding,
    /// Base learning rate, no weight decay: bias, norm, scale.
    NoDecay,
    /// The LoRA+ learning rate (`base * ratio`, Hayou et al., ICML 2024).
    LoraB,
}

/// An optimizer per parameter group, all of one [`OptimizerType`].
///
/// Bias, norm and scale parameters train without weight decay, embeddings at
/// their own learning rate, and with LoRA+ the LoRA B matrices at
/// `base_lr * ratio`. [`Self::set_learning_rate`] moves every group's rate
/// with the base rate, keeping the ratios.
#[derive(Debug)]
pub struct ParamGroupOptimizer {
    regular: TrainOptimizer,
    lora_b: Option<TrainOptimizer>,
    embedding: TrainOptimizer,
    no_decay: TrainOptimizer,
    base_lr: f32,
    embedding_ratio: f32,
    loraplus_ratio: Option<f32>,
    config: ParameterGroupConfig,
    param_cache: HashMap<Rc<str>, ParamClass>,
}

impl ParamGroupOptimizer {
    /// Which kind of optimizer every group runs.
    pub fn kind(&self) -> OptimizerType {
        self.regular.kind()
    }

    fn classify_param(&mut self, name: &Rc<str>) -> ParamClass {
        if let Some(&class) = self.param_cache.get(name) {
            return class;
        }
        let name_lower = name.to_lowercase();
        let class = if self.config.is_no_decay(name) {
            ParamClass::NoDecay
        } else if self
            .config
            .embedding_patterns
            .iter()
            .any(|pattern| name_lower.contains(&pattern.to_lowercase()))
        {
            ParamClass::Embedding
        } else if self.lora_b.is_some()
            && (name_lower.contains("lora_b") || name_lower.contains("lorab"))
        {
            ParamClass::LoraB
        } else {
            ParamClass::Regular
        };
        self.param_cache.insert(name.clone(), class);
        class
    }

    fn group_mut(&mut self, class: ParamClass) -> &mut TrainOptimizer {
        match class {
            ParamClass::Regular => &mut self.regular,
            ParamClass::Embedding => &mut self.embedding,
            ParamClass::NoDecay => &mut self.no_decay,
            ParamClass::LoraB => self.lora_b.as_mut().unwrap_or(&mut self.regular),
        }
    }

    fn groups(&self) -> impl Iterator<Item = &TrainOptimizer> {
        [&self.regular, &self.embedding, &self.no_decay]
            .into_iter()
            .chain(self.lora_b.as_ref())
    }

    fn groups_mut(&mut self) -> impl Iterator<Item = &mut TrainOptimizer> {
        [&mut self.regular, &mut self.embedding, &mut self.no_decay]
            .into_iter()
            .chain(self.lora_b.as_mut())
    }

    /// `(base_lr, embedding_lr)` for the next update.
    pub fn learning_rates(&self) -> (f32, f32) {
        (self.regular.lr(), self.embedding.lr())
    }

    /// Set the base learning rate. The embedding and LoRA B groups keep their
    /// ratio to it; the no-decay group runs at it.
    pub fn set_learning_rate(&mut self, base_lr: f32) {
        self.base_lr = base_lr;
        self.regular.set_lr(base_lr);
        self.no_decay.set_lr(base_lr);
        self.embedding.set_lr(base_lr * self.embedding_ratio);
        if let (Some(b), Some(ratio)) = (self.lora_b.as_mut(), self.loraplus_ratio) {
            b.set_lr(base_lr * ratio);
        }
    }

    /// One line describing the groups, for logs.
    pub fn summary(&self) -> String {
        let (base_lr, emb_lr) = self.learning_rates();
        let mut s = format!(
            "{:?}: {} regular params (lr={base_lr:.2e}), {} embedding params (lr={emb_lr:.2e}), \
             {} no-decay params (wd=0)",
            self.kind(),
            self.regular.num_params(),
            self.embedding.num_params(),
            self.no_decay.num_params(),
        );
        if let Some(b) = &self.lora_b {
            s.push_str(&format!(
                ", {} LoRA B params (lr={:.2e})",
                b.num_params(),
                b.lr()
            ));
        }
        s
    }
}

impl Optimizer for ParamGroupOptimizer {
    /// The optimizer is its own state; each group keeps its own.
    type State = Self;
    fn state(&self) -> &Self::State {
        self
    }
    fn state_mut(&mut self) -> &mut Self::State {
        self
    }
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        let class = self.classify_param(key);
        self.group_mut(class)
            .update_single(key, gradient, parameter)
    }
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<()> {
        // Every group advances together so bias corrections and Adafactor's
        // decay see the same step whichever parameters a group holds.
        for group in self.groups_mut() {
            group.advance_step();
        }
        apply_to_model(self, model, gradients)
    }
}

impl Updatable for ParamGroupOptimizer {
    fn updatable_states_len(&self) -> usize {
        self.groups().map(|g| g.updatable_states_len()).sum()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        self.groups().flat_map(|g| g.updatable_states()).collect()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        self.groups_mut()
            .flat_map(|g| g.updatable_states_mut())
            .collect()
    }
}

/// Builder for [`ParamGroupOptimizer`].
#[derive(Debug, Clone)]
pub struct ParamGroupOptimizerBuilder {
    kind: OptimizerType,
    base_lr: f32,
    embedding_lr: Option<f32>,
    weight_decay: f32,
    embedding_patterns: Vec<String>,
    loraplus_lr_ratio: Option<f32>,
}

impl ParamGroupOptimizerBuilder {
    /// Groups of `kind` optimizers at base learning rate `base_lr`, with
    /// weight decay 0.01 until [`Self::with_weight_decay`] says otherwise.
    pub fn new(kind: OptimizerType, base_lr: f32) -> Self {
        Self {
            kind,
            base_lr,
            embedding_lr: None,
            weight_decay: 0.01,
            embedding_patterns: ParameterGroupConfig::default().embedding_patterns,
            loraplus_lr_ratio: None,
        }
    }

    /// The groups `config` describes: its optimizer, learning rate and weight
    /// decay, and its embedding learning rate if it sets one.
    pub fn from_config(config: &TrainingConfig) -> Self {
        let mut builder = Self::new(config.optimizer, config.learning_rate as f32)
            .with_weight_decay(config.weight_decay as f32);
        if let Some(lr) = config.embedding_learning_rate {
            builder = builder.with_embedding_lr(lr as f32);
        }
        builder
    }

    /// Train embedding parameters at `lr`, scaled with the base rate.
    pub fn with_embedding_lr(mut self, lr: f32) -> Self {
        self.embedding_lr = Some(lr);
        self
    }

    /// Weight decay for every group but the no-decay one.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Count names containing `pattern` as embedding parameters.
    pub fn add_embedding_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.embedding_patterns.push(pattern.into());
        self
    }

    /// Train LoRA B matrices at `base_lr * ratio` (LoRA+; the paper suggests 16).
    pub fn with_loraplus_lr_ratio(mut self, ratio: f32) -> Self {
        self.loraplus_lr_ratio = Some(ratio);
        self
    }

    /// Build the groups.
    pub fn build(self) -> ParamGroupOptimizer {
        let base_lr = self.base_lr;
        let embedding_ratio = match self.embedding_lr {
            Some(lr) if base_lr > 0.0 => lr / base_lr,
            _ => 1.0,
        };
        let make = |lr: f32, wd: f32| TrainOptimizer::new(self.kind, lr, wd);
        let lora_b = self.loraplus_lr_ratio.map(|ratio| {
            tracing::info!(
                "LoRA+ enabled: A matrices lr={:.2e}, B matrices lr={:.2e} (ratio={ratio})",
                base_lr,
                base_lr * ratio,
            );
            make(base_lr * ratio, self.weight_decay)
        });
        ParamGroupOptimizer {
            regular: make(base_lr, self.weight_decay),
            lora_b,
            embedding: make(base_lr * embedding_ratio, self.weight_decay),
            no_decay: make(base_lr, 0.0),
            base_lr,
            embedding_ratio,
            loraplus_ratio: self.loraplus_lr_ratio,
            config: ParameterGroupConfig {
                base_lr: base_lr as f64,
                embedding_lr: self.embedding_lr.map(f64::from),
                weight_decay: self.weight_decay as f64,
                embedding_patterns: self.embedding_patterns,
                ..Default::default()
            },
            param_cache: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests;
