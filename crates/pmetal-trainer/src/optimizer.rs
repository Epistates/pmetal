//! The optimizers `TrainingConfig::optimizer` selects, and the parameter
//! groups every training loop wraps them in.
//!
//! [`TrainOptimizer`] is one optimizer of the kind a [`OptimizerType`] names.
//! [`ParamGroupOptimizer`] holds one per parameter group
//! ([`ParameterGroupConfig`]): embeddings at their own learning rate, LoRA B
//! matrices at the LoRA+ rate, and biases and normalization weights without
//! weight decay. Every group follows the learning rate the scheduler sets
//! through [`ParamGroupOptimizer::set_learning_rate`], keeping its ratio to
//! the base rate.
//!
//! Each update rule follows the paper that introduced it, with that paper's
//! default hyperparameters, and computes what its reference implementation
//! does (`tests/optimizer_parity.rs` holds each to it, step by step):
//!
//! | Kind | Rule | Defaults | Reference |
//! |------|------|----------|-----------|
//! | AdamW | Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (arXiv 1711.05101), bias-corrected | β = (0.9, 0.999), ε = 1e-8 | `torch.optim.AdamW` |
//! | SGD | heavy-ball momentum, Sutskever et al., *On the importance of initialization and momentum in deep learning* (ICML 2013); L2 weight decay added to the gradient | momentum 0.9 | `torch.optim.SGD` |
//! | Lion | Chen et al., *Symbolic Discovery of Optimization Algorithms* (arXiv 2302.06675) | β = (0.9, 0.99), decoupled weight decay | the authors' PyTorch implementation |
//! | Adafactor | Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* (arXiv 1804.04235) | ε₁ = 1e-30, update clip d = 1, β̂₂ₜ = 1 − t^-0.8, no first moment, external learning rate | the reference Adafactor with `relative_step=False, scale_parameter=False` |
//! | Muon | Jordan et al., *Muon: An optimizer for hidden layers in neural networks* (2024): momentum 0.95 with Nesterov, five quintic Newton-Schulz steps; update scaled by `0.2·sqrt(max(rows, cols))` to AdamW's RMS (Liu et al., arXiv 2502.16982), decoupled weight decay; embeddings, the head, norms and biases train with AdamW | momentum 0.95, ns steps 5 | `torch.optim.Muon` with `adjust_lr_fn="match_rms_adamw"` |
//! | Schedule-free AdamW | Defazio et al., *The Road Less Scheduled* (arXiv 2405.15682): iterate averaging, weight decay at `y`, `r = 0`, `weight_lr_power = 2`; a constant rate after warmup | β = (0.9, 0.999), ε = 1e-8 | the authors' `AdamWScheduleFree` |
//! | 8-bit AdamW | Dettmers et al., *8-bit Optimizers via Block-wise Quantization* (arXiv 2110.02861): both moments one byte per element, blocks of 256, dynamic code; tensors under 4096 elements keep f32 moments | as AdamW | the authors' blockwise 8-bit AdamW |
//!
//! The learning rate and weight decay are the caller's. They are not
//! interchangeable between kinds: the Lion paper recommends a learning rate
//! 3-10x smaller and a weight decay 3-10x larger than AdamW's, and Adafactor
//! with an external learning rate is usually run around 1e-3.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_runtime::compat::{
    Array, Exception, FlattenedModuleParam,
    module::ModuleParameters,
    optimizers::{
        AdamW, AdamW8bit, AdamWBuilder, LrAdjust, Muon, MuonAdamW, Optimizer, OptimizerState,
        ScheduleFreeAdamW, StatefulOptimizer, Updatable, apply_updates,
    },
};
// The update rules live beside AdamW in the runtime; they are re-exported here
// where the trainer's callers have always found them.
use pmetal_core::{OptimizerType, TrainingConfig};
pub use pmetal_runtime::compat::optimizers::{Adafactor, Lion, MomentumSgd, SecondMoment};

use crate::{ParamClass, ParameterGroupConfig};

type Result<T> = std::result::Result<T, Exception>;

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
    /// Muon on the hidden matrices, AdamW on the rest.
    Muon(Box<MuonAdamW>),
    /// Schedule-free AdamW.
    ScheduleFree(ScheduleFreeAdamW),
    /// AdamW with 8-bit moments.
    AdamW8bit(Box<AdamW8bit>),
}

impl TrainOptimizer {
    /// An optimizer of `kind` at learning rate `lr` and weight decay
    /// `weight_decay`, with the kind's paper defaults for everything else.
    ///
    /// Muon runs Moonshot's recipe: its update is scaled to AdamW's RMS
    /// ([`LrAdjust::MatchRmsAdamw`]), so `lr` and `weight_decay` are the
    /// AdamW values for both halves.
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
            OptimizerType::Muon => Self::Muon(Box::new(
                MuonAdamW::new(lr, lr, weight_decay)
                    .with_muon(Muon::new(lr, weight_decay).with_lr_adjust(LrAdjust::MatchRmsAdamw)),
            )),
            OptimizerType::AdamWScheduleFree => {
                Self::ScheduleFree(ScheduleFreeAdamW::new(lr, weight_decay))
            }
            OptimizerType::AdamW8bit => Self::AdamW8bit(Box::new(AdamW8bit::new(lr, weight_decay))),
        }
    }

    /// The optimizer of `kind` for the parameters of `class`.
    ///
    /// Muon is for the hidden matrices only: the embedding and no-decay
    /// groups (the token embedding, the output head, norms and biases) train
    /// with AdamW at their rates.
    pub fn for_class(kind: OptimizerType, class: ParamClass, lr: f32, weight_decay: f32) -> Self {
        match (kind, class) {
            (OptimizerType::Muon, ParamClass::Embedding | ParamClass::NoDecay) => {
                Self::new(OptimizerType::AdamW, lr, weight_decay)
            }
            _ => Self::new(kind, lr, weight_decay),
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
            Self::Muon(_) => OptimizerType::Muon,
            Self::ScheduleFree(_) => OptimizerType::AdamWScheduleFree,
            Self::AdamW8bit(_) => OptimizerType::AdamW8bit,
        }
    }

    /// The learning rate the next update uses.
    pub fn lr(&self) -> f32 {
        match self {
            Self::AdamW(o) => o.lr(),
            Self::Sgd(o) => o.lr,
            Self::Lion(o) => o.lr,
            Self::Adafactor(o) => o.lr,
            Self::Muon(o) => o.lr(),
            Self::ScheduleFree(o) => o.lr,
            Self::AdamW8bit(o) => o.lr(),
        }
    }

    /// Set the learning rate the next update uses.
    pub fn set_lr(&mut self, lr: f32) {
        match self {
            Self::AdamW(o) => o.set_lr(lr),
            Self::Sgd(o) => o.lr = lr,
            Self::Lion(o) => o.lr = lr,
            Self::Adafactor(o) => o.lr = lr,
            Self::Muon(o) => o.set_lr(lr),
            Self::ScheduleFree(o) => o.lr = lr,
            Self::AdamW8bit(o) => o.set_lr(lr),
        }
    }

    /// Advance the optimizer's step counter. Once per update, before the
    /// per-parameter calls; [`Optimizer::update`] does this itself.
    pub fn advance_step(&mut self) {
        match self {
            Self::AdamW(o) => o.advance_step(),
            Self::Adafactor(o) => o.advance_step(),
            Self::Muon(o) => o.advance_step(),
            Self::ScheduleFree(o) => o.advance_step(),
            Self::AdamW8bit(o) => o.advance_step(),
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
            Self::Muon(o) => o.muon().state.len() + o.adamw().state.len(),
            Self::ScheduleFree(o) => o.num_params(),
            Self::AdamW8bit(o) => o.num_params(),
        }
    }

    /// Swap `model` to the weights to evaluate and save. Only schedule-free
    /// AdamW has any: it trains on `y` and evaluates the average `x`. Every
    /// other optimizer trains the weights it evaluates, and this does
    /// nothing.
    pub fn eval_weights<M: ModuleParameters>(&mut self, model: &mut M) {
        if let Self::ScheduleFree(o) = self {
            o.eval_weights(model);
        }
    }

    /// Swap `model` back to the weights to train on after
    /// [`eval_weights`](Self::eval_weights).
    pub fn train_weights<M: ModuleParameters>(&mut self, model: &mut M) {
        if let Self::ScheduleFree(o) = self {
            o.train_weights(model);
        }
    }

    fn inner_state(&self) -> OptimizerState {
        match self {
            Self::AdamW(o) => o.export_state(),
            Self::Sgd(o) => o.export_state(),
            Self::Lion(o) => o.export_state(),
            Self::Adafactor(o) => o.export_state(),
            Self::Muon(o) => o.export_state(),
            Self::ScheduleFree(o) => o.export_state(),
            Self::AdamW8bit(o) => o.export_state(),
        }
    }
}

impl StatefulOptimizer for TrainOptimizer {
    fn export_state(&self) -> OptimizerState {
        self.inner_state()
    }

    fn import_state(&mut self, state: &OptimizerState) -> Result<()> {
        match self {
            Self::AdamW(o) => o.import_state(state),
            Self::Sgd(o) => o.import_state(state),
            Self::Lion(o) => o.import_state(state),
            Self::Adafactor(o) => o.import_state(state),
            Self::Muon(o) => o.import_state(state),
            Self::ScheduleFree(o) => o.import_state(state),
            Self::AdamW8bit(o) => o.import_state(state),
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
            Self::Muon(o) => o.update_single(key, gradient, parameter),
            Self::ScheduleFree(o) => o.update_single(key, gradient, parameter),
            Self::AdamW8bit(o) => o.update_single(key, gradient, parameter),
        }
    }
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<()> {
        self.advance_step();
        apply_updates(self, model, gradients)
    }
}

impl Updatable for TrainOptimizer {
    fn updatable_states_len(&self) -> usize {
        self.updatable_states().len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        match self {
            Self::AdamW(o) => o.updatable_states(),
            Self::Sgd(o) => o.updatable_states(),
            Self::Lion(o) => o.updatable_states(),
            Self::Adafactor(o) => o.updatable_states(),
            Self::Muon(o) => o.updatable_states(),
            Self::ScheduleFree(o) => o.updatable_states(),
            Self::AdamW8bit(o) => o.updatable_states(),
        }
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        match self {
            Self::AdamW(o) => o.updatable_states_mut(),
            Self::Sgd(o) => o.updatable_states_mut(),
            Self::Lion(o) => o.updatable_states_mut(),
            Self::Adafactor(o) => o.updatable_states_mut(),
            Self::Muon(o) => o.updatable_states_mut(),
            Self::ScheduleFree(o) => o.updatable_states_mut(),
            Self::AdamW8bit(o) => o.updatable_states_mut(),
        }
    }
}

/// An optimizer per parameter group, all of one [`OptimizerType`].
///
/// The groups are a [`ParameterGroupConfig`]'s: biases and normalization
/// weights train without weight decay, embeddings at their own learning
/// rate, and with LoRA+ the LoRA B matrices at `base_lr * ratio`.
/// [`Self::set_learning_rate`] moves every group's rate with the base rate,
/// keeping the ratios.
#[derive(Debug)]
pub struct ParamGroupOptimizer {
    kind: OptimizerType,
    regular: TrainOptimizer,
    lora_b: Option<TrainOptimizer>,
    embedding: TrainOptimizer,
    no_decay: TrainOptimizer,
    groups: ParameterGroupConfig,
    param_cache: HashMap<Rc<str>, ParamClass>,
}

impl ParamGroupOptimizer {
    /// `kind` optimizers over `groups`, at base learning rate `base_lr`.
    pub fn new(kind: OptimizerType, base_lr: f32, groups: ParameterGroupConfig) -> Self {
        let make = |class| {
            TrainOptimizer::for_class(
                kind,
                class,
                groups.learning_rate(class, base_lr),
                groups.weight_decay(class),
            )
        };
        let lora_b = groups.rates.loraplus_ratio.map(|ratio| {
            tracing::info!(
                "LoRA+ enabled: A matrices lr={:.2e}, B matrices lr={:.2e} (ratio={ratio})",
                base_lr,
                groups.learning_rate(ParamClass::LoraB, base_lr),
            );
            make(ParamClass::LoraB)
        });
        Self {
            kind,
            regular: make(ParamClass::Regular),
            lora_b,
            embedding: make(ParamClass::Embedding),
            no_decay: make(ParamClass::NoDecay),
            groups,
            param_cache: HashMap::new(),
        }
    }

    /// Which kind of optimizer the groups run (Muon's embedding and no-decay
    /// groups run AdamW, as Muon is for the hidden matrices only).
    pub fn kind(&self) -> OptimizerType {
        self.kind
    }

    /// Whether training and evaluating use different weights
    /// ([`eval_weights`](Self::eval_weights)): schedule-free AdamW's.
    pub fn has_eval_weights(&self) -> bool {
        self.kind == OptimizerType::AdamWScheduleFree
    }

    /// Swap `model` to the weights to evaluate and save; see
    /// [`TrainOptimizer::eval_weights`]. Pair with
    /// [`train_weights`](Self::train_weights).
    pub fn eval_weights<M: ModuleParameters>(&mut self, model: &mut M) {
        for group in self.optimizers_mut() {
            group.eval_weights(model);
        }
    }

    /// Swap `model` back to the weights to train on.
    pub fn train_weights<M: ModuleParameters>(&mut self, model: &mut M) {
        for group in self.optimizers_mut() {
            group.train_weights(model);
        }
    }

    /// The groups the parameters train in.
    pub fn parameter_groups(&self) -> &ParameterGroupConfig {
        &self.groups
    }

    fn classify_param(&mut self, name: &Rc<str>) -> ParamClass {
        if let Some(&class) = self.param_cache.get(name) {
            return class;
        }
        let class = self.groups.classify(name);
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

    fn optimizers(&self) -> impl Iterator<Item = &TrainOptimizer> {
        [&self.regular, &self.embedding, &self.no_decay]
            .into_iter()
            .chain(self.lora_b.as_ref())
    }

    fn optimizers_mut(&mut self) -> impl Iterator<Item = &mut TrainOptimizer> {
        [&mut self.regular, &mut self.embedding, &mut self.no_decay]
            .into_iter()
            .chain(self.lora_b.as_mut())
    }

    /// `(base_lr, embedding_lr)` for the next update.
    pub fn learning_rates(&self) -> (f32, f32) {
        (self.regular.lr(), self.embedding.lr())
    }

    /// Set the base learning rate. Every group runs at its multiple of it.
    pub fn set_learning_rate(&mut self, base_lr: f32) {
        let rates = self.groups.rates;
        let rate = |class| rates.learning_rate(class, base_lr);
        self.regular.set_lr(rate(ParamClass::Regular));
        self.no_decay.set_lr(rate(ParamClass::NoDecay));
        self.embedding.set_lr(rate(ParamClass::Embedding));
        if let Some(b) = self.lora_b.as_mut() {
            b.set_lr(rate(ParamClass::LoraB));
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
        for group in self.optimizers_mut() {
            group.advance_step();
        }
        apply_updates(self, model, gradients)
    }
}

/// The names a group's state is nested under in a [`ParamGroupOptimizer`]'s.
const GROUP_NAMES: [(&str, ParamClass); 4] = [
    ("regular", ParamClass::Regular),
    ("embedding", ParamClass::Embedding),
    ("no_decay", ParamClass::NoDecay),
    ("lora_b", ParamClass::LoraB),
];

impl ParamGroupOptimizer {
    fn group(&self, class: ParamClass) -> Option<&TrainOptimizer> {
        match class {
            ParamClass::Regular => Some(&self.regular),
            ParamClass::Embedding => Some(&self.embedding),
            ParamClass::NoDecay => Some(&self.no_decay),
            ParamClass::LoraB => self.lora_b.as_ref(),
        }
    }
}

impl StatefulOptimizer for ParamGroupOptimizer {
    fn export_state(&self) -> OptimizerState {
        let mut out = OptimizerState::new("param_groups");
        for (name, class) in GROUP_NAMES {
            if let Some(group) = self.group(class) {
                group.export_state().nest(name, &mut out);
            }
        }
        out
    }

    fn import_state(&mut self, state: &OptimizerState) -> Result<()> {
        state.expect_kind("param_groups")?;
        for (name, class) in GROUP_NAMES {
            let Some(group) = (match class {
                ParamClass::Regular => Some(&mut self.regular),
                ParamClass::Embedding => Some(&mut self.embedding),
                ParamClass::NoDecay => Some(&mut self.no_decay),
                ParamClass::LoraB => self.lora_b.as_mut(),
            }) else {
                continue;
            };
            let kind = group.export_state().kind;
            group.import_state(&state.unnest(name, &kind))?;
        }
        Ok(())
    }
}

impl Updatable for ParamGroupOptimizer {
    fn updatable_states_len(&self) -> usize {
        self.optimizers().map(|g| g.updatable_states_len()).sum()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        self.optimizers()
            .flat_map(|g| g.updatable_states())
            .collect()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        self.optimizers_mut()
            .flat_map(|g| g.updatable_states_mut())
            .collect()
    }
}

/// Builder for [`ParamGroupOptimizer`].
#[derive(Debug, Clone)]
pub struct ParamGroupOptimizerBuilder {
    kind: OptimizerType,
    base_lr: f32,
    groups: ParameterGroupConfig,
}

impl ParamGroupOptimizerBuilder {
    /// Groups of `kind` optimizers at base learning rate `base_lr`, with
    /// weight decay 0.01 until [`Self::with_weight_decay`] says otherwise.
    pub fn new(kind: OptimizerType, base_lr: f32) -> Self {
        Self {
            kind,
            base_lr,
            groups: ParameterGroupConfig::new(0.01),
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
        self.groups = self.groups.with_embedding_lr(lr, self.base_lr);
        self
    }

    /// Weight decay for every group but the no-decay one.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.groups.rates.weight_decay = wd;
        self
    }

    /// Count parameters under a module named `module` as embedding
    /// parameters.
    pub fn add_embedding_module(mut self, module: impl Into<String>) -> Self {
        self.groups = self.groups.add_embedding_module(module);
        self
    }

    /// Train LoRA B matrices at `base_lr * ratio` (LoRA+; the paper suggests 16).
    pub fn with_loraplus_lr_ratio(mut self, ratio: f32) -> Self {
        self.groups = self.groups.with_loraplus_lr_ratio(ratio);
        self
    }

    /// Build the groups.
    pub fn build(self) -> ParamGroupOptimizer {
        ParamGroupOptimizer::new(self.kind, self.base_lr, self.groups)
    }
}

#[cfg(test)]
mod tests;
