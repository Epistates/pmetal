//! The optimizer trait and the optimizers over compat arrays: [`AdamW`],
//! [`Sgd`], [`MomentumSgd`], [`Lion`], [`Adafactor`] and [`Muon`].

use super::{Array, Exception, FlattenedModuleParam, ModuleParameters, ModuleParametersExt};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

mod adamw_8bit;
mod first_order;
mod muon;
mod schedule_free;
mod state;

pub use adamw_8bit::{
    AdamW8bit, BLOCK as ADAMW_8BIT_BLOCK, MIN_8BIT_SIZE, dynamic_map, nearest_code,
};
pub use first_order::{Adafactor, Lion, MomentumSgd, SecondMoment, Sgd};
pub use muon::{LrAdjust, Muon, MuonAdamW, NS_COEFFICIENTS, newton_schulz_orthogonalize};
pub use schedule_free::ScheduleFreeAdamW;
pub use state::{OptimizerState, StatefulOptimizer};

/// Optimizer state: stores (momentum, velocity) tensors per parameter.
pub type State<V> = HashMap<Rc<str>, V>;

/// Common optimizer interface.
pub trait Optimizer {
    type State;
    fn state(&self) -> &Self::State;
    fn state_mut(&mut self) -> &mut Self::State;
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception>;
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<(), Exception> {
        apply_updates(self, model, gradients)
    }
}

/// Run [`Optimizer::update_single`] over every parameter that has a
/// gradient, stopping at the first update that fails.
///
/// A bridge op that throws hands back a scalar-zero placeholder instead of
/// unwinding, so a gradient that doesn't fit its parameter used to leave the
/// parameter where it was while the step still reported a loss. The bridge's
/// first unobserved error is read before the updates (a failure in the
/// forward or backward pass is reported as that) and after each parameter,
/// so an update failure names the parameter it broke.
pub fn apply_updates<O: Optimizer + ?Sized, M: ModuleParameters>(
    optimizer: &mut O,
    model: &mut M,
    gradients: FlattenedModuleParam,
) -> Result<(), Exception> {
    crate::error::check_unobserved_error().map_err(|e| {
        Exception::custom(format!(
            "a bridge op failed before the optimizer update: {e}"
        ))
    })?;
    // Gradient keys are dotted paths ("layers.0.self_attn.q_proj.lora_a"),
    // so they're matched against the flattened parameter tree.
    let mut flat = model.flatten_params_mut();
    for (key, grad) in &gradients {
        if let Some(arr) = flat.get_mut(key.as_ref()) {
            optimizer.update_single(key, grad, arr)?;
            crate::error::check_unobserved_error()
                .map_err(|e| Exception::custom(format!("optimizer update of `{key}`: {e}")))?;
        }
    }
    Ok(())
}

/// Updatable: exposes state arrays for eval/checkpointing.
pub trait Updatable {
    fn updatable_states_len(&self) -> usize;
    fn updatable_states(&self) -> Vec<&Array>;
    fn updatable_states_mut(&mut self) -> Vec<&mut Array>;
}

/// AdamW optimizer, updating as `torch.optim.AdamW` does (see
/// [`crate::optimizer::AdamW::step_single`]).
///
/// Every parameter trains at the one learning rate [`AdamW::set_lr`] sets;
/// biases and normalization weights skip weight decay
/// ([`crate::optimizer::excluded_from_weight_decay`]). Separate embedding or
/// LoRA+ rates are parameter groups, which the trainer builds from several
/// of these.
pub struct AdamW {
    inner: crate::optimizer::AdamW,
    pub state: State<(Array, Array)>,
}

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW(compat)").finish()
    }
}

impl AdamW {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self::from_inner(crate::optimizer::AdamW::new(lr, weight_decay))
    }

    fn from_inner(inner: crate::optimizer::AdamW) -> Self {
        Self {
            inner: inner.with_classifier(crate::optimizer::decay_only_classifier),
            state: HashMap::new(),
        }
    }

    /// Advance the inner optimizer's step counter by one.
    ///
    /// Must be called once per training step before calling `update_single`
    /// in a loop. The [`Optimizer::update`] override does this automatically.
    pub fn advance_step(&mut self) {
        self.inner.advance_step();
    }

    /// Current optimizer step count.
    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    /// Set the learning rate the next update uses (for LR scheduling).
    pub fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    /// The learning rate the next update uses.
    pub fn lr(&self) -> f32 {
        self.inner.lr()
    }

    /// The f32 master copies of the half-precision parameters that have taken
    /// a step, by parameter name. A checkpoint that resumes such a run exactly
    /// needs them: the parameters hold only the rounded weights.
    pub fn master_weights(&self) -> impl Iterator<Item = (&str, &Array)> {
        self.inner
            .states
            .iter()
            .filter_map(|(key, state)| state.master.value().map(|m| (key.as_str(), m)))
    }

    /// Restore optimizer state from a checkpoint.
    ///
    /// Sets the step counter and populates both the public `state` map and
    /// the inner per-parameter `AdamState` entries so that the next
    /// `update()` call picks up where training left off.
    pub fn restore_state(&mut self, step_count: u64, state: State<(Array, Array)>) {
        self.inner.set_step_count(step_count);
        for (key, (m, v)) in &state {
            self.inner.states.insert(
                key.to_string(),
                crate::optimizer::AdamState::new(m.clone(), v.clone()),
            );
        }
        self.state = state;
    }
}

impl Optimizer for AdamW {
    type State = State<(Array, Array)>;
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
    ) -> Result<(), Exception> {
        self.inner.step_single(key.as_ref(), gradient, parameter);
        // Sync the inner optimizer's moment state into the public state
        // map so that Updatable, checkpointing, and test assertions see it.
        if let Some(inner_state) = self.inner.states.get(key.as_ref()) {
            self.state
                .insert(key.clone(), (inner_state.m.clone(), inner_state.v.clone()));
        }
        Ok(())
    }
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<(), Exception> {
        // Advance the step counter ONCE per training step, not per parameter.
        self.inner.advance_step();
        apply_updates(self, model, gradients)
    }
}

impl StatefulOptimizer for AdamW {
    fn export_state(&self) -> OptimizerState {
        let mut out = OptimizerState::new("adamw");
        out.set_scalar("step", self.inner.step_count() as f64);
        for (key, s) in &self.inner.states {
            out.set_tensor("m", key, &s.m);
            out.set_tensor("v", key, &s.v);
            if let Some(master) = s.master.value() {
                out.set_tensor("master", key, master);
            }
        }
        out
    }

    fn import_state(&mut self, state: &OptimizerState) -> Result<(), Exception> {
        state.expect_kind("adamw")?;
        let step = state.scalar("step")? as u64;
        let mut restored = BTreeMap::new();
        for (key, m) in state.slot("m") {
            let v = state.tensor("v", key).ok_or_else(|| {
                Exception::custom(format!(
                    "optimizer state has a first moment for `{key}` but no second"
                ))
            })?;
            restored.insert(
                key.to_string(),
                crate::optimizer::AdamState::restored(
                    m.clone(),
                    v.clone(),
                    state.tensor("master", key).cloned(),
                ),
            );
        }
        self.inner.set_step_count(step);
        self.state = restored
            .iter()
            .map(|(k, s)| (Rc::from(k.as_str()), (s.m.clone(), s.v.clone())))
            .collect();
        self.inner.states = restored;
        Ok(())
    }
}

impl Updatable for AdamW {
    fn updatable_states_len(&self) -> usize {
        self.updatable_states().len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        // The moments in `state` are the inner optimizer's own; half-precision
        // parameters' f32 master copies live only there.
        self.state
            .values()
            .flat_map(|(m, v)| [m, v])
            .chain(self.inner.states.values().filter_map(|s| s.master.value()))
            .collect()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        self.state
            .values_mut()
            .flat_map(|(m, v)| [m, v])
            .chain(
                self.inner
                    .states
                    .values_mut()
                    .filter_map(|s| s.master.value_mut()),
            )
            .collect()
    }
}

/// Builder for AdamW.
#[derive(Debug, Clone)]
pub struct AdamWBuilder {
    lr: f32,
    weight_decay: f32,
    betas: (f32, f32),
    eps: f32,
}

impl AdamWBuilder {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            weight_decay: 0.01,
            betas: (0.9, 0.999),
            eps: 1e-8,
        }
    }
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
    pub fn betas(mut self, b: (f32, f32)) -> Self {
        self.betas = b;
        self
    }
    pub fn eps(mut self, e: f32) -> Self {
        self.eps = e;
        self
    }
    pub fn build(self) -> Result<AdamW, Exception> {
        Ok(AdamW::from_inner(
            crate::optimizer::AdamW::new(self.lr, self.weight_decay)
                .with_betas(self.betas.0, self.betas.1)
                .with_eps(self.eps),
        ))
    }
}

impl<M, O: Updatable> Updatable for (M, O) {
    fn updatable_states_len(&self) -> usize {
        self.1.updatable_states_len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        self.1.updatable_states()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        self.1.updatable_states_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f32 master copies come out in parameter-name order, the same on
    /// every run: a checkpoint writes them in that order.
    #[test]
    fn master_weights_are_in_name_order() {
        let bf16 = super::super::Dtype::Bfloat16.as_i32();
        // Inserted out of order, and enough of them that a hash map would not
        // happen to walk them sorted.
        let names: Vec<Rc<str>> = (0..40)
            .rev()
            .map(|i| Rc::from(format!("layers.{i}.w")))
            .collect();
        let mut opt = AdamW::new(1e-3, 0.0);
        opt.advance_step();
        for name in &names {
            let mut param = Array::from_f32_slice(&[1.0, 2.0], &[2]).as_dtype(bf16);
            let grad = Array::from_f32_slice(&[0.5, 0.5], &[2]).as_dtype(bf16);
            opt.update_single(name, &grad, &mut param).unwrap();
        }
        let mut sorted: Vec<&str> = names.iter().map(|n| &**n).collect();
        sorted.sort_unstable();
        let masters: Vec<&str> = opt.master_weights().map(|(name, _)| name).collect();
        assert_eq!(masters, sorted);
    }
}
