use super::{Array, Exception, FlattenedModuleParam, ModuleParameters, ModuleParametersExt};
use std::collections::HashMap;
use std::rc::Rc;

/// Optimizer state: stores (momentum, velocity) tensors per parameter.
pub type State<V> = HashMap<Rc<str>, V>;

/// Common optimizer interface — matches `mlx_rs::optimizers::Optimizer`.
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
        // Flatten the nested parameter tree so that dotted keys from
        // value_and_grad (e.g. "layers.0.self_attn.q_proj.lora_a") can
        // be matched directly.  The old code looked up flat keys against
        // the nested root map, where they could never be found.
        let mut flat = model.flatten_params_mut();
        for (key, grad) in &gradients {
            if let Some(arr) = flat.get_mut(key.as_ref()) {
                let _ = self.update_single(key, grad, arr);
            }
        }
        Ok(())
    }
}

/// Updatable: exposes state arrays for eval/checkpointing.
pub trait Updatable {
    fn updatable_states_len(&self) -> usize;
    fn updatable_states(&self) -> Vec<&Array>;
    fn updatable_states_mut(&mut self) -> Vec<&mut Array>;
}

/// AdamW optimizer compatible with mlx_rs::optimizers::AdamW interface.
pub struct AdamW {
    inner: crate::optimizer::AdamW,
    pub lr: Array,
    pub state: State<(Array, Array)>,
}

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW(compat)").finish()
    }
}

impl AdamW {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            inner: crate::optimizer::AdamW::new(lr, weight_decay),
            lr: Array::from_f32(lr),
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

    /// Update the base learning rate (for LR scheduling).
    pub fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
        self.lr = Array::from_f32(lr);
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
                crate::optimizer::AdamState {
                    m: m.clone(),
                    v: v.clone(),
                },
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
        let mut flat = model.flatten_params_mut();
        for (key, grad) in &gradients {
            if let Some(arr) = flat.get_mut(key.as_ref()) {
                let _ = self.update_single(key, grad, arr);
            }
        }
        Ok(())
    }
}

impl Updatable for AdamW {
    fn updatable_states_len(&self) -> usize {
        self.state.len() * 2
    }
    fn updatable_states(&self) -> Vec<&Array> {
        self.state.values().flat_map(|(m, v)| [m, v]).collect()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        self.state.values_mut().flat_map(|(m, v)| [m, v]).collect()
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
        Ok(AdamW::new(self.lr, self.weight_decay))
    }
}

/// SGD optimizer — vanilla stochastic gradient descent with no momentum.
pub struct Sgd {
    pub lr: Array,
    pub state: State<()>,
}

impl std::fmt::Debug for Sgd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sgd").finish()
    }
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            lr: Array::from_f32(lr),
            state: HashMap::new(),
        }
    }
}

impl Optimizer for Sgd {
    type State = State<()>;
    fn state(&self) -> &Self::State {
        &self.state
    }
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
    fn update_single(
        &mut self,
        _key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        let lr_val = self.lr.clone().item_f32();
        let lr_arr = Array::from_f32(lr_val);
        *parameter = parameter.subtract(&gradient.multiply(&lr_arr));
        Ok(())
    }
}

impl Updatable for Sgd {
    fn updatable_states_len(&self) -> usize {
        0
    }
    fn updatable_states(&self) -> Vec<&Array> {
        vec![]
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        vec![]
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
