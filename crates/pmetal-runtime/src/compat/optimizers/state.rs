//! Optimizer state as data: what a checkpoint holds so that a resumed run
//! takes the steps the uninterrupted one would have.
//!
//! An [`OptimizerState`] is a set of counters and named tensors. The tensors
//! are named `<slot>/<parameter>` (`m/model.layers.0.q_proj.lora_a`), one slot
//! per kind of buffer an optimizer keeps for a parameter: the moments, the
//! f32 master copy of a half-precision weight, the schedule-free iterate. An
//! optimizer that wraps others (parameter groups, Muon with AdamW) prefixes
//! each part's names with [`OptimizerState::nest`] and splits them again with
//! [`OptimizerState::unnest`].
//!
//! The learning rate is not state: the scheduler sets it before every step,
//! from the step count the checkpoint's metadata holds.

use crate::compat::{Array, Exception};
use crate::optimizer::MasterWeight;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

/// An optimizer's counters and buffers, by name.
#[derive(Debug, Default, Clone)]
pub struct OptimizerState {
    /// Which update rule wrote the state (`"adamw"`, `"muon"`, ...). Loading a
    /// state into an optimizer of another kind is an error, not a guess.
    pub kind: String,
    /// Step counters and other scalars. `f64` holds every `u64` a run can
    /// reach exactly.
    pub scalars: BTreeMap<String, f64>,
    /// The buffers, named `<slot>/<parameter>`.
    pub tensors: BTreeMap<String, Array>,
}

impl OptimizerState {
    /// An empty state of `kind`.
    pub fn new(kind: &str) -> Self {
        Self {
            kind: kind.to_string(),
            ..Self::default()
        }
    }

    /// Record the scalar `name`.
    pub fn set_scalar(&mut self, name: &str, value: f64) {
        self.scalars.insert(name.to_string(), value);
    }

    /// The scalar `name`.
    ///
    /// # Errors
    /// If the state has none by that name.
    pub fn scalar(&self, name: &str) -> Result<f64, Exception> {
        self.scalars.get(name).copied().ok_or_else(|| {
            Exception::custom(format!(
                "the {} optimizer state has no `{name}` counter",
                self.kind
            ))
        })
    }

    /// Record `array` as `slot` of `parameter`.
    pub fn set_tensor(&mut self, slot: &str, parameter: &str, array: &Array) {
        self.tensors
            .insert(format!("{slot}/{parameter}"), array.clone());
    }

    /// Every parameter that has a `slot` buffer, with it.
    pub fn slot<'a>(&'a self, slot: &'a str) -> impl Iterator<Item = (&'a str, &'a Array)> + 'a {
        let prefix = format!("{slot}/");
        self.tensors.iter().filter_map(move |(name, array)| {
            name.strip_prefix(prefix.as_str())
                .map(|param| (param, array))
        })
    }

    /// `parameter`'s `slot` buffer, if it has one.
    pub fn tensor(&self, slot: &str, parameter: &str) -> Option<&Array> {
        self.tensors.get(&format!("{slot}/{parameter}"))
    }

    /// This state's names behind `prefix`, for embedding it in a larger one:
    /// scalars become `<prefix>.<name>` and tensors `<prefix>:<slot>/<param>`.
    pub fn nest(&self, prefix: &str, into: &mut OptimizerState) {
        for (name, value) in &self.scalars {
            into.scalars.insert(format!("{prefix}.{name}"), *value);
        }
        for (name, array) in &self.tensors {
            into.tensors
                .insert(format!("{prefix}:{name}"), array.clone());
        }
    }

    /// The part of this state [`nest`](Self::nest)ed under `prefix`, as a
    /// state of `kind`.
    pub fn unnest(&self, prefix: &str, kind: &str) -> OptimizerState {
        let scalar_prefix = format!("{prefix}.");
        let tensor_prefix = format!("{prefix}:");
        let mut part = OptimizerState::new(kind);
        for (name, value) in &self.scalars {
            if let Some(rest) = name.strip_prefix(&scalar_prefix) {
                part.scalars.insert(rest.to_string(), *value);
            }
        }
        for (name, array) in &self.tensors {
            if let Some(rest) = name.strip_prefix(&tensor_prefix) {
                part.tensors.insert(rest.to_string(), array.clone());
            }
        }
        part
    }

    /// Fail unless the state was written by an optimizer of `kind`.
    ///
    /// # Errors
    /// If it was written by another.
    pub fn expect_kind(&self, kind: &str) -> Result<(), Exception> {
        if self.kind == kind {
            Ok(())
        } else {
            Err(Exception::custom(format!(
                "optimizer state of a `{}` optimizer cannot be loaded into a `{kind}` one",
                self.kind
            )))
        }
    }

    /// Every array in the state, for evaluating it before it is written.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.tensors.values()
    }
}

/// An optimizer whose state a checkpoint can hold.
pub trait StatefulOptimizer {
    /// The optimizer's counters and buffers.
    fn export_state(&self) -> OptimizerState;

    /// Continue from `state`, replacing whatever the optimizer holds. The
    /// parameters the buffers belong to must already hold the checkpoint's
    /// weights.
    ///
    /// # Errors
    /// If `state` was written by another kind of optimizer or lacks a counter.
    fn import_state(&mut self, state: &OptimizerState) -> Result<(), Exception>;
}

/// Record every master copy of `masters` in `into` under the slot `master`.
pub(super) fn export_masters(masters: &HashMap<Rc<str>, MasterWeight>, into: &mut OptimizerState) {
    for (key, master) in masters {
        if let Some(value) = master.value() {
            into.set_tensor("master", key, value);
        }
    }
}

/// The master copies `state` holds, as [`MasterWeight::restore`]d.
pub(super) fn import_masters(state: &OptimizerState) -> HashMap<Rc<str>, MasterWeight> {
    state
        .slot("master")
        .map(|(key, value)| (Rc::from(key), MasterWeight::restore(value.clone())))
        .collect()
}

/// The buffers of `slot` in `state`, by parameter, as an optimizer's own map.
pub(super) fn import_slot(state: &OptimizerState, slot: &str) -> HashMap<Rc<str>, Array> {
    state
        .slot(slot)
        .map(|(key, value)| (Rc::from(key), value.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{
        Adafactor, AdamW, Lion, MomentumSgd, Muon, MuonAdamW, Optimizer, Sgd, StatefulOptimizer,
    };
    use super::*;
    use crate::compat::{Dtype, random};

    fn read(a: &Array) -> Vec<f32> {
        let mut a = a.as_dtype(Dtype::Float32.as_i32());
        a.eval();
        let n = a.size();
        let v = a.to_f32_vec(n).expect("readable");
        crate::check_unobserved_error().expect("no bridge error");
        v
    }

    /// Train a bf16 matrix for six steps straight, and for three, a state
    /// export, a fresh optimizer fed it and a reloaded weight, and three more:
    /// the two end in the same bits. Updates are far below bf16's resolution,
    /// so the run is only right if the f32 master copy comes through too.
    fn resumes_exactly<O: Optimizer + StatefulOptimizer>(
        what: &str,
        make: impl Fn() -> O,
        advance: impl Fn(&mut O),
    ) {
        random::seed(95);
        let bf16 = Dtype::Bfloat16.as_i32();
        let key: Rc<str> = Rc::from("model.layers.0.mlp.up_proj.weight");
        let p0 = random::uniform_range(-1.0, 1.0, &[8, 16], Dtype::Float32).as_dtype(bf16);
        let grads: Vec<Array> = (0..6)
            .map(|_| random::normal(&[8, 16], Dtype::Float32))
            .collect();
        let run = |opt: &mut O, p: &mut Array, range: std::ops::Range<usize>| {
            for i in range {
                advance(opt);
                opt.update_single(&key, &grads[i], p).expect("update");
                p.eval();
            }
        };
        let mut whole = make();
        let mut pw = p0.clone();
        run(&mut whole, &mut pw, 0..6);

        let mut first = make();
        let mut p = p0.clone();
        run(&mut first, &mut p, 0..3);
        let state = first.export_state();
        let mut second = make();
        second.import_state(&state).expect("import");
        // The weight a resumed run loads: a new array with the saved values.
        let mut p2 = Array::from_f32_slice(&read(&p), &[8, 16]).as_dtype(bf16);
        run(&mut second, &mut p2, 3..6);
        assert_eq!(read(&p2), read(&pw), "{what}");
        assert_ne!(read(&pw), read(&p0), "{what} moved the weight");
    }

    #[test]
    fn every_optimizer_resumes_bit_exactly() {
        let lr = 1e-3;
        resumes_exactly("adamw", || AdamW::new(lr, 0.01), |o| o.advance_step());
        resumes_exactly("sgd", || Sgd::new(lr * 100.0), |_| {});
        resumes_exactly("momentum sgd", || MomentumSgd::new(lr, 0.01), |_| {});
        resumes_exactly("lion", || Lion::new(lr, 0.01), |_| {});
        resumes_exactly("adafactor", || Adafactor::new(lr, 0.01), |o| o.advance_step());
        resumes_exactly("muon", || Muon::new(lr, 0.01), |_| {});
        resumes_exactly(
            "muon + adamw",
            || MuonAdamW::new(lr, lr, 0.01),
            |o| o.advance_step(),
        );
    }

    /// The test above has teeth: a resume that drops the f32 master copies
    /// ends somewhere else, because the sub-resolution updates the master
    /// had accumulated are lost.
    #[test]
    fn a_resume_without_the_master_copy_diverges() {
        random::seed(96);
        let bf16 = Dtype::Bfloat16.as_i32();
        let key: Rc<str> = Rc::from("w");
        let p0 = random::uniform_range(-1.0, 1.0, &[8, 16], Dtype::Float32).as_dtype(bf16);
        let grads: Vec<Array> = (0..6)
            .map(|_| random::normal(&[8, 16], Dtype::Float32).mul_scalar(0.1))
            .collect();
        let run = |opt: &mut AdamW, p: &mut Array, range: std::ops::Range<usize>| {
            for i in range {
                opt.advance_step();
                opt.update_single(&key, &grads[i], p).expect("update");
                p.eval();
            }
        };
        let mut whole = AdamW::new(1e-3, 0.0);
        let mut pw = p0.clone();
        run(&mut whole, &mut pw, 0..6);
        let mut first = AdamW::new(1e-3, 0.0);
        let mut p = p0.clone();
        run(&mut first, &mut p, 0..3);
        let mut state = first.export_state();
        state.tensors.retain(|name, _| !name.starts_with("master/"));
        let mut second = AdamW::new(1e-3, 0.0);
        second.import_state(&state).expect("import");
        let mut p2 = Array::from_f32_slice(&read(&p), &[8, 16]).as_dtype(bf16);
        run(&mut second, &mut p2, 3..6);
        assert_ne!(read(&p2), read(&pw));
    }

    #[test]
    fn a_state_loads_only_into_its_own_kind() {
        let state = Lion::new(1e-3, 0.0).export_state();
        let err = AdamW::new(1e-3, 0.0)
            .import_state(&state)
            .expect_err("a Lion state is not AdamW's");
        assert!(err.to_string().contains("lion"), "{err}");
    }

    #[test]
    fn nested_states_do_not_collide() {
        let mut inner = OptimizerState::new("adamw");
        inner.set_scalar("step", 7.0);
        inner.set_tensor("m", "w", &Array::from_f32(1.0));
        let mut outer = OptimizerState::new("group");
        inner.nest("regular", &mut outer);
        inner.nest("embedding", &mut outer);
        let back = outer.unnest("embedding", "adamw");
        assert_eq!(back.scalar("step").unwrap(), 7.0);
        assert!(back.tensor("m", "w").is_some());
        assert_eq!(outer.tensors.len(), 2);
    }
}
