//! Schedule-free AdamW: iterate averaging in place of a decaying learning-rate
//! schedule (Defazio et al., *The Road Less Scheduled*, arXiv 2405.15682).
//!
//! The optimizer keeps three sequences per parameter. `z` takes Adam-normalized
//! gradient steps at a constant rate; `x` is the running average of the `z`'s
//! (the weights to evaluate and ship); the gradient is taken at
//! `y = (1 − β₁)·z + β₁·x`, the weights the model holds while it trains. With
//! `lr` the (warmed-up) rate of step `t`, `lr_max` the largest rate seen and
//! `c = w_t / Σw` the averaging weight (`w_t = t^r · lr_max²`, `r = 0`):
//!
//! ```text
//! v = β₂·v + (1−β₂)·g²
//! ĝ = g / (√(v/(1−β₂ᵗ)) + ε) + λ·y            decay is taken at y
//! y = y + c·(z − y) + lr·(β₁·(1−c) − 1)·ĝ     the new y, x never formed
//! z = z − lr·ĝ
//! ```
//!
//! This is `schedulefree.AdamWScheduleFree`'s update, step for step. Because
//! `x` is never formed during training, evaluating or saving needs the swap
//! [`ScheduleFreeAdamW::eval_weights`] does (`x = (y − (1−β₁)·z)/β₁`) and
//! [`ScheduleFreeAdamW::train_weights`] undoes. The swap loses nothing: the
//! optimizer keeps its `y` exactly, in f32, and puts it back as it was.

use super::{
    Optimizer, OptimizerState, State, StatefulOptimizer, Updatable, apply_updates,
    first_order::f32_of,
};
use crate::compat::{
    Array, Exception, FlattenedModuleParam, ModuleParameters, ModuleParametersExt,
};
use std::collections::HashMap;
use std::rc::Rc;

/// What the optimizer keeps for one parameter.
#[derive(Debug)]
struct SfState {
    /// The slow iterate `z`, f32.
    z: Array,
    /// Second moment, f32.
    v: Array,
    /// The training weights `y`, f32. A half-precision parameter is this
    /// rounded; the f32 is what the next step continues from.
    y: Array,
    /// The parameter array the last step (or swap) left in the model, to tell
    /// when something else has replaced it (a rollback) and `y` is stale.
    produced: Option<Array>,
}

/// The scalars of one step, fixed by [`ScheduleFreeAdamW::advance_step`].
#[derive(Debug, Clone, Copy, Default)]
struct StepScalars {
    lr: f32,
    /// `c`, the averaging weight of the new iterate.
    ckp1: f32,
    bias_correction2: f32,
}

/// Schedule-free AdamW. Train at a constant learning rate (warmed up, if you
/// like) and evaluate [`eval_weights`](Self::eval_weights).
#[derive(Debug)]
pub struct ScheduleFreeAdamW {
    /// Learning rate: the scheduler's, which for this optimizer is constant
    /// after warmup.
    pub lr: f32,
    /// Weight decay, taken at `y` and scaled by the learning rate.
    pub weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    /// Exponent of the step in the averaging weight; 0 averages the iterates
    /// by learning rate alone.
    r: f64,
    /// Exponent of the largest learning rate in the averaging weight.
    weight_lr_power: f64,
    /// Steps taken.
    k: u64,
    lr_max: f64,
    weight_sum: f64,
    scalars: StepScalars,
    train_mode: bool,
    /// Per-parameter buffers.
    state: State<SfState>,
}

impl ScheduleFreeAdamW {
    /// Schedule-free AdamW with the reference's defaults: β = (0.9, 0.999),
    /// ε = 1e-8, `r = 0`, `weight_lr_power = 2`.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            r: 0.0,
            weight_lr_power: 2.0,
            k: 0,
            lr_max: 0.0,
            weight_sum: 0.0,
            scalars: StepScalars::default(),
            train_mode: true,
            state: HashMap::new(),
        }
    }

    /// Override the betas.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Override the denominator epsilon.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Weight the iterates by `step^r` as well as by learning rate (`r = 0`
    /// is the reference's default).
    pub fn with_r(mut self, r: f32) -> Self {
        self.r = f64::from(r);
        self
    }

    /// Steps taken.
    pub fn step_count(&self) -> u64 {
        self.k
    }

    /// Whether the model holds the training weights `y` (as opposed to the
    /// averaged `x`).
    pub fn is_train_mode(&self) -> bool {
        self.train_mode
    }

    /// Number of parameters the optimizer holds state for.
    pub fn num_params(&self) -> usize {
        self.state.len()
    }

    /// Fix the scalars of the next step from the current learning rate. Once
    /// per optimizer step, before the per-parameter calls;
    /// [`Optimizer::update`] does it itself.
    pub fn advance_step(&mut self) {
        let k = self.k as f64;
        let lr = f64::from(self.lr);
        self.lr_max = self.lr_max.max(lr);
        let weight = (k + 1.0).powf(self.r) * self.lr_max.powf(self.weight_lr_power);
        self.weight_sum += weight;
        let ckp1 = if self.weight_sum == 0.0 {
            0.0
        } else {
            weight / self.weight_sum
        };
        self.scalars = StepScalars {
            lr: self.lr,
            ckp1: ckp1 as f32,
            bias_correction2: (1.0 - f64::from(self.beta2).powf(k + 1.0)) as f32,
        };
        self.k += 1;
    }

    /// `y` for `key`: the optimizer's own when `parameter` is still the array
    /// it left, else the parameter (replaced by a rollback or a load).
    fn current_y(state: &SfState, parameter: &Array) -> Array {
        match &state.produced {
            Some(produced) if produced.id() == parameter.id() => state.y.clone(),
            _ => f32_of(parameter),
        }
    }

    /// Put `y` into `parameter`, rounded to its dtype, and remember it.
    fn commit(state: &mut SfState, parameter: &mut Array, y: Array) {
        *parameter = y.as_dtype(parameter.dtype_raw());
        state.y = y;
        state.produced = Some(parameter.clone());
    }

    /// Swap every parameter `model` holds to the averaged weights `x`, which
    /// is what to evaluate and to save. A no-op when already there.
    pub fn eval_weights<M: ModuleParameters>(&mut self, model: &mut M) {
        if !self.train_mode {
            return;
        }
        let c = 1.0 - 1.0 / self.beta1;
        let mut flat = model.flatten_params_mut();
        for (key, parameter) in flat.iter_mut() {
            let Some(state) = self.state.get_mut::<str>(key.as_str()) else {
                continue;
            };
            let y = Self::current_y(state, parameter);
            state.y = y.clone();
            // x = y + (1 − 1/β₁)·(z − y)
            let x = y.add(&state.z.subtract(&y).mul_scalar(c));
            **parameter = x.as_dtype(parameter.dtype_raw());
            state.produced = None;
        }
        self.train_mode = false;
    }

    /// Swap back to the training weights `y`, exactly as they were. A no-op
    /// when already there.
    pub fn train_weights<M: ModuleParameters>(&mut self, model: &mut M) {
        if self.train_mode {
            return;
        }
        let mut flat = model.flatten_params_mut();
        for (key, parameter) in flat.iter_mut() {
            let Some(state) = self.state.get_mut::<str>(key.as_str()) else {
                continue;
            };
            let y = state.y.clone();
            Self::commit(state, parameter, y);
        }
        self.train_mode = true;
    }
}

impl Optimizer for ScheduleFreeAdamW {
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
    ) -> Result<(), Exception> {
        if !self.train_mode {
            return Err(Exception::custom(
                "the schedule-free optimizer was asked to step while the model holds its \
                 evaluation weights; call train_weights first",
            ));
        }
        let StepScalars {
            lr,
            ckp1,
            bias_correction2,
        } = self.scalars;
        let g = f32_of(gradient);
        let state = self.state.entry(key.clone()).or_insert_with(|| {
            let y = f32_of(parameter);
            SfState {
                z: y.clone(),
                v: y.zeros_like(),
                y,
                produced: None,
            }
        });
        let y = Self::current_y(state, parameter);

        state.v = state
            .v
            .mul_scalar(self.beta2)
            .add(&g.square().mul_scalar(1.0 - self.beta2));
        let denom = state
            .v
            .div_scalar(bias_correction2)
            .sqrt()
            .add_scalar(self.eps);
        let mut normalized = g.divide(&denom);
        if self.weight_decay != 0.0 {
            normalized = normalized.add(&y.mul_scalar(self.weight_decay));
        }
        let y_next = y
            .add(&state.z.subtract(&y).mul_scalar(ckp1))
            .add(&normalized.mul_scalar(lr * (self.beta1 * (1.0 - ckp1) - 1.0)));
        state.z = state.z.subtract(&normalized.mul_scalar(lr));
        Self::commit(state, parameter, y_next);
        Ok(())
    }
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<(), Exception> {
        self.advance_step();
        apply_updates(self, model, gradients)
    }
}

impl StatefulOptimizer for ScheduleFreeAdamW {
    fn export_state(&self) -> OptimizerState {
        let mut out = OptimizerState::new("adamw_schedule_free");
        out.set_scalar("k", self.k as f64);
        out.set_scalar("lr_max", self.lr_max);
        out.set_scalar("weight_sum", self.weight_sum);
        out.set_scalar("train_mode", f64::from(u8::from(self.train_mode)));
        for (key, s) in &self.state {
            out.set_tensor("z", key, &s.z);
            out.set_tensor("v", key, &s.v);
            out.set_tensor("y", key, &s.y);
        }
        out
    }

    fn import_state(&mut self, state: &OptimizerState) -> Result<(), Exception> {
        state.expect_kind("adamw_schedule_free")?;
        self.k = state.scalar("k")? as u64;
        self.lr_max = state.scalar("lr_max")?;
        self.weight_sum = state.scalar("weight_sum")?;
        self.train_mode = state.scalar("train_mode")? != 0.0;
        self.state.clear();
        for (key, z) in state.slot("z") {
            let get = |slot: &str| {
                state.tensor(slot, key).cloned().ok_or_else(|| {
                    Exception::custom(format!(
                        "schedule-free optimizer state has `z` for `{key}` but no `{slot}`"
                    ))
                })
            };
            self.state.insert(
                Rc::from(key),
                SfState {
                    z: z.clone(),
                    v: get("v")?,
                    y: get("y")?,
                    produced: None,
                },
            );
        }
        Ok(())
    }
}

impl Updatable for ScheduleFreeAdamW {
    fn updatable_states_len(&self) -> usize {
        self.updatable_states().len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        self.state.values().flat_map(|s| [&s.z, &s.v, &s.y]).collect()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        self.state
            .values_mut()
            .flat_map(|s| [&mut s.z, &mut s.v, &mut s.y])
            .collect()
    }
}

#[cfg(test)]
mod tests {
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

    /// `schedulefree.AdamWScheduleFree`'s step on a vector, in f64, as the
    /// reference writes it.
    struct Reference {
        y: Vec<f64>,
        z: Vec<f64>,
        v: Vec<f64>,
        k: u64,
        lr_max: f64,
        weight_sum: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        r: f64,
        power: f64,
    }

    impl Reference {
        fn new(p0: &[f64], (beta1, beta2, eps, wd): (f64, f64, f64, f64)) -> Self {
            Self {
                y: p0.to_vec(),
                z: p0.to_vec(),
                v: vec![0.0; p0.len()],
                k: 0,
                lr_max: -1.0,
                weight_sum: 0.0,
                beta1,
                beta2,
                eps,
                wd,
                r: 0.0,
                power: 2.0,
            }
        }

        fn step(&mut self, g: &[f64], lr: f64) {
            let k = self.k as f64;
            let bias_correction2 = 1.0 - self.beta2.powf(k + 1.0);
            self.lr_max = self.lr_max.max(lr);
            let weight = (k + 1.0).powf(self.r) * self.lr_max.powf(self.power);
            self.weight_sum += weight;
            let ckp1 = if self.weight_sum == 0.0 {
                0.0
            } else {
                weight / self.weight_sum
            };
            for i in 0..self.y.len() {
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g[i] * g[i];
                let denom = (self.v[i] / bias_correction2).sqrt() + self.eps;
                let mut gn = g[i] / denom;
                if self.wd != 0.0 {
                    gn += self.wd * self.y[i];
                }
                self.y[i] += ckp1 * (self.z[i] - self.y[i]);
                self.y[i] += lr * (self.beta1 * (1.0 - ckp1) - 1.0) * gn;
                self.z[i] -= lr * gn;
            }
            self.k += 1;
        }

        fn x(&self) -> Vec<f64> {
            self.y
                .iter()
                .zip(&self.z)
                .map(|(y, z)| (y - (1.0 - self.beta1) * z) / self.beta1)
                .collect()
        }
    }

    fn close(got: &[f32], want: &[f64], tol: f64, what: &str) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (*g as f64 - w).abs() <= tol * (1.0 + w.abs()),
                "{what}[{i}]: {g} vs {w}"
            );
        }
    }

    struct Weights {
        w: Array,
    }
    crate::impl_module_params!(Weights; w);

    /// A `[2, 3]` weight and six steps of gradients, with a warmup-shaped
    /// learning rate whose first value is zero (which the averaging weight
    /// must survive).
    fn setup() -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
        random::seed(81);
        let p0 = read(&random::uniform_range(-1.0, 1.0, &[2, 3], Dtype::Float32));
        let grads = (0..6)
            .map(|_| read(&random::uniform_range(-2.0, 2.0, &[2, 3], Dtype::Float32)))
            .collect();
        (p0, grads, vec![0.0, 0.01, 0.02, 0.02, 0.02, 0.015])
    }

    #[test]
    fn steps_as_the_reference_does_in_train_and_eval_weights() {
        let (p0, grads, lrs) = setup();
        let (wd, b1, b2, eps) = (0.1_f32, 0.9_f32, 0.999_f32, 1e-8_f32);
        let mut opt = ScheduleFreeAdamW::new(lrs[0], wd);
        let mut want = Reference::new(
            &p0.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            (b1 as f64, b2 as f64, eps as f64, wd as f64),
        );
        let key: Rc<str> = Rc::from("w");
        let mut model = Weights {
            w: Array::from_f32_slice(&p0, &[2, 3]),
        };
        for (g, &lr) in grads.iter().zip(&lrs) {
            opt.lr = lr;
            opt.advance_step();
            opt.update_single(&key, &Array::from_f32_slice(g, &[2, 3]), &mut model.w)
                .expect("update");
            want.step(&g.iter().map(|&v| v as f64).collect::<Vec<_>>(), lr as f64);
            close(&read(&model.w), &want.y, 2e-5, "y");
        }

        // The weights to evaluate are the average x, and swapping to them and
        // back returns the training weights exactly.
        let before = read(&model.w);
        opt.eval_weights(&mut model);
        assert!(!opt.is_train_mode());
        close(&read(&model.w), &want.x(), 5e-5, "x");
        opt.train_weights(&mut model);
        assert_eq!(read(&model.w), before, "y after a round trip");
    }

    #[test]
    fn refuses_to_step_while_the_model_holds_the_averaged_weights() {
        let mut opt = ScheduleFreeAdamW::new(0.01, 0.0);
        let key: Rc<str> = Rc::from("w");
        let ones = || Array::ones(&[2, 2], Dtype::Float32.as_i32());
        let mut model = Weights { w: ones() };
        opt.advance_step();
        opt.update_single(&key, &ones(), &mut model.w)
            .expect("first step");
        opt.eval_weights(&mut model);
        opt.advance_step();
        let err = opt
            .update_single(&key, &ones(), &mut model.w)
            .expect_err("evaluation weights");
        assert!(err.to_string().contains("train_weights"), "{err}");
    }

    /// Stopping after three steps, saving the state and loading it into a
    /// fresh optimizer takes the steps after it the uninterrupted run takes,
    /// bit for bit, with the weights saved as the averaged x.
    #[test]
    fn resumes_bit_exactly_from_its_state() {
        let (p0, grads, lrs) = setup();
        let key: Rc<str> = Rc::from("w");
        let run = |opt: &mut ScheduleFreeAdamW, p: &mut Array, steps: std::ops::Range<usize>| {
            for i in steps {
                opt.lr = lrs[i];
                opt.advance_step();
                opt.update_single(&key, &Array::from_f32_slice(&grads[i], &[2, 3]), p)
                    .expect("update");
            }
        };
        let mut whole = ScheduleFreeAdamW::new(0.0, 0.05);
        let mut pw = Array::from_f32_slice(&p0, &[2, 3]);
        run(&mut whole, &mut pw, 0..6);

        let mut first = ScheduleFreeAdamW::new(0.0, 0.05);
        let mut model = Weights {
            w: Array::from_f32_slice(&p0, &[2, 3]),
        };
        run(&mut first, &mut model.w, 0..3);
        // A checkpoint: the averaged weights and the optimizer's state.
        first.eval_weights(&mut model);
        let saved_weights = read(&model.w);
        let state = first.export_state();

        let mut second = ScheduleFreeAdamW::new(0.0, 0.05);
        second.import_state(&state).expect("import");
        let mut model = Weights {
            w: Array::from_f32_slice(&saved_weights, &[2, 3]),
        };
        second.train_weights(&mut model);
        run(&mut second, &mut model.w, 3..6);
        assert_eq!(read(&model.w), read(&pw));
    }

    /// A bf16 parameter whose per-step change is far below bf16's resolution
    /// still trains: the iterates are f32.
    #[test]
    fn trains_a_bf16_parameter() {
        let bf16 = Dtype::Bfloat16.as_i32();
        let mut opt = ScheduleFreeAdamW::new(1e-4, 0.0);
        let key: Rc<str> = Rc::from("w");
        let mut p = Array::ones(&[4, 4], bf16);
        let g = Array::ones(&[4, 4], bf16);
        for _ in 0..300 {
            opt.advance_step();
            opt.update_single(&key, &g, &mut p).expect("update");
            p.eval();
        }
        assert_eq!(p.dtype(), Dtype::Bfloat16);
        assert!(1.0 - read(&p)[0] > 0.01);
    }
}
