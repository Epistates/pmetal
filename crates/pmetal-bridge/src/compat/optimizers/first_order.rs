//! SGD with momentum, Lion and Adafactor.
//!
//! Each update rule follows the paper that introduced it, with that paper's
//! default hyperparameters, and computes what its reference implementation
//! does. All of them do their arithmetic in f32 whatever the parameter's
//! dtype and update a half-precision parameter through an f32
//! [`MasterWeight`], so a bf16 model trains rather than rounding its updates
//! away.

use super::{Optimizer, State, Updatable, apply_updates};
use crate::compat::{Array, Dtype, Exception, FlattenedModuleParam, ModuleParameters};
use crate::optimizer::MasterWeight;
use std::collections::HashMap;
use std::rc::Rc;

pub(super) fn f32_of(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32.as_i32())
}

pub(super) fn master_of<'a>(
    masters: &'a mut State<MasterWeight>,
    key: &Rc<str>,
) -> &'a mut MasterWeight {
    masters.entry(key.clone()).or_default()
}

/// Plain stochastic gradient descent, `p -= lr·g`, with no state.
#[derive(Debug)]
pub struct Sgd {
    /// Learning rate.
    pub lr: f32,
    /// Nothing to hold: the type of the optimizer's (empty) state.
    pub state: State<()>,
    /// Master copies of half-precision parameters.
    pub masters: State<MasterWeight>,
}

impl Sgd {
    /// SGD at learning rate `lr`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            state: HashMap::new(),
            masters: HashMap::new(),
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
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        let master = master_of(&mut self.masters, key);
        let weight = master.weight(parameter);
        master.commit(
            parameter,
            weight.subtract(&f32_of(gradient).mul_scalar(self.lr)),
        );
        Ok(())
    }
}

impl Updatable for Sgd {
    fn updatable_states_len(&self) -> usize {
        0
    }
    fn updatable_states(&self) -> Vec<&Array> {
        Vec::new()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        Vec::new()
    }
}

/// SGD with heavy-ball momentum, optionally Nesterov, and L2 weight decay, as
/// `torch.optim.SGD` computes it: `d = g + λ·p`, `buf = μ·buf + d` (the first
/// step's `buf` is `d`), then `p -= lr·(d + μ·buf)` under Nesterov momentum
/// (Sutskever et al., ICML 2013) and `p -= lr·buf` otherwise.
#[derive(Debug)]
pub struct MomentumSgd {
    /// Learning rate.
    pub lr: f32,
    /// L2 weight decay, added to the gradient.
    pub weight_decay: f32,
    /// Momentum coefficient.
    pub momentum: f32,
    /// Whether to take Nesterov's look-ahead step.
    pub nesterov: bool,
    /// Momentum buffer per parameter, f32.
    pub state: State<Array>,
    /// Master copies of half-precision parameters.
    pub masters: State<MasterWeight>,
}

impl MomentumSgd {
    /// Momentum coefficient (Sutskever et al., 2013).
    pub const MOMENTUM: f32 = 0.9;

    /// SGD with the default momentum, without Nesterov's step.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            momentum: Self::MOMENTUM,
            nesterov: false,
            state: HashMap::new(),
            masters: HashMap::new(),
        }
    }

    /// Set the momentum coefficient.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Take Nesterov's look-ahead step.
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
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
    ) -> Result<(), Exception> {
        let master = master_of(&mut self.masters, key);
        let weight = master.weight(parameter);
        let mut d = f32_of(gradient);
        if self.weight_decay != 0.0 {
            d = d.add(&weight.mul_scalar(self.weight_decay));
        }
        let buf = match self.state.remove(key) {
            Some(buf) => buf.mul_scalar(self.momentum).add(&d),
            None => d.clone(),
        };
        let step = if self.nesterov {
            d.add(&buf.mul_scalar(self.momentum))
        } else {
            buf.clone()
        };
        master.commit(parameter, weight.subtract(&step.mul_scalar(self.lr)));
        self.state.insert(key.clone(), buf);
        Ok(())
    }
}

/// Lion: the sign of an interpolated momentum, with decoupled weight decay
/// (Chen et al., *Symbolic Discovery of Optimization Algorithms*, arXiv
/// 2302.06675).
///
/// ```text
/// c = β₁·m + (1−β₁)·g
/// p = p − lr·(sign(c) + λ·p)
/// m = β₂·m + (1−β₂)·g
/// ```
#[derive(Debug)]
pub struct Lion {
    /// Learning rate.
    pub lr: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    beta1: f32,
    beta2: f32,
    /// Momentum per parameter, f32.
    pub state: State<Array>,
    /// Master copies of half-precision parameters.
    pub masters: State<MasterWeight>,
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
            masters: HashMap::new(),
        }
    }

    /// Override the betas.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
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
    ) -> Result<(), Exception> {
        let master = master_of(&mut self.masters, key);
        let weight = master.weight(parameter);
        let g = f32_of(gradient);
        let m = self.state.remove(key).unwrap_or_else(|| g.zeros_like());
        let c = m
            .mul_scalar(self.beta1)
            .add(&g.mul_scalar(1.0 - self.beta1));
        let weight = if self.weight_decay != 0.0 {
            weight.subtract(&weight.mul_scalar(self.lr * self.weight_decay))
        } else {
            weight
        };
        master.commit(parameter, weight.subtract(&c.sign().mul_scalar(self.lr)));
        let m = m
            .mul_scalar(self.beta2)
            .add(&g.mul_scalar(1.0 - self.beta2));
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

/// Adafactor with an external learning rate (Shazeer & Stern, arXiv
/// 1804.04235).
///
/// Matrices keep only the row and column means of the squared gradient; the
/// update is that estimate's inverse square root times the gradient, scaled
/// down so its RMS is at most `clip_threshold`. The decay of the estimate
/// grows with the step, `β̂₂ₜ = 1 − t^decay_rate`. The learning rate is the
/// scheduler's: relative step sizes and parameter-scale multiplication, the
/// paper's other two options, are off, as for fine-tuning with a schedule.
#[derive(Debug)]
pub struct Adafactor {
    /// Learning rate.
    pub lr: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    eps1: f32,
    clip_threshold: f32,
    decay_rate: f32,
    step: u64,
    /// Second-moment estimate per parameter, f32.
    pub state: State<SecondMoment>,
    /// Master copies of half-precision parameters.
    pub masters: State<MasterWeight>,
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
            masters: HashMap::new(),
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
    ) -> Result<(), Exception> {
        let t = self.step.max(1) as f32;
        let beta2t = 1.0 - t.powf(self.decay_rate);
        let g = f32_of(gradient);
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

        let master = master_of(&mut self.masters, key);
        let mut p = master.weight(parameter);
        if self.weight_decay != 0.0 {
            p = p.subtract(&p.mul_scalar(self.weight_decay * self.lr));
        }
        master.commit(parameter, p.subtract(&update.mul_scalar(self.lr)));
        self.state.insert(key.clone(), moment);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn read(a: &Array) -> Vec<f32> {
        let mut a = a.as_dtype(Dtype::Float32.as_i32());
        a.eval();
        let n = a.size();
        let v = a.to_f32_vec(n).expect("readable");
        crate::check_unobserved_error().expect("no bridge error");
        v
    }

    /// `torch.optim.SGD` in f64 on one scalar: three steps of gradients 1, 2, 3.
    fn sgd_reference(p0: f64, (lr, wd, mu, nesterov): (f64, f64, f64, bool)) -> f64 {
        let (mut p, mut buf) = (p0, None::<f64>);
        for g in [1.0, 2.0, 3.0] {
            let d = g + wd * p;
            let b = match buf {
                Some(b) => mu * b + d,
                None => d,
            };
            buf = Some(b);
            p -= lr * if nesterov { d + mu * b } else { b };
        }
        p
    }

    #[test]
    fn sgd_with_momentum_follows_torch_with_and_without_nesterov() {
        for nesterov in [false, true] {
            let (lr, wd, mu) = (0.1_f32, 0.01_f32, 0.9_f32);
            let mut opt = MomentumSgd::new(lr, wd)
                .with_momentum(mu)
                .with_nesterov(nesterov);
            let key: Rc<str> = Rc::from("w");
            let mut p = Array::from_f32_slice(&[2.0], &[1]);
            for g in [1.0f32, 2.0, 3.0] {
                opt.update_single(&key, &Array::from_f32_slice(&[g], &[1]), &mut p)
                    .expect("update");
            }
            let want = sgd_reference(2.0, (lr as f64, wd as f64, mu as f64, nesterov));
            let got = read(&p)[0] as f64;
            assert!(
                (got - want).abs() < 1e-6,
                "nesterov={nesterov}: {got} vs {want}"
            );
        }
    }

    /// One step of each optimizer on a bf16 parameter whose per-step update is
    /// far below bf16's resolution at its value, which in bf16 alone never
    /// moves it.
    fn train_bf16(mut step: impl FnMut(&Rc<str>, &Array, &mut Array)) -> f32 {
        let bf16 = Dtype::Bfloat16.as_i32();
        let key: Rc<str> = Rc::from("w");
        let mut p = Array::ones(&[4, 4], bf16);
        let g = Array::ones(&[4, 4], bf16);
        for _ in 0..300 {
            step(&key, &g, &mut p);
            p.eval();
        }
        assert_eq!(p.dtype(), Dtype::Bfloat16);
        read(&p)[0]
    }

    #[test]
    fn every_optimizer_trains_a_bf16_parameter() {
        let lr = 1e-4_f32;
        let mut sgd = Sgd::new(lr);
        let mut momentum = MomentumSgd::new(lr, 0.0);
        let mut lion = Lion::new(lr, 0.0);
        let mut adafactor = Adafactor::new(lr, 0.0);
        let results = [
            (
                "sgd",
                train_bf16(|k, g, p| sgd.update_single(k, g, p).unwrap()),
            ),
            (
                "momentum",
                train_bf16(|k, g, p| momentum.update_single(k, g, p).unwrap()),
            ),
            (
                "lion",
                train_bf16(|k, g, p| lion.update_single(k, g, p).unwrap()),
            ),
            (
                "adafactor",
                train_bf16(|k, g, p| {
                    adafactor.advance_step();
                    adafactor.update_single(k, g, p).unwrap()
                }),
            ),
        ];
        for (name, value) in results {
            assert!(1.0 - value > 0.01, "{name} left a bf16 weight at {value}");
        }
    }
}
