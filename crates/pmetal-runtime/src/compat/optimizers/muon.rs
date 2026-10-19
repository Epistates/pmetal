//! Muon: momentum orthogonalized by Newton–Schulz iteration, for the weight
//! matrices of a network's hidden layers.
//!
//! Jordan et al., *Muon: An optimizer for hidden layers in neural networks*
//! (2024), with the update `torch.optim.Muon` computes and its defaults:
//!
//! ```text
//! B = μ·B + g                          momentum buffer
//! B̃ = g + μ·B  (Nesterov)  or  B       what is orthogonalized
//! O = NewtonSchulz₅(B̃)                 the nearest semi-orthogonal matrix, roughly
//! p = p·(1 − lr·λ)                     decoupled weight decay, at the rate before adjustment
//! p = p − adjusted_lr·O
//! ```
//!
//! Orthogonalizing gives every direction in the update the same size, which
//! plain momentum does not, and each step costs a few matrix products rather
//! than a second-moment state: Muon holds one buffer per matrix against
//! AdamW's two. Moonshot's *Muon is Scalable for LLM Training* (arXiv
//! 2502.16982) adds the `0.2·√max(rows, cols)` scale that gives the update
//! AdamW's RMS, so an AdamW learning rate and weight decay carry over
//! ([`LrAdjust::MatchRmsAdamw`]).
//!
//! Muon is defined for matrices only. [`Muon`] refuses anything else;
//! [`MuonAdamW`] is what a whole model trains with: Muon for the hidden
//! matrices, AdamW for embeddings, the output head, norms and biases.

use super::first_order::{f32_of, master_of};
use super::state::{export_masters, import_masters, import_slot};
use super::{
    AdamW, Optimizer, OptimizerState, State, StatefulOptimizer, Updatable, apply_updates,
};
use crate::compat::{Array, Dtype, Exception, FlattenedModuleParam, ModuleParameters};
use crate::optimizer::{MasterWeight, is_embedding_parameter};
use std::collections::HashMap;
use std::rc::Rc;

/// The quintic's coefficients `(a, b, c)`, tuned so the iteration drives
/// singular values to roughly `[0.7, 1.2]` in five steps (Jordan et al.).
pub const NS_COEFFICIENTS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

/// How the learning rate is scaled by the matrix's shape (`rows` = outputs,
/// `cols` = inputs), as `torch.optim.Muon`'s `adjust_lr_fn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LrAdjust {
    /// `lr·√max(1, rows/cols)`: wide matrices at the learning rate, tall ones
    /// scaled up.
    Original,
    /// `0.2·lr·√max(rows, cols)`: the update has AdamW's RMS (Moonshot).
    MatchRmsAdamw,
    /// `lr·√(rows/cols)`.
    SpectralUnclamped,
    /// The learning rate as given.
    None,
}

impl LrAdjust {
    /// The learning rate for a `[rows, cols]` matrix.
    pub fn apply(self, lr: f32, rows: i32, cols: i32) -> f32 {
        let (r, c) = (rows as f32, cols as f32);
        match self {
            Self::Original => lr * (r / c).max(1.0).sqrt(),
            Self::MatchRmsAdamw => 0.2 * lr * r.max(c).sqrt(),
            Self::SpectralUnclamped => lr * (r / c).sqrt(),
            Self::None => lr,
        }
    }
}

/// The nearest semi-orthogonal matrix to `g` that `steps` of the quintic
/// Newton–Schulz iteration reach: with `X₀ = G/(‖G‖_F + ε)`, each step is
/// `X ← aX + (b·XXᵀ + c·(XXᵀ)²)X`, on the wide orientation (`G` is
/// transposed first when it has more rows than columns). The arithmetic is in
/// `dtype`; the reference implementation uses bf16.
pub fn newton_schulz_orthogonalize(
    g: &Array,
    steps: usize,
    (a, b, c): (f32, f32, f32),
    eps: f32,
    dtype: Dtype,
) -> Array {
    let transposed = g.dim(0) > g.dim(1);
    let g32 = f32_of(g);
    let norm = g32.square().sum_all().sqrt().add_scalar(eps);
    let mut x = g32.divide(&norm).as_dtype(dtype.as_i32());
    if transposed {
        x = x.t();
    }
    for _ in 0..steps {
        let gram = x.matmul(&x.t());
        let poly = gram.mul_scalar(b).add(&gram.matmul(&gram).mul_scalar(c));
        x = x.mul_scalar(a).add(&poly.matmul(&x));
    }
    if transposed {
        x = x.t();
    }
    x
}

/// Muon over matrices: momentum, then Newton–Schulz orthogonalization, then
/// the step, as in the module's description above.
#[derive(Debug)]
pub struct Muon {
    /// Learning rate, before the shape adjustment.
    pub lr: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Momentum coefficient.
    pub momentum: f32,
    /// Whether to orthogonalize `g + μ·B` rather than `B`.
    pub nesterov: bool,
    /// Newton–Schulz iterations.
    pub ns_steps: usize,
    /// The iteration's `(a, b, c)`.
    pub ns_coefficients: (f32, f32, f32),
    /// Added to the gradient's norm before it is divided out.
    pub eps: f32,
    /// How the learning rate follows the matrix's shape.
    pub adjust_lr: LrAdjust,
    /// The dtype the iteration runs in.
    pub ns_dtype: Dtype,
    /// Momentum buffer per matrix, f32.
    pub state: State<Array>,
    /// Master copies of half-precision parameters.
    pub masters: State<MasterWeight>,
}

impl Muon {
    /// Muon with `torch.optim.Muon`'s defaults: momentum 0.95 with Nesterov,
    /// five iterations in bf16, ε = 1e-7 and the original learning-rate
    /// adjustment.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            momentum: 0.95,
            nesterov: true,
            ns_steps: 5,
            ns_coefficients: NS_COEFFICIENTS,
            eps: 1e-7,
            adjust_lr: LrAdjust::Original,
            ns_dtype: Dtype::Bfloat16,
            state: HashMap::new(),
            masters: HashMap::new(),
        }
    }

    /// Choose how the learning rate follows the matrix's shape.
    pub fn with_lr_adjust(mut self, adjust: LrAdjust) -> Self {
        self.adjust_lr = adjust;
        self
    }

    /// Run the Newton–Schulz iteration in `dtype`.
    pub fn with_ns_dtype(mut self, dtype: Dtype) -> Self {
        self.ns_dtype = dtype;
        self
    }

    /// Set the momentum coefficient and whether it is Nesterov's.
    pub fn with_momentum(mut self, momentum: f32, nesterov: bool) -> Self {
        self.momentum = momentum;
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for Muon {
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
        if parameter.ndim() != 2 || gradient.shape() != parameter.shape() {
            return Err(Exception::custom(format!(
                "Muon updates matrices only, and `{key}` is {:?} with a gradient of {:?}: train \
                 embeddings, norms and biases with AdamW (see MuonAdamW)",
                parameter.shape(),
                gradient.shape()
            )));
        }
        let g = f32_of(gradient);
        let buf = match self.state.remove(key) {
            Some(buf) => buf.mul_scalar(self.momentum).add(&g),
            None => g.clone(),
        };
        let direction = if self.nesterov {
            g.add(&buf.mul_scalar(self.momentum))
        } else {
            buf.clone()
        };
        let orthogonal = newton_schulz_orthogonalize(
            &direction,
            self.ns_steps,
            self.ns_coefficients,
            self.eps,
            self.ns_dtype,
        );
        let lr = self
            .adjust_lr
            .apply(self.lr, parameter.dim(0), parameter.dim(1));

        let master = master_of(&mut self.masters, key);
        let mut weight = master.weight(parameter);
        if self.weight_decay != 0.0 {
            weight = weight.mul_scalar(1.0 - self.lr * self.weight_decay);
        }
        master.commit(
            parameter,
            weight.subtract(&f32_of(&orthogonal).mul_scalar(lr)),
        );
        self.state.insert(key.clone(), buf);
        Ok(())
    }
}

/// Muon for a model's hidden matrices and AdamW for everything else: the
/// recipe Muon is meant to be trained with.
///
/// A parameter goes to Muon when it is a matrix and not under a token
/// embedding or output head ([`is_embedding_parameter`]); embeddings, norms,
/// biases and scalars go to AdamW. `set_lr` moves both learning rates
/// together, keeping the ratio AdamW's has to Muon's.
#[derive(Debug)]
pub struct MuonAdamW {
    muon: Muon,
    adamw: AdamW,
    /// AdamW's learning rate over Muon's.
    adamw_ratio: f32,
}

impl MuonAdamW {
    /// Muon at `muon_lr` over the hidden matrices, AdamW at `adamw_lr` over
    /// the rest, both with weight decay `weight_decay` (AdamW leaves biases
    /// and norms undecayed).
    pub fn new(muon_lr: f32, adamw_lr: f32, weight_decay: f32) -> Self {
        Self {
            muon: Muon::new(muon_lr, weight_decay),
            adamw: AdamW::new(adamw_lr, weight_decay),
            adamw_ratio: if muon_lr > 0.0 {
                adamw_lr / muon_lr
            } else {
                1.0
            },
        }
    }

    /// Replace the Muon half, for settings `new` doesn't take.
    pub fn with_muon(mut self, muon: Muon) -> Self {
        self.muon = muon;
        self
    }

    /// Whether `name` is trained by Muon given its `parameter`.
    pub fn uses_muon(name: &str, parameter: &Array) -> bool {
        parameter.ndim() == 2 && !is_embedding_parameter(name)
    }

    /// Muon's learning rate.
    pub fn lr(&self) -> f32 {
        self.muon.lr
    }

    /// Set Muon's learning rate, and AdamW's at its ratio to it.
    pub fn set_lr(&mut self, lr: f32) {
        self.muon.lr = lr;
        self.adamw.set_lr(lr * self.adamw_ratio);
    }

    /// The Muon half.
    pub fn muon(&self) -> &Muon {
        &self.muon
    }

    /// The AdamW half.
    pub fn adamw(&self) -> &AdamW {
        &self.adamw
    }

    /// Advance AdamW's step counter. Once per update, before the
    /// per-parameter calls; [`Optimizer::update`] does it itself.
    pub fn advance_step(&mut self) {
        self.adamw.advance_step();
    }
}

impl Optimizer for MuonAdamW {
    /// The optimizer is its own state; each half keeps its own.
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
        if Self::uses_muon(key, parameter) {
            self.muon.update_single(key, gradient, parameter)
        } else {
            self.adamw.update_single(key, gradient, parameter)
        }
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

impl StatefulOptimizer for Muon {
    fn export_state(&self) -> OptimizerState {
        let mut out = OptimizerState::new("muon");
        for (key, buf) in &self.state {
            out.set_tensor("buf", key, buf);
        }
        export_masters(&self.masters, &mut out);
        out
    }

    fn import_state(&mut self, state: &OptimizerState) -> Result<(), Exception> {
        state.expect_kind("muon")?;
        self.state = import_slot(state, "buf");
        self.masters = import_masters(state);
        Ok(())
    }
}

impl StatefulOptimizer for MuonAdamW {
    fn export_state(&self) -> OptimizerState {
        let mut out = OptimizerState::new("muon_adamw");
        self.muon.export_state().nest("muon", &mut out);
        self.adamw.export_state().nest("adamw", &mut out);
        out
    }

    fn import_state(&mut self, state: &OptimizerState) -> Result<(), Exception> {
        state.expect_kind("muon_adamw")?;
        self.muon.import_state(&state.unnest("muon", "muon"))?;
        self.adamw.import_state(&state.unnest("adamw", "adamw"))
    }
}

impl Updatable for Muon {
    fn updatable_states_len(&self) -> usize {
        self.updatable_states().len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        let masters = self.masters.values().filter_map(|m| m.value());
        self.state.values().chain(masters).collect()
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        let masters = self.masters.values_mut().filter_map(|m| m.value_mut());
        self.state.values_mut().chain(masters).collect()
    }
}

impl Updatable for MuonAdamW {
    fn updatable_states_len(&self) -> usize {
        self.updatable_states().len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        let mut states = self.muon.updatable_states();
        states.extend(self.adamw.updatable_states());
        states
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        let mut states = self.muon.updatable_states_mut();
        states.extend(self.adamw.updatable_states_mut());
        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::random;

    fn read(a: &Array) -> Vec<f32> {
        let mut a = a.as_dtype(Dtype::Float32.as_i32());
        a.eval();
        let n = a.size();
        let v = a.to_f32_vec(n).expect("readable");
        crate::check_unobserved_error().expect("no bridge error");
        v
    }

    type Mat = Vec<Vec<f64>>;

    fn mat(values: &[f32], rows: usize, cols: usize) -> Mat {
        (0..rows)
            .map(|r| (0..cols).map(|c| values[r * cols + c] as f64).collect())
            .collect()
    }

    fn matmul(a: &Mat, b: &Mat) -> Mat {
        (0..a.len())
            .map(|i| {
                (0..b[0].len())
                    .map(|j| (0..b.len()).map(|k| a[i][k] * b[k][j]).sum())
                    .collect()
            })
            .collect()
    }

    fn transpose(a: &Mat) -> Mat {
        (0..a[0].len())
            .map(|j| (0..a.len()).map(|i| a[i][j]).collect())
            .collect()
    }

    /// Keller Jordan's `zeropower_via_newtonschulz5`, in f64.
    fn reference_ns(g: &Mat, steps: usize, (a, b, c): (f64, f64, f64), eps: f64) -> Mat {
        let tall = g.len() > g[0].len();
        let norm = g.iter().flatten().map(|x| x * x).sum::<f64>().sqrt() + eps;
        let mut x: Mat = g
            .iter()
            .map(|r| r.iter().map(|v| v / norm).collect())
            .collect();
        if tall {
            x = transpose(&x);
        }
        for _ in 0..steps {
            let gram = matmul(&x, &transpose(&x));
            let gram2 = matmul(&gram, &gram);
            let poly: Mat = (0..gram.len())
                .map(|i| {
                    (0..gram.len())
                        .map(|j| b * gram[i][j] + c * gram2[i][j])
                        .collect()
                })
                .collect();
            let px = matmul(&poly, &x);
            x = (0..x.len())
                .map(|i| (0..x[0].len()).map(|j| a * x[i][j] + px[i][j]).collect())
                .collect();
        }
        if tall { transpose(&x) } else { x }
    }

    #[test]
    fn newton_schulz_follows_the_reference_iteration_on_wide_and_tall_matrices() {
        random::seed(31);
        let (a, b, c) = NS_COEFFICIENTS;
        for (rows, cols) in [(3usize, 5usize), (5, 3), (4, 4)] {
            let g = random::uniform_range(-1.0, 1.0, &[rows as i32, cols as i32], Dtype::Float32);
            let want = reference_ns(
                &mat(&read(&g), rows, cols),
                5,
                (a as f64, b as f64, c as f64),
                1e-7,
            );
            let got = read(&newton_schulz_orthogonalize(
                &g,
                5,
                NS_COEFFICIENTS,
                1e-7,
                Dtype::Float32,
            ));
            for i in 0..rows {
                for j in 0..cols {
                    assert!(
                        (got[i * cols + j] as f64 - want[i][j]).abs() < 1e-4,
                        "{rows}x{cols} [{i}][{j}]: {} vs {}",
                        got[i * cols + j],
                        want[i][j]
                    );
                }
            }
        }
    }

    /// The iteration's whole point: whatever the input's spectrum, the
    /// output's singular values are all near 1.
    #[test]
    fn newton_schulz_flattens_the_spectrum() {
        random::seed(32);
        // A matrix with singular values spread over three decades.
        let u = random::normal(&[16, 16], Dtype::Float32);
        let v = random::normal(&[32, 16], Dtype::Float32);
        let scales: Vec<f32> = (0..16)
            .map(|i| 10f32.powf(-3.0 * i as f32 / 15.0))
            .collect();
        let g = u
            .multiply(&Array::from_f32_slice(&scales, &[1, 16]))
            .matmul(&v.t());
        let (_, s_in, _) = g.svd();
        let o = newton_schulz_orthogonalize(&g, 5, NS_COEFFICIENTS, 1e-7, Dtype::Float32);
        let (_, s_out, _) = o.svd();
        let (s_in, s_out) = (read(&s_in), read(&s_out));
        assert!(
            s_in[0] / s_in[15] > 100.0,
            "the input's spectrum is not spread"
        );
        // The top 12 singular values (above the iteration's reach for the
        // smallest few) land in [0.5, 1.5].
        for (i, s) in s_out.iter().take(12).enumerate() {
            assert!((0.5..1.5).contains(s), "singular value {i} is {s}");
        }
    }

    #[test]
    fn the_learning_rate_follows_the_shape_as_torch_does() {
        let (lr, rows, cols) = (0.01_f32, 8, 2);
        assert!((LrAdjust::Original.apply(lr, rows, cols) - 0.01 * 2.0).abs() < 1e-7);
        assert!((LrAdjust::Original.apply(lr, cols, rows) - 0.01).abs() < 1e-7);
        assert!(
            (LrAdjust::MatchRmsAdamw.apply(lr, rows, cols) - 0.2 * 0.01 * 8f32.sqrt()).abs() < 1e-7
        );
        assert!((LrAdjust::SpectralUnclamped.apply(lr, rows, cols) - 0.02).abs() < 1e-7);
        assert_eq!(LrAdjust::None.apply(lr, rows, cols), lr);
    }

    /// Two steps of the update against `torch.optim.Muon`'s algorithm written
    /// out in f64: momentum with Nesterov's blend, the orthogonalized
    /// direction, decoupled decay at the unadjusted rate, the adjusted step.
    #[test]
    fn muon_updates_as_the_reference_does() {
        random::seed(33);
        let (rows, cols) = (4usize, 3usize);
        let (lr, wd, mu) = (0.05_f32, 0.1_f32, 0.95_f32);
        let p0 = random::uniform_range(-1.0, 1.0, &[rows as i32, cols as i32], Dtype::Float32);
        let grads: Vec<Array> = (0..2)
            .map(|_| random::uniform_range(-1.0, 1.0, &[rows as i32, cols as i32], Dtype::Float32))
            .collect();

        let mut opt = Muon::new(lr, wd).with_ns_dtype(Dtype::Float32);
        let key: Rc<str> = Rc::from("layers.0.mlp.up_proj.weight");
        let mut p = p0.clone();
        for g in &grads {
            opt.update_single(&key, g, &mut p).expect("update");
        }

        let (a, b, c) = NS_COEFFICIENTS;
        let mut want = mat(&read(&p0), rows, cols);
        let mut buf: Option<Mat> = None;
        for g in &grads {
            let g = mat(&read(g), rows, cols);
            let next: Mat = match &buf {
                Some(prev) => (0..rows)
                    .map(|i| {
                        (0..cols)
                            .map(|j| mu as f64 * prev[i][j] + g[i][j])
                            .collect()
                    })
                    .collect(),
                None => g.clone(),
            };
            let direction: Mat = (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| g[i][j] + mu as f64 * next[i][j])
                        .collect()
                })
                .collect();
            let o = reference_ns(&direction, 5, (a as f64, b as f64, c as f64), 1e-7);
            let adjusted = lr as f64 * (rows as f64 / cols as f64).max(1.0).sqrt();
            for i in 0..rows {
                for j in 0..cols {
                    want[i][j] = want[i][j] * (1.0 - lr as f64 * wd as f64) - adjusted * o[i][j];
                }
            }
            buf = Some(next);
        }
        let got = read(&p);
        for i in 0..rows {
            for j in 0..cols {
                assert!(
                    (got[i * cols + j] as f64 - want[i][j]).abs() < 2e-4,
                    "[{i}][{j}]: {} vs {}",
                    got[i * cols + j],
                    want[i][j]
                );
            }
        }
    }

    /// With Moonshot's adjustment (`0.2·√max(rows, cols)`) one step of a
    /// random gradient moves a matrix by an update whose RMS is AdamW's
    /// (around 0.2), whatever the matrix's shape: the point of the scale,
    /// and what lets an AdamW learning rate and weight decay carry over.
    /// The step itself is the f64 reference's, decay and scale included.
    #[test]
    fn the_moonshot_scale_gives_the_update_adamw_rms_on_every_shape() {
        random::seed(34);
        let (lr, wd) = (0.01_f32, 0.1_f32);
        for (rows, cols) in [(64usize, 128usize), (128, 64), (96, 96), (16, 256)] {
            let shape = [rows as i32, cols as i32];
            let p0 = random::uniform_range(-1.0, 1.0, &shape, Dtype::Float32);
            let g = random::normal(&shape, Dtype::Float32);
            let mut opt = Muon::new(lr, wd)
                .with_lr_adjust(LrAdjust::MatchRmsAdamw)
                .with_ns_dtype(Dtype::Float32);
            let key: Rc<str> = Rc::from("layers.0.q_proj.weight");
            let mut p = p0.clone();
            opt.update_single(&key, &g, &mut p).expect("update");

            // First step: B = g, Nesterov direction g + μ·g.
            let g_mat = mat(&read(&g), rows, cols);
            let direction: Mat = g_mat
                .iter()
                .map(|r| r.iter().map(|v| v * (1.0 + 0.95)).collect())
                .collect();
            let (a, b, c) = NS_COEFFICIENTS;
            let o = reference_ns(&direction, 5, (a as f64, b as f64, c as f64), 1e-7);
            let scale = 0.2 * lr as f64 * (rows.max(cols) as f64).sqrt();
            let p0m = mat(&read(&p0), rows, cols);
            let got = read(&p);
            let mut sq = 0.0;
            for i in 0..rows {
                for j in 0..cols {
                    let want = p0m[i][j] * (1.0 - lr as f64 * wd as f64) - scale * o[i][j];
                    assert!(
                        (got[i * cols + j] as f64 - want).abs() < 2e-4,
                        "{rows}x{cols} [{i}][{j}]"
                    );
                    let step = got[i * cols + j] as f64 - p0m[i][j] * (1.0 - lr as f64 * wd as f64);
                    sq += step * step;
                }
            }
            let rms = (sq / (rows * cols) as f64).sqrt() / lr as f64;
            assert!(
                (0.12..0.3).contains(&rms),
                "{rows}x{cols}: update RMS {rms} over lr, expected about 0.2"
            );
        }
    }

    #[test]
    fn muon_refuses_what_is_not_a_matrix() {
        let mut opt = Muon::new(0.01, 0.0);
        let key: Rc<str> = Rc::from("model.norm.weight");
        let mut p = Array::ones(&[8], Dtype::Float32.as_i32());
        let g = Array::ones(&[8], Dtype::Float32.as_i32());
        let err = opt.update_single(&key, &g, &mut p).expect_err("1-D");
        assert!(err.to_string().contains("matrices only"), "{err}");
    }

    /// Hidden matrices go to Muon; embeddings, the output head, norms and
    /// biases go to AdamW. A bf16 matrix trains through its master copy.
    #[test]
    fn the_hybrid_routes_by_shape_and_name() {
        let mut opt = MuonAdamW::new(0.02, 0.001, 0.0);
        let bf16 = Dtype::Bfloat16.as_i32();
        let matrix = |n: i32| Array::ones(&[n, n], bf16);
        let routed = [
            ("model.layers.0.self_attn.q_proj.weight", matrix(8), true),
            ("model.embed_tokens.weight", matrix(8), false),
            ("lm_head.weight", matrix(8), false),
            ("model.norm.weight", Array::ones(&[8], bf16), false),
            (
                "model.layers.0.self_attn.q_proj.bias",
                Array::ones(&[8], bf16),
                false,
            ),
        ];
        opt.advance_step();
        for (name, param, muon) in routed {
            assert_eq!(MuonAdamW::uses_muon(name, &param), muon, "{name}");
            let key: Rc<str> = Rc::from(name);
            let mut p = param.clone();
            let g = Array::from_f32_slice(&[0.5; 8], &[8]).as_dtype(bf16);
            let g = if p.ndim() == 2 {
                Array::ones(&[8, 8], bf16)
            } else {
                g
            };
            opt.update_single(&key, &g, &mut p).expect("update");
            assert_eq!(p.dtype(), Dtype::Bfloat16, "{name} keeps its dtype");
            assert_eq!(opt.muon().state.contains_key(&key), muon, "{name}");
            assert_eq!(opt.adamw().state.contains_key(&key), !muon, "{name}");
        }
        crate::check_unobserved_error().expect("no bridge error");
        opt.set_lr(0.01);
        assert!(
            (opt.adamw().lr() - 0.0005).abs() < 1e-9,
            "AdamW keeps its ratio to Muon"
        );
    }
}
