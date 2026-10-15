//! MiniLLM (Gu et al., ICLR 2024) against its equations, in float64.
//!
//! The loss is a surrogate whose gradient is the paper's policy gradient:
//! importance weights, the per-token rewards `R^Norm` and the teacher-mixed
//! behavior distribution are constants, and everything else is differentiated.
//! The reference below evaluates the same surrogate on the host with those
//! constants frozen at the base point, so its finite differences are the
//! gradient the implementation must produce.

use pmetal_distill::losses::MiniLlmLoss;
use pmetal_runtime::compat::nn::value_and_grad_explicit;
use pmetal_runtime::compat::{Array, Dtype};
use serial_test::serial;

const B: usize = 2;
const T: usize = 5;
const V: usize = 6;

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn logits(seed: f32, scale: f32) -> Vec<f32> {
    (0..B * T * V)
        .map(|i| scale * ((i as f32) * 0.83 + seed).sin() + 0.02 * (i % V) as f32)
        .collect()
}

const TOKENS: [i32; B * T] = [2, 5, 0, 3, 1, 4, 4, 1, 0, 2];
/// Row 0: the first position is a prompt token; row 1 is padded after three.
const MASK: [f32; B * T] = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];

fn log_softmax(row: &[f64]) -> Vec<f64> {
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    row.iter().map(|x| x - lse).collect()
}

struct Reference {
    alpha: f64,
    eps: f64,
    normalize: bool,
    teacher: Vec<f64>,
    /// The student the constants are frozen at.
    base: Vec<f64>,
}

impl Reference {
    fn row(x: &[f64], pos: usize) -> Vec<f64> {
        log_softmax(&x[pos * V..(pos + 1) * V])
    }

    /// `R^Norm_{t+1}` for every position, at the base point.
    fn rewards(&self) -> Vec<f64> {
        let mut out = vec![0.0; B * T];
        for b in 0..B {
            let r: Vec<f64> = (0..T)
                .map(|t| {
                    let pos = b * T + t;
                    let y = TOKENS[pos] as usize;
                    MASK[pos] as f64
                        * (Self::row(&self.teacher, pos)[y] - Self::row(&self.base, pos)[y])
                })
                .collect();
            for t in 0..T {
                let after: f64 = r[t + 1..].iter().sum();
                let count: f64 = (t + 1..T).map(|u| MASK[b * T + u] as f64).sum();
                out[b * T + t] = if self.normalize {
                    after / count.max(1.0)
                } else {
                    after
                };
            }
        }
        out
    }

    /// `rho_t = q(y_t) / p~(y_t)` at student `z`, with `p~` mixed from the base.
    fn rho(&self, z: &[f64], pos: usize) -> f64 {
        let y = TOKENS[pos] as usize;
        let q = Self::row(z, pos)[y].exp();
        let q0 = Self::row(&self.base, pos)[y].exp();
        let p = Self::row(&self.teacher, pos)[y].exp();
        q / (self.alpha * p + (1.0 - self.alpha) * q0)
    }

    fn loss(&self, z: &[f64], rewards: &[f64], weights: &[f64]) -> (f64, f64, f64) {
        let (mut single, mut long, mut count) = (0.0, 0.0, 0.0);
        for pos in 0..B * T {
            let m = MASK[pos] as f64;
            if m == 0.0 {
                continue;
            }
            let lq = Self::row(z, pos);
            let lp = Self::row(&self.teacher, pos);
            let kl: f64 = lq.iter().zip(&lp).map(|(a, b)| a.exp() * (a - b)).sum();
            single += weights[pos] * kl;
            let rho = self.rho(z, pos);
            let clipped = rho.clamp(1.0 - self.eps, 1.0 + self.eps);
            long += -(rho * rewards[pos]).min(clipped * rewards[pos]);
            count += m;
        }
        ((single + long) / count, single / count, long / count)
    }
}

fn f64s(x: &[f32]) -> Vec<f64> {
    x.iter().map(|v| *v as f64).collect()
}

fn arrays(teacher: &[f32], student: &[f32]) -> (Array, Array, Array, Array) {
    let shape = [B as i32, T as i32, V as i32];
    (
        Array::from_f32_slice(teacher, &shape),
        Array::from_f32_slice(student, &shape),
        Array::from_i32_slice_shaped(&TOKENS, &[B as i32, T as i32]),
        Array::from_f32_slice(&MASK, &[B as i32, T as i32]),
    )
}

#[test]
#[serial]
fn value_and_gradient_match_the_float64_surrogate() {
    let (teacher, student) = (logits(0.4, 2.5), logits(1.7, 2.5));
    for (alpha, eps, normalize) in [
        (0.2_f32, 0.2_f32, true),
        (0.2, 0.2, false),
        (0.0, 0.3, true),
        (0.5, 0.1, true),
    ] {
        let loss = MiniLlmLoss::new(alpha)
            .with_clip_epsilon(eps)
            .with_length_normalization(normalize);
        let (t, s, tokens, mask) = arrays(&teacher, &student);

        let (value, mut grads) = value_and_grad_explicit(
            |a: &[Array]| {
                loss.compute(&t, &a[0], None, &tokens, Some(&mask))
                    .unwrap()
                    .total
            },
            std::slice::from_ref(&s),
            &[],
        )
        .unwrap();
        value.eval();
        grads[0].eval();
        drain();
        let got = value.item::<f32>() as f64;
        let grad = grads[0].to_f32_vec(B * T * V).unwrap();

        let reference = Reference {
            alpha: alpha as f64,
            eps: eps as f64,
            normalize,
            teacher: f64s(&teacher),
            base: f64s(&student),
        };
        let rewards = reference.rewards();
        let weights: Vec<f64> = (0..B * T)
            .map(|pos| {
                reference
                    .rho(&reference.base, pos)
                    .clamp(1.0 - reference.eps, 1.0 + reference.eps)
            })
            .collect();
        let (want, _, _) = reference.loss(&reference.base, &rewards, &weights);
        assert!(
            (got - want).abs() <= 2e-5 * want.abs().max(1e-3),
            "alpha={alpha} eps={eps} normalize={normalize}: value {got}, float64 {want}"
        );

        let h = 1e-5;
        for i in 0..B * T * V {
            let mut z = reference.base.clone();
            z[i] += h;
            let up = reference.loss(&z, &rewards, &weights).0;
            z[i] -= 2.0 * h;
            let down = reference.loss(&z, &rewards, &weights).0;
            let fd = (up - down) / (2.0 * h);
            assert!(
                (grad[i] as f64 - fd).abs() <= 2e-5 + 2e-3 * fd.abs(),
                "alpha={alpha} eps={eps} normalize={normalize} entry {i}: autograd {} vs finite differences {fd}",
                grad[i]
            );
        }
    }
}

/// A student equal to the teacher has nothing to learn: no loss, no gradient.
#[test]
#[serial]
fn identical_models_give_zero_loss_and_gradient() {
    let z = logits(0.4, 2.5);
    let (t, s, tokens, mask) = arrays(&z, &z);
    let loss = MiniLlmLoss::new(0.2);
    let (value, mut grads) = value_and_grad_explicit(
        |a: &[Array]| {
            loss.compute(&t, &a[0], None, &tokens, Some(&mask))
                .unwrap()
                .total
        },
        &[s],
        &[],
    )
    .unwrap();
    value.eval();
    grads[0].eval();
    drain();
    assert!(value.item::<f32>().abs() < 1e-6);
    let grad = grads[0].to_f32_vec(B * T * V).unwrap();
    assert!(grad.iter().all(|g| g.abs() < 1e-6), "{grad:?}");
}

/// Prompt and padding positions contribute nothing, even with junk logits.
#[test]
#[serial]
fn masked_positions_carry_no_gradient() {
    let (teacher, student) = (logits(0.4, 2.5), logits(1.7, 2.5));
    let (t, s, tokens, mask) = arrays(&teacher, &student);
    let loss = MiniLlmLoss::new(0.2);
    let (_, mut grads) = value_and_grad_explicit(
        |a: &[Array]| {
            loss.compute(&t, &a[0], None, &tokens, Some(&mask))
                .unwrap()
                .total
        },
        &[s],
        &[],
    )
    .unwrap();
    grads[0].eval();
    drain();
    let grad = grads[0].to_f32_vec(B * T * V).unwrap();
    for pos in 0..B * T {
        if MASK[pos] == 0.0 {
            assert!(
                grad[pos * V..(pos + 1) * V].iter().all(|g| *g == 0.0),
                "masked position {pos} has gradient"
            );
        }
    }
}

/// `p~ = alpha*p + (1-alpha)*q` is a distribution, for the rollout to sample from.
#[test]
#[serial]
fn teacher_mixed_distribution_is_normalised() {
    let (teacher, student) = (logits(0.4, 2.5), logits(1.7, 2.5));
    let (t, s, _, _) = arrays(&teacher, &student);
    let loss = MiniLlmLoss::new(0.2);
    let log_p = loss.mixed_log_probs(&t, &s);
    let sums = log_p
        .exp()
        .sum_axes(&[-1], false)
        .as_dtype(Dtype::Float32.as_i32())
        .to_f32_vec(B * T)
        .unwrap();
    drain();
    assert!(sums.iter().all(|s| (s - 1.0).abs() < 1e-5), "{sums:?}");
}
