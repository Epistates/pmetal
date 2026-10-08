//! The distillation losses are trained by differentiating them, so their
//! gradient with respect to the student is the contract, not their value.
//!
//! Each test differentiates the real loss (not a copy of its formula) with
//! MLX autograd and checks the result two ways: against central finite
//! differences of the same loss, and, where the derivative has a closed form,
//! against that form. A loss that hands back a scalar rebuilt from evaluated
//! data is a constant to autograd, so its gradient comes out as zeros and
//! fails both checks.

use pmetal_bridge::compat::Array;
use pmetal_bridge::compat::nn::value_and_grad_explicit;
use pmetal_distill::losses::{
    DistillLoss, HiddenStateLoss, JensenShannonLoss, KlDivergenceLoss, SoftCrossEntropyLoss,
};
use pmetal_distill::{DistillConfig, DistillMethod, Distiller, LossConfig, LossType};
use serial_test::serial;

const SHAPE: [i32; 3] = [1, 3, 5];

fn teacher_data() -> Vec<f32> {
    vec![
        1.2, -0.4, 2.1, 0.3, -1.0, //
        0.0, 0.7, -0.2, 1.9, 0.4, //
        -1.5, 2.4, 0.8, 0.1, -0.6,
    ]
}

fn student_data() -> Vec<f32> {
    vec![
        0.3, 0.9, -0.7, 1.4, 0.2, //
        1.1, -0.5, 0.6, 0.2, -1.3, //
        0.4, 0.4, 1.7, -0.9, 0.5,
    ]
}

fn mask() -> Array {
    Array::from_f32_slice(&[1.0_f32, 0.0, 1.0], &[1, 3])
}

fn drain() {
    pmetal_bridge::check_last_error().expect("a bridge op threw");
}

/// Loss value and autograd gradient of `f` at `x`.
fn autograd(f: &dyn Fn(&Array) -> Array, x: &[f32], shape: &[i32]) -> (f32, Vec<f32>) {
    let input = Array::from_f32_slice(x, shape);
    let (loss, grads) = value_and_grad_explicit(|a: &[Array]| f(&a[0]), &[input], &[]).unwrap();
    loss.eval();
    grads[0].eval();
    drain();
    let grad = grads[0]
        .as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32())
        .to_f32_vec(x.len())
        .unwrap();
    (loss.item::<f32>(), grad)
}

/// Central finite differences of `f` at `x`.
fn finite_differences(f: &dyn Fn(&Array) -> Array, x: &[f32], shape: &[i32]) -> Vec<f32> {
    let h = 5e-3_f32;
    (0..x.len())
        .map(|i| {
            let mut plus = x.to_vec();
            let mut minus = x.to_vec();
            plus[i] += h;
            minus[i] -= h;
            let fp = f(&Array::from_f32_slice(&plus, shape)).item::<f32>();
            let fm = f(&Array::from_f32_slice(&minus, shape)).item::<f32>();
            (fp - fm) / (2.0 * h)
        })
        .collect()
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 3e-4 + 2e-2 * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{label}: entry {i} autograd={g} reference={w}\n  autograd: {got:?}\n  reference: {want:?}"
        );
    }
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn logit_losses() -> Vec<(&'static str, Box<dyn DistillLoss>)> {
    vec![
        ("kl_divergence", Box::new(KlDivergenceLoss::new())),
        (
            "reverse_kl_divergence",
            Box::new(KlDivergenceLoss::reverse()),
        ),
        ("jensen_shannon", Box::new(JensenShannonLoss::new())),
        ("soft_cross_entropy", Box::new(SoftCrossEntropyLoss::new())),
    ]
}

#[test]
#[serial]
fn every_logit_loss_matches_finite_differences() {
    let teacher = Array::from_f32_slice(&teacher_data(), &SHAPE);
    let student = student_data();
    let mask = mask();

    for (name, loss) in logit_losses() {
        for temperature in [1.0_f32, 2.0] {
            for weights in [None, Some(&mask)] {
                let f = |s: &Array| {
                    loss.compute_weighted(&teacher, s, temperature, weights)
                        .unwrap()
                };
                let label = format!("{name} T={temperature} weighted={}", weights.is_some());
                let (_, grad) = autograd(&f, &student, &SHAPE);
                let reference = finite_differences(&f, &student, &SHAPE);
                assert!(
                    norm(&grad) > 1e-3,
                    "{label}: the student received no gradient: {grad:?}"
                );
                assert_close(&label, &grad, &reference);

                // A masked-out token must contribute nothing.
                if weights.is_some() {
                    assert!(
                        grad[5..10].iter().all(|g| *g == 0.0),
                        "{label}: masked token got gradient {:?}",
                        &grad[5..10]
                    );
                }
            }
        }
    }
}

/// `compute_masked` averages over the unmasked tokens only.
#[test]
#[serial]
fn compute_masked_ignores_masked_tokens() {
    let teacher = Array::from_f32_slice(&teacher_data(), &SHAPE);
    let student = Array::from_f32_slice(&student_data(), &SHAPE);
    let mask = mask();
    for (name, loss) in logit_losses() {
        let masked: f32 = loss
            .compute_masked(&teacher, &student, 2.0, &mask)
            .unwrap()
            .item();
        let weighted: f32 = loss
            .compute_weighted(&teacher, &student, 2.0, Some(&mask))
            .unwrap()
            .item();
        let unmasked: f32 = loss.compute(&teacher, &student, 2.0).unwrap().item();
        drain();
        assert!(
            (masked - weighted).abs() < 1e-6,
            "{name}: {masked} vs {weighted}"
        );
        assert!(
            (masked - unmasked).abs() > 1e-4,
            "{name}: the mask changed nothing ({masked})"
        );
    }
}

fn softmax_rows(x: &[f32], vocab: usize, temperature: f32) -> Vec<f32> {
    x.chunks(vocab)
        .flat_map(|row| {
            let max = row.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let e: Vec<f32> = row
                .iter()
                .map(|v| ((v - max) / temperature).exp())
                .collect();
            let z: f32 = e.iter().sum();
            e.into_iter().map(move |v| v / z)
        })
        .collect()
}

fn kl_distiller(temperature: f32, alpha: f32) -> Distiller {
    Distiller::new(DistillConfig {
        teacher: "t".to_string(),
        student: "s".to_string(),
        method: DistillMethod::Online,
        loss: LossConfig {
            loss_type: LossType::KlDivergence,
            temperature,
            alpha,
            ..LossConfig::default()
        },
        offline: None,
        output_path: None,
        training: Default::default(),
    })
    .unwrap()
}

/// Hinton et al. (2015) §2: the soft term is `T² · KL(p_T ‖ q_T)`, whose
/// gradient with respect to student logit `z_i` is `T · (q_i − p_i)`. With a
/// per-token weight `w`, token `n` carries `w_n / Σw` of it.
///
/// This runs the trainer's own path: `Distiller::compute_loss` with labels and
/// the label mask as weights, `alpha = 1` so only the soft term is left.
#[test]
#[serial]
fn distiller_soft_gradient_is_t_squared_scaled_kl() {
    let teacher_raw = teacher_data();
    let teacher = Array::from_f32_slice(&teacher_raw, &SHAPE);
    let student = student_data();
    let labels = Array::from_i32_slice(&[2_i32, 0, 1]).reshape(&[1, 3]);
    let weights = [1.0_f32, 0.0, 1.0];
    let mask = mask();
    let vocab = SHAPE[2] as usize;

    for temperature in [1.0_f32, 2.0, 4.0] {
        let distiller = kl_distiller(temperature, 1.0);
        let f = |s: &Array| {
            distiller
                .compute_loss(&teacher, s, Some(&labels), Some(&mask), 0, 1)
                .unwrap()
                .total
        };
        let (_, grad) = autograd(&f, &student, &SHAPE);

        let p = softmax_rows(&teacher_raw, vocab, temperature);
        let q = softmax_rows(&student, vocab, temperature);
        let total_weight: f32 = weights.iter().sum();
        let analytic: Vec<f32> = (0..student.len())
            .map(|i| temperature * (weights[i / vocab] / total_weight) * (q[i] - p[i]))
            .collect();

        assert_close(&format!("T={temperature}"), &grad, &analytic);
    }
}

/// With `alpha = 0.5` the total gradient is the average of the soft and hard
/// gradients; the soft half must be there.
#[test]
#[serial]
fn distiller_total_carries_both_terms() {
    let teacher = Array::from_f32_slice(&teacher_data(), &SHAPE);
    let student = student_data();
    let labels = Array::from_i32_slice(&[2_i32, 0, 1]).reshape(&[1, 3]);

    let grad_of = |alpha: f32| {
        let distiller = kl_distiller(2.0, alpha);
        let f = |s: &Array| {
            distiller
                .compute_loss(&teacher, s, Some(&labels), None, 0, 1)
                .unwrap()
                .total
        };
        autograd(&f, &student, &SHAPE).1
    };
    let soft = grad_of(1.0);
    let hard = grad_of(0.0);
    let half = grad_of(0.5);
    let expected: Vec<f32> = soft.iter().zip(&hard).map(|(s, h)| 0.5 * (s + h)).collect();
    assert!(norm(&soft) > 1e-3, "soft term has no gradient");
    assert_close("alpha=0.5", &half, &expected);
}

/// Teacher and student rows that are permutations of each other give some
/// tokens the same log-probability on both sides, where `log(p + q)` has to
/// split its derivative evenly between them.
#[test]
#[serial]
fn jensen_shannon_gradient_is_exact_where_teacher_and_student_tie() {
    let shape = [1, 2, 4];
    let teacher = Array::from_f32_slice(&[0.0_f32, 1.0, 2.0, 3.0, 2.0, 0.5, 1.0, 0.5], &shape);
    let student = [0.0_f32, 1.0, 3.0, 2.0, 2.0, 1.0, 0.5, 0.5];
    let loss = JensenShannonLoss::new();
    let f = |s: &Array| loss.compute(&teacher, s, 1.0).unwrap();
    let (_, grad) = autograd(&f, &student, &shape);
    let reference = finite_differences(&f, &student, &shape);
    assert_close("jsd ties", &grad, &reference);
}

#[test]
#[serial]
fn hidden_state_losses_match_finite_differences() {
    let shape = [1, 3, 5];
    let teacher = Array::from_f32_slice(&teacher_data(), &shape);
    let student = student_data();
    for (name, loss) in [
        ("mse", HiddenStateLoss::mse()),
        ("cosine", HiddenStateLoss::cosine()),
        ("l1", HiddenStateLoss::l1()),
    ] {
        let f = |s: &Array| loss.compute(&teacher, s).unwrap();
        let (_, grad) = autograd(&f, &student, &shape);
        let reference = finite_differences(&f, &student, &shape);
        assert!(norm(&grad) > 1e-3, "{name}: no gradient");
        assert_close(name, &grad, &reference);
    }
}
