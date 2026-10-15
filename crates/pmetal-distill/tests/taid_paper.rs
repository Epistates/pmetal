//! TAID against the paper (Shing et al., ICLR 2025): the intermediate
//! distribution, the objective, its gradient and the adaptive schedule of `t`,
//! each checked against a float64 computation on the host.

use pmetal_distill::{TaidConfig, TaidDistiller};
use pmetal_runtime::compat::Array;
use pmetal_runtime::compat::nn::value_and_grad_explicit;
use serial_test::serial;

const SHAPE: [i32; 3] = [1, 2, 5];
const VOCAB: usize = 5;

const TEACHER: [f32; 10] = [1.2, -0.4, 2.1, 0.3, -1.0, 0.0, 0.7, -0.2, 1.9, 0.4];
const STUDENT: [f32; 10] = [0.3, 0.9, -0.7, 1.4, 0.2, 1.1, -0.5, 0.6, 0.2, -1.3];

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|x| x / sum).collect()
}

/// `p_t = softmax((1 - t) z_S + t z_T)` per token, float64.
fn intermediate(teacher: &[f32], student: &[f32], t: f64) -> Vec<Vec<f64>> {
    teacher
        .chunks(VOCAB)
        .zip(student.chunks(VOCAB))
        .map(|(zt, zs)| {
            let mixed: Vec<f64> = zt
                .iter()
                .zip(zs)
                .map(|(a, b)| t * *a as f64 + (1.0 - t) * *b as f64)
                .collect();
            softmax(&mixed)
        })
        .collect()
}

fn student_probs(student: &[f32]) -> Vec<Vec<f64>> {
    student
        .chunks(VOCAB)
        .map(|z| softmax(&z.iter().map(|x| *x as f64).collect::<Vec<_>>()))
        .collect()
}

/// Mean over the tokens of `KL(p_t || q)`.
fn objective(teacher: &[f32], student: &[f32], t: f64, mask: &[f64]) -> f64 {
    let p = intermediate(teacher, student, t);
    let q = student_probs(student);
    let total: f64 = p
        .iter()
        .zip(&q)
        .zip(mask)
        .map(|((p, q), w)| w * p.iter().zip(q).map(|(a, b)| a * (a / b).ln()).sum::<f64>())
        .sum();
    total / mask.iter().sum::<f64>()
}

fn config_at(t_start: f64) -> TaidConfig {
    TaidConfig::default().with_t_start(t_start)
}

/// The loss is the KL from the logit-space mixture to the student, whose
/// own contribution to the mixture is a constant.
#[test]
#[serial]
fn the_objective_is_forward_kl_from_the_logit_space_mixture() {
    let teacher = Array::from_f32_slice(&TEACHER, &SHAPE);
    let student = Array::from_f32_slice(&STUDENT, &SHAPE);
    for t in [0.2_f64, 0.4, 0.9, 1.0] {
        let taid = TaidDistiller::new(config_at(t)).unwrap();
        let out = taid.compute_loss(&teacher, &student, None, None).unwrap();
        out.total.eval();
        drain();
        let got = out.total.item::<f32>() as f64;
        let want = objective(&TEACHER, &STUDENT, t, &[1.0, 1.0]);
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1e-3),
            "t={t}: got {got}, float64 reference {want}"
        );
        assert_eq!(out.t, t);
    }
}

/// At `t = 1` the target is the teacher: the objective is `KL(teacher || student)`.
#[test]
#[serial]
fn at_t_one_the_target_is_the_teacher() {
    let teacher = Array::from_f32_slice(&TEACHER, &SHAPE);
    let student = Array::from_f32_slice(&STUDENT, &SHAPE);
    let taid = TaidDistiller::new(config_at(1.0)).unwrap();
    let out = taid.compute_loss(&teacher, &student, None, None).unwrap();
    out.total.eval();
    drain();

    let p = student_probs(&TEACHER);
    let q = student_probs(&STUDENT);
    let kl: f64 = p
        .iter()
        .zip(&q)
        .map(|(p, q)| p.iter().zip(q).map(|(a, b)| a * (a / b).ln()).sum::<f64>())
        .sum::<f64>()
        / 2.0;
    let got = out.total.item::<f32>() as f64;
    assert!((got - kl).abs() <= 1e-5 * kl, "{got} vs KL {kl}");
}

/// The target is detached from the student, so the gradient with respect to
/// the student logits is the softmax-cross-entropy one: `(q - p_t) / N` per
/// token. Differentiating the mixture through the student as well would add a
/// term the paper does not have.
#[test]
#[serial]
fn the_gradient_treats_the_mixture_as_a_constant() {
    let t = 0.4;
    let teacher = Array::from_f32_slice(&TEACHER, &SHAPE);
    let taid = TaidDistiller::new(config_at(t)).unwrap();
    let (value, mut grads) = value_and_grad_explicit(
        |a: &[Array]| {
            taid.compute_loss(&teacher, &a[0], None, None)
                .unwrap()
                .total
        },
        &[Array::from_f32_slice(&STUDENT, &SHAPE)],
        &[],
    )
    .unwrap();
    value.eval();
    grads[0].eval();
    drain();
    let grad = grads[0].to_f32_vec(10).unwrap();

    let p = intermediate(&TEACHER, &STUDENT, t);
    let q = student_probs(&STUDENT);
    for tok in 0..2 {
        for i in 0..VOCAB {
            let want = (q[tok][i] - p[tok][i]) / 2.0;
            let got = grad[tok * VOCAB + i] as f64;
            assert!(
                (got - want).abs() <= 1e-5 + 1e-3 * want.abs(),
                "token {tok} entry {i}: gradient {got}, expected (q - p_t)/N = {want}"
            );
        }
    }
}

/// A masked-out token contributes neither loss nor gradient.
#[test]
#[serial]
fn masked_tokens_are_excluded() {
    let teacher = Array::from_f32_slice(&TEACHER, &SHAPE);
    let mask = Array::from_f32_slice(&[1.0_f32, 0.0], &[1, 2]);
    let taid = TaidDistiller::new(config_at(0.4)).unwrap();
    let (value, mut grads) = value_and_grad_explicit(
        |a: &[Array]| {
            taid.compute_loss(&teacher, &a[0], Some(&mask), None)
                .unwrap()
                .total
        },
        &[Array::from_f32_slice(&STUDENT, &SHAPE)],
        &[],
    )
    .unwrap();
    value.eval();
    grads[0].eval();
    drain();
    let want = objective(&TEACHER, &STUDENT, 0.4, &[1.0, 0.0]);
    let got = value.item::<f32>() as f64;
    assert!((got - want).abs() <= 1e-5 * want, "{got} vs {want}");
    let grad = grads[0].to_f32_vec(10).unwrap();
    assert!(grad[VOCAB..].iter().all(|g| *g == 0.0), "{grad:?}");
}

/// The adaptive schedule, replayed in float64 from a sequence of objectives:
/// `t` follows the paper's momentum update and never falls below the linear
/// floor.
#[test]
fn t_follows_the_momentum_update_and_the_linear_floor() {
    let config = TaidConfig::default()
        .with_t_start(0.3)
        .with_alpha(0.05)
        .with_beta(0.9);
    let total = 40_usize;
    let mut taid = TaidDistiller::new(config.clone()).unwrap();

    // A loss that improves, stalls, then worsens.
    let objectives: Vec<f64> = (1..=total)
        .map(|n| match n {
            1..=15 => 2.0 / (n as f64),
            16..=28 => 0.13,
            _ => 0.13 + 0.02 * (n - 28) as f64,
        })
        .collect();

    let (mut t, mut m, mut previous) = (config.t_start, 0.0_f64, None::<f64>);
    for (i, j) in objectives.iter().enumerate() {
        let n = i + 1;
        assert!(
            (taid.t() - t).abs() < 1e-12,
            "step {n}: t={} expected {t}",
            taid.t()
        );
        taid.observe(*j, n, total);

        let delta = previous.map_or(0.0, |p: f64| (p - j) / (p + config.epsilon));
        previous = Some(*j);
        m = config.beta * m + (1.0 - config.beta) * delta;
        let sigmoid = 1.0 / (1.0 + (-m).exp());
        let linear = config.t_start + (config.t_end - config.t_start) * (n as f64 / total as f64);
        t = (t + config.alpha * sigmoid * (config.t_end - t))
            .max(linear)
            .min(config.t_end);
        assert!(taid.t() >= linear - 1e-12 && taid.t() <= 1.0);
    }
    // The floor reaches the teacher at the last step.
    assert!((taid.t() - 1.0).abs() < 1e-12, "final t = {}", taid.t());
}

/// An objective that is NaN (a diverged step) must not poison `t`.
#[test]
fn a_non_finite_objective_does_not_poison_t() {
    let mut taid = TaidDistiller::new(TaidConfig::default()).unwrap();
    taid.observe(1.0, 1, 100);
    taid.observe(f64::NAN, 2, 100);
    taid.observe(0.9, 3, 100);
    assert!(taid.t().is_finite() && taid.momentum().is_finite());
    assert!(taid.t() >= taid.linear_t(3, 100) - 1e-12);
}

/// Mismatched vocabularies are an error, not a silent truncation: the mixture
/// is taken per token id.
#[test]
#[serial]
fn mismatched_vocabularies_are_rejected() {
    let taid = TaidDistiller::new(TaidConfig::default()).unwrap();
    let teacher = Array::from_f32_slice(&[0.0_f32; 12], &[1, 2, 6]);
    let student = Array::from_f32_slice(&[0.0_f32; 10], &[1, 2, 5]);
    assert!(taid.compute_loss(&teacher, &student, None, None).is_err());
}
