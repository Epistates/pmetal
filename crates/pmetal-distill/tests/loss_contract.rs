//! Contracts every [`DistillLoss`] owes its caller, checked against float64
//! references computed on the host.
//!
//! * the weighted mean divides by the total weight, so a uniform weight of any
//!   size gives the unweighted mean;
//! * the teacher is a constant: no gradient reaches it, even when it was
//!   computed inside the differentiated function;
//! * logits of any dtype are normalized in f32.

use pmetal_distill::losses::{
    DistillLoss, HiddenStateLoss, HingeRankingLoss, JensenShannonLoss, JsdSkewedLoss,
    KlDivergenceLoss, LogisticRankingLoss, MiniLlmLoss, MseLoss, SoftCrossEntropyLoss, TvdLoss,
    UniversalLogitLoss,
};
use pmetal_runtime::compat::nn::value_and_grad_explicit;
use pmetal_runtime::compat::{Array, Dtype};
use serial_test::serial;

const TOKENS: usize = 2;
const VOCAB: usize = 7;

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

/// Logits spread over a range wide enough that a bf16 normalizer is visibly
/// wrong, but with a distinct maximum per row.
fn logits(seed: f32) -> Vec<f32> {
    (0..TOKENS * VOCAB)
        .map(|i| 9.0 * ((i as f32) * 1.37 + seed).sin() + 0.3 * (i as f32))
        .collect()
}

fn array(data: &[f32], dtype: Dtype) -> Array {
    Array::from_f32_slice(data, &[1, TOKENS as i32, VOCAB as i32]).as_dtype(dtype.as_i32())
}

/// What an array holds after the cast to `dtype` and back: the values the
/// loss actually sees.
fn rounded(data: &[f32], dtype: Dtype) -> Vec<f64> {
    array(data, dtype)
        .as_dtype(Dtype::Float32.as_i32())
        .to_f32_vec(data.len())
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

fn scalar(a: &Array) -> f32 {
    a.eval();
    drain();
    a.item::<f32>()
}

fn every_loss() -> Vec<(&'static str, Box<dyn DistillLoss>)> {
    vec![
        ("kl", Box::new(KlDivergenceLoss::new())),
        ("reverse_kl", Box::new(KlDivergenceLoss::reverse())),
        ("jsd", Box::new(JensenShannonLoss::new())),
        ("soft_ce", Box::new(SoftCrossEntropyLoss::new())),
        ("tvd", Box::new(TvdLoss::new())),
        ("jsd_skewed", Box::new(JsdSkewedLoss::new(0.3))),
        ("minillm", Box::new(MiniLlmLoss::new(0.8))),
        ("uld", Box::new(UniversalLogitLoss::new())),
        ("mse", Box::new(MseLoss::new())),
        ("hinge", Box::new(HingeRankingLoss::new().with_top_k(5))),
        (
            "logistic",
            Box::new(LogisticRankingLoss::new().with_top_k(5)),
        ),
    ]
}

/// Weights of 0.25 on two tokens sum to 0.5. The mean over weighted tokens is
/// the same as the unweighted mean; dividing by `max(sum, 1)` instead halves
/// it.
#[test]
#[serial]
fn uniform_fractional_weights_leave_the_mean_unchanged() {
    let teacher = array(&logits(0.3), Dtype::Float32);
    let student = array(&logits(1.9), Dtype::Float32);
    let weights = Array::from_f32_slice(&[0.25_f32, 0.25], &[1, 2]);
    for (name, loss) in every_loss() {
        let plain = scalar(&loss.compute(&teacher, &student, 2.0).unwrap());
        let weighted = scalar(
            &loss
                .compute_weighted(&teacher, &student, 2.0, Some(&weights))
                .unwrap(),
        );
        assert!(
            (plain - weighted).abs() <= 1e-4 * plain.abs().max(1.0),
            "{name}: unweighted {plain} vs uniform-weight {weighted}"
        );
    }
}

/// Every token masked out: the loss is zero, not NaN.
#[test]
#[serial]
fn all_zero_weights_give_zero_loss() {
    let teacher = array(&logits(0.3), Dtype::Float32);
    let student = array(&logits(1.9), Dtype::Float32);
    let weights = Array::from_f32_slice(&[0.0_f32, 0.0], &[1, 2]);
    for (name, loss) in every_loss() {
        let v = scalar(
            &loss
                .compute_weighted(&teacher, &student, 2.0, Some(&weights))
                .unwrap(),
        );
        assert!(v == 0.0, "{name}: got {v}");
    }
}

/// The teacher is a target. Differentiating a loss with respect to both
/// logits (as happens when the teacher forward runs inside the differentiated
/// function) must leave the teacher's gradient at zero.
#[test]
#[serial]
fn the_teacher_receives_no_gradient() {
    let t = logits(0.3);
    let s = logits(1.9);
    for (name, loss) in every_loss() {
        let shape = [1, TOKENS as i32, VOCAB as i32];
        let f = |a: &[Array]| loss.compute(&a[0], &a[1], 2.0).unwrap();
        let (value, mut grads) = value_and_grad_explicit(
            f,
            &[
                Array::from_f32_slice(&t, &shape),
                Array::from_f32_slice(&s, &shape),
            ],
            &[],
        )
        .unwrap();
        value.eval();
        grads[0].eval();
        grads[1].eval();
        drain();
        let g_teacher = grads[0].to_f32_vec(t.len()).unwrap();
        let g_student = grads[1].to_f32_vec(s.len()).unwrap();
        assert!(
            g_student.iter().any(|g| g.abs() > 1e-6),
            "{name}: the student received no gradient"
        );
        assert!(
            g_teacher.iter().all(|g| *g == 0.0),
            "{name}: the teacher received gradient {g_teacher:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// float64 references
// ---------------------------------------------------------------------------

fn log_softmax(row: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = row.iter().map(|x| x / temperature).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + scaled.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    scaled.iter().map(|x| x - lse).collect()
}

/// Mean over tokens of `per_token(log p_teacher, log p_student)`.
fn reference(
    teacher: &[f64],
    student: &[f64],
    temperature: f64,
    per_token: impl Fn(&[f64], &[f64]) -> f64,
) -> f64 {
    (0..TOKENS)
        .map(|t| {
            let lt = log_softmax(&teacher[t * VOCAB..(t + 1) * VOCAB], temperature);
            let ls = log_softmax(&student[t * VOCAB..(t + 1) * VOCAB], temperature);
            per_token(&lt, &ls)
        })
        .sum::<f64>()
        / TOKENS as f64
}

fn tvd(lt: &[f64], ls: &[f64]) -> f64 {
    0.5 * lt
        .iter()
        .zip(ls)
        .map(|(a, b)| (a.exp() - b.exp()).abs())
        .sum::<f64>()
}

/// `alpha * KL(T || M) + (1 - alpha) * KL(S || M)`, `M = alpha T + (1 - alpha) S`.
fn skewed_jsd(alpha: f64) -> impl Fn(&[f64], &[f64]) -> f64 {
    move |lt, ls| {
        lt.iter()
            .zip(ls)
            .map(|(a, b)| {
                let (pt, ps) = (a.exp(), b.exp());
                let m = alpha * pt + (1.0 - alpha) * ps;
                alpha * pt * (a - m.ln()) + (1.0 - alpha) * ps * (b - m.ln())
            })
            .sum()
    }
}

/// `KL(S || mix T + (1 - mix) S)`.
fn mixed_reverse_kl(mix: f64) -> impl Fn(&[f64], &[f64]) -> f64 {
    move |lt, ls| {
        lt.iter()
            .zip(ls)
            .map(|(a, b)| {
                let (pt, ps) = (a.exp(), b.exp());
                let m = mix * pt + (1.0 - mix) * ps;
                ps * (b - m.ln())
            })
            .sum()
    }
}

fn kl(lt: &[f64], ls: &[f64]) -> f64 {
    lt.iter().zip(ls).map(|(a, b)| a.exp() * (a - b)).sum()
}

/// A bf16 logit tensor is normalized in f32: the loss matches float64 on the
/// same (bf16-rounded) logits to f32 accuracy, not to bf16's three digits.
#[test]
#[serial]
fn bf16_logits_are_normalised_in_f32() {
    let (t, s) = (logits(0.3), logits(1.9));
    let teacher = array(&t, Dtype::Bfloat16);
    let student = array(&s, Dtype::Bfloat16);
    let (t64, s64) = (rounded(&t, Dtype::Bfloat16), rounded(&s, Dtype::Bfloat16));
    let temperature = 2.0_f32;

    type Case = (
        &'static str,
        Box<dyn DistillLoss>,
        Box<dyn Fn(&[f64], &[f64]) -> f64>,
    );
    let cases: Vec<Case> = vec![
        ("kl", Box::new(KlDivergenceLoss::new()), Box::new(kl)),
        ("tvd", Box::new(TvdLoss::new()), Box::new(tvd)),
        (
            "jsd_skewed",
            Box::new(JsdSkewedLoss::new(0.3)),
            Box::new(skewed_jsd(0.3)),
        ),
        (
            "minillm",
            Box::new(MiniLlmLoss::new(0.8)),
            Box::new(mixed_reverse_kl(0.8)),
        ),
    ];
    for (name, loss, per_token) in cases {
        let got = scalar(&loss.compute(&teacher, &student, temperature).unwrap()) as f64;
        let want = reference(&t64, &s64, temperature as f64, per_token);
        assert!(
            (got - want).abs() <= 2e-5 * want.abs().max(1e-2),
            "{name}: got {got}, float64 reference {want}"
        );
    }
}

/// Universal Logit Distillation (Boizard et al., 2024, eq. 5): the L1 distance
/// between the two probability vectors sorted in decreasing order, the shorter
/// padded with zeros. A sum over the sorted entries, not a mean, and not
/// over their cumulative sums.
fn uld_reference(teacher: &[f64], student: &[f64], temperature: f64, top_k: Option<usize>) -> f64 {
    let sorted = |row: &[f64]| {
        let mut p: Vec<f64> = log_softmax(row, temperature)
            .into_iter()
            .map(f64::exp)
            .collect();
        p.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if let Some(k) = top_k {
            p.truncate(k);
        }
        p
    };
    let (t, s) = (sorted(teacher), sorted(student));
    (0..t.len().max(s.len()))
        .map(|i| (t.get(i).unwrap_or(&0.0) - s.get(i).unwrap_or(&0.0)).abs())
        .sum()
}

fn uld_mean_reference(
    teacher: &[f64],
    teacher_vocab: usize,
    student: &[f64],
    student_vocab: usize,
    temperature: f64,
    top_k: Option<usize>,
) -> f64 {
    (0..TOKENS)
        .map(|t| {
            uld_reference(
                &teacher[t * teacher_vocab..(t + 1) * teacher_vocab],
                &student[t * student_vocab..(t + 1) * student_vocab],
                temperature,
                top_k,
            )
        })
        .sum::<f64>()
        / TOKENS as f64
}

/// ULD against the paper's closed form, with equal and with different
/// vocabularies.
#[test]
#[serial]
fn uld_is_the_l1_distance_between_sorted_distributions() {
    let teacher_vocab = VOCAB;
    let student_vocab = 5;
    let t = logits(0.3);
    let s: Vec<f32> = (0..TOKENS * student_vocab)
        .map(|i| 3.0 * ((i as f32) * 0.77 + 1.1).cos())
        .collect();
    let teacher = array(&t, Dtype::Float32);
    let student = Array::from_f32_slice(&s, &[1, TOKENS as i32, student_vocab as i32]);
    let t64: Vec<f64> = t.iter().map(|x| *x as f64).collect();
    let s64: Vec<f64> = s.iter().map(|x| *x as f64).collect();

    for (top_k, loss) in [
        (None, UniversalLogitLoss::new()),
        (Some(4), UniversalLogitLoss::new().with_top_k(4)),
    ] {
        for temperature in [1.0_f32, 2.0] {
            let got = scalar(&loss.compute(&teacher, &student, temperature).unwrap()) as f64;
            let want = uld_mean_reference(
                &t64,
                teacher_vocab,
                &s64,
                student_vocab,
                temperature as f64,
                top_k,
            );
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1e-2),
                "top_k={top_k:?} T={temperature}: got {got}, float64 reference {want}"
            );
        }
    }
}

/// The student's gradient through the sort. (The permutation that sorts the
/// student is a function of the student; if it enters the autodiff graph, MLX
/// refuses to differentiate the gather, and no run could train.)
#[test]
#[serial]
fn uld_gradient_matches_finite_differences() {
    let teacher = array(&logits(0.3), Dtype::Float32);
    let s = logits(1.9);
    let shape = [1, TOKENS as i32, VOCAB as i32];
    let loss = UniversalLogitLoss::new();
    let f = |x: &Array| loss.compute(&teacher, x, 1.5).unwrap();

    let (value, mut grads) = value_and_grad_explicit(
        |a: &[Array]| f(&a[0]),
        &[Array::from_f32_slice(&s, &shape)],
        &[],
    )
    .unwrap();
    value.eval();
    grads[0].eval();
    drain();
    let grad = grads[0].to_f32_vec(s.len()).unwrap();

    let h = 2e-3_f32;
    for i in 0..s.len() {
        let (mut plus, mut minus) = (s.clone(), s.clone());
        plus[i] += h;
        minus[i] -= h;
        let fd = (scalar(&f(&Array::from_f32_slice(&plus, &shape)))
            - scalar(&f(&Array::from_f32_slice(&minus, &shape))))
            / (2.0 * h);
        assert!(
            (grad[i] - fd).abs() <= 2e-4 + 2e-2 * fd.abs(),
            "entry {i}: autograd {} vs finite differences {fd}",
            grad[i]
        );
    }
}

/// A logit regression never softens a distribution, so the temperature must
/// not change what the student is trained toward. `Distiller` multiplies a
/// softened divergence by `T²` to restore its gradient; doing the same to MSE
/// inflated it sixteenfold at `T = 4`.
#[test]
#[serial]
fn mse_distillation_is_independent_of_the_temperature() {
    use pmetal_distill::{DistillConfig, DistillMethod, Distiller, LossConfig, LossType};

    let loss_at = |temperature: f32| {
        let distiller = Distiller::new(DistillConfig {
            teacher: "t".to_string(),
            student: "s".to_string(),
            method: DistillMethod::Online,
            loss: LossConfig {
                loss_type: LossType::MseLoss,
                temperature,
                alpha: 1.0,
                ..LossConfig::default()
            },
            offline: None,
            output_path: None,
            training: Default::default(),
        })
        .unwrap();
        let teacher = array(&logits(0.3), Dtype::Float32);
        let student = array(&logits(1.9), Dtype::Float32);
        scalar(
            &distiller
                .compute_loss(&teacher, &student, None, None, 0, 1)
                .unwrap()
                .total,
        )
    };
    let (cold, hot) = (loss_at(1.0), loss_at(4.0));
    assert!(
        (cold - hot).abs() <= 1e-5 * cold.abs(),
        "MSE at T=1 is {cold}, at T=4 is {hot}"
    );
}

// ---------------------------------------------------------------------------
// Hidden-state alignment
// ---------------------------------------------------------------------------

const HIDDEN: usize = 6;

fn hidden(seed: f32, tokens: usize) -> Vec<f32> {
    (0..tokens * HIDDEN)
        .map(|i| 2.0 * ((i as f32) * 0.91 + seed).sin() + 0.1 * (i % 5) as f32)
        .collect()
}

fn hidden_array(data: &[f32], dtype: Dtype) -> Array {
    Array::from_f32_slice(data, &[1, (data.len() / HIDDEN) as i32, HIDDEN as i32])
        .as_dtype(dtype.as_i32())
}

/// Cosine loss on a zero hidden vector (a padded position, or a layer
/// that has not warmed up) must have a finite gradient. The norm was a bare
/// `sqrt` of the squared sum, whose derivative at zero is infinite and whose
/// product with the zero vector is NaN.
#[test]
#[serial]
fn cosine_hidden_loss_has_a_finite_gradient_at_zero_vectors() {
    let tokens = 3;
    let teacher = hidden(0.2, tokens);
    let mut student = hidden(1.4, tokens);
    // The middle student token is all zeros, and so is the first teacher token.
    student[HIDDEN..2 * HIDDEN].fill(0.0);
    let mut teacher = teacher;
    teacher[..HIDDEN].fill(0.0);

    let loss = HiddenStateLoss::cosine();
    let t = hidden_array(&teacher, Dtype::Float32);
    let (value, mut grads) = value_and_grad_explicit(
        |a: &[Array]| loss.compute(&t, &a[0]).unwrap(),
        &[hidden_array(&student, Dtype::Float32)],
        &[],
    )
    .unwrap();
    value.eval();
    grads[0].eval();
    drain();
    let grad = grads[0].to_f32_vec(student.len()).unwrap();
    assert!(value.item::<f32>().is_finite());
    assert!(
        grad.iter().all(|g| g.is_finite()),
        "gradient has non-finite entries: {grad:?}"
    );
}

/// Padded positions must not move the loss: with a 0/1 token mask, the
/// hidden-state losses average over the unmasked tokens only.
#[test]
#[serial]
fn hidden_state_losses_respect_the_token_mask() {
    let tokens = 4;
    let teacher = hidden(0.2, tokens);
    let student = hidden(1.4, tokens);
    let mask = Array::from_f32_slice(&[1.0_f32, 1.0, 0.0, 0.0], &[1, 4]);

    // The first two tokens only, as the reference.
    let head = 2 * HIDDEN;
    for (name, loss) in [
        ("mse", HiddenStateLoss::mse()),
        ("cosine", HiddenStateLoss::cosine()),
        ("l1", HiddenStateLoss::l1()),
    ] {
        let reference = scalar(
            &loss
                .compute(
                    &hidden_array(&teacher[..head], Dtype::Float32),
                    &hidden_array(&student[..head], Dtype::Float32),
                )
                .unwrap(),
        );
        // Garbage in the padded positions.
        let mut junk = student.clone();
        junk[head..].fill(50.0);
        let masked = scalar(
            &loss
                .compute_with_mask(
                    &hidden_array(&teacher, Dtype::Float32),
                    &hidden_array(&junk, Dtype::Float32),
                    Some(&mask),
                )
                .unwrap(),
        );
        assert!(
            (reference - masked).abs() <= 1e-5 * reference.abs().max(1e-3),
            "{name}: first two tokens {reference}, masked {masked}"
        );
    }
}

/// Hidden states are reduced in f32: a bf16 tensor's mean is not rounded to
/// bf16's three digits.
#[test]
#[serial]
fn bf16_hidden_states_are_reduced_in_f32() {
    let tokens = 64;
    let (t, s) = (hidden(0.2, tokens), hidden(1.4, tokens));
    let rounded = |x: &[f32]| -> Vec<f64> {
        hidden_array(x, Dtype::Bfloat16)
            .as_dtype(Dtype::Float32.as_i32())
            .to_f32_vec(x.len())
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    };
    let (t64, s64) = (rounded(&t), rounded(&s));
    let mse: f64 = t64
        .iter()
        .zip(&s64)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / t64.len() as f64;
    let got = scalar(
        &HiddenStateLoss::mse()
            .compute(
                &hidden_array(&t, Dtype::Bfloat16),
                &hidden_array(&s, Dtype::Bfloat16),
            )
            .unwrap(),
    ) as f64;
    assert!(
        (got - mse).abs() <= 1e-5 * mse,
        "bf16 hidden MSE {got} vs float64 {mse}"
    );
}

/// A layer pair beyond the layers the models returned is a configuration
/// error, not a pair to skip quietly.
#[test]
#[serial]
fn a_layer_pair_out_of_range_is_an_error() {
    use pmetal_distill::losses::hidden_state::LayerDistillation;

    let h = hidden_array(&hidden(0.2, 2), Dtype::Float32);
    let distill = LayerDistillation::new(vec![(0, 0), (5, 1)], HiddenStateLoss::mse(), 1.0);
    let err = distill
        .compute(std::slice::from_ref(&h), &[h.clone(), h.clone()])
        .expect_err("teacher layer 5 does not exist");
    assert!(err.to_string().contains("out of range"), "{err}");
}

/// A hidden-state loss handed to the builder is the one that runs, whatever
/// the config says; the config used to be consulted and the injected loss
/// ignored.
#[test]
#[serial]
fn the_injected_hidden_loss_is_the_one_used() {
    use pmetal_distill::{
        DistillConfig, DistillMethod, DistillerBuilder, HiddenStateConfig, HiddenStateLossType,
        LossConfig,
    };

    let config = DistillConfig {
        teacher: "t".to_string(),
        student: "s".to_string(),
        method: DistillMethod::Online,
        loss: LossConfig {
            hidden_state: Some(HiddenStateConfig {
                layer_mapping: vec![(0, 0)],
                loss_type: HiddenStateLossType::Mse,
                weight: 1.0,
                use_projection: false,
            }),
            ..LossConfig::default()
        },
        offline: None,
        output_path: None,
        training: Default::default(),
    };
    let distiller = DistillerBuilder::new()
        .with_config(config)
        .with_hidden_loss(HiddenStateLoss::l1())
        .build()
        .unwrap();

    let (t, s) = (hidden(0.2, 2), hidden(1.4, 2));
    let logits_t = array(&logits(0.3), Dtype::Float32);
    let logits_s = array(&logits(1.9), Dtype::Float32);
    let out = distiller
        .compute_loss_with_hidden(
            &logits_t,
            &logits_s,
            &[hidden_array(&t, Dtype::Float32)],
            &[hidden_array(&s, Dtype::Float32)],
            None,
            None,
            0,
            1,
        )
        .unwrap();
    let got = scalar(&out.hidden.expect("a hidden term")) as f64;
    let l1: f64 = t
        .iter()
        .zip(&s)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / t.len() as f64;
    assert!((got - l1).abs() <= 1e-5 * l1, "hidden term {got}, L1 {l1}");
}
