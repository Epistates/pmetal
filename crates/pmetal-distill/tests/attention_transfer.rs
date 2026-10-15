//! Attention-map transfer losses against float64, and their place in `Distiller`.

use pmetal_distill::losses::AttentionTransferLoss;
use pmetal_distill::{
    AttentionConfig, AttentionHeadReduction, AttentionLossType, DistillConfig, DistillMethod,
    Distiller, LossConfig,
};
use pmetal_runtime::compat::nn::value_and_grad_explicit;
use pmetal_runtime::compat::{Array, Dtype};
use serial_test::serial;

const B: usize = 2;
const H: usize = 3;
const Q: usize = 4;
const K: usize = 4;
const EPS: f64 = 1e-8;

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

/// Row-stochastic attention maps `[B, H, Q, K]` (causal: query `q` sees keys
/// `<= q`), as f32 values.
fn attention(seed: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; B * H * Q * K];
    for row in 0..B * H * Q {
        let q = row % Q;
        let scores: Vec<f64> = (0..=q)
            .map(|k| 2.0 * ((row * K + k) as f64 * 0.91 + seed as f64).sin())
            .collect();
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = scores.iter().map(|s| (s - max).exp()).sum();
        for (k, s) in scores.iter().enumerate() {
            out[row * K + k] = ((s - max).exp() / z) as f32;
        }
    }
    out
}

fn array(x: &[f32], dtype: Dtype) -> Array {
    Array::from_f32_slice(x, &[B as i32, H as i32, Q as i32, K as i32]).as_dtype(dtype.as_i32())
}

fn rounded(x: &[f32], dtype: Dtype) -> Vec<f64> {
    array(x, dtype)
        .as_dtype(Dtype::Float32.as_i32())
        .to_f32_vec(x.len())
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

fn scalar(a: &Array) -> f64 {
    a.eval();
    drain();
    a.item::<f32>() as f64
}

fn mse(t: &[f64], s: &[f64]) -> f64 {
    t.iter().zip(s).map(|(a, b)| (b - a) * (b - a)).sum::<f64>() / t.len() as f64
}

fn kl(t: &[f64], s: &[f64]) -> f64 {
    let rows = t.len() / K;
    t.chunks(K)
        .zip(s.chunks(K))
        .map(|(t, s)| {
            t.iter()
                .zip(s)
                .map(|(a, b)| a * ((a + EPS).ln() - (b + EPS).ln()))
                .sum::<f64>()
        })
        .sum::<f64>()
        / rows as f64
}

/// Zagoruyko & Komodakis: sum over heads, L2-normalise each row, mean squared
/// difference.
fn attention_transfer(t: &[f64], s: &[f64]) -> f64 {
    let summed = |x: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; B * Q * K];
        for b in 0..B {
            for h in 0..H {
                for qk in 0..Q * K {
                    out[b * Q * K + qk] += x[(b * H + h) * Q * K + qk];
                }
            }
        }
        out.chunks(K)
            .flat_map(|row| {
                let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                row.iter()
                    .map(move |v| v / (norm + EPS))
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    mse(&summed(t), &summed(s))
}

#[test]
#[serial]
fn bf16_attention_maps_are_reduced_in_f32() {
    let (t, s) = (attention(0.3), attention(1.7));
    let (t64, s64) = (rounded(&t, Dtype::Bfloat16), rounded(&s, Dtype::Bfloat16));
    let cases: [(&str, AttentionTransferLoss, fn(&[f64], &[f64]) -> f64); 3] = [
        ("mse", AttentionTransferLoss::mse(), mse),
        ("kl", AttentionTransferLoss::kl(), kl),
        (
            "attention_transfer",
            AttentionTransferLoss::attention_transfer(),
            attention_transfer,
        ),
    ];
    for (name, loss, reference) in cases {
        let got = scalar(
            &loss
                .compute(&array(&t, Dtype::Bfloat16), &array(&s, Dtype::Bfloat16))
                .unwrap(),
        );
        let want = reference(&t64, &s64);
        assert!(
            (got - want).abs() <= 2e-5 * want.abs().max(1e-4),
            "{name}: got {got}, float64 reference {want}"
        );
    }
}

#[test]
#[serial]
fn the_teacher_attention_receives_no_gradient() {
    let (t, s) = (attention(0.3), attention(1.7));
    for (name, loss) in [
        ("mse", AttentionTransferLoss::mse()),
        ("kl", AttentionTransferLoss::kl()),
        ("at", AttentionTransferLoss::attention_transfer()),
    ] {
        let (value, mut grads) = value_and_grad_explicit(
            |a: &[Array]| loss.compute(&a[0], &a[1]).unwrap(),
            &[array(&t, Dtype::Float32), array(&s, Dtype::Float32)],
            &[],
        )
        .unwrap();
        value.eval();
        grads[0].eval();
        grads[1].eval();
        drain();
        let n = t.len();
        let g_teacher = grads[0].to_f32_vec(n).unwrap();
        let g_student = grads[1].to_f32_vec(n).unwrap();
        assert!(
            g_student.iter().any(|g| g.abs() > 1e-9),
            "{name}: no student gradient"
        );
        assert!(
            g_teacher.iter().all(|g| *g == 0.0),
            "{name}: teacher got gradient"
        );
    }
}

fn distiller(attention: Option<AttentionConfig>) -> Distiller {
    Distiller::new(DistillConfig {
        teacher: "t".to_string(),
        student: "s".to_string(),
        method: DistillMethod::Online,
        loss: LossConfig {
            attention,
            ..LossConfig::default()
        },
        offline: None,
        output_path: None,
        training: Default::default(),
    })
    .unwrap()
}

/// The configured attention term: the mean over the mapped layer pairs of the
/// chosen loss, times the weight; none without a configuration.
#[test]
#[serial]
fn the_distiller_computes_the_configured_attention_term() {
    let layers_t: Vec<Vec<f32>> = (0..3).map(|i| attention(0.2 * i as f32)).collect();
    let layers_s: Vec<Vec<f32>> = (0..2).map(|i| attention(1.0 + 0.4 * i as f32)).collect();
    let teacher: Vec<Array> = layers_t.iter().map(|x| array(x, Dtype::Float32)).collect();
    let student: Vec<Array> = layers_s.iter().map(|x| array(x, Dtype::Float32)).collect();

    assert!(
        distiller(None)
            .compute_attention_loss(&teacher, &student)
            .unwrap()
            .is_none()
    );

    let config = AttentionConfig {
        layer_mapping: vec![(0, 0), (2, 1)],
        loss_type: AttentionLossType::Mse,
        weight: 0.5,
        head_reduction: AttentionHeadReduction::Exact,
    };
    let got = scalar(
        &distiller(Some(config))
            .compute_attention_loss(&teacher, &student)
            .unwrap()
            .expect("an attention term"),
    );
    let pair = |t: usize, s: usize| {
        mse(
            &rounded(&layers_t[t], Dtype::Float32),
            &rounded(&layers_s[s], Dtype::Float32),
        )
    };
    let want = 0.5 * (pair(0, 0) + pair(2, 1)) / 2.0;
    assert!(
        (got - want).abs() <= 2e-5 * want,
        "got {got}, expected {want}"
    );

    // A layer the model did not return.
    let bad = AttentionConfig {
        layer_mapping: vec![(7, 0)],
        loss_type: AttentionLossType::Mse,
        weight: 1.0,
        head_reduction: AttentionHeadReduction::Exact,
    };
    assert!(
        distiller(Some(bad))
            .compute_attention_loss(&teacher, &student)
            .is_err()
    );
}
