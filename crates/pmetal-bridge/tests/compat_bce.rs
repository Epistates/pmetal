//! `compat::losses::BinaryCrossEntropy` is MLX's
//! `nn.losses.binary_cross_entropy`: per-element BCE, independent of the
//! other elements. Expected values are written out from the definition.

use pmetal_bridge::check_last_error;
use pmetal_bridge::compat::Array;
use pmetal_bridge::compat::losses::{BinaryCrossEntropyBuilder, LossReduction};

fn read(a: &Array) -> Vec<f32> {
    let mut a = a.clone();
    a.eval();
    let n = a.size();
    let v = a.to_f32_vec(n).expect("to_f32_vec");
    check_last_error().expect("bridge error");
    v
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-5 * w.abs().max(1.0);
        assert!((g as f64 - w).abs() < tol, "{what}[{i}]: got {g}, want {w}");
    }
}

/// `softplus(x) - x·t` in f64.
fn bce_logits(x: f64, t: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p() - x * t
}

#[test]
fn bce_with_logits_is_per_element_and_does_not_overflow() {
    let x = [0.0, 2.0, -3.0, 50.0, -200.0];
    let t = [1.0, 0.0, 1.0, 0.0, 1.0];
    let logits = Array::from_f32_slice(&x.map(|v| v as f32), &[5]);
    let targets = Array::from_f32_slice(&t.map(|v| v as f32), &[5]);
    let want: Vec<f64> = x.iter().zip(&t).map(|(&x, &t)| bce_logits(x, t)).collect();

    let none = BinaryCrossEntropyBuilder::new()
        .with_logits(true)
        .reduction(LossReduction::None)
        .build()
        .call(&logits, &targets);
    assert_close(&read(&none), &want, "bce none");

    let mean = BinaryCrossEntropyBuilder::new()
        .with_logits(true)
        .reduction(LossReduction::Mean)
        .build()
        .call(&logits, &targets);
    let want_mean = want.iter().sum::<f64>() / want.len() as f64;
    assert_close(&read(&mean), &[want_mean], "bce mean");
}

#[test]
fn bce_on_probabilities_clips_the_log_at_minus_100() {
    let p = Array::from_f32_slice(&[0.5, 0.0, 0.9], &[3]);
    let t = Array::from_f32_slice(&[1.0, 1.0, 0.0], &[3]);
    let loss = BinaryCrossEntropyBuilder::new()
        .with_logits(false)
        .reduction(LossReduction::None)
        .build()
        .call(&p, &t);
    // -ln 0.5, -max(ln 0, -100), -ln(1 - 0.9)
    let want = [std::f64::consts::LN_2, 100.0, -(0.1f32 as f64).ln()];
    assert_close(&read(&loss), &want, "bce probabilities");
}
