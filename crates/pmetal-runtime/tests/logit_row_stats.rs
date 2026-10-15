//! `logit_row_stats`: the per-row logsumexp, target logit and logit sum of a
//! logit matrix, from one Metal pass, with the logits' gradient from another.
//! Each is checked against its definition in f64.

use pmetal_runtime::InlineArray;
use pmetal_runtime::check_unobserved_error;
use pmetal_runtime::compat::{Dtype, random};
use pmetal_runtime::inline_array::{logit_row_stats, value_and_grad};

fn read(a: &InlineArray) -> Vec<f32> {
    let mut a = a.as_dtype(Dtype::Float32.as_i32());
    a.eval();
    let n = a.size();
    let v = a.to_f32_vec(n).expect("readable");
    check_unobserved_error().expect("no bridge error");
    v
}

/// Logits of `dtype` and the f32 values they hold.
fn logits(n: usize, v: usize, dtype: Dtype, spread: f32) -> (InlineArray, Vec<f32>) {
    let z = random::normal(&[n as i32, v as i32], Dtype::Float32)
        .mul_scalar(spread)
        .as_dtype(dtype.as_i32());
    let values = read(&z);
    (z, values)
}

fn targets_for(n: usize, v: usize) -> Vec<i32> {
    (0..n as i32).map(|i| (i * 7919 + 3) % v as i32).collect()
}

struct Row {
    lse: f64,
    softmax: Vec<f64>,
}

fn row(values: &[f32]) -> Row {
    let m = values.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = values.iter().map(|z| (*z as f64 - m).exp()).sum();
    let lse = sum.ln() + m;
    Row {
        lse,
        softmax: values.iter().map(|z| (*z as f64 - lse).exp()).collect(),
    }
}

/// Vocabularies that don't divide the 256 threads, nor by 4 or 32, and one
/// that is a row's worth of a real model.
const SHAPES: [(usize, usize); 4] = [(5, 1), (7, 1000), (9, 32_001), (3, 151_936)];

#[test]
fn the_statistics_match_their_definitions_in_every_dtype() {
    random::seed(11);
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for (n, v) in SHAPES {
            let (z, values) = logits(n, v, dtype, 2.5);
            let targets = targets_for(n, v);
            let t = InlineArray::from_i32_slice_shaped(&targets, &[n as i32]);
            let stats = logit_row_stats(&z, &t);
            let (lse, target, sum) = (
                read(&stats.lse),
                read(&stats.target_logit),
                read(&stats.logit_sum),
            );
            for i in 0..n {
                let r = &values[i * v..(i + 1) * v];
                let want_lse = row(r).lse;
                let want_sum: f64 = r.iter().map(|z| *z as f64).sum();
                assert!(
                    (lse[i] as f64 - want_lse).abs() < 2e-5 * (1.0 + want_lse.abs()),
                    "{dtype:?} {n}x{v} row {i}: lse {} vs {want_lse}",
                    lse[i]
                );
                assert_eq!(
                    target[i], r[targets[i] as usize],
                    "{dtype:?} {n}x{v} row {i}"
                );
                assert!(
                    (sum[i] as f64 - want_sum).abs() < 1e-3 * (1.0 + want_sum.abs().max(v as f64)),
                    "{dtype:?} {n}x{v} row {i}: sum {} vs {want_sum}",
                    sum[i]
                );
            }
        }
    }
}

/// The gradient of `a·lse + b·z[target] + c·Σz` is
/// `a·softmax + b·onehot + c`, in the logits' dtype, for any cotangents.
#[test]
fn the_logits_gradient_is_softmax_plus_onehot_plus_one() {
    random::seed(12);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for (n, v) in [(6, 1000), (4, 32_001)] {
            let (z, values) = logits(n, v, dtype, 2.0);
            let targets = targets_for(n, v);
            let t = InlineArray::from_i32_slice_shaped(&targets, &[n as i32]);
            let a: Vec<f32> = (0..n).map(|i| 0.5 + 0.25 * i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| -1.0 + 0.5 * i as f32).collect();
            let c: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 - 2.0)).collect();
            let (a_arr, b_arr, c_arr) = (
                InlineArray::from_f32_slice(&a, &[n as i32]),
                InlineArray::from_f32_slice(&b, &[n as i32]),
                InlineArray::from_f32_slice(&c, &[n as i32]),
            );
            let (_, grads) = value_and_grad(
                |p| {
                    let s = logit_row_stats(&p[0], &p[1]);
                    s.lse
                        .multiply(&p[2])
                        .add(&s.target_logit.multiply(&p[3]))
                        .add(&s.logit_sum.multiply(&p[4]))
                        .sum_all()
                },
                std::slice::from_ref(&z),
                &[t, a_arr, b_arr, c_arr],
            );
            assert_eq!(
                grads[0].dtype(),
                dtype,
                "the gradient takes the logits' dtype"
            );
            let got = read(&grads[0]);
            // bf16 rounds the gradient once: 2^-8 relative.
            let tol = if dtype == Dtype::Bfloat16 { 8e-3 } else { 1e-5 };
            for i in 0..n {
                let softmax = row(&values[i * v..(i + 1) * v]).softmax;
                for j in 0..v {
                    let onehot = if j == targets[i] as usize { 1.0 } else { 0.0 };
                    let want = a[i] as f64 * softmax[j] + b[i] as f64 * onehot + c[i] as f64;
                    let g = got[i * v + j] as f64;
                    assert!(
                        (g - want).abs() <= tol * want.abs() + 1e-7,
                        "{dtype:?} {n}x{v} d/dz[{i}][{j}]: {g} vs {want}"
                    );
                }
            }
        }
    }
}

/// A target outside the vocabulary has no logit and no one-hot, which is how
/// an ignored label reaches the kernel if a caller doesn't mask it first.
#[test]
fn a_target_outside_the_vocabulary_picks_nothing() {
    let z = InlineArray::from_f32_slice(&[0.5, 1.5, -2.0, 4.0, 3.0, 0.0, 1.0, 2.0], &[2, 4]);
    for bad in [-100, -1, 4, 99] {
        let t = InlineArray::from_i32_slice_shaped(&[bad, 2], &[2]);
        let (_, grads) = value_and_grad(
            |p| {
                let s = logit_row_stats(&p[0], &p[1]);
                s.target_logit.sum_all()
            },
            std::slice::from_ref(&z),
            std::slice::from_ref(&t),
        );
        let stats = logit_row_stats(&z, &t);
        let target = read(&stats.target_logit);
        assert_eq!(target[0], 0.0, "target {bad}");
        assert_eq!(
            target[1],
            0.0 + 1.0,
            "the in-range row still reads its logit"
        );
        let g = read(&grads[0]);
        assert!(
            g[..4].iter().all(|x| *x == 0.0),
            "target {bad}: {:?}",
            &g[..4]
        );
        assert_eq!(&g[4..], &[0.0, 0.0, 1.0, 0.0]);
    }
}

/// A second derivative goes through MLX ops, which the kernel has no VJP to
/// stand in for: `d/dz Σ ‖∂lse/∂z‖²` against the same computed from `logsumexp`.
#[test]
fn a_second_derivative_matches_the_op_built_one() {
    random::seed(13);
    let (n, v) = (3, 50);
    let (z, _) = logits(n, v, Dtype::Float32, 1.5);
    let t = InlineArray::from_i32_slice_shaped(&targets_for(n, v), &[n as i32]);

    let via_kernel = |z: &InlineArray| {
        let (_, g) = value_and_grad(
            |q| logit_row_stats(&q[0], &q[1]).lse.sum_all(),
            std::slice::from_ref(z),
            std::slice::from_ref(&t),
        );
        g[0].square().sum_all()
    };
    let via_ops = |z: &InlineArray| {
        let (_, g) = value_and_grad(
            |q| q[0].logsumexp(-1, false).sum_all(),
            std::slice::from_ref(z),
            &[],
        );
        g[0].square().sum_all()
    };
    let (_, want) = value_and_grad(|p| via_ops(&p[0]), std::slice::from_ref(&z), &[]);
    let (_, got) = value_and_grad(|p| via_kernel(&p[0]), std::slice::from_ref(&z), &[]);
    let (want, got) = (read(&want[0]), read(&got[0]));
    assert!(
        want.iter().any(|w| w.abs() > 1e-3),
        "a vanishing reference proves nothing"
    );
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!((w - g).abs() < 1e-5 + 1e-4 * w.abs(), "d2[{i}]: {g} vs {w}");
    }
}

#[test]
fn a_mismatched_target_count_is_an_error_not_a_crash() {
    let z = InlineArray::from_f32_slice(&[0.0; 12], &[3, 4]);
    let t = InlineArray::from_i32_slice_shaped(&[0, 1], &[2]);
    pmetal_runtime::expect_runtime_errors(|| {
        let _ = logit_row_stats(&z, &t);
        assert!(
            pmetal_runtime::check_last_error().is_err(),
            "two targets for three rows"
        );
    });
}
