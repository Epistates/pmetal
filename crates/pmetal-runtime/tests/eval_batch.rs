//! `compat::eval` submits a set of arrays as one evaluation.
//!
//! The correctness tests run everywhere; the timing comparison is ignored:
//! `cargo test --release -p pmetal-runtime --test eval_batch -- --ignored --nocapture`

use pmetal_runtime::InlineArray;
use pmetal_runtime::compat::{self, Dtype, transforms};
use std::time::Instant;

/// A small independent graph per index, so every array is its own unit of work.
fn graphs(n: usize, dim: i32) -> Vec<InlineArray> {
    (0..n)
        .map(|i| {
            let x = InlineArray::ones(&[dim, dim], Dtype::Float32).mul_scalar(i as f32 + 1.0);
            x.matmul(&x).add(&x)
        })
        .collect()
}

#[test]
fn batched_eval_matches_one_at_a_time() {
    let one_by_one = graphs(9, 16);
    for a in &one_by_one {
        a.eval().expect("eval");
    }
    let batched = graphs(9, 16);
    transforms::eval(&batched).expect("batched eval");
    compat::eval(&batched).expect("re-eval of evaluated arrays");

    for (a, b) in one_by_one.iter().zip(&batched) {
        assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
    }
}

#[test]
fn batched_eval_accepts_empty_and_repeated_arrays() {
    transforms::eval(std::iter::empty()).expect("empty set");
    let a = graphs(1, 8).remove(0);
    transforms::eval([&a, &a, &a]).expect("repeats");
    assert_eq!(a.as_slice::<f32>().len(), 64);
}

#[test]
#[ignore = "timing comparison; run with --release --nocapture"]
fn batched_eval_timing() {
    let (n, dim, reps) = (600, 64, 20);
    let time = |label: &str, run: &dyn Fn(&[InlineArray])| {
        let mut ms = Vec::new();
        for _ in 0..reps + 2 {
            let set = graphs(n, dim);
            let t = Instant::now();
            run(&set);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.drain(..2);
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("{label}: median {:.2} ms for {n} arrays", ms[ms.len() / 2]);
    };
    time("one eval per array", &|set| {
        for a in set {
            a.eval().expect("eval");
        }
    });
    time("compat::eval (set)   ", &|set| {
        compat::eval(set).expect("eval");
    });
}
