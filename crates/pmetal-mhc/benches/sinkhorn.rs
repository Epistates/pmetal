//! Benchmarks for Sinkhorn-Knopp algorithm.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use pmetal_mhc::{SinkhornConfig, sinkhorn_knopp, sinkhorn_knopp_backward};
use std::hint::black_box;

fn logits(n: usize) -> Array2<f32> {
    Array2::from_shape_fn((n, n), |(i, j)| ((i * 7 + j * 3) % 11) as f32 * 0.2 - 1.0)
}

fn bench_sinkhorn_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinkhorn_knopp");
    let config = SinkhornConfig::default();

    for n in [4, 8, 16, 32] {
        let matrix = logits(n);
        group.bench_with_input(BenchmarkId::new("forward_n", n), &n, |b, _| {
            b.iter(|| sinkhorn_knopp(black_box(&matrix), &config));
        });
        let grad = Array2::<f32>::ones((n, n));
        group.bench_with_input(BenchmarkId::new("backward_n", n), &n, |b, _| {
            b.iter(|| sinkhorn_knopp_backward(black_box(&matrix), black_box(&grad), &config));
        });
    }

    group.finish();
}

fn bench_sinkhorn_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinkhorn_iterations");
    let matrix = logits(4);

    for iterations in [5, 10, 20, 50] {
        let config = SinkhornConfig {
            iterations,
            ..SinkhornConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::new("iters", iterations),
            &iterations,
            |b, _| {
                b.iter(|| sinkhorn_knopp(black_box(&matrix), &config));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sinkhorn_sizes, bench_sinkhorn_iterations);
criterion_main!(benches);
