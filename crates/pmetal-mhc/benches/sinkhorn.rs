//! Benchmarks for Sinkhorn-Knopp algorithm.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use pmetal_mhc::{sinkhorn_knopp, SinkhornConfig};

fn bench_sinkhorn_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinkhorn_knopp");
    let config = SinkhornConfig::default();

    for n in [4, 8, 16, 32] {
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, &n| {
            let mut matrix =
                Array2::<f32>::from_shape_fn((n, n), |(i, j)| ((i + j) as f32 * 0.1).exp());
            b.iter(|| {
                sinkhorn_knopp(black_box(&mut matrix), &config);
            });
        });
    }

    group.finish();
}

fn bench_sinkhorn_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinkhorn_iterations");
    let n = 4;

    for iters in [5, 10, 20, 50] {
        let config = SinkhornConfig {
            max_iterations: iters,
            ..SinkhornConfig::default()
        };

        group.bench_with_input(BenchmarkId::new("iters", iters), &iters, |b, _| {
            let mut matrix =
                Array2::<f32>::from_shape_fn((n, n), |(i, j)| ((i + j) as f32 * 0.1).exp());
            b.iter(|| {
                sinkhorn_knopp(black_box(&mut matrix), &config);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sinkhorn_sizes, bench_sinkhorn_iterations);
criterion_main!(benches);
