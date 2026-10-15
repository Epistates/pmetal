//! Benchmarks of the merge primitives and methods.
//!
//! Run with: cargo bench -p pmetal-merge
//!
//! Every iteration ends in a scalar readback of the result, so the timings
//! cover the evaluated computation and not just MLX graph construction.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pmetal_merge::{
    LinearMerge, MergeMethod, MergeParameters, SignElection, SlerpMerge, TiesMerge,
    disjoint_merge, sparsify_by_magnitude,
};
use pmetal_runtime::compat::Array;
use std::hint::black_box;

/// Deterministic pseudo-random test data in roughly [-10, 10].
fn generate_test_data(size: usize, salt: usize) -> Array {
    let data: Vec<f32> = (0..size)
        .map(|i| ((i + salt * 7919) as f32 * 1.234567).sin() * 10.0)
        .collect();
    Array::from_slice(&data, &[size as i32])
}

/// Force evaluation of `result`.
fn force(result: &Array) -> f32 {
    result.sum_all().item_f32()
}

fn params(n: usize, density: f32) -> Vec<MergeParameters> {
    (0..n)
        .map(|_| MergeParameters {
            weight: Some(1.0_f32.into()),
            density: Some(density.into()),
            ..Default::default()
        })
        .collect()
}

fn bench_sparsification(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsification");
    for size in [1024, 16384, 262144, 1048576] {
        let tensor = generate_test_data(size, 0);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("top_half", size), &size, |b, _| {
            b.iter(|| {
                let out = sparsify_by_magnitude(black_box(&tensor), black_box(0.5)).unwrap();
                force(&out)
            });
        });
    }
    group.finish();
}

fn bench_disjoint_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("disjoint_merge");
    let size = 65536;
    for num_models in [2, 4, 8] {
        let tensors: Vec<Array> = (0..num_models)
            .map(|i| generate_test_data(size, i))
            .collect();
        let weights = vec![1.0; num_models];
        group.throughput(Throughput::Elements((size * num_models) as u64));
        group.bench_with_input(BenchmarkId::new("models", num_models), &num_models, |b, _| {
            b.iter(|| {
                let out = disjoint_merge(
                    black_box(&tensors),
                    black_box(&weights),
                    SignElection::Mass,
                    true,
                )
                .unwrap();
                force(&out)
            });
        });
    }
    group.finish();
}

fn bench_ties_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("ties_merge");
    let method = TiesMerge::new();
    let global = MergeParameters::default();
    for size in [16384, 65536, 262144] {
        let base = generate_test_data(size, 0);
        let tensors = [generate_test_data(size, 1), generate_test_data(size, 2)];
        let params = params(2, 0.5);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("two_models", size), &size, |b, _| {
            b.iter(|| {
                let out = method
                    .merge(black_box(&tensors), Some(&base), &params, &global)
                    .unwrap();
                force(&out)
            });
        });
    }
    group.finish();
}

fn bench_linear_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_merge");
    let method = LinearMerge::new();
    let global = MergeParameters::default();
    let size = 65536;
    for num_models in [2, 4, 8] {
        let tensors: Vec<Array> = (0..num_models)
            .map(|i| generate_test_data(size, i))
            .collect();
        let params = params(num_models, 1.0);
        group.throughput(Throughput::Elements((size * num_models) as u64));
        group.bench_with_input(BenchmarkId::new("models", num_models), &num_models, |b, _| {
            b.iter(|| {
                let out = method
                    .merge(black_box(&tensors), None, &params, &global)
                    .unwrap();
                force(&out)
            });
        });
    }
    group.finish();
}

fn bench_slerp_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("slerp_merge");
    let method = SlerpMerge::new();
    let global = MergeParameters::default();
    for size in [16384, 65536, 262144] {
        let tensors = [generate_test_data(size, 1), generate_test_data(size, 2)];
        let params = params(2, 1.0);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &size, |b, _| {
            b.iter(|| {
                let out = method
                    .merge(black_box(&tensors), None, &params, &global)
                    .unwrap();
                force(&out)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sparsification,
    bench_disjoint_merge,
    bench_ties_merge,
    bench_linear_merge,
    bench_slerp_merge,
);
criterion_main!(benches);
