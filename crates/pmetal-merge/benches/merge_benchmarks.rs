//! Benchmarks comparing original vs optimized merge implementations.
//!
//! Run with: cargo bench -p pmetal-merge

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mlx_rs::Array;
use pmetal_merge::{
    gpu_merge::GpuMerger, sign_consensus, sparsify_batch_by_magnitude, sparsify_by_magnitude,
    sparsify_by_magnitude_online,
};

/// Generate random test data.
fn generate_test_data(size: usize) -> Array {
    // Create deterministic test data
    let data: Vec<f32> = (0..size)
        .map(|i| (i as f32 * 1.234567).sin() * 10.0)
        .collect();
    Array::from_slice(&data, &[size as i32])
}

/// Benchmark standard sparsification vs online (quickselect) sparsification.
fn bench_sparsification(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsification");

    for size in [1024, 16384, 262144, 1048576].iter() {
        let tensor = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("standard", size), size, |b, _| {
            b.iter(|| {
                let _ = sparsify_by_magnitude(black_box(&tensor), black_box(0.5));
            });
        });

        group.bench_with_input(
            BenchmarkId::new("online_quickselect", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _ = sparsify_by_magnitude_online(black_box(&tensor), black_box(0.5));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch sparsification.
fn bench_batch_sparsification(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sparsification");

    for num_tensors in [2, 4, 8].iter() {
        let size = 65536;
        let tensors: Vec<Array> = (0..*num_tensors)
            .map(|_| generate_test_data(size))
            .collect();
        let densities: Vec<f32> = vec![0.5; *num_tensors];

        group.throughput(Throughput::Elements((size * *num_tensors) as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential", num_tensors),
            num_tensors,
            |b, _| {
                b.iter(|| {
                    let _: Vec<_> = tensors
                        .iter()
                        .zip(densities.iter())
                        .map(|(t, &d)| sparsify_by_magnitude(t, d).unwrap())
                        .collect();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batched_online", num_tensors),
            num_tensors,
            |b, _| {
                b.iter(|| {
                    let _ = sparsify_batch_by_magnitude(black_box(&tensors), black_box(&densities));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sign consensus computation.
fn bench_sign_consensus(c: &mut Criterion) {
    let mut group = c.benchmark_group("sign_consensus");

    for num_models in [2, 4, 8].iter() {
        let size = 65536;
        let tensors: Vec<Array> = (0..*num_models)
            .map(|i| {
                let data: Vec<f32> = (0..size)
                    .map(|j| {
                        let x = ((i * size + j) as f32 * 1.234567).sin() * 10.0;
                        if x > 0.0 {
                            x
                        } else {
                            -x
                        }
                    })
                    .collect();
                Array::from_slice(&data, &[size as i32])
            })
            .collect();
        let weights: Vec<f32> = vec![1.0 / *num_models as f32; *num_models];

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("models", num_models),
            num_models,
            |b, _| {
                b.iter(|| {
                    let _ = sign_consensus(black_box(&tensors), black_box(&weights));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full TIES merge pipeline.
fn bench_ties_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("ties_merge");

    let merger = GpuMerger::new().unwrap();

    for size in [16384, 65536, 262144].iter() {
        let base = generate_test_data(*size);
        let t1 = generate_test_data(*size);
        let t2 = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("gpu_accelerated", size), size, |b, _| {
            b.iter(|| {
                let _ = merger.ties_merge(
                    black_box(&[t1.clone(), t2.clone()]),
                    black_box(&base),
                    black_box(&[0.5, 0.5]),
                    black_box(&[0.5, 0.5]),
                    black_box(1.0),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark linear merge.
fn bench_linear_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_merge");

    let merger = GpuMerger::new().unwrap();

    for num_models in [2, 4, 8].iter() {
        let size = 65536;
        let tensors: Vec<Array> = (0..*num_models).map(|_| generate_test_data(size)).collect();
        let weights: Vec<f32> = vec![1.0 / *num_models as f32; *num_models];

        group.throughput(Throughput::Elements((size * *num_models) as u64));

        group.bench_with_input(
            BenchmarkId::new("models", num_models),
            num_models,
            |b, _| {
                b.iter(|| {
                    let _ = merger.linear_merge(black_box(&tensors), black_box(&weights));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SLERP merge.
fn bench_slerp_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("slerp_merge");

    let merger = GpuMerger::new().unwrap();

    for size in [16384, 65536, 262144].iter() {
        let t1 = generate_test_data(*size);
        let t2 = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("size", size), size, |b, _| {
            b.iter(|| {
                let _ = merger.slerp_merge(black_box(&t1), black_box(&t2), black_box(0.5));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sparsification,
    bench_batch_sparsification,
    bench_sign_consensus,
    bench_ties_merge,
    bench_linear_merge,
    bench_slerp_merge,
);
criterion_main!(benches);
