//! Benchmarks for the mHC mapping computations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array2, Array3};
use pmetal_mhc::{
    MhcConfig, MhcParams, MhcPreset, apply_post_res_mapping, apply_pre_mapping, compute_mappings,
    compute_mappings_backward,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::hint::black_box;

/// Tokens per call.
const TOKENS: usize = 16;

fn streams(config: &MhcConfig) -> Array3<f32> {
    let mut rng = StdRng::seed_from_u64(1);
    Array3::from_shape_fn((TOKENS, config.expansion_rate, config.hidden_dim), |_| {
        rng.random_range(-1.0f32..1.0)
    })
}

fn bench_compute_mappings(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mappings");

    for preset in [MhcPreset::Small, MhcPreset::Medium, MhcPreset::Large] {
        let config = MhcConfig::from_preset(preset);
        let params = MhcParams::with_rng(&config, &mut StdRng::seed_from_u64(0));
        let x = streams(&config);

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{preset:?}")),
            &(&config, &params, &x),
            |b, (config, params, x)| {
                b.iter(|| compute_mappings(black_box(x), black_box(params), black_box(config)));
            },
        );

        let n = config.expansion_rate;
        let g_pre = Array2::<f32>::ones((TOKENS, n));
        let g_post = Array2::<f32>::ones((TOKENS, n));
        let g_res = Array3::<f32>::ones((TOKENS, n, n));
        group.bench_with_input(
            BenchmarkId::new("backward", format!("{preset:?}")),
            &(&config, &params, &x),
            |b, (config, params, x)| {
                b.iter(|| {
                    compute_mappings_backward(
                        black_box(x),
                        black_box(params),
                        black_box(config),
                        &g_pre,
                        &g_post,
                        &g_res,
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_apply_mappings(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_mappings");

    let config = MhcConfig::from_preset(MhcPreset::Medium);
    let params = MhcParams::with_rng(&config, &mut StdRng::seed_from_u64(0));
    let hidden_dim = config.hidden_dim;

    let x = streams(&config);
    let mappings = compute_mappings(&x, &params, &config);

    group.bench_with_input(
        BenchmarkId::new("pre_mapping", hidden_dim),
        &(&x, &mappings),
        |b, (x, m)| {
            b.iter(|| apply_pre_mapping(black_box(x), black_box(&m.h_pre)));
        },
    );

    let h_out = Array2::<f32>::ones((TOKENS, hidden_dim));
    group.bench_with_input(
        BenchmarkId::new("post_res_mapping", hidden_dim),
        &(&x, &h_out, &mappings),
        |b, (x, h_out, m)| {
            b.iter(|| {
                apply_post_res_mapping(
                    black_box(x),
                    black_box(h_out),
                    black_box(&m.h_post),
                    black_box(&m.h_res),
                )
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_compute_mappings, bench_apply_mappings);
criterion_main!(benches);
