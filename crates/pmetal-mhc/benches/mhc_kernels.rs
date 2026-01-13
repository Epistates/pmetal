//! Benchmarks for mHC Metal kernels.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array3;
use pmetal_mhc::{
    apply_post_res_mapping, apply_pre_mapping, compute_mappings, MhcConfig, MhcParams, MhcPreset,
};

fn bench_compute_mappings(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mappings");

    for preset in [MhcPreset::Small, MhcPreset::Medium, MhcPreset::Large] {
        let config = MhcConfig::from_preset(preset);
        let params = MhcParams::new(&config);
        let x = Array3::zeros((4, config.expansion_rate, config.hidden_dim));

        group.bench_with_input(
            BenchmarkId::new("preset", format!("{:?}", preset)),
            &(config, params, x),
            |b, (config, params, x)| {
                b.iter(|| {
                    compute_mappings(black_box(x), black_box(params), black_box(config));
                });
            },
        );
    }

    group.finish();
}

fn bench_apply_mappings(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_mappings");

    let config = MhcConfig::from_preset(MhcPreset::Medium);
    let params = MhcParams::new(&config);
    let n = config.expansion_rate;
    let hidden_dim = config.hidden_dim;

    // Create input tensor
    let x = Array3::<f32>::zeros((4, n, hidden_dim));
    let mappings = compute_mappings(&x, &params, &config);

    group.bench_with_input(
        BenchmarkId::new("pre_mapping", hidden_dim),
        &(x.clone(), mappings.clone()),
        |b, (x, m)| {
            b.iter(|| {
                apply_pre_mapping(black_box(x), black_box(&m.h_pre));
            });
        },
    );

    let h_out = ndarray::Array2::<f32>::zeros((4, hidden_dim));
    group.bench_with_input(
        BenchmarkId::new("post_res_mapping", hidden_dim),
        &(x.clone(), h_out.clone(), mappings.clone()),
        |b, (x, h_out, m)| {
            b.iter(|| {
                apply_post_res_mapping(
                    black_box(x),
                    black_box(h_out),
                    black_box(&m.h_post),
                    black_box(&m.h_res),
                );
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_compute_mappings, bench_apply_mappings);
criterion_main!(benches);
