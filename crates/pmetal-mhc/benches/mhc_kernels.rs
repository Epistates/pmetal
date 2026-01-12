//! Benchmarks for mHC Metal kernels.

use pmetal_mhc::{
    apply_post_res_mapping, apply_pre_mapping, compute_mappings, MhcConfig, MhcParams, MhcPreset,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

fn bench_compute_mappings(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mappings");

    for preset in [MhcPreset::Small, MhcPreset::Medium, MhcPreset::Large] {
        let config = MhcConfig::from_preset(preset);
        let params = MhcParams::new(&config);

        group.bench_with_input(
            BenchmarkId::new("preset", format!("{:?}", preset)),
            &(config, params),
            |b, (config, params)| {
                b.iter(|| {
                    compute_mappings(black_box(params), black_box(config));
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
    let mappings = compute_mappings(&params, &config);
    let n = config.expansion_rate;

    for hidden_dim in [512, 2048, 4096] {
        let x = Array2::<f32>::zeros((n, hidden_dim));

        group.bench_with_input(
            BenchmarkId::new("pre_mapping/dim", hidden_dim),
            &(x.clone(), mappings.clone()),
            |b, (x, m)| {
                b.iter(|| {
                    apply_pre_mapping(black_box(&m.h_pre), black_box(&x.view()));
                });
            },
        );

        let h_out = Array2::<f32>::zeros((n, hidden_dim));
        group.bench_with_input(
            BenchmarkId::new("post_res_mapping/dim", hidden_dim),
            &(x.clone(), h_out.clone(), mappings.clone()),
            |b, (x, h_out, m)| {
                b.iter(|| {
                    apply_post_res_mapping(
                        black_box(&m.h_res),
                        black_box(&m.h_post),
                        black_box(&x.view()),
                        black_box(&h_out.view()),
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_compute_mappings, bench_apply_mappings);
criterion_main!(benches);
