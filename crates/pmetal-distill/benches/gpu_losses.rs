//! Benchmarks for GPU-first distillation losses.
//!
//! Run with: cargo bench -p pmetal-distill

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use pmetal_distill::losses::{
    KlDivergenceLoss, JensenShannonLoss, SoftCrossEntropyLoss,
    HiddenStateLoss, DistillLoss, is_gpu_available,
};
use mlx_rs::Array;

/// Generate random logits for benchmarking.
fn random_logits(batch: usize, seq: usize, vocab: usize) -> Array {
    let size = batch * seq * vocab;
    // Use deterministic values for reproducibility
    let data: Vec<f32> = (0..size)
        .map(|i| {
            let x = (i as f32 * 0.1234567) % 10.0 - 5.0;
            x
        })
        .collect();
    Array::from_slice(&data, &[batch as i32, seq as i32, vocab as i32])
}

/// Generate random hidden states for benchmarking.
fn random_hidden(batch: usize, seq: usize, hidden: usize) -> Array {
    let size = batch * seq * hidden;
    let data: Vec<f32> = (0..size)
        .map(|i| {
            let x = (i as f32 * 0.7654321) % 2.0 - 1.0;
            x
        })
        .collect();
    Array::from_slice(&data, &[batch as i32, seq as i32, hidden as i32])
}

fn bench_kl_divergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("kl_divergence");
    group.sample_size(50);

    let configs = [
        // (batch, seq, vocab, description)
        (1, 8, 1024, "tiny_1x8x1k"),
        (4, 8, 8192, "small_4x8x8k"),
        (4, 32, 32000, "medium_4x32x32k"),
        (8, 64, 32000, "large_8x64x32k"),
        (16, 128, 128000, "xlarge_16x128x128k"),
    ];

    for (batch, seq, vocab, name) in configs {
        let teacher = random_logits(batch, seq, vocab);
        let student = random_logits(batch, seq, vocab);
        let loss = KlDivergenceLoss::new();

        let throughput = (batch * seq) as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("gpu_first", name),
            &(&teacher, &student),
            |b, (t, s)| {
                b.iter(|| {
                    let _ = black_box(loss.compute(t, s, 2.0));
                });
            },
        );
    }

    group.finish();
}

fn bench_kl_temperature(c: &mut Criterion) {
    let mut group = c.benchmark_group("kl_temperature");
    group.sample_size(50);

    let batch = 4;
    let seq = 32;
    let vocab = 32000;

    let teacher = random_logits(batch, seq, vocab);
    let student = random_logits(batch, seq, vocab);
    let loss = KlDivergenceLoss::new();

    for temp in [0.5, 1.0, 2.0, 4.0, 10.0] {
        group.bench_with_input(
            BenchmarkId::new("T", format!("{:.1}", temp)),
            &temp,
            |b, t| {
                b.iter(|| {
                    let _ = black_box(loss.compute(&teacher, &student, *t));
                });
            },
        );
    }

    group.finish();
}

fn bench_jensen_shannon(c: &mut Criterion) {
    let mut group = c.benchmark_group("jensen_shannon");
    group.sample_size(50);

    let configs = [
        (1, 8, 1024, "tiny_1x8x1k"),
        (4, 8, 8192, "small_4x8x8k"),
        (4, 32, 32000, "medium_4x32x32k"),
        (8, 64, 32000, "large_8x64x32k"),
    ];

    for (batch, seq, vocab, name) in configs {
        let teacher = random_logits(batch, seq, vocab);
        let student = random_logits(batch, seq, vocab);
        let loss = JensenShannonLoss::new();

        let throughput = (batch * seq) as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("gpu_first", name),
            &(&teacher, &student),
            |b, (t, s)| {
                b.iter(|| {
                    let _ = black_box(loss.compute(t, s, 2.0));
                });
            },
        );
    }

    group.finish();
}

fn bench_soft_cross_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_cross_entropy");
    group.sample_size(50);

    let configs = [
        (1, 8, 1024, "tiny_1x8x1k"),
        (4, 8, 8192, "small_4x8x8k"),
        (4, 32, 32000, "medium_4x32x32k"),
        (8, 64, 32000, "large_8x64x32k"),
    ];

    for (batch, seq, vocab, name) in configs {
        let teacher = random_logits(batch, seq, vocab);
        let student = random_logits(batch, seq, vocab);
        let loss = SoftCrossEntropyLoss::new();

        let throughput = (batch * seq) as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("gpu_first", name),
            &(&teacher, &student),
            |b, (t, s)| {
                b.iter(|| {
                    let _ = black_box(loss.compute(t, s, 2.0));
                });
            },
        );
    }

    group.finish();
}

fn bench_hidden_mse(c: &mut Criterion) {
    let mut group = c.benchmark_group("hidden_mse");
    group.sample_size(50);

    let configs = [
        (4, 32, 512, "small_4x32x512"),
        (4, 64, 2048, "medium_4x64x2k"),
        (8, 128, 4096, "large_8x128x4k"),
    ];

    for (batch, seq, hidden, name) in configs {
        let teacher = random_hidden(batch, seq, hidden);
        let student = random_hidden(batch, seq, hidden);
        let loss = HiddenStateLoss::mse();

        let throughput = (batch * seq) as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("gpu_first", name),
            &(&teacher, &student),
            |b, (t, s)| {
                b.iter(|| {
                    let _ = black_box(loss.compute(t, s));
                });
            },
        );
    }

    group.finish();
}

fn bench_hidden_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("hidden_cosine");
    group.sample_size(50);

    let configs = [
        (4, 32, 512, "small_4x32x512"),
        (4, 64, 2048, "medium_4x64x2k"),
        (8, 128, 4096, "large_8x128x4k"),
    ];

    for (batch, seq, hidden, name) in configs {
        let teacher = random_hidden(batch, seq, hidden);
        let student = random_hidden(batch, seq, hidden);
        let loss = HiddenStateLoss::cosine();

        let throughput = (batch * seq) as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("gpu_first", name),
            &(&teacher, &student),
            |b, (t, s)| {
                b.iter(|| {
                    let _ = black_box(loss.compute(t, s));
                });
            },
        );
    }

    group.finish();
}

fn bench_vocab_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("vocab_scaling");
    group.sample_size(30);

    let batch = 4;
    let seq = 32;

    // Test how performance scales with vocabulary size
    for vocab in [1024, 4096, 8192, 16384, 32000, 65536, 128000] {
        let teacher = random_logits(batch, seq, vocab);
        let student = random_logits(batch, seq, vocab);
        let loss = KlDivergenceLoss::new();

        let throughput = (batch * seq * vocab) as u64; // Total elements
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("kl_div", format!("vocab_{}", vocab)),
            &(&teacher, &student),
            |b, (t, s)| {
                b.iter(|| {
                    let _ = black_box(loss.compute(t, s, 2.0));
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scaling");
    group.sample_size(30);

    let seq = 32;
    let vocab = 32000;

    // Test how performance scales with batch size
    for batch in [1, 2, 4, 8, 16, 32] {
        let teacher = random_logits(batch, seq, vocab);
        let student = random_logits(batch, seq, vocab);
        let loss = KlDivergenceLoss::new();

        let throughput = (batch * seq) as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("kl_div", format!("batch_{}", batch)),
            &(&teacher, &student),
            |b, (t, s)| {
                b.iter(|| {
                    let _ = black_box(loss.compute(t, s, 2.0));
                });
            },
        );
    }

    group.finish();
}

fn bench_comparison_all_losses(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_comparison");
    group.sample_size(50);

    // Standard configuration for comparison
    let batch = 4;
    let seq = 32;
    let vocab = 32000;

    let teacher = random_logits(batch, seq, vocab);
    let student = random_logits(batch, seq, vocab);

    let throughput = (batch * seq) as u64;
    group.throughput(Throughput::Elements(throughput));

    // KL Divergence (forward)
    let kl_fwd = KlDivergenceLoss::new();
    group.bench_function("kl_forward", |b| {
        b.iter(|| {
            let _ = black_box(kl_fwd.compute(&teacher, &student, 2.0));
        });
    });

    // KL Divergence (reverse)
    let kl_rev = KlDivergenceLoss::reverse();
    group.bench_function("kl_reverse", |b| {
        b.iter(|| {
            let _ = black_box(kl_rev.compute(&teacher, &student, 2.0));
        });
    });

    // Jensen-Shannon
    let js = JensenShannonLoss::new();
    group.bench_function("jensen_shannon", |b| {
        b.iter(|| {
            let _ = black_box(js.compute(&teacher, &student, 2.0));
        });
    });

    // Soft Cross-Entropy
    let soft_ce = SoftCrossEntropyLoss::new();
    group.bench_function("soft_cross_entropy", |b| {
        b.iter(|| {
            let _ = black_box(soft_ce.compute(&teacher, &student, 2.0));
        });
    });

    group.finish();
}

fn gpu_status() {
    println!("\n=== GPU Distillation Loss Benchmarks ===");
    println!("GPU available: {}", is_gpu_available());
    println!("");
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(2));
    targets =
        bench_kl_divergence,
        bench_kl_temperature,
        bench_jensen_shannon,
        bench_soft_cross_entropy,
        bench_hidden_mse,
        bench_hidden_cosine,
        bench_vocab_scaling,
        bench_batch_scaling,
        bench_comparison_all_losses
);

criterion_main!(benches);
