//! Criterion benchmarks for ANE-adjacent Metal kernel primitives.
//!
//! Run with:
//!     cargo bench -p pmetal-metal --bench ane_kernels
//!
//! The `#[cfg(target_os = "macos")]` guard in the crate root means these
//! benchmarks only compile on macOS.

#![cfg(target_os = "macos")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pmetal_metal::accelerate;
use pmetal_metal::ane::iosurface::IoSurface;

// ---------------------------------------------------------------------------
// IOSurface staging throughput
// ---------------------------------------------------------------------------

/// Benchmark `IoSurface::write_f32_as_fp16` at a range of tensor sizes.
///
/// This path is on the critical path of every ANE training step: all weights
/// and activations are staged through IOSurface before ANE dispatch.
/// Throughput is reported in bytes/s (f32 input side).
fn bench_iosurface_staging(c: &mut Criterion) {
    let mut group = c.benchmark_group("iosurface_write_f32_as_fp16");

    // (channels, spatial) pairs — representative of real training tensors:
    //   (64, 32)      → tiny model activation [dim=64, seq=32]
    //   (256, 128)    → medium model activation
    //   (512, 256)    → weight matrix row [hidden_dim=512, dim=256]
    //   (2048, 256)   → FFN weight [hidden=2048, dim=256]
    //   (32768, 1)    → embedding lookup row [vocab=32K, single token]
    let sizes: &[(usize, usize)] = &[(64, 32), (256, 128), (512, 256), (2048, 256), (32768, 1)];

    for &(channels, spatial) in sizes {
        let n = channels * spatial;
        // Input bytes: n × 4 (f32)
        group.throughput(Throughput::Bytes((n * 4) as u64));

        // Pre-allocate surface and source data outside the timed region.
        let surface =
            IoSurface::for_tensor(channels, spatial).expect("IoSurface::for_tensor failed");
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("channels_x_spatial", format!("{channels}x{spatial}")),
            &(channels, spatial),
            |b, &(ch, sp)| {
                b.iter(|| {
                    surface.write_f32_as_fp16(black_box(&data), black_box(ch), black_box(sp));
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// CPU cross-entropy (fwd + bwd fused)
// ---------------------------------------------------------------------------

/// Benchmark `accelerate::cross_entropy_loss` at different vocab sizes with a
/// fixed sequence length of 32.
///
/// Cross-entropy is purely on CPU (vDSP softmax + log). It runs once per
/// training step after the final ANE projection and its cost grows linearly
/// with vocab size.
fn bench_cross_entropy(c: &mut Criterion) {
    let seq = 32usize;
    let mut group = c.benchmark_group("cross_entropy_fwd_bwd");

    // Vocab sizes covering common tokenisers: 8K (tiny), 32K (LLaMA2),
    // 50K (GPT-2), 128K (Llama3), and a pathological 256K case.
    for &vocab in &[8_000usize, 32_000, 50_000, 128_000, 256_000] {
        // Total elements: vocab × seq
        group.throughput(Throughput::Elements((vocab * seq) as u64));

        // Build fixed logits and targets outside the timed region.
        let logits: Vec<f32> = (0..vocab * seq)
            .map(|i| (i as f32 % 256.0 - 128.0) * 0.01)
            .collect();
        let targets: Vec<u32> = (0..seq).map(|t| (t % vocab) as u32).collect();
        let mut dlogits = vec![0.0f32; vocab * seq];

        group.bench_with_input(BenchmarkId::new("vocab", vocab), &vocab, |b, &v| {
            b.iter(|| {
                accelerate::cross_entropy_loss(
                    black_box(&mut dlogits),
                    black_box(&logits),
                    black_box(&targets),
                    black_box(v),
                    black_box(seq),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(benches, bench_iosurface_staging, bench_cross_entropy);
criterion_main!(benches);
