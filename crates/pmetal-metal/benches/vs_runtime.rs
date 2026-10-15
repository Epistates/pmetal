//! Each hand-written kernel against the runtime op it would replace.
//!
//! Run with:
//!     cargo bench -p pmetal-metal --bench vs_runtime -- [section ...]
//!
//! Sections: `dwgemm`.
//!
//! Every row times the *kernel alone* with its inputs already resident: the
//! runtime op is built lazily and evaluated, the Metal kernel is dispatched and
//! waited on. Medians over a fixed number of timed runs after warm-up. The GPU
//! is shared with whatever else is running, so treat the numbers as relative
//! and re-run interleaved.

#![cfg(target_os = "macos")]

use std::sync::Arc;
use std::time::Instant;

use pmetal_metal::buffer::{BufferUsage, MetalBuffer};
use pmetal_metal::context::MetalContext;
use pmetal_metal::kernels::BatchedCommandBuffer;
use pmetal_metal::kernels::dw_gemm::DwGemm;
use pmetal_runtime::InlineArray;

const WARMUP: usize = 3;
const RUNS: usize = 15;

fn median_ms(mut f: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples: Vec<f64> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[RUNS / 2]
}

/// Deterministic pseudo-normal data (sum of uniforms).
fn data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n)
        .map(|_| (next() + next() + next() + next() - 2.0) * 1.7 * scale)
        .collect()
}

fn row(label: &str, runtime_ms: f64, ours_ms: f64, flops: f64) {
    println!(
        "  {label:<28} runtime {runtime_ms:>8.3} ms ({:>5.2} TFLOP/s)   ours {ours_ms:>8.3} ms ({:>5.2} TFLOP/s)   ours/runtime {:>5.2}x",
        flops / runtime_ms / 1e9,
        flops / ours_ms / 1e9,
        ours_ms / runtime_ms
    );
}

/// Largest absolute difference relative to the reference's largest magnitude.
fn rel_err(ours: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(ours.len(), reference.len());
    let scale = reference
        .iter()
        .fold(0f32, |m, v| m.max(v.abs()))
        .max(1e-20);
    ours.iter()
        .zip(reference)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

fn dwgemm(ctx: &Arc<MetalContext>) {
    println!("\n== dW GEMM (f32): C += A B^T ==");
    let kern = DwGemm::new(ctx.clone()).unwrap();
    for &(m, n, k) in &[
        (1024usize, 2816usize, 512usize),
        (4096, 4096, 2048),
        (4096, 11008, 2048),
    ] {
        let a = data(m * k, 1, 1.0);
        let b = data(n * k, 2, 1.0);
        let aa = InlineArray::from_f32_slice(&a, &[m as i32, k as i32]);
        let ba = InlineArray::from_f32_slice(&b, &[n as i32, k as i32]);
        aa.eval();
        ba.eval();
        let runtime_ms = median_ms(|| {
            aa.matmul(&ba.t()).eval();
        });

        let mk = |v: &[f32]| MetalBuffer::from_slice(ctx, v, BufferUsage::Shared).unwrap();
        let (ab, bb) = (mk(&a), mk(&b));
        let cb = MetalBuffer::<f32>::zeros(ctx, m * n, BufferUsage::Shared).unwrap();
        let run = |beta: f32| {
            let mut batch = BatchedCommandBuffer::new(ctx.clone()).unwrap();
            kern.queue_gemm_accum(&mut batch, &ab, &bb, &cb, m, n, k, 1.0, beta)
                .unwrap();
            batch.execute().unwrap();
        };
        run(0.0);
        let want = aa.matmul(&ba.t());
        println!(
            "    rel.err vs runtime: {:.2e}",
            rel_err(cb.as_slice(), want.as_slice::<f32>())
        );
        let ours_ms = median_ms(|| run(1.0));
        row(
            &format!("M={m} N={n} K={k}"),
            runtime_ms,
            ours_ms,
            2.0 * (m * n * k) as f64,
        );
    }
}

fn main() {
    let filter: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();
    let wants = |name: &str| filter.is_empty() || filter.iter().any(|f| f == name);
    let ctx = MetalContext::global().expect("a Metal device");
    println!("device: {}", ctx.properties().name);
    if wants("dwgemm") {
        dwgemm(&ctx);
    }
    pmetal_runtime::check_unobserved_error().expect("a runtime op failed");
}
