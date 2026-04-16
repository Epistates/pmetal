//! Criterion benchmarks for the shared Metal compute-kernel dispatch path.
//!
//! Run with:
//!     cargo bench -p pmetal-metal --bench kernel_dispatch
//!
//! ## What this measures
//!
//! `dispatch_linear_kernel` (via `DequantKernels::dequantize_q4_0`) is the
//! simplest helper on top of the 7-step boilerplate. It does one
//! `dispatchThreads → commit → waitUntilCompleted`; the total wall-time
//! therefore captures:
//!   1. Pipeline cache lookup (first call) / cache hit (subsequent).
//!   2. Command-buffer + encoder allocation overhead.
//!   3. Buffer-binding overhead per input (2 buffers here).
//!   4. GPU kernel execution + Apple-Silicon synchronisation.
//!
//! At small element counts the dispatch overhead dominates; past ~64K
//! elements the GPU side dominates. Both regimes are useful to watch
//! for regressions — the audit called out the "no Metal bench suite"
//! SOTA gap as the reason the MoE audit shipped a slower kernel.
//!
//! ## Why dequant as the canary
//!
//! The dequant kernel is the one migrated onto `dispatch_linear_kernel` in
//! Phase 9 and has no function-constant specialisation or threadgroup
//! scratch memory — so any regression we see here points at the helper
//! itself, not at shader-specific changes.
//!
//! The `#[cfg(target_os = "macos")]` guard keeps the bench out of the
//! matrix on non-Apple hosts where there's no Metal runtime to allocate
//! against.

#![cfg(target_os = "macos")]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use pmetal_metal::buffer::{BufferUsage, MetalBuffer};
use pmetal_metal::context::MetalContext;
use pmetal_metal::kernels::dequant::DequantKernels;

/// Dispatch the Q4_0 dequant kernel at a range of element counts that
/// cover both the dispatch-overhead-bound regime (<64K) and the
/// GPU-bound regime (>=256K).
///
/// Q4_0 packs 32 values per 18-byte block (2 bytes scale + 16 bytes
/// nibbles), so the input buffer is sized accordingly.
fn bench_dequant_q4_0_dispatch(c: &mut Criterion) {
    let ctx = match MetalContext::global() {
        Ok(c) => c,
        Err(_) => {
            eprintln!(
                "kernel_dispatch bench: skipping — Metal context unavailable \
                 (likely no-Metal test environment)"
            );
            return;
        }
    };
    let kernels = match DequantKernels::new(&ctx) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("kernel_dispatch bench: DequantKernels::new failed: {e:?}");
            return;
        }
    };

    let mut group = c.benchmark_group("dequantize_q4_0");

    // Q4_0 block layout constants.
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2B scale + 16B packed nibbles

    // Element counts that exercise both dispatch-bound and GPU-bound
    // regimes. All are multiples of BLOCK_ELEMENTS so the input size is
    // an exact number of Q4_0 blocks.
    let sizes: &[usize] = &[1024, 8_192, 65_536, 262_144, 1_048_576];

    for &n in sizes {
        // `usize::is_multiple_of` only stabilised in Rust 1.87 and the
        // workspace MSRV is 1.86 — use the arithmetic form to stay
        // compatible without raising the toolchain floor.
        assert!(n % BLOCK_ELEMENTS == 0, "n must be divisible by 32");
        let n_blocks = n / BLOCK_ELEMENTS;
        let input_bytes = n_blocks * BLOCK_BYTES;

        // Zeroed input is fine — we measure dispatch cost, not the
        // numerical output. The GPU still executes the kernel for every
        // block regardless of content.
        let input = match MetalBuffer::<u8>::zeros(&ctx, input_bytes, BufferUsage::Shared) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("bench: failed to allocate input buffer for n={n}: {e:?}");
                continue;
            }
        };
        let output = match MetalBuffer::<f32>::zeros(&ctx, n, BufferUsage::Shared) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("bench: failed to allocate output buffer for n={n}: {e:?}");
                continue;
            }
        };

        // Throughput in output-f32 bytes (4 bytes × n).
        group.throughput(Throughput::Bytes((n * std::mem::size_of::<f32>()) as u64));

        let input_ret = input.as_retained();
        let output_ret = output.as_retained();

        group.bench_with_input(BenchmarkId::new("n_elements", n), &n, |b, &n_elem| {
            b.iter(|| {
                kernels
                    .dequantize_q4_0(
                        &ctx,
                        black_box(&input_ret),
                        black_box(&output_ret),
                        black_box(n_elem),
                    )
                    .expect("dequantize_q4_0 dispatch failed");
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dequant_q4_0_dispatch);
criterion_main!(benches);
