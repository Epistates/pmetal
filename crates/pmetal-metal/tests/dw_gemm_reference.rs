//! `DwGemm` against a float64 host reference: `C = alpha * A B^T + beta * C`.
//!
//! The ANE trainer calls it with whatever `[tokens, dim]` shapes the model has,
//! so every dimension is exercised off the tile size (64 x 64 x 16) as well as
//! on it, and the BLAS convention that `beta == 0` never reads `C` is held to.

#![cfg(target_os = "macos")]

use pmetal_metal::buffer::{BufferUsage, MetalBuffer};
use pmetal_metal::context::MetalContext;
use pmetal_metal::kernels::BatchedCommandBuffer;
use pmetal_metal::kernels::dw_gemm::DwGemm;

/// Deterministic pseudo-normal values.
fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n)
        .map(|_| (next() + next() + next() + next() - 2.0) * 1.7)
        .collect()
}

/// `alpha * A B^T + beta * C` in f64, plus the per-element magnitude
/// `sum |a b|` that bounds the f32 accumulation error.
fn reference(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    (m, n, k): (usize, usize, usize),
    alpha: f32,
    beta: f32,
) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; m * n];
    let mut mag = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut dot, mut abs) = (0.0f64, 0.0f64);
            for p in 0..k {
                let prod = a[i * k + p] as f64 * b[j * k + p] as f64;
                dot += prod;
                abs += prod.abs();
            }
            let carried = if beta == 0.0 {
                0.0
            } else {
                beta as f64 * c[i * n + j] as f64
            };
            out[i * n + j] = alpha as f64 * dot + carried;
            mag[i * n + j] = (alpha as f64).abs() * abs + carried.abs();
        }
    }
    (out, mag)
}

fn run(
    ctx: &std::sync::Arc<MetalContext>,
    (m, n, k): (usize, usize, usize),
    alpha: f32,
    beta: f32,
    c_init: &[f32],
) -> Vec<f32> {
    let a = data(m * k, 1);
    let b = data(n * k, 2);
    let ab = MetalBuffer::from_slice(ctx, &a, BufferUsage::Shared).unwrap();
    let bb = MetalBuffer::from_slice(ctx, &b, BufferUsage::Shared).unwrap();
    let cb = MetalBuffer::from_slice(ctx, c_init, BufferUsage::Shared).unwrap();
    let kernel = DwGemm::new(ctx.clone()).unwrap();
    let mut batch = BatchedCommandBuffer::new(ctx.clone()).unwrap();
    kernel
        .queue_gemm_accum(&mut batch, &ab, &bb, &cb, m, n, k, alpha, beta)
        .unwrap();
    batch.execute().unwrap();
    cb.to_vec().unwrap()
}

fn check(shape: (usize, usize, usize), alpha: f32, beta: f32, poison_c: bool) {
    let ctx = MetalContext::global().expect("a Metal device");
    let (m, n, k) = shape;
    let a = data(m * k, 1);
    let b = data(n * k, 2);
    let c_init = if poison_c {
        vec![f32::NAN; m * n]
    } else {
        data(m * n, 3)
    };
    let got = run(&ctx, shape, alpha, beta, &c_init);
    let (want, mag) = reference(&a, &b, &c_init, shape, alpha, beta);
    // A float32 dot product of k terms is off by at most ~k * 2^-24 of the
    // sum of magnitudes; sqrt(k) is the typical case, so k * 2^-23 is a
    // comfortable bound that still catches a dropped or doubled term.
    let eps = k.max(1) as f64 * 2f64.powi(-23);
    for idx in 0..m * n {
        let err = (got[idx] as f64 - want[idx]).abs();
        assert!(
            got[idx].is_finite() && err <= eps * mag[idx] + 1e-30,
            "shape {shape:?} alpha {alpha} beta {beta}: C[{}][{}] = {} but the reference is {} (|err| {err:.3e}, bound {:.3e})",
            idx / n,
            idx % n,
            got[idx],
            want[idx],
            eps * mag[idx]
        );
    }
}

#[test]
fn matches_the_f64_reference_on_and_off_the_tile_grid() {
    for shape in [
        (1, 1, 1),
        (7, 5, 3),
        (64, 64, 16),
        (65, 63, 17),
        (63, 65, 15),
        (130, 70, 33),
        (257, 129, 100),
        (96, 200, 1000),
        (384, 512, 300),
    ] {
        check(shape, 1.0, 0.0, false);
        check(shape, 1.0, 1.0, false);
        check(shape, 0.5, 2.0, false);
        check(shape, 2.0, -1.5, false);
    }
}

#[test]
fn beta_zero_does_not_read_the_accumulator() {
    // BLAS convention: with beta == 0, C is output only, and may hold
    // anything (a fresh, uninitialized buffer) without poisoning the result.
    for shape in [(7, 5, 3), (65, 63, 17), (130, 70, 33)] {
        check(shape, 1.0, 0.0, true);
    }
}
