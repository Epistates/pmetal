//! The compiled fused ops share MLX's process-wide compile cache, which MLX
//! does not lock. Threads that build graphs through them at the same time
//! (each on its own stream, so each adds its own cache entries) corrupted it.

use std::sync::{Arc, Barrier};

use pmetal_runtime::compat::Dtype;
use pmetal_runtime::inline_array::{self as bridge, InlineArray};

const THREADS: usize = 16;
const ROUNDS: usize = 40;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Every thread runs every fused op on every float dtype, so the cache sees
/// first calls (new entries) from all threads at once, then replays.
fn worker(barrier: Arc<Barrier>, id: usize) {
    // Half the threads share the default stream (and so one cache entry per
    // op and dtype), half work on a stream of their own.
    let own_stream = id % 2 == 0;
    if own_stream {
        bridge::new_generation_stream();
        bridge::set_generation_stream();
    }
    barrier.wait();

    let dtypes = [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16];
    for round in 0..ROUNDS {
        for dtype in dtypes {
            let width = 4 + (id + round) % 5;
            let data: Vec<f32> = (0..width).map(|i| i as f32 * 0.25 - 0.5).collect();
            let x = InlineArray::from_f32_slice(&data, &[width as i32]).as_dtype(dtype.as_i32());
            let y = x.fused_silu();
            let gated = InlineArray::fused_swiglu(&x, &x);
            let mut y32 = y.as_dtype(Dtype::Float32.as_i32());
            let mut g32 = gated.as_dtype(Dtype::Float32.as_i32());
            let got_y = y32.to_f32_vec(width).expect("silu readback");
            let got_g = g32.to_f32_vec(width).expect("swiglu readback");
            for (i, (&gy, &gg)) in got_y.iter().zip(&got_g).enumerate() {
                let v = i as f32 * 0.25 - 0.5;
                // Half-precision inputs are rounded before the op; the
                // tolerance is a few ulps of bf16 (2^-8 relative) at these
                // magnitudes.
                assert!((gy - silu(v)).abs() < 2e-2, "silu({v}) = {gy}");
                assert!((gg - silu(v) * v).abs() < 2e-2, "swiglu({v}) = {gg}");
            }
        }
    }
    bridge::synchronize();
    if own_stream {
        bridge::reset_default_stream();
    }
}

#[test]
fn fused_ops_are_safe_to_first_call_from_many_threads() {
    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|id| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || worker(barrier, id))
        })
        .collect();
    for h in handles {
        h.join().expect("worker thread");
    }
    pmetal_runtime::check_unobserved_error().expect("bridge error");
}
