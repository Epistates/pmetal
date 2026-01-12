//! Benchmark to measure FFI overhead in mlx-rs vs raw operations.
//!
//! Run with: cargo bench --bench ffi_overhead

use std::time::Instant;
use mlx_rs::{Array, ops::indexing::argmax, transforms::eval};

fn main() {
    // Warmup
    let warmup = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
    let _ = argmax(&warmup, None).unwrap();
    eval(&[&warmup]).unwrap();

    // Create test array similar to logits
    let vocab_size = 151936; // Qwen3 vocab size
    let logits_data: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
    let logits = Array::from_slice(&logits_data, &[1, vocab_size as i32]);
    eval(&[&logits]).unwrap();

    // Benchmark argmax operations
    let n_iters = 1000;

    println!("Benchmarking {} argmax operations on [{}, {}] array...", n_iters, 1, vocab_size);

    let start = Instant::now();
    for _ in 0..n_iters {
        let result = argmax(&logits, Some(-1)).unwrap();
        eval(&[&result]).unwrap();
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Operations per second: {:.0}", 1_000_000.0 / per_op_us);

    // Now benchmark with reshape (like our generation loop)
    println!("\nBenchmarking argmax + reshape (generation loop pattern)...");

    let token = Array::from_slice(&[42i32], &[1]);
    eval(&[&token]).unwrap();

    let start = Instant::now();
    for _ in 0..n_iters {
        // Reshape token [1] -> [1, 1] (like our loop)
        let input = token.reshape(&[1, 1]).unwrap();
        // Simulate logits extraction
        let result = argmax(&logits, Some(-1)).unwrap();
        eval(&[&input, &result]).unwrap();
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Equivalent tok/s: {:.0}", 1_000_000.0 / per_op_us);

    println!("\nDone!");
}
