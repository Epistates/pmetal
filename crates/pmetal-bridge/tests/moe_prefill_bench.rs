//! A routed layer's prefill, with the `(token, expert)` pairs in router order
//! and grouped by expert. Run on an otherwise idle machine:
//! `cargo test -p pmetal-bridge --release --test moe_prefill_bench -- --ignored --nocapture`.

use std::time::Instant;

use pmetal_bridge::compat::{Dtype, random};
use pmetal_bridge::native_moe::switch_glu_grouped;
use pmetal_bridge::native_weight::{LayerWeight, QuantParams};
use pmetal_bridge::{InlineArray, QuantizedMode};

const EXPERTS: i32 = 128;
const TOP_K: i32 = 8;
const HIDDEN: i32 = 2048;
const INTERMEDIATE: i32 = 768;

/// Distinct experts per token from a small LCG.
fn routes(tokens: i32) -> InlineArray {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut inds = Vec::with_capacity((tokens * TOP_K) as usize);
    for _ in 0..tokens {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < TOP_K as usize {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((state >> 33) % EXPERTS as u64) as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        inds.extend(row);
    }
    InlineArray::from_u32_slice(&inds, &[tokens, TOP_K])
}

fn experts(rows: i32, cols: i32, seed: u64, packed: bool) -> LayerWeight {
    // `[E, out, in]`, the checkpoint layout.
    let key = pmetal_bridge::decode::draw_key(seed, 0);
    let w = random::normal_keyed(&[EXPERTS, cols, rows], Dtype::Bfloat16, Some(&key)).multiply(
        &InlineArray::scalar_with_dtype(0.02, Dtype::Bfloat16.as_i32()),
    );
    if packed {
        let params = QuantParams::defaults_for(QuantizedMode::Affine);
        let (weight, scales, biases) = w.quantize_weights(params.group_size, params.bits);
        LayerWeight::new(weight, Some(scales), Some(biases), params)
    } else {
        // `[E, in, out]`, as the engines hold a dense stack.
        LayerWeight::Dense(w.transpose_axes(&[0, 2, 1]))
    }
}

fn median_ms(mut run: impl FnMut() -> InlineArray) -> f64 {
    for _ in 0..3 {
        run().eval();
    }
    let mut times: Vec<f64> = (0..9)
        .map(|_| {
            let out = run();
            let start = Instant::now();
            out.eval();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

#[test]
#[ignore = "benchmark"]
fn grouped_and_router_order_prefill() {
    for packed in [false, true] {
        let (gate, up, down) = (
            experts(HIDDEN, INTERMEDIATE, 1, packed),
            experts(HIDDEN, INTERMEDIATE, 2, packed),
            experts(INTERMEDIATE, HIDDEN, 3, packed),
        );
        for tokens in [128, 512, 2048] {
            let key = pmetal_bridge::decode::draw_key(9, 0);
            let x = random::normal_keyed(&[tokens, HIDDEN], Dtype::Bfloat16, Some(&key));
            let inds = routes(tokens);
            let scores = InlineArray::from_f32_slice(
                &vec![1.0 / TOP_K as f32; (tokens * TOP_K) as usize],
                &[tokens, TOP_K],
            )
            .as_dtype(Dtype::Bfloat16.as_i32());
            let timed = |grouped: bool| {
                median_ms(|| {
                    switch_glu_grouped(
                        &x,
                        &gate,
                        &up,
                        &down,
                        &inds,
                        &scores,
                        InlineArray::fused_swiglu,
                        grouped,
                    )
                })
            };
            let in_router_order = timed(false);
            let grouped = timed(true);
            pmetal_bridge::check_unobserved_error().expect("no bridge op failed");
            eprintln!(
                "{} experts, {tokens:>4} tokens: router order {in_router_order:8.2} ms, \
                 grouped {grouped:8.2} ms ({:.2}x)",
                if packed { "4-bit" } else { "bf16 " },
                in_router_order / grouped
            );
        }
    }
}
