//! TurboQuant QJL ablation harness.
//!
//! Run with: `cargo bench --bench turboquant_qjl_ablation -p pmetal-bridge --features tq-ablation`
//!
//! # What this measures
//!
//! TurboQuant's score kernel computes
//! `score = key_norm * (codebook_dot + residual_scale * sign * qproj)`.
//! The `residual_scale * sign * qproj` term is the **QJL residual** — a
//! 1-bit sign hash of the per-row residual after Beta-codebook reconstruction,
//! projected through a Gaussian J. The reference TurboQuant paper's own
//! ablation ("Karpathy loop") found this term contributes ≈ 0 to attention
//! scores; dropping it reclaims a bit per coordinate for higher codebook
//! resolution (Variant F). We want to reproduce or refute that on our
//! targeted models before flipping the production default in Phase C.
//!
//! # What this harness does today
//!
//! This is a **synthetic-data smoke harness** that exercises the ablation
//! plumbing end-to-end:
//!
//! 1. Build a [`QuantizedKvCache`] for a configurable head_dim / bits / seq.
//! 2. Append synthetic Gaussian K/V tensors.
//! 3. Run [`QuantizedKvCache::append_and_compute_attention`] against a
//!    synthetic query, with [`ablation::qjl_disabled`] = `false`.
//! 4. Reset, repeat with `qjl_disabled` = `true`.
//! 5. Compare the two output tensors element-wise — report mean |Δ|,
//!    max |Δ|, and cosine similarity over the attention output.
//!
//! The synthetic-data score-drift number is a *first-order sanity check*:
//! it shows the ablation knob is actually wired in (a non-zero |Δ| confirms
//! the QJL term is contributing on this synthetic distribution). It does
//! **not** decide Phase C — that requires the real wikitext-2 perplexity
//! sweep across actual model weights (see "TODO: real-model integration"
//! below).
//!
//! # Decision criterion (for the real measurement)
//!
//! - ΔPPL < 0.5% across all (model, bits, ctx) cells → reproduces, ship
//!   Variant F as default in Phase C.
//! - ΔPPL ≥ 0.5% on any cell → does not reproduce on our targets, ship
//!   Variant F as opt-in only.
//!
//! # TODO: real-model integration
//!
//! The full sweep (4 models × 3 bits × 2 ctx × 2 QJL states) requires
//! loading actual model weights and a wikitext-2 corpus. That belongs in
//! `pmetal-models/benches/turboquant_qjl_ablation.rs` (so it can call
//! `pmetal_models::generation::token_logprobs` for per-token NLL); the
//! pmetal-bridge crate intentionally has no dep on pmetal-models. When
//! that bench lands, gate it on the `tq-ablation` feature flowing through
//! `pmetal-bridge/tq-ablation`.

use pmetal_bridge::InlineArray;
use pmetal_bridge::turboquant::{QuantizedKvCache, TurboQuantConfig};

#[cfg(feature = "tq-ablation")]
use pmetal_bridge::turboquant::ablation;

/// Ablation cell: one (head_dim, bits, seq, batch, q_heads, kv_heads) point.
#[derive(Debug, Clone, Copy)]
struct Cell {
    head_dim: usize,
    bits: u8,
    seq: i32,
    q_heads: i32,
    kv_heads: i32,
}

const CELLS: &[Cell] = &[
    // Cover the dims the GPU score kernels actually take: 128 + 256.
    // Bits sweep matches the plan's 4b / 3b / 2b — we round 3b down to the
    // smallest viable codebook (mse_bits = bits - 1 = 2).
    Cell { head_dim: 128, bits: 4, seq: 1024, q_heads: 4, kv_heads: 4 },
    Cell { head_dim: 128, bits: 3, seq: 1024, q_heads: 4, kv_heads: 4 },
    Cell { head_dim: 128, bits: 2, seq: 1024, q_heads: 4, kv_heads: 4 },
    Cell { head_dim: 256, bits: 4, seq: 1024, q_heads: 4, kv_heads: 4 },
    Cell { head_dim: 256, bits: 3, seq: 1024, q_heads: 4, kv_heads: 4 },
    Cell { head_dim: 256, bits: 2, seq: 1024, q_heads: 4, kv_heads: 4 },
];

fn synthetic_gaussian(shape: &[i32], seed: u64) -> InlineArray {
    let n: i32 = shape.iter().product();
    // Box-Muller from a deterministic LCG keeps the values reproducible
    // across runs without pulling in the rand crate's distribution code.
    let mut state = seed;
    let mut step = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32) as f32 / (u32::MAX as f32)
    };
    let mut data = Vec::with_capacity(n as usize);
    while data.len() < n as usize {
        let u1 = step().max(1e-12);
        let u2 = step();
        let mag = (-2.0_f32 * u1.ln()).sqrt();
        let z0 = mag * (std::f32::consts::TAU * u2).cos();
        let z1 = mag * (std::f32::consts::TAU * u2).sin();
        data.push(z0);
        if data.len() < n as usize {
            data.push(z1);
        }
    }
    InlineArray::from_f32_slice(&data, shape)
}

fn run_cell(cell: Cell, qjl_disabled: bool) -> InlineArray {
    #[cfg(feature = "tq-ablation")]
    ablation::set_qjl_disabled(qjl_disabled);
    #[cfg(not(feature = "tq-ablation"))]
    {
        // Without the feature, the toggle is a no-op — we still run the
        // path so the harness produces output, and the qjl_disabled=true
        // arm will report `Δ ≈ 0` as a confirmation that the feature flag
        // is actually load-bearing.
        let _ = qjl_disabled;
    }

    let b: i32 = 1;
    let d: i32 = cell.head_dim as i32;
    let config = TurboQuantConfig::uniform(cell.bits, cell.bits)
        .with_recent_window(None);
    let mut cache = QuantizedKvCache::new(config);

    // Prefill with seq tokens then run a single decode-style query step.
    let prefill_keys = synthetic_gaussian(&[b, cell.kv_heads, cell.seq, d], 0xA);
    let prefill_values = synthetic_gaussian(&[b, cell.kv_heads, cell.seq, d], 0xB);
    cache
        .append(&prefill_keys, &prefill_values)
        .expect("prefill append");

    let queries = synthetic_gaussian(&[b, cell.q_heads, 1, d], 0xC);
    let step_keys = synthetic_gaussian(&[b, cell.kv_heads, 1, d], 0xD);
    let step_values = synthetic_gaussian(&[b, cell.kv_heads, 1, d], 0xE);
    let scale = 1.0f32 / (cell.head_dim as f32).sqrt();

    cache
        .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
        .expect("attention")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    (dot / denom) as f32
}

fn main() {
    println!("# TurboQuant QJL ablation — synthetic-data smoke harness");
    println!();
    #[cfg(feature = "tq-ablation")]
    println!("Feature `tq-ablation` is ENABLED — qjl_disabled toggle is live.");
    #[cfg(not(feature = "tq-ablation"))]
    println!(
        "Feature `tq-ablation` is DISABLED — toggle is a no-op. Re-run with \
         `--features tq-ablation` to exercise the ablation."
    );
    println!();
    println!(
        "{:<5}  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}",
        "head", "bits", "seq", "mean|Δ|", "max|Δ|", "cos(out)"
    );

    for cell in CELLS {
        let mut out_on = run_cell(*cell, false);
        let mut out_off = run_cell(*cell, true);

        let total = (cell.q_heads as usize) * (cell.head_dim);
        let v_on = out_on.to_f32_vec(total).expect("on→f32");
        let v_off = out_off.to_f32_vec(total).expect("off→f32");

        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&a, &b) in v_on.iter().zip(v_off.iter()) {
            let d = (a - b).abs();
            sum_abs += d as f64;
            if d > max_abs {
                max_abs = d;
            }
        }
        let mean_abs = (sum_abs / v_on.len() as f64) as f32;
        let cos = cosine_similarity(&v_on, &v_off);

        println!(
            "{:<5}  {:>6}  {:>6}  {:>10.6}  {:>10.6}  {:>10.6}",
            cell.head_dim, cell.bits, cell.seq, mean_abs, max_abs, cos
        );
    }
}
