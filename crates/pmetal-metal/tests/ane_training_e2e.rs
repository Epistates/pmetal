//! ANE training end-to-end validation.
//!
//! These tests require ANE hardware (Apple Silicon with a Neural Engine) and
//! are marked `#[ignore]` so that `cargo test` skips them by default.
//!
//! Run with:
//!     cargo test -p pmetal-metal --test ane_training_e2e --release -- --ignored
//!
//! The `#[cfg(target_os = "macos")]` guard ensures the module compiles only on
//! macOS, matching the `#![cfg(target_os = "macos")]` gate in the crate root.

#![cfg(target_os = "macos")]

use pmetal_metal::ane::dynamic_trainer::{DynamicAneTrainer, DynamicAneTrainerConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the total flat-weight vector length for `load_weights_flat`.
///
/// Layout (matches `DynamicAneTrainer::load_weights_flat`):
///   embed:   v * d
///   per layer × n_layers:
///     rms_att: d
///     wq:      q_dim * d
///     wk:      kv_dim * d
///     wv:      kv_dim * d
///     wo:      d * q_dim
///     rms_ffn: d
///     w1:      h * d
///     w2:      d * h
///     w3:      h * d
///   rms_final: d
fn flat_weight_len(
    dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: Option<usize>,
    n_layers: usize,
    vocab_size: usize,
) -> usize {
    let hd = head_dim.unwrap_or(dim / n_heads);
    let qd = n_heads * hd;
    let kvd = n_kv_heads * hd;
    let d = dim;
    let h = hidden_dim;
    let nl = n_layers;
    let v = vocab_size;

    // embed
    let mut total = v * d;

    // per-layer weights
    let per_layer = d       // rms_att
        + qd * d            // wq
        + kvd * d           // wk
        + kvd * d           // wv
        + d * qd            // wo
        + d                 // rms_ffn
        + h * d             // w1
        + d * h             // w2
        + h * d; // w3

    total += per_layer * nl;

    // rms_final
    total += d;

    total
}

/// Build a pseudo-random f32 weight vector using a simple LCG seeded with
/// `seed`. Weights are drawn from [-0.02, 0.02] — small enough to keep
/// activations in a healthy range without any normalisation warm-up.
fn random_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // 64-bit LCG (Knuth)
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map high 32 bits to [-0.02, 0.02]
        let u = ((state >> 32) as u32) as f32 / u32::MAX as f32; // [0, 1]
        out.push((u - 0.5) * 0.04);
    }
    out
}

/// Build a deterministic batch of (input, target) token sequences.
///
/// Each sequence is `seq_len` tokens long. Inputs cycle through [0, vocab)
/// and targets are shifted by one position (next-token prediction).
fn make_batch(batch_size: usize, seq_len: usize, vocab_size: usize) -> Vec<(Vec<u16>, Vec<u16>)> {
    (0..batch_size)
        .map(|b| {
            let input: Vec<u16> = (0..seq_len)
                .map(|t| ((b * seq_len + t) % vocab_size) as u16)
                .collect();
            let target: Vec<u16> = (0..seq_len)
                .map(|t| ((b * seq_len + t + 1) % vocab_size) as u16)
                .collect();
            (input, target)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that ANE training reduces loss on a trivial memorisation task.
///
/// Uses a tiny model (dim=64, hidden=128, 2 layers, vocab=1000, seq=32) so
/// that kernel compilation finishes quickly and memory pressure stays low.
/// The model is trained on a fixed synthetic next-token prediction task.
/// We assert:
///   - no NaN / Inf appears in any loss value, and
///   - the loss after 10 steps is strictly lower than the initial loss.
#[test]
#[ignore]
fn ane_training_reduces_loss() {
    let dim = 64;
    let hidden_dim = 128;
    let n_heads = 4;
    let n_kv_heads = 2;
    let head_dim = Some(16); // 4 heads × 16 = 64 = dim
    let n_layers = 2;
    let vocab_size = 1000;
    let seq_len = 32;

    let config = DynamicAneTrainerConfig {
        dim,
        hidden_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        n_layers,
        vocab_size,
        seq_len,
        learning_rate: 1e-3,
        adam_beta1: 0.9,
        adam_beta2: 0.999,
        adam_eps: 1e-8,
        gradient_clip_norm: 1.0,
        accum_steps: 1,
        warmup_steps: 0,
        min_lr_ratio: 0.1,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        loss_scale: 1.0,
        embedding_lr: None,
    };

    let mut trainer = DynamicAneTrainer::new(config.clone());

    // Load reproducible pseudo-random weights
    let n_weights = flat_weight_len(
        dim, hidden_dim, n_heads, n_kv_heads, head_dim, n_layers, vocab_size,
    );
    let weights = random_weights(n_weights, 42);
    trainer.load_weights_flat(&weights);

    // Compile ANE kernels (requires ANE hardware)
    trainer
        .compile_kernels()
        .expect("ANE kernel compilation failed");

    // Fixed synthetic batch: 4 sequences of sequential token IDs
    let batch = make_batch(4, seq_len, vocab_size);

    let mut losses = Vec::with_capacity(10);

    for _step in 0..10 {
        let loss = trainer
            .train_batch(&batch, /*max_steps=*/ 1)
            .expect("train_batch failed");

        assert!(
            loss.is_finite(),
            "Loss is not finite at step {_step}: {loss}"
        );
        losses.push(loss);
    }

    let first_loss = losses[0];
    let last_loss = losses[9];

    assert!(
        last_loss < first_loss,
        "Expected loss to decrease over 10 steps: first={first_loss:.4}, last={last_loss:.4}"
    );
}

/// Verify ANE kernel compilation succeeds for multiple model configurations.
///
/// Each configuration exercises a different combination of GQA ratios and
/// dimension scales. Compilation is the expensive gate; we only verify that
/// it returns `Ok(())`.
#[test]
#[ignore]
fn ane_kernels_compile_all_configs() {
    struct Case {
        name: &'static str,
        dim: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: Option<usize>,
    }

    let cases = [
        Case {
            name: "small",
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: Some(16),
        },
        Case {
            name: "medium",
            dim: 256,
            hidden_dim: 512,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: None, // 256 / 8 = 32
        },
        Case {
            name: "gqa_extreme",
            dim: 512,
            hidden_dim: 1024,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: None, // 512 / 16 = 32
        },
        Case {
            name: "large_hidden",
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
            n_kv_heads: 4,
            head_dim: Some(64),
        },
    ];

    for case in &cases {
        let config = DynamicAneTrainerConfig {
            dim: case.dim,
            hidden_dim: case.hidden_dim,
            n_heads: case.n_heads,
            n_kv_heads: case.n_kv_heads,
            head_dim: case.head_dim,
            n_layers: 2,
            vocab_size: 1000,
            seq_len: 32,
            learning_rate: 1e-3,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            gradient_clip_norm: 1.0,
            accum_steps: 1,
            warmup_steps: 0,
            min_lr_ratio: 0.1,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            loss_scale: 1.0,
            embedding_lr: None,
        };

        let mut trainer = DynamicAneTrainer::new(config.clone());

        // Load unit weights so all activations are non-zero (avoids degenerate
        // zero-output kernels that could mask compilation bugs).
        let n_weights = flat_weight_len(
            case.dim,
            case.hidden_dim,
            case.n_heads,
            case.n_kv_heads,
            case.head_dim,
            2,
            1000,
        );
        let weights = vec![0.01f32; n_weights];
        trainer.load_weights_flat(&weights);

        let result = trainer.compile_kernels();
        assert!(
            result.is_ok(),
            "compile_kernels() failed for config '{}': {:?}",
            case.name,
            result.err()
        );
    }
}
