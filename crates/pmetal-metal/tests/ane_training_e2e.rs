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

// ---------------------------------------------------------------------------
// Forward parity against a CPU reference
// ---------------------------------------------------------------------------

struct Dims {
    d: usize,
    h: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    nl: usize,
    v: usize,
    s: usize,
}

/// `n` values in `[center - spread, center + spread]` from a Knuth LCG.
fn lcg_values(n: usize, seed: u64, center: f32, spread: f32) -> Vec<f32> {
    random_weights(n, seed)
        .into_iter()
        .map(|u| center + u / 0.02 * spread)
        .collect()
}

/// How big the residual stream is.
#[derive(Clone, Copy, Debug)]
enum Residual {
    /// Embeddings within ±0.3.
    Small,
    /// Embeddings within ±60, past where an fp16 sum of squares over 64
    /// channels overflows (RMS 32), as a real checkpoint's residual is. Wo
    /// and W2 are scaled up so the blocks still move a residual that size,
    /// and the final norm down to keep the tied logits in range.
    Large,
}

/// Weights in `load_weights_flat` order, sized so attention depends on
/// position: with the ±0.02 of `random_weights`, attention is close to
/// uniform and a model with no positional signal scores nearly the same.
fn parity_weights(m: &Dims, residual: Residual) -> Vec<f32> {
    let (d, h, qd, kvd) = (m.d, m.h, m.nh * m.hd, m.nkv * m.hd);
    let (embed, block_out, final_norm) = match residual {
        Residual::Small => (0.3, 0.2, 1.0),
        Residual::Large => (60.0, 20.0, 0.005),
    };
    let mut w = lcg_values(m.v * d, 1, 0.0, embed);
    for l in 0..m.nl as u64 {
        let seed = 100 * (l + 1);
        w.extend(lcg_values(d, seed, 1.0, 0.2));
        w.extend(lcg_values(qd * d, seed + 1, 0.0, 0.5));
        w.extend(lcg_values(kvd * d, seed + 2, 0.0, 0.5));
        w.extend(lcg_values(kvd * d, seed + 3, 0.0, 0.2));
        w.extend(lcg_values(d * qd, seed + 4, 0.0, block_out));
        w.extend(lcg_values(d, seed + 5, 1.0, 0.2));
        w.extend(lcg_values(h * d, seed + 6, 0.0, 0.2));
        w.extend(lcg_values(d * h, seed + 7, 0.0, block_out));
        w.extend(lcg_values(h * d, seed + 8, 0.0, 0.2));
    }
    w.extend(lcg_values(d, 9, final_norm, 0.2 * final_norm));
    w
}

/// Mean next-token cross-entropy of a Llama forward pass on the CPU, in f64,
/// written independently of the trainer. `rope_theta: None` leaves out RoPE.
fn reference_loss(
    m: &Dims,
    w: &[f32],
    input: &[u16],
    target: &[u16],
    eps: f64,
    rope_theta: Option<f64>,
) -> f64 {
    let (d, h, s, hd) = (m.d, m.h, m.s, m.hd);
    let (qd, kvd) = (m.nh * hd, m.nkv * hd);
    let mut at = 0;
    let mut take = |n: usize| {
        let t: Vec<f64> = w[at..at + n].iter().map(|&x| x as f64).collect();
        at += n;
        t
    };
    // All activations channel-first: [channels, s].
    let rmsnorm = |x: &[f64], g: &[f64]| {
        let mut out = vec![0.0; d * s];
        for t in 0..s {
            let ms = (0..d).map(|c| x[c * s + t] * x[c * s + t]).sum::<f64>() / d as f64;
            let r = 1.0 / (ms + eps).sqrt();
            for c in 0..d {
                out[c * s + t] = x[c * s + t] * r * g[c];
            }
        }
        out
    };
    let matmul = |wt: &[f64], rows: usize, cols: usize, x: &[f64]| {
        let mut out = vec![0.0; rows * s];
        for r in 0..rows {
            for c in 0..cols {
                let wv = wt[r * cols + c];
                for t in 0..s {
                    out[r * s + t] += wv * x[c * s + t];
                }
            }
        }
        out
    };
    let rope = |x: &mut [f64], heads: usize| {
        let Some(theta) = rope_theta else { return };
        let half = hd / 2;
        for head in 0..heads {
            for i in 0..half {
                let freq = theta.powf(-2.0 * i as f64 / hd as f64);
                for t in 0..s {
                    let (sin, cos) = (t as f64 * freq).sin_cos();
                    let a = (head * hd + i) * s + t;
                    let b = a + half * s;
                    let (x1, x2) = (x[a], x[b]);
                    x[a] = x1 * cos - x2 * sin;
                    x[b] = x1 * sin + x2 * cos;
                }
            }
        }
    };

    let embed = take(m.v * d);
    let mut x = vec![0.0; d * s];
    for (t, &tok) in input.iter().enumerate() {
        for c in 0..d {
            x[c * s + t] = embed[tok as usize * d + c];
        }
    }
    for _ in 0..m.nl {
        let (rms_att, wq, wk, wv, wo) = (
            take(d),
            take(qd * d),
            take(kvd * d),
            take(kvd * d),
            take(d * qd),
        );
        let (rms_ffn, w1, w2, w3) = (take(d), take(h * d), take(d * h), take(h * d));

        let xn = rmsnorm(&x, &rms_att);
        let (mut q, mut k, v) = (
            matmul(&wq, qd, d, &xn),
            matmul(&wk, kvd, d, &xn),
            matmul(&wv, kvd, d, &xn),
        );
        rope(&mut q, m.nh);
        rope(&mut k, m.nkv);
        let mut attn = vec![0.0; qd * s];
        for head in 0..m.nh {
            let kv = head / (m.nh / m.nkv);
            for t in 0..s {
                let scores: Vec<f64> = (0..=t)
                    .map(|u| {
                        (0..hd)
                            .map(|i| q[(head * hd + i) * s + t] * k[(kv * hd + i) * s + u])
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|sc| (sc - max).exp()).collect();
                let z: f64 = e.iter().sum();
                for i in 0..hd {
                    attn[(head * hd + i) * s + t] =
                        (0..=t).map(|u| e[u] / z * v[(kv * hd + i) * s + u]).sum();
                }
            }
        }
        let o = matmul(&wo, d, qd, &attn);
        for i in 0..d * s {
            x[i] += o[i];
        }
        let xn = rmsnorm(&x, &rms_ffn);
        let (g, u) = (matmul(&w1, h, d, &xn), matmul(&w3, h, d, &xn));
        let act: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let down = matmul(&w2, d, h, &act);
        for i in 0..d * s {
            x[i] += down[i];
        }
    }
    let xf = rmsnorm(&x, &take(d));

    let mut loss = 0.0;
    for t in 0..s {
        let logits: Vec<f64> = (0..m.v)
            .map(|tok| (0..d).map(|c| embed[tok * d + c] * xf[c * s + t]).sum())
            .collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
        loss += lse - logits[target[t] as usize];
    }
    loss / s as f64
}

fn forward_parity_case(seq_len: usize, residual: Residual) {
    let m = Dims {
        d: 64,
        h: 128,
        nh: 4,
        nkv: 2,
        hd: 16,
        nl: 2,
        v: 256,
        s: seq_len,
    };
    let rope_theta = 10_000.0;
    let eps = 1e-5;
    let config = DynamicAneTrainerConfig {
        dim: m.d,
        hidden_dim: m.h,
        n_heads: m.nh,
        n_kv_heads: m.nkv,
        head_dim: Some(m.hd),
        n_layers: m.nl,
        vocab_size: m.v,
        seq_len: m.s,
        rms_norm_eps: eps,
        rope_theta,
        accum_steps: 1,
        warmup_steps: 0,
        ..DynamicAneTrainerConfig::default()
    };
    let weights = parity_weights(&m, residual);
    let tokens: Vec<u16> = random_weights(m.s + 1, 7)
        .iter()
        .map(|u| ((u + 0.02) / 0.04 * (m.v - 1) as f32) as u16)
        .collect();
    let (input, target) = (tokens[..m.s].to_vec(), tokens[1..].to_vec());

    let want = reference_loss(
        &m,
        &weights,
        &input,
        &target,
        eps as f64,
        Some(rope_theta as f64),
    );
    let without_rope = reference_loss(&m, &weights, &input, &target, eps as f64, None);

    let mut trainer = DynamicAneTrainer::new(config);
    trainer.load_weights_flat(&weights);
    trainer
        .compile_kernels()
        .expect("ANE kernel compilation failed");
    // The loss comes from the forward pass, before this step's update.
    let got = trainer
        .train_batch(&[(input, target)], 1)
        .expect("train_batch") as f64;

    let tolerance = 0.02;
    assert!(
        (want - without_rope).abs() > 5.0 * tolerance,
        "seq {seq_len} {residual:?}: the check can't tell RoPE from none ({want} vs {without_rope})"
    );
    assert!(
        (got - want).abs() < tolerance,
        "seq {seq_len} {residual:?}: ANE loss {got:.4}, CPU reference {want:.4} (without RoPE: {without_rope:.4})"
    );
}

/// The trainer's forward pass computes the model the checkpoint defines.
/// It once left out RoPE entirely, so a pretrained model started from the
/// wrong loss and degraded with training, while the synthetic loss-goes-down
/// test above still passed. Sequence 32 runs the fused attention kernel;
/// 1024 is past its size limit and takes the decomposed path (ANE
/// projections, attention on the CPU). The large residual is a real
/// checkpoint's scale, where an RMSNorm computed in fp16 on the ANE
/// overflowed and the loss was NaN.
#[test]
#[ignore]
fn ane_training_forward_matches_cpu_reference() {
    forward_parity_case(32, Residual::Small);
    forward_parity_case(1024, Residual::Small);
    forward_parity_case(32, Residual::Large);
}

/// Each tensor's `(name, offset, len)` in the flat layout.
fn flat_tensors(m: &Dims) -> Vec<(String, usize, usize)> {
    let (d, h, qd, kvd) = (m.d, m.h, m.nh * m.hd, m.nkv * m.hd);
    let mut out = vec![("embed".to_string(), 0, m.v * d)];
    let mut at = m.v * d;
    for l in 0..m.nl {
        for (name, len) in [
            ("rms_att", d),
            ("wq", qd * d),
            ("wk", kvd * d),
            ("wv", kvd * d),
            ("wo", d * qd),
            ("rms_ffn", d),
            ("w1", h * d),
            ("w2", d * h),
            ("w3", h * d),
        ] {
            out.push((format!("layers.{l}.{name}"), at, len));
            at += len;
        }
    }
    out.push(("rms_final".to_string(), at, d));
    out
}

fn backward_sign_case(seq_len: usize, samples: usize) {
    let m = Dims {
        d: 64,
        h: 128,
        nh: 4,
        nkv: 2,
        hd: 16,
        nl: 2,
        v: 256,
        s: seq_len,
    };
    let (rope_theta, eps, lr) = (10_000.0f32, 1e-5f32, 1e-3f32);
    let config = DynamicAneTrainerConfig {
        dim: m.d,
        hidden_dim: m.h,
        n_heads: m.nh,
        n_kv_heads: m.nkv,
        head_dim: Some(m.hd),
        n_layers: m.nl,
        vocab_size: m.v,
        seq_len: m.s,
        rms_norm_eps: eps,
        rope_theta,
        learning_rate: lr,
        accum_steps: 1,
        warmup_steps: 0,
        ..DynamicAneTrainerConfig::default()
    };
    let weights = parity_weights(&m, Residual::Small);
    let tokens: Vec<u16> = random_weights(m.s + 1, 7)
        .iter()
        .map(|u| ((u + 0.02) / 0.04 * (m.v - 1) as f32) as u16)
        .collect();
    let (input, target) = (tokens[..m.s].to_vec(), tokens[1..].to_vec());

    let mut trainer = DynamicAneTrainer::new(config);
    trainer.load_weights_flat(&weights);
    trainer
        .compile_kernels()
        .expect("ANE kernel compilation failed");
    trainer
        .train_batch(&[(input.clone(), target.clone())], 1)
        .expect("train_batch");
    let after = trainer.export_weights_flat();

    let loss_at =
        |w: &[f32]| reference_loss(&m, w, &input, &target, eps as f64, Some(rope_theta as f64));
    let mut failures = Vec::new();
    for (name, offset, len) in flat_tensors(&m) {
        let step = (len / samples).max(1);
        let mut grads = Vec::new();
        for i in (0..len).step_by(step).take(samples) {
            let at = offset + i;
            let h = 1e-3f32;
            let mut w = weights.clone();
            w[at] += h;
            let up = loss_at(&w);
            w[at] -= 2.0 * h;
            let down = loss_at(&w);
            grads.push((at, (up - down) / (2.0 * h as f64)));
        }
        let largest = grads.iter().fold(0f64, |acc, (_, g)| acc.max(g.abs()));
        let (mut agree, mut checked) = (0, 0);
        for &(at, g) in &grads {
            if g.abs() < 0.1 * largest || g.abs() < 1e-4 {
                continue;
            }
            checked += 1;
            let moved = after[at] - weights[at];
            if moved != 0.0 && (moved < 0.0) == (g > 0.0) {
                agree += 1;
            }
        }
        if checked == 0 || agree < checked {
            failures.push(format!("{name}: {agree}/{checked}"));
        }
    }
    assert!(
        failures.is_empty(),
        "seq {seq_len}: gradient signs disagree for {failures:?}"
    );
}

/// The backward pass computes the gradient of the loss. Adam's first step
/// moves each weight by `-lr * sign(g)`, so after one step every weight's
/// move should oppose a finite-difference gradient of the CPU reference.
/// A scrambled weight layout or a zero dQ/dK/dV agrees only by chance.
#[test]
#[ignore]
fn ane_training_gradients_match_cpu_reference() {
    backward_sign_case(32, 12);
    backward_sign_case(1024, 6);
}
