//! Extend kernels against a CPU reference, on ANE hardware.
//!
//!     cargo test -p pmetal-metal --test ane_extend --release -- --ignored

#![cfg(target_os = "macos")]

use pmetal_metal::ane::extend::{ExtendConfig, ExtendModel, LayerWeights, WeightFormat};
use pmetal_metal::ane::kernel::quantize_int8_rows;

/// `n` values in `[-spread, spread]` from a Knuth LCG.
fn values(n: usize, seed: u64, center: f32, spread: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            center + (u - 0.5) * 2.0 * spread
        })
        .collect()
}

/// Random weights for one layer, big enough that attention depends on
/// position. `qk_norm` centers the q/k norm weights, which set how large the
/// attention scores get.
fn random_layer(cfg: &ExtendConfig, seed: u64, qk_norm: f32) -> LayerWeights {
    let (d, h, qd, kvd, hd) = (
        cfg.dim,
        cfg.hidden_dim,
        cfg.q_dim(),
        cfg.kv_dim(),
        cfg.head_dim,
    );
    LayerWeights {
        rms_att: values(d, seed, 1.0, 0.2),
        wq: values(qd * d, seed + 1, 0.0, 0.5),
        wk: values(kvd * d, seed + 2, 0.0, 0.5),
        wv: values(kvd * d, seed + 3, 0.0, 0.3),
        wo: values(d * qd, seed + 4, 0.0, 0.2),
        q_norm: values(hd, seed + 5, qk_norm, 0.3),
        k_norm: values(hd, seed + 6, qk_norm, 0.3),
        rms_ffn: values(d, seed + 7, 1.0, 0.2),
        w_gate: values(h * d, seed + 8, 0.0, 0.2),
        w_up: values(h * d, seed + 9, 0.0, 0.2),
        w_down: values(d * h, seed + 10, 0.0, 0.2),
    }
}

/// The projection weights the kernel computes with.
fn as_stored(layer: &LayerWeights, format: WeightFormat, cfg: &ExtendConfig) -> LayerWeights {
    let (d, h, qd, kvd) = (cfg.dim, cfg.hidden_dim, cfg.q_dim(), cfg.kv_dim());
    let stored = |w: &[f32], rows: usize, cols: usize| -> Vec<f32> {
        match format {
            WeightFormat::Fp16 => w.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect(),
            WeightFormat::Int8 => {
                let (q, s) = quantize_int8_rows(w, rows, cols);
                q.iter()
                    .enumerate()
                    .map(|(i, &v)| v as f32 * s[i / cols].to_f32())
                    .collect()
            }
        }
    };
    LayerWeights {
        wq: stored(&layer.wq, qd, d),
        wk: stored(&layer.wk, kvd, d),
        wv: stored(&layer.wv, kvd, d),
        wo: stored(&layer.wo, d, qd),
        w_gate: stored(&layer.w_gate, h, d),
        w_up: stored(&layer.w_up, h, d),
        w_down: stored(&layer.w_down, d, h),
        ..layer.clone()
    }
}

/// The residual stream after every layer for `x`, `[dim, s]` channel-first,
/// computed causally over all `s` positions in f64.
fn reference(
    cfg: &ExtendConfig,
    layers: &[LayerWeights],
    x: &[f32],
    s: usize,
    theta: f64,
) -> Vec<f64> {
    let (d, h, nh, nkv, hd) = (
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
    );
    let (qd, kvd) = (cfg.q_dim(), cfg.kv_dim());
    let eps = cfg.rms_norm_eps as f64;
    let mut x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    // RMSNorm of the `rows` values v[at(0)], v[at(1)], ...
    let rms = |v: &[f64], g: &[f32], rows: usize, at: &dyn Fn(usize) -> usize| -> Vec<f64> {
        let ms = (0..rows).map(|r| v[at(r)].powi(2)).sum::<f64>() / rows as f64;
        let k = 1.0 / (ms + eps).sqrt();
        (0..rows).map(|r| v[at(r)] * k * g[r] as f64).collect()
    };
    let mat = |w: &[f32], rows: usize, cols: usize, v: &[f64]| -> Vec<f64> {
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] as f64 * v[c]).sum())
            .collect()
    };
    let rope = |v: &mut [f64], t: usize| {
        for i in 0..hd / 2 {
            let f = theta.powf(-2.0 * i as f64 / hd as f64);
            let (sn, cs) = (t as f64 * f).sin_cos();
            let (a, b) = (v[i], v[i + hd / 2]);
            v[i] = a * cs - b * sn;
            v[i + hd / 2] = a * sn + b * cs;
        }
    };
    for layer in layers {
        // Per position: normed input, q (per head), k, v.
        let mut qs = Vec::new();
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for t in 0..s {
            let xn = rms(&x, &layer.rms_att, d, &|r| r * s + t);
            let q = mat(&layer.wq, qd, d, &xn);
            let k = mat(&layer.wk, kvd, d, &xn);
            let v = mat(&layer.wv, kvd, d, &xn);
            let mut qh: Vec<Vec<f64>> = (0..nh)
                .map(|hh| rms(&q, &layer.q_norm, hd, &|r| hh * hd + r))
                .collect();
            let mut kh: Vec<Vec<f64>> = (0..nkv)
                .map(|hh| rms(&k, &layer.k_norm, hd, &|r| hh * hd + r))
                .collect();
            qh.iter_mut().for_each(|q| rope(q, t));
            kh.iter_mut().for_each(|k| rope(k, t));
            qs.push(qh);
            ks.push(kh);
            vs.push(v);
        }
        let mut x2 = x.clone();
        for t in 0..s {
            let mut attn = vec![0.0f64; qd];
            for hh in 0..nh {
                let g = hh / (nh / nkv);
                let sc: Vec<f64> = (0..=t)
                    .map(|u| {
                        (0..hd).map(|i| qs[t][hh][i] * ks[u][g][i]).sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = sc.iter().map(|v| (v - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for i in 0..hd {
                    attn[hh * hd + i] = (0..=t).map(|u| e[u] / z * vs[u][g * hd + i]).sum();
                }
            }
            let o = mat(&layer.wo, d, qd, &attn);
            for c in 0..d {
                x2[c * s + t] += o[c];
            }
        }
        for t in 0..s {
            let xn = rms(&x2, &layer.rms_ffn, d, &|r| r * s + t);
            let g = mat(&layer.w_gate, h, d, &xn);
            let u = mat(&layer.w_up, h, d, &xn);
            let a: Vec<f64> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let y = mat(&layer.w_down, d, h, &a);
            for c in 0..d {
                x2[c * s + t] += y[c];
            }
        }
        x = x2;
    }
    x
}

/// A small model: three layers in two chunks.
fn tiny(format: WeightFormat) -> ExtendConfig {
    ExtendConfig {
        dim: 64,
        hidden_dim: 128,
        n_heads: 4,
        n_kv_heads: 2,
        // 32 keeps V's cached rows 64 bytes, the ANE's row alignment, and
        // makes q_dim (128) differ from dim.
        head_dim: 32,
        width: 8,
        capacity: 64,
        rms_norm_eps: 1e-6,
        weights: format,
    }
}

/// Run `steps` tokens at a time through `n_layers` random layers, in chunks
/// of two, and check every position against the reference.
fn case(cfg: ExtendConfig, n_layers: usize, steps: &[usize], qk_norm: f32) {
    let format = cfg.weights;
    let theta = 10_000.0f32;
    let layers: Vec<LayerWeights> = (0..n_layers as u64)
        .map(|i| random_layer(&cfg, 100 * (i + 1), qk_norm))
        .collect();
    let mut model = ExtendModel::compile(cfg.clone(), theta, layers.len(), 2, |i| {
        Ok(layers[i].clone())
    })
    .expect("extend kernels compile");

    let s: usize = steps.iter().sum();
    let d = cfg.dim;
    let x = values(d * s, 7, 0.0, 1.0);
    let column = |from: usize, n: usize| -> Vec<f32> {
        (0..d)
            .flat_map(|c| x[c * s + from..c * s + from + n].to_vec())
            .collect()
    };
    let mut got = vec![0.0f32; d * s];
    let mut at = 0;
    for &n in steps {
        let out = model.extend(&column(at, n), n).expect("extend");
        for c in 0..d {
            got[c * s + at..c * s + at + n].copy_from_slice(&out[c * n..(c + 1) * n]);
        }
        at += n;
    }
    assert_eq!(model.position(), s);

    let stored: Vec<LayerWeights> = layers.iter().map(|l| as_stored(l, format, &cfg)).collect();
    let want = reference(&cfg, &stored, &x, s, theta as f64);
    // `out` holds positions `from..from + n`, `[dim, n]`.
    let check = |out: &[f32], from: usize, n: usize, what: &str| {
        for t in 0..n {
            let at = |c: usize| (out[c * n + t] as f64, want[c * s + from + t]);
            let err: f64 = (0..d)
                .map(|c| (at(c).0 - at(c).1).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm: f64 = (0..d).map(|c| at(c).1.powi(2)).sum::<f64>().sqrt();
            assert!(
                err / norm < 1e-2,
                "{format:?} {what}, position {}: relative error {:.2e}",
                from + t,
                err / norm
            );
        }
    };
    check(&got, 0, s, "as run");

    // Rewinding and running the last 4 again, now as one block, recomputes
    // them.
    let from = s - 4;
    model.truncate(from);
    let again = model
        .extend(&column(from, 4), 4)
        .expect("extend after truncate");
    check(&again, from, 4, "after truncate");
}

#[test]
#[ignore = "requires ANE hardware"]
fn extend_matches_cpu_reference_fp16() {
    // Prefill in full blocks, decode one at a time, then a partial block.
    case(tiny(WeightFormat::Fp16), 3, &[8, 8, 1, 1, 1, 3], 1.0);
}

#[test]
#[ignore = "requires ANE hardware"]
fn extend_matches_cpu_reference_int8() {
    case(tiny(WeightFormat::Int8), 3, &[8, 8, 1, 1, 1, 3], 1.0);
}

/// Qwen3-0.6B's shapes: a short prompt, then single tokens. Its q/k norm
/// weights put attention scores in the tens, as a real model's are: past
/// ~11, where exp overflows fp16, the ANE's reduce_log_sum_exp broke decode.
#[test]
#[ignore = "requires ANE hardware"]
fn extend_matches_cpu_reference_qwen3_shape() {
    let cfg = ExtendConfig {
        dim: 1024,
        hidden_dim: 3072,
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 128,
        width: 32,
        capacity: 512,
        rms_norm_eps: 1e-6,
        weights: WeightFormat::Fp16,
    };
    case(cfg, 2, &[5, 1, 1, 1, 1], 3.0);
}

/// Time one chunk of Qwen3-4B-shaped layers (random int8 weights) and
/// extrapolate to the model's 36. Prints; doesn't assert.
#[test]
#[ignore = "requires ANE hardware; benchmark"]
fn extend_throughput_qwen3_4b_shape() {
    let layers_per_chunk = 4;
    for (width, capacity) in [(32usize, 1024usize), (32, 4096), (64, 4096)] {
        let cfg = ExtendConfig {
            dim: 2560,
            hidden_dim: 9728,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            width,
            capacity,
            rms_norm_eps: 1e-6,
            weights: WeightFormat::Int8,
        };
        let layer = random_layer(&cfg, 1, 1.0);
        let start = std::time::Instant::now();
        let mut model =
            ExtendModel::compile(cfg.clone(), 1e6, layers_per_chunk, layers_per_chunk, |_| {
                Ok(layer.clone())
            })
            .expect("compile");
        let compile = start.elapsed();
        let x = values(cfg.dim, 3, 0.0, 1.0);
        for _ in 0..3 {
            model.extend(&x, 1).unwrap();
        }
        let iters = 20;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            model.extend(&x, 1).unwrap();
        }
        let per_call = start.elapsed().as_secs_f64() / iters as f64;
        let per_token_36 = per_call * 36.0 / layers_per_chunk as f64;
        eprintln!(
            "W={width} L={capacity}: compile {compile:.1?} for {layers_per_chunk} layers; \
             {:.2} ms per call ({:.2} ms/layer) -> 36 layers {:.1} ms = {:.1} passes/s, \
             x{width} tokens per pass if all accepted",
            per_call * 1e3,
            per_call * 1e3 / layers_per_chunk as f64,
            per_token_36 * 1e3,
            1.0 / per_token_36,
        );
    }
}
