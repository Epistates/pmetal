//! Attention against a float64 implementation: every mask, grouped heads,
//! logit softcapping, attention sinks and a key sequence longer than the
//! query's; values in f32 and bf16, and the gradients of the queries, keys,
//! values and sinks against central finite differences.

use super::*;
use crate::compat::ops;
use crate::f64_reference::*;
use serial_test::serial;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mask {
    None,
    Causal,
    Window(usize),
    /// An additive `[1, 1, Lq, Lk]` array with a pattern of `-inf`.
    Custom,
}

#[derive(Clone)]
struct Case {
    b: usize,
    h: usize,
    hk: usize,
    lq: usize,
    lk: usize,
    d: usize,
    dv: usize,
    scale: f64,
    softcap: Option<f64>,
    mask: Mask,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    sinks: Option<Vec<f64>>,
}

#[derive(Clone, Copy)]
struct Spec {
    b: usize,
    h: usize,
    hk: usize,
    lq: usize,
    lk: usize,
    d: usize,
    dv: usize,
    softcap: Option<f64>,
    mask: Mask,
    sinks: bool,
}

impl Spec {
    fn new(h: usize, hk: usize, lq: usize, lk: usize) -> Self {
        Self {
            b: 2,
            h,
            hk,
            lq,
            lk,
            d: 8,
            dv: 8,
            softcap: None,
            mask: Mask::Causal,
            sinks: false,
        }
    }

    fn mask(self, mask: Mask) -> Self {
        Self { mask, ..self }
    }

    fn softcap(self, cap: f64) -> Self {
        Self {
            softcap: Some(cap),
            ..self
        }
    }

    fn sinks(self) -> Self {
        Self {
            sinks: true,
            ..self
        }
    }
}

fn case(spec: Spec, seed: u64) -> Case {
    let mut rng = Rng(seed);
    let mut draw = |n: usize, scale: f64| -> Vec<f64> {
        (0..n)
            .map(|_| round32(scale * rng.normal()))
            .collect::<Vec<_>>()
    };
    let Spec {
        b,
        h,
        hk,
        lq,
        lk,
        d,
        dv,
        ..
    } = spec;
    // Large enough scores that a softcap bends them.
    let q = draw(b * h * lq * d, 2.0);
    let k = draw(b * hk * lk * d, 2.0);
    let v = draw(b * hk * lk * dv, 1.0);
    let sinks = spec.sinks.then(|| draw(h, 1.5));
    Case {
        b,
        h,
        hk,
        lq,
        lk,
        d,
        dv,
        scale: 1.0 / (d as f64).sqrt(),
        softcap: spec.softcap,
        mask: spec.mask,
        q,
        k,
        v,
        sinks,
    }
}

/// Whether query `i` may attend to key `j` (a bottom-right aligned band, as
/// the cached continuation of a prompt needs), and the additive value of the
/// custom mask there.
fn allowed(c: &Case, i: usize, j: usize) -> bool {
    let reach = (i + c.lk) as i64 - c.lq as i64; // the last key query `i` sees
    let j = j as i64;
    match c.mask {
        Mask::None => true,
        Mask::Causal => j <= reach,
        Mask::Window(w) => j <= reach && j > reach - w as i64,
        Mask::Custom => custom_allowed(i, j as usize),
    }
}

/// The custom mask's pattern: keys with `(i + 2j) % 5 == 4` are hidden, which
/// leaves every row some key to attend to.
fn custom_allowed(i: usize, j: usize) -> bool {
    (i + 2 * j) % 5 != 4
}

/// The attention output `[B, H, Lq, Dv]` in f64.
fn reference(c: &Case) -> Vec<f64> {
    let group = c.h / c.hk;
    let mut out = vec![0.0; c.b * c.h * c.lq * c.dv];
    for b in 0..c.b {
        for h in 0..c.h {
            let kh = h / group;
            for i in 0..c.lq {
                let qo = ((b * c.h + h) * c.lq + i) * c.d;
                let mut logits: Vec<f64> = Vec::with_capacity(c.lk + 1);
                for j in 0..c.lk {
                    if !allowed(c, i, j) {
                        logits.push(f64::NEG_INFINITY);
                        continue;
                    }
                    let ko = ((b * c.hk + kh) * c.lk + j) * c.d;
                    let dot: f64 = (0..c.d).map(|x| c.q[qo + x] * c.k[ko + x]).sum();
                    let mut s = c.scale * dot;
                    if let Some(cap) = c.softcap {
                        s = cap * (s / cap).tanh();
                    }
                    logits.push(s);
                }
                if let Some(sinks) = &c.sinks {
                    logits.push(sinks[h]);
                }
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                let oo = ((b * c.h + h) * c.lq + i) * c.dv;
                for (j, weight) in weights.iter().take(c.lk).enumerate() {
                    let vo = ((b * c.hk + kh) * c.lk + j) * c.dv;
                    for x in 0..c.dv {
                        out[oo + x] += weight / total * c.v[vo + x];
                    }
                }
            }
        }
    }
    out
}

fn config(c: &Case) -> FusedAttentionConfig {
    let mut config =
        FusedAttentionConfig::new(c.h as i32, c.hk as i32, c.d as i32).with_scale(c.scale as f32);
    config = config.with_mask_type(match c.mask {
        Mask::None | Mask::Custom => AttentionMaskType::None,
        Mask::Causal => AttentionMaskType::Causal,
        Mask::Window(w) => AttentionMaskType::SlidingWindow(w as i32),
    });
    if let Some(cap) = c.softcap {
        config = config.with_logit_softcapping(cap as f32);
    }
    config
}

fn custom_mask(c: &Case) -> Array {
    let values: Vec<f32> = (0..c.lq * c.lk)
        .map(|n| {
            if custom_allowed(n / c.lk, n % c.lk) {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    Array::from_f32_slice(&values, &[1, 1, c.lq as i32, c.lk as i32])
}

/// The implementation under test: the inputs `[q, k, v]` (and the sinks when
/// the case has them) to the output.
fn run(c: &Case, x: &[Array]) -> Array {
    let mask = (c.mask == Mask::Custom).then(|| custom_mask(c));
    let config = config(c);
    match &c.sinks {
        Some(_) => sink_sdpa(&x[0], &x[1], &x[2], &config, mask.as_ref(), &x[3]).unwrap(),
        None => fused_sdpa(&x[0], &x[1], &x[2], &config, mask.as_ref()).unwrap(),
    }
}

fn fields(c: &Case) -> Vec<Vec<f64>> {
    let mut f = vec![c.q.clone(), c.k.clone(), c.v.clone()];
    f.extend(c.sinks.clone());
    f
}

fn shapes(c: &Case) -> Vec<Vec<usize>> {
    let mut s = vec![
        vec![c.b, c.h, c.lq, c.d],
        vec![c.b, c.hk, c.lk, c.d],
        vec![c.b, c.hk, c.lk, c.dv],
    ];
    if c.sinks.is_some() {
        s.push(vec![c.h]);
    }
    s
}

fn names(c: &Case) -> Vec<&'static str> {
    let mut n = vec!["q", "k", "v"];
    if c.sinks.is_some() {
        n.push("sinks");
    }
    n
}

fn describe(c: &Case) -> String {
    format!(
        "{:?}, {}/{} heads, Lq {}, Lk {}{}{}",
        c.mask,
        c.h,
        c.hk,
        c.lq,
        c.lk,
        c.softcap
            .map_or(String::new(), |cap| format!(", softcap {cap}")),
        if c.sinks.is_some() { ", sinks" } else { "" }
    )
}

fn check_values(c: &Case, dtype: Dtype, rel: f64) {
    let mut x: Vec<Array> = fields(c)
        .iter()
        .zip(shapes(c))
        .map(|(f, s)| arr(f, &s))
        .collect();
    for a in x.iter_mut().take(3) {
        *a = a.as_dtype(dtype.as_i32());
    }
    let got = run(c, &x);
    drain();
    assert_eq!(
        got.dtype(),
        dtype,
        "{}: the output keeps the dtype",
        describe(c)
    );
    assert_close(&describe(c), &vec_of(&got), &reference(c), rel);
}

const MASKS: [Mask; 4] = [Mask::None, Mask::Causal, Mask::Window(3), Mask::Custom];

/// Plain attention under every mask, with grouped heads and with a query
/// shorter than the keys (the continuation of a cached prompt).
#[test]
#[serial]
fn attention_is_the_definition() {
    for mask in MASKS {
        for (h, hk, lq, lk) in [(4, 4, 6, 6), (4, 2, 6, 6), (4, 1, 3, 9), (4, 2, 1, 7)] {
            let c = case(Spec::new(h, hk, lq, lk).mask(mask), 100 + lq as u64);
            check_values(&c, Dtype::Float32, 2e-5);
        }
    }
}

/// Gemma 2's attention logit softcapping: `cap · tanh(scores / cap)` before
/// the softmax, with grouped heads and every mask.
#[test]
#[serial]
fn softcapped_attention_is_the_definition() {
    for mask in MASKS {
        for (h, hk, lq, lk) in [(4, 4, 6, 6), (4, 2, 5, 9)] {
            let c = case(Spec::new(h, hk, lq, lk).mask(mask).softcap(5.0), 200);
            check_values(&c, Dtype::Float32, 2e-5);
        }
    }
}

/// Attention sinks (gpt-oss): a per-head logit joins each softmax as a key
/// that contributes no value.
#[test]
#[serial]
fn attention_with_sinks_is_the_definition() {
    for mask in MASKS {
        for (h, hk, lq, lk) in [(4, 4, 6, 6), (4, 2, 5, 9), (4, 2, 1, 7)] {
            let c = case(Spec::new(h, hk, lq, lk).mask(mask).sinks(), 300);
            check_values(&c, Dtype::Float32, 2e-5);
        }
    }
}

/// Half precision: bf16 inputs, an f32 reference built from the same
/// rounded values. The scores are formed in f32 inside the fused kernel; the
/// softcapped path has to do the same, since a bf16 score of 40 is only good
/// to 0.1, which the softmax turns into a 10% error in a weight.
#[test]
#[serial]
fn bf16_attention_keeps_f32_scores() {
    let bf16 = |c: &mut Case| {
        for x in [&mut c.q, &mut c.k, &mut c.v] {
            x.iter_mut()
                .for_each(|v| *v = half::bf16::from_f64(*v).to_f64());
        }
        if let Some(s) = &mut c.sinks {
            s.iter_mut()
                .for_each(|v| *v = half::bf16::from_f64(*v).to_f64());
        }
    };
    for spec in [
        Spec::new(4, 2, 12, 12),
        Spec::new(4, 2, 12, 12).softcap(5.0),
        Spec::new(4, 2, 12, 12).sinks(),
        Spec::new(4, 2, 4, 12).mask(Mask::Window(5)).softcap(30.0),
        Spec::new(4, 2, 6, 12).mask(Mask::Custom),
        Spec::new(4, 2, 4, 12).mask(Mask::Window(5)),
    ] {
        let mut c = case(spec, 400);
        bf16(&mut c);
        check_values(&c, Dtype::Bfloat16, 1.5e-2);
    }
}

/// Gemma 2's regime: attention logits that reach the tens (the reason for a
/// cap of 50). The scores are formed in f32: in bf16 a score of 40 is only
/// good to 0.1, which the softmax turns into a several-percent error in the
/// weight of the keys that matter.
#[test]
#[serial]
fn softcapped_scores_in_the_tens_are_formed_in_f32() {
    let mut c = case(Spec::new(4, 2, 12, 24).softcap(50.0), 410);
    for x in [&mut c.q, &mut c.k] {
        x.iter_mut()
            .for_each(|v| *v = half::bf16::from_f64(3.0 * *v).to_f64());
    }
    c.v.iter_mut()
        .for_each(|v| *v = half::bf16::from_f64(*v).to_f64());
    check_values(&c, Dtype::Bfloat16, 8e-3);
}

/// A boolean mask means "may attend" (true) as MLX defines it, on every
/// path; coerced to the queries' dtype it would be read as the additive
/// values 0 and 1 and attend to everything, silently.
#[test]
#[serial]
fn a_boolean_mask_means_attend() {
    for spec in [
        Spec::new(4, 2, 6, 6).mask(Mask::Custom),
        Spec::new(4, 2, 6, 6).mask(Mask::Custom).softcap(5.0),
        Spec::new(4, 2, 6, 6).mask(Mask::Custom).sinks(),
    ] {
        let c = case(spec, 600);
        let mut x: Vec<Array> = fields(&c)
            .iter()
            .zip(shapes(&c))
            .map(|(f, s)| arr(f, &s))
            .collect();
        let allowed: Vec<f32> = (0..c.lq * c.lk)
            .map(|n| f32::from(u8::from(custom_allowed(n / c.lk, n % c.lk))))
            .collect();
        let mask = Array::from_f32_slice(&allowed, &[1, 1, c.lq as i32, c.lk as i32])
            .as_dtype(Dtype::Bool.as_i32());
        let config = config(&c);
        let got = match &c.sinks {
            Some(_) => sink_sdpa(&x[0], &x[1], &x[2], &config, Some(&mask), &x[3]).unwrap(),
            None => fused_sdpa(&x[0], &x[1], &x[2], &config, Some(&mask)).unwrap(),
        };
        drain();
        x.clear();
        assert_close(&describe(&c), &vec_of(&got), &reference(&c), 2e-5);
    }
}

/// Gradients of the queries, keys, values and sinks against finite
/// differences of the float64 attention.
#[test]
#[serial]
fn attention_gradients_are_the_definitions() {
    let specs = [
        Spec::new(4, 2, 5, 5),
        Spec::new(4, 2, 5, 5).mask(Mask::Window(3)),
        Spec::new(4, 2, 3, 7).mask(Mask::None),
        Spec::new(4, 2, 4, 4).mask(Mask::Custom),
        Spec::new(4, 2, 5, 5).softcap(4.0),
        Spec::new(4, 2, 5, 5).sinks(),
        Spec::new(4, 1, 3, 6).mask(Mask::Window(2)).sinks(),
    ];
    for (n, spec) in specs.into_iter().enumerate() {
        let c = case(spec, 500 + n as u64);
        let mut rng = Rng(77);
        let out_len = c.b * c.h * c.lq * c.dv;
        let w: Vec<f64> = (0..out_len).map(|_| round32(rng.normal())).collect();
        let w_arr = arr(&w, &[c.b, c.h, c.lq, c.dv]);
        check_gradient(
            &describe(&c),
            &names(&c),
            &fields(&c),
            &shapes(&c),
            |x| run(&c, x).multiply(&w_arr).sum_all(),
            |x| {
                let mut p = c.clone();
                p.q = x[0].clone();
                p.k = x[1].clone();
                p.v = x[2].clone();
                p.sinks = x.get(3).cloned();
                dot(&reference(&p), &w)
            },
            2e-4,
        );
    }
}

/// Causal attention in f32 from primitives with known VJPs (a materialised
/// `[L, L]` score matrix, grouped keys and values expanded per query head),
/// with the sinks' extra softmax column when given.
fn ops_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    softcap: Option<f32>,
    sinks: Option<&Array>,
) -> Array {
    let (batch, heads, len) = (q.dim(0), q.dim(1), q.dim(2));
    let group = heads / k.dim(1);
    let expand = |x: &Array| {
        let (b, hk, l, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
        x.reshape(&[b, hk, 1, l, d])
            .broadcast_to(&[b, hk, group, l, d])
            .reshape(&[b, hk * group, l, d])
    };
    let (k, v) = (expand(k), expand(v));
    let mut scores = q
        .matmul(&k.transpose_axes(&[0, 1, 3, 2]))
        .multiply(&Array::from_f32(scale));
    if let Some(cap) = softcap {
        let cap = Array::from_f32(cap);
        scores = ops::tanh(&scores.divide(&cap)).multiply(&cap);
    }
    scores = scores.add(&create_causal_mask(len, len).unwrap());
    if let Some(sinks) = sinks {
        let column = sinks
            .reshape(&[1, heads, 1, 1])
            .broadcast_to(&[batch, heads, len, 1]);
        scores = ops::concatenate_axis(&[&scores, &column], -1);
    }
    let weights = scores.softmax(-1);
    let weights = if sinks.is_some() {
        ops::slice_axis(&weights, -1, 0, len)
    } else {
        weights
    };
    weights.matmul(&v)
}

/// At a length where MLX takes its fused forward and backward kernels (a
/// long causal prefill, no array mask), the gradients of the queries, keys,
/// values and sinks are those of plain attention built from primitives.
#[test]
#[serial]
fn long_sequence_gradients_match_plain_attention() {
    use crate::compat::nn::value_and_grad_explicit;
    use crate::compat::random;
    let (heads, kv_heads, head_dim, len) = (4, 2, 128, 1024);
    random::seed(3);
    let draw =
        |heads: i32| random::uniform_range(-1.0, 1.0, &[1, heads, len, head_dim], Dtype::Float32);
    let (q, k, v) = (draw(heads), draw(kv_heads), draw(kv_heads));
    let sinks = random::uniform_range(-1.0, 1.0, &[heads], Dtype::Float32);
    let config = FusedAttentionConfig::new(heads, kv_heads, head_dim);
    let scale = config.scale;

    // Only the last few query rows enter the loss, so it stays O(1).
    let rows = 4;
    let mut w = vec![0.0_f32; (heads * len * head_dim) as usize];
    for h in 0..heads {
        for r in (len - rows)..len {
            for c in 0..head_dim {
                let i = ((h * len + r) * head_dim + c) as usize;
                w[i] = ((i * 31 % 97) as f32 / 97.0) - 0.5;
            }
        }
    }
    let w = Array::from_f32_slice(&w, &[1, heads, len, head_dim]);

    for with_sinks in [false, true] {
        let inputs = [q.clone(), k.clone(), v.clone(), sinks.clone()];
        let used = if with_sinks { 4 } else { 3 };
        let grads = |f: &dyn Fn(&[Array]) -> Array| {
            let (_, g) = value_and_grad_explicit(
                |a: &[Array]| f(a).multiply(&w).sum_all(),
                &inputs[..used],
                &[],
            )
            .unwrap();
            g.iter().for_each(|x| {
                x.eval();
            });
            drain();
            g
        };
        let fused = grads(&|a| {
            if with_sinks {
                sink_sdpa(&a[0], &a[1], &a[2], &config, None, &a[3]).unwrap()
            } else {
                fused_sdpa(&a[0], &a[1], &a[2], &config, None).unwrap()
            }
        });
        let reference = grads(&|a| ops_attention(&a[0], &a[1], &a[2], scale, None, a.get(3)));
        for (name, (got, want)) in ["dq", "dk", "dv", "dsinks"]
            .iter()
            .zip(fused.iter().zip(&reference))
        {
            let (got, want) = (vec_of(got), vec_of(want));
            assert!(
                max_abs(&want) > 0.0,
                "{name}: the reference gradient is zero"
            );
            assert_close(&format!("{name} (sinks: {with_sinks})"), &got, &want, 1e-3);
        }
    }
}

/// A training step's worth of sink attention (forward and backward, at a
/// gpt-oss layer's shapes in bf16): the runtime's fused op against plain ops
/// that form the `[L, L]` scores. Time and peak memory; run with
/// `--ignored --nocapture`.
#[test]
#[serial]
#[ignore = "benchmark"]
fn sink_attention_step_against_plain_ops() {
    use crate::compat::nn::value_and_grad_explicit;
    use crate::compat::random;
    use crate::inline_array::{get_peak_memory, reset_peak_memory};
    use std::time::Instant;

    let (heads, kv_heads, head_dim) = (64, 8, 64);
    let bf16 = Dtype::Bfloat16.as_i32();
    for len in [1024, 4096] {
        let draw = |h: i32| random::normal(&[1, h, len, head_dim], Dtype::Float32).as_dtype(bf16);
        let (q, k, v) = (draw(heads), draw(kv_heads), draw(kv_heads));
        let sinks = random::normal(&[heads], Dtype::Float32).as_dtype(bf16);
        let config = FusedAttentionConfig::new(heads, kv_heads, head_dim);
        let scale = config.scale;
        for (name, fused) in [("plain ops", false), ("sdpa_with_sinks", true)] {
            let step = || {
                let (_, grads) = value_and_grad_explicit(
                    |a: &[Array]| {
                        let out = if fused {
                            sink_sdpa(&a[0], &a[1], &a[2], &config, None, &a[3]).unwrap()
                        } else {
                            ops_attention(&a[0], &a[1], &a[2], scale, None, Some(&a[3]))
                        };
                        out.as_dtype(Dtype::Float32.as_i32()).square().sum_all()
                    },
                    &[q.clone(), k.clone(), v.clone(), sinks.clone()],
                    &[],
                )
                .unwrap();
                grads.iter().for_each(|g| {
                    g.eval();
                });
                drain();
            };
            step();
            reset_peak_memory();
            let start = Instant::now();
            for _ in 0..3 {
                step();
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / 3.0;
            let peak = get_peak_memory() as f64 / (1 << 20) as f64;
            eprintln!("L={len:5} {name:16} {ms:9.1} ms/step   peak {peak:8.0} MiB");
        }
    }
}

/// A training step's worth of Gemma 2's softcapped attention (forward and
/// backward, bf16, 16 query heads over 8 key heads of width 256): scores in
/// f32 with a broadcast group axis against plain ops (scores in the queries'
/// dtype, keys and values expanded per query head, which is what the path
/// did). Time and peak memory; run with `--ignored --nocapture`.
#[test]
#[serial]
#[ignore = "benchmark"]
fn softcapped_attention_step_against_plain_ops() {
    use crate::compat::nn::value_and_grad_explicit;
    use crate::compat::random;
    use crate::inline_array::{get_peak_memory, reset_peak_memory};
    use std::time::Instant;

    let (heads, kv_heads, head_dim) = (16, 8, 256);
    let bf16 = Dtype::Bfloat16.as_i32();
    for len in [1024, 2048] {
        let draw = |h: i32| random::normal(&[1, h, len, head_dim], Dtype::Float32).as_dtype(bf16);
        let (q, k, v) = (draw(heads), draw(kv_heads), draw(kv_heads));
        let config =
            FusedAttentionConfig::new(heads, kv_heads, head_dim).with_logit_softcapping(50.0);
        let scale = config.scale;
        for (name, current) in [("plain ops", false), ("f32 scores", true)] {
            let step = || {
                let (_, grads) = value_and_grad_explicit(
                    |a: &[Array]| {
                        let out = if current {
                            fused_sdpa(&a[0], &a[1], &a[2], &config, None).unwrap()
                        } else {
                            ops_attention(&a[0], &a[1], &a[2], scale, Some(50.0), None)
                        };
                        out.as_dtype(Dtype::Float32.as_i32()).square().sum_all()
                    },
                    &[q.clone(), k.clone(), v.clone()],
                    &[],
                )
                .unwrap();
                grads.iter().for_each(|g| {
                    g.eval();
                });
                drain();
            };
            step();
            reset_peak_memory();
            let start = Instant::now();
            for _ in 0..3 {
                step();
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / 3.0;
            let peak = get_peak_memory() as f64 / (1 << 20) as f64;
            eprintln!("L={len:5} {name:12} {ms:9.1} ms/step   peak {peak:8.0} MiB");
        }
    }
}
