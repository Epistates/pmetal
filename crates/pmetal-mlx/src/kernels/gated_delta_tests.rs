//! The gated delta rule in float64, and the checks that hold the f32 paths to
//! it: values and gradients, sequential and chunked, over the shapes the
//! models use.
//!
//! The rule (Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta
//! Rule", ICLR 2025), per head with state `S ∈ ℝ^{Dv×Dk}`:
//!
//! ```text
//! S ← S · diag(g_t)                     (g_t scalar or per key dimension)
//! S ← S + β_t (v_t − S k_t) k_tᵀ
//! y_t = S q_t
//! ```
//!
//! Gradients are checked against central finite differences of the float64
//! reference at sampled coordinates: with `h = 1e-6` the difference is good
//! to about 1e-9, far inside the f32 paths' own error.

use super::*;
use crate::f64_reference::*;
use pmetal_runtime::compat::nn::value_and_grad_explicit;
use serial_test::serial;

// ---------------------------------------------------------------------------
// Deterministic data
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Case {
    b: usize,
    t: usize,
    hk: usize,
    dk: usize,
    hv: usize,
    dv: usize,
    /// `true` for a gate per key dimension, `[B, T, Hv, Dk]`.
    vector_gate: bool,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    state: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Spec {
    b: usize,
    t: usize,
    hk: usize,
    dk: usize,
    hv: usize,
    dv: usize,
    vector_gate: bool,
    /// Normalise `q` and `k` per head, as the model layers do (`l2norm`),
    /// which keeps the WY system well conditioned.
    normalize: bool,
}

impl Spec {
    fn new(b: usize, t: usize, hk: usize, dk: usize, hv: usize, dv: usize) -> Self {
        Self {
            b,
            t,
            hk,
            dk,
            hv,
            dv,
            vector_gate: false,
            normalize: true,
        }
    }
}

fn case(spec: Spec, seed: u64) -> Case {
    let mut rng = Rng(seed);
    let Spec {
        b,
        t,
        hk,
        dk,
        hv,
        dv,
        vector_gate,
        normalize,
    } = spec;
    let heads = |rng: &mut Rng, n_heads: usize| -> Vec<f64> {
        let mut x: Vec<f64> = (0..b * t * n_heads * dk).map(|_| rng.normal()).collect();
        if normalize {
            for row in x.chunks_mut(dk) {
                let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                row.iter_mut().for_each(|v| *v /= norm);
            }
        }
        x.into_iter().map(round32).collect()
    };
    let q = heads(&mut rng, hk);
    let k = heads(&mut rng, hk);
    let v = (0..b * t * hv * dv)
        .map(|_| round32(rng.normal()))
        .collect();
    let gate_len = b * t * hv * if vector_gate { dk } else { 1 };
    // Decays across the range a trained model has (0.3 .. 1), sparing a
    // fully-forgetting gate that the log-space chunk path clamps.
    let g = (0..gate_len)
        .map(|_| round32(0.3 + 0.7 * rng.uniform()))
        .collect();
    let beta = (0..b * t * hv)
        .map(|_| round32(0.05 + 0.9 * rng.uniform()))
        .collect();
    let state = (0..b * hv * dv * dk)
        .map(|_| round32(0.5 * rng.normal()))
        .collect();
    Case {
        b,
        t,
        hk,
        dk,
        hv,
        dv,
        vector_gate,
        q,
        k,
        v,
        g,
        beta,
        state,
    }
}

// ---------------------------------------------------------------------------
// The reference
// ---------------------------------------------------------------------------

/// `(y [B,T,Hv,Dv], final state [B,Hv,Dv,Dk])` of the rule, sequentially, in
/// f64. `mask` (`[B, T]`, true = a real token) leaves the state alone at the
/// masked steps; the outputs there are not defined and are not compared.
fn reference(c: &Case, mask: Option<&[bool]>) -> (Vec<f64>, Vec<f64>) {
    let (b, t, hk, dk, hv, dv) = (c.b, c.t, c.hk, c.dk, c.hv, c.dv);
    let rep = hv / hk;
    let mut y = vec![0.0; b * t * hv * dv];
    let mut state = c.state.clone();
    for bi in 0..b {
        for h in 0..hv {
            let kh = h / rep;
            let s0 = (bi * hv + h) * dv * dk;
            for ti in 0..t {
                if mask.is_some_and(|m| !m[bi * t + ti]) {
                    continue;
                }
                let qo = ((bi * t + ti) * hk + kh) * dk;
                let vo = ((bi * t + ti) * hv + h) * dv;
                let bo = (bi * t + ti) * hv + h;
                let s = &mut state[s0..s0 + dv * dk];
                // decay
                for r in 0..dv {
                    for col in 0..dk {
                        let g = if c.vector_gate {
                            c.g[bo * dk + col]
                        } else {
                            c.g[bo]
                        };
                        s[r * dk + col] *= g;
                    }
                }
                // delta
                let mut delta = vec![0.0; dv];
                for r in 0..dv {
                    let kv: f64 = (0..dk).map(|col| s[r * dk + col] * c.k[qo + col]).sum();
                    delta[r] = c.beta[bo] * (c.v[vo + r] - kv);
                }
                for r in 0..dv {
                    for col in 0..dk {
                        s[r * dk + col] += delta[r] * c.k[qo + col];
                    }
                }
                for r in 0..dv {
                    y[vo + r] = (0..dk).map(|col| s[r * dk + col] * c.q[qo + col]).sum();
                }
            }
        }
    }
    (y, state)
}

// ---------------------------------------------------------------------------
// f32 side
// ---------------------------------------------------------------------------

struct Inputs {
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
}

impl Inputs {
    fn of(c: &Case) -> Self {
        let gate_shape: Vec<usize> = if c.vector_gate {
            vec![c.b, c.t, c.hv, c.dk]
        } else {
            vec![c.b, c.t, c.hv]
        };
        Self {
            q: arr(&c.q, &[c.b, c.t, c.hk, c.dk]),
            k: arr(&c.k, &[c.b, c.t, c.hk, c.dk]),
            v: arr(&c.v, &[c.b, c.t, c.hv, c.dv]),
            g: arr(&c.g, &gate_shape),
            beta: arr(&c.beta, &[c.b, c.t, c.hv]),
            state: arr(&c.state, &[c.b, c.hv, c.dv, c.dk]),
        }
    }

    fn as_vec(&self) -> Vec<Array> {
        vec![
            self.q.clone(),
            self.k.clone(),
            self.v.clone(),
            self.g.clone(),
            self.beta.clone(),
            self.state.clone(),
        ]
    }
}

type Impl = dyn Fn(&[Array], Option<&Array>) -> (Array, Array);

/// The sequential ops path.
fn ops_impl(x: &[Array], mask: Option<&Array>) -> (Array, Array) {
    gated_delta_ops(&x[0], &x[1], &x[2], &x[3], &x[4], Some(&x[5]), mask).unwrap()
}

/// The chunked path at `chunk` positions per chunk.
fn chunk_impl(chunk: i32) -> impl Fn(&[Array], Option<&Array>) -> (Array, Array) {
    move |x, mask| {
        gated_delta_chunk_ops_impl(&x[0], &x[1], &x[2], &x[3], &x[4], &x[5], mask, chunk).unwrap()
    }
}

/// Values of an implementation against the f64 reference.
fn check_values(what: &str, c: &Case, f: &Impl, mask: Option<&[bool]>, rel: f64) {
    let inputs = Inputs::of(c);
    let mask_arr = mask.map(|m| {
        let m32: Vec<f32> = m.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        Array::from_f32_slice(&m32, &[c.b as i32, c.t as i32]).as_dtype(Dtype::Bool.as_i32())
    });
    let (y, s) = f(&inputs.as_vec(), mask_arr.as_ref());
    let (ry, rs) = reference(c, mask);
    drain();
    let mut y = vec_of(&y);
    let mut ry = ry;
    if let Some(m) = mask {
        // Outputs at masked positions are not defined.
        let per_pos = c.hv * c.dv;
        for (pos, &valid) in m.iter().enumerate() {
            if !valid {
                y[pos * per_pos..(pos + 1) * per_pos].fill(0.0);
                ry[pos * per_pos..(pos + 1) * per_pos].fill(0.0);
            }
        }
    }
    assert_close(&format!("{what}: output"), &y, &ry, rel);
    assert_close(&format!("{what}: final state"), &vec_of(&s), &rs, rel);
}

const NAMES: [&str; 6] = ["q", "k", "v", "g", "beta", "state"];

/// The case's inputs, in [`NAMES`] order.
fn fields(c: &Case) -> Vec<Vec<f64>> {
    vec![
        c.q.clone(),
        c.k.clone(),
        c.v.clone(),
        c.g.clone(),
        c.beta.clone(),
        c.state.clone(),
    ]
}

/// The shapes of [`fields`].
fn field_shapes(c: &Case) -> Vec<Vec<usize>> {
    let gate = if c.vector_gate {
        vec![c.b, c.t, c.hv, c.dk]
    } else {
        vec![c.b, c.t, c.hv]
    };
    vec![
        vec![c.b, c.t, c.hk, c.dk],
        vec![c.b, c.t, c.hk, c.dk],
        vec![c.b, c.t, c.hv, c.dv],
        gate,
        vec![c.b, c.t, c.hv],
        vec![c.b, c.hv, c.dv, c.dk],
    ]
}

/// `c` with its inputs replaced, in [`NAMES`] order.
fn with_fields(c: &Case, mut x: Vec<Vec<f64>>) -> Case {
    let mut c = c.clone();
    c.state = x.pop().unwrap();
    c.beta = x.pop().unwrap();
    c.g = x.pop().unwrap();
    c.v = x.pop().unwrap();
    c.k = x.pop().unwrap();
    c.q = x.pop().unwrap();
    c
}

/// Random weights for the outputs and the final state, to contract them to a
/// scalar loss: `Σ wy·y + Σ ws·S`.
fn loss_weights(c: &Case) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng(0xFEED);
    let wy = (0..c.b * c.t * c.hv * c.dv)
        .map(|_| round32(rng.normal()))
        .collect();
    let ws = (0..c.b * c.hv * c.dv * c.dk)
        .map(|_| round32(rng.normal()))
        .collect();
    (wy, ws)
}

/// Gradients through `f`, for every input of the rule.
fn check_gradients(what: &str, c: &Case, f: &Impl, rel: f64) {
    let (wy, ws) = loss_weights(c);
    let wy_arr = arr(&wy, &[c.b, c.t, c.hv, c.dv]);
    let ws_arr = arr(&ws, &[c.b, c.hv, c.dv, c.dk]);
    check_gradient(
        what,
        &NAMES,
        &fields(c),
        &field_shapes(c),
        |x| {
            let (y, s) = f(x, None);
            y.multiply(&wy_arr)
                .sum_all()
                .add(&s.multiply(&ws_arr).sum_all())
        },
        |x| {
            let (y, s) = reference(&with_fields(c, x.to_vec()), None);
            dot(&y, &wy) + dot(&s, &ws)
        },
        rel,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The sequential path is the rule: scalar and per-key-dimension gates,
/// grouped heads, a carried state.
#[test]
#[serial]
fn the_sequential_path_is_the_rule() {
    for (name, spec) in [
        ("scalar gate", Spec::new(2, 24, 2, 16, 4, 8)),
        (
            "vector gate",
            Spec {
                vector_gate: true,
                ..Spec::new(1, 17, 1, 8, 2, 8)
            },
        ),
    ] {
        check_values(name, &case(spec, 1), &ops_impl, None, 2e-5);
    }
}

/// A one-token call takes the decode path.
#[test]
#[serial]
fn the_decode_path_is_the_rule() {
    let c = case(Spec::new(2, 1, 2, 64, 4, 32), 2);
    check_values("decode", &c, &ops_impl, None, 2e-5);
}

/// The chunked path is the rule too, at several chunk sizes, with a length
/// that does not divide the chunk, a carried state and grouped heads.
#[test]
#[serial]
fn the_chunked_path_is_the_rule() {
    for (chunk, t) in [
        (16, 40),
        (16, 64),
        (32, 100),
        (64, 130),
        (128, 300),
        (256, 600),
    ] {
        let c = case(Spec::new(2, t, 2, 16, 4, 8), 3 + t as u64);
        check_values(
            &format!("chunk {chunk}, T {t}"),
            &c,
            &chunk_impl(chunk),
            None,
            5e-4,
        );
    }
}

/// Unnormalised keys, as the older tests used: the WY system is much worse
/// conditioned and the chunk path still has to track the rule.
#[test]
#[serial]
fn the_chunked_path_survives_unnormalised_keys() {
    let spec = Spec {
        normalize: false,
        ..Spec::new(1, 96, 1, 8, 1, 8)
    };
    let c = case(spec, 9);
    check_values("unnormalised", &c, &chunk_impl(32), None, 5e-3);
}

/// Chunked and sequential agree on the state a split sequence ends in: the
/// state carried across a chunk boundary is the sequential one.
#[test]
#[serial]
fn the_chunked_state_carries_across_calls() {
    let full = case(Spec::new(1, 96, 2, 16, 2, 8), 5);
    let (_, want) = reference(&full, None);

    let half = 48;
    let slice = |c: &Case, from: usize, to: usize| -> Case {
        Case {
            t: to - from,
            q: c.q[from * c.hk * c.dk..to * c.hk * c.dk].to_vec(),
            k: c.k[from * c.hk * c.dk..to * c.hk * c.dk].to_vec(),
            v: c.v[from * c.hv * c.dv..to * c.hv * c.dv].to_vec(),
            g: c.g[from * c.hv..to * c.hv].to_vec(),
            beta: c.beta[from * c.hv..to * c.hv].to_vec(),
            ..c.clone()
        }
    };
    let first = slice(&full, 0, half);
    let mut second = slice(&full, half, 96);
    let f = chunk_impl(16);
    let (_, mid) = f(&Inputs::of(&first).as_vec(), None);
    second.state = vec_of(&mid);
    let (_, end) = f(&Inputs::of(&second).as_vec(), None);
    drain();
    assert_close("carried state", &vec_of(&end), &want, 5e-4);
}

/// Masked positions leave the state alone on the chunked path.
#[test]
#[serial]
fn the_chunked_path_skips_masked_positions() {
    let c = case(Spec::new(2, 70, 2, 16, 2, 8), 7);
    let mut mask = vec![true; 2 * 70];
    for (i, m) in mask.iter_mut().enumerate() {
        // Ragged tails: the second row loses its last 30 tokens.
        *m = !(i >= 70 + 40);
    }
    check_values("masked", &c, &chunk_impl(32), Some(&mask), 5e-4);
}

/// Gradients of the sequential path against the f64 reference.
#[test]
#[serial]
fn the_sequential_gradients_are_the_rules() {
    let c = case(Spec::new(1, 12, 2, 8, 4, 8), 11);
    check_gradients("sequential", &c, &ops_impl, 2e-4);
}

/// Gradients of the chunked path against the f64 reference: every input.
#[test]
#[serial]
fn the_chunked_gradients_are_the_rules() {
    let c = case(Spec::new(1, 40, 2, 8, 4, 8), 13);
    check_gradients("chunked", &c, &chunk_impl(16), 2e-4);
    let long = case(Spec::new(1, 150, 1, 8, 2, 8), 14);
    check_gradients("chunked, 128", &long, &chunk_impl(128), 2e-4);
}

/// The inverse of `I + A` for strictly lower `A` by repeated squaring is the
/// inverse, for chunk sizes that are and are not powers of two, with the
/// entries of a WY system (`beta * decay * k . k`, normalised keys) and with
/// larger ones.
#[test]
#[serial]
fn the_unit_lower_inverse_is_the_inverse() {
    for (chunk, amplitude) in [(16usize, 0.4), (50, 0.2), (64, 0.3), (64, 1.0)] {
        let mut rng = Rng(chunk as u64);
        let mut m = vec![0.0; chunk * chunk];
        for i in 0..chunk {
            m[i * chunk + i] = 1.0;
            for j in 0..i {
                m[i * chunk + j] =
                    round32(amplitude * (2.0 * rng.uniform() - 1.0) / (i as f64).sqrt().max(1.0));
            }
        }
        // Forward substitution in f64.
        let mut want = vec![0.0; chunk * chunk];
        for col in 0..chunk {
            for i in col..chunk {
                let rhs = if i == col { 1.0 } else { 0.0 };
                let s: f64 = (col..i)
                    .map(|j| m[i * chunk + j] * want[j * chunk + col])
                    .sum();
                want[i * chunk + col] = rhs - s;
            }
        }
        let got = vec_of(&unit_lower_inverse(&arr(
            &m,
            &[2, chunk, chunk].map(|d| d)[1..],
        )));
        drain();
        assert_close(
            &format!("C {chunk}, amplitude {amplitude}"),
            &got,
            &want,
            1e-4,
        );
    }
}

/// The default chunk is 64 tokens, and 128 from 512 tokens; an override wins
/// and is kept in 16..=512, and zero or less asks for the sequential path.
#[test]
fn the_default_chunk_grows_with_the_sequence() {
    assert!(
        std::env::var(GDN_CHUNK_SIZE_ENV_VAR).is_err(),
        "unset {GDN_CHUNK_SIZE_ENV_VAR} to run this test"
    );
    assert_eq!(resolve_chunk_size_override(None, 100), Some(64));
    assert_eq!(resolve_chunk_size_override(None, 511), Some(64));
    assert_eq!(resolve_chunk_size_override(None, 512), Some(128));
    assert_eq!(resolve_chunk_size_override(None, 4096), Some(128));
    assert_eq!(resolve_chunk_size_override(Some(32), 4096), Some(32));
    assert_eq!(resolve_chunk_size_override(Some(7), 100), Some(16));
    assert_eq!(resolve_chunk_size_override(Some(9000), 100), Some(512));
    assert_eq!(resolve_chunk_size_override(Some(0), 4096), None);
    assert_eq!(resolve_chunk_size_override(Some(-3), 4096), None);
}

/// A head that nearly forgets at every step (a decay of 0.005 to 0.05): over
/// a chunk the log-decay sums past what `exp` holds in f32 (`ln(f32::MAX)` is
/// 88), so the decay matrix's upper triangle, which the mask zeroes, is `inf`
/// before it is masked. Forward values and every gradient stay finite and the
/// rule's.
#[test]
#[serial]
fn a_forgetful_head_keeps_the_chunked_path_finite() {
    let mut c = case(Spec::new(1, 80, 2, 8, 2, 8), 17);
    let mut rng = Rng(5);
    c.g.iter_mut()
        .for_each(|g| *g = round32(0.005 + 0.045 * rng.uniform()));
    for chunk in [16, 64] {
        check_values(
            &format!("chunk {chunk}"),
            &c,
            &chunk_impl(chunk),
            None,
            5e-4,
        );
        check_gradients(&format!("chunk {chunk}"), &c, &chunk_impl(chunk), 5e-4);
    }
}

/// Round to bf16 and back.
fn round_bf16(x: f64) -> f64 {
    half::bf16::from_f64(x).to_f64()
}

/// A case whose every input is exactly representable in bf16 (the state in
/// bf16 as well, as a cache holds it), so the reference sees what the
/// implementation is given.
fn bf16_case(spec: Spec, seed: u64) -> Case {
    let mut c = case(spec, seed);
    for x in [
        &mut c.q,
        &mut c.k,
        &mut c.v,
        &mut c.g,
        &mut c.beta,
        &mut c.state,
    ] {
        x.iter_mut().for_each(|v| *v = round_bf16(*v));
    }
    c
}

/// `f` run on bf16 copies of its inputs, checking that the output and the
/// state keep their dtype.
fn in_bf16(
    f: impl Fn(&[Array], Option<&Array>) -> (Array, Array),
) -> impl Fn(&[Array], Option<&Array>) -> (Array, Array) {
    move |x, mask| {
        let bf16 = Dtype::Bfloat16.as_i32();
        let x: Vec<Array> = x.iter().map(|a| a.as_dtype(bf16)).collect();
        let (y, s) = f(&x, mask);
        assert_eq!(
            y.dtype(),
            Dtype::Bfloat16,
            "the output keeps the input dtype"
        );
        assert_eq!(
            s.dtype(),
            Dtype::Bfloat16,
            "the state keeps the state dtype"
        );
        (y, s)
    }
}

/// Half-precision inputs give the recurrence's answer, not a half-precision
/// recurrence's: the state is accumulated in f32 inside a call and only the
/// result is rounded, as the fused kernel does. Over a few hundred steps a
/// recurrence carried in bf16 drifts far outside bf16's own rounding.
#[test]
#[serial]
fn bf16_inputs_run_an_f32_recurrence() {
    let c = bf16_case(Spec::new(1, 200, 2, 16, 4, 8), 21);
    // bf16 carries about three significant digits: the result is held to
    // that, not to f32's.
    check_values("ops, bf16", &c, &in_bf16(ops_impl), None, 1.5e-2);
    check_values("chunked, bf16", &c, &in_bf16(chunk_impl(64)), None, 1.5e-2);
}

// ---------------------------------------------------------------------------
// The public entry points: raw gate inputs, dispatch, rollback
// ---------------------------------------------------------------------------

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}

/// The decay `exp(−exp(A_log) · softplus(a + dt_bias))`.
fn decay(a_log: f64, a: f64, dt_bias: f64) -> f64 {
    (-a_log.exp() * softplus(a + dt_bias)).exp()
}

/// The raw inputs `gated_delta_update` turns into `g` and `β`.
#[derive(Clone)]
struct Raw {
    /// `[B, T, Hv]`
    a: Vec<f64>,
    /// `[B, T, Hv]`
    b: Vec<f64>,
    /// `[Hv]`
    a_log: Vec<f64>,
    /// `[Hv]`
    dt_bias: Vec<f64>,
}

fn raw_inputs(c: &Case, seed: u64) -> Raw {
    let mut rng = Rng(seed);
    let n = c.b * c.t * c.hv;
    Raw {
        a: (0..n).map(|_| round32(rng.normal())).collect(),
        b: (0..n).map(|_| round32(rng.normal())).collect(),
        // `A_log` is the log of a decay rate between 1 and 16 in trained
        // checkpoints; `dt_bias` is of order one.
        a_log: (0..c.hv)
            .map(|_| round32((1.0 + 15.0 * rng.uniform()).ln()))
            .collect(),
        dt_bias: (0..c.hv).map(|_| round32(rng.normal())).collect(),
    }
}

/// `base` with the gates the raw inputs give.
fn gated(base: &Case, raw: &Raw) -> Case {
    let mut c = base.clone();
    let hv = c.hv;
    c.g = raw
        .a
        .iter()
        .enumerate()
        .map(|(i, &a)| decay(raw.a_log[i % hv], a, raw.dt_bias[i % hv]))
        .collect();
    c.beta = raw.b.iter().map(|&b| 1.0 / (1.0 + (-b).exp())).collect();
    c
}

const UPDATE_NAMES: [&str; 8] = ["q", "k", "v", "a", "b", "a_log", "dt_bias", "state"];

/// `gated_delta_update_with_chunk_size_override` on `base`'s q, k, v and state
/// and `raw`'s gates.
fn run_update(x: &[Array], chunk_override: Option<i32>, mask: Option<&Array>) -> (Array, Array) {
    gated_delta_update_with_chunk_size_override(
        &x[0],
        &x[1],
        &x[2],
        &x[3],
        &x[4],
        &x[5],
        &x[6],
        Some(&x[7]),
        mask,
        chunk_override,
    )
    .unwrap()
}

fn update_arrays(base: &Case, raw: &Raw) -> Vec<Array> {
    vec![
        arr(&base.q, &[base.b, base.t, base.hk, base.dk]),
        arr(&base.k, &[base.b, base.t, base.hk, base.dk]),
        arr(&base.v, &[base.b, base.t, base.hv, base.dv]),
        arr(&raw.a, &[base.b, base.t, base.hv]),
        arr(&raw.b, &[base.b, base.t, base.hv]),
        arr(&raw.a_log, &[base.hv]),
        arr(&raw.dt_bias, &[base.hv]),
        arr(&base.state, &[base.b, base.hv, base.dv, base.dk]),
    ]
}

/// The values `gated_delta_update` returns, by every route the dispatcher can
/// take, against the rule: the fused kernel (a key width of 32, no mask, no
/// gradient), the chunked path (forced, a mask, or a prompt past a chunk) and
/// the sequential one.
#[test]
#[serial]
fn the_update_is_the_rule_by_every_route() {
    let kernel_shape = Spec::new(2, 40, 2, 32, 4, 16);
    let ops_shape = Spec::new(2, 40, 2, 16, 4, 16);
    let routes: [(&str, Spec, Option<i32>, bool); 7] = [
        ("kernel", kernel_shape, None, false),
        (
            "kernel, one token",
            Spec::new(2, 1, 2, 32, 4, 16),
            None,
            false,
        ),
        ("forced chunks", ops_shape, Some(16), false),
        ("forced sequential", ops_shape, Some(0), false),
        ("past a chunk", Spec::new(1, 150, 2, 16, 4, 8), None, false),
        (
            "a long sequence (wide chunks)",
            Spec::new(1, 600, 1, 16, 2, 8),
            None,
            false,
        ),
        ("masked", kernel_shape, None, true),
    ];
    for (what, spec, chunk_override, masked) in routes {
        let base = case(spec, 31);
        let raw = raw_inputs(&base, 32);
        let want_case = gated(&base, &raw);
        let mask: Option<Vec<bool>> =
            masked.then(|| (0..base.b * base.t).map(|i| i % base.t < 30).collect());
        let mask_arr = mask.as_ref().map(|m| {
            let m32: Vec<f32> = m.iter().map(|&v| f32::from(u8::from(v))).collect();
            Array::from_f32_slice(&m32, &[base.b as i32, base.t as i32])
                .as_dtype(Dtype::Bool.as_i32())
        });
        let (y, s) = run_update(
            &update_arrays(&base, &raw),
            chunk_override,
            mask_arr.as_ref(),
        );
        drain();
        let (mut ry, rs) = reference(&want_case, mask.as_deref());
        let mut y = vec_of(&y);
        if let Some(m) = &mask {
            let per_pos = base.hv * base.dv;
            for (pos, &valid) in m.iter().enumerate() {
                if !valid {
                    y[pos * per_pos..(pos + 1) * per_pos].fill(0.0);
                    ry[pos * per_pos..(pos + 1) * per_pos].fill(0.0);
                }
            }
        }
        assert_close(&format!("{what}: output"), &y, &ry, 5e-4);
        assert_close(&format!("{what}: state"), &vec_of(&s), &rs, 5e-4);
    }
}

/// The fused kernel with half-precision inputs: bf16 projections and a bf16
/// state, f32 `A_log`, as a checkpoint holds them.
#[test]
#[serial]
fn the_kernel_route_takes_bf16() {
    let base = bf16_case(Spec::new(1, 33, 2, 32, 4, 16), 41);
    let raw = raw_inputs(&base, 42);
    let raw = Raw {
        a: raw.a.iter().copied().map(round_bf16).collect(),
        b: raw.b.iter().copied().map(round_bf16).collect(),
        dt_bias: raw.dt_bias.iter().copied().map(round_bf16).collect(),
        ..raw
    };
    let bf16 = Dtype::Bfloat16.as_i32();
    let mut x = update_arrays(&base, &raw);
    for i in [0, 1, 2, 3, 4, 6, 7] {
        x[i] = x[i].as_dtype(bf16);
    }
    let (y, s) = run_update(&x, None, None);
    drain();
    assert_eq!(y.dtype(), Dtype::Bfloat16);
    assert_eq!(s.dtype(), Dtype::Bfloat16);
    let (ry, rs) = reference(&gated(&base, &raw), None);
    assert_close("kernel, bf16: output", &vec_of(&y), &ry, 2e-2);
    assert_close("kernel, bf16: state", &vec_of(&s), &rs, 2e-2);
}

/// Gradients of the update with respect to everything it takes, through the
/// routes autograd can follow. A key width of 32 with no mask is what the
/// fused kernel takes outside a gradient trace; under one the dispatcher has
/// to leave it, and the gradient has to be the rule's.
#[test]
#[serial]
fn the_update_gradients_are_the_rules() {
    for (what, spec, chunk_override) in [
        ("short, sequential", Spec::new(1, 6, 2, 32, 4, 8), None),
        ("forced chunks", Spec::new(1, 48, 2, 32, 4, 8), Some(16)),
        ("forced sequential", Spec::new(1, 20, 2, 32, 4, 8), Some(0)),
        ("past a chunk", Spec::new(1, 80, 1, 16, 2, 8), None),
    ] {
        let base = case(spec, 51);
        let raw = raw_inputs(&base, 52);
        let (wy, ws) = loss_weights(&base);
        let wy_arr = arr(&wy, &[base.b, base.t, base.hv, base.dv]);
        let ws_arr = arr(&ws, &[base.b, base.hv, base.dv, base.dk]);

        let arrays = update_arrays(&base, &raw);
        let bases: Vec<Vec<f64>> = vec![
            base.q.clone(),
            base.k.clone(),
            base.v.clone(),
            raw.a.clone(),
            raw.b.clone(),
            raw.a_log.clone(),
            raw.dt_bias.clone(),
            base.state.clone(),
        ];
        let shapes: Vec<Vec<usize>> = arrays
            .iter()
            .map(|a| a.shape().iter().map(|&d| d as usize).collect())
            .collect();
        check_gradient(
            what,
            &UPDATE_NAMES,
            &bases,
            &shapes,
            |x| {
                let (y, s) = run_update(x, chunk_override, None);
                y.multiply(&wy_arr)
                    .sum_all()
                    .add(&s.multiply(&ws_arr).sum_all())
            },
            |x| {
                let mut c = base.clone();
                c.q = x[0].clone();
                c.k = x[1].clone();
                c.v = x[2].clone();
                c.state = x[7].clone();
                let raw = Raw {
                    a: x[3].clone(),
                    b: x[4].clone(),
                    a_log: x[5].clone(),
                    dt_bias: x[6].clone(),
                };
                let (y, s) = reference(&gated(&c, &raw), None);
                dot(&y, &wy) + dot(&s, &ws)
            },
            5e-4,
        );
    }
}

/// `compute_g` is `exp(−exp(A_log) · softplus(a + dt_bias))`, including where
/// the softplus is large (the decay underflows to zero, not NaN).
#[test]
#[serial]
fn the_decay_is_the_formula() {
    let a_log = [0.0_f64, 1.0, 2.5];
    let dt_bias = [0.1_f64, -0.4, 0.9];
    let a: Vec<f64> = [-30.0, -3.0, -0.5, 0.0, 0.7, 4.0, 25.0, 60.0]
        .iter()
        .flat_map(|&v| [v, v, v])
        .collect();
    let want: Vec<f64> = a
        .iter()
        .enumerate()
        .map(|(i, &x)| decay(a_log[i % 3], x, dt_bias[i % 3]))
        .collect();
    let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let g = compute_g(
        &Array::from_f32_slice(&f(&a_log), &[3]),
        &Array::from_f32_slice(&f(&a), &[1, 8, 3]),
        &Array::from_f32_slice(&f(&dt_bias), &[3]),
    )
    .unwrap();
    let raw_g = compute_g_impl(
        &Array::from_f32_slice(&f(&a_log), &[3]),
        &Array::from_f32_slice(&f(&a), &[1, 8, 3]),
        &Array::from_f32_slice(&f(&dt_bias), &[3]),
    )
    .unwrap();
    drain();
    for (what, got) in [("compute_g", g), ("compute_g_impl", raw_g)] {
        let got = vec_of(&got);
        assert!(got.iter().all(|v| v.is_finite()), "{what}: {got:?}");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst <= 1e-6, "{what}: off by {worst:e}");
    }
}

/// The rollback primitive replays a prefix: the state after `n` of a verified
/// block's tokens is what feeding those `n` tokens through the rule gives,
/// by the kernel (a key width of 32) and by the ops (any other).
#[test]
#[serial]
fn the_state_advance_replays_a_prefix() {
    for (what, spec) in [
        ("kernel", Spec::new(2, 12, 2, 32, 4, 16)),
        ("ops", Spec::new(2, 12, 2, 16, 4, 8)),
    ] {
        let c = case(spec, 61);
        let accepted = 7;
        // The first `accepted` tokens of every row.
        let prefix = |x: &[f64], per_token: usize| -> Vec<f64> {
            (0..c.b)
                .flat_map(|b| x[b * c.t * per_token..(b * c.t + accepted) * per_token].to_vec())
                .collect()
        };
        let head = Case {
            t: accepted,
            q: prefix(&c.q, c.hk * c.dk),
            k: prefix(&c.k, c.hk * c.dk),
            v: prefix(&c.v, c.hv * c.dv),
            g: prefix(&c.g, c.hv),
            beta: prefix(&c.beta, c.hv),
            ..c.clone()
        };
        let (_, want) = reference(&head, None);
        let n = Inputs::of(&head);
        let got = gated_delta_state_advance(&n.state, &n.k, &n.v, &n.g, &n.beta).unwrap();
        drain();
        assert_close(
            &format!("{what}: advanced state"),
            &vec_of(&got),
            &want,
            5e-4,
        );
    }
}

/// A training step's worth of the recurrence (forward and backward, at a
/// layer's shapes) by the chunked and the sequential path: time and peak
/// memory. Run with `--ignored --nocapture`; it measures, it asserts only
/// that the two gradients agree.
#[test]
#[serial]
#[ignore = "benchmark"]
fn chunked_training_step_against_sequential() {
    use crate::memory::{get_peak_memory, reset_peak_memory};
    use std::time::Instant;

    let (hk, hv, dk, dv) = (16, 32, 128, 128);
    for t in [256_i32, 1024, 4096] {
        let rng_inputs = |shape: &[i32]| {
            pmetal_runtime::compat::random::normal(shape, Dtype::Float32)
                .multiply(&Array::from_f32(0.5))
                .as_dtype(Dtype::Bfloat16.as_i32())
        };
        let l2 = |x: Array| {
            let norm = x.square().sum_axis(-1, true).sqrt();
            x.divide(&norm)
        };
        let q = l2(rng_inputs(&[1, t, hk, dk]));
        let k = l2(rng_inputs(&[1, t, hk, dk]));
        let v = rng_inputs(&[1, t, hv, dv]);
        let a = rng_inputs(&[1, t, hv]);
        let b = rng_inputs(&[1, t, hv]);
        let a_log = Array::from_f32_slice(&vec![1.0_f32; hv as usize], &[hv]);
        let dt_bias = Array::from_f32_slice(&vec![0.0_f32; hv as usize], &[hv])
            .as_dtype(Dtype::Bfloat16.as_i32());

        let mut results = Vec::new();
        for (name, chunk) in [
            ("sequential", Some(0)),
            ("chunk 64", Some(64)),
            ("chunk 128", Some(128)),
            ("chunk 256", Some(256)),
        ] {
            let step = || {
                let (_, grads) = value_and_grad_explicit(
                    |x: &[Array]| {
                        let (y, _) = gated_delta_update_with_chunk_size_override(
                            &x[0], &x[1], &x[2], &a, &b, &a_log, &dt_bias, None, None, chunk,
                        )
                        .unwrap();
                        y.as_dtype(Dtype::Float32.as_i32()).square().sum_all()
                    },
                    &[q.clone(), k.clone(), v.clone()],
                    &[],
                )
                .unwrap();
                grads.iter().for_each(|g| {
                    g.eval();
                });
                drain();
                grads
            };
            step();
            reset_peak_memory();
            let start = Instant::now();
            let runs = 3;
            let mut grads = Vec::new();
            for _ in 0..runs {
                grads = step();
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / runs as f64;
            let peak = get_peak_memory() as f64 / (1 << 20) as f64;
            eprintln!("T={t:5} {name:10} {ms:9.1} ms/step   peak {peak:8.0} MiB");
            results.push(grads);
        }
        for (i, name) in ["dq", "dk", "dv"].iter().enumerate() {
            let (s, c) = (vec_of(&results[0][i]), vec_of(&results[1][i]));
            assert_close(&format!("T={t} {name}"), &c, &s, 5e-2);
        }
    }
}
