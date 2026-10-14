//! Gradient and forward checks against an independent float64 transcription of
//! the paper's equations (arXiv:2512.24880, eqs. 3, 7, 8, 9), with the
//! Sinkhorn recipe of the released DeepSeek-V4 kernel.
//!
//! The reference below shares no code with the crate. Gradients are central
//! finite differences of the reference's scalar loss, taken in `f64` with a
//! step of 1e-6, so they are accurate to ~1e-9 and the only error left in the
//! comparison is the `f32` rounding of the crate under test.

use ndarray::{Array1, Array2, Array3};
use pmetal_mhc::{
    MhcConfig, MhcGradients, MhcLayer, MhcParams, SinkhornConfig, compute_mappings,
    compute_mappings_backward, sinkhorn_knopp, sinkhorn_knopp_backward,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

// ---------------------------------------------------------------------------
// float64 reference
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct P64 {
    n: usize,
    c: usize,
    eps: f64,
    iters: usize,
    dynamic: bool,
    alpha: [f64; 3],
    b_pre: Vec<f64>,
    b_post: Vec<f64>,
    b_res: Vec<f64>,
    /// `[nC][2n + n²]`, flattened row-major.
    phi: Vec<f64>,
}

impl P64 {
    fn from_layer(p: &MhcParams, config: &MhcConfig) -> Self {
        let f = |a: &mut dyn Iterator<Item = &f32>| a.map(|&v| f64::from(v)).collect::<Vec<_>>();
        Self {
            n: config.expansion_rate,
            c: config.hidden_dim,
            eps: f64::from(config.epsilon),
            iters: config.sinkhorn_iterations,
            dynamic: config.use_dynamic_mappings,
            alpha: [p.alpha_pre, p.alpha_post, p.alpha_res].map(f64::from),
            b_pre: f(&mut p.b_pre.iter()),
            b_post: f(&mut p.b_post.iter()),
            b_res: f(&mut p.b_res.iter()),
            phi: f(&mut p.phi.iter()),
        }
    }
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// The release's Sinkhorn on an `n × n` row-major matrix of logits.
fn sinkhorn64(z: &[f64], n: usize, iters: usize, eps: f64) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        let max = z[i * n..(i + 1) * n]
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        let row: Vec<f64> = (0..n).map(|j| (z[i * n + j] - max).exp()).collect();
        let s: f64 = row.iter().sum();
        for j in 0..n {
            m[i * n + j] = row[j] / s + eps;
        }
    }
    let col = |m: &mut Vec<f64>| {
        for j in 0..n {
            let s: f64 = (0..n).map(|i| m[i * n + j]).sum::<f64>() + eps;
            for i in 0..n {
                m[i * n + j] /= s;
            }
        }
    };
    let row = |m: &mut Vec<f64>| {
        for i in 0..n {
            let s: f64 = m[i * n..(i + 1) * n].iter().sum::<f64>() + eps;
            for j in 0..n {
                m[i * n + j] /= s;
            }
        }
    };
    col(&mut m);
    for _ in 1..iters {
        row(&mut m);
        col(&mut m);
    }
    m
}

/// `(h_pre, h_post, h_res)` for one token whose streams are `x` (`n × c`, row-major).
fn mappings64(p: &P64, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, nc) = (p.n, p.n * p.c);
    let mix = 2 * n + n * n;
    let ms = x.iter().map(|v| v * v).sum::<f64>() / nc as f64;
    let xn: Vec<f64> = x.iter().map(|v| v / (ms + p.eps).sqrt()).collect();
    let proj: Vec<f64> = (0..mix)
        .map(|m| {
            if p.dynamic {
                (0..nc).map(|k| xn[k] * p.phi[k * mix + m]).sum()
            } else {
                0.0
            }
        })
        .collect();
    let h_pre = (0..n)
        .map(|i| sigmoid(p.alpha[0] * proj[i] + p.b_pre[i]) + p.eps)
        .collect();
    let h_post = (0..n)
        .map(|i| 2.0 * sigmoid(p.alpha[1] * proj[n + i] + p.b_post[i]))
        .collect();
    let z: Vec<f64> = (0..n * n)
        .map(|k| p.alpha[2] * proj[2 * n + k] + p.b_res[k])
        .collect();
    (h_pre, h_post, sinkhorn64(&z, n, p.iters, p.eps))
}

/// The sublayer: `h_out = tanh(s ⊙ h_in + t)`.
fn sublayer64(h: &[f64], s: &[f64], t: &[f64]) -> Vec<f64> {
    h.iter()
        .zip(s.iter().zip(t))
        .map(|(h, (s, t))| (s * h + t).tanh())
        .collect()
}

/// One layer for one token: `x' = H^res x + H^postᵀ F(H^pre x)`.
fn layer64(p: &P64, x: &[f64], s: &[f64], t: &[f64]) -> Vec<f64> {
    let (n, c) = (p.n, p.c);
    let (h_pre, h_post, h_res) = mappings64(p, x);
    let h_in: Vec<f64> = (0..c)
        .map(|k| (0..n).map(|i| h_pre[i] * x[i * c + k]).sum())
        .collect();
    let h_out = sublayer64(&h_in, s, t);
    (0..n * c)
        .map(|idx| {
            let (i, k) = (idx / c, idx % c);
            (0..n).map(|j| h_res[i * n + j] * x[j * c + k]).sum::<f64>() + h_post[i] * h_out[k]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

const BATCH: usize = 2;

fn config(dynamic: bool) -> MhcConfig {
    MhcConfig {
        expansion_rate: 4,
        hidden_dim: 3,
        use_dynamic_mappings: dynamic,
        sinkhorn_iterations: 20,
        ..MhcConfig::default()
    }
}

/// Parameters with every entry order 1, so the sigmoids, the softmax and the
/// normalizations all run in their non-saturated, non-linear range.
fn rich_params(config: &MhcConfig, seed: u64) -> MhcParams {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut u = |scale: f32| rng.random_range(-scale..scale);
    let n = config.expansion_rate;
    let mut p = MhcParams::identity_init(config);
    p.alpha_pre = 0.6;
    p.alpha_post = -0.5;
    p.alpha_res = 0.8;
    p.b_pre = Array1::from_shape_fn(n, |_| u(1.0));
    p.b_post = Array1::from_shape_fn(n, |_| u(1.0));
    p.b_res = Array2::from_shape_fn((n, n), |_| u(1.0));
    p.phi = Array2::from_shape_fn(p.phi.raw_dim(), |_| u(1.0));
    p
}

fn x_fixture(config: &MhcConfig) -> Array3<f32> {
    let mut rng = StdRng::seed_from_u64(99);
    // Scale 2.5 puts the RMS well away from 1, so a missing normalization
    // factor in the RMSNorm backward cannot hide.
    Array3::from_shape_fn((BATCH, config.expansion_rate, config.hidden_dim), |_| {
        rng.random_range(-2.5f32..2.5)
    })
}

fn weights(shape: (usize, usize, usize), seed: u64) -> Array3<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array3::from_shape_fn(shape, |_| rng.random_range(-1.0f32..1.0))
}

fn flat64(a: &Array3<f32>, b: usize) -> Vec<f64> {
    a.index_axis(ndarray::Axis(0), b)
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

/// Largest `|analytic − fd| / max(1, |fd|)`.
fn rel_err(analytic: f64, fd: f64) -> f64 {
    (analytic - fd).abs() / fd.abs().max(1.0)
}

/// Central difference of `loss` in coordinate `k` of `v`.
fn fd(loss: &dyn Fn(&[f64]) -> f64, v: &[f64], k: usize) -> f64 {
    let h = 1e-6;
    let mut vp = v.to_vec();
    let mut vm = v.to_vec();
    vp[k] += h;
    vm[k] -= h;
    (loss(&vp) - loss(&vm)) / (2.0 * h)
}

// ---------------------------------------------------------------------------
// Sinkhorn
// ---------------------------------------------------------------------------

#[test]
fn sinkhorn_forward_matches_float64_reference() {
    let mut rng = StdRng::seed_from_u64(5);
    for iters in [1, 2, 5, 20] {
        let z = Array2::from_shape_fn((4, 4), |_| rng.random_range(-3.0f32..3.0));
        let cfg = SinkhornConfig {
            iterations: iters,
            epsilon: 1e-6,
        };
        let got = sinkhorn_knopp(&z, &cfg).matrix;
        let zf: Vec<f64> = z.iter().map(|&v| f64::from(v)).collect();
        let want = sinkhorn64(&zf, 4, iters, 1e-6);
        for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(g) - w).abs() < 3e-6,
                "iters {iters} [{k}] {g} vs {w}"
            );
        }
    }
}

#[test]
fn sinkhorn_backward_matches_finite_differences() {
    let mut rng = StdRng::seed_from_u64(6);
    for (n, iters) in [(3usize, 1usize), (3, 2), (4, 20), (5, 7)] {
        let z = Array2::from_shape_fn((n, n), |_| rng.random_range(-2.0f32..2.0));
        let w = Array2::from_shape_fn((n, n), |_| rng.random_range(-1.0f32..1.0));
        let cfg = SinkhornConfig {
            iterations: iters,
            epsilon: 1e-6,
        };
        let analytic = sinkhorn_knopp_backward(&z, &w, &cfg);
        let zf: Vec<f64> = z.iter().map(|&v| f64::from(v)).collect();
        let wf: Vec<f64> = w.iter().map(|&v| f64::from(v)).collect();
        let loss = |z: &[f64]| -> f64 {
            sinkhorn64(z, n, iters, 1e-6)
                .iter()
                .zip(&wf)
                .map(|(m, w)| m * w)
                .sum()
        };
        for k in 0..n * n {
            let want = fd(&loss, &zf, k);
            let got = f64::from(analytic[[k / n, k % n]]);
            assert!(
                rel_err(got, want) < 2e-4,
                "n {n} iters {iters} [{k}]: analytic {got} vs fd {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Mappings
// ---------------------------------------------------------------------------

fn check_mappings_backward(dynamic: bool) {
    let cfg = config(dynamic);
    let params = rich_params(&cfg, 1);
    let x = x_fixture(&cfg);
    let n = cfg.expansion_rate;
    let w_pre = weights((BATCH, 1, n), 11);
    let w_post = weights((BATCH, 1, n), 12);
    let w_res = weights((BATCH, n, n), 13);

    let g_pre = w_pre.index_axis(ndarray::Axis(1), 0).to_owned();
    let g_post = w_post.index_axis(ndarray::Axis(1), 0).to_owned();
    let (grad_x, grads) = compute_mappings_backward(&x, &params, &cfg, &g_pre, &g_post, &w_res);

    let p64 = P64::from_layer(&params, &cfg);
    // loss = Σ_tokens ⟨w_pre, H^pre⟩ + ⟨w_post, H^post⟩ + ⟨w_res, H^res⟩
    let loss_of = |p: &P64, xs: &[Vec<f64>]| -> f64 {
        (0..BATCH)
            .map(|b| {
                let (hp, ho, hr) = mappings64(p, &xs[b]);
                let wp = flat64(&w_pre, b);
                let wo = flat64(&w_post, b);
                let wr = flat64(&w_res, b);
                hp.iter().zip(&wp).map(|(a, w)| a * w).sum::<f64>()
                    + ho.iter().zip(&wo).map(|(a, w)| a * w).sum::<f64>()
                    + hr.iter().zip(&wr).map(|(a, w)| a * w).sum::<f64>()
            })
            .sum()
    };
    let xs: Vec<Vec<f64>> = (0..BATCH).map(|b| flat64(&x, b)).collect();

    // Gradient with respect to the input.
    let nc = n * cfg.hidden_dim;
    for b in 0..BATCH {
        let loss = |v: &[f64]| {
            let mut xs2 = xs.clone();
            xs2[b] = v.to_vec();
            loss_of(&p64, &xs2)
        };
        for k in 0..nc {
            let want = fd(&loss, &xs[b], k);
            let got = f64::from(grad_x[[b, k / cfg.hidden_dim, k % cfg.hidden_dim]]);
            if dynamic {
                assert!(
                    rel_err(got, want) < 5e-4,
                    "dx[{b},{k}] analytic {got} vs fd {want}"
                );
            } else {
                assert!(
                    want.abs() < 1e-9 && got == 0.0,
                    "static mappings: dx[{b},{k}] {got} vs {want}"
                );
            }
        }
    }

    // Gradient with respect to every parameter.
    let check = |name: &str, analytic: Vec<f64>, get: &dyn Fn(&mut P64) -> &mut Vec<f64>| {
        let base = get(&mut p64.clone()).clone();
        for (k, &got) in analytic.iter().enumerate() {
            let loss = |v: &[f64]| {
                let mut p = p64.clone();
                *get(&mut p) = v.to_vec();
                loss_of(&p, &xs)
            };
            let want = fd(&loss, &base, k);
            assert!(
                rel_err(got, want) < 5e-4,
                "d{name}[{k}] analytic {got} vs fd {want}"
            );
        }
    };
    let up = |a: &Array1<f32>| a.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let up2 = |a: &Array2<f32>| a.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    check("b_pre", up(&grads.d_b_pre), &|p| &mut p.b_pre);
    check("b_post", up(&grads.d_b_post), &|p| &mut p.b_post);
    check("b_res", up2(&grads.d_b_res), &|p| &mut p.b_res);
    if dynamic {
        check("phi", up2(&grads.d_phi), &|p| &mut p.phi);
        let alphas = [grads.d_alpha_pre, grads.d_alpha_post, grads.d_alpha_res];
        for (a, &got) in alphas.iter().enumerate() {
            let loss = |v: &[f64]| {
                let mut p = p64.clone();
                p.alpha[a] = v[0];
                loss_of(&p, &xs)
            };
            let want = fd(&loss, &[p64.alpha[a]], 0);
            assert!(
                rel_err(f64::from(got), want) < 5e-4,
                "dalpha[{a}] {got} vs {want}"
            );
        }
    } else {
        assert!(grads.d_phi.iter().all(|&v| v == 0.0));
        assert_eq!(
            [grads.d_alpha_pre, grads.d_alpha_post, grads.d_alpha_res],
            [0.0; 3]
        );
    }
}

#[test]
fn dynamic_mappings_backward_matches_finite_differences() {
    check_mappings_backward(true);
}

#[test]
fn static_mappings_backward_matches_finite_differences() {
    check_mappings_backward(false);
}

#[test]
fn mappings_forward_matches_float64_reference() {
    let cfg = config(true);
    let params = rich_params(&cfg, 2);
    let x = x_fixture(&cfg);
    let m = compute_mappings(&x, &params, &cfg);
    let p64 = P64::from_layer(&params, &cfg);
    for b in 0..BATCH {
        let (hp, ho, hr) = mappings64(&p64, &flat64(&x, b));
        for i in 0..4 {
            assert!((f64::from(m.h_pre[[b, i]]) - hp[i]).abs() < 2e-6);
            assert!((f64::from(m.h_post[[b, i]]) - ho[i]).abs() < 2e-6);
            for j in 0..4 {
                assert!((f64::from(m.h_res[[b, i, j]]) - hr[i * 4 + j]).abs() < 3e-6);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The whole layer, sublayer in the middle
// ---------------------------------------------------------------------------

fn sublayer_params(c: usize) -> (Vec<f32>, Vec<f32>) {
    let s = (0..c).map(|k| 0.8 + 0.4 * k as f32).collect();
    let t = (0..c).map(|k| 0.3 - 0.2 * k as f32).collect();
    (s, t)
}

fn check_layer_backward(dynamic: bool) {
    let cfg = config(dynamic);
    let params = rich_params(&cfg, 3);
    let layer = MhcLayer::with_params(cfg.clone(), params.clone()).unwrap();
    let x = x_fixture(&cfg);
    let (n, c) = (cfg.expansion_rate, cfg.hidden_dim);
    let g_out = weights((BATCH, n, c), 21);
    let (s32, t32) = sublayer_params(c);

    // Forward through the crate, then backward with the sublayer's own derivative.
    let (h_in, mappings) = layer.pre_layer(&x).unwrap();
    let h_out = Array2::from_shape_fn(h_in.raw_dim(), |(b, k)| {
        (s32[k] * h_in[[b, k]] + t32[k]).tanh()
    });
    let y = layer.post_res_layer(&x, &h_out, &mappings).unwrap();
    let (grad_x, grads) = layer
        .backward(&x, &h_out, &mappings, &g_out, |g_h_out| {
            Array2::from_shape_fn(g_h_out.raw_dim(), |(b, k)| {
                g_h_out[[b, k]] * (1.0 - h_out[[b, k]].powi(2)) * s32[k]
            })
        })
        .unwrap();

    let p64 = P64::from_layer(&params, &cfg);
    let (s, t): (Vec<f64>, Vec<f64>) = (
        s32.iter().map(|&v| f64::from(v)).collect(),
        t32.iter().map(|&v| f64::from(v)).collect(),
    );
    let xs: Vec<Vec<f64>> = (0..BATCH).map(|b| flat64(&x, b)).collect();
    let loss_of = |p: &P64, xs: &[Vec<f64>]| -> f64 {
        (0..BATCH)
            .map(|b| {
                let y = layer64(p, &xs[b], &s, &t);
                y.iter()
                    .zip(flat64(&g_out, b))
                    .map(|(a, w)| a * w)
                    .sum::<f64>()
            })
            .sum()
    };

    // The forward agrees with the reference.
    for b in 0..BATCH {
        let want = layer64(&p64, &xs[b], &s, &t);
        for (k, &w) in want.iter().enumerate() {
            let got = f64::from(y[[b, k / c, k % c]]);
            assert!((got - w).abs() < 3e-6, "y[{b},{k}] {got} vs {w}");
        }
    }

    // Input gradient: the three paths (residual, H^pre, the mappings) together.
    for b in 0..BATCH {
        let loss = |v: &[f64]| {
            let mut xs2 = xs.clone();
            xs2[b] = v.to_vec();
            loss_of(&p64, &xs2)
        };
        for k in 0..n * c {
            let want = fd(&loss, &xs[b], k);
            let got = f64::from(grad_x[[b, k / c, k % c]]);
            assert!(
                rel_err(got, want) < 5e-4,
                "dx[{b},{k}] analytic {got} vs fd {want}"
            );
        }
    }

    // Parameter gradients.
    let up = |a: &Array1<f32>| a.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let up2 = |a: &Array2<f32>| a.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let check = |name: &str, analytic: Vec<f64>, get: &dyn Fn(&mut P64) -> &mut Vec<f64>| {
        let base = get(&mut p64.clone()).clone();
        for (k, &got) in analytic.iter().enumerate() {
            let loss = |v: &[f64]| {
                let mut p = p64.clone();
                *get(&mut p) = v.to_vec();
                loss_of(&p, &xs)
            };
            let want = fd(&loss, &base, k);
            assert!(
                rel_err(got, want) < 5e-4,
                "d{name}[{k}] analytic {got} vs fd {want}"
            );
        }
    };
    check("b_pre", up(&grads.d_b_pre), &|p| &mut p.b_pre);
    check("b_post", up(&grads.d_b_post), &|p| &mut p.b_post);
    check("b_res", up2(&grads.d_b_res), &|p| &mut p.b_res);
    if dynamic {
        check("phi", up2(&grads.d_phi), &|p| &mut p.phi);
        for (a, got) in [grads.d_alpha_pre, grads.d_alpha_post, grads.d_alpha_res]
            .into_iter()
            .enumerate()
        {
            let loss = |v: &[f64]| {
                let mut p = p64.clone();
                p.alpha[a] = v[0];
                loss_of(&p, &xs)
            };
            let want = fd(&loss, &[p64.alpha[a]], 0);
            assert!(
                rel_err(f64::from(got), want) < 5e-4,
                "dalpha[{a}] {got} vs {want}"
            );
        }
    }
}

#[test]
fn layer_backward_matches_finite_differences_dynamic() {
    check_layer_backward(true);
}

#[test]
fn layer_backward_matches_finite_differences_static() {
    check_layer_backward(false);
}

#[test]
fn gradients_accumulate_across_micro_batches() {
    // Gradients are linear in the batch: two halves summed equal the whole.
    let cfg = config(true);
    let layer = MhcLayer::with_params(cfg.clone(), rich_params(&cfg, 4)).unwrap();
    let x = x_fixture(&cfg);
    let g = weights(x.dim(), 31);
    let run = |x: &Array3<f32>, g: &Array3<f32>| -> (Array3<f32>, MhcGradients) {
        let (h_in, m) = layer.pre_layer(x).unwrap();
        let h_out = h_in.mapv(f32::tanh);
        layer
            .backward(x, &h_out, &m, g, |gh| {
                gh * &h_in.mapv(|v| 1.0 - v.tanh().powi(2))
            })
            .unwrap()
    };
    let (_, whole) = run(&x, &g);
    let take = |a: &Array3<f32>, b: usize| a.slice(ndarray::s![b..b + 1, .., ..]).to_owned();
    let (_, mut sum) = run(&take(&x, 0), &take(&g, 0));
    let (_, second) = run(&take(&x, 1), &take(&g, 1));
    sum.accumulate(&second);
    let tol = 1e-4 * (1.0 + whole.norm());
    assert!((sum.norm() - whole.norm()).abs() < tol);
    assert!(
        sum.d_phi
            .iter()
            .zip(&whole.d_phi)
            .all(|(a, b)| (a - b).abs() < tol),
        "phi gradients differ"
    );
}
