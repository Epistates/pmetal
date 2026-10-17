//! How an adapter's `A` and `B` start out.
//!
//! `Standard` is Hu et al. (2021): `A` uniform, `B` zero, so the adapter is a
//! no-op and the base weight is untouched. The others start the adapter on
//! something the base weight tells them, and take that part out of the base so
//! that `W_base + s·B·A` still equals the weight they were given:
//!
//! - **PiSSA** (Meng et al., 2024): `A` and `B` hold the top-`r` singular
//!   triplets of `W`, the base keeps the residual. The adapter then trains the
//!   directions that matter most, at the full learning rate, and the frozen
//!   part is the low-energy tail (which a quantizer also handles better).
//! - **OLoRA** (Büyükakyüz, 2024): from the QR decomposition `W = QR`, `B` is
//!   the first `r` columns of `Q` and `A` the first `r` rows of `R`.
//! - **LoftQ** (Li et al., 2023): for a quantized base. `A` and `B` hold the
//!   top-`r` SVD of the quantization residual `W − Q`, with `Q` re-quantized
//!   from `W − s·B·A` alternately, so the adapter starts where the quantizer
//!   left the weight.
//!
//! All of them work on the weight in f32 and round the new base once to the
//! dtype it had.

use pmetal_core::LoraInit;
use pmetal_runtime::compat::{Array, Dtype, Exception, Linear, random};

use crate::LoraError;

/// Power iterations of the randomized SVD (Halko et al., 2011). Four is what
/// the PiSSA paper's fast SVD uses.
pub const RANDOMIZED_SVD_ITERATIONS: i32 = 4;

/// Below this many columns on the short side an exact SVD is cheap enough that
/// the randomized one is not worth its approximation.
const EXACT_SVD_LIMIT: i32 = 384;

/// Extra columns the randomized sketch carries beyond the rank asked for.
const OVERSAMPLE: i32 = 16;

/// Alternations of LoftQ's quantize / SVD loop (the paper's `T`).
pub const LOFTQ_ITERATIONS: usize = 5;

/// The factors and the new base an initialisation produces, in f32.
pub(crate) struct Factors {
    /// `[rank, in]`.
    pub a: Array,
    /// `[out, rank]`.
    pub b: Array,
    /// The weight the base keeps, `[out, in]`.
    pub base: Array,
}

/// Evaluate `arrays` and surface any error the runtime raised doing it: a
/// failed factorisation does not throw, it leaves a placeholder behind.
fn settle(what: &str, arrays: &[&Array]) -> Result<(), LoraError> {
    for array in arrays {
        Array::clone(array).eval();
    }
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| LoraError::Mlx(Exception::custom(format!("{what}: {e}"))))
}

fn f32_of(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32.as_i32())
}

/// Reduced QR of a tall `[m, n]` matrix, `Q` `[m, min(m, n)]`.
fn orthonormal_basis(y: &Array) -> Array {
    y.qr().0
}

/// The top-`rank` singular triplets of `w`: `(U [m, rank], S [rank],
/// Vt [rank, n])`.
///
/// Exact (a thin SVD) when the short side is small or `niter` is zero,
/// otherwise the randomized range finder with `niter` power iterations and a
/// few columns of oversampling (Halko et al., 2011, algorithm 4.4 with the
/// SVD of the small projected matrix), which costs a few matmuls and a QR of
/// a tall-skinny matrix where the exact one is cubic in the short side.
pub fn truncated_svd(
    w: &Array,
    rank: i32,
    niter: i32,
) -> Result<(Array, Array, Array), LoraError> {
    let (m, n) = (w.dim(0), w.dim(1));
    let short = m.min(n);
    if rank <= 0 || rank > short {
        return Err(LoraError::InvalidState(format!(
            "rank {rank} does not fit a {m}x{n} weight"
        )));
    }
    let w = f32_of(w);
    let take = |u: Array, s: Array, vt: Array| {
        (
            u.slice(&[0, 0], &[m, rank]),
            s.slice(&[0], &[rank]),
            vt.slice(&[0, 0], &[rank, n]),
        )
    };
    let sketch = rank + OVERSAMPLE;
    if short <= EXACT_SVD_LIMIT || niter <= 0 || sketch >= short {
        let (u, s, vt) = w.svd();
        settle("svd", &[&u, &s, &vt])?;
        return Ok(take(u, s, vt));
    }
    let wt = w.t();
    let omega = random::normal(&[n, sketch], Dtype::Float32);
    let mut q = orthonormal_basis(&w.matmul(&omega));
    for _ in 0..niter {
        let z = orthonormal_basis(&wt.matmul(&q));
        q = orthonormal_basis(&w.matmul(&z));
    }
    // W ≈ Q (QᵀW), and the small matrix's SVD carries over.
    let projected = q.t().matmul(&w);
    let (ub, s, vt) = projected.svd();
    settle("randomized svd", &[&q, &ub, &s, &vt])?;
    let u = q.matmul(&ub);
    Ok(take(u, s, vt))
}

/// `U[:, :r]·√S` as `B` and `√S·Vt[:r]` as `A`, with `S` divided by the
/// adapter's scale so that `scale·B·A` is the rank-`r` part itself.
fn balanced_factors(u: &Array, s: &Array, vt: &Array, scale: f32) -> (Array, Array) {
    let root = s.mul_scalar(1.0 / scale).sqrt();
    let rank = root.dim(0);
    let b = u.multiply(&root.reshape(&[1, rank]));
    let a = vt.multiply(&root.reshape(&[rank, 1]));
    (a, b)
}

/// PiSSA: the adapter holds `W`'s top-`rank` principal components, the base
/// the rest. `scale` is the adapter's `alpha / rank` (or `alpha / √rank`).
pub(crate) fn pissa(w: &Array, rank: i32, scale: f32, niter: i32) -> Result<Factors, LoraError> {
    let w = f32_of(w);
    let (u, s, vt) = truncated_svd(&w, rank, niter)?;
    let (a, b) = balanced_factors(&u, &s, &vt, scale);
    let base = w.subtract(&b.matmul(&a).mul_scalar(scale));
    Ok(Factors { a, b, base })
}

/// OLoRA: `W = QR`, `B = Q[:, :r]`, `A = R[:r]`, and the base keeps
/// `W − s·B·A`. Needs `rank ≤ min(out, in)`.
pub(crate) fn olora(w: &Array, rank: i32, scale: f32) -> Result<Factors, LoraError> {
    let (m, n) = (w.dim(0), w.dim(1));
    if rank <= 0 || rank > m.min(n) {
        return Err(LoraError::InvalidState(format!(
            "rank {rank} does not fit a {m}x{n} weight"
        )));
    }
    let w = f32_of(w);
    let (q, r) = w.qr();
    settle("qr", &[&q, &r])?;
    let b = q.slice(&[0, 0], &[m, rank]);
    let a = r.slice(&[0, 0], &[rank, n]);
    let base = w.subtract(&b.matmul(&a).mul_scalar(scale));
    Ok(Factors { a, b, base })
}

/// LoftQ (Li et al., 2023, algorithm 1). `quantize` maps a dense weight to
/// what the quantized base will compute with (pack, then unpack).
///
/// Returns the dense weight to hand to the quantizer afterwards (`W − s·B·A`
/// from the last round, which packs to the `Q` the factors were fitted to),
/// and the factors: the top-`rank` SVD of `W − Q`.
pub(crate) fn loftq(
    w: &Array,
    rank: i32,
    scale: f32,
    iterations: usize,
    quantize: &dyn Fn(&Array) -> Result<Array, LoraError>,
) -> Result<Factors, LoraError> {
    let w = f32_of(w);
    let mut a = Array::zeros(&[rank, w.dim(1)], Dtype::Float32.as_i32());
    let mut b = Array::zeros(&[w.dim(0), rank], Dtype::Float32.as_i32());
    let mut base = w.clone();
    for _ in 0..iterations.max(1) {
        // The part of W the factors do not carry is what gets quantized.
        base = w.subtract(&b.matmul(&a).mul_scalar(scale));
        let q = f32_of(&quantize(&base)?);
        let (u, s, vt) = truncated_svd(&w.subtract(&q), rank, RANDOMIZED_SVD_ITERATIONS)?;
        (a, b) = balanced_factors(&u, &s, &vt, scale);
    }
    Ok(Factors { a, b, base })
}

/// Initialise the adapter attached to `linear` (`init` other than
/// `Standard`), replacing the base weight with the residual.
///
/// `quantize`, for LoftQ, packs and unpacks a dense weight the way the
/// layer's base will be packed.
pub(crate) fn initialise(
    linear: &mut Linear,
    init: LoraInit,
    quantize: Option<&dyn Fn(&Array) -> Result<Array, LoraError>>,
) -> Result<(), LoraError> {
    if init == LoraInit::Standard {
        return Ok(());
    }
    if linear.quant.is_some() {
        return Err(LoraError::InvalidState(format!(
            "the {init} initialisation reads the dense weight, and this projection is already \
             packed; initialise first and pack after (`AdaptedModel::attach_quantized`)"
        )));
    }
    let dtype = linear.weight.value.dtype();
    if dtype == Dtype::Uint8 {
        return Err(LoraError::InvalidState(format!(
            "the {init} initialisation needs a floating-point base weight, and this one is FP8"
        )));
    }
    let adapter = linear
        .adapter
        .as_deref()
        .ok_or_else(|| LoraError::InvalidState("no adapter attached".into()))?;
    let (rank, scale) = (adapter.rank, adapter.scale);
    let w = linear.weight.value.clone();
    let factors = match init {
        LoraInit::Standard => unreachable!("handled above"),
        LoraInit::Pissa => pissa(&w, rank, scale, RANDOMIZED_SVD_ITERATIONS)?,
        LoraInit::Olora => olora(&w, rank, scale)?,
        LoraInit::Loftq => {
            let quantize = quantize.ok_or_else(|| {
                LoraError::InvalidState(
                    "the loftq initialisation fits the adapter to a quantizer; load the model \
                     with a QLoRA scheme (`AdaptedModel::attach_quantized`)"
                        .into(),
                )
            })?;
            loftq(&w, rank, scale, LOFTQ_ITERATIONS, quantize)?
        }
    };
    let adapter = linear
        .adapter
        .as_deref_mut()
        .ok_or_else(|| LoraError::InvalidState("no adapter attached".into()))?;
    adapter.a = factors.a;
    adapter.b = factors.b;
    linear.weight.value = factors.base.as_dtype(dtype.as_i32());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QLoraConfig, QLoraScheme};
    use pmetal_runtime::compat::Param;

    /// `‖a − b‖_F / ‖b‖_F`.
    fn relative_error(a: &Array, b: &Array) -> f32 {
        let num = f32_of(a).subtract(&f32_of(b)).square().sum(None);
        let den = f32_of(b).square().sum(None);
        let (num, den) = (num.item_f32(), den.item_f32());
        assert!(num.is_finite() && den > 0.0, "num {num}, den {den}");
        (num / den).sqrt()
    }

    fn drain() {
        pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    }

    /// `U diag(s) Vt` with orthonormal `U`, `V` (from QR of Gaussians) and a
    /// geometrically decaying spectrum, so the top singular triplets are
    /// well separated and the truncation is unambiguous.
    fn decaying(m: i32, n: i32, decay: f32) -> (Array, Vec<f32>) {
        random::seed(21);
        let k = m.min(n);
        let u = random::normal(&[m, k], Dtype::Float32).qr().0;
        let v = random::normal(&[n, k], Dtype::Float32).qr().0;
        let s: Vec<f32> = (0..k).map(|i| decay.powi(i)).collect();
        let sdiag = Array::from_f32_slice(&s, &[1, k]);
        (u.multiply(&sdiag).matmul(&v.t()), s)
    }

    fn values(a: &Array) -> Vec<f32> {
        let mut a = f32_of(a);
        a.eval();
        let n = a.shape().iter().map(|d| *d as usize).product();
        a.to_f32_vec(n).expect("readable")
    }

    /// PiSSA, by its definition (Meng et al., 2024, eq. 2-3): with
    /// `W = U S Vᵀ`, `A = √S_r Vᵣᵀ`, `B = U_r √S_r`, and the base keeps
    /// `W − U_r S_r Vᵣᵀ`.
    #[test]
    fn pissa_splits_the_weight_into_its_principal_part_and_the_rest() {
        let (w, s) = decaying(48, 64, 0.7);
        let (rank, scale) = (4, 2.0_f32);
        let f = pissa(&w, rank, scale, RANDOMIZED_SVD_ITERATIONS).unwrap();
        drain();

        // The adapter and the base together are the weight.
        let rebuilt = f.base.add(&f.b.matmul(&f.a).mul_scalar(scale));
        assert!(relative_error(&rebuilt, &w) < 1e-5);

        // The adapter is the best rank-r approximation (Eckart-Young): the
        // residual's norm is the tail of the spectrum.
        let tail: f32 = s[rank as usize..].iter().map(|x| x * x).sum::<f32>().sqrt();
        let residual = f.base.square().sum(None).item_f32().sqrt();
        assert!(
            (residual - tail).abs() / tail < 1e-3,
            "residual norm {residual}, tail of the spectrum {tail}"
        );

        // `A` and `B` are balanced: AAᵀ = BᵀB = diag(S_r / scale).
        let want: Vec<f32> = s[..rank as usize].iter().map(|x| x / scale).collect();
        for gram in [f.a.matmul(&f.a.t()), f.b.t().matmul(&f.b)] {
            let g = values(&gram);
            for i in 0..rank as usize {
                for j in 0..rank as usize {
                    let expect = if i == j { want[i] } else { 0.0 };
                    let got = g[i * rank as usize + j];
                    assert!(
                        (got - expect).abs() < 1e-4,
                        "gram[{i}][{j}] = {got}, want {expect}"
                    );
                }
            }
        }
    }

    /// The randomized SVD finds the same principal part the exact one does,
    /// on a weight past the exact cutoff.
    #[test]
    fn the_randomized_svd_agrees_with_the_exact_one() {
        let (w, s) = decaying(448, 512, 0.93);
        let rank = 8;
        assert!(
            w.dim(0).min(w.dim(1)) > EXACT_SVD_LIMIT,
            "fixture is on the exact path"
        );
        let (u, got, vt) = truncated_svd(&w, rank, RANDOMIZED_SVD_ITERATIONS).unwrap();
        drain();
        let got = values(&got);
        for (i, g) in got.iter().enumerate() {
            assert!((g - s[i]).abs() / s[i] < 2e-3, "σ{i}: {g} against {}", s[i]);
        }
        // The rank-r reconstruction is the exact truncation's.
        let approx = u
            .multiply(&Array::from_f32_slice(&got, &[1, rank]))
            .matmul(&vt);
        let (eu, es, evt) = truncated_svd(&w, rank, 0).unwrap();
        let exact = eu.multiply(&es.reshape(&[1, rank])).matmul(&evt);
        assert!(relative_error(&approx, &exact) < 5e-3);
    }

    /// OLoRA, by its definition (Büyükakyüz, 2024): `W = QR`, `B = Q[:, :r]`,
    /// `A = R[:r]`, base `W − s·B·A`.
    #[test]
    fn olora_takes_the_first_columns_of_q_and_rows_of_r() {
        let (w, _) = decaying(48, 64, 0.9);
        let (rank, scale) = (5, 0.5_f32);
        let f = olora(&w, rank, scale).unwrap();
        drain();
        let rebuilt = f.base.add(&f.b.matmul(&f.a).mul_scalar(scale));
        assert!(relative_error(&rebuilt, &w) < 1e-5);

        // `B` has orthonormal columns and `A` is upper triangular.
        let g = values(&f.b.t().matmul(&f.b));
        for i in 0..rank as usize {
            for j in 0..rank as usize {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((g[i * rank as usize + j] - expect).abs() < 1e-4);
            }
        }
        let a = values(&f.a);
        let n = w.dim(1) as usize;
        for i in 1..rank as usize {
            for j in 0..i {
                assert!(a[i * n + j].abs() < 1e-5, "A[{i}][{j}] = {}", a[i * n + j]);
            }
        }
        assert!(a.iter().any(|x| x.abs() > 1e-3), "A is zero");
    }

    #[test]
    fn a_rank_beyond_the_weight_is_refused() {
        let (w, _) = decaying(8, 16, 0.9);
        assert!(pissa(&w, 9, 1.0, 0).is_err());
        assert!(olora(&w, 9, 1.0).is_err());
        assert!(pissa(&w, 0, 1.0, 0).is_err());
    }

    /// LoftQ (Li et al., 2023, algorithm 1): the quantized base plus the
    /// adapter is closer to `W` than the quantized base alone, and alternating
    /// improves on a single round.
    #[test]
    fn loftq_fits_the_adapter_to_the_quantization_residual() {
        random::seed(5);
        let w = random::normal(&[64, 128], Dtype::Float32).mul_scalar(0.05);
        let qlora = QLoraConfig {
            scheme: QLoraScheme::Nf4,
            group_size: 64,
            double_quant: false,
        };
        let quantize = |x: &Array| qlora.round_trip(x);
        let (rank, scale) = (8, 2.0_f32);

        let naive = relative_error(&quantize(&w).unwrap(), &w);
        let error_after = |iterations: usize| {
            let f = loftq(&w, rank, scale, iterations, &quantize).unwrap();
            // What the layer computes: the base as it will be packed, plus
            // the adapter.
            let q = quantize(&f.base).unwrap();
            relative_error(&q.add(&f.b.matmul(&f.a).mul_scalar(scale)), &w)
        };
        let one = error_after(1);
        let many = error_after(LOFTQ_ITERATIONS);
        drain();
        assert!(
            one < naive * 0.97,
            "one round {one}, plain quantization {naive}"
        );
        assert!(
            many <= one * 1.0001,
            "{LOFTQ_ITERATIONS} rounds {many}, one round {one}"
        );
    }

    /// On a layer: the dtype of the base is kept, the weight the layer computes
    /// is the one it was given, and the adapter is live.
    #[test]
    fn initialising_a_bf16_layer_keeps_its_dtype_and_function() {
        random::seed(8);
        let w = random::normal(&[32, 64], Dtype::Float32).mul_scalar(0.1);
        for init in [LoraInit::Pissa, LoraInit::Olora] {
            let mut layer = Linear {
                weight: Param::new(w.as_dtype(Dtype::Bfloat16.as_i32())),
                bias: Param::new(None),
                adapter: None,
                quant: None,
            };
            let before = layer.weight.value.clone();
            layer.attach_lora(4, 8.0, false).unwrap();
            initialise(&mut layer, init, None).unwrap();
            assert_eq!(layer.weight.value.dtype(), Dtype::Bfloat16, "{init}");
            let adapter = layer.adapter.as_ref().unwrap();
            assert!(
                adapter.b.square().sum(None).item_f32() > 0.0,
                "{init}: B is zero"
            );
            // The folded weight is the original one up to bf16 rounding of
            // the residual.
            let folded = adapter.merged_weight(&layer.weight.value);
            assert!(relative_error(&folded, &before) < 8e-3, "{init}");
            // And the layer's output agrees on an input.
            let x = random::normal(&[3, 64], Dtype::Float32).as_dtype(Dtype::Bfloat16.as_i32());
            let got = layer.forward(&x);
            let want = x.matmul(&before.t());
            assert!(relative_error(&got, &want) < 2e-2, "{init}");
        }
        drain();
    }

    #[test]
    fn an_initialisation_that_cannot_run_says_why() {
        random::seed(9);
        let w = random::normal(&[32, 64], Dtype::Float32);
        let mut layer = Linear {
            weight: Param::new(w),
            bias: Param::new(None),
            adapter: None,
            quant: None,
        };
        layer.attach_lora(4, 8.0, false).unwrap();
        let err = initialise(&mut layer, LoraInit::Loftq, None).unwrap_err();
        assert!(err.to_string().contains("quantizer"), "{err}");
        // Packed already: the dense weight is gone.
        layer.detach_lora();
        layer
            .quantize_codebook(&crate::NF4_CODEBOOK, 64, false)
            .unwrap();
        layer.attach_lora(4, 8.0, false).unwrap();
        let err = initialise(&mut layer, LoraInit::Pissa, None).unwrap_err();
        assert!(err.to_string().contains("already packed"), "{err}");
    }
}
