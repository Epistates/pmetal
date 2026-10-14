//! Properties that hold for every input, not just the fixtures: what the
//! Sinkhorn projection guarantees whatever the logits, and how the mappings
//! behave at the numeric extremes a bf16 residual stream can reach.

use ndarray::{Array2, Array3};
use pmetal_mhc::{
    MhcConfig, MhcParams, SinkhornConfig, amax_gain_backward, compute_mappings,
    is_doubly_stochastic, sinkhorn_knopp,
};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn cfg() -> SinkhornConfig {
    SinkhornConfig::default()
}

fn square(max_abs: f32) -> impl Strategy<Value = Array2<f32>> {
    (1usize..=8).prop_flat_map(move |n| {
        proptest::collection::vec(-max_abs..max_abs, n * n)
            .prop_map(move |v| Array2::from_shape_vec((n, n), v).unwrap())
    })
}

proptest! {
    /// Whatever the logits, even ±60 (where `exp` of the larger overflows
    /// `f32` three times over), the result is finite, non-negative, has columns
    /// summing to 1 (the last step normalizes them) and a backward gain ≤ 1.
    #[test]
    fn projection_is_finite_nonnegative_and_column_stochastic(z in square(60.0)) {
        let r = sinkhorn_knopp(&z, &cfg());
        prop_assert!(r.matrix.iter().all(|v| v.is_finite() && *v >= 0.0));
        prop_assert!(r.col_error < 1e-5, "col error {}", r.col_error);
        prop_assert!(amax_gain_backward(&r.matrix) <= 1.0 + 1e-5);
    }

    /// For logits within ±1.5 (a factor of e^3 between entries), the paper's 20
    /// iterations leave the rows within 1e-3 of 1 and the columns exact. The
    /// projection is approximate by design, and slower the more skewed the
    /// matrix: at ±3 a 2x2 case needs more than 20 iterations to reach 1e-2.
    #[test]
    fn moderate_logits_converge_to_doubly_stochastic(z in square(1.5)) {
        let r = sinkhorn_knopp(&z, &cfg());
        prop_assert!(is_doubly_stochastic(&r.matrix, 1e-3), "{:?}", r);
    }

    /// The softmax start makes the projection invariant to adding a constant
    /// to a row of the logits (the paper's `exp` + first row step does too).
    #[test]
    fn row_shifts_of_the_logits_change_nothing(
        z in square(3.0),
        shifts in proptest::collection::vec(-20.0f32..20.0, 8),
    ) {
        let n = z.nrows();
        let shifted = Array2::from_shape_fn((n, n), |(i, j)| z[[i, j]] + shifts[i]);
        let a = sinkhorn_knopp(&z, &cfg()).matrix;
        let b = sinkhorn_knopp(&shifted, &cfg()).matrix;
        for (x, y) in a.iter().zip(&b) {
            // f32 rounding of `z + shift` costs ~2e-6 of logit.
            prop_assert!((x - y).abs() < 2e-5, "{x} vs {y}");
        }
    }

    /// More iterations never leave the manifold: the error after 40 is no
    /// worse than after 4 (up to rounding) for moderate logits.
    #[test]
    fn more_iterations_do_not_hurt(z in square(3.0)) {
        let few = sinkhorn_knopp(&z, &SinkhornConfig { iterations: 4, ..cfg() });
        let many = sinkhorn_knopp(&z, &SinkhornConfig { iterations: 40, ..cfg() });
        prop_assert!(many.row_error <= few.row_error + 1e-6);
    }
}

// ---------------------------------------------------------------------------
// bf16 and extreme magnitudes
// ---------------------------------------------------------------------------

/// Round an `f32` to the nearest bf16 value (round half to even), as a bf16
/// residual stream would store it. Finite inputs only.
fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
}

fn layer_fixture() -> (MhcConfig, MhcParams, Array3<f32>) {
    let config = MhcConfig {
        hidden_dim: 32,
        ..MhcConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(17);
    let mut params = MhcParams::with_rng(&config, &mut rng);
    // Gates at 0.3 so the dynamic part moves the mappings visibly.
    params.alpha_pre = 0.3;
    params.alpha_post = 0.3;
    params.alpha_res = 0.3;
    params.phi.mapv_inplace(|v| v * 10.0);
    let x = Array3::from_shape_fn((6, 4, 32), |_| rng.random_range(-3.0f32..3.0));
    (config, params, x)
}

#[test]
fn bf16_streams_move_the_mappings_by_their_rounding_only() {
    let (config, params, x) = layer_fixture();
    let xb = x.mapv(bf16);
    // bf16 keeps 8 significant bits: each entry moves by at most 2^-9 relative.
    let max_rel = x
        .iter()
        .zip(&xb)
        .map(|(a, b)| ((a - b) / a).abs())
        .fold(0.0f32, f32::max);
    assert!(max_rel <= 2f32.powi(-8), "{max_rel}");

    let a = compute_mappings(&x, &params, &config);
    let b = compute_mappings(&xb, &params, &config);
    let diff = |p: &Array2<f32>, q: &Array2<f32>| {
        p.iter()
            .zip(q)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        diff(&a.h_pre, &b.h_pre) < 2e-3,
        "{}",
        diff(&a.h_pre, &b.h_pre)
    );
    assert!(
        diff(&a.h_post, &b.h_post) < 4e-3,
        "{}",
        diff(&a.h_post, &b.h_post)
    );
    let res_diff = a
        .h_res
        .iter()
        .zip(&b.h_res)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(res_diff < 2e-3, "{res_diff}");
    // The perturbed matrices are as doubly stochastic as the unperturbed ones:
    // columns exact, rows as converged as the same projection leaves them.
    for (ma, mb) in a.h_res.outer_iter().zip(b.h_res.outer_iter()) {
        let err = |m: ndarray::ArrayView2<'_, f32>| {
            (0..4)
                .map(|i| (m.row(i).sum() - 1.0).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(err(mb) <= err(ma) + 1e-3, "{} vs {}", err(mb), err(ma));
        assert!(
            mb.columns()
                .into_iter()
                .all(|c| (c.sum() - 1.0).abs() < 1e-5)
        );
    }
}

#[test]
fn mappings_are_invariant_to_the_scale_of_the_streams() {
    // RMSNorm divides the scale out once it dominates its epsilon; scales
    // whose squares overflow `f32` must not break it (the mean of squares is
    // accumulated in f64).
    let (config, params, x) = layer_fixture();
    let base = compute_mappings(&x, &params, &config);
    for scale in [1e2f32, 1e3, 1e15, 1e30] {
        let m = compute_mappings(&x.mapv(|v| v * scale), &params, &config);
        for (a, b) in base.h_pre.iter().zip(&m.h_pre) {
            assert!((a - b).abs() < 1e-4, "scale {scale}: {a} vs {b}");
        }
        for (a, b) in base.h_res.iter().zip(&m.h_res) {
            assert!((a - b).abs() < 1e-4, "scale {scale}: {a} vs {b}");
        }
    }
}

#[test]
fn zero_and_tiny_streams_give_the_static_mappings() {
    let (config, params, x) = layer_fixture();
    let static_cfg = MhcConfig {
        use_dynamic_mappings: false,
        ..config.clone()
    };
    let want = compute_mappings(&x, &params, &static_cfg);
    for scale in [0.0f32, 1e-30] {
        let m = compute_mappings(&x.mapv(|v| v * scale), &params, &config);
        assert!(m.h_pre.iter().chain(&m.h_post).all(|v| v.is_finite()));
        for (a, b) in want.h_res.iter().zip(&m.h_res) {
            assert!((a - b).abs() < 1e-5, "scale {scale}: {a} vs {b}");
        }
    }
}
