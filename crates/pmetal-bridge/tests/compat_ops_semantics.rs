//! `compat::ops` functions compute what their namesakes in MLX (and NumPy)
//! compute: the same values, including halfway cases, values past `i32`,
//! tiny arguments, NaN and infinity, and the same output dtype and shape.
//!
//! Expected values are written out by hand from the NumPy / MLX definitions,
//! never read back from another MLX op.

use pmetal_bridge::check_last_error;
use pmetal_bridge::compat::ops::PadMode;
use pmetal_bridge::compat::{Array, Dtype, ops};

/// The op under test, checked for a bridge error before anything else runs
/// (a later successful op clears the thread's error slot).
fn checked(out: Array, what: &str) -> Array {
    check_last_error().unwrap_or_else(|e| panic!("{what}: bridge error {e}"));
    out
}

fn read(a: &Array) -> Vec<f32> {
    let mut a = a.clone();
    a.eval();
    let n = a.size();
    let v = a.to_f32_vec(n).expect("to_f32_vec");
    check_last_error().expect("readback");
    v
}

fn f32s(v: &[f32]) -> Array {
    Array::from_f32_slice(v, &[v.len() as i32])
}

fn bf16s(v: &[f32]) -> Array {
    f32s(v).as_dtype(Dtype::Bfloat16.as_i32())
}

/// Exact match, with NaN equal to NaN and the sign of zero compared.
fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let same = (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits();
        assert!(same, "{what}[{i}]: got {g:?}, want {w:?} (all: {got:?})");
    }
}

// ── log1p ──────────────────────────────────────────────────────────────────

#[test]
fn log1p_keeps_precision_for_tiny_arguments() {
    // log(1 + 1e-10) in f32 is log(1.0) = 0; log1p keeps the 1e-10.
    let x = f32s(&[1e-10, -1e-10, 3e-8]);
    let got = read(&checked(ops::log1p(&x), "log1p"));
    for (g, w) in got.iter().zip([1e-10f32, -1e-10, 3e-8]) {
        assert!(((g - w) / w).abs() < 1e-6, "log1p: got {g:e}, want {w:e}");
    }
}

#[test]
fn log1p_edges_and_dtype() {
    let x = f32s(&[0.0, -1.0, f32::INFINITY, f32::NAN, 1.0]);
    let got = read(&checked(ops::log1p(&x), "log1p"));
    assert_same(
        &got,
        &[
            0.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            std::f32::consts::LN_2,
        ],
        "log1p",
    );
    let y = checked(ops::log1p(&bf16s(&[0.5])), "log1p bf16");
    assert_eq!(y.dtype(), Dtype::Bfloat16, "log1p must keep bf16");
}

#[test]
fn softplus_via_logaddexp_does_not_overflow() {
    // softplus(x) = logaddexp(0, x); log1p(exp(x)) is inf for x > ~88 in f32.
    let x = f32s(&[-50.0, 0.0, 100.0, 1000.0]);
    let got = read(&checked(
        ops::logaddexp(&Array::from_f32(0.0), &x),
        "logaddexp",
    ));
    // softplus(-50) = log1p(e^-50) ≈ e^-50 = 1.9287e-22.
    assert!(
        (got[0] / 1.928_749_8e-22 - 1.0).abs() < 1e-5,
        "softplus(-50) = {:e}",
        got[0]
    );
    assert!((got[1] - std::f32::consts::LN_2).abs() < 1e-7);
    assert_eq!(got[2], 100.0);
    assert_eq!(got[3], 1000.0);
}

// ── round / floor / ceil ───────────────────────────────────────────────────

#[test]
fn round_breaks_ties_to_even() {
    let x = f32s(&[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.4, 2.6, -2.6]);
    let got = read(&checked(ops::round(&x), "round"));
    assert_same(
        &got,
        &[0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0, 2.0, 3.0, -3.0],
        "round",
    );
}

#[test]
fn round_floor_ceil_beyond_i32() {
    // 3e9 and -3e9 are integers in f32 but outside i32.
    let x = f32s(&[3e9, -3e9, 1e20]);
    for (name, out) in [
        ("round", ops::round(&x)),
        ("floor", ops::floor(&x)),
        ("ceil", ops::ceil(&x)),
    ] {
        let got = read(&checked(out, name));
        assert_same(&got, &[3e9, -3e9, 1e20], name);
    }
}

#[test]
fn floor_and_ceil_values() {
    let x = f32s(&[-1.5, 1.5, -0.0, 2.0, f32::INFINITY, f32::NAN]);
    let got = read(&checked(ops::floor(&x), "floor"));
    assert_same(
        &got,
        &[-2.0, 1.0, -0.0, 2.0, f32::INFINITY, f32::NAN],
        "floor",
    );
    let got = read(&checked(ops::ceil(&x), "ceil"));
    assert_same(
        &got,
        &[-1.0, 2.0, -0.0, 2.0, f32::INFINITY, f32::NAN],
        "ceil",
    );
}

#[test]
fn rounding_ops_keep_the_input_dtype() {
    let x = bf16s(&[-1.5, 2.5]);
    for (name, out) in [
        ("round", ops::round(&x)),
        ("floor", ops::floor(&x)),
        ("ceil", ops::ceil(&x)),
    ] {
        assert_eq!(checked(out, name).dtype(), Dtype::Bfloat16, "{name}");
    }
}

// ── tanh / negative ────────────────────────────────────────────────────────

#[test]
fn tanh_is_accurate_near_zero_and_saturates() {
    // 2·σ(2x) − 1 cancels to 0 (or a multiple of f32 epsilon) for tiny x.
    let x = f32s(&[1e-8, -3e-6, 20.0, -20.0, f32::INFINITY, f32::NAN]);
    let got = read(&checked(ops::tanh(&x), "tanh"));
    assert!(
        (got[0] / 1e-8 - 1.0).abs() < 1e-6,
        "tanh(1e-8) = {:e}",
        got[0]
    );
    assert!(
        (got[1] / -3e-6 - 1.0).abs() < 1e-6,
        "tanh(-3e-6) = {:e}",
        got[1]
    );
    assert_same(&got[2..], &[1.0, -1.0, 1.0, f32::NAN], "tanh");
}

#[test]
fn elementwise_unary_ops_keep_the_input_dtype() {
    let x = bf16s(&[-1.5, 0.25, 3.0]);
    for (name, out) in [
        ("tanh", ops::tanh(&x)),
        ("negative", ops::negative(&x)),
        ("ceil", ops::ceil(&x)),
    ] {
        assert_eq!(checked(out, name).dtype(), Dtype::Bfloat16, "{name}");
    }
    let i = Array::from_i32_slice(&[3, -4]);
    let n = checked(ops::negative(&i), "negative int32");
    assert_eq!(n.dtype(), Dtype::Int32, "negative must keep int32");
    assert_same(&read(&n), &[-3.0, 4.0], "negative int32");
}

#[test]
fn negative_flips_the_sign_of_zero() {
    let got = read(&checked(ops::negative(&f32s(&[0.0, -0.0])), "negative"));
    assert_same(&got, &[-0.0, 0.0], "negative");
}

// ── any ────────────────────────────────────────────────────────────────────

/// `[2, 3, 4]` bool, true only at `[1, 2, 3]` and `[0, 0, 1]`.
fn sparse_mask() -> Array {
    let mut v = vec![0.0f32; 24];
    v[12 + 2 * 4 + 3] = 1.0;
    v[1] = 1.0;
    Array::from_f32_slice(&v, &[2, 3, 4]).as_dtype(Dtype::Bool.as_i32())
}

#[test]
fn any_over_several_axes_reduces_those_axes() {
    let m = sparse_mask();
    // Axes 0 and 2 → shape [3]: row 0 (from [0,0,1]) and row 2 (from [1,2,3]).
    let a = checked(ops::any(&m, Some(&[0, 2]), false), "any [0, 2]");
    assert_eq!(a.shape(), vec![3]);
    assert_eq!(a.dtype(), Dtype::Bool);
    assert_same(&read(&a), &[1.0, 0.0, 1.0], "any [0, 2]");

    // Order and negative axes don't matter.
    let b = checked(ops::any(&m, Some(&[-1, 0]), false), "any [-1, 0]");
    assert_same(&read(&b), &[1.0, 0.0, 1.0], "any [-1, 0]");

    // Axes 1 and 2 → shape [2]: both batches have a true.
    let c = checked(ops::any(&m, Some(&[1, 2]), false), "any [1, 2]");
    assert_same(&read(&c), &[1.0, 1.0], "any [1, 2]");
}

#[test]
fn any_honours_keep_dims() {
    let m = sparse_mask();
    let a = checked(ops::any(&m, Some(&[0, 2]), true), "any keep");
    assert_eq!(a.shape(), vec![1, 3, 1]);
    assert_same(&read(&a), &[1.0, 0.0, 1.0], "any keep");

    let all = checked(ops::any(&m, None, true), "any all keep");
    assert_eq!(all.shape(), vec![1, 1, 1]);
    let none = checked(ops::any(&m, None, false), "any all");
    assert_eq!(none.ndim(), 0);
    assert_same(&read(&none), &[1.0], "any all");
}

#[test]
fn any_treats_nan_as_true() {
    // NumPy/MLX truthiness: NaN is nonzero.
    let x = f32s(&[0.0, f32::NAN]);
    let a = checked(ops::any(&x, None, false), "any nan");
    assert_same(&read(&a), &[1.0], "any nan");
}

// ── pad ────────────────────────────────────────────────────────────────────

#[test]
fn pad_modes_follow_numpy() {
    let x = f32s(&[1.0, 2.0, 3.0]);
    let cases: [(PadMode, [f32; 6]); 4] = [
        (PadMode::Constant, [7.0, 7.0, 1.0, 2.0, 3.0, 7.0]),
        (PadMode::Edge, [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]),
        (PadMode::Reflect, [3.0, 2.0, 1.0, 2.0, 3.0, 2.0]),
        (PadMode::Symmetric, [2.0, 1.0, 1.0, 2.0, 3.0, 3.0]),
    ];
    for (mode, want) in cases {
        let name = mode.name();
        let out = checked(ops::pad(&x, &[(2, 1)], Some(mode), Some(7.0)), name);
        assert_same(&read(&out), &want, name);
        assert_eq!(name.parse::<PadMode>(), Ok(mode));
    }
}

#[test]
fn pad_defaults_to_zero_constant_and_keeps_dtype() {
    let x = bf16s(&[1.0, 2.0]).reshape(&[1, 2]);
    let out = checked(ops::pad(&x, &[(1, 0), (0, 1)], None, None), "pad");
    assert_eq!(out.shape(), vec![2, 3]);
    assert_eq!(out.dtype(), Dtype::Bfloat16);
    assert_same(&read(&out), &[0.0, 0.0, 0.0, 1.0, 2.0, 0.0], "pad");
}

#[test]
fn pad_modes_mlx_lacks_are_refused_by_name() {
    for name in ["wrap", "linear_ramp", "mean", "Reflect"] {
        let err = name.parse::<PadMode>().expect_err(name);
        assert!(err.contains(&format!("`{name}`")), "{err}");
    }
}

// ── tile / split ───────────────────────────────────────────────────────────

#[test]
fn tile_repeats_the_whole_array() {
    let x = f32s(&[1.0, 2.0]);
    let t = checked(ops::tile(&x, &[2]), "tile");
    assert_same(&read(&t), &[1.0, 2.0, 1.0, 2.0], "tile");

    // reps longer than ndim prepends axes: [2] tiled by [2, 2] is [2, 4].
    let t = checked(ops::tile(&x, &[2, 2]), "tile 2d");
    assert_eq!(t.shape(), vec![2, 4]);
    assert_same(
        &read(&t),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        "tile 2d",
    );
}

#[test]
fn split_into_equal_sections() {
    let x = f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let parts = ops::split(&x, 3, 0);
    check_last_error().expect("split");
    assert_eq!(parts.len(), 3);
    assert_same(&read(&parts[2]), &[4.0, 5.0], "split");

    // 6 doesn't divide into 4 sections: MLX refuses rather than going uneven.
    let parts = ops::split(&x, 4, 0);
    assert!(check_last_error().is_err(), "uneven split must fail");
    assert!(parts.is_empty());
}

// ── predicates and logical ops ─────────────────────────────────────────────

#[test]
fn predicates_and_logical_ops() {
    let x = bf16s(&[0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0]);
    let nan = checked(ops::is_nan(&x), "is_nan");
    assert_eq!(nan.dtype(), Dtype::Bool);
    assert_same(&read(&nan), &[0.0, 1.0, 0.0, 0.0, 0.0], "is_nan");
    let inf = checked(ops::is_inf(&x), "is_inf");
    assert_same(&read(&inf), &[0.0, 0.0, 1.0, 1.0, 0.0], "is_inf");

    // Truthiness: nonzero (NaN included) is true.
    let not = checked(ops::logical_not(&x), "logical_not");
    assert_eq!(not.dtype(), Dtype::Bool);
    assert_same(&read(&not), &[1.0, 0.0, 0.0, 0.0, 0.0], "logical_not");
    let and = checked(ops::logical_and(&nan, &inf), "logical_and");
    assert_same(&read(&and), &[0.0; 5], "logical_and");
    let or = checked(ops::logical_or(&nan, &inf), "logical_or");
    assert_same(&read(&or), &[0.0, 1.0, 1.0, 1.0, 0.0], "logical_or");
    assert!(ops::item_bool(&checked(ops::any(&or, None, false), "any")));
}

#[test]
fn arange_from_is_exact_past_f32_integers() {
    // 2^24 + 1 is not representable in f32.
    let start = (1 << 24) + 1;
    let a = checked(ops::arange_from(start, start + 2), "arange_from");
    assert_eq!(a.dtype(), Dtype::Int32);
    assert_eq!(a.as_slice::<i32>(), &[start, start + 1]);
}
