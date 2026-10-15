//! Helpers shared by the unit tests.

use pmetal_runtime::compat::Array;

use crate::MergeParameters;

/// An f32 tensor from a flat slice and a shape.
pub(crate) fn arr(values: &[f32], shape: &[i32]) -> Array {
    Array::from_f32_slice(values, shape)
}

/// The values of `array` as f32. A runtime op that threw returns a
/// placeholder instead of failing, so this also fails on any error the
/// runtime recorded since the last check.
pub(crate) fn values(array: &Array) -> Vec<f32> {
    let n: usize = array.shape().iter().map(|&d| d as usize).product();
    let out = array.clone().to_f32_vec(n).expect("array readable as f32");
    pmetal_runtime::check_unobserved_error().expect("runtime op failed");
    out
}

/// Assert `got` equals `want` element-wise within `tol`.
#[track_caller]
pub(crate) fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "index {i}: got {g}, want {w} (tol {tol})\n got:  {got:?}\n want: {want:?}"
        );
    }
}

/// Per-model parameters carrying only a weight and a density.
pub(crate) fn weight_density(weight: f32, density: f32) -> MergeParameters {
    MergeParameters {
        weight: Some(weight.into()),
        density: Some(density.into()),
        ..Default::default()
    }
}
