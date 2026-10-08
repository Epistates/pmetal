//! Multi-axis `squeeze_axes` / `expand_dims_axes` take NumPy's axes: for
//! squeeze they name axes of the input, for expand_dims positions in the
//! output, negative ones counted from the end, in any order.

use pmetal_bridge::check_last_error;
use pmetal_bridge::compat::{Array, Dtype};

#[test]
fn squeeze_axes_with_negative_axes() {
    // squeeze(zeros([2, 1, 1]), axis=(-2, 2)) is [2].
    let x = Array::zeros(&[2, 1, 1], Dtype::Float32.as_i32());
    for axes in [[-2, 2], [2, -2], [1, -1], [-1, 1]] {
        let s = x.squeeze_axes(&axes);
        check_last_error().unwrap_or_else(|e| panic!("squeeze_axes {axes:?}: {e}"));
        assert_eq!(s.shape(), vec![2], "squeeze_axes {axes:?}");
    }
}

#[test]
fn expand_dims_axes_with_negative_axes() {
    let x = Array::zeros(&[3], Dtype::Float32.as_i32());
    for (axes, want) in [
        (vec![-2, -1], vec![3, 1, 1]),
        (vec![-1, -2], vec![3, 1, 1]),
        (vec![-1, 0], vec![1, 3, 1]),
        (vec![0, -1], vec![1, 3, 1]),
    ] {
        let e = x.expand_dims_axes(&axes);
        check_last_error().unwrap_or_else(|err| panic!("expand_dims_axes {axes:?}: {err}"));
        assert_eq!(e.shape(), want, "expand_dims_axes {axes:?}");
    }
    let y = Array::zeros(&[4, 5, 6, 7], Dtype::Float32.as_i32());
    let e = y.expand_dims_axes(&[4, 2]);
    check_last_error().expect("expand_dims_axes [4, 2]");
    assert_eq!(e.shape(), vec![4, 5, 1, 6, 1, 7]);
}
