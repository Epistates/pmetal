use super::{Array, Dtype};
pub use crate::inline_array::PadMode;
use crate::inline_array::RawBuf;
use std::mem::MaybeUninit;

pub fn maximum(a: &Array, b: &Array) -> Array {
    a.maximum(b)
}
pub fn minimum(a: &Array, b: &Array) -> Array {
    a.minimum(b)
}
pub fn matmul(a: &Array, b: &Array) -> Array {
    a.matmul(b)
}
pub fn softmax_axis(a: &Array, axis: i32) -> Array {
    a.softmax(axis)
}
/// `log(1 + x)`, accurate for tiny `x` (where `1 + x` rounds to 1), like
/// `mx.log1p`. Keeps a floating input's dtype.
pub fn log1p(a: &Array) -> Array {
    a.log1p()
}
/// `log(exp(a) + exp(b))` without overflow, like `mx.logaddexp`.
///
/// `logaddexp(0, x)` is softplus, `log(1 + exp(x))`, finite for every finite
/// `x`.
pub fn logaddexp(a: &Array, b: &Array) -> Array {
    a.logaddexp(b)
}
pub fn broadcast_to(a: &Array, shape: &[i32]) -> Array {
    a.broadcast_to(shape)
}

/// Concatenate a slice of arrays along `axis`.
///
/// Uses the fast two-array path when `arrays.len() == 2`, otherwise chains
/// `concatenate_2` left-to-right (equivalent to `mx.concatenate`).
pub fn concatenate_axis(arrays: &[&Array], axis: i32) -> Array {
    assert!(!arrays.is_empty(), "concatenate_axis: empty array slice");
    if arrays.len() == 1 {
        return arrays[0].clone();
    }
    if arrays.len() == 2 {
        return arrays[0].concatenate_2(arrays[1], axis);
    }
    // For three or more arrays, use the contiguous-buffer MLX path via
    // the `mlx_inline_concatenate` FFI (all RawBufs must be contiguous).
    // We clone each array into a Vec<Array> so we hold the live buffers,
    // then collect raw pointers into a contiguous slice.
    let owned: Vec<Array> = arrays.iter().map(|a| (*a).clone()).collect();
    concatenate_owned_axis(&owned, axis)
}

/// Concatenate owned arrays along `axis`.  The arrays must all remain live
/// for the duration of the call (they are dropped after the FFI returns).
pub fn concatenate_owned_axis(arrays: &[Array], axis: i32) -> Array {
    assert!(
        !arrays.is_empty(),
        "concatenate_owned_axis: empty array slice"
    );
    if arrays.len() == 1 {
        return arrays[0].clone();
    }
    if arrays.len() == 2 {
        return arrays[0].concatenate_2(&arrays[1], axis);
    }
    // Collect RawBufs into a contiguous Vec so the pointer is valid.
    // SAFETY: RawBuf is Copy, and InlineArray exposes its raw field via
    // the internal `as_raw_ptr` method.  We must not let `arrays` drop
    // while we hold the pointer.
    let raw_copies: Vec<RawBuf> = arrays
        .iter()
        .map(|a| {
            // Copy the raw buffer — this is a C++ copy-construct (ref-count bump).
            let mut dst = MaybeUninit::<RawBuf>::uninit();
            unsafe {
                // mlx_inline_init_copy is the copy-constructor trampoline.
                // It is `pub(crate)` in inline_array but we are inside the
                // same crate so this is fine.
                crate::inline_array::raw_copy_buf(dst.as_mut_ptr(), a.as_raw_ptr());
                dst.assume_init()
            }
        })
        .collect();

    let mut dst_raw = MaybeUninit::<RawBuf>::uninit();
    unsafe {
        crate::inline_array::raw_concatenate(
            dst_raw.as_mut_ptr(),
            raw_copies.as_ptr(),
            raw_copies.len() as i32,
            axis,
        );
    }
    // Destroy the temporary copies we made.
    for mut rb in raw_copies {
        unsafe {
            crate::inline_array::raw_destroy(&mut rb);
        }
    }
    unsafe { crate::inline_array::from_raw_buf(dst_raw.assume_init()) }
}

/// Stack arrays along a new axis.
pub fn stack_axis(arrays: &[Array], axis: i32) -> Array {
    assert!(!arrays.is_empty(), "stack_axis: empty array slice");
    let raw_copies: Vec<RawBuf> = arrays
        .iter()
        .map(|a| {
            let mut dst = MaybeUninit::<RawBuf>::uninit();
            unsafe {
                crate::inline_array::raw_copy_buf(dst.as_mut_ptr(), a.as_raw_ptr());
                dst.assume_init()
            }
        })
        .collect();

    let mut dst_raw = MaybeUninit::<RawBuf>::uninit();
    unsafe {
        crate::inline_array::raw_stack(
            dst_raw.as_mut_ptr(),
            raw_copies.as_ptr(),
            raw_copies.len() as i32,
            axis,
        );
    }
    for mut rb in raw_copies {
        unsafe {
            crate::inline_array::raw_destroy(&mut rb);
        }
    }
    unsafe { crate::inline_array::from_raw_buf(dst_raw.assume_init()) }
}

pub fn expand_dims(a: &Array, axis: i32) -> Array {
    a.expand_dims(axis)
}
pub fn repeat_axis(a: Array, repeats: i32, axis: i32) -> Array {
    a.repeat(repeats, axis)
}
/// Stack arrays along a new axis 0 — equivalent to `mlx_rs::ops::stack`.
pub fn stack(arrays: &[Array]) -> Array {
    stack_axis(arrays, 0)
}

pub fn tri(n: i32, m: i32, k: i32, dtype: Dtype) -> Array {
    Array::tri(n, m, k, dtype.as_i32())
}

pub fn sigmoid(a: &Array) -> Array {
    a.sigmoid()
}

pub fn clip(a: &Array, lo: Option<&Array>, hi: Option<&Array>) -> Array {
    a.clip(lo, hi)
}

pub fn argsort_axis(a: &Array, axis: i32) -> Array {
    a.argsort(axis)
}
pub fn argpartition_axis(a: &Array, kth: i32, axis: i32) -> Array {
    a.argpartition(kth, axis)
}
pub fn cumsum(a: &Array, axis: i32) -> Array {
    a.cumsum(axis)
}
pub fn tril(a: &Array, k: i32) -> Array {
    a.tril(k)
}
pub fn argmax(a: &Array, axis: i32) -> Array {
    a.argmax(axis)
}
pub fn argmin(a: &Array, axis: i32) -> Array {
    a.argmin(axis)
}
pub fn zeros(shape: &[i32], dtype: Dtype) -> Array {
    Array::zeros(shape, dtype.as_i32())
}
pub fn ones(shape: &[i32], dtype: Dtype) -> Array {
    Array::ones(shape, dtype.as_i32())
}
pub fn full(shape: &[i32], val: f32, dtype: Dtype) -> Array {
    Array::full(shape, val, dtype.as_i32())
}
pub fn arange(n: i32, dtype: Dtype) -> Array {
    Array::arange(n, dtype.as_i32())
}
/// `arange(0, n, 1)` — integer range as int32.
pub fn arange_n(n: i32) -> Array {
    Array::arange(n, Dtype::Int32.as_i32())
}
/// `arange(start, stop, 1)` — integer range; equivalent to `mlx_rs::ops::arange::<i32,i32>(start, stop, 1)`.
pub fn arange_from(start: i32, stop: i32) -> Array {
    let n = (stop - start).max(0);
    let base = Array::arange(n, Dtype::Int32.as_i32());
    if start == 0 {
        base
    } else {
        base.add(&Array::from_i32(start))
    }
}
pub fn zeros_like(a: &Array) -> Array {
    a.zeros_like()
}
pub fn eye(n: i32, dtype: Dtype) -> Array {
    Array::eye(n, dtype.as_i32())
}
pub fn flatten(a: &Array, start: i32, end: i32) -> Array {
    a.flatten(start, end)
}
pub fn transpose(a: &Array) -> Array {
    a.t()
}
pub fn transpose_axes(a: &Array, axes: &[i32]) -> Array {
    a.transpose_axes(axes)
}
pub fn reshape(a: &Array, shape: &[i32]) -> Array {
    a.reshape(shape)
}
pub fn squeeze(a: &Array, axis: i32) -> Array {
    a.squeeze(axis)
}
pub fn sum_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
    a.sum_axis(axis, keepdims)
}
pub fn sum_axes(a: &Array, axes: &[i32], keepdims: bool) -> Array {
    a.sum_axes(axes, keepdims)
}
pub fn sum_all(a: &Array) -> Array {
    a.sum_all()
}
pub fn mean_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
    a.mean_axis(axis, keepdims)
}
pub fn mean_all(a: &Array) -> Array {
    a.mean_all()
}
pub fn max_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
    a.max_axis(axis, keepdims)
}
pub fn min_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
    a.min_axis(axis, keepdims)
}
pub fn logsumexp(a: &Array, axis: i32, keepdims: bool) -> Array {
    a.logsumexp(axis, keepdims)
}
pub fn exp(a: &Array) -> Array {
    a.exp()
}
pub fn log(a: &Array) -> Array {
    a.log()
}
pub fn sqrt(a: &Array) -> Array {
    a.sqrt()
}
pub fn abs(a: &Array) -> Array {
    a.abs_val()
}
pub fn square(a: &Array) -> Array {
    a.square()
}
pub fn pow(a: &Array, b: &Array) -> Array {
    a.pow(b)
}
pub fn where_fn(cond: &Array, a: &Array, b: &Array) -> Array {
    cond.where_cond(a, b)
}
/// `r#where` — alias for `where_fn` matching the mlx-rs `ops::r#where` name.
#[allow(non_snake_case)]
pub fn r#where(cond: &Array, a: &Array, b: &Array) -> Array {
    cond.where_cond(a, b)
}
pub fn equal(a: &Array, b: &Array) -> Array {
    a.equal(b)
}
pub fn not_equal(a: &Array, b: &Array) -> Array {
    a.not_equal(b)
}
pub fn greater(a: &Array, b: &Array) -> Array {
    a.greater(b)
}
pub fn less(a: &Array, b: &Array) -> Array {
    a.less(b)
}
pub fn greater_equal(a: &Array, b: &Array) -> Array {
    a.greater_equal(b)
}
pub fn less_equal(a: &Array, b: &Array) -> Array {
    a.less_equal(b)
}
pub fn stop_gradient(a: &Array) -> Array {
    a.stop_gradient()
}
pub fn take_axis(a: &Array, indices: &Array, axis: i32) -> Array {
    a.take_axis(indices, axis)
}
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Array {
    a.take_along_axis(indices, axis)
}
/// Pad every axis, like `mx.pad`. `pad_widths` is one `(before, after)` per
/// axis; `mode` defaults to [`PadMode::Constant`], and `fill_value` (default
/// 0) is used by that mode only.
pub fn pad(
    a: &Array,
    pad_widths: &[(i32, i32)],
    mode: Option<PadMode>,
    fill_value: Option<f32>,
) -> Array {
    let fill = fill_value.unwrap_or(0.0);
    let flat: Vec<i32> = pad_widths.iter().flat_map(|(b, e)| [*b, *e]).collect();
    a.pad(&flat, mode.unwrap_or_default(), fill)
}
/// Wrapper matching `mlx_rs::ops::arange::<i32, f32>` signature used in vocoder.
/// Produces a float32 arange from `start` to `stop` (exclusive), step 1.
pub fn arange_range(start: i32, stop: i32) -> Array {
    let n = (stop - start).max(0);
    // arange(n) gives [0..n); add start offset if needed
    let a = Array::arange(n, Dtype::Float32.as_i32());
    if start == 0 {
        a
    } else {
        let offset = Array::from_f32(start as f32);
        a.add(&offset)
    }
}
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Array {
    input.conv1d(weight, stride, padding, dilation, groups)
}
/// Hyperbolic tangent, `mx.tanh`. Keeps the input dtype.
pub fn tanh(a: &Array) -> Array {
    a.tanh()
}

// ── arithmetic helpers ────────────────────────────────────────────────────

/// Element-wise addition — alias for `a.add(b)`.
pub fn add(a: &Array, b: &Array) -> Array {
    a.add(b)
}
/// Element-wise subtraction.
pub fn subtract(a: &Array, b: &Array) -> Array {
    a.subtract(b)
}
/// Element-wise multiplication.
pub fn multiply(a: &Array, b: &Array) -> Array {
    a.multiply(b)
}
/// Element-wise division.
pub fn divide(a: &Array, b: &Array) -> Array {
    a.divide(b)
}
/// Negate: `-a`, `mx.negative`. Keeps the input dtype.
pub fn negative(a: &Array) -> Array {
    a.negative()
}

// ── trigonometry ─────────────────────────────────────────────────────────

pub fn sin(a: &Array) -> Array {
    a.sin()
}
pub fn cos(a: &Array) -> Array {
    a.cos()
}

// ── aliases and missing variants ──────────────────────────────────────────

/// Alias for `zeros` — `zeros_dtype(shape, dtype)` matches mlx-rs naming.
pub fn zeros_dtype(shape: &[i32], dtype: Dtype) -> Array {
    Array::zeros(shape, dtype.as_i32())
}
/// Alias: `argmax` with keepdims=false.
pub fn argmax_axis(a: &Array, axis: i32) -> Array {
    a.argmax(axis)
}
/// Alias: `argmin` with keepdims=false.
pub fn argmin_axis(a: &Array, axis: i32) -> Array {
    a.argmin(axis)
}
/// `which(cond, x, y)` — alias for `where_fn`.
pub fn which(cond: &Array, x: &Array, y: &Array) -> Array {
    cond.where_cond(x, y)
}

/// Tile the whole array `reps` times along each dimension, like `mx.tile` /
/// `np.tile`: `tile([1, 2], [2])` is `[1, 2, 1, 2]`.
///
/// `reps` lines up with the array's trailing axes; whichever is shorter is
/// padded with leading 1s.
pub fn tile(a: &Array, reps: &[i32]) -> Array {
    a.tile(reps)
}

/// Split array into `num_sections` equal pieces along `axis`, like
/// `mx.split(a, num_sections, axis)`.
///
/// The axis must divide evenly, as in MLX and `np.split`. Otherwise MLX's
/// error is on [`check_last_error`](crate::check_last_error) and the result
/// is empty.
pub fn split(a: &Array, num_sections: i32, axis: i32) -> Vec<Array> {
    a.split_sections(num_sections, axis)
}

/// Split array at given indices along `axis`.
///
/// Equivalent to `np.split(a, indices, axis=axis)` or `mx.split(a, indices, axis)`.
/// `indices` are the positions *before* which splits are made (i.e. [i0, i1] → 3 pieces:
/// `[:i0]`, `[i0:i1]`, `[i1:]`).
pub fn split_sections(a: &Array, indices: &[i32], axis: i32) -> Vec<Array> {
    a.split(indices, axis)
}

/// Scatter: create a new array where `a[indices]` = `updates` along `axis`.
///
/// Returns a new array; does not modify `a` in place.
/// Equivalent to `mlx_rs::ops::put_along_axis`.
pub fn put_along_axis(a: &Array, indices: &Array, updates: &Array, axis: i32) -> Array {
    a.put_along_axis_op(indices, updates, axis)
}

/// `async_eval` — schedule GPU evaluation of each array on the active stream
/// without blocking the caller. Used by the decode pipeline to launch the
/// forward for token N+1 before the host extracts token N.
///
/// Mirrors `mx.async_eval(...)` in Python MLX. Unlike the synchronous
/// `eval`, this does not wait for GPU completion — callers rely on a
/// downstream `.item()` / `.as_slice()` to drive the host sync.
pub fn async_eval<'a>(arrays: impl IntoIterator<Item = &'a Array>) {
    for arr in arrays {
        arr.async_eval_ref();
    }
}

/// `logsumexp_axis` — alias for `logsumexp(a, axis, false)`.
pub fn logsumexp_axis(a: &Array, axis: i32) -> Array {
    a.logsumexp(axis, false)
}

/// `logsumexp_axis_keepdims` — `logsumexp(a, axis, keepdims)`.
pub fn logsumexp_axis_keepdims(a: &Array, axis: i32, keepdims: bool) -> Array {
    a.logsumexp(axis, keepdims)
}

/// Reciprocal square root: `1/sqrt(x)`.
pub fn rsqrt(a: &Array) -> Array {
    a.rsqrt()
}

/// Quantize to MLX native FP8 (E4M3 format, stored as uint8).
pub fn to_fp8(x: &Array) -> Result<Array, super::Exception> {
    x.try_to_fp8()
        .map_err(|err| super::Exception::custom(err.to_string()))
}

/// Dequantize from MLX native FP8 (E4M3 uint8 payload) to the target dtype.
pub fn from_fp8(x: &Array, dtype: super::Dtype) -> Result<Array, super::Exception> {
    x.try_from_fp8(dtype.as_i32())
        .map_err(|err| super::Exception::custom(err.to_string()))
}

/// Evenly-spaced values from `start` to `stop` (inclusive).
pub fn linspace(start: f32, stop: f32, n: i32, dtype: Dtype) -> Array {
    Array::linspace(start, stop, n, dtype.as_i32())
}

/// Floor, `mx.floor`: largest integer ≤ x. Keeps the input dtype; exact for
/// every float, including values past the `i32` range, ±inf and NaN.
pub fn floor(a: &Array) -> Array {
    a.floor()
}

/// Ceil, `mx.ceil`: smallest integer ≥ x. Keeps the input dtype.
pub fn ceil(a: &Array) -> Array {
    a.ceil()
}

/// Round to the nearest integer, ties to even, like `mx.round` / `np.round`:
/// `0.5 → 0`, `1.5 → 2`, `2.5 → 2`. Keeps the input dtype.
pub fn round(a: &Array) -> Array {
    a.round()
}

/// Logical NOT to bool, `mx.logical_not`: true where `a` is zero.
pub fn logical_not(a: &Array) -> Array {
    a.logical_not()
}

/// Logical AND to bool, `mx.logical_and`.
pub fn logical_and(a: &Array, b: &Array) -> Array {
    a.logical_and(b)
}

/// Logical OR to bool, `mx.logical_or`.
pub fn logical_or(a: &Array, b: &Array) -> Array {
    a.logical_or(b)
}

/// Bool array, true where `a` is NaN, `mx.isnan`.
pub fn is_nan(a: &Array) -> Array {
    a.isnan()
}

/// Bool array, true where `a` is ±inf, `mx.isinf`.
pub fn is_inf(a: &Array) -> Array {
    a.isinf()
}

/// Logical-OR reduction to bool, `mx.any(a, axes, keepdims)`.
///
/// `axes: None` reduces every axis. The listed axes are reduced together
/// (order and sign don't matter), and `keep_dims` keeps each one as size 1.
pub fn any(a: &Array, axes: Option<&[i32]>, keep_dims: bool) -> Array {
    a.any(axes, keep_dims)
}

/// The value of a one-element bool array, evaluating it first.
pub fn item_bool(a: &Array) -> bool {
    let a_clone = a.clone();
    a_clone.eval();
    // Cast to f32 and check if value > 0.5
    let f = a_clone.as_dtype(Dtype::Float32.as_i32()).item_f32();
    f > 0.5
}

/// Select a single index along `axis`, removing that dimension.
/// Equivalent to `a[..., idx, ...]` in Python/mlx.
pub fn select_axis(a: &Array, idx: i32, axis: i32) -> Array {
    let ndim = a.ndim();
    let ax = if axis < 0 { ndim + axis } else { axis };
    let i = Array::from_i32_slice_shaped(&[idx], &[1]);
    let out = a.take_axis(&i, ax);
    out.squeeze(ax)
}

/// Slice the last axis from 0 to `end` (exclusive), all other axes full.
/// Equivalent to `a[..., :end]` in Python/mlx.
pub fn slice_last_to(a: &Array, end: i32) -> Array {
    let ndim = a.ndim() as usize;
    let shape = a.shape();
    let s: Vec<i32> = vec![0; ndim];
    let mut e: Vec<i32> = shape.to_vec();
    e[ndim - 1] = end;
    a.slice(&s, &e)
}

/// Slice the last axis from `start` to the end, all other axes full.
/// Equivalent to `a[..., start:]` in Python/mlx.
pub fn slice_last_from(a: &Array, start: i32) -> Array {
    let ndim = a.ndim() as usize;
    let shape = a.shape();
    let mut s: Vec<i32> = vec![0; ndim];
    let e: Vec<i32> = shape.to_vec();
    s[ndim - 1] = start;
    a.slice(&s, &e)
}

/// Slice a specific axis from `start` to `end`, all other axes full.
/// Equivalent to `a[..., start:end, ...]` at the given axis.
pub fn slice_axis(a: &Array, axis: i32, start: i32, end: i32) -> Array {
    let ndim = a.ndim();
    let ax = if axis < 0 {
        (ndim + axis) as usize
    } else {
        axis as usize
    };
    let shape = a.shape();
    let mut s: Vec<i32> = vec![0; ndim as usize];
    let mut e: Vec<i32> = shape.to_vec();
    s[ax] = start;
    e[ax] = end;
    a.slice(&s, &e)
}

/// Slice a specific axis from `start` to the end, all other axes full.
pub fn slice_axis_from(a: &Array, axis: i32, start: i32) -> Array {
    let ndim = a.ndim();
    let ax = if axis < 0 {
        (ndim + axis) as usize
    } else {
        axis as usize
    };
    let shape = a.shape();
    let mut s: Vec<i32> = vec![0; ndim as usize];
    let e: Vec<i32> = shape.to_vec();
    s[ax] = start;
    a.slice(&s, &e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_roundtrip_preserves_signed_float_values() {
        let x = Array::from_f32_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], &[2, 3]);
        let q = to_fp8(&x).expect("to_fp8");

        assert_eq!(q.dtype(), Dtype::Uint8);

        let mut y = from_fp8(&q, Dtype::Float32).expect("from_fp8");
        y.eval();
        let got = y.to_f32_vec(6).expect("to_f32_vec");

        assert!(got[0] < -0.75, "expected negative value, got {}", got[0]);
        assert!(got[1] < -0.25, "expected negative value, got {}", got[1]);
        assert!(
            got[2].abs() < 0.05,
            "expected near-zero value, got {}",
            got[2]
        );
        assert!(got[3] > 0.25, "expected positive value, got {}", got[3]);
        assert!(got[4] > 0.75, "expected positive value, got {}", got[4]);
        assert!(got[5] > 1.5, "expected positive value, got {}", got[5]);
    }
}
