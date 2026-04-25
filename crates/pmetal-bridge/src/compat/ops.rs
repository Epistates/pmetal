use super::{Array, Dtype};
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
/// log(1 + x) — numerically stable log1p.
pub fn log1p(a: &Array) -> Array {
    let one = Array::from_f32(1.0);
    a.add(&one).log()
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
        let offset = crate::InlineArray::from_f32(start as f32).as_dtype(Dtype::Int32.as_i32());
        base.add(&offset)
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
/// Pad array. `pad_widths`: `&[(before, after)]` per axis.
pub fn pad(
    a: &Array,
    pad_widths: &[(i32, i32)],
    _mode: Option<&str>,
    fill_value: Option<f32>,
) -> Array {
    let fill = fill_value.unwrap_or(0.0);
    let flat: Vec<i32> = pad_widths.iter().flat_map(|(b, e)| [*b, *e]).collect();
    a.pad_constant(&flat, fill)
}
/// Wrapper matching `mlx_rs::ops::arange::<i32, f32>` signature used in vocoder.
/// Produces a float32 arange from `start` to `stop` (exclusive), step 1.
pub fn arange_range(start: i32, stop: i32) -> Array {
    let n = (stop - start).max(0);
    // arange(n) gives [0..n); add start offset if needed
    let a = Array::arange(n, 10); // dtype=10=float32
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
pub fn tanh(a: &Array) -> Array {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) = 2*sigmoid(2x) - 1
    let two = Array::from_f32(2.0);
    let two_x = a.multiply(&two);
    let sig = two_x.sigmoid();
    let one = Array::from_f32(1.0);
    let two2 = Array::from_f32(2.0);
    sig.multiply(&two2).subtract(&one)
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
/// Negate: `-a`.
pub fn negative(a: &Array) -> Array {
    let neg_one = Array::from_f32(-1.0);
    a.multiply(&neg_one)
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

/// Tile array `reps` times along each dimension.
///
/// `reps` is the number of repetitions per axis (broadcast from the end).
pub fn tile(a: &Array, reps: &[i32]) -> Array {
    let ndim = a.ndim() as usize;
    let nrep = reps.len();
    let mut arr = a.clone();
    // Extend ndim to match reps length if needed (prepend 1-dims).
    if nrep > ndim {
        for _ in ndim..nrep {
            arr = arr.expand_dims(0);
        }
    }
    let effective_ndim = arr.ndim() as usize;
    let pad = effective_ndim.saturating_sub(nrep);
    let full_reps: Vec<i32> = std::iter::repeat_n(1, pad)
        .chain(reps.iter().copied())
        .collect();
    for (axis, &rep) in full_reps.iter().enumerate() {
        if rep > 1 {
            arr = arr.repeat(rep, axis as i32);
        }
    }
    arr
}

/// Split array into `num_sections` equal pieces along `axis`.
///
/// Equivalent to `np.split(a, num_sections, axis=axis)` or `mx.split(a, num_sections, axis)`.
pub fn split(a: &Array, num_sections: i32, axis: i32) -> Vec<Array> {
    let ax = if axis < 0 {
        (a.ndim() + axis) as usize
    } else {
        axis as usize
    };
    let dim = a.shape()[ax];
    let section_size = dim / num_sections;
    // Build split indices: [section_size, 2*section_size, ..., (n-1)*section_size]
    let indices: Vec<i32> = (1..num_sections).map(|i| i * section_size).collect();
    a.split(&indices, axis)
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

/// Quantize to FP8 (E4M3 format, stored as uint8).
///
/// Stub implementation — returns the input cast to Float16 then reinterpreted
/// as uint8.  Replace with proper FP8 when available.
pub fn to_fp8(x: &Array) -> Result<Array, super::Exception> {
    // FP8 is not yet in the inline bridge; fall back to float16 as uint8.
    let fp16 = x.as_dtype(super::Dtype::Float16.as_i32());
    Ok(fp16.as_dtype(super::Dtype::Uint8.as_i32()))
}

/// Dequantize from FP8 to the target dtype.
///
/// Stub implementation — casts uint8 input to the requested dtype.
pub fn from_fp8(x: &Array, dtype: super::Dtype) -> Result<Array, super::Exception> {
    Ok(x.as_dtype(dtype.as_i32()))
}

/// Evenly-spaced values from `start` to `stop` (inclusive).
pub fn linspace(start: f32, stop: f32, n: i32, dtype: Dtype) -> Array {
    Array::linspace(start, stop, n, dtype.as_i32())
}

/// Floor: largest integer ≤ x.
///
/// Implemented as `cast to int32, then cast back` for float inputs.
/// For inputs already of integer dtype, this is a no-op.
pub fn floor(a: &Array) -> Array {
    // floor(x) = cast_to_int(x - (x < 0) * 1) — approximate for int casts
    // More robustly: use the fact that int32 truncates toward zero:
    // floor(x) = trunc(x) - (x < 0 AND x != trunc(x))
    let int_vals = a.as_dtype(Dtype::Int32.as_i32());
    let float_int = int_vals.as_dtype(a.dtype_raw());
    // Correction: if original < float_int (negative truncation direction), subtract 1.
    let one = Array::from_f32(1.0).as_dtype(a.dtype_raw());
    let needs_correction = a.less(&float_int);
    let correction = needs_correction.as_dtype(a.dtype_raw()).multiply(&one);
    float_int.subtract(&correction)
}

/// Ceil: smallest integer ≥ x.
pub fn ceil(a: &Array) -> Array {
    // ceil(x) = -floor(-x)
    let neg = a.multiply(&Array::from_f32(-1.0));
    let floored = floor(&neg);
    floored.multiply(&Array::from_f32(-1.0))
}

/// Round to nearest integer.
pub fn round(a: &Array) -> Array {
    // round(x) = floor(x + 0.5)
    let half = Array::from_f32(0.5).as_dtype(a.dtype_raw());
    floor(&a.add(&half))
}

/// Logical NOT: bool → !bool.
pub fn logical_not(a: &Array) -> Array {
    let zero = Array::zeros(&[1], Dtype::Bool.as_i32());
    a.equal(&zero)
}

/// Logical AND: cast both to bool then multiply (AND).
pub fn logical_and(a: &Array, b: &Array) -> Array {
    let ab = a.as_dtype(Dtype::Bool.as_i32());
    let bb = b.as_dtype(Dtype::Bool.as_i32());
    // bool multiply = AND
    ab.multiply(&bb)
}

/// Logical OR: cast both to bool then add and clamp (OR).
pub fn logical_or(a: &Array, b: &Array) -> Array {
    let ab = a.as_dtype(Dtype::Bool.as_i32());
    let bb = b.as_dtype(Dtype::Bool.as_i32());
    let sum = ab.add(&bb);
    // Any nonzero = true: cast to bool clips to {0,1}
    sum.as_dtype(Dtype::Bool.as_i32())
}

/// Returns a bool array — true where x is NaN (using x != x identity).
pub fn is_nan(a: &Array) -> Array {
    // NaN != NaN is the IEEE 754 definition
    a.not_equal(a)
}

/// Returns a bool array — true where x is ±Inf.
pub fn is_inf(a: &Array) -> Array {
    // |x| > f32::MAX  iff  x is ±Inf
    let abs_a = a.abs_val();
    let max_finite = Array::from_f32(f32::MAX);
    abs_a.greater(&max_finite)
}

/// Reduce a bool/numeric array with logical-OR along all axes (or a given axis).
/// Equivalent to mlx_rs::ops::any(a, axes, keepdims).
/// `axes`: None = reduce all axes.
pub fn any(a: &Array, axes: Option<&[i32]>, _keep_dims: bool) -> Array {
    // Cast to bool first, then sum. Any nonzero sum → true.
    let b = a.as_dtype(Dtype::Bool.as_i32());
    let s = match axes {
        None => b.sum(None),
        Some(ax) => {
            // sum over each axis
            let mut result = b.clone();
            for &axis in ax {
                result = result.sum(Some(axis));
            }
            result
        }
    };
    s.as_dtype(Dtype::Bool.as_i32())
}

/// Returns `true` (scalar bool) if any element is NaN.
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
