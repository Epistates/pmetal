//! Safe entry points of the runtime must not let a caller's arguments reach
//! the C++ side as undefined behaviour or as a process abort.
//!
//! Each case here used to read memory it did not own, return another dtype's
//! bytes as a value, or let a C++ exception unwind into Rust. Expected values
//! are written out by hand.

use pmetal_bridge::InlineArray;
use pmetal_bridge::check_unobserved_error;
use pmetal_bridge::compat::Dtype;

fn scalar(value: f32, dtype: Dtype) -> InlineArray {
    InlineArray::from_f32(value).as_dtype(dtype.as_i32())
}

/// `item_f32` handed back the first four bytes of the buffer, so a bf16 `3.0`
/// came out as 2e-39 (its two bytes plus two bytes of whatever followed) and
/// an int32 `7` as 1e-44. A loss logged from a bf16 model was garbage.
#[test]
fn item_returns_the_value_whatever_the_dtype() {
    for dtype in [
        Dtype::Bfloat16,
        Dtype::Float16,
        Dtype::Float32,
        Dtype::Int8,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Uint8,
        Dtype::Uint32,
    ] {
        let x = scalar(3.0, dtype);
        assert_eq!(x.item_f32(), 3.0, "item_f32 of {dtype:?}");
        assert_eq!(x.item::<i32>(), 3, "item::<i32> of {dtype:?}");
        assert_eq!(x.item_u32(), 3, "item_u32 of {dtype:?}");
    }
    check_unobserved_error().expect("bridge error");
}

#[test]
fn item_keeps_the_sign_and_the_fraction() {
    assert_eq!(scalar(-5.5, Dtype::Bfloat16).item_f32(), -5.5);
    assert_eq!(scalar(-2.25, Dtype::Float16).item_f32(), -2.25);
    assert_eq!(scalar(-5.0, Dtype::Int32).item::<i32>(), -5);
    assert_eq!(scalar(-5.0, Dtype::Int64).item::<i32>(), -5);
    assert_eq!(scalar(-5.0, Dtype::Int8).item_f32(), -5.0);
    // An f32 value read as an integer is converted (truncated), not
    // reinterpreted: 2.5f32 is 0x40200000.
    assert_eq!(scalar(2.5, Dtype::Float32).item::<i32>(), 2);
    check_unobserved_error().expect("bridge error");
}

/// A 0-d array's shape is an empty slice, and its `dim` is an error, not a
/// read past the end of the shape.
#[test]
fn shape_of_a_scalar_is_empty() {
    let x = InlineArray::from_f32(1.0);
    assert_eq!(x.shape(), &[] as &[i32]);
    assert_eq!(x.ndim(), 0);
}

/// `mlx_inline_dim` indexed the shape with `.at()` outside any try block, so
/// an out-of-range axis threw a C++ exception through the `extern "C"`
/// frame and aborted the process.
#[test]
#[should_panic(expected = "axis 3 is out of range")]
fn dim_out_of_range_panics_like_indexing() {
    let x = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
    let _ = x.dim(3);
}

#[test]
fn dim_accepts_negative_axes() {
    let x = InlineArray::zeros(&[2, 3, 4], Dtype::Float32.as_i32());
    assert_eq!(x.dim(0), 2);
    assert_eq!(x.dim(-1), 4);
    assert_eq!(x.dim(-3), 2);
}

/// The slice constructors copy `product(shape)` elements from the pointer. A
/// slice shorter than that was an over-read of the caller's allocation.
#[test]
#[should_panic(expected = "needs 4 elements")]
fn from_f32_slice_rejects_a_short_slice() {
    let _ = InlineArray::from_f32_slice(&[1.0, 2.0], &[4]);
}

#[test]
#[should_panic(expected = "needs 6 elements")]
fn from_u32_slice_rejects_a_short_slice() {
    let _ = InlineArray::from_u32_slice(&[1, 2], &[2, 3]);
}

#[test]
#[should_panic(expected = "needs 6 elements")]
fn from_u8_slice_rejects_a_short_slice() {
    let _ = InlineArray::from_u8_slice(&[1, 2], &[2, 3]);
}

#[test]
#[should_panic(expected = "needs 3 elements")]
fn from_u16_bits_slice_rejects_a_short_slice() {
    let _ = InlineArray::from_u16_bits_slice(&[0, 0], &[3], Dtype::Bfloat16.as_i32());
}

#[test]
#[should_panic(expected = "needs 8 elements")]
fn from_i32_slice_shaped_rejects_a_short_slice() {
    let _ = InlineArray::from_i32_slice_shaped(&[1, 2, 3, 4], &[2, 4]);
}

#[test]
#[should_panic(expected = "negative dimension")]
fn a_negative_dimension_is_rejected() {
    let _ = InlineArray::from_f32_slice(&[1.0, 2.0], &[-1, 2]);
}

/// A longer slice than the shape needs is legal: the array takes the prefix.
#[test]
fn a_longer_slice_is_read_up_to_the_shape() {
    let mut x = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2]);
    assert_eq!(x.to_f32_vec(2).unwrap(), vec![1.0, 2.0]);
    check_unobserved_error().expect("bridge error");
}

/// `slice` read `start.len()` entries from `stop`'s pointer.
#[test]
#[should_panic(expected = "start and stop")]
fn slice_rejects_mismatched_bounds() {
    let x = InlineArray::zeros(&[4, 4], Dtype::Float32.as_i32());
    let _ = x.slice(&[0, 0], &[2]);
}

#[test]
#[should_panic(expected = "start and stop")]
fn slice_set_rejects_mismatched_bounds() {
    let x = InlineArray::zeros(&[4, 4], Dtype::Float32.as_i32());
    let v = InlineArray::zeros(&[2, 2], Dtype::Float32.as_i32());
    let _ = x.slice_set(&v, &[0, 0], &[2]);
}

fn ints(values: &[i32], dtype: Dtype) -> InlineArray {
    InlineArray::from_i32_slice(values).as_dtype(dtype.as_i32())
}

/// `scalar_like` cast the value to the peer's dtype first, so `x * 0.5` on an
/// int32 array multiplied by 0 (and `x + 0.5` added 0), silently.
#[test]
fn a_fractional_scalar_promotes_an_integer_array_instead_of_truncating() {
    let x = ints(&[1, 2, 3], Dtype::Int32);
    let y = x.mul_scalar(0.5);
    assert_eq!(y.dtype(), Dtype::Float32);
    assert_eq!(y.as_slice::<f32>(), &[0.5, 1.0, 1.5]);

    let z = x.add_scalar(0.25);
    assert_eq!(z.dtype(), Dtype::Float32);
    assert_eq!(z.as_slice::<f32>(), &[1.25, 2.25, 3.25]);
    check_unobserved_error().expect("bridge error");
}

/// A whole number the integer dtype holds keeps the array's dtype, as a
/// Python int does under NumPy's rules.
#[test]
fn a_whole_scalar_keeps_an_integer_arrays_dtype() {
    let x = ints(&[1, 2, 3], Dtype::Int32);
    let y = x.add_scalar(10.0);
    assert_eq!(y.dtype(), Dtype::Int32);
    assert_eq!(y.as_slice::<i32>(), &[11, 12, 13]);

    let u = ints(&[1, 2, 3], Dtype::Uint8);
    assert_eq!(u.mul_scalar(2.0).dtype(), Dtype::Uint8);
    // 300 does not fit a u8, so the array promotes rather than wrapping 300.
    let wide = u.add_scalar(300.0);
    assert_eq!(wide.dtype(), Dtype::Float32);
    assert_eq!(wide.as_slice::<f32>(), &[301.0, 302.0, 303.0]);
    // Negative numbers don't fit unsigned dtypes either.
    assert_eq!(u.add_scalar(-1.0).dtype(), Dtype::Float32);
    check_unobserved_error().expect("bridge error");
}

#[test]
fn a_float_peer_still_gets_a_scalar_in_its_own_dtype() {
    for dtype in [Dtype::Bfloat16, Dtype::Float16, Dtype::Float32] {
        let x = scalar(2.0, dtype);
        assert_eq!(x.mul_scalar(0.5).dtype(), dtype, "{dtype:?}");
        assert_eq!(InlineArray::scalar_like(0.5, &x).dtype(), dtype);
    }
}

#[test]
fn swap_axes_takes_negative_axes() {
    let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let x = InlineArray::from_f32_slice(&data, &[2, 3]);
    let t = x.swap_axes(-1, 0);
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.as_slice::<f32>(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    check_unobserved_error().expect("bridge error");
}

/// An out-of-range axis made `perm.swap` panic with an index error.
#[test]
fn swap_axes_out_of_range_is_a_bridge_error() {
    let x = InlineArray::zeros(&[2, 3], Dtype::Float32.as_i32());
    let _ = pmetal_bridge::expect_bridge_errors(|| x.swap_axes(0, 7));
    let err = pmetal_bridge::check_last_error().expect_err("axis 7 of a 2-d array");
    assert!(err.to_string().contains("swapaxes"), "{err}");
}

#[test]
fn split_sections_splits_evenly() {
    let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let x = InlineArray::from_f32_slice(&data, &[6]);
    let parts = x.split_sections(3, 0);
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[1].as_slice::<f32>(), &[2.0, 3.0]);
    check_unobserved_error().expect("bridge error");
}

/// `split_sections` sized its output buffer from the caller's section count:
/// zero indexed an empty buffer and panicked, a negative one wrapped to a
/// huge allocation, and a large one asked for gigabytes. A bad count is an
/// MLX error and an empty result.
#[test]
fn split_sections_with_a_bad_count_is_an_error_not_a_panic_or_an_allocation() {
    let x = InlineArray::from_f32_slice(&[0.0; 6], &[6]);
    for sections in [0, -3, 4, 1_000_000_000] {
        let parts = pmetal_bridge::expect_bridge_errors(|| x.split_sections(sections, 0));
        assert!(parts.is_empty(), "{sections} sections gave {}", parts.len());
        assert!(
            pmetal_bridge::check_last_error().is_err(),
            "{sections} sections raised no error"
        );
    }
}

/// The constructors and casts take the typed `Dtype` as well as the raw code.
#[test]
fn a_dtype_argument_can_be_typed_or_a_raw_code() {
    let typed = InlineArray::zeros(&[2, 2], Dtype::Bfloat16);
    assert_eq!(typed.dtype(), Dtype::Bfloat16);
    assert_eq!(InlineArray::ones(&[2], 11).dtype(), Dtype::Bfloat16);
    let x = InlineArray::from_f32_slice(&[1.5, 2.5], &[2]);
    assert_eq!(x.as_dtype(Dtype::Int32).dtype(), Dtype::Int32);
    assert_eq!(x.as_dtype(Dtype::Float16.as_i32()).dtype(), Dtype::Float16);
    assert_eq!(InlineArray::arange(4, Dtype::Int32).dtype(), Dtype::Int32);
    assert_eq!(
        InlineArray::full(&[2], 3.0, Dtype::Float16).dtype(),
        Dtype::Float16
    );
    check_unobserved_error().expect("bridge error");
}

/// An unknown dtype code came back as float32.
#[test]
fn an_unknown_dtype_code_is_an_error_not_float32() {
    let made = pmetal_bridge::expect_bridge_errors(|| InlineArray::zeros(&[2], 99));
    let err = pmetal_bridge::check_last_error().expect_err("dtype code 99");
    assert!(err.to_string().contains("unknown dtype code 99"), "{err}");
    // What comes back is the bridge's placeholder, not a [2] float32 array.
    assert_ne!(made.shape(), &[2]);
}
