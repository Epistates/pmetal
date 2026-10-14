//! Safe entry points of the runtime must not let a caller's arguments reach
//! the C++ side as undefined behaviour or as a process abort.
//!
//! Each case here used to read memory it did not own, return another dtype's
//! bytes as a value, or let a C++ exception unwind into Rust. Expected values
//! are written out by hand.

use pmetal_bridge::check_unobserved_error;
use pmetal_bridge::compat::Dtype;
use pmetal_bridge::InlineArray;

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
