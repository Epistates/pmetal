//! Trait-based dtype mapping for `InlineArray` casting and scalar extraction.
//!
//! - [`AsDtype`] maps Rust primitive types to MLX dtype IDs (drives `as_type::<T>()`).
//! - [`ArrayElement`] drives the generic `from_slice<T>()` entry point.
//! - [`BridgeScalar`] drives `item<T>()` (only `f32`, `u32`, `i32` are bridge-visible).

use super::InlineArray;

// ── AsDtype: sealed trait for as_type<T>() ──────────────────────────────

/// Sealed trait mapping Rust primitive types to MLX dtype IDs.
///
/// Used by [`InlineArray::as_type::<T>()`] to cast arrays by Rust type.
pub trait AsDtype {
    const DTYPE_ID: i32;
}

impl AsDtype for f32 {
    const DTYPE_ID: i32 = 10;
} // Float32
impl AsDtype for f16 {
    const DTYPE_ID: i32 = 9;
} // Float16 (using half::f16 or similar)
impl AsDtype for u8 {
    const DTYPE_ID: i32 = 1;
} // Uint8
impl AsDtype for u16 {
    const DTYPE_ID: i32 = 2;
} // Uint16
impl AsDtype for u32 {
    const DTYPE_ID: i32 = 3;
} // Uint32
impl AsDtype for u64 {
    const DTYPE_ID: i32 = 4;
} // Uint64
impl AsDtype for i8 {
    const DTYPE_ID: i32 = 5;
} // Int8
impl AsDtype for i16 {
    const DTYPE_ID: i32 = 6;
} // Int16
impl AsDtype for i32 {
    const DTYPE_ID: i32 = 7;
} // Int32
impl AsDtype for i64 {
    const DTYPE_ID: i32 = 8;
} // Int64
impl AsDtype for bool {
    const DTYPE_ID: i32 = 0;
} // Bool

/// Half-precision float marker type for `as_type::<f16>()`.
/// Use `half::f16` from the `half` crate, or this zero-sized stub.
#[allow(non_camel_case_types)]
pub struct f16;

/// Bfloat16 marker type for `as_type::<bf16>()`.
#[allow(non_camel_case_types)]
pub struct bf16;

impl AsDtype for bf16 {
    const DTYPE_ID: i32 = 11;
} // Bfloat16

// ── ArrayElement: trait for from_slice<T>() ──────────────────────────────

/// Trait for element types supported by [`InlineArray::from_slice`].
///
/// Implemented for `f32`, `i32`, `u32`, and `i64`.
pub trait ArrayElement {
    fn into_array(data: &[Self], shape: &[i32]) -> InlineArray
    where
        Self: Sized;
}

impl ArrayElement for f32 {
    fn into_array(data: &[f32], shape: &[i32]) -> InlineArray {
        InlineArray::from_f32_slice(data, shape)
    }
}

impl ArrayElement for i32 {
    fn into_array(data: &[i32], shape: &[i32]) -> InlineArray {
        InlineArray::from_i32_slice_shaped(data, shape)
    }
}

impl ArrayElement for u32 {
    fn into_array(data: &[u32], shape: &[i32]) -> InlineArray {
        InlineArray::from_u32_slice(data, shape)
    }
}

impl ArrayElement for i64 {
    fn into_array(data: &[i64], shape: &[i32]) -> InlineArray {
        let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
        InlineArray::from_i32_slice_shaped(&i32_data, shape)
    }
}

impl ArrayElement for usize {
    fn into_array(data: &[usize], shape: &[i32]) -> InlineArray {
        let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
        InlineArray::from_i32_slice_shaped(&i32_data, shape)
    }
}

// ── BridgeScalar: sealed trait for item<T>() ─────────────────────────────

/// Sealed trait for extracting a scalar value from an [`InlineArray`].
///
/// Only `f32`, `u32`, and `i32` are supported — they are the types the bridge
/// FFI exposes via `mlx_inline_item_f32` / `mlx_inline_item_u32`. The trait
/// reads through `&InlineArray` because the underlying readback is logically
/// const (eval is interior-mutable on the MLX side).
pub trait BridgeScalar: private::Sealed {
    fn extract(arr: &InlineArray) -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

impl BridgeScalar for f32 {
    fn extract(arr: &InlineArray) -> f32 {
        arr.item_f32()
    }
}

impl BridgeScalar for u32 {
    fn extract(arr: &InlineArray) -> u32 {
        arr.item_u32()
    }
}

impl BridgeScalar for i32 {
    fn extract(arr: &InlineArray) -> i32 {
        arr.item_u32() as i32
    }
}
