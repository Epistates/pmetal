//! Public dtype codes for bridge FFI arguments.
//!
//! The raw `i32` dtype codes expected by `mlx_inline_*` functions are the same
//! integers MLX uses internally. Writing them inline as magic numbers is a
//! footgun — `0` is `Bool`, not `Float32`, and a silent off-by-ten bug yields
//! tensors that every downstream op rejects or (worse) silently corrupts.
//!
//! Downstream crates should import one of these named constants instead of
//! hard-coding literals:
//!
//! ```
//! use pmetal_runtime::dtype;
//! let code = dtype::F32;          // 10, not 0
//! ```
//!
//! The high-level [`crate::compat::Dtype`] enum (re-exported here as
//! [`Dtype`]) is the preferred API — these raw constants exist for call
//! sites that already pass `i32` through FFI and don't want to carry an
//! enum through the call chain.
//!
//! Values match `mlx::core::Dtype` in the MLX C++ headers and are identical
//! across platforms. Keep this module in sync with
//! `crates/pmetal-runtime/cpp/bridge_internal.h::dtype_from_int`.

// Re-export the high-level enum — most callers should use this.
pub use crate::compat::Dtype;

/// `Bool` — scalar boolean elements.
pub const BOOL: i32 = 0;
/// `Uint8` — 8-bit unsigned integer.
pub const U8: i32 = 1;
/// `Uint16` — 16-bit unsigned integer.
pub const U16: i32 = 2;
/// `Uint32` — 32-bit unsigned integer.
pub const U32: i32 = 3;
/// `Uint64` — 64-bit unsigned integer.
pub const U64: i32 = 4;
/// `Int8` — 8-bit signed integer.
pub const I8: i32 = 5;
/// `Int16` — 16-bit signed integer.
pub const I16: i32 = 6;
/// `Int32` — 32-bit signed integer.
pub const I32: i32 = 7;
/// `Int64` — 64-bit signed integer.
pub const I64: i32 = 8;
/// `Float16` — IEEE 754 half-precision.
pub const F16: i32 = 9;
/// `Float32` — IEEE 754 single-precision.
pub const F32: i32 = 10;
/// `Bfloat16` — brain floating-point 16.
pub const BF16: i32 = 11;
/// `Complex64` — complex, fp32 real + fp32 imaginary.
pub const COMPLEX64: i32 = 12;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_dtype_enum() {
        assert_eq!(BOOL, Dtype::Bool.as_i32());
        assert_eq!(U8, Dtype::Uint8.as_i32());
        assert_eq!(U16, Dtype::Uint16.as_i32());
        assert_eq!(U32, Dtype::Uint32.as_i32());
        assert_eq!(U64, Dtype::Uint64.as_i32());
        assert_eq!(I8, Dtype::Int8.as_i32());
        assert_eq!(I16, Dtype::Int16.as_i32());
        assert_eq!(I32, Dtype::Int32.as_i32());
        assert_eq!(I64, Dtype::Int64.as_i32());
        assert_eq!(F16, Dtype::Float16.as_i32());
        assert_eq!(F32, Dtype::Float32.as_i32());
        assert_eq!(BF16, Dtype::Bfloat16.as_i32());
        assert_eq!(COMPLEX64, Dtype::Complex64.as_i32());
    }
}
