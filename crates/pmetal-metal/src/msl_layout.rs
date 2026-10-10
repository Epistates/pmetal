//! Rust structs that a Metal shader reads byte for byte.
//!
//! A params struct goes to the GPU as raw bytes (`setBytes`, or a buffer
//! made from its bytes), and the shader reads those bytes through its own MSL
//! struct. Nothing at run time checks that the two agree: a missing field, a
//! reordered one or a different width just shifts every later field, and the
//! kernel computes with the wrong numbers. Every such struct is `#[repr(C)]`
//! and is declared with [`msl_layout!`], which pins its size and each field's
//! offset to the MSL struct's at compile time.
//!
//! MSL lays a struct out like C: each scalar at its natural alignment
//! (`bool` 1 byte, `half` 2, `uint`/`int`/`float` 4), the struct padded to
//! its largest member's alignment
//! (Metal Shading Language Specification, table 2.1, "Metal scalar data
//! types"). Code that sends a params struct names the concrete type, never a
//! generic `T`, so the struct it sends is always one checked here.

/// Assert, at compile time, a `#[repr(C)]` struct's size and the offset of
/// each listed field, as the MSL struct of the same role lays them out.
/// List every field, padding included.
macro_rules! msl_layout {
    ($ty:ty, size = $size:expr, { $($field:ident : $offset:expr),+ $(,)? }) => {
        const _: () = {
            assert!(
                ::core::mem::size_of::<$ty>() == $size,
                concat!(stringify!($ty), " is not the size of its MSL struct"),
            );
            $(
                assert!(
                    ::core::mem::offset_of!($ty, $field) == $offset,
                    concat!(
                        stringify!($ty), ".", stringify!($field),
                        " is not at its MSL struct's offset",
                    ),
                );
            )+
        };
    };
}
pub(crate) use msl_layout;
