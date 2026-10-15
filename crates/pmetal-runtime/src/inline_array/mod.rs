//! Zero-allocation MLX array — stores `mlx::core::array` inline on the Rust stack.
//!
//! This eliminates ALL per-op heap allocation, matching Python/nanobind's direct
//! C++ binding performance. Each op is a single `extern "C"` call with placement-new
//! into a caller-provided buffer.

use std::mem::MaybeUninit;

mod dtype;
pub use dtype::{ArrayElement, AsDtype, IntoDtype, RuntimeScalar, bf16, f16};

mod diagnostics;
pub use diagnostics::{
    DeviceInfo, GenerationStream, clear_cache, device_info, disable_compile, enable_compile,
    eval_and_detach_many, get_active_memory, get_cache_memory, get_max_recommended_size,
    get_memory_limit, get_peak_memory, graph_desc_count, graph_dump, graph_node_count, init_device,
    metal_start_capture, metal_stop_capture, metallib_path, new_generation_stream,
    reset_default_stream, reset_peak_memory, set_cache_limit, set_generation_stream,
    set_memory_limit, set_metallib_path, set_wired_limit, set_wired_limit_max, synchronize,
    validate_metallib, verify_buffer_layout,
};

mod safetensors;
pub use safetensors::{load_safetensors_shard, random_seed, try_load_safetensors_shard};

mod autograd;
pub use autograd::{checkpoint_apply, checkpoint_apply_unchecked, value_and_grad};

mod cce;
pub use cce::{
    FusedCceOutput, FusedCceParams, LogitRowStats, cut_cross_entropy_fused, logit_row_stats,
};

mod compiled;
mod eval;
mod factory;
mod fast;
pub use fast::PadMode;
mod gather;
mod gdn_methods;
mod interop;
mod ops;
mod quantized;
pub use quantized::QuantizedMode;
mod reductions;
mod shape_ops;

/// Size of `mlx::core::array` in bytes. Must match MLX_ARRAY_SIZE in bridge.h.
const ARRAY_BUF_SIZE: usize = 128;
/// Alignment of `mlx::core::array`.
const ARRAY_BUF_ALIGN: usize = 8;

/// Raw inline array buffer — matches `mlx_inline_array` in C.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub(crate) struct RawBuf {
    pub(crate) buf: [u8; ARRAY_BUF_SIZE],
}

/// One projection weight as the compiled blocks take it — matches
/// `mlx_inline_qweight` in C.
///
/// A null `scales` means dense, and a null `biases` means a mode with no bias
/// term. Build one with [`crate::native_weight::LayerWeight::as_raw`]; the
/// borrow keeps the arrays it points at alive for the call.
#[repr(C)]
pub(crate) struct QWeightRaw<'a> {
    pub(crate) weight: *const RawBuf,
    pub(crate) scales: *const RawBuf,
    pub(crate) biases: *const RawBuf,
    /// A per-tensor factor applied to the packed product. See `tensor_scale`
    /// on `LayerWeight::Quantized`.
    pub(crate) tensor_scale: *const RawBuf,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
    pub(crate) mode: i32,
    pub(crate) _owner: std::marker::PhantomData<&'a InlineArray>,
}

mod ffi;
use ffi::*;

// ── InlineArray ───────────────────────────────────────────────────────────

/// Stack-allocated MLX array. Zero heap allocation per op.
///
/// `repr(transparent)` over the raw buffer: code that hands C++ a run of
/// arrays (a `Vec<InlineArray>` as a `*const RawBuf`) relies on the two having
/// one layout, which an unannotated struct does not promise.
#[repr(transparent)]
pub struct InlineArray {
    pub(crate) raw: RawBuf,
}

const _: () = assert!(
    std::mem::size_of::<InlineArray>() == std::mem::size_of::<RawBuf>()
        && std::mem::align_of::<InlineArray>() == std::mem::align_of::<RawBuf>()
);

#[derive(Clone, Debug)]
pub struct EvalToken {
    prior: crate::error::RuntimeResult<()>,
}

impl Default for EvalToken {
    fn default() -> Self {
        Self { prior: Ok(()) }
    }
}

impl EvalToken {
    #[inline]
    pub(crate) fn new(prior: crate::error::RuntimeResult<()>) -> Self {
        Self { prior }
    }

    #[inline]
    pub fn into_result(self) -> crate::error::RuntimeResult<()> {
        let after = crate::error::check_last_error();
        match self.prior {
            Ok(()) => after,
            Err(err) => {
                let _ = after;
                // Reporting it observes it: it must not come out of the
                // unobserved channel a second time.
                crate::error::clear_last_error();
                Err(err)
            }
        }
    }

    #[inline]
    pub fn unwrap(self) {
        self.into_result().unwrap();
    }

    #[inline]
    pub fn expect(self, msg: &str) {
        self.into_result().expect(msg);
    }
}

impl Drop for InlineArray {
    #[inline]
    fn drop(&mut self) {
        unsafe { mlx_inline_destroy(&mut self.raw) };
    }
}

impl Clone for InlineArray {
    fn clone(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_init_copy(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }
}

// SAFETY: an `mlx::core::array` is a handle to a reference-counted
// description, so cloning, dropping and reading the metadata of a handle from
// any thread is sound.
//
// What this does NOT promise is that two threads may *evaluate* the same
// unevaluated array at once: MLX does not lock a graph while it is being
// evaluated, and `eval(&self)` takes a shared reference. An array that more
// than one thread reads (shared weights, say) should be evaluated on one
// thread before it is shared, as model loading does.
unsafe impl Send for InlineArray {}
unsafe impl Sync for InlineArray {}

impl std::fmt::Debug for InlineArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InlineArray(ndim={}, shape={:?})",
            self.ndim(),
            self.shape()
        )
    }
}

macro_rules! binop {
    ($name:ident, $cfn:ident) => {
        #[inline]
        pub fn $name(&self, other: &Self) -> Self {
            let mut dst = ::std::mem::MaybeUninit::<$crate::inline_array::RawBuf>::uninit();
            unsafe {
                $cfn(dst.as_mut_ptr(), &self.raw, &other.raw);
                Self {
                    raw: dst.assume_init(),
                }
            }
        }
    };
}
pub(super) use binop;

macro_rules! unop {
    ($name:ident, $cfn:ident) => {
        #[inline]
        pub fn $name(&self) -> Self {
            let mut dst = ::std::mem::MaybeUninit::<$crate::inline_array::RawBuf>::uninit();
            unsafe {
                $cfn(dst.as_mut_ptr(), &self.raw);
                Self {
                    raw: dst.assume_init(),
                }
            }
        }
    };
}
pub(super) use unop;

// ── Trait impls ────────────────────────────────────────────────────────────

impl AsRef<InlineArray> for InlineArray {
    #[inline]
    fn as_ref(&self) -> &InlineArray {
        self
    }
}

// ── Crate-internal helpers for compat.rs ─────────────────────────────────

/// Copy-construct a RawBuf — equivalent to `mlx::core::array` copy constructor.
/// Used by `compat::ops` when it needs to build a contiguous slice of buffers.
#[inline]
pub(crate) unsafe fn raw_copy_buf(dst: *mut RawBuf, src: *const RawBuf) {
    unsafe { mlx_inline_init_copy(dst, src) }
}

/// Destroy a raw buffer — calls the `mlx::core::array` destructor.
#[inline]
pub(crate) unsafe fn raw_destroy(a: *mut RawBuf) {
    unsafe { mlx_inline_destroy(a) }
}

/// Concatenate a contiguous slice of RawBufs along `axis`.
#[inline]
pub(crate) unsafe fn raw_concatenate(dst: *mut RawBuf, arrays: *const RawBuf, num: i32, axis: i32) {
    unsafe { mlx_inline_concatenate(dst, arrays, num, axis) }
}

/// Stack a contiguous slice of RawBufs along a new `axis`.
#[inline]
pub(crate) unsafe fn raw_stack(dst: *mut RawBuf, arrays: *const RawBuf, num: i32, axis: i32) {
    unsafe { mlx_inline_stack(dst, arrays, num, axis) }
}

/// Wrap a raw RawBuf (already placement-new'd by C++) into an `InlineArray`.
///
/// # Safety
/// `raw` must have been initialised by a C++ placement-new (e.g. via one of
/// the `mlx_inline_*` FFI functions).  Ownership is transferred: `Drop` will
/// call the C++ destructor.
#[inline]
pub(crate) unsafe fn from_raw_buf(raw: RawBuf) -> InlineArray {
    InlineArray { raw }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_slice_set_round_trip(mut base: InlineArray, value: InlineArray, expected: &[f32]) {
        let start = [0, 1];
        let stop = [2, 3];
        base = base.slice_set(&value, &start, &stop);
        let got = base.to_f32_vec(expected.len()).expect("to_f32_vec");
        assert_eq!(got, expected);
    }

    fn assert_tail_write_after_kv_cache_append(
        mut base: InlineArray,
        zeros: InlineArray,
        value: InlineArray,
        expected: &[f32],
    ) {
        base = base.kv_cache_append(&zeros, 2);
        let start = [0, 0, 3, 0];
        let stop = [1, 1, 4, 2];
        base = base.slice_set(&value, &start, &stop);
        let got = base.to_f32_vec(expected.len()).expect("to_f32_vec");
        assert_eq!(got, expected);
    }

    #[test]
    fn test_buffer_layout() {
        verify_buffer_layout();
    }

    #[test]
    fn test_scalar_roundtrip() {
        let a = InlineArray::from_f32(std::f32::consts::PI);
        a.eval();
        let v = a.item_f32();
        assert!((v - std::f32::consts::PI).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn test_add_scalars() {
        let a = InlineArray::from_f32(2.0);
        let b = InlineArray::from_f32(3.0);
        let c = a.add(&b);
        c.eval();
        let v = c.item_f32();
        assert!((v - 5.0).abs() < 1e-6, "expected 5.0, got {v}");
    }

    #[test]
    fn eval_token_surfaces_pending_bridge_error() {
        let a = InlineArray::from_f32_slice(&[1.0; 6], &[2, 3]);
        let b = InlineArray::from_f32_slice(&[1.0; 20], &[4, 5]);
        let sentinel = crate::expect_runtime_errors(|| a.matmul(&b));

        let err = sentinel
            .eval()
            .into_result()
            .expect_err("matmul shape error should survive eval");
        match err {
            crate::RuntimeError::CxxException(msg) => {
                assert!(msg.contains("[matmul]"), "expected op tag, got {msg}");
            }
            crate::RuntimeError::Unknown(msg) => {
                panic!("expected CxxException, got Unknown: {msg}");
            }
        }
    }

    #[test]
    fn test_slice_set_f32() {
        let base = InlineArray::zeros(&[2, 4], crate::compat::Dtype::Float32.as_i32());
        let value = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_slice_set_round_trip(base, value, &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_slice_set_u8() {
        let base = InlineArray::zeros(&[2, 4], crate::compat::Dtype::Uint8.as_i32());
        let value = InlineArray::from_u8_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_slice_set_round_trip(base, value, &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_slice_set_u32() {
        let base = InlineArray::zeros(&[2, 4], crate::compat::Dtype::Uint32.as_i32());
        let value = InlineArray::from_u32_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_slice_set_round_trip(base, value, &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_tail_slice_set_after_kv_cache_append_f32() {
        let base =
            InlineArray::from_f32_slice(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0], &[1, 1, 3, 2]);
        let zeros = InlineArray::zeros(&[1, 1, 1, 2], crate::compat::Dtype::Float32.as_i32());
        let value = InlineArray::from_f32_slice(&[20.0, 21.0], &[1, 1, 1, 2]);
        assert_tail_write_after_kv_cache_append(
            base,
            zeros,
            value,
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0],
        );
    }

    #[test]
    fn test_tail_slice_set_after_kv_cache_append_u8() {
        let base = InlineArray::from_u8_slice(&[10, 11, 12, 13, 14, 15], &[1, 1, 3, 2]);
        let zeros = InlineArray::zeros(&[1, 1, 1, 2], crate::compat::Dtype::Uint8.as_i32());
        let value = InlineArray::from_u8_slice(&[20, 21], &[1, 1, 1, 2]);
        assert_tail_write_after_kv_cache_append(
            base,
            zeros,
            value,
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0],
        );
    }

    #[test]
    fn test_tail_slice_set_after_kv_cache_append_u32() {
        let base = InlineArray::from_u32_slice(&[10, 11, 12, 13, 14, 15], &[1, 1, 3, 2]);
        let zeros = InlineArray::zeros(&[1, 1, 1, 2], crate::compat::Dtype::Uint32.as_i32());
        let value = InlineArray::from_u32_slice(&[20, 21], &[1, 1, 1, 2]);
        assert_tail_write_after_kv_cache_append(
            base,
            zeros,
            value,
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0],
        );
    }
}
