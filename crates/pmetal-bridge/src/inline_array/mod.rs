//! Zero-allocation MLX array — stores `mlx::core::array` inline on the Rust stack.
//!
//! This eliminates ALL per-op heap allocation, matching Python/nanobind's direct
//! C++ binding performance. Each op is a single `extern "C"` call with placement-new
//! into a caller-provided buffer.

use std::mem::MaybeUninit;

mod dtype;
pub use dtype::{ArrayElement, AsDtype, BridgeScalar, bf16, f16};

mod diagnostics;
pub use diagnostics::{
    clear_cache, disable_compile, enable_compile, eval_and_detach_many, get_active_memory,
    get_cache_memory, get_max_recommended_size, get_peak_memory, graph_desc_count, graph_dump,
    graph_node_count, metal_start_capture, metal_stop_capture, new_generation_stream,
    reset_default_stream, reset_peak_memory, set_generation_stream, set_wired_limit,
    set_wired_limit_max, synchronize, verify_buffer_layout,
};

mod safetensors;
pub use safetensors::{load_safetensors_shard, random_seed};

mod autograd;
pub use autograd::{checkpoint_apply, value_and_grad};

mod compiled;
mod eval;
mod factory;
mod fast;
mod gather;
mod gdn_methods;
mod interop;
mod ops;
mod quantized;
pub use quantized::QuantizedMode;
mod reductions;
mod shape_ops;
mod turboquant_methods;

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

mod ffi;
use ffi::*;

// ── Full Qwen3.5 forward pass ─────────────────────────────────────────────

/// Run the entire Qwen3.5 forward pass (all N layers) as a single C++ call,
/// eliminating per-op FFI overhead (~1800 round trips per decode step).
///
/// # Safety
/// All raw pointers in `weight_ptrs` and `cache_ptrs` must point to live,
/// placement-new'd `mlx::core::array` objects (i.e. valid `InlineArray.raw`
/// fields).  The arrays must remain live for the duration of this call.
///
/// `attn_kv_offsets` and `rope_offset` are updated in-place by C++.
pub(crate) unsafe fn qwen35_decode_step(
    token_ids: &InlineArray,
    weight_ptrs: &[*const RawBuf],
    cache_ptrs: &mut [*mut RawBuf],
    attn_kv_offsets: &mut [i32],
    rope_offset: &mut i32,
    config_ints: &[i32],
    config_floats: &[f32],
) -> InlineArray {
    let mut dst = InlineArray::uninit();
    unsafe {
        mlx_inline_qwen35_decode_step(
            dst.as_raw_ptr_mut(),
            token_ids.as_raw_ptr(),
            weight_ptrs.as_ptr(),
            weight_ptrs.len() as i32,
            cache_ptrs.as_mut_ptr(),
            cache_ptrs.len() as i32,
            attn_kv_offsets.as_mut_ptr(),
            rope_offset,
            config_ints.as_ptr(),
            config_ints.len() as i32,
            config_floats.as_ptr(),
            config_floats.len() as i32,
        );
    }
    dst
}

// ── InlineArray ───────────────────────────────────────────────────────────

/// Stack-allocated MLX array. Zero heap allocation per op.
pub struct InlineArray {
    pub(crate) raw: RawBuf,
}

#[derive(Clone, Debug)]
pub struct EvalToken {
    prior: crate::error::BridgeResult<()>,
}

impl Default for EvalToken {
    fn default() -> Self {
        Self { prior: Ok(()) }
    }
}

impl EvalToken {
    #[inline]
    pub(crate) fn new(prior: crate::error::BridgeResult<()>) -> Self {
        Self { prior }
    }

    #[inline]
    pub fn into_result(self) -> crate::error::BridgeResult<()> {
        let after = crate::error::check_last_error();
        match self.prior {
            Ok(()) => after,
            Err(err) => {
                let _ = after;
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
        let sentinel = a.matmul(&b);

        let err = sentinel
            .eval()
            .into_result()
            .expect_err("matmul shape error should survive eval");
        match err {
            crate::BridgeError::CxxException(msg) => {
                assert!(msg.contains("[matmul]"), "expected op tag, got {msg}");
            }
            crate::BridgeError::Unknown(msg) => {
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
