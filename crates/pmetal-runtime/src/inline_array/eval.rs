//! Evaluation and readback on [`InlineArray`].
//!
//! - `eval` / `async_eval` / `eval_2`: materialize the backing computation
//!   graph. `detach` severs the graph to free input references between
//!   decode steps.
//! - `item*` / `as_slice`: extract scalar / borrowed-slice views (require a
//!   prior eval).

use super::dtype::RuntimeScalar;
use super::ffi::*;
use super::{EvalToken, InlineArray, RawBuf};
use crate::error::RuntimeResult;

impl InlineArray {
    // ── Eval ─────────────────────────────────────────────────────────────

    /// Materialize the array. The returned token reports a failure of this
    /// evaluation or of the op just before it (a success wipes the last-op
    /// slot, so the token remembers what was there).
    ///
    /// The token is only *looked at* here: dropping it, as nearly every caller
    /// does (`x.eval();`), leaves an earlier op's error on the channels for
    /// the next check. It used to take the error out of them, so `eval`
    /// discarded the failure of the op before it.
    pub fn eval(&self) -> EvalToken {
        let prior = crate::error::peek_last_error();
        // MLX array handles are internally mutable; eval materializes the backing
        // graph state but does not change the logical Rust ownership model.
        unsafe { mlx_inline_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken::new(prior)
    }

    /// [`eval`](Self::eval), without waiting for the result.
    pub fn async_eval(&self) -> EvalToken {
        let prior = crate::error::peek_last_error();
        unsafe { mlx_inline_async_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken::new(prior)
    }

    /// Materialize every array of a set in ONE evaluation: one submission and
    /// one wait for the lot, where `eval` on each would wait once apiece. The
    /// arrays stay attached to their graphs (unlike [`eval_and_detach_many`]
    /// (crate::inline_array::eval_and_detach_many)) and the values are the
    /// ones `eval` would give. An empty set does nothing.
    pub fn eval_many<'a>(arrays: impl IntoIterator<Item = &'a InlineArray>) -> EvalToken {
        let prior = crate::error::peek_last_error();
        let mut ptrs: Vec<*mut RawBuf> = arrays
            .into_iter()
            .map(|a| std::ptr::from_ref(&a.raw).cast_mut())
            .collect();
        if !ptrs.is_empty() {
            // SAFETY: each pointer addresses a live array; the bridge copies
            // the handles (sharing their graphs) and does not write through
            // them, as `eval` does not.
            unsafe { mlx_inline_eval_many(ptrs.as_mut_ptr(), ptrs.len() as i32) }
        }
        EvalToken::new(prior)
    }

    /// Evaluate and return any bridge-side exception as a Rust error.
    pub fn try_eval(&self) -> RuntimeResult<()> {
        self.eval().into_result()
    }

    /// Asynchronously evaluate and return any bridge-side exception as a Rust error.
    pub fn try_async_eval(&self) -> RuntimeResult<()> {
        self.async_eval().into_result()
    }

    /// Eval two arrays in one call (avoids two FFI round-trips).
    #[inline]
    pub fn eval_2(a: &mut Self, b: &mut Self) {
        unsafe { mlx_inline_eval_2(&mut a.raw, &mut b.raw) }
    }

    /// Sever the computation graph, freeing all input references.
    /// CRITICAL for cache arrays: without this, cache updates chain across
    /// decode steps, keeping ALL previous steps' Metal buffers alive.
    /// Call on cache arrays after each eval to prevent memory accumulation.
    #[inline]
    pub fn detach(&mut self) {
        unsafe { mlx_inline_detach(&mut self.raw) }
    }

    // ── Async eval on borrowed ref ──────────────────────────────────────

    pub fn async_eval_ref(&self) {
        unsafe { mlx_inline_async_eval_arr(&self.raw) }
    }

    // ── Slice access (requires prior eval) ───────────────────────────────

    /// Borrow the array's elements as a flat slice, in row-major order.
    ///
    /// Evaluates the array if it isn't already (and waits on an in-flight
    /// `async_eval`). A strided view (a transpose, a slice of an inner axis
    /// or with a step, a broadcast) is materialized into a row-contiguous
    /// buffer first, in place on the shared MLX array; a row-contiguous array
    /// is borrowed with no copy. An empty array gives an empty slice.
    ///
    /// # Panics
    /// If `T` doesn't match the dtype (`f32` needs float32; `u32`/`i32` need
    /// a 32-bit integer dtype), or if evaluation fails.
    pub fn as_slice<T: crate::inline_array::RuntimeScalar>(&self) -> &[T] {
        let dtype = self.dtype_raw();
        assert!(
            T::SLICE_DTYPES.contains(&dtype),
            "as_slice::<{}>() on a {:?} array",
            std::any::type_name::<T>(),
            crate::compat::Dtype::from_raw(dtype),
        );
        let n = self.size();
        let ptr = match self.try_data_ptr() {
            Ok(ptr) => ptr as *const T,
            Err(e) => panic!("as_slice: evaluating the array failed: {e}"),
        };
        if n == 0 {
            return &[];
        }
        assert!(!ptr.is_null(), "as_slice: null data pointer");
        // SAFETY: `data_ptr` evaluated the array and left it row-contiguous,
        // so `ptr` addresses `n` elements of `T` (dtype checked above) in a
        // buffer the array desc owns for as long as `self` lives.
        unsafe { std::slice::from_raw_parts(ptr, n) }
    }

    // ── Item extraction ──────────────────────────────────────────────────
    //
    // The C++ FFI signatures take `mlx_inline_array*` for ABI uniformity but
    // both ops only mutate through MLX's internal interior-mutability: eval
    // materializes the backing graph and item<T> reads the resulting scalar.
    // Cloning self before the call would only bump the underlying refcount
    // without changing observable behavior, so we route the read through a
    // const→mut pointer cast (matches `eval()` above).

    /// The element of a one-element array as `f32`, whatever its dtype
    /// (a bf16, f16 or integer element is converted).
    pub fn item_f32(&self) -> f32 {
        self.eval().expect("item_f32 eval failed");
        unsafe { mlx_inline_item_f32(std::ptr::from_ref(&self.raw).cast_mut()) }
    }

    /// The element of a one-element array as `u32`, converted as
    /// [`as_dtype`](Self::as_dtype) converts: a float is truncated.
    pub fn item_u32(&self) -> u32 {
        self.eval().expect("item_u32 eval failed");
        unsafe { mlx_inline_item_u32(std::ptr::from_ref(&self.raw).cast_mut()) }
    }

    /// The element of a one-element array as `i32`, converted as
    /// [`as_dtype`](Self::as_dtype) converts: a float is truncated.
    pub fn item_i32(&self) -> i32 {
        self.eval().expect("item_i32 eval failed");
        unsafe { mlx_inline_item_i32(std::ptr::from_ref(&self.raw).cast_mut()) }
    }

    // ── Item extraction (generic) ────────────────────────────────────────

    /// Extract the element of a one-element array, converted to `T`.
    /// Evaluates lazily if needed. `T` must be `f32`, `u32`, or `i32`.
    pub fn item<T: RuntimeScalar>(&self) -> T {
        T::extract(self)
    }
}
