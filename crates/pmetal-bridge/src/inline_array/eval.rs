//! Evaluation and readback on [`InlineArray`].
//!
//! - `eval` / `async_eval` / `eval_2`: materialize the backing computation
//!   graph. `detach` severs the graph to free input references between
//!   decode steps.
//! - `item*` / `as_slice`: extract scalar / borrowed-slice views (require a
//!   prior eval).

use super::dtype::BridgeScalar;
use super::ffi::*;
use super::{EvalToken, InlineArray};
use crate::error::{BridgeResult, check_last_error};

impl InlineArray {
    // ── Eval ─────────────────────────────────────────────────────────────

    pub fn eval(&self) -> EvalToken {
        let prior = check_last_error();
        // MLX array handles are internally mutable; eval materializes the backing
        // graph state but does not change the logical Rust ownership model.
        unsafe { mlx_inline_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken::new(prior)
    }
    pub fn async_eval(&self) -> EvalToken {
        let prior = check_last_error();
        unsafe { mlx_inline_async_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken::new(prior)
    }

    /// Evaluate and return any bridge-side exception as a Rust error.
    pub fn try_eval(&self) -> BridgeResult<()> {
        self.eval().into_result()
    }

    /// Asynchronously evaluate and return any bridge-side exception as a Rust error.
    pub fn try_async_eval(&self) -> BridgeResult<()> {
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
    pub fn as_slice<T: crate::inline_array::BridgeScalar>(&self) -> &[T] {
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

    pub fn item_f32(&self) -> f32 {
        self.eval().expect("item_f32 eval failed");
        unsafe { mlx_inline_item_f32(std::ptr::from_ref(&self.raw).cast_mut()) }
    }
    pub fn item_u32(&self) -> u32 {
        self.eval().expect("item_u32 eval failed");
        unsafe { mlx_inline_item_u32(std::ptr::from_ref(&self.raw).cast_mut()) }
    }

    // ── Item extraction (generic) ────────────────────────────────────────

    /// Extract the scalar value from a 0-d array. Evaluates lazily if needed.
    /// `T` must be `f32`, `u32`, or `i32` (the bridge-exported scalar types).
    pub fn item<T: BridgeScalar>(&self) -> T {
        self.eval().expect("item eval failed");
        T::extract(self)
    }
}
