//! Generic `mlx::core::compile()` wrapper.
//!
//! mlx::core::compile traces a callable into a single fused graph on first
//! invocation and replays that trace on every subsequent call. The perf
//! memory (`project_perf_gap_mlxlm.md` / `project_mlxrs_perf_findings.md`)
//! explicitly calls out compiled graphs as the dominant remaining lever
//! for inference throughput — measured ~7× over the uncompiled path on
//! Qwen3.5 decode.
//!
//! The pmetal bridge already uses `compile()` internally for specific
//! fused ops (`bridge_compiled.cpp::make_compiled`). This module exposes
//! it *generically* so any Rust caller can compile an arbitrary closure
//! and call it repeatedly.
//!
//! ## Usage
//!
//! ```ignore
//! use pmetal_bridge::{CompiledFn, InlineArray};
//!
//! // Compile a SwiGLU(gate, up) closure once.
//! let mut swiglu = CompiledFn::new(1, /* shapeless= */ true, |inputs| {
//!     let g = &inputs[0];
//!     let u = &inputs[1];
//!     let activated = g.multiply(&g.sigmoid()); // silu(g)
//!     vec![activated.multiply(u)]
//! })?;
//!
//! // Call it as many times as you like — the trace is re-used.
//! let gate = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
//! let up   = InlineArray::from_f32_slice(&[0.5, 0.5], &[2]);
//! let outs = swiglu.call(&[&gate, &up])?;
//! ```
//!
//! ## Ownership + lifetime
//!
//! `CompiledFn` owns:
//! 1. An opaque C++ handle (`*mut c_void`) — the MLX compiled closure.
//! 2. A `Box<dyn FnMut(&[InlineArray]) -> Vec<InlineArray>>` — the Rust
//!    closure, kept alive via raw pointer in the C++ trampoline's `ctx`.
//!
//! On drop the C++ handle is destroyed first (no more trampoline calls
//! possible), then the Box is reclaimed and dropped.
//!
//! The wrapper is **`!Send` + `!Sync`** — the Rust closure might capture
//! non-thread-safe state, and MLX's compile cache isn't known to be
//! thread-safe for the same closure handle. Build one handle per thread
//! if you need concurrent compiled dispatch.
//!
//! ## Atexit caveat
//!
//! Upstream MLX has a cross-DSO atexit ordering issue where destroying a
//! `mlx::core::CompiledFn` after libmlx has been unloaded causes a
//! segfault (see the comment on `bridge_compiled.cpp::make_compiled`).
//! Dropping `CompiledFn` *during* program runtime is safe — the problem
//! is specifically holding one in a `static` / `OnceCell` / lazy-init
//! singleton that only gets dropped at process exit. If you need a
//! process-wide compiled closure, `Box::leak` it or wrap it in
//! something that explicitly never runs Drop.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::InlineArray;
use crate::error::{BridgeError, BridgeResult, check_last_error};
use crate::inline_array::RawBuf;

type BoxedClosure = Box<dyn FnMut(&[InlineArray]) -> Vec<InlineArray>>;

#[allow(non_camel_case_types)]
type mlx_rust_compile_forward_fn = unsafe extern "C" fn(
    inputs: *const *const RawBuf,
    n_inputs: i32,
    outputs: *mut RawBuf,
    n_outputs_written: *mut i32,
    ctx: *mut c_void,
);

// Redeclared here so we don't need to widen the visibility of the
// `extern "C"` block in `inline_array.rs`. Rust treats these as distinct
// items; the linker resolves them to the same underlying C symbols.
unsafe extern "C" {
    fn mlx_inline_compile_make(
        fn_ptr: mlx_rust_compile_forward_fn,
        ctx: *mut c_void,
        n_outputs_max: i32,
        shapeless: bool,
    ) -> *mut c_void;

    fn mlx_inline_compile_call(
        compiled_handle: *mut c_void,
        inputs: *const *const RawBuf,
        n_inputs: i32,
        outputs: *mut RawBuf,
        n_outputs_max: i32,
        n_outputs_written: *mut i32,
    ) -> i32;

    fn mlx_inline_compile_free(compiled_handle: *mut c_void);

    fn mlx_inline_init_copy(dst: *mut RawBuf, src: *const RawBuf);
    fn mlx_inline_init_empty(dst: *mut RawBuf);
}

// ---------------------------------------------------------------------------
// Trampoline
// ---------------------------------------------------------------------------
//
// MLX invokes this extern "C" function on every compiled-closure call.
// `ctx` is the `*mut BoxedClosure` we handed to the C++ side; we
// reinterpret it, convert the input buffers to borrowed `InlineArray`
// values, invoke the closure, and write the outputs back into the
// caller-allocated buffer slots.

unsafe extern "C" fn trampoline(
    inputs: *const *const RawBuf,
    n_inputs: i32,
    outputs: *mut RawBuf,
    n_outputs_written: *mut i32,
    ctx: *mut c_void,
) {
    // SAFETY: `ctx` originated from `Box::into_raw(Box<BoxedClosure>)`.
    let closure = unsafe { &mut *(ctx as *mut BoxedClosure) };

    let n_inputs = n_inputs.max(0) as usize;
    // Wrap each input pointer as an owned `InlineArray` backed by a
    // placement-copy of the C++ array. Matches the value_and_grad
    // trampoline pattern.
    let input_arrays: Vec<InlineArray> = (0..n_inputs)
        .map(|i| {
            let raw_ptr = unsafe { *inputs.add(i) };
            let mut dst = MaybeUninit::<RawBuf>::uninit();
            unsafe { mlx_inline_init_copy(dst.as_mut_ptr(), raw_ptr) };
            InlineArray {
                raw: unsafe { dst.assume_init() },
            }
        })
        .collect();

    let results = closure(&input_arrays);

    // Write outputs back. The C++ side pre-initialised every output slot
    // with an empty array via `mlx_inline_init_empty`, so we destructively
    // overwrite up to results.len() slots by placement-copying.
    let n_out = results.len();
    for (i, out_arr) in results.into_iter().enumerate() {
        let slot = unsafe { outputs.add(i) };
        // Destroy the placeholder empty array first.
        unsafe {
            std::ptr::drop_in_place(slot as *mut InlineArray as *mut ());
        }
        unsafe { mlx_inline_init_copy(slot, out_arr.as_raw_ptr()) };
        // `out_arr` drops at end of this iteration, releasing its own copy.
    }

    unsafe {
        *n_outputs_written = n_out as i32;
    }
}

// ---------------------------------------------------------------------------
// Public wrapper
// ---------------------------------------------------------------------------

/// A Rust closure traced through `mlx::core::compile()`.
///
/// See [module docs](self) for the ownership model + thread-safety
/// constraints + atexit caveat.
pub struct CompiledFn {
    handle: NonNull<c_void>,
    // Kept alive until Drop so the C++ trampoline's `ctx` pointer stays
    // valid across every invocation.
    closure_box: *mut BoxedClosure,
    n_outputs_max: i32,
    // !Send + !Sync — see module docs.
    _not_send: PhantomData<*const ()>,
}

impl CompiledFn {
    /// Compile a Rust closure into a reusable fused graph.
    ///
    /// * `n_outputs_max` — upper bound on the number of arrays the closure
    ///   will produce. Must be > 0.
    /// * `shapeless` — when `true`, the trace is reused across different
    ///   input shapes; `false` retraces on every shape change. Shapeless
    ///   compile matches most inference uses (T=1 decode).
    /// * `closure` — receives a slice of input arrays and returns a
    ///   vector of output arrays whose length is ≤ `n_outputs_max`.
    ///
    /// Returns [`BridgeError`] when MLX's `compile()` rejects the closure
    /// or the handle allocation fails.
    pub fn new<F>(n_outputs_max: usize, shapeless: bool, closure: F) -> BridgeResult<Self>
    where
        F: FnMut(&[InlineArray]) -> Vec<InlineArray> + 'static,
    {
        if n_outputs_max == 0 {
            return Err(BridgeError::CxxException(
                "CompiledFn::new: n_outputs_max must be > 0".into(),
            ));
        }
        let boxed: BoxedClosure = Box::new(closure);
        let boxed_ptr: *mut BoxedClosure = Box::into_raw(Box::new(boxed));

        // SAFETY: trampoline matches the mlx_rust_compile_forward_fn type;
        // boxed_ptr is a valid pointer for the closure's lifetime.
        let raw_handle = unsafe {
            mlx_inline_compile_make(
                trampoline,
                boxed_ptr as *mut c_void,
                n_outputs_max as i32,
                shapeless,
            )
        };

        let handle = match NonNull::new(raw_handle) {
            Some(h) => h,
            None => {
                // Compile failed — reclaim the Box before returning.
                unsafe {
                    drop(Box::from_raw(boxed_ptr));
                }
                return Err(match check_last_error() {
                    Err(e) => e,
                    Ok(()) => {
                        BridgeError::CxxException("mlx::core::compile() returned null".into())
                    }
                });
            }
        };

        Ok(Self {
            handle,
            closure_box: boxed_ptr,
            n_outputs_max: n_outputs_max as i32,
            _not_send: PhantomData,
        })
    }

    /// Invoke the compiled closure.
    ///
    /// `inputs` are borrowed — they're placement-copied into MLX's
    /// `std::vector<array>` on the C++ side, so the caller retains
    /// ownership. The returned vector holds up to `n_outputs_max` arrays
    /// (the exact count is whatever the closure's last call produced).
    pub fn call(&mut self, inputs: &[&InlineArray]) -> BridgeResult<Vec<InlineArray>> {
        let input_ptrs: Vec<*const RawBuf> =
            inputs.iter().map(|a| &a.raw as *const RawBuf).collect();

        // Allocate output slots — they'll receive placement-new'd arrays
        // from the C++ side on success.
        let mut outputs: Vec<InlineArray> = (0..self.n_outputs_max as usize)
            .map(|_| {
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe { mlx_inline_init_empty(dst.as_mut_ptr()) };
                InlineArray {
                    raw: unsafe { dst.assume_init() },
                }
            })
            .collect();

        let mut n_written: i32 = 0;
        // SAFETY: handle is valid (NonNull, created by compile_make);
        // input_ptrs contains pointers to the caller's InlineArray::raw
        // fields; outputs has n_outputs_max valid slots.
        let rc = unsafe {
            mlx_inline_compile_call(
                self.handle.as_ptr(),
                input_ptrs.as_ptr(),
                input_ptrs.len() as i32,
                outputs.as_mut_ptr() as *mut RawBuf,
                self.n_outputs_max,
                &mut n_written as *mut i32,
            )
        };

        if rc != 0 {
            return Err(check_last_error()
                .err()
                .unwrap_or(BridgeError::CxxException(
                    "compile_call returned non-zero without setting the error channel".into(),
                )));
        }

        outputs.truncate(n_written.max(0) as usize);
        Ok(outputs)
    }

    /// Number of outputs this compiled closure can produce per call.
    pub fn n_outputs_max(&self) -> usize {
        self.n_outputs_max.max(0) as usize
    }
}

impl Drop for CompiledFn {
    fn drop(&mut self) {
        // Order matters: destroy the C++ handle first so MLX can't invoke
        // the trampoline after we've dropped the Box.
        // SAFETY: handle was returned by compile_make and hasn't been
        // freed elsewhere (we own it exclusively).
        unsafe { mlx_inline_compile_free(self.handle.as_ptr()) };

        if !self.closure_box.is_null() {
            // SAFETY: closure_box was allocated via Box::into_raw in new()
            // and hasn't been reclaimed since. No more trampoline calls
            // can reach it after compile_free returned.
            unsafe {
                drop(Box::from_raw(self.closure_box));
            }
        }
    }
}

impl std::fmt::Debug for CompiledFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFn")
            .field("handle", &self.handle.as_ptr())
            .field("n_outputs_max", &self.n_outputs_max)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canary: compile a trivial 1-input, 1-output pass-through and
    /// verify it runs end-to-end twice. Proves the FFI + trampoline are
    /// wired, the compile cache replays, and the error channel is clean.
    #[test]
    fn compiles_and_reuses_trace() {
        crate::error::clear_last_error();

        let mut add_one = CompiledFn::new(1, /* shapeless= */ true, |inputs| {
            let x = &inputs[0];
            vec![x.add_scalar(1.0)]
        })
        .expect("compile should succeed for a simple add-scalar closure");

        // First invocation — traces + executes.
        let x1 = InlineArray::from_f32_slice(&[2.0, 3.0], &[2]);
        let outs1 = add_one.call(&[&x1]).expect("first compile_call");
        assert_eq!(outs1.len(), 1);
        assert_eq!(outs1[0].shape(), &[2]);

        // Second invocation — replays the trace.
        let x2 = InlineArray::from_f32_slice(&[10.0, 20.0], &[2]);
        let outs2 = add_one.call(&[&x2]).expect("second compile_call");
        assert_eq!(outs2.len(), 1);
        assert_eq!(outs2[0].shape(), &[2]);

        // Verify arithmetic: x + 1.
        let mut got = outs2.into_iter().next().unwrap();
        let v = got.to_f32_vec(2).expect("to_f32_vec");
        assert!((v[0] - 11.0).abs() < 1e-5, "got {}", v[0]);
        assert!((v[1] - 21.0).abs() < 1e-5, "got {}", v[1]);

        // Error channel stays clean after two successful calls.
        assert_eq!(check_last_error(), Ok(()));
    }

    #[test]
    fn rejects_zero_outputs_max() {
        let result = CompiledFn::new(0, true, |_| vec![]);
        assert!(result.is_err());
    }

    // Compile-time assertion: CompiledFn must not implement Send.
    //
    // Enforced by the `_not_send: PhantomData<*const ()>` field on the
    // struct; any future refactor that drops it will be caught by
    // downstream callers that try to share a CompiledFn across threads
    // (which Rust will reject because the raw pointers inside are !Send).
}
