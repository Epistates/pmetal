//! Autograd entry points: `value_and_grad` and `checkpoint_apply`.
//!
//! Both cross the FFI boundary via trampoline callbacks so the inner closure
//! can use normal `InlineArray` ops while C++ manages the autograd tape.

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

// ── value_and_grad ───────────────────────────────────────────────────────

/// Compute loss + gradients via callback-based autograd.
///
/// `loss_fn` receives all arrays (params first, then inputs) and must return
/// a scalar loss. Gradients are computed w.r.t. the first `params.len()` arrays.
///
/// Returns `(loss, gradients)` where `gradients[i]` is `dloss/dparams[i]`.
pub fn value_and_grad<F>(
    mut loss_fn: F,
    params: &[InlineArray],
    inputs: &[InlineArray],
) -> (InlineArray, Vec<InlineArray>)
where
    F: FnMut(&[InlineArray]) -> InlineArray,
{
    // Trampoline: C++ calls this with InlineArray-sized buffers
    unsafe extern "C" fn trampoline<F: FnMut(&[InlineArray]) -> InlineArray>(
        all_arrays: *const *const RawBuf,
        n_total: i32,
        loss_out: *mut RawBuf,
        ctx: *mut std::ffi::c_void,
    ) {
        let f = unsafe { &mut *(ctx as *mut F) };
        // Wrap raw pointers as borrowed InlineArrays (no ownership transfer)
        let arrays: Vec<InlineArray> = (0..n_total as usize)
            .map(|i| {
                let ptr = unsafe { *all_arrays.add(i) };
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe { mlx_inline_init_copy(dst.as_mut_ptr(), ptr) };
                InlineArray {
                    raw: unsafe { dst.assume_init() },
                }
            })
            .collect();
        let loss = f(&arrays);
        // Write loss into output buffer (placement-copy)
        unsafe { mlx_inline_init_copy(loss_out, &loss.raw) };
        // arrays and loss drop here (calling mlx_inline_destroy for each)
    }

    let n_params = params.len();
    let n_total = n_params + inputs.len();

    // Build flat pointer array: [param0, param1, ..., input0, input1, ...]
    let all_ptrs: Vec<*const RawBuf> = params
        .iter()
        .chain(inputs.iter())
        .map(|a| &a.raw as *const RawBuf)
        .collect();

    let mut loss = InlineArray::from_f32(0.0);
    let mut grads: Vec<InlineArray> = (0..n_params).map(|_| InlineArray::from_f32(0.0)).collect();
    let mut grad_ptrs: Vec<*mut RawBuf> = grads
        .iter_mut()
        .map(|g| &mut g.raw as *mut RawBuf)
        .collect();

    unsafe {
        mlx_inline_value_and_grad(
            trampoline::<F>,
            &mut loss_fn as *mut F as *mut std::ffi::c_void,
            all_ptrs.as_ptr(),
            n_params as i32,
            n_total as i32,
            &mut loss.raw,
            grad_ptrs.as_mut_ptr(),
        );
    }

    (loss, grads)
}

// ── Gradient checkpointing ───────────────────────────────────────────────

/// Capacity of the output buffer handed to the checkpointed callback.
///
/// The callback reports how many outputs it actually produced, so this only has
/// to be an upper bound: a decoder layer running one at a time produces at most
/// a handful (hidden, kv_k, kv_v, state).  Both sides clamp to it.
const MAX_OUTPUTS: usize = 64;

/// Apply gradient checkpointing to a forward function.
///
/// `inner_fn` receives the input arrays and must return a `Vec<InlineArray>`.
/// The returned arrays are computed through `mlx::core::checkpoint()`, which
/// discards all intermediate activations after the forward pass and recomputes
/// them during the backward pass.  This reduces peak activation memory from
/// O(layers × batch × seq × hidden) to O(1 layer) at the cost of one extra
/// forward pass per gradient step.
///
/// # Gradients only reach `inputs`
///
/// `checkpoint` installs a `custom_vjp`, and a `custom_vjp` differentiates with
/// respect to its explicit primals.  Anything `inner_fn` captures is a constant
/// to the tape and comes back with a **zero** gradient, silently.  Every tensor
/// that needs to be trained has to arrive through `inputs` and be read out of
/// the slice, which is why the module-level wrapper threads a layer's trainable
/// parameters in and writes them back before running the layer.
///
/// # Usage
///
/// ```ignore
/// let outputs = checkpoint_apply(&inputs, |arrays| {
///     // normal forward computation using InlineArray ops
///     let h = arrays[0].matmul(&arrays[1]);
///     vec![h.relu()]
/// });
/// ```
///
/// # Errors
///
/// Returns an empty `Vec` if the C++ side threw; the reason is available from
/// [`crate::check_last_error`].
pub fn checkpoint_apply<F>(inputs: &[InlineArray], inner_fn: F) -> Vec<InlineArray>
where
    F: FnMut(&[InlineArray]) -> Vec<InlineArray> + 'static,
{
    // Safety: `F: 'static`, so the closure cannot dangle no matter how long MLX
    // holds on to the graph.
    unsafe { checkpoint_apply_unchecked(inputs, inner_fn) }
}

/// [`checkpoint_apply`] without the `'static` bound on the closure.
///
/// # Safety
///
/// `inner_fn` is invoked again during the backward pass, which happens after
/// this function returns, and MLX keeps the closure alive for as long as any
/// array still references the resulting graph.  The caller must guarantee that
/// everything `inner_fn` borrows outlives every differentiation of that graph.
///
/// The intended use is a decoder layer borrowed from a model that the training
/// loop owns: the model outlives the step, and each step's graph is
/// differentiated once.  Retaining an output array past the model's lifetime,
/// or differentiating the same graph a second time after the borrow has ended,
/// is undefined behaviour.
///
/// The trap is quieter than it looks.  A closure written without `move`
/// captures even plain `Copy` locals by reference, so an index or a length
/// belonging to the calling frame becomes a dangling read on the recompute and
/// comes back as whatever is on the stack by then.  Write the closure as
/// `move` and let it own everything except the long-lived borrows this contract
/// is actually about.
pub unsafe fn checkpoint_apply_unchecked<F>(inputs: &[InlineArray], inner_fn: F) -> Vec<InlineArray>
where
    F: FnMut(&[InlineArray]) -> Vec<InlineArray>,
{
    // Trampoline: C++ calls this with InlineArray-sized bufs for both the
    // input arrays and the flat output buffer.
    unsafe extern "C" fn trampoline<F: FnMut(&[InlineArray]) -> Vec<InlineArray>>(
        all_arrays: *const *const RawBuf,
        n_total: i32,
        outputs_out: *mut RawBuf,
        n_outputs_out: *mut i32,
        ctx: *mut std::ffi::c_void,
    ) {
        let f = unsafe { &mut *(ctx as *mut F) };

        // Borrow-wrap each input pointer as an InlineArray (copy-construct).
        let arrays: Vec<InlineArray> = (0..n_total as usize)
            .map(|i| {
                let ptr = unsafe { *all_arrays.add(i) };
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe { mlx_inline_init_copy(dst.as_mut_ptr(), ptr) };
                InlineArray {
                    raw: unsafe { dst.assume_init() },
                }
            })
            .collect();

        let results = f(&arrays);
        // `outputs_out` holds MAX_OUTPUTS slots; writing past them would run off
        // the end of the caller's buffer.
        let n = results.len().min(MAX_OUTPUTS);

        // Write each output via placement-copy into the caller's flat buffer.
        for (i, r) in results.iter().take(n).enumerate() {
            unsafe { mlx_inline_init_copy(outputs_out.add(i), &r.raw) };
        }
        unsafe { *n_outputs_out = n as i32 };
        // `arrays` and `results` drop here, calling mlx_inline_destroy for each.
    }

    // Frees the boxed closure once C++ drops the last copy of the checkpointed
    // function.  Paired with the `Box::into_raw` below.
    unsafe extern "C" fn drop_ctx<F>(ctx: *mut std::ffi::c_void) {
        drop(unsafe { Box::from_raw(ctx as *mut F) });
    }

    let n_total = inputs.len();
    // Build flat pointer array for the C++ side.
    let all_ptrs: Vec<*const RawBuf> = inputs.iter().map(|a| &a.raw as *const RawBuf).collect();

    // The slots are left uninitialised for C++ to placement-new into, and
    // `set_len` afterwards claims exactly the ones it wrote.  Pre-filling them
    // with real arrays instead would leak one per slot per call, since
    // placement-new does not run the destructor of what it overwrites.
    let mut output_storage: Vec<InlineArray> = Vec::with_capacity(MAX_OUTPUTS);

    // Hand the closure to C++, which keeps it alive until the graph is gone.
    let ctx = Box::into_raw(Box::new(inner_fn)) as *mut std::ffi::c_void;

    let mut n_written: i32 = 0;
    unsafe {
        mlx_inline_checkpoint(
            trampoline::<F>,
            ctx,
            drop_ctx::<F>,
            all_ptrs.as_ptr(),
            n_total as i32,
            MAX_OUTPUTS as i32,
            // Safety: InlineArray is a single-field struct wrapping RawBuf, so
            // the Vec's buffer is MAX_OUTPUTS contiguous RawBuf-sized slots.
            output_storage.as_mut_ptr() as *mut RawBuf,
            &mut n_written,
        );
    }

    // Safety: C++ placement-new'd exactly `n_written` slots (clamped to
    // MAX_OUTPUTS on its side), and wrote 0 if it threw.
    let n = (n_written as usize).min(MAX_OUTPUTS);
    unsafe { output_storage.set_len(n) };
    output_storage
}
