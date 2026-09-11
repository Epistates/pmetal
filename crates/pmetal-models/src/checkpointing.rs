//! Running a decoder layer under gradient checkpointing.
//!
//! The mechanism is [`pmetal_bridge::compat::checkpointed`], which is `unsafe`
//! for a specific reason: `mlx::core::checkpoint` re-invokes the forward
//! function during the backward pass, so MLX keeps the closure (and everything
//! it borrows) alive for as long as any array still references the graph. The
//! borrow therefore has to outlive the call that created it, which Rust cannot
//! check here.
//!
//! This module is the one place in the crate that opts back into `unsafe`, and
//! it is confined to `pub(crate)` so the invariant only has to hold for callers
//! inside this crate. It does, by construction: [`checkpointed_layer`] is only
//! ever called by a model on a layer it owns, from inside that model's own
//! forward pass. The layer cannot outlive the model, the model belongs to the
//! training loop, and the loop differentiates each step's graph before moving
//! on. Exposing this publicly would hand out a contract an outside caller could
//! break by keeping an output array past the model's lifetime.

// SAFETY: see the module docs. The single `unsafe` call below is a lifetime
// contract on a borrow, not a memory-layout or FFI-pointer claim, and the
// `pub(crate)` bound is what discharges it.
#![allow(unsafe_code)]

use pmetal_bridge::compat::{Array, Exception, ModuleParametersExt};

/// Run one decoder layer under gradient checkpointing.
///
/// The layer's interior activations are discarded when it returns and
/// recomputed during the backward pass, so the trunk's peak activation memory
/// stops scaling with depth. One decoder layer is the unit, matching what
/// PyTorch and mlx-lm both checkpoint; there is no block size to tune.
///
/// Only worth calling when a gradient is going to be taken. Inference has no
/// backward pass to save anything for, so the recompute would be pure cost and
/// callers gate this on there being no KV cache.
///
/// `mask` is threaded through as an input rather than captured. The closure
/// outlives this call, so a captured reference to the caller's mask would be a
/// dangling read on the recompute.
pub(crate) fn checkpointed_layer<L, F>(
    layer: &mut L,
    h: &Array,
    mask: Option<&Array>,
    mut forward: F,
) -> Result<Array, Exception>
where
    L: ModuleParametersExt,
    F: FnMut(&mut L, &Array, Option<&Array>) -> Result<Array, Exception>,
{
    let inputs: Vec<Array> = match mask {
        Some(mask) => vec![h.clone(), mask.clone()],
        None => vec![h.clone()],
    };
    let has_mask = mask.is_some();

    // SAFETY: `layer` is borrowed from the model running this forward pass, so
    // it outlives the graph being built. See the module docs.
    unsafe {
        pmetal_bridge::compat::checkpointed(layer, &inputs, move |layer, inputs| {
            let mask = has_mask.then(|| &inputs[1]);
            forward(layer, &inputs[0], mask)
        })
    }
}
