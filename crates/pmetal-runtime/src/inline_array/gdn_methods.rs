//! GDN (Gated Delta Network) recurrence methods on [`InlineArray`].
//!
//! - `gdn_metal_step` / `gdn_metal_state_update` dispatch to the fused Metal
//!   kernel (used during inference for Qwen3Next-style hybrid archs).
//!
//! The gated delta rule's dispatcher, with its ops-based sequential and
//! chunkwise forms, is [`crate::nn::gated_delta`].

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

impl InlineArray {
    // ── GDN Metal kernel step ───────────────────────────────────────────

    /// GDN recurrence with pre-computed g and beta. Uses Metal kernel (1 dispatch)
    /// when dk%32==0 && dk<=256, otherwise falls back to ops.
    #[inline]
    pub fn gdn_metal_step(
        q: &Self,
        k: &Self,
        v: &Self,
        g: &Self,
        beta: &Self,
        state: &Self,
        t: i32,
    ) -> (Self, Self) {
        let mut dst_y = MaybeUninit::<RawBuf>::uninit();
        let mut dst_state = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gdn_metal_step(
                dst_y.as_mut_ptr(),
                dst_state.as_mut_ptr(),
                &q.raw,
                &k.raw,
                &v.raw,
                &g.raw,
                &beta.raw,
                &state.raw,
                t,
            );
            (
                Self {
                    raw: dst_y.assume_init(),
                },
                Self {
                    raw: dst_state.assume_init(),
                },
            )
        }
    }

    /// GDN state-only advance for speculative-decoding rollback replay.
    ///
    /// Same contract as `gdn_metal_step` but skips the query / output
    /// projection — only the post-replay state is returned. Used when the
    /// caller has accepted a prefix of a drafted block and wants to
    /// reconstruct the recurrent state at the accepted position without
    /// paying for the output computation.
    #[inline]
    pub fn gdn_metal_state_update(
        k: &Self,
        v: &Self,
        g: &Self,
        beta: &Self,
        state: &Self,
        t: i32,
    ) -> Self {
        let mut dst_state = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gdn_metal_state_update(
                dst_state.as_mut_ptr(),
                &k.raw,
                &v.raw,
                &g.raw,
                &beta.raw,
                &state.raw,
                t,
            );
            Self {
                raw: dst_state.assume_init(),
            }
        }
    }
}
