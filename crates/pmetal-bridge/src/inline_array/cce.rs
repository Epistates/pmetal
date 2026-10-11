//! Fused cut cross-entropy: the Metal kernels and custom VJP in
//! `cpp/bridge_cce.cpp`.

use std::ffi::CStr;
use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;
use crate::error::{BridgeError, BridgeResult};

/// The LM head's logit transform and the backward's knobs.
#[derive(Debug, Clone, Copy)]
pub struct FusedCceParams {
    /// Multiplies the head's output (1 for most heads).
    pub logit_scale: f32,
    /// `cap · tanh(logits / cap)` after the scale; 0 disables it.
    pub softcap: f32,
    /// The backward leaves out a 64-row tile of the vocabulary for a group of
    /// eight tokens whose softmax stays under this across it; 0 leaves out
    /// nothing.
    pub filter_eps: f32,
    /// Vocabulary columns of logit gradient the backward holds at once.
    pub vocab_chunk: usize,
}

/// Each token's logsumexp over the vocabulary and its target logit.
#[derive(Debug)]
pub struct FusedCceOutput {
    /// `[tokens]`, f32.
    pub lse: InlineArray,
    /// `[tokens]`, f32; zero where the target is outside the vocabulary.
    pub target_logit: InlineArray,
}

/// Each token's logsumexp and target logit under the logits
/// `softcap(scale · (hidden · weightᵀ + bias))`, computed by fused Metal
/// kernels that never hold the `[tokens, vocab]` logits.
///
/// Differentiable: a custom VJP gives the hidden states, the head and the
/// bias their gradients. `Ok(Err(reason))` when the kernels can't take these
/// inputs (no Metal GPU, a dtype other than f32 / f16 / bf16, shapes that
/// don't multiply), for the caller to take another path.
///
/// * `hidden` - `[tokens, hidden]`
/// * `weight` - `[vocab, hidden]`
/// * `targets` - `[tokens]`, integer
/// * `bias` - `[vocab]`, when the head has one
pub fn cut_cross_entropy_fused(
    hidden: &InlineArray,
    weight: &InlineArray,
    targets: &InlineArray,
    bias: Option<&InlineArray>,
    params: FusedCceParams,
) -> BridgeResult<Result<FusedCceOutput, &'static str>> {
    // Which gradients autograd will want: an array it differentiates is a
    // tracer, and so is everything computed from one. This only shapes the
    // backward's memory schedule; both gradients are built either way.
    let want_dx = hidden.is_tracer();
    let want_dw = weight.is_tracer() || bias.is_some_and(InlineArray::is_tracer);
    let mut lse = MaybeUninit::<RawBuf>::uninit();
    let mut target = MaybeUninit::<RawBuf>::uninit();
    let mut reason: *const std::ffi::c_char = std::ptr::null();
    let status = unsafe {
        mlx_inline_cce_fused(
            lse.as_mut_ptr(),
            target.as_mut_ptr(),
            &hidden.raw,
            &weight.raw,
            &targets.raw,
            bias.map_or(std::ptr::null(), |b| &b.raw as *const RawBuf),
            params.logit_scale,
            params.softcap,
            params.filter_eps,
            params.vocab_chunk.min(i32::MAX as usize) as i32,
            want_dx,
            want_dw,
            &mut reason,
        )
    };
    // Every path leaves both outputs holding an array.
    let (lse, target) = unsafe {
        (
            InlineArray {
                raw: lse.assume_init(),
            },
            InlineArray {
                raw: target.assume_init(),
            },
        )
    };
    match status {
        0 => Ok(Ok(FusedCceOutput {
            lse,
            target_logit: target,
        })),
        1 => {
            let why = if reason.is_null() {
                "unsupported inputs"
            } else {
                // A string literal in the bridge.
                unsafe { CStr::from_ptr(reason) }
                    .to_str()
                    .unwrap_or("unsupported inputs")
            };
            Ok(Err(why))
        }
        _ => Err(crate::check_last_error().err().unwrap_or_else(|| {
            BridgeError::Unknown("cut cross-entropy failed without an error".into())
        })),
    }
}
