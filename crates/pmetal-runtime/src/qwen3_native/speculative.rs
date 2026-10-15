//! Undoing the rejected tail of a speculative verify.
//!
//! A verify runs the target over a drafted block and keeps a prefix of it.
//! Attention layers forget the rest by moving their cache offset back
//! ([`super::rollback_cache`]), but a GDN layer folds every token into one
//! recurrent state. So the verify records each GDN layer's recurrence inputs
//! ([`GdnReplay`]), and [`rewind_verify`] replays them over the kept prefix
//! from the state the block started at, as the reference implementation of
//! block drafting does. The replay runs the same kernel as the verify, and
//! the convolution state is the last inputs of the kept prefix.

use crate::InlineArray;

use super::cache::NativeCache;
use super::forward::rollback_cache;

/// One GDN layer's recurrence inputs over a verified block.
#[derive(Clone)]
pub struct GdnReplay {
    /// `[B, T, Hk, Dk]`, normed and scaled.
    pub queries: InlineArray,
    /// `[B, T, Hk, Dk]`, normed.
    pub keys: InlineArray,
    /// `[B, T, Hv, Dv]`.
    pub values: InlineArray,
    /// Decay, `[B, T, Hv]`.
    pub g: InlineArray,
    /// `[B, T, Hv]`.
    pub beta: InlineArray,
    /// The recurrent state before the block, `[B, Hv, Dv, Dk]`.
    pub initial_state: InlineArray,
    /// The convolution's input: the `conv_kernel - 1` inputs before the
    /// block, then the block's, `[B, conv_kernel - 1 + T, conv_dim]`.
    pub conv_input: InlineArray,
    pub conv_kernel: i32,
}

/// Leave `cache` as if only the first `accepted` of the `verified` tokens
/// recorded in `replays` (one per GDN layer, in layer order) had been run.
pub fn rewind_verify(
    cache: &mut NativeCache,
    replays: &[GdnReplay],
    verified: i32,
    accepted: i32,
) -> Result<(), String> {
    if accepted >= verified {
        return Ok(());
    }
    if accepted < 0 {
        return Err(format!("rewind_verify: accepted {accepted} of {verified}"));
    }
    if replays.len() != cache.gdn_caches.len() {
        return Err(format!(
            "rewind_verify: {} GDN replays for {} GDN layers",
            replays.len(),
            cache.gdn_caches.len()
        ));
    }
    rollback_cache(cache, verified - accepted);
    for (layer, replay) in cache.gdn_caches.iter_mut().zip(replays) {
        let block = replay.keys.dim(1);
        if block != verified {
            return Err(format!(
                "rewind_verify: a replay holds {block} tokens, the verify {verified}"
            ));
        }
        let prefix = |x: &InlineArray| {
            let mut stop = x.shape().to_vec();
            stop[1] = accepted;
            x.slice(&vec![0; stop.len()], &stop)
        };
        layer.ssm_state = Some(if accepted == 0 {
            replay.initial_state.clone()
        } else {
            InlineArray::gdn_metal_step(
                &prefix(&replay.queries),
                &prefix(&replay.keys),
                &prefix(&replay.values),
                &prefix(&replay.g),
                &prefix(&replay.beta),
                &replay.initial_state,
                accepted,
            )
            .1
        });
        let (b, cd) = (replay.conv_input.dim(0), replay.conv_input.dim(2));
        layer.conv_state = Some(replay.conv_input.slice(
            &[0, accepted, 0],
            &[b, accepted + replay.conv_kernel - 1, cd],
        ));
    }
    Ok(())
}
