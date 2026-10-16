//! The mask a sliding-window layer attends under.
//!
//! A layer given no mask asks the attention kernel for its window
//! (`AttentionMaskType::SlidingWindow`). A layer given one (packed sequences'
//! block-diagonal mask, a padding mask) hands the kernel that mask instead, and
//! the kernel attends as far back as it allows, which for a causal mask is the
//! whole sequence. [`windowed_mask`] closes that gap for every local layer.

use pmetal_runtime::compat::{Array, Exception};
use pmetal_runtime::nn::restrict_to_sliding_window;

/// The caller's `mask` restricted to the causal band of `window` keys, for a
/// layer whose attention is local; `None` when there is nothing to restrict (no
/// caller mask, or a layer that attends globally).
///
/// A caller's mask says which sequence a token belongs to, not how far back a
/// local layer reaches, so a sliding-window model trained under it would attend
/// past its window inside every sequence longer than the window: a different
/// model from the one served. Use as
///
/// ```ignore
/// let windowed = windowed_mask(mask, self.local_window(), q_len, keys.dim(2))?;
/// let mask = windowed.as_ref().or(mask);
/// ```
pub fn windowed_mask(
    mask: Option<&Array>,
    window: Option<i32>,
    query_len: i32,
    key_len: i32,
) -> Result<Option<Array>, Exception> {
    match (mask, window) {
        (Some(mask), Some(window)) => {
            restrict_to_sliding_window(mask, query_len, key_len, window).map(Some)
        }
        _ => Ok(None),
    }
}
