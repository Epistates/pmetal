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

/// Whether layer `layer_idx` of a Qwen-style model attends within a sliding
/// window, as `transformers` decides it.
///
/// The window exists only with `use_sliding_window` (the config drops
/// `sliding_window` otherwise). Then a `layer_types` entry names the layer's
/// type, `"sliding_attention"` or `"full_attention"`; without one the layers
/// from `max_window_layers` (28 unless the config says) up are sliding.
pub fn layer_is_sliding(
    use_sliding_window: bool,
    sliding_window: Option<i32>,
    max_window_layers: Option<i32>,
    layer_types: Option<&[String]>,
    layer_idx: usize,
) -> bool {
    if !use_sliding_window || sliding_window.is_none() {
        return false;
    }
    match layer_types.and_then(|types| types.get(layer_idx)) {
        Some(layer_type) => layer_type == "sliding_attention",
        None => layer_idx as i32 >= max_window_layers.unwrap_or(28),
    }
}
