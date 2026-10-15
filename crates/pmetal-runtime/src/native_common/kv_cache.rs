//! KV cache allocation and growth primitives shared by the native
//! architectures.
//!
//! ## Growth policy
//!
//! Every native architecture grew its KV cache slightly differently before
//! this module landed:
//!
//! * `qwen3_native` rounded `next` up to the next 256-token chunk and grew
//!   to that target in a single `kv_cache_append`.
//! * `gpt_oss_native`, `llama4_native`, and `deepseek_native` grew by a
//!   *fixed* 256-token chunk per call, which silently undersized the buffer
//!   when a single forward pass needed more than 256 new tokens (e.g. a
//!   long-context prefill following a short warmup).
//!
//! The shared policy implemented here:
//!
//! 1. Round `next` up to the nearest [`CHUNK_TOKENS`]-token chunk to amortise
//!    MLX graph-node bookkeeping over multi-token prefills and steady-state
//!    decode.
//! 2. On growth, take `max(allocated * 2, round_up(next))` so a long prefill
//!    triggers at most `O(log(N))` `kv_cache_append` calls instead of `O(N)`
//!    — a measurable wall-clock win on llama4 / gpt_oss long-context prefill
//!    and a structural fix for the 256-cap bug above.
//!
//! All helpers operate on `Option<InlineArray>` slots in place; callers never
//! see the intermediate `take`/replace dance.

use crate::InlineArray;

/// Capacity-rounding chunk in tokens. Matches the historical qwen3 alloc
/// stride. Exported so call sites can audit the rounding decision and so the
/// constant has exactly one source of truth.
pub const CHUNK_TOKENS: i32 = 256;

/// Round a token count up to the nearest [`CHUNK_TOKENS`]-token boundary.
#[inline]
pub fn round_up_to_chunk(n: i32) -> i32 {
    debug_assert!(n >= 0, "round_up_to_chunk: negative token count");
    // i32::div_ceil is unstable on the toolchain pmetal targets, so spell it
    // out: ((n + chunk - 1) / chunk) * chunk.
    ((n + CHUNK_TOKENS - 1) / CHUNK_TOKENS) * CHUNK_TOKENS
}

/// Growth strategy for [`next_capacity`] / [`alloc_or_grow_kv`].
///
/// The native architectures use two distinct call patterns:
///
/// * **AmortizedChunked** — incremental append from inside an attention
///   forward pass. Called many times during a decode/prefill loop; rounding
///   up and doubling on growth keeps `kv_cache_append` count in `O(log N)`.
/// * **Exact** — one-shot reservation when the final size is known up front
///   (e.g. `NativeCache::reserve_decode_inputs` before a bounded generation
///   run). Allocates exactly what was asked for so we don't burn memory on
///   short generations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowthPolicy {
    /// Exact target capacity with no rounding or doubling.
    Exact,
    /// Round `next` up to [`CHUNK_TOKENS`], at least double on growth.
    AmortizedChunked,
}

/// Compute the target capacity for a cache that must hold `next` tokens given
/// `allocated` already in place. See [`GrowthPolicy`] for behaviour.
#[inline]
pub fn next_capacity(policy: GrowthPolicy, allocated: Option<i32>, next: i32) -> Option<i32> {
    match policy {
        GrowthPolicy::Exact => match allocated {
            None => Some(next),
            Some(alloc) if next > alloc => Some(next),
            Some(_) => None,
        },
        GrowthPolicy::AmortizedChunked => {
            let target = round_up_to_chunk(next);
            match allocated {
                None => Some(target),
                Some(alloc) if next > alloc => Some(alloc.saturating_mul(2).max(target)),
                Some(_) => None,
            }
        }
    }
}

/// Allocate or grow a paired `[B, H, T, head_dim]` KV buffer set in place.
///
/// On first call (both `keys` and `values` are `None`) this allocates fresh
/// zero-initialised buffers sized according to [`next_capacity`]. On
/// subsequent calls it grows the buffers in place via `kv_cache_append`
/// along the time axis. No-op if the existing capacity already covers `next`.
///
/// `head_count` is the cache head count (`n_kv_heads` for GQA-aware caches),
/// `head_dim` is the per-head feature dim. `dtype` is the MLX integer dtype
/// code (`crate::compat::Dtype::*.as_i32()`).
#[allow(clippy::too_many_arguments)]
pub fn alloc_or_grow_kv(
    policy: GrowthPolicy,
    keys: &mut Option<InlineArray>,
    values: &mut Option<InlineArray>,
    batch: i32,
    head_count: i32,
    next: i32,
    head_dim: i32,
    dtype: i32,
) {
    let allocated = keys.as_ref().map(|k| k.dim(2));
    let Some(target) = next_capacity(policy, allocated, next) else {
        return;
    };
    match allocated {
        None => {
            *keys = Some(InlineArray::zeros(
                &[batch, head_count, target, head_dim],
                dtype,
            ));
            *values = Some(InlineArray::zeros(
                &[batch, head_count, target, head_dim],
                dtype,
            ));
        }
        Some(alloc) => {
            let extend = target - alloc;
            let old_k = keys.take().expect("keys present when allocated is Some");
            let old_v = values
                .take()
                .expect("values present when allocated is Some");
            let ext_k = InlineArray::zeros(&[batch, head_count, extend, head_dim], dtype);
            let ext_v = InlineArray::zeros(&[batch, head_count, extend, head_dim], dtype);
            *keys = Some(old_k.kv_cache_append(&ext_k, 2));
            *values = Some(old_v.kv_cache_append(&ext_v, 2));
        }
    }
}

/// Allocate or grow a single 4-D buffer along its time axis. `shape_for_cap`
/// returns the full tensor shape given a time-axis capacity (lets callers vary
/// the last dim — e.g. DeepSeek MLA stores `kv_latent` and `k_pe` with
/// different last dims but the same time axis).
pub fn alloc_or_grow_buffer(
    policy: GrowthPolicy,
    buf: &mut Option<InlineArray>,
    next: i32,
    time_axis: i32,
    dtype: i32,
    shape_for_cap: impl Fn(i32) -> [i32; 4],
) {
    let allocated = buf.as_ref().map(|b| b.dim(time_axis));
    let Some(target) = next_capacity(policy, allocated, next) else {
        return;
    };
    match allocated {
        None => {
            *buf = Some(InlineArray::zeros(&shape_for_cap(target), dtype));
        }
        Some(alloc) => {
            let extend = target - alloc;
            let old = buf.take().expect("buf present when allocated is Some");
            let ext = InlineArray::zeros(&shape_for_cap(extend), dtype);
            *buf = Some(old.kv_cache_append(&ext, time_axis));
        }
    }
}

/// Quantized-tuple-aware allocator used by the qwen3 mixed-bit cache. Allocates
/// or grows the (packed, scales, biases) trio that backs an affine-quantized
/// `[B, H, T, last_dim]` buffer. Re-uses [`alloc_or_grow_buffer`] for each leg.
#[allow(clippy::too_many_arguments)]
pub fn alloc_or_grow_quantized_tuple(
    policy: GrowthPolicy,
    packed: &mut Option<InlineArray>,
    scales: &mut Option<InlineArray>,
    biases: &mut Option<InlineArray>,
    batch: i32,
    head_count: i32,
    next: i32,
    packed_last_dim: i32,
    scales_last_dim: i32,
    packed_dtype: i32,
    scales_dtype: i32,
) {
    alloc_or_grow_buffer(policy, packed, next, 2, packed_dtype, |cap| {
        [batch, head_count, cap, packed_last_dim]
    });
    alloc_or_grow_buffer(policy, scales, next, 2, scales_dtype, |cap| {
        [batch, head_count, cap, scales_last_dim]
    });
    alloc_or_grow_buffer(policy, biases, next, 2, scales_dtype, |cap| {
        [batch, head_count, cap, scales_last_dim]
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amortized_initial_round_up() {
        let p = GrowthPolicy::AmortizedChunked;
        assert_eq!(next_capacity(p, None, 0), Some(0));
        assert_eq!(next_capacity(p, None, 1), Some(CHUNK_TOKENS));
        assert_eq!(next_capacity(p, None, CHUNK_TOKENS), Some(CHUNK_TOKENS));
        assert_eq!(
            next_capacity(p, None, CHUNK_TOKENS + 1),
            Some(2 * CHUNK_TOKENS)
        );
    }

    #[test]
    fn amortized_no_growth_needed() {
        let p = GrowthPolicy::AmortizedChunked;
        assert_eq!(next_capacity(p, Some(CHUNK_TOKENS), CHUNK_TOKENS), None);
        assert_eq!(next_capacity(p, Some(1024), 999), None);
    }

    #[test]
    fn amortized_doubles_on_growth() {
        // Existing 1024, need 1025 → max(1024*2, ceil(1025)) = 2048.
        assert_eq!(
            next_capacity(GrowthPolicy::AmortizedChunked, Some(1024), 1025),
            Some(2048)
        );
    }

    #[test]
    fn amortized_jumps_past_double_when_needed() {
        // Existing 256, need 4097 → max(256*2, ceil(4097)) = max(512, 4352) = 4352.
        assert_eq!(
            next_capacity(GrowthPolicy::AmortizedChunked, Some(256), 4097),
            Some(4352)
        );
    }

    #[test]
    fn exact_grows_to_target_no_rounding() {
        let p = GrowthPolicy::Exact;
        assert_eq!(next_capacity(p, None, 17), Some(17));
        assert_eq!(next_capacity(p, Some(16), 17), Some(17));
        assert_eq!(next_capacity(p, Some(17), 17), None);
        assert_eq!(next_capacity(p, Some(20), 17), None);
    }
}
