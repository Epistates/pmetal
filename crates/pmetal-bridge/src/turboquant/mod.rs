//! TurboQuant KV cache — zero mlx-rs dependency.
//!
//! Self-contained implementation of the TurboQuant-inspired KV cache using only
//! [`InlineArray`] and pure-Rust math.  The module is intentionally free of any
//! mlx-rs or pmetal-metal imports; all GPU work is driven through
//! `InlineArray::matmul` which dispatches to MLX's Metal backend automatically.
//!
//! ## Storage layout invariants (audit-pinned 2026-04-25)
//!
//! Pack/encode/decode/score MUST agree on these axis orders. This contract is
//! mirrored in `cpp/bridge/turboquant.h`'s top-of-file comment; keep both in
//! sync. The [`tests::layout_invariants`] tests pin the derived dimensions
//! (e.g. [`bits::packed_qjl_words`]) so a silent off-by-one in one of the
//! helpers will fail CI rather than corrupt cache reads.
//!
//! Live cache layouts (read by score kernels):
//!
//! * `indices`        — `[N, D, S_cap]` `u8` (7-bit centroid id)
//! * `qjl_signs`      — `[N, ceil(D/32), S_cap]` `u32` (packed sign bits)
//! * `norms`          — `[N, S_cap]` `f32`
//! * `residual_norms` — `[N, S_cap]` `f32`
//! * `codebook`       — `[C]` `f32` (sorted Lloyd-Max centroids)
//!
//! q8 fullbyte (D=256, n_seq>=1024) uses a transposed seq-major shadow:
//!
//! * `q8_fullbyte_seq`     — `[N, S_cap, D]` `u8`
//! * `q8_slot_scales_seq`  — `[N, S_cap]` `f16`
//! * `d256_rot_values_seq` — `[N, S_cap, D]` `bf16`
//!
//! GPU-attention coverage:
//!
//! * Uniform + head_dim==128 → fused d128 attention kernel
//! * Uniform + head_dim==256 → fused d256 attention kernel
//!   (+ q8 fullbyte specialisation when bits==8 and n_seq >= 1024)
//! * Uniform + other head_dim → dequantize + SDPA fallback
//! * Mixed precision → dequantize + SDPA fallback (no GPU kernel today)
//!
//! # Algorithm overview
//!
//! **Keys** (inner-product optimised):
//!   1. Normalise each vector onto the unit sphere; store the L2 norm.
//!   2. Apply the orthogonal rotation Π: `r = Π · k_norm`.
//!   3. Nearest-centroid scalar quantisation of every coordinate using the
//!      Lloyd-Max codebook for the Beta distribution (MSE at `b-1` bits).
//!   4. Compute the residual `e = k_norm - Π^T · codebook[idx]` and project it
//!      through a Gaussian matrix J: sign(J · e) gives 1-bit QJL signs.
//!
//! **Values** (MSE optimised, no QJL stage):
//!   1. Normalise + store norm.
//!   2. Rotate then quantise with the full `b`-bit codebook.
//!
//! **Outlier-aware mixed-bit** (optional):
//!   Per-row, the top-`k` coordinates by magnitude are flagged as "outliers"
//!   and stored at a higher bit-width in a separate sub-vector.
//!
//! # Module layout (post-Phase 0 split)
//!
//! - [`config`] — public config types ([`TurboQuantConfig`], [`TurboQuantTensorConfig`]).
//! - [`core`] — [`TurboQuantCore`] (shared rotation matrices + codebooks).
//! - [`state`] — [`TurboQuantState`] (per-tensor runtime tables).
//! - [`bits`] — [`PackedBits`] bit-packing primitive used by the host stores.
//! - [`math`] — pure math helpers (Beta codebook, FWHT, Rademacher signs).
//! - [`gpu_keystore`] — GPU-resident `Gpu*KeyStore` / `Gpu*ValueStore`.
//! - [`host_keystore`] — host-side [`QuantizedKeyStore`] / [`QuantizedValueStore`].
//! - [`encode`] — CPU encode/decode primitives + outlier helpers.
//! - [`dispatch`] — GPU quantise/dequantise kernel glue.
//! - [`cache`] — [`QuantizedKvCache`] state machine (impl of public API).
//! - [`tests`] — round-trip + layout-invariant tests (`#[cfg(test)]`).
//!
//! # What is NOT in this module
//!
//! - The mlx-rs `Array` integration code.
//! - The `TurboQuantKvCache` struct (see `KvLayerCache` in qwen3_native).
//! - The pmetal-metal `TurboQuantTransform` (InlineArray.matmul replaces it).

use std::sync::Arc;
use std::time::Instant;

use crate::InlineArray;

// ── Constants ────────────────────────────────────────────────────────────────

/// Deterministic seed — same as the mlx-rs reference implementation.
const TURBOQUANT_SEED: u64 = 0x5442_5155_414e_544d;
/// Vectors with L2 norm below this are treated as zero.
pub(super) const ZERO_EPSILON: f32 = 1e-12;
/// Defensive upper bound on encoded residual L2 norms, used to prevent Inf/NaN
/// from upstream fp16 corruption from reaching the QJL term in the score and
/// attention kernels. Derived from `||k_rot||=1` + triangle inequality plus a
/// conservative margin over the Beta-codebook reconstruction norm; realistic
/// values are below 1.0 for any bit-width b≥2. Any residual norm above this
/// cap would already violate Theorem 2's distortion bound — clipping is safe.
pub(super) const MAX_RESIDUAL_NORM: f32 = 4.0;

pub(super) fn turboquant_trace_enabled() -> bool {
    std::env::var_os("PMETAL_TRACE_TURBOQUANT").is_some()
}

fn turboquant_wht_enabled() -> bool {
    std::env::var_os("PMETAL_TQ_USE_WHT")
        .map(|value| value != "0")
        .unwrap_or(true)
}

/// Returns true when the active dim should use signed-FWHT instead of dense
/// `[d×d]` matmul for rotation and QJL projection.
///
/// FWHT is `O(d log d)`; dense matmul is `O(d²)`. For every transformer KV
/// head_dim in current use (32, 64, 96, 128, 192, 256, 320…) the FWHT
/// alternative wins on both compute and memory. We restrict to power-of-two
/// dims ≥ 4 because the underlying Walsh-Hadamard kernel requires it; non-pow2
/// (192, 80, etc.) keep the dense path.
pub(super) fn dim_uses_fwht(dim: usize) -> bool {
    turboquant_wht_enabled() && dim >= 4 && dim.is_power_of_two()
}

pub(super) fn turboquant_q8_fullbyte_enabled() -> bool {
    std::env::var_os("PMETAL_TQ_Q8_FULLBYTE")
        .map(|value| value != "0")
        .unwrap_or(false)
}

pub(super) fn trace_turboquant_bridge(message: &str) {
    if turboquant_trace_enabled() {
        eprintln!("[TURBOQUANT TRACE][BRIDGE] {message}");
    }
}

pub(super) fn eval_stage_micros(array: &InlineArray) -> u128 {
    let start = Instant::now();
    array.eval();
    crate::inline_array::synchronize();
    start.elapsed().as_micros()
}

mod config;
pub use config::{DEFAULT_RECENT_WINDOW, TurboQuantConfig, TurboQuantTensorConfig};

mod core;
pub use core::TurboQuantCore;

mod state;
pub use state::TurboQuantState;

mod bits;
pub use bits::PackedBits;

mod math;
pub use math::{beta_codebook, generate_rademacher_signs, signed_fwht_forward};
pub(crate) use math::{generate_random_orthogonal, generate_random_projection};

mod gpu_keystore;
mod host_keystore;
pub use host_keystore::{QuantizedKeyStore, QuantizedValueStore};

mod encode;

mod dispatch;

mod cache;
pub use cache::{QuantizedKvCache, UniformAttentionBenchMode};

#[cfg(test)]
mod tests;

// ═══════════════════════════════════════════════════════════════════════════
// Public convenience constructors
// ═══════════════════════════════════════════════════════════════════════════

/// Build a shared [`TurboQuantState`] for the given dimensions and config.
///
/// This is the expensive step (~100 ms per unique dim).  Call once at model
/// load time and share the `Arc` across all layers.
///
/// Not every `(config, dim)` combination has a fused GPU attention path; see
/// [`TurboQuantState::has_gpu_attention_support`] for the full predicate.
/// Mixed-precision configs (`outlier_count > 0`) currently fall back to
/// dequantize+SDPA at attention time. We log a warning at state-build time
/// when that fallback will be hit so the cost is visible during model load
/// rather than being discovered as a perf regression at first decode.
pub fn build_state(
    key_dim: usize,
    value_dim: usize,
    config: TurboQuantConfig,
) -> Arc<TurboQuantState> {
    let state = Arc::new(TurboQuantState::new(key_dim, value_dim, config));
    if !state.has_gpu_attention_support() {
        eprintln!(
            "[turboquant] WARNING: state for key_dim={key_dim} value_dim={value_dim} \
             will fall back to dequantize+SDPA at attention time. \
             Reason: {}",
            state.gpu_attention_unsupported_reason()
        );
    }
    state
}

/// Look up the dispatch reason for a particular state — returned to callers
/// who want to surface "your config is on the slow path" diagnostics in
/// model-load logs without re-implementing the predicate.
pub fn gpu_attention_unsupported_reason_for(
    state: &TurboQuantState,
) -> Option<&'static str> {
    if state.has_gpu_attention_support() {
        None
    } else {
        Some(state.gpu_attention_unsupported_reason())
    }
}

/// Create a [`QuantizedKvCache`] that will lazily build its state on first use.
pub fn new_cache(config: TurboQuantConfig) -> QuantizedKvCache {
    QuantizedKvCache::new(config)
}

/// Create a [`QuantizedKvCache`] using a pre-built shared [`TurboQuantState`].
pub fn new_cache_with_state(
    config: TurboQuantConfig,
    state: Arc<TurboQuantState>,
) -> QuantizedKvCache {
    QuantizedKvCache::with_state(config, state)
}
