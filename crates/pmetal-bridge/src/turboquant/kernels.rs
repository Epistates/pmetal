//! Safe wrappers over the TurboQuant Metal kernels.
//!
//! Each wrapper checks the bridge's error channel after its call and returns
//! the failure instead of a placeholder array.

use std::mem::MaybeUninit;

use crate::InlineArray;
use crate::error::{BridgeResult, check_last_error};
use crate::inline_array::RawBuf;

bridge_extern! {
    pub(super) fn mlx_inline_turboquant_quantize_pack(
        out_packed: *mut RawBuf,
        out_scales: *mut RawBuf,
        rotated: *const RawBuf,
        thresholds: *const RawBuf,
        codebook: *const RawBuf,
        dim: u32,
        bits: u32,
        n_rows: u32,
    ) -> i32;
    pub(super) fn mlx_inline_turboquant_dequantize(
        out: *mut RawBuf,
        packed: *const RawBuf,
        scales: *const RawBuf,
        codebook: *const RawBuf,
        dim: u32,
        bits: u32,
        n_rows: u32,
    ) -> i32;
    pub(super) fn mlx_inline_turboquant_attention_partials(
        out_acc: *mut RawBuf,
        out_m: *mut RawBuf,
        out_l: *mut RawBuf,
        out_row_shift: *mut RawBuf,
        q: *const RawBuf,
        q_rot: *const RawBuf,
        k_packed: *const RawBuf,
        k_scale: *const RawBuf,
        v_packed: *const RawBuf,
        v_scale: *const RawBuf,
        cb_k: *const RawBuf,
        cb_v: *const RawBuf,
        dense_k: *const RawBuf,
        dense_v: *const RawBuf,
        params: *const RawBuf,
        center: *const RawBuf,
        dim: u32,
        value_dim: u32,
        key_bits: u32,
        value_bits: u32,
        kv_heads: u32,
        groups: u32,
        n_rows: u32,
        n_kv_rows: u32,
        n_blocks: u32,
    ) -> i32;
    pub(super) fn mlx_inline_turboquant_attention_merge(
        out: *mut RawBuf,
        part_acc: *const RawBuf,
        part_m: *const RawBuf,
        part_l: *const RawBuf,
        inv_rot: *const RawBuf,
        sinks: *const RawBuf,
        row_shift: *const RawBuf,
        params: *const RawBuf,
        value_dim: u32,
        q_heads: u32,
        n_rows: u32,
        n_blocks: u32,
        out_dtype: i32,
    ) -> i32;
}

/// Index of each field in the kernels' `params` array. The attention kernels
/// read everything that changes from one decode step to the next from this
/// int32 array, never from a template argument, which would compile a new
/// kernel per value.
pub(super) mod param {
    pub(in crate::turboquant) const COLD_TOKENS: usize = 0;
    pub(in crate::turboquant) const SINK_TOKENS: usize = 1;
    pub(in crate::turboquant) const RING_CAPACITY: usize = 2;
    pub(in crate::turboquant) const RING_START: usize = 3;
    pub(in crate::turboquant) const RING_LENGTH: usize = 4;
    pub(in crate::turboquant) const TOKENS_PER_BLOCK: usize = 5;
    pub(in crate::turboquant) const COLD_BLOCKS: usize = 6;
    pub(in crate::turboquant) const SCALE_BITS: usize = 7;
    pub(in crate::turboquant) const COLD_CAPACITY: usize = 8;
    pub(in crate::turboquant) const DENSE_CAPACITY: usize = 9;
    pub(in crate::turboquant) const TOTAL_BLOCKS: usize = 10;
    pub(in crate::turboquant) const HAS_SINK_LOGITS: usize = 11;
    pub(in crate::turboquant) const COLD_FIRST: usize = 12;
    pub(in crate::turboquant) const DENSE_FIRST: usize = 13;
    pub(in crate::turboquant) const HAS_CENTER: usize = 14;
    pub(in crate::turboquant) const COUNT: usize = 15;
}

fn uninit() -> MaybeUninit<RawBuf> {
    MaybeUninit::uninit()
}

/// Normalise, quantize and pack `rotated` (`[N, dim]` f32 rows that are
/// `x · Πᵀ`). Returns `(packed [N, words] u32, scales [N] f32)`, where a
/// row's scale is its norm times the least-squares gain of its
/// reconstruction.
pub(super) fn quantize_pack(
    rotated: &InlineArray,
    thresholds: &InlineArray,
    codebook: &InlineArray,
    dim: usize,
    bits: u8,
    n_rows: usize,
) -> BridgeResult<(InlineArray, InlineArray)> {
    let mut packed = uninit();
    let mut scales = uninit();
    // SAFETY: both outputs are written (a value or a placeholder) before the
    // call returns, and the inputs outlive it.
    unsafe {
        mlx_inline_turboquant_quantize_pack(
            packed.as_mut_ptr(),
            scales.as_mut_ptr(),
            &rotated.raw,
            &thresholds.raw,
            &codebook.raw,
            dim as u32,
            u32::from(bits),
            n_rows as u32,
        );
        let out = (
            InlineArray {
                raw: packed.assume_init(),
            },
            InlineArray {
                raw: scales.assume_init(),
            },
        );
        check_last_error()?;
        Ok(out)
    }
}

/// `scale · codebook[index]` for every coordinate of every packed row:
/// `[N, dim]` f32 in the rotated basis.
pub(super) fn dequantize(
    packed: &InlineArray,
    scales: &InlineArray,
    codebook: &InlineArray,
    dim: usize,
    bits: u8,
    n_rows: usize,
) -> BridgeResult<InlineArray> {
    let mut out = uninit();
    // SAFETY: as in `quantize_pack`.
    unsafe {
        mlx_inline_turboquant_dequantize(
            out.as_mut_ptr(),
            &packed.raw,
            &scales.raw,
            &codebook.raw,
            dim as u32,
            u32::from(bits),
            n_rows as u32,
        );
        let out = InlineArray {
            raw: out.assume_init(),
        };
        check_last_error()?;
        Ok(out)
    }
}

/// Shapes and operands of one attention-partials launch.
pub(super) struct PartialsArgs<'a> {
    pub(super) q: &'a InlineArray,
    pub(super) q_rot: &'a InlineArray,
    pub(super) k_packed: &'a InlineArray,
    pub(super) k_scale: &'a InlineArray,
    pub(super) v_packed: &'a InlineArray,
    pub(super) v_scale: &'a InlineArray,
    pub(super) cb_k: &'a InlineArray,
    pub(super) cb_v: &'a InlineArray,
    pub(super) dense_k: &'a InlineArray,
    pub(super) dense_v: &'a InlineArray,
    pub(super) params: &'a InlineArray,
    /// `[KV, D]` f32 key centre; read only when `params[HAS_CENTER]` is set.
    pub(super) center: &'a InlineArray,
    pub(super) dim: usize,
    pub(super) value_dim: usize,
    pub(super) key_bits: u8,
    pub(super) value_bits: u8,
    pub(super) kv_heads: usize,
    pub(super) groups: usize,
    pub(super) n_rows: usize,
    pub(super) n_kv_rows: usize,
    pub(super) n_blocks: usize,
}

/// What pass 1 produces.
pub(super) struct Partials {
    /// Per block, the unnormalised accumulator `[R, B, Dv]`.
    pub(super) acc: InlineArray,
    /// Per block, the running max `[R, B]` (base 2).
    pub(super) m: InlineArray,
    /// Per block, the running sum `[R, B]`.
    pub(super) l: InlineArray,
    /// Per row, `q . mu` scaled into the base-2 logit domain, `[R]`.
    pub(super) row_shift: InlineArray,
}

/// Pass 1: per-block partial attention.
pub(super) fn attention_partials(a: &PartialsArgs<'_>) -> BridgeResult<Partials> {
    let mut acc = uninit();
    let mut m = uninit();
    let mut l = uninit();
    let mut row_shift = uninit();
    // SAFETY: as in `quantize_pack`.
    unsafe {
        mlx_inline_turboquant_attention_partials(
            acc.as_mut_ptr(),
            m.as_mut_ptr(),
            l.as_mut_ptr(),
            row_shift.as_mut_ptr(),
            &a.q.raw,
            &a.q_rot.raw,
            &a.k_packed.raw,
            &a.k_scale.raw,
            &a.v_packed.raw,
            &a.v_scale.raw,
            &a.cb_k.raw,
            &a.cb_v.raw,
            &a.dense_k.raw,
            &a.dense_v.raw,
            &a.params.raw,
            &a.center.raw,
            a.dim as u32,
            a.value_dim as u32,
            u32::from(a.key_bits),
            u32::from(a.value_bits),
            a.kv_heads as u32,
            a.groups as u32,
            a.n_rows as u32,
            a.n_kv_rows as u32,
            a.n_blocks as u32,
        );
        let out = Partials {
            acc: InlineArray {
                raw: acc.assume_init(),
            },
            m: InlineArray {
                raw: m.assume_init(),
            },
            l: InlineArray {
                raw: l.assume_init(),
            },
            row_shift: InlineArray {
                raw: row_shift.assume_init(),
            },
        };
        check_last_error()?;
        Ok(out)
    }
}

/// Pass 2: merge the blocks of every row, rotate the cold part back with
/// `inv_rot` (`Π`, `[Dv, Dv]` f32), add the dense part, normalise.
/// Returns `[R, Dv]` in `out_dtype`.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_merge(
    partials: &Partials,
    inv_rot: &InlineArray,
    sinks: &InlineArray,
    params: &InlineArray,
    value_dim: usize,
    q_heads: usize,
    n_rows: usize,
    n_blocks: usize,
    out_dtype: i32,
) -> BridgeResult<InlineArray> {
    let mut out = uninit();
    // SAFETY: as in `quantize_pack`.
    unsafe {
        mlx_inline_turboquant_attention_merge(
            out.as_mut_ptr(),
            &partials.acc.raw,
            &partials.m.raw,
            &partials.l.raw,
            &inv_rot.raw,
            &sinks.raw,
            &partials.row_shift.raw,
            &params.raw,
            value_dim as u32,
            q_heads as u32,
            n_rows as u32,
            n_blocks as u32,
            out_dtype,
        );
        let out = InlineArray {
            raw: out.assume_init(),
        };
        check_last_error()?;
        Ok(out)
    }
}
