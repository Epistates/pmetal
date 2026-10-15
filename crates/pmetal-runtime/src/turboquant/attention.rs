//! Decode-step attention over the compressed cold store plus the dense tail.
//!
//! This builds the launch for the two kernels in
//! `cpp/bridge_turboquant_attention.cpp`; see that file for the algorithm and
//! the meaning of every `params` field.

use crate::InlineArray;
use crate::compat::Dtype;

use super::kernels::{self, PartialsArgs, param};
use super::quantizer::Quantizer;
use super::store::{ColdStore, DenseStore};

/// Tokens per block are rounded to a multiple of this (one simdgroup tile).
const TILE: usize = 32;
/// Fewest and most tokens one block covers.
const MIN_TOKENS_PER_BLOCK: usize = 128;
const MAX_TOKENS_PER_BLOCK: usize = 2048;
/// Aim for this many blocks over all KV heads, so the GPU stays busy at
/// moderate context without drowning the merge at long context.
const TARGET_BLOCKS: usize = 1024;

/// How the tokens are divided among threadgroups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlockPlan {
    pub(super) tokens_per_block: usize,
    pub(super) cold_blocks: usize,
    pub(super) total_blocks: usize,
}

/// Divide `cold` compressed and `dense` uncompressed tokens of `kv_rows` KV
/// heads into blocks. Both segments use the same block length, and the cold
/// blocks come first.
pub(super) fn plan_blocks(cold: usize, dense: usize, kv_rows: usize) -> BlockPlan {
    let total = cold + dense;
    let wanted = total.div_ceil((TARGET_BLOCKS / kv_rows.max(1)).max(1));
    let tokens_per_block = wanted
        .clamp(MIN_TOKENS_PER_BLOCK, MAX_TOKENS_PER_BLOCK)
        .div_ceil(TILE)
        * TILE;
    let cold_blocks = cold.div_ceil(tokens_per_block);
    BlockPlan {
        tokens_per_block,
        cold_blocks,
        total_blocks: cold_blocks + dense.div_ceil(tokens_per_block),
    }
}

/// Everything one decode-step attention needs besides the cache arrays.
#[derive(Debug, Clone, Copy)]
pub(super) struct DecodeShape {
    pub(super) batch: usize,
    pub(super) q_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) dim: usize,
    pub(super) value_dim: usize,
}

/// Which tokens a decode step attends to, and what joins the softmax besides.
#[derive(Debug, Clone, Copy)]
pub(super) struct Visibility<'a> {
    /// Absolute position of the first token attended to; everything before it
    /// is masked (a sliding window, a chunk).
    pub(super) first_token: usize,
    /// One logit per query head that joins the softmax normaliser without
    /// contributing a value (gpt-oss attention sinks).
    pub(super) sink_logits: Option<&'a InlineArray>,
}

/// The first cold token and the first dense token a step attends to, given
/// that the sequence is `sinks`, then `cold`, then the ring, and that
/// everything before absolute position `first_token` is masked.
fn first_visible(
    first_token: usize,
    sink_len: usize,
    cold_len: usize,
    dense_len: usize,
) -> (usize, usize) {
    let cold_first = first_token.saturating_sub(sink_len).min(cold_len);
    let dense_first = if first_token < sink_len {
        first_token
    } else {
        sink_len + first_token.saturating_sub(sink_len + cold_len)
    };
    (cold_first, dense_first.min(dense_len))
}

/// Threadgroup memory a kernel launch may use.
const THREADGROUP_MEMORY: usize = 32 * 1024;
/// Query heads one launch serves per KV head: a lane keeps `groups` running
/// accumulators of `ceil(Dv / 32)` values, and the registers of a lane cap how
/// many before the compiler spills.
const REGISTER_BUDGET: usize = 128;
const MAX_GROUPS: usize = 16;

/// The most query heads per KV head one launch can serve for these shapes.
pub(super) fn max_groups(dim: usize, value_dim: usize, key_bits: u8, value_bits: u8) -> usize {
    let codebooks = (1usize << key_bits) + (1usize << value_bits);
    let by_memory = (THREADGROUP_MEMORY / 4).saturating_sub(codebooks) / dim;
    let by_registers = REGISTER_BUDGET / (value_dim.div_ceil(32) + 4);
    by_memory.min(by_registers).clamp(1, MAX_GROUPS)
}

/// Attention of one query token per row over `cold` and `dense`.
///
/// `queries` is `[B, Hq, 1, D]` in any float dtype; the result is
/// `[B, Hq, 1, Dv]` in the same dtype. When a KV head serves more query heads
/// than a launch can (many query heads per KV head at a wide head dimension),
/// the query heads are split into equal groups, one launch each: the cache is
/// read once per group.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_attention(
    queries: &InlineArray,
    scale: f32,
    visibility: Visibility<'_>,
    shape: &DecodeShape,
    keys: &Quantizer,
    values: &Quantizer,
    cold: Option<&ColdStore>,
    dense: &DenseStore,
) -> Result<InlineArray, String> {
    let DecodeShape {
        batch,
        q_heads,
        kv_heads,
        dim,
        value_dim,
    } = *shape;
    if kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(format!(
            "TurboQuant: {q_heads} query heads is not a multiple of {kv_heads} KV heads"
        ));
    }
    let groups = q_heads / kv_heads;
    let cap = max_groups(dim, value_dim, keys.bits(), values.bits());
    if groups <= cap {
        return decode_attention_launch(
            queries, scale, visibility, shape, keys, values, cold, dense,
        );
    }
    // The largest divisor of `groups` that fits.
    let per_launch = (1..=cap).rev().find(|g| groups % g == 0).unwrap_or(1);
    let sub_shape = DecodeShape {
        q_heads: kv_heads * per_launch,
        ..*shape
    };
    let grouped_queries =
        queries.reshape(&[batch as i32, kv_heads as i32, groups as i32, dim as i32]);
    let mut outputs: Option<InlineArray> = None;
    for part in 0..groups / per_launch {
        let lo = (part * per_launch) as i32;
        let hi = ((part + 1) * per_launch) as i32;
        let sub_queries = grouped_queries
            .slice(
                &[0, 0, lo, 0],
                &[batch as i32, kv_heads as i32, hi, dim as i32],
            )
            .reshape(&[batch as i32, (kv_heads * per_launch) as i32, 1, dim as i32]);
        let sub_sinks = visibility.sink_logits.map(|s| {
            s.reshape(&[kv_heads as i32, groups as i32])
                .slice(&[0, lo], &[kv_heads as i32, hi])
                .reshape(&[(kv_heads * per_launch) as i32])
        });
        let out = decode_attention_launch(
            &sub_queries,
            scale,
            Visibility {
                first_token: visibility.first_token,
                sink_logits: sub_sinks.as_ref(),
            },
            &sub_shape,
            keys,
            values,
            cold,
            dense,
        )?
        .reshape(&[
            batch as i32,
            kv_heads as i32,
            per_launch as i32,
            value_dim as i32,
        ]);
        outputs = Some(match outputs {
            None => out,
            Some(so_far) => so_far.concatenate_2(&out, 2),
        });
    }
    Ok(outputs.expect("at least one launch").reshape(&[
        batch as i32,
        q_heads as i32,
        1,
        value_dim as i32,
    ]))
}

/// One launch: `Hq / Hkv` must not exceed [`max_groups`].
#[allow(clippy::too_many_arguments)]
fn decode_attention_launch(
    queries: &InlineArray,
    scale: f32,
    visibility: Visibility<'_>,
    shape: &DecodeShape,
    keys: &Quantizer,
    values: &Quantizer,
    cold: Option<&ColdStore>,
    dense: &DenseStore,
) -> Result<InlineArray, String> {
    let Visibility {
        first_token,
        sink_logits: sinks,
    } = visibility;
    let DecodeShape {
        batch,
        q_heads,
        kv_heads,
        dim,
        value_dim,
    } = *shape;
    if kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(format!(
            "TurboQuant: {q_heads} query heads is not a multiple of {kv_heads} KV heads"
        ));
    }
    let groups = q_heads / kv_heads;
    let n_rows = batch * q_heads;
    let n_kv_rows = batch * kv_heads;
    let cold_tokens = cold.map_or(0, |c| c.len);
    let dense_tokens = dense.tokens();
    let (cold_first, dense_first) =
        first_visible(first_token, dense.sink_len, cold_tokens, dense_tokens);
    let visible_cold = cold_tokens - cold_first;
    let visible_dense = dense_tokens - dense_first;
    if visible_cold + visible_dense == 0 {
        return Err(format!(
            "TurboQuant: no cached token to attend to ({} cached, attending from position {first_token})",
            cold_tokens + dense_tokens
        ));
    }
    let plan = plan_blocks(visible_cold, visible_dense, n_kv_rows);

    let q_rows = queries.reshape(&[n_rows as i32, dim as i32]);
    // Cold keys were divided channel by channel by `scale`; multiplying the
    // query by it leaves every logit as it was.
    let q_cold = match cold
        .and_then(|c| c.frame.as_ref())
        .and_then(|f| f.scale.as_ref())
    {
        Some(scale) => q_rows
            .as_dtype(Dtype::Float32.as_i32())
            .reshape(&[n_kv_rows as i32, groups as i32, dim as i32])
            .multiply(&scale.expand_dims(1))
            .reshape(&[n_rows as i32, dim as i32]),
        None => q_rows.clone(),
    };
    let q_rot = keys.rotate(&q_cold);

    let placeholder;
    let cold_arrays = match cold {
        Some(c) => c,
        None => {
            placeholder = ColdStore::empty(n_kv_rows, keys, values);
            &placeholder
        }
    };

    let mut params = [0i32; param::COUNT];
    params[param::COLD_TOKENS] = cold_tokens as i32;
    params[param::SINK_TOKENS] = dense.sink_len as i32;
    params[param::RING_CAPACITY] = dense.ring_capacity.max(1) as i32;
    params[param::RING_START] = dense.ring_start as i32;
    params[param::RING_LENGTH] = dense.ring_len as i32;
    params[param::TOKENS_PER_BLOCK] = plan.tokens_per_block as i32;
    params[param::COLD_BLOCKS] = plan.cold_blocks as i32;
    params[param::SCALE_BITS] = (scale * std::f32::consts::LOG2_E).to_bits() as i32;
    params[param::COLD_CAPACITY] = cold_arrays.capacity as i32;
    params[param::DENSE_CAPACITY] = dense.capacity as i32;
    params[param::TOTAL_BLOCKS] = plan.total_blocks as i32;
    params[param::HAS_SINK_LOGITS] = i32::from(sinks.is_some());
    params[param::COLD_FIRST] = cold_first as i32;
    params[param::DENSE_FIRST] = dense_first as i32;
    // The centre the cold keys were stored with. A cold store that has one
    // always has it set before it holds a token.
    let frame = cold.and_then(|c| c.frame.as_ref());
    let centre = frame.map(|f| &f.center);
    params[param::HAS_CENTER] = i32::from(centre.is_some());
    let params = InlineArray::from_i32_slice(&params);
    let no_centre;
    let centre = match centre {
        Some(c) => c,
        None => {
            no_centre = InlineArray::zeros(&[1], Dtype::Float32.as_i32());
            &no_centre
        }
    };

    let no_sinks;
    let sink_logits = match sinks {
        Some(s) => s,
        None => {
            no_sinks = InlineArray::zeros(&[1], Dtype::Float32.as_i32());
            &no_sinks
        }
    };

    let partials = kernels::attention_partials(&PartialsArgs {
        q: &q_rows,
        q_rot: &q_rot,
        k_packed: &cold_arrays.k.packed,
        k_scale: &cold_arrays.k.scales,
        v_packed: &cold_arrays.v.packed,
        v_scale: &cold_arrays.v.scales,
        cb_k: keys.codebook_array(),
        cb_v: values.codebook_array(),
        dense_k: &dense.keys,
        dense_v: &dense.values,
        params: &params,
        center: centre,
        dim,
        value_dim,
        key_bits: keys.bits(),
        value_bits: values.bits(),
        kv_heads,
        groups,
        n_rows,
        n_kv_rows,
        n_blocks: plan.total_blocks,
    })
    .map_err(|e| e.to_string())?;

    let out = kernels::attention_merge(
        &partials,
        values.rotation_array(),
        sink_logits,
        &params,
        value_dim,
        q_heads,
        n_rows,
        plan.total_blocks,
        queries.dtype_raw(),
    )
    .map_err(|e| e.to_string())?;
    Ok(out.reshape(&[batch as i32, q_heads as i32, 1, value_dim as i32]))
}
