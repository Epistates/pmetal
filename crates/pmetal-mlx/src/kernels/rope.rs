//! Applying Rotary Position Embedding (RoPE) at contiguous or explicit
//! positions.
//!
//! What a config's RoPE scaling means (linear, dynamic NTK, YaRN, LongRoPE,
//! Llama 3 bands, proportional) lives in one place, [`pmetal_bridge::rope`];
//! an architecture holds a [`RotaryEmbedding`] from there and rotates with
//! [`rope_embedding`]. The helpers here thread [`RopePositions`] (a cache
//! offset, or one position per token for packed batches) through the fused
//! kernel and the explicit cos/sin tables.

use pmetal_bridge::compat::{Array, Dtype, Exception, fast, ops};
use pmetal_bridge::rope::{RotaryEmbedding, rotate_with_cos_sin};

/// Where a rotary embedding takes its positions from.
///
/// [`Offset`] is the contiguous run `offset, offset + 1, …`: every cached
/// decode, and every forward over one unbroken sequence. [`Explicit`] is one
/// position per token, which is what a packed batch needs so the second
/// sequence in a row restarts at 0 instead of continuing the first.
///
/// Architectures thread `Option<&Array>` down from their own forward and call
/// [`RopePositions::resolve`] once against the cache offset, rather than
/// branching separately at each `q` and `k` rotation.
///
/// [`Offset`]: RopePositions::Offset
/// [`Explicit`]: RopePositions::Explicit
#[derive(Debug, Clone, Copy)]
pub enum RopePositions<'a> {
    /// Positions `offset, offset + 1, …, offset + seq_len - 1`.
    Offset(i32),
    /// One position per token, shape `[seq_len]`.
    Explicit(&'a Array),
}

impl<'a> RopePositions<'a> {
    /// The caller's explicit positions when it has them, the contiguous run
    /// from `offset` otherwise.
    pub fn resolve(positions: Option<&'a Array>, offset: i32) -> Self {
        match positions {
            Some(ids) => Self::Explicit(ids),
            None => Self::Offset(offset),
        }
    }
}

/// Apply RoPE at `positions` using a scalar base frequency.
///
/// The contiguous arm is the fused `mx.fast.rope` kernel; the explicit arm
/// builds the cos/sin tables from the position vector. They compute the same
/// rotation, which this module's tests pin down for both `traditional`
/// settings, at a non-zero offset, and for partial RoPE.
pub fn rope(
    x: &Array,
    positions: RopePositions<'_>,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    match positions {
        RopePositions::Offset(offset) => apply_rope(x, dims, traditional, base, scale, offset),
        RopePositions::Explicit(ids) => {
            apply_rope_with_positions(x, ids, dims, traditional, base, scale)
        }
    }
}

/// Rotate `x` at `positions` with a [`RotaryEmbedding`]: whatever scaling its
/// config named, its attention factor included.
pub fn rope_embedding(x: &Array, positions: RopePositions<'_>, rope: &RotaryEmbedding) -> Array {
    match positions {
        RopePositions::Offset(offset) => rope.apply(x, offset),
        RopePositions::Explicit(ids) => rope.apply_at(x, ids),
    }
}

/// [`rope`] with an explicit `[dims / 2]` inverse-frequency table.
///
/// Needed wherever no single `base` describes the rotation: Phi-3 LongRoPE
/// scales each band by its own `long_factor`, and YaRN blends per-band ramps.
pub fn rope_with_inv_freq(
    x: &Array,
    positions: RopePositions<'_>,
    inv_freq: &Array,
    dims: i32,
    traditional: bool,
) -> Result<Array, Exception> {
    match positions {
        RopePositions::Offset(offset) => {
            apply_rope_with_freqs(x, inv_freq, dims, traditional, offset)
        }
        RopePositions::Explicit(ids) => {
            rope_with_positions_and_inv_freq(x, ids, inv_freq, dims, traditional, 1.0)
        }
    }
}

/// [`rope`] with an explicit `[dims / 2]` *period* table, the form
/// `mx.fast.rope` takes through its `freqs=` argument.
///
/// Llama 3's frequency-band scaling is published this way, so the contiguous
/// arm stays on the fused kernel instead of rebuilding the tables per call.
pub fn rope_with_periods(
    x: &Array,
    positions: RopePositions<'_>,
    periods: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
) -> Result<Array, Exception> {
    match positions {
        RopePositions::Offset(offset) => Ok(fast::rope_with_freqs(
            x,
            dims,
            traditional,
            scale,
            offset,
            periods,
        )),
        RopePositions::Explicit(ids) => {
            apply_rope_with_positions_and_periods(x, ids, periods, dims, traditional, scale)
        }
    }
}

/// Apply RoPE to a tensor (functional version).
///
/// # Arguments
/// * `x` - Input tensor of shape [..., seq_len, head_dim]
/// * `dims` - Number of dimensions to apply RoPE to
/// * `traditional` - If true, use traditional RoPE implementation
/// * `base` - Base frequency for the embeddings
/// * `scale` - Scale for the positions
/// * `offset` - Position offset
///
/// # Returns
/// Tensor with rotary embeddings applied.
pub fn apply_rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> Result<Array, Exception> {
    Ok(fast::rope(x, dims, traditional, base, scale, offset))
}

/// Apply RoPE with explicit position IDs.
///
/// This is essential for packed sequence training where multiple sequences
/// are concatenated and position IDs need to reset for each sequence.
///
/// Uses the non-traditional (efficient) RoPE implementation where dimensions
/// are split in half rather than interleaved.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, heads, seq_len, head_dim]
/// * `position_ids` - Position indices of shape [seq_len]
/// * `dims` - Number of dimensions to apply RoPE to (usually head_dim)
/// * `traditional` - If true, use traditional (interleaved) RoPE
/// * `base` - Base frequency for the embeddings (default 10000.0)
/// * `scale` - Scale factor for positions (default 1.0)
///
/// # Returns
/// Tensor with rotary embeddings applied according to position_ids.
///
/// # Example
/// ```ignore
/// // Packed sequences: [seq1_tok1, seq1_tok2, seq2_tok1, seq2_tok2, seq2_tok3]
/// // Position IDs:     [0,         1,         0,         1,         2]
/// let x = Array::zeros::<f32>(&[1, 4, 5, 64]); // batch=1, heads=4, seq=5, dim=64
/// let position_ids = Array::from_i32_slice(&[0_i32, 1, 0, 1, 2], &[5]);
/// let output = apply_rope_with_positions(&x, &position_ids, 64, false, 10000.0, 1.0)?;
/// ```
pub fn apply_rope_with_positions(
    x: &Array,
    position_ids: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    // Compute inverse frequencies: inv_freq[i] = 1.0 / (base^(2i/dims))
    // indices: [0, 1, ..., half_dims-1] as float
    let indices = ops::arange_range(0, dims / 2); // [half_dims] float32
    let neg_two_over_dims = Array::from_f32(-2.0 / dims as f32);
    let exponents = indices.multiply(&neg_two_over_dims);
    let base_arr = Array::from_f32(base);
    let inv_freq = base_arr.pow(&exponents); // [half_dims]

    rope_with_positions_and_inv_freq(x, position_ids, &inv_freq, dims, traditional, scale)
}

/// Apply RoPE with explicit position IDs and an explicit period table.
///
/// Sibling of [`apply_rope_with_positions`] for scalings that rescale each
/// frequency band separately, where no single `base` describes the rotation
/// (Llama 3). `periods` is the same `[dims / 2]` table `mx.fast.rope` takes —
/// periods, not inverse frequencies — so one table serves both the fused
/// contiguous-offset path and this packed-sequence one.
pub fn apply_rope_with_positions_and_periods(
    x: &Array,
    position_ids: &Array,
    periods: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
) -> Result<Array, Exception> {
    let inv_freq = Array::from_f32(1.0).divide(periods);
    rope_with_positions_and_inv_freq(x, position_ids, &inv_freq, dims, traditional, scale)
}

/// Shared body: rotate `x` at the given per-token positions using a
/// precomputed `[dims / 2]` inverse-frequency table.
fn rope_with_positions_and_inv_freq(
    x: &Array,
    position_ids: &Array,
    inv_freq: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
) -> Result<Array, Exception> {
    // x shape: [batch, heads, seq_len, head_dim]
    let half_dims = dims / 2;

    // position_ids: [seq_len] as i32 → float and scale
    let pos_float = position_ids.as_dtype(Dtype::Float32.as_i32());
    let scale_arr = Array::from_f32(scale);
    let scaled_pos = pos_float.multiply(&scale_arr); // [seq_len]

    // Compute angles: [seq_len, half_dims]
    let pos_expanded = scaled_pos.expand_dims(-1); // [seq_len, 1]
    let inv_freq_expanded = inv_freq.expand_dims(0); // [1, half_dims]
    let angles = pos_expanded.multiply(&inv_freq_expanded); // [seq_len, half_dims]

    // Compute cos and sin
    let cos_theta = angles.cos(); // [seq_len, half_dims]
    let sin_theta = angles.sin(); // [seq_len, half_dims]

    // Reshape for broadcasting with x: [1, 1, seq_len, half_dims]
    let cos_theta = cos_theta.reshape(&[1, 1, -1, half_dims]);
    let sin_theta = sin_theta.reshape(&[1, 1, -1, half_dims]);

    Ok(rotate_with_cos_sin(
        x,
        &cos_theta,
        &sin_theta,
        dims,
        traditional,
    ))
}

/// Apply RoPE using explicit per-dimension inverse frequencies.
///
/// Unlike [`apply_rope`] (which derives `inv_freq[i] = base^(-2i/dims)` from a
/// single scalar `base`), this takes a precomputed `inv_freq` table of length
/// `dims/2`. This is required for Phi-3 LongRoPE / SuRoPE, where each
/// frequency is independently scaled by a per-dimension `long_factor`:
/// `inv_freq[i] = 1 / (long_factor[i] * base^(2i/dims))`.
///
/// Positions are the contiguous range `[offset, offset + seq_len)`. Any
/// magnitude (mscale) rescaling of the activations must be applied by the
/// caller *before* this call — this function only rotates.
///
/// # Arguments
/// * `x` - `[batch, heads, seq_len, head_dim]`
/// * `inv_freq` - `[dims/2]` angular frequencies
/// * `dims` - rotary dimension (may be < head_dim for partial RoPE)
/// * `traditional` - interleaved (true) vs split-half (false)
/// * `offset` - absolute position of the first token (KV-cache aware)
pub fn apply_rope_with_freqs(
    x: &Array,
    inv_freq: &Array,
    dims: i32,
    traditional: bool,
    offset: i32,
) -> Result<Array, Exception> {
    let seq_len = x.shape()[2];
    let half_dims = dims / 2;

    // positions: [offset, offset+1, ..., offset+seq_len-1] as float32
    let positions = ops::arange_range(offset, offset + seq_len);
    let angles = positions
        .expand_dims(-1) // [seq_len, 1]
        .multiply(&inv_freq.expand_dims(0)); // [1, half_dims] → [seq_len, half_dims]

    let cos_theta = angles.cos().reshape(&[1, 1, -1, half_dims]);
    let sin_theta = angles.sin().reshape(&[1, 1, -1, half_dims]);

    Ok(rotate_with_cos_sin(
        x,
        &cos_theta,
        &sin_theta,
        dims,
        traditional,
    ))
}

/// Apply RoPE with per-batch-row position IDs.
///
/// Sibling of [`apply_rope_with_positions`] for the fused continuous-batching
/// decode path. Each batch row carries its own absolute offsets, so the
/// computed cos/sin are broadcast as `[batch, 1, seq_len, half_dims]` rather
/// than `[1, 1, seq_len, half_dims]`.
///
/// # Arguments
/// * `x` - Input tensor of shape `[batch, heads, seq_len, head_dim]`
/// * `position_ids` - Position indices of shape `[batch, seq_len]` (int32)
/// * `dims` - Number of dimensions to apply RoPE to
/// * `traditional` - If true, use traditional (interleaved) RoPE
/// * `base` - Base frequency for the embeddings
/// * `scale` - Scale factor for positions
pub fn apply_rope_with_per_batch_positions(
    x: &Array,
    position_ids: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Exception::custom(format!(
            "apply_rope_with_per_batch_positions: expected rank-4 input, got {shape:?}"
        )));
    }
    let batch = shape[0];
    let head_dim = shape[3];
    let half_dims = dims / 2;

    let pos_shape = position_ids.shape();
    if pos_shape.len() != 2 || pos_shape[0] != batch {
        return Err(Exception::custom(format!(
            "apply_rope_with_per_batch_positions: position_ids must be [batch={batch}, seq_len], got {pos_shape:?}"
        )));
    }

    // inv_freq[i] = base ^ (-2i / dims)
    let indices = ops::arange_range(0, half_dims);
    let neg_two_over_dims = Array::from_f32(-2.0 / dims as f32);
    let exponents = indices.multiply(&neg_two_over_dims);
    let base_arr = Array::from_f32(base);
    let inv_freq = base_arr.pow(&exponents); // [half_dims]

    // scaled positions: [batch, seq_len]
    let pos_float = position_ids.as_dtype(Dtype::Float32.as_i32());
    let scale_arr = Array::from_f32(scale);
    let scaled_pos = pos_float.multiply(&scale_arr);

    // angles: [batch, seq_len, half_dims]
    let pos_expanded = scaled_pos.expand_dims(-1); // [batch, seq_len, 1]
    let inv_freq_expanded = inv_freq.reshape(&[1, 1, half_dims]); // [1, 1, half_dims]
    let angles = pos_expanded.multiply(&inv_freq_expanded);

    let cos_theta = angles.cos();
    let sin_theta = angles.sin();

    // Reshape to [batch, 1, seq_len, half_dims] for broadcasting with
    // x shaped [batch, heads, seq_len, head_dim].
    let cos_theta = cos_theta.reshape(&[batch, 1, -1, half_dims]);
    let sin_theta = sin_theta.reshape(&[batch, 1, -1, half_dims]);

    if traditional {
        let x_rope = if dims < head_dim {
            x.split(&[dims], -1).remove(0)
        } else {
            x.clone()
        };
        let rope_shape = x_rope.shape();
        let batch_r = rope_shape[0];
        let heads = rope_shape[1];
        let seq_len = rope_shape[2];
        let x_pairs = x_rope.reshape(&[batch_r, heads, seq_len, half_dims, 2]);
        let x_even = x_pairs
            .slice(&[0, 0, 0, 0, 0], &[batch_r, heads, seq_len, half_dims, 1])
            .squeeze(-1);
        let x_odd = x_pairs
            .slice(&[0, 0, 0, 0, 1], &[batch_r, heads, seq_len, half_dims, 2])
            .squeeze(-1);

        let r_even = x_even
            .multiply(&cos_theta)
            .subtract(&x_odd.multiply(&sin_theta));
        let r_odd = x_even.multiply(&sin_theta).add(&x_odd.multiply(&cos_theta));

        let stacked = ops::stack_axis(vec![r_even, r_odd].as_slice(), -1);
        let x_rotated = stacked.reshape(&[batch_r, heads, seq_len, dims]);

        if dims < head_dim {
            let parts = x.split(&[dims], -1);
            Ok(ops::concatenate_axis(&[&x_rotated, &parts[1]], -1))
        } else {
            Ok(x_rotated)
        }
    } else {
        let parts = if dims == head_dim {
            x.split(&[half_dims], -1)
        } else {
            x.split(&[half_dims, dims], -1)
        };
        let x1 = &parts[0];
        let x2 = &parts[1];
        let rx1 = x1.multiply(&cos_theta).subtract(&x2.multiply(&sin_theta));
        let rx2 = x1.multiply(&sin_theta).add(&x2.multiply(&cos_theta));
        let x_rotated = ops::concatenate_axis(&[&rx1, &rx2], -1);
        if dims < head_dim && parts.len() > 2 {
            Ok(ops::concatenate_axis(&[&x_rotated, &parts[2]], -1))
        } else {
            Ok(x_rotated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{Array, Dtype, random};

    #[test]
    fn test_rope_functional() {
        let x = random::normal(&[8, 4, 64], Dtype::Float32);
        let output = apply_rope(&x, 64, false, 10000.0, 1.0, 0).unwrap();
        assert_eq!(output.shape()[1], 4); // seq_len
        assert_eq!(output.shape()[2], 64); // head_dim
    }

    // ------------------------------------------------------------------
    // RopePositions
    //
    // The two arms take different code paths — a fused Metal kernel versus
    // cos/sin tables built here — so "same rotation" is an assertion, not a
    // definition. Everything downstream of `RopePositions` assumes it.
    // ------------------------------------------------------------------

    fn ramp(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin()).collect();
        Array::from_slice(&values, shape)
    }

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let n = a.shape().iter().product::<i32>() as usize;
        let a = a.subtract(b).abs();
        a.try_eval().expect("eval");
        a.as_slice::<f32>()[..n]
            .iter()
            .fold(0.0f32, |worst, v| worst.max(v.abs()))
    }

    fn contiguous_positions(offset: i32, seq_len: i32) -> Array {
        let values: Vec<i32> = (offset..offset + seq_len).collect();
        Array::from_i32_slice_shaped(&values, &[seq_len])
    }

    #[test]
    fn contiguous_and_explicit_positions_agree() {
        let (seq_len, dims) = (6, 16);
        let x = ramp(&[1, 2, seq_len, dims]);

        for traditional in [false, true] {
            for offset in [0, 5] {
                let contiguous = rope(
                    &x,
                    RopePositions::Offset(offset),
                    dims,
                    traditional,
                    10000.0,
                    1.0,
                )
                .unwrap();
                let ids = contiguous_positions(offset, seq_len);
                let explicit = rope(
                    &x,
                    RopePositions::Explicit(&ids),
                    dims,
                    traditional,
                    10000.0,
                    1.0,
                )
                .unwrap();

                let diff = max_abs_diff(&contiguous, &explicit);
                assert!(
                    diff < 1e-4,
                    "traditional={traditional} offset={offset}: max |Δ| = {diff:e}"
                );
            }
        }
    }

    /// Partial RoPE (`dims < head_dim`) has to leave the tail untouched on
    /// both arms, not just the fused one.
    #[test]
    fn contiguous_and_explicit_positions_agree_on_partial_rope() {
        let (seq_len, head_dim, dims) = (4, 16, 8);
        let x = ramp(&[1, 2, seq_len, head_dim]);
        let ids = contiguous_positions(3, seq_len);

        for traditional in [false, true] {
            let contiguous = rope(
                &x,
                RopePositions::Offset(3),
                dims,
                traditional,
                10000.0,
                1.0,
            )
            .unwrap();
            let explicit = rope(
                &x,
                RopePositions::Explicit(&ids),
                dims,
                traditional,
                10000.0,
                1.0,
            )
            .unwrap();

            let diff = max_abs_diff(&contiguous, &explicit);
            assert!(diff < 1e-4, "traditional={traditional}: max |Δ| = {diff:e}");
        }
    }

    #[test]
    fn contiguous_and_explicit_positions_agree_with_an_inv_freq_table() {
        let (seq_len, dims) = (5, 16);
        let x = ramp(&[1, 2, seq_len, dims]);
        // A table no scalar base produces: every other band stretched.
        let table: Vec<f32> = (0..dims / 2)
            .map(|i| 1.0 / (10000.0f32.powf(2.0 * i as f32 / dims as f32) * (1.0 + i as f32)))
            .collect();
        let inv_freq = Array::from_slice(&table, &[dims / 2]);
        let ids = contiguous_positions(2, seq_len);

        let contiguous =
            rope_with_inv_freq(&x, RopePositions::Offset(2), &inv_freq, dims, false).unwrap();
        let explicit =
            rope_with_inv_freq(&x, RopePositions::Explicit(&ids), &inv_freq, dims, false).unwrap();

        let diff = max_abs_diff(&contiguous, &explicit);
        assert!(diff < 1e-4, "max |Δ| = {diff:e}");
    }

    #[test]
    fn contiguous_and_explicit_positions_agree_with_a_period_table() {
        let (seq_len, dims) = (5, 64);
        let x = ramp(&[1, 2, seq_len, dims]);
        let periods = pmetal_bridge::rope::Rotary::from_config(
            dims,
            pmetal_bridge::rope::RopeConfig::from_json(&serde_json::json!({
                "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
            })),
            500_000.0,
            1.0,
        )
        .unwrap()
        .periods(0);
        let periods = Array::from_slice(&periods, &[dims / 2]);
        let ids = contiguous_positions(7, seq_len);

        let contiguous =
            rope_with_periods(&x, RopePositions::Offset(7), &periods, dims, false, 1.0).unwrap();
        let explicit = rope_with_periods(
            &x,
            RopePositions::Explicit(&ids),
            &periods,
            dims,
            false,
            1.0,
        )
        .unwrap();

        let diff = max_abs_diff(&contiguous, &explicit);
        assert!(diff < 1e-4, "max |Δ| = {diff:e}");
    }

    /// The reason the type exists: two sequences packed into one row must
    /// rotate as if each had been run on its own.
    #[test]
    fn packed_positions_restart_the_rotation_at_a_sequence_boundary() {
        let dims = 16;
        let first = ramp(&[1, 1, 2, dims]);
        let second = ramp(&[1, 1, 3, dims]);
        let packed = ops::concatenate_axis(&[&first, &second], 2);

        let separate = ops::concatenate_axis(
            &[
                &rope(&first, RopePositions::Offset(0), dims, false, 10000.0, 1.0).unwrap(),
                &rope(&second, RopePositions::Offset(0), dims, false, 10000.0, 1.0).unwrap(),
            ],
            2,
        );

        let ids = Array::from_i32_slice_shaped(&[0, 1, 0, 1, 2], &[5]);
        let together = rope(
            &packed,
            RopePositions::Explicit(&ids),
            dims,
            false,
            10000.0,
            1.0,
        )
        .unwrap();
        assert!(max_abs_diff(&separate, &together) < 1e-4);

        // And that the run-through positions a packed forward gets today are
        // genuinely different, so the test above is not vacuous.
        let run_through =
            rope(&packed, RopePositions::Offset(0), dims, false, 10000.0, 1.0).unwrap();
        assert!(max_abs_diff(&separate, &run_through) > 1e-2);
    }

    #[test]
    fn resolve_prefers_explicit_positions_over_the_cache_offset() {
        let ids = Array::from_i32_slice_shaped(&[0, 1, 0], &[3]);
        assert!(matches!(
            RopePositions::resolve(Some(&ids), 17),
            RopePositions::Explicit(_)
        ));
        assert!(matches!(
            RopePositions::resolve(None, 17),
            RopePositions::Offset(17)
        ));
    }
}
