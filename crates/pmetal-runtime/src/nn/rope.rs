//! Applying Rotary Position Embedding (RoPE) at contiguous or explicit
//! positions.
//!
//! What a config's RoPE scaling means (linear, dynamic NTK, YaRN, LongRoPE,
//! Llama 3 bands, proportional) lives in one place, [`crate::rope`];
//! an architecture holds a [`RotaryEmbedding`] from there and rotates with
//! [`rope_embedding`]. The helpers here thread [`RopePositions`] (a cache
//! offset, or one position per token for packed batches) through the fused
//! kernel and the explicit cos/sin tables.
//!
//! The rotation is computed in f32 and returned in the input's dtype on every
//! route: the cos/sin tables are f32, and a product with them would otherwise
//! promote a half-precision input.

use crate::compat::{Array, Dtype, Exception, fast, ops};
use crate::rope::{RotaryEmbedding, rotate_with_cos_sin};

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
///
/// * `x` - `[batch, heads, seq_len, head_dim]`
/// * `dims` - the number of leading channels to rotate (the rest pass through)
/// * `traditional` - pairs `(x[2i], x[2i+1])` instead of `(x[i], x[i + dims/2])`
/// * `scale` - multiplies the positions
pub fn rope(
    x: &Array,
    positions: RopePositions<'_>,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    match positions {
        RopePositions::Offset(offset) => Ok(fast::rope(x, dims, traditional, base, scale, offset)),
        RopePositions::Explicit(ids) => {
            let inv_freq = base_inverse_frequencies(base, dims);
            Ok(rotate_at(x, ids, &inv_freq, dims, traditional, scale))
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
            let inv_freq = Array::from_f32(1.0).divide(periods);
            Ok(rotate_at(x, ids, &inv_freq, dims, traditional, scale))
        }
    }
}

/// Apply RoPE with per-batch-row position IDs.
///
/// For the fused continuous-batching decode path, where each batch row
/// carries its own absolute offset: the cos/sin tables are broadcast as
/// `[batch, 1, seq_len, dims / 2]` rather than `[1, 1, seq_len, dims / 2]`.
///
/// * `x` - `[batch, heads, seq_len, head_dim]`
/// * `position_ids` - `[batch, seq_len]`, integer
pub fn apply_rope_with_per_batch_positions(
    x: &Array,
    position_ids: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    if x.ndim() != 4 {
        return Err(Exception::custom(format!(
            "apply_rope_with_per_batch_positions: expected rank-4 input, got {:?}",
            x.shape()
        )));
    }
    if position_ids.ndim() != 2 || position_ids.dim(0) != x.dim(0) {
        return Err(Exception::custom(format!(
            "apply_rope_with_per_batch_positions: position_ids must be [batch={}, seq_len], got {:?}",
            x.dim(0),
            position_ids.shape()
        )));
    }
    let inv_freq = base_inverse_frequencies(base, dims);
    Ok(rotate_at(
        x,
        position_ids,
        &inv_freq,
        dims,
        traditional,
        scale,
    ))
}

/// `base^(-2i / dims)` for `i` in `0..dims / 2`.
fn base_inverse_frequencies(base: f32, dims: i32) -> Array {
    let exponents = ops::arange_range(0, dims / 2).multiply(&Array::from_f32(-2.0 / dims as f32));
    Array::from_f32(base).pow(&exponents)
}

/// Rotate `x` (`[batch, heads, seq_len, head_dim]`) at one position per token
/// (`position_ids`: `[seq_len]` shared by the batch, or `[batch, seq_len]`)
/// with a `[dims / 2]` inverse-frequency table, in f32, and return `x`'s dtype.
fn rotate_at(
    x: &Array,
    position_ids: &Array,
    inv_freq: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
) -> Array {
    let half = dims / 2;
    // angles: [seq_len, half] or [batch, seq_len, half]
    let angles = position_ids
        .as_dtype(Dtype::Float32.as_i32())
        .multiply(&Array::from_f32(scale))
        .expand_dims(-1)
        .multiply(&inv_freq.reshape(&[half]));
    // Laid out against x: [1 | batch, 1, seq_len, half].
    let rows = if position_ids.ndim() == 2 {
        position_ids.dim(0)
    } else {
        1
    };
    let shape = [rows, 1, -1, half];
    let (cos, sin) = (angles.cos().reshape(&shape), angles.sin().reshape(&shape));
    rotate_with_cos_sin(
        &x.as_dtype(Dtype::Float32.as_i32()),
        &cos,
        &sin,
        dims,
        traditional,
    )
    .as_dtype(x.dtype().as_i32())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::{Array, Dtype};

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
    fn contiguous_and_explicit_positions_agree_with_a_period_table() {
        let (seq_len, dims) = (5, 64);
        let x = ramp(&[1, 2, seq_len, dims]);
        let periods = crate::rope::Rotary::from_config(
            dims,
            crate::rope::RopeConfig::from_json(&serde_json::json!({
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

    /// Rotation by `position * scale * base^(-2i / dims)` in f64, over
    /// `x [B, H, L, D]` at `positions [B][L]`; the channels past `dims` pass
    /// through. Pairs are `(x[i], x[i + dims/2])`, or `(x[2i], x[2i+1])` when
    /// `traditional`.
    #[allow(clippy::too_many_arguments)]
    fn rotate_f64(
        x: &[f64],
        shape: [usize; 4],
        positions: &[Vec<f64>],
        dims: usize,
        traditional: bool,
        base: f64,
        scale: f64,
    ) -> Vec<f64> {
        let [_, heads, len, width] = shape;
        let mut out = x.to_vec();
        for (b, row_positions) in positions.iter().enumerate() {
            for h in 0..heads {
                for (l, position) in row_positions.iter().enumerate().take(len) {
                    let row = ((b * heads + h) * len + l) * width;
                    for i in 0..dims / 2 {
                        let angle = position * scale * base.powf(-2.0 * i as f64 / dims as f64);
                        let (sin, cos) = angle.sin_cos();
                        let (a, c) = if traditional {
                            (row + 2 * i, row + 2 * i + 1)
                        } else {
                            (row + i, row + i + dims / 2)
                        };
                        out[a] = x[a] * cos - x[c] * sin;
                        out[c] = x[a] * sin + x[c] * cos;
                    }
                }
            }
        }
        out
    }

    fn values(a: &Array) -> Vec<f64> {
        let mut a = a.as_dtype(Dtype::Float32.as_i32());
        a.eval();
        a.to_f32_vec(a.size())
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    }

    /// Every way of giving positions rotates as the definition does, in f64:
    /// a cache offset, one position per token, one per token per batch row;
    /// both pairings, partial rotation, a position scale; and the result
    /// keeps the input's dtype (the cos/sin tables are f32, and a product with
    /// them promotes a half-precision input to f32 unless it is cast back).
    #[test]
    fn rotations_are_the_definition_and_keep_the_dtype() {
        let (batch, heads, len, width) = (2usize, 3usize, 5usize, 16usize);
        let shape = [batch as i32, heads as i32, len as i32, width as i32];
        let base_x = ramp(&shape);
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let x = base_x.as_dtype(dtype.as_i32());
            let x64 = values(&x);
            let tol = if dtype == Dtype::Float32 { 2e-5 } else { 2e-2 };
            for (dims, traditional, scale) in [
                (16, false, 1.0),
                (16, true, 0.5),
                (8, false, 1.0),
                (8, true, 2.0),
            ] {
                // A shared run of positions from an offset.
                let offset = 7;
                let shared: Vec<Vec<f64>> = (0..batch)
                    .map(|_| (0..len).map(|l| (offset + l) as f64).collect())
                    .collect();
                // Per-row positions, differing by row.
                let per_row: Vec<Vec<f64>> = (0..batch)
                    .map(|b| (0..len).map(|l| (3 * b + 2 * l + 1) as f64).collect())
                    .collect();
                let shared_ids = Array::from_i32_slice_shaped(
                    &shared[0].iter().map(|&p| p as i32).collect::<Vec<_>>(),
                    &[len as i32],
                );
                let row_ids = Array::from_i32_slice_shaped(
                    &per_row
                        .iter()
                        .flatten()
                        .map(|&p| p as i32)
                        .collect::<Vec<_>>(),
                    &[batch as i32, len as i32],
                );
                let cases: [(&str, Array, &[Vec<f64>]); 3] = [
                    (
                        "offset",
                        rope(
                            &x,
                            RopePositions::Offset(offset as i32),
                            dims,
                            traditional,
                            10000.0,
                            scale as f32,
                        )
                        .unwrap(),
                        &shared,
                    ),
                    (
                        "explicit",
                        rope(
                            &x,
                            RopePositions::Explicit(&shared_ids),
                            dims,
                            traditional,
                            10000.0,
                            scale as f32,
                        )
                        .unwrap(),
                        &shared,
                    ),
                    (
                        "per batch row",
                        apply_rope_with_per_batch_positions(
                            &x,
                            &row_ids,
                            dims,
                            traditional,
                            10000.0,
                            scale as f32,
                        )
                        .unwrap(),
                        &per_row,
                    ),
                ];
                for (name, got, positions) in cases {
                    let what = format!("{name}, {dtype:?}, dims {dims}, traditional {traditional}");
                    assert_eq!(
                        got.dtype(),
                        dtype,
                        "{what}: the output keeps the input's dtype"
                    );
                    let want = rotate_f64(
                        &x64,
                        [batch, heads, len, width],
                        positions,
                        dims as usize,
                        traditional,
                        10000.0,
                        scale,
                    );
                    let worst = values(&got)
                        .iter()
                        .zip(&want)
                        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                    assert!(worst <= tol, "{what}: off by {worst:e}");
                }
            }
        }
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
