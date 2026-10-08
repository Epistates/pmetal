//! Interleaved multimodal RoPE (mRoPE) for the Qwen 3.5 family.
//!
//! A text token has one position; an image or video token has three, its
//! temporal, row and column index (`get_rope_index` in the reference). The
//! rotary frequencies are split between the three axes *interleaved*: with
//! `mrope_section = [t, h, w]`, frequency `i` rotates by the row position when
//! `i % 3 == 1` and `i < 3h`, by the column position when `i % 3 == 2` and
//! `i < 3w`, and by the temporal position otherwise (`Qwen3_5TextRotaryEmbedding
//! .recomposition_frequencies`). Transformers applies that layout whatever
//! `mrope_interleaved` says, so this does too.
//!
//! When all three positions are equal, as they are for every text token, this
//! is ordinary RoPE at that position. Text-only prompts therefore never come
//! here: they keep the fused kernel, and only a prompt with media builds these
//! tables, once per forward, shared by every full-attention layer.

use crate::compat::{Array, Dtype, ops};

/// The cos/sin tables of one forward pass, `[1, 1, L, rope_dims / 2]` f32.
#[derive(Clone)]
pub struct MropeTables {
    cos: Array,
    sin: Array,
    rope_dims: i32,
}

impl std::fmt::Debug for MropeTables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MropeTables")
            .field("rope_dims", &self.rope_dims)
            .field("shape", &self.cos.shape())
            .finish()
    }
}

/// The axis (0 = temporal, 1 = row, 2 = column) each of the `half` rotary
/// frequencies takes its position from.
pub fn frequency_axes(half: usize, section: [usize; 3]) -> Vec<usize> {
    (0..half)
        .map(|i| match i % 3 {
            1 if i < 3 * section[1] => 1,
            2 if i < 3 * section[2] => 2,
            _ => 0,
        })
        .collect()
}

/// The reference's inverse frequencies, `1 / theta^(2i / rope_dims)`, in f32.
pub fn inverse_frequencies(rope_dims: usize, theta: f32) -> Vec<f32> {
    (0..rope_dims / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / rope_dims as f32))
        .collect()
}

impl MropeTables {
    /// Build the tables for `positions`, a `[3, L]` int32 array of temporal,
    /// row and column positions.
    pub fn new(positions: &Array, rope_dims: i32, theta: f32, section: [usize; 3]) -> Self {
        Self::with_frequencies(
            positions,
            rope_dims,
            &inverse_frequencies(rope_dims as usize, theta),
            1.0,
            section,
        )
    }

    /// [`new`](Self::new) with explicit `[rope_dims / 2]` inverse frequencies
    /// and a gain on `cos` and `sin`, as transformers' rotary embedding takes
    /// them from its `rope_type` (YaRN: blended frequencies and its attention
    /// factor; see [`super::family::Rotary`]).
    pub fn with_frequencies(
        positions: &Array,
        rope_dims: i32,
        inverse_frequencies: &[f32],
        gain: f32,
        section: [usize; 3],
    ) -> Self {
        let half = (rope_dims / 2) as usize;
        assert_eq!(
            inverse_frequencies.len(),
            half,
            "one inverse frequency per rotary pair"
        );
        let axes: Vec<i32> = frequency_axes(half, section)
            .into_iter()
            .map(|axis| axis as i32)
            .collect();
        let axes = Array::from_i32_slice(&axes);
        // [3, L] -> [half, L]: each frequency's own position row.
        let per_frequency = positions
            .as_dtype(Dtype::Float32.as_i32())
            .take_axis(&axes, 0);
        let inv_freq = Array::from_f32_slice(inverse_frequencies, &[half as i32]);
        // [L, half]
        let angles = per_frequency
            .transpose_axes(&[1, 0])
            .multiply(&inv_freq.reshape(&[1, half as i32]));
        let length = angles.dim(0);
        let shape = [1, 1, length, half as i32];
        let (mut cos, mut sin) = (ops::cos(&angles), ops::sin(&angles));
        if gain != 1.0 {
            let gain = Array::from_f32(gain);
            cos = cos.multiply(&gain);
            sin = sin.multiply(&gain);
        }
        Self {
            cos: cos.reshape(&shape),
            sin: sin.reshape(&shape),
            rope_dims,
        }
    }

    /// Tables for one position per token (`[L]` int32), the same on all three
    /// axes: the text-only case, for a rotary embedding the fused kernel
    /// cannot express on its own.
    pub fn for_text_positions(
        positions: &Array,
        rope_dims: i32,
        inverse_frequencies: &[f32],
        gain: f32,
    ) -> Self {
        let length = positions.dim(0);
        let positions = positions.reshape(&[1, length]);
        let stacked = ops::concatenate_axis(&[&positions, &positions, &positions], 0);
        Self::with_frequencies(&stacked, rope_dims, inverse_frequencies, gain, [0, 0, 0])
    }

    /// Positions the tables were built for.
    pub fn len(&self) -> i32 {
        self.cos.dim(2)
    }

    /// Whether the tables cover no positions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rotate the first `rope_dims` channels of `x` (`[B, heads, L, head_dim]`),
    /// split-half (non-traditional) layout, in f32; the result has `x`'s
    /// dtype.
    pub fn apply(&self, x: &Array) -> Array {
        let dtype = x.dtype();
        let head_dim = x.dim(3);
        let half = self.rope_dims / 2;
        let rotated = ops::slice_axis(x, 3, 0, self.rope_dims).as_dtype(Dtype::Float32.as_i32());
        let x1 = ops::slice_axis(&rotated, 3, 0, half);
        let x2 = ops::slice_axis(&rotated, 3, half, self.rope_dims);
        let out1 = x1.multiply(&self.cos).subtract(&x2.multiply(&self.sin));
        let out2 = x2.multiply(&self.cos).add(&x1.multiply(&self.sin));
        let out = ops::concatenate_axis(&[&out1, &out2], 3).as_dtype(dtype.as_i32());
        if self.rope_dims == head_dim {
            out
        } else {
            let pass = ops::slice_axis(x, 3, self.rope_dims, head_dim);
            ops::concatenate_axis(&[&out, &pass], 3)
        }
    }
}

/// A rotary embedding the fused kernel's `(base, scale)` pair cannot express:
/// YaRN's blended frequencies, with its attention factor on the rotated
/// channels (transformers multiplies `cos` and `sin` by it, so the channels
/// past `rope_dims` keep their scale).
///
/// Contiguous positions run the fused kernel with explicit periods and then
/// the gain; explicit positions (tree verification) and three-axis positions
/// (a prompt with media) go through [`MropeTables`] built from the same
/// frequencies.
#[derive(Clone)]
pub struct ScaledRope {
    rope_dims: i32,
    inverse_frequencies: Vec<f32>,
    periods: Array,
    gain: f32,
    /// `[head_dim]` f32: `gain` on the rotated channels, 1 on the rest.
    gain_per_channel: Option<Array>,
}

impl std::fmt::Debug for ScaledRope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScaledRope")
            .field("rope_dims", &self.rope_dims)
            .field("gain", &self.gain)
            .finish()
    }
}

impl ScaledRope {
    /// `None` for plain RoPE, which the fused kernel runs from `theta`.
    pub fn new(rotary: &super::family::Rotary, head_dim: i32) -> Option<Self> {
        if rotary.is_plain() {
            return None;
        }
        let inverse_frequencies = rotary.inverse_frequencies();
        let half = inverse_frequencies.len() as i32;
        let periods: Vec<f32> = inverse_frequencies.iter().map(|f| 1.0 / f).collect();
        let gain = rotary.attention_factor();
        let gain_per_channel = (gain != 1.0).then(|| {
            let channels: Vec<f32> = (0..head_dim)
                .map(|c| if c < rotary.dims { gain } else { 1.0 })
                .collect();
            Array::from_f32_slice(&channels, &[head_dim])
        });
        Some(Self {
            rope_dims: rotary.dims,
            inverse_frequencies,
            periods: Array::from_f32_slice(&periods, &[half]),
            gain,
            gain_per_channel,
        })
    }

    /// Rotate `[B, heads, L, head_dim]` at positions `offset..offset + L`.
    pub fn apply(&self, x: &Array, offset: i32) -> Array {
        let rotated = x.rope_with_freqs(self.rope_dims, false, 1.0, offset, &self.periods);
        match &self.gain_per_channel {
            Some(gain) => rotated
                .as_dtype(Dtype::Float32.as_i32())
                .multiply(gain)
                .as_dtype(x.dtype().as_i32()),
            None => rotated,
        }
    }

    /// Rotate `[B, heads, L, head_dim]` at one explicit position per token
    /// (`[L]` int32).
    pub fn apply_at(&self, x: &Array, positions: &Array) -> Array {
        MropeTables::for_text_positions(
            positions,
            self.rope_dims,
            &self.inverse_frequencies,
            self.gain,
        )
        .apply(x)
    }

    /// Tables for a prompt's `[3, L]` multimodal positions.
    pub fn tables(&self, positions: &Array, section: [usize; 3]) -> MropeTables {
        MropeTables::with_frequencies(
            positions,
            self.rope_dims,
            &self.inverse_frequencies,
            self.gain,
            section,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every path of a YaRN rotation agrees: the fused kernel with explicit
    /// periods and gain, explicit per-token positions, and three equal
    /// position axes.
    #[test]
    fn scaled_rope_paths_agree() {
        let (heads, len, head_dim) = (2, 6, 32);
        let text = serde_json::json!({
            "head_dim": head_dim, "partial_rotary_factor": 0.25, "rope_theta": 10000.0,
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                "original_max_position_embeddings": 16.0, "beta_fast": 32.0,
                "beta_slow": 1.0, "truncate": true, "attention_factor": 1.1386}
        });
        let rotary = super::super::family::Rotary::from_text_config(&text).unwrap();
        let scaled = ScaledRope::new(&rotary, head_dim).expect("yarn is not plain");
        let x = crate::compat::random::uniform_f32(&[1, heads, len, head_dim]);
        let offset = 20;
        let fused = scaled.apply(&x, offset);
        let ids: Vec<i32> = (offset..offset + len).collect();
        let explicit = scaled.apply_at(&x, &Array::from_i32_slice(&ids));
        let three: Vec<i32> = (0..3).flat_map(|_| offset..offset + len).collect();
        let tables = scaled
            .tables(&Array::from_i32_slice_shaped(&three, &[3, len]), [2, 1, 1])
            .apply(&x);
        let max_diff = |a: &Array, b: &Array| {
            a.subtract(b)
                .abs()
                .max_axis(-1, false)
                .max_axis(-1, false)
                .max_axis(-1, false)
                .max_axis(-1, false)
                .item_f32()
        };
        let d1 = max_diff(&fused, &explicit);
        let d2 = max_diff(&fused, &tables);
        // The untouched channels keep their values, unscaled.
        let pass = max_diff(
            &ops::slice_axis(&fused, 3, 8, head_dim),
            &ops::slice_axis(&x, 3, 8, head_dim),
        );
        crate::check_last_error().unwrap();
        assert!(
            d1 < 1e-5 && d2 < 1e-5,
            "fused vs explicit {d1}, vs tables {d2}"
        );
        assert_eq!(pass, 0.0);
    }

    /// Qwen3.8's `[11, 11, 10]` over 32 frequencies: 11 temporal, 11 row, 10
    /// column, interleaved.
    #[test]
    fn released_section_interleaves() {
        let axes = frequency_axes(32, [11, 11, 10]);
        assert_eq!(&axes[..6], &[0, 1, 2, 0, 1, 2]);
        assert_eq!(axes[30], 0);
        assert_eq!(axes[31], 1);
        assert_eq!(axes.iter().filter(|&&a| a == 0).count(), 11);
        assert_eq!(axes.iter().filter(|&&a| a == 1).count(), 11);
        assert_eq!(axes.iter().filter(|&&a| a == 2).count(), 10);
    }

    /// Equal positions on all three axes are plain RoPE, which the fused
    /// kernel computes.
    #[test]
    fn text_positions_match_the_fused_kernel() {
        let (heads, len, head_dim, rope_dims) = (2, 5, 16, 8);
        let x = crate::compat::random::uniform_f32(&[1, heads, len, head_dim]);
        let positions: Vec<i32> = (0..3).flat_map(|_| 3..3 + len).collect();
        let positions = Array::from_i32_slice_shaped(&positions, &[3, len]);
        let tables = MropeTables::new(&positions, rope_dims, 10_000.0, [2, 1, 1]);
        let got = tables.apply(&x);
        let want = x.rope(rope_dims, false, 10_000.0, 1.0, 3);
        let diff = got
            .subtract(&want)
            .abs()
            .max_axis(-1, false)
            .max_axis(-1, false);
        let diff = diff.max_axis(-1, false).max_axis(-1, false).item_f32();
        crate::check_last_error().unwrap();
        assert!(diff < 1e-5, "max difference {diff}");
    }
}
