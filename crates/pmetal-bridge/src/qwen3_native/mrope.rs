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
//! here: they rotate through [`RotaryEmbedding`], and only a prompt with media
//! builds these tables, once per forward, shared by every full-attention
//! layer. The frequencies and attention factor are the embedding's own, so a
//! scaled (YaRN) rotation splits across the axes the same way a plain one
//! does.

use crate::compat::{Array, Dtype, ops};
use crate::rope::{RotaryEmbedding, rotate_with_cos_sin};

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

impl MropeTables {
    /// Build the tables for `positions`, a `[3, L]` int32 array of temporal,
    /// row and column positions, from `rope`'s frequencies and attention
    /// factor (transformers multiplies both `cos` and `sin` by it).
    pub fn new(positions: &Array, rope: &RotaryEmbedding, section: [usize; 3]) -> Self {
        let reach = if rope.rotary().scaling.depends_on_reach() {
            positions
                .as_dtype(Dtype::Float32.as_i32())
                .max(None)
                .item_f32() as i64
                + 1
        } else {
            0
        };
        Self::with_frequencies(
            positions,
            rope.dims(),
            &rope.inverse_frequencies(reach),
            rope.attention_factor(),
            section,
        )
    }

    fn with_frequencies(
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
        rotate_with_cos_sin(
            &x.as_dtype(Dtype::Float32.as_i32()),
            &self.cos,
            &self.sin,
            self.rope_dims,
            false,
        )
        .as_dtype(x.dtype().as_i32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::{RopeConfig, Rotary};

    fn max_diff(a: &Array, b: &Array) -> f32 {
        a.subtract(b)
            .abs()
            .max_axis(-1, false)
            .max_axis(-1, false)
            .max_axis(-1, false)
            .max_axis(-1, false)
            .item_f32()
    }

    /// Three equal position axes are ordinary RoPE at that position, YaRN's
    /// frequencies and attention factor included: the tables agree with the
    /// embedding's own contiguous rotation.
    #[test]
    fn equal_axes_are_the_embeddings_own_rotation() {
        let (heads, len, head_dim) = (2, 6, 32);
        let plain = serde_json::json!({"rope_theta": 10000.0, "partial_rotary_factor": 0.25});
        let yarn = serde_json::json!({
            "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                "original_max_position_embeddings": 16.0}
        });
        let x = crate::compat::random::uniform_f32(&[1, heads, len, head_dim]);
        let offset = 20;
        let three: Vec<i32> = (0..3).flat_map(|_| offset..offset + len).collect();
        let positions = Array::from_i32_slice_shaped(&three, &[3, len]);
        for config in [plain, yarn] {
            let rotary =
                Rotary::from_config(head_dim, RopeConfig::from_json(&config), 1e4, 1.0).unwrap();
            let rope = RotaryEmbedding::new(rotary, false);
            let want = rope.apply(&x, offset);
            let got = MropeTables::new(&positions, &rope, [2, 1, 1]).apply(&x);
            let d = max_diff(&got, &want);
            // The untouched channels keep their values, unscaled.
            let pass = max_diff(
                &ops::slice_axis(&got, 3, 8, head_dim),
                &ops::slice_axis(&x, 3, 8, head_dim),
            );
            crate::check_last_error().unwrap();
            assert!(d < 1e-5, "{config}: tables vs embedding {d}");
            assert_eq!(pass, 0.0);
        }
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
}
