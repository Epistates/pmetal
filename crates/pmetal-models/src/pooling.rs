//! Pooling strategies for converting sequence representations to fixed-size embeddings.

use pmetal_runtime::compat::{Array, Dtype, Exception, ops};
/// Pooling modes for sentence embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PoolingMode {
    /// Mean of all non-padding tokens.
    #[default]
    Mean,
    /// First token (`[CLS]` for BERT, `<s>` for RoBERTa).
    Cls,
    /// Element-wise max across sequence dimension.
    Max,
    /// Last non-padding token (useful for causal models used as embedders).
    LastToken,
    /// Weighted mean with linearly increasing position weights.
    WeightedMean,
}

impl std::fmt::Display for PoolingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Cls => write!(f, "cls"),
            Self::Max => write!(f, "max"),
            Self::LastToken => write!(f, "last_token"),
            Self::WeightedMean => write!(f, "weighted_mean"),
        }
    }
}

/// Pool sequence hidden states into a single embedding vector.
///
/// The arithmetic runs in f32 whatever the model's dtype and the result is cast
/// back to it: summing a long sequence, or weighting positions up to the
/// sequence length, loses real precision in bf16 (257 is not representable),
/// and an f32 scalar mixed into bf16 operands silently promotes the whole
/// result to f32.
///
/// # Arguments
/// * `hidden_states` - `[batch, seq_len, hidden_dim]`
/// * `attention_mask` - `[batch, seq_len]` (1 for real tokens, 0 for padding)
/// * `mode` - Pooling strategy
///
/// # Returns
/// Embedding tensor `[batch, hidden_dim]`, in the dtype of `hidden_states`.
pub fn pool(
    hidden_states: &Array,
    attention_mask: &Array,
    mode: PoolingMode,
) -> Result<Array, Exception> {
    let batch = hidden_states.dim(0);
    let seq_len = hidden_states.dim(1);
    let out_dtype = hidden_states.dtype().as_i32();
    let f32_dtype = Dtype::Float32.as_i32();
    let hidden = hidden_states.as_dtype(f32_dtype);
    // `[batch, seq_len, 1]`, 1.0 where a real token sits.
    let mask = attention_mask
        .reshape(&[batch, seq_len, 1])
        .as_dtype(f32_dtype);
    let tiny = Array::from_f32(1e-9);

    let pooled = match mode {
        PoolingMode::Mean => {
            let sum = hidden.multiply(&mask).sum_axes(&[1], false);
            let count = ops::maximum(&mask.sum_axes(&[1], false), &tiny);
            sum.divide(&count)
        }
        PoolingMode::Cls => ops::slice_axis(&hidden, 1, 0, 1).squeeze_axes(&[1]),
        PoolingMode::Max => {
            // Padding positions are pushed far below any real state.
            let padding = Array::from_f32(1.0).subtract(&mask);
            hidden
                .add(&padding.multiply(&Array::from_f32(-1e9)))
                .max_axis(1, false)
        }
        PoolingMode::LastToken => {
            // The last *attended* position, wherever the padding is: right
            // padding ends the sequence early, left padding (what last-token
            // embedders use) starts it late, and `count - 1` is right only for
            // the first. A row with nothing attended reads position 0.
            let positions = ops::arange(seq_len, Dtype::Int32).reshape(&[1, seq_len]);
            let attended = attention_mask
                .as_dtype(Dtype::Int32.as_i32())
                .gt(&Array::from_int(0));
            let last =
                ops::where_fn(&attended, &positions, &Array::from_int(-1)).max_axis(1, false);
            let last = ops::maximum(&last, &Array::from_int(0));
            let hidden_dim = hidden.dim(2);
            let indices = ops::broadcast_to(&last.reshape(&[batch, 1, 1]), &[batch, 1, hidden_dim]);
            hidden.take_along_axis(&indices, 1).squeeze_axes(&[1])
        }
        PoolingMode::WeightedMean => {
            // Linearly increasing weights: position i gets weight (i+1)
            let weights = ops::arange(seq_len, Dtype::Float32)
                .add(&Array::from_f32(1.0))
                .reshape(&[1, seq_len, 1]);
            let sum = hidden
                .multiply(&weights)
                .multiply(&mask)
                .sum_axes(&[1], false);
            let weight_sum = ops::maximum(&weights.multiply(&mask).sum_axes(&[1], false), &tiny);
            sum.divide(&weight_sum)
        }
    };
    Ok(pooled.as_dtype(out_dtype))
}

/// L2-normalize embeddings along the last axis.
///
/// Computed in f32 and returned in the dtype of `embeddings`.
///
/// # Arguments
/// * `embeddings` - `[batch, dim]`
///
/// # Returns
/// Unit-normalized embeddings `[batch, dim]`
pub fn normalize_embeddings(embeddings: &Array) -> Result<Array, Exception> {
    let wide = embeddings.as_dtype(Dtype::Float32.as_i32());
    let norm = wide.square().sum_axes(&[-1], true).sqrt();
    let norm = ops::maximum(&norm, &Array::from_f32(1e-12));
    Ok(wide.divide(&norm).as_dtype(embeddings.dtype().as_i32()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hidden(batch: i32, seq: i32, dim: i32) -> Array {
        Array::from_slice(
            &vec![1.0f32; (batch * seq * dim) as usize],
            &[batch, seq, dim],
        )
    }

    #[test]
    fn test_mean_pooling_no_padding() {
        let hidden = make_hidden(2, 4, 8);
        let mask = Array::from_slice(&[1i32, 1, 1, 1, 1, 1, 1, 1], &[2, 4]);
        let out = pool(&hidden, &mask, PoolingMode::Mean).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[2, 8]);
        // Mean of all-ones is 1.0 — check first element
        let v: f32 = out.flatten(0, -1).index(0).item();
        assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {}", v);
    }

    #[test]
    fn test_cls_pooling() {
        let batch = 2i32;
        let seq = 4i32;
        let dim = 8i32;
        let mut data = vec![1.0f32; (batch * seq * dim) as usize];
        for b in 0..batch as usize {
            for d in 0..dim as usize {
                data[b * seq as usize * dim as usize + d] = 2.0;
            }
        }
        let hidden = Array::from_slice(&data, &[batch, seq, dim]);
        let mask = Array::from_slice(&[1i32, 1, 1, 1, 1, 1, 1, 1], &[2, 4]);
        let out = pool(&hidden, &mask, PoolingMode::Cls).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[2, 8]);
        let v: f32 = out.flatten(0, -1).index(0).item();
        assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {}", v);
    }

    /// `[batch, seq, 1]` hidden states whose value at position `p` is `p`, so a
    /// pooled value names the position it came from.
    fn positional(batch: i32, seq: i32) -> Array {
        let values: Vec<f32> = (0..batch * seq).map(|i| (i % seq) as f32).collect();
        Array::from_slice(&values, &[batch, seq, 1])
    }

    fn values(array: &Array) -> Vec<f32> {
        let n = array.size();
        let mut array = array.as_dtype(Dtype::Float32.as_i32());
        array.to_f32_vec(n).unwrap()
    }

    /// ⚠️ The last token is the last *attended* one, wherever the padding is.
    /// Qwen3-Embedding (and every other last-token embedder) pads on the left,
    /// where `count - 1` indexes a pad position instead of the final token.
    #[test]
    fn last_token_pooling_handles_left_and_right_padding() {
        let hidden = positional(3, 5);
        let mask = Array::from_slice(
            &[
                0i32, 0, 1, 1, 1, // left-padded: last real token is position 4
                1, 1, 1, 0, 0, // right-padded: position 2
                1, 1, 1, 1, 1, // unpadded: position 4
            ],
            &[3, 5],
        );
        let pooled = pool(&hidden, &mask, PoolingMode::LastToken).unwrap();
        assert_eq!(pooled.shape(), &[3, 1]);
        assert_eq!(values(&pooled), vec![4.0, 2.0, 4.0]);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// A row with nothing attended pools to position 0 rather than reading
    /// out of bounds.
    #[test]
    fn last_token_pooling_of_an_empty_row_stays_in_bounds() {
        let hidden = positional(1, 4);
        let mask = Array::from_slice(&[0i32, 0, 0, 0], &[1, 4]);
        let pooled = pool(&hidden, &mask, PoolingMode::LastToken).unwrap();
        assert_eq!(values(&pooled), vec![0.0]);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// Pooling a bf16 model's states must hand back bf16, not an f32 the
    /// scalars promoted it to.
    #[test]
    fn pooling_keeps_the_dtype_of_the_hidden_states() {
        let hidden = positional(2, 4).as_dtype(Dtype::Bfloat16.as_i32());
        let mask = Array::from_slice(&[1i32, 1, 1, 0, 1, 1, 1, 1], &[2, 4]);
        for mode in [
            PoolingMode::Mean,
            PoolingMode::Cls,
            PoolingMode::Max,
            PoolingMode::LastToken,
            PoolingMode::WeightedMean,
        ] {
            let pooled = pool(&hidden, &mask, mode).unwrap();
            assert_eq!(pooled.dtype(), Dtype::Bfloat16, "{mode}");
        }
        let normed = normalize_embeddings(&hidden.reshape(&[2, 4])).unwrap();
        assert_eq!(normed.dtype(), Dtype::Bfloat16);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// Mean and weighted-mean over a long bf16 sequence accumulate in f32: the
    /// position weights 1..=seq are not all representable in bf16 (257 is not),
    /// so weighting in the model dtype shifts the answer.
    #[test]
    fn weighted_mean_is_exact_over_a_long_bf16_sequence() {
        let seq = 600;
        let hidden = Array::from_slice(&vec![1.0f32; seq as usize], &[1, seq, 1])
            .as_dtype(Dtype::Bfloat16.as_i32());
        let mask = Array::from_slice(&vec![1i32; seq as usize], &[1, seq]);
        let pooled = pool(&hidden, &mask, PoolingMode::WeightedMean).unwrap();
        // Every state is 1, so any weighting of them is 1.
        assert_eq!(values(&pooled), vec![1.0]);
        let mean = pool(&hidden, &mask, PoolingMode::Mean).unwrap();
        assert_eq!(values(&mean), vec![1.0]);
    }

    #[test]
    fn mean_and_weighted_mean_skip_padding() {
        let hidden = positional(1, 4); // 0 1 2 3
        let mask = Array::from_slice(&[1i32, 1, 0, 0], &[1, 4]);
        let mean = values(&pool(&hidden, &mask, PoolingMode::Mean).unwrap());
        assert!((mean[0] - 0.5).abs() < 1e-6, "{mean:?}");
        // Weights 1 and 2 on states 0 and 1: (0*1 + 1*2) / (1 + 2).
        let weighted = values(&pool(&hidden, &mask, PoolingMode::WeightedMean).unwrap());
        assert!((weighted[0] - 2.0 / 3.0).abs() < 1e-6, "{weighted:?}");
    }

    #[test]
    fn max_pooling_ignores_padding() {
        let hidden = positional(1, 4);
        let mask = Array::from_slice(&[1i32, 1, 0, 0], &[1, 4]);
        let pooled = pool(&hidden, &mask, PoolingMode::Max).unwrap();
        assert_eq!(values(&pooled), vec![1.0]);
    }

    #[test]
    fn test_normalize_embeddings() {
        let emb = Array::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let normed = normalize_embeddings(&emb).unwrap();
        normed.eval().unwrap();
        // norm = 5.0, so [0.6, 0.8]
        let flat = normed.flatten(0, -1);
        let v0: f32 = flat.index(0).item();
        let v1: f32 = flat.index(1).item();
        assert!((v0 - 0.6).abs() < 1e-5);
        assert!((v1 - 0.8).abs() < 1e-5);
    }
}
