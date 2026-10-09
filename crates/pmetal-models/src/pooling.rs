//! Pooling strategies for converting sequence representations to fixed-size embeddings.

use pmetal_bridge::compat::{Array, Exception};
/// Pooling modes for sentence embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PoolingMode {
    /// Mean of all non-padding tokens.
    #[default]
    Mean,
    /// First token ([CLS] for BERT, <s> for RoBERTa).
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
/// # Arguments
/// * `hidden_states` - `[batch, seq_len, hidden_dim]`
/// * `attention_mask` - `[batch, seq_len]` (1 for real tokens, 0 for padding)
/// * `mode` - Pooling strategy
///
/// # Returns
/// Embedding tensor `[batch, hidden_dim]`
pub fn pool(
    hidden_states: &Array,
    attention_mask: &Array,
    mode: PoolingMode,
) -> Result<Array, Exception> {
    let batch = hidden_states.dim(0);
    let seq_len = hidden_states.dim(1);

    match mode {
        PoolingMode::Mean => {
            // Expand mask to [batch, seq_len, 1] for broadcasting
            let mask_expanded = attention_mask
                .reshape(&[batch, seq_len, 1])
                .as_dtype(hidden_states.dtype().as_i32());
            // Zero out padding positions, sum over sequence, divide by token count
            let masked = hidden_states.multiply(&mask_expanded);
            let sum = masked.sum_axes(&[1], false);
            let count = mask_expanded.sum_axes(&[1], false);
            let count = pmetal_bridge::compat::ops::maximum(&count, &Array::from_f32(1e-9));
            Ok(sum.divide(&count))
        }
        PoolingMode::Cls => {
            // Take first token embedding [batch, hidden_dim]
            Ok(pmetal_bridge::compat::ops::slice_axis(hidden_states, 1, 0, 1).squeeze_axes(&[1]))
        }
        PoolingMode::Max => {
            // Replace padding positions with -inf then take max over sequence
            let mask_expanded = attention_mask
                .reshape(&[batch, seq_len, 1])
                .as_dtype(hidden_states.dtype().as_i32());
            // inv_mask is 1 where padding, 0 where real token
            let inv_mask = Array::from_f32(1.0).subtract(&mask_expanded);
            let neg_inf = Array::from_f32(-1e9);
            let masked = hidden_states.add(&inv_mask.multiply(&neg_inf));
            Ok(masked.max_axis(1, false))
        }
        PoolingMode::LastToken => {
            // Vectorized gather: compute last non-padding position per sequence and
            // use take_along_axis to extract in a single operation (no Python-style loop).
            let hidden_dim = hidden_states.dim(2);

            // lengths [batch] → last index [batch] as Int32
            let last_indices = attention_mask
                .as_dtype(pmetal_bridge::compat::Dtype::Int32.as_i32())
                .sum_axes(&[1], false) // [batch]
                .subtract(&Array::from_int(1)); // [batch]

            // Reshape to [batch, 1, 1] then broadcast to [batch, 1, hidden_dim]
            // so take_along_axis can gather along the sequence dimension.
            let indices_expanded = last_indices.reshape(&[batch, 1, 1]);
            let indices_broadcast = pmetal_bridge::compat::ops::broadcast_to(
                &indices_expanded,
                &[batch, 1, hidden_dim],
            );

            // take_along_axis(hidden_states, indices, axis=1) → [batch, 1, hidden_dim]
            let gathered = hidden_states.take_along_axis(&indices_broadcast, 1);

            // Squeeze the singleton seq dimension → [batch, hidden_dim]
            Ok(gathered.squeeze_axes(&[1]))
        }
        PoolingMode::WeightedMean => {
            // Linearly increasing weights: position i gets weight (i+1)
            let weights: Vec<f32> = (1..=seq_len).map(|i| i as f32).collect();
            let weights = Array::from_slice(&weights, &[1, seq_len, 1])
                .as_dtype(hidden_states.dtype().as_i32());
            let mask_expanded = attention_mask
                .reshape(&[batch, seq_len, 1])
                .as_dtype(hidden_states.dtype().as_i32());
            let weighted = hidden_states.multiply(&weights).multiply(&mask_expanded);
            let sum = weighted.sum_axes(&[1], false);
            let weight_sum = weights.multiply(&mask_expanded).sum_axes(&[1], false);
            let weight_sum =
                pmetal_bridge::compat::ops::maximum(&weight_sum, &Array::from_f32(1e-9));
            Ok(sum.divide(&weight_sum))
        }
    }
}

/// L2-normalize embeddings along the last axis.
///
/// # Arguments
/// * `embeddings` - `[batch, dim]`
///
/// # Returns
/// Unit-normalized embeddings `[batch, dim]`
pub fn normalize_embeddings(embeddings: &Array) -> Result<Array, Exception> {
    let norm = embeddings.square().sum_axes(&[-1], true).sqrt();
    let norm = pmetal_bridge::compat::ops::maximum(&norm, &Array::from_f32(1e-12));
    Ok(embeddings.divide(&norm))
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
