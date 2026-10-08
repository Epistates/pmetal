//! MLX MoE combine operations.
//!
//! Pure MLX ops for MoE expert output combination. The previous Metal side-channel
//! (`fused_moe_combine`) has been removed — the 6 MLX ops are already async on GPU
//! and adding synchronization barriers (4x eval() + waitUntilCompleted) made the
//! Metal path 5-20x slower than the MLX ops it replaced.

use pmetal_bridge::compat::{Array, Dtype, Exception, random};

/// MoE combine: weighted expert sum + sigmoid-gated shared expert.
///
/// Computes:
/// ```text
/// y = (expert_outs * weights.unsqueeze(-1)).sum(-2)  // weighted sum
/// shared_gate = sigmoid(shared_gate_logit)
/// y += shared_gate * shared_out
/// ```
///
/// This is the MoE block's output alone. The decoder layer adds its own
/// residual, the same as it does for a dense MLP; adding the block's input here
/// as well counted the normalized hidden state twice.
///
/// All ops run asynchronously on GPU via MLX's lazy evaluation graph.
///
/// # Arguments
/// * `expert_outs` - Expert outputs `[batch_seq, K, D]`
/// * `expert_weights` - Routing weights `[batch_seq, K]`
/// * `shared_out` - Shared expert output `[batch_seq, D]`
/// * `shared_gate_logit` - Shared expert gate logit `[batch_seq, 1]` or scalar
/// * `k` - Number of active experts (top-k)
/// * `batch_seq` - Batch * sequence length
pub fn moe_combine_mlx(
    expert_outs: &Array,
    expert_weights: &Array,
    shared_out: &Array,
    shared_gate_logit: &Array,
    k: i32,
    batch_seq: i32,
) -> Result<Array, Exception> {
    let y = expert_outs
        .multiply(&expert_weights.reshape(&[batch_seq, k, 1]))
        .sum_axis(-2, false);

    // Shared expert with gate
    let shared_gate = shared_gate_logit.sigmoid();
    let shared_y = shared_gate.multiply(shared_out);

    Ok(y.add(&shared_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_moe_combine_mlx_basic() {
        let dim = 64i32;
        let k = 4i32;
        let batch_seq = 1i32;

        let expert_outs = random::normal(&[batch_seq, k, dim], Dtype::Float32);
        let expert_weights = Array::from_f32_slice(&[0.3f32, 0.25, 0.25, 0.2], &[batch_seq, k]);
        let shared_out = random::normal(&[batch_seq, dim], Dtype::Float32);
        let shared_gate_logit = Array::from_f32(0.5);

        let result = moe_combine_mlx(
            &expert_outs,
            &expert_weights,
            &shared_out,
            &shared_gate_logit,
            k,
            batch_seq,
        )
        .unwrap();

        result.eval();
        assert_eq!(result.shape(), &[batch_seq, dim]);

        // Verify no NaN
        let data: Vec<f32> = result.as_slice().to_vec();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "NaN at index {}", i);
        }
    }

    #[test]
    #[serial]
    fn test_moe_combine_mlx_batched() {
        let dim = 32i32;
        let k = 2i32;
        let batch_seq = 4i32;

        let expert_outs = random::normal(&[batch_seq, k, dim], Dtype::Float32);
        let expert_weights = random::normal(&[batch_seq, k], Dtype::Float32);
        let shared_out = random::normal(&[batch_seq, dim], Dtype::Float32);
        let shared_gate_logit = random::normal(&[batch_seq, 1], Dtype::Float32);

        let result = moe_combine_mlx(
            &expert_outs,
            &expert_weights,
            &shared_out,
            &shared_gate_logit,
            k,
            batch_seq,
        )
        .unwrap();

        result.eval();
        assert_eq!(result.shape(), &[batch_seq, dim]);
    }

    /// The combine is the MoE block's output, not the layer's: the decoder adds
    /// the residual, so with silent experts and a silent shared expert nothing
    /// comes out. It used to add the block input back, which a decoder layer
    /// then added a second time.
    #[test]
    #[serial]
    fn test_moe_combine_mlx_adds_no_residual() {
        let (dim, k, batch_seq) = (8i32, 2i32, 3i32);
        let expert_outs = Array::zeros_f32(&[batch_seq, k, dim]);
        let expert_weights = Array::from_f32_slice(&[0.5f32; 6], &[batch_seq, k]);
        let shared_out = Array::zeros_f32(&[batch_seq, dim]);
        let shared_gate_logit = Array::zeros_f32(&[batch_seq, 1]);

        let result = moe_combine_mlx(
            &expert_outs,
            &expert_weights,
            &shared_out,
            &shared_gate_logit,
            k,
            batch_seq,
        )
        .unwrap();
        result.eval();
        let data: Vec<f32> = result.as_slice().to_vec();
        assert!(
            data.iter().all(|&v| v == 0.0),
            "combine leaked a residual: {data:?}"
        );
    }
}
