//! Ring attention implementation for context parallelism.
//!
//! Each rank holds a local chunk of the sequence. KV blocks are passed
//! around the ring, and partial attention is computed at each step.
//! The running output is accumulated using online softmax normalization
//! so the final result is mathematically equivalent to full attention.
//!
//! # Algorithm (Pass-KV mode)
//!
//! ```text
//! For step in 0..world_size:
//!   1. Compute local attention: score = Q_local @ K_remote.T / sqrt(d)
//!   2. Apply causal mask (if applicable)
//!   3. Compute local softmax: exp_score, local_lse
//!   4. Update running output using online softmax correction:
//!      - new_lse = log(exp(old_lse) + exp(local_lse))
//!      - output = output * exp(old_lse - new_lse) + local_out * exp(local_lse - new_lse)
//!   5. Send K,V to next rank; receive K,V from previous rank
//! ```
//!
//! # Reference
//!
//! - Ring Attention with Blockwise Transformers (Liu et al., 2023)
//! - Striped Attention (Brandon et al., 2023)

use crate::mlx_dist::group::DistributedGroup;
use crate::mlx_dist::ops;
use pmetal_runtime::compat::{Array, Exception, ops as mlx_ops};

/// Context parallelism communication mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CPMode {
    /// Pass KV blocks around the ring. Each rank holds local Q.
    /// Best for prefill where sequence is split and KV is being built.
    PassKV,
    /// Pass Q blocks around the ring. Each rank holds full KV.
    /// Best for decode where KV cache is large and Q is small.
    PassQ,
}

/// Ring attention forward pass.
///
/// Computes attention where the sequence is split across ranks.
/// KV (or Q) blocks are exchanged in a ring pattern, with partial
/// attention accumulated using online softmax.
///
/// # Arguments
///
/// * `query` — Local Q chunk `[B, H, S_local, D]`
/// * `key` — Local K chunk `[B, H, S_local, D]` (will be exchanged)
/// * `value` — Local V chunk `[B, H, S_local, D]` (will be exchanged)
/// * `scale` — Attention scale factor (typically `1/sqrt(head_dim)`)
/// * `group` — Distributed group for ring communication
/// * `mode` — PassKV or PassQ
///
/// # Returns
///
/// Attention output `[B, H, S_local, D]` (same shape as query).
pub fn ring_attention_forward(
    query: &Array,
    key: &Array,
    value: &Array,
    scale: f32,
    group: &DistributedGroup,
    mode: CPMode,
) -> Result<Array, Exception> {
    match mode {
        CPMode::PassKV => ring_attention_pass_kv(query, key, value, scale, group),
        CPMode::PassQ => ring_attention_pass_q(query, key, value, scale, group),
    }
}

/// Ring attention with KV passing (prefill mode).
fn ring_attention_pass_kv(
    query: &Array,
    key: &Array,
    value: &Array,
    scale: f32,
    group: &DistributedGroup,
) -> Result<Array, Exception> {
    let world_size = group.size() as usize;
    let rank = group.rank() as usize;
    let next_rank = ((rank + 1) % world_size) as i32;
    let prev_rank = ((rank + world_size - 1) % world_size) as i32;

    // Initialize: compute attention with local KV.
    let (mut output, mut running_lse) = local_attention(query, key, value, scale)?;

    // Current KV being sent around the ring.
    let mut current_k = key.clone();
    let mut current_v = value.clone();

    // Ring exchange: world_size - 1 steps.
    for _step in 1..world_size {
        // Send current KV to next rank, receive from previous rank.
        let send_k = ops::send(&current_k, next_rank, Some(group))?;
        let send_v = ops::send(&current_v, next_rank, Some(group))?;
        let recv_k = ops::recv_like(&current_k, prev_rank, Some(group))?;
        let recv_v = ops::recv_like(&current_v, prev_rank, Some(group))?;

        // Evaluate sends and receives.
        send_k.eval();
        send_v.eval();
        recv_k.eval();
        recv_v.eval();

        // Compute attention with received KV block.
        let (step_output, step_lse) = local_attention(query, &recv_k, &recv_v, scale)?;

        // Online softmax update:
        // new_lse = log(exp(running_lse) + exp(step_lse))
        // output = output * exp(running_lse - new_lse) + step_output * exp(step_lse - new_lse)
        let (new_output, new_lse) =
            online_softmax_update(&output, &running_lse, &step_output, &step_lse)?;

        output = new_output;
        running_lse = new_lse;
        current_k = recv_k;
        current_v = recv_v;
    }

    Ok(output)
}

/// Ring attention with Q passing (decode mode).
fn ring_attention_pass_q(
    query: &Array,
    key: &Array,
    value: &Array,
    scale: f32,
    group: &DistributedGroup,
) -> Result<Array, Exception> {
    let world_size = group.size() as usize;
    let rank = group.rank() as usize;
    let next_rank = ((rank + 1) % world_size) as i32;
    let prev_rank = ((rank + world_size - 1) % world_size) as i32;

    // In Pass-Q mode, each rank has full KV. Q is passed around.
    let (mut output, mut running_lse) = local_attention(query, key, value, scale)?;

    let mut current_q = query.clone();

    for _step in 1..world_size {
        let send_q = ops::send(&current_q, next_rank, Some(group))?;
        let recv_q = ops::recv_like(&current_q, prev_rank, Some(group))?;
        send_q.eval();
        recv_q.eval();

        let (step_output, step_lse) = local_attention(&recv_q, key, value, scale)?;
        let (new_output, new_lse) =
            online_softmax_update(&output, &running_lse, &step_output, &step_lse)?;

        output = new_output;
        running_lse = new_lse;
        current_q = recv_q;
    }

    Ok(output)
}

/// Compute local attention and return (output, log_sum_exp).
///
/// Attention: softmax(Q @ K.T * scale) @ V
/// Also returns the log-sum-exp for online softmax accumulation.
fn local_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
) -> Result<(Array, Array), Exception> {
    // scores = Q @ K^T * scale  → [B, H, S_q, S_k]
    // Transpose last two dims: ndim-2 and ndim-1.
    let ndim = k.ndim();
    let mut axes: Vec<i32> = (0..ndim).collect();
    let last = axes.len() - 1;
    let second_last = last - 1;
    axes.swap(second_last, last);
    let kt = k.transpose_axes(&axes);

    let scale_arr = Array::from_f32_slice(&[scale], &[1]);
    // matmul and multiply both return Array (no error).
    let scores = q.matmul(&kt).multiply(&scale_arr);

    // log_sum_exp for online softmax: [B, H, S_q, 1]
    let lse = scores.logsumexp(-1, true);

    // Numerically stable softmax: subtract lse before exp.
    let scores_shifted = scores.subtract(&lse);
    let weights = scores_shifted.exp();

    // Output = weights @ V  → [B, H, S_q, D]
    let output = weights.matmul(v);

    Ok((output, lse))
}

/// Online softmax update: combine two partial attention results.
///
/// Given two partial results (output1, lse1) and (output2, lse2),
/// compute the combined result that is equivalent to computing
/// attention over the concatenation of their key/value sets.
fn online_softmax_update(
    output1: &Array,
    lse1: &Array,
    output2: &Array,
    lse2: &Array,
) -> Result<(Array, Array), Exception> {
    // new_lse = log(exp(lse1) + exp(lse2))
    // = max(lse1, lse2) + log(exp(lse1 - max) + exp(lse2 - max))
    let max_lse = mlx_ops::maximum(lse1, lse2);
    let diff1 = lse1.subtract(&max_lse);
    let diff2 = lse2.subtract(&max_lse);
    // exp() returns Array directly.
    let sum_exp = diff1.exp().add(&diff2.exp());
    let new_lse = max_lse.add(&sum_exp.log());

    // Correction factors.
    let correction1 = lse1.subtract(&new_lse).exp();
    let correction2 = lse2.subtract(&new_lse).exp();

    // Combined output.
    let new_output = output1
        .multiply(&correction1)
        .add(&output2.multiply(&correction2));

    Ok((new_output, new_lse))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_attention_basic_shape() {
        // [1, 2, 4, 8] — batch=1, heads=2, seq=4, dim=8
        let q = Array::from_f32_slice(&vec![0.1f32; 64], &[1, 2, 4, 8]);
        let k = Array::from_f32_slice(&vec![0.1f32; 64], &[1, 2, 4, 8]);
        let v = Array::from_f32_slice(&vec![0.1f32; 64], &[1, 2, 4, 8]);

        let (out, lse) = local_attention(&q, &k, &v, 0.125).unwrap();
        assert_eq!(out.shape(), &[1, 2, 4, 8]);
        assert_eq!(lse.shape(), &[1, 2, 4, 1]);
    }

    #[test]
    fn online_softmax_update_combines() {
        let out1 = Array::from_f32_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let lse1 = Array::from_f32_slice(&[0.5f32], &[1, 1, 1, 1]);
        let out2 = Array::from_f32_slice(&[3.0f32, 4.0], &[1, 1, 1, 2]);
        let lse2 = Array::from_f32_slice(&[0.5f32], &[1, 1, 1, 1]);

        let (mut combined, new_lse) = online_softmax_update(&out1, &lse1, &out2, &lse2).unwrap();

        combined.eval();
        new_lse.eval();

        // Equal LSEs → equal weighting → output = (out1 + out2) / 2
        let data = combined.to_f32_vec(2).unwrap();
        assert!((data[0] - 2.0).abs() < 0.01);
        assert!((data[1] - 3.0).abs() < 0.01);
    }
}
