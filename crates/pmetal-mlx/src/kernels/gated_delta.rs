//! Gated Delta Network (GDN) linear attention implementation.
//!
//! Implements the delta-rule-based recurrence for Qwen 3.5's linear attention layers.
//! The GDN replaces quadratic softmax attention with a linear recurrence:
//!
//!   state = state * g + k^T * (beta * (v - state @ k))
//!   y = state @ q
//!
//! This gives O(1) per-token generation cost vs O(n) for standard attention.
//!
//! # Implementations
//!
//! - `gated_delta_ops`: Pure MLX ops sequential loop (used for prefill and training)
//! - `gated_delta_update`: Top-level API that dispatches between ops and kernel paths
//!
//! # Shapes
//!
//! - q, k: `[B, T, Hk, Dk]` (key heads, may differ from value heads)
//! - v: `[B, T, Hv, Dv]`
//! - g: `[B, T, Hv]` (scalar gating) or `[B, T, Hv, Dk]` (vectorized)
//! - beta: `[B, T, Hv]`
//! - state: `[B, Hv, Dv, Dk]`
//!
//! # Reference
//!
//! Ported from `mlx-lm/models/gated_delta.py` (Apple, 2025).

use mlx_rs::{
    Array, Dtype,
    error::Exception,
    nn,
    ops::{self, indexing::IndexOp},
};

/// Compute gating decay: g = exp(-exp(A_log) * softplus(a + dt_bias))
///
/// Operates in f32 for numerical stability, then casts back to input dtype.
///
/// # Arguments
/// * `a_log` - Log of decay rates, shape `[Hv]`
/// * `a` - Per-token gating input, shape `[B, T, Hv]`
/// * `dt_bias` - Learnable bias, shape `[Hv]`
///
/// # Returns
/// Gating decay values, shape `[B, T, Hv]`
pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let input_dtype = a_log.dtype();

    // Upcast to f32 for stability
    let a_log_f32 = if input_dtype != Dtype::Float32 {
        a_log.as_type::<f32>()?
    } else {
        a_log.clone()
    };

    // exp(A_log) gives the decay rate A
    let decay_rate = a_log_f32.exp()?;

    // softplus(a + dt_bias)
    let a_biased = a.add(dt_bias)?;
    let sp = nn::softplus(&a_biased)?;

    // g = exp(-A * softplus(a + dt_bias))
    let g = decay_rate.multiply(&sp)?.negative()?.exp()?;

    // Cast back to input dtype
    if input_dtype != Dtype::Float32 {
        g.as_dtype(input_dtype)
    } else {
        Ok(g)
    }
}

/// Single recurrent step (ops-based reference).
///
/// Computes one step of the delta-rule recurrence using standard MLX ops.
///
/// # Arguments
/// * `q`, `k` - Query/key for this step, shape `[B, H, Dk]`
/// * `v` - Value for this step, shape `[B, H, Dv]`
/// * `g` - Gating decay, shape `[B, H]` or `[B, H, Dk]`
/// * `beta` - Beta gate, shape `[B, H]`
/// * `state` - Current recurrent state, shape `[B, H, Dv, Dk]`
/// * `mask` - Optional mask for this step, shape `[B]`
///
/// # Returns
/// (output `[B, H, Dv]`, new_state `[B, H, Dv, Dk]`)
fn gated_delta_step_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let old_state = state.clone();

    // Decay: state = state * g
    // g can be [B, H] (scalar) or [B, H, Dk] (vectorized)
    let decayed_state = match g.ndim() {
        2 => {
            // [B, H] -> [B, H, 1, 1] for broadcasting with [B, H, Dv, Dk]
            let g_expanded = g.reshape(&[g.dim(0), g.dim(1), 1, 1])?;
            state.multiply(&g_expanded)?
        }
        3 => {
            // [B, H, Dk] -> [B, H, 1, Dk] for broadcasting with [B, H, Dv, Dk]
            let g_expanded = g.reshape(&[g.dim(0), g.dim(1), 1, g.dim(2)])?;
            state.multiply(&g_expanded)?
        }
        _ => {
            return Err(Exception::custom(format!(
                "Unsupported gating shape: {:?}",
                g.shape()
            )));
        }
    };

    // kv_mem = sum(decayed_state * k, axis=-1) -> [B, H, Dv]
    // k is [B, H, Dk], expand to [B, H, 1, Dk]
    let k_expanded = k.reshape(&[k.dim(0), k.dim(1), 1, k.dim(2)])?;
    let kv_mem = decayed_state.multiply(&k_expanded)?.sum_axis(-1, false)?;

    // delta = (v - kv_mem) * beta
    // v is [B, H, Dv], beta is [B, H] -> [B, H, 1]
    let beta_expanded = beta.reshape(&[beta.dim(0), beta.dim(1), 1])?;
    let delta = v.subtract(&kv_mem)?.multiply(&beta_expanded)?;

    // new_state = decayed_state + k^T * delta (outer product)
    // k_expanded: [B, H, 1, Dk], delta: [B, H, Dv] -> [B, H, Dv, 1]
    let delta_expanded = delta.reshape(&[delta.dim(0), delta.dim(1), delta.dim(2), 1])?;
    let new_state = decayed_state.add(&k_expanded.multiply(&delta_expanded)?)?;

    // y = sum(new_state * q, axis=-1) -> [B, H, Dv]
    let q_expanded = q.reshape(&[q.dim(0), q.dim(1), 1, q.dim(2)])?;
    let y = new_state.multiply(&q_expanded)?.sum_axis(-1, false)?;

    // Apply mask: if masked, keep old state
    let new_state = if let Some(mask) = mask {
        // mask is [B], expand to [B, 1, 1, 1] for broadcasting
        let mask_expanded = mask.reshape(&[mask.dim(0), 1, 1, 1])?;
        ops::r#where(&mask_expanded, &new_state, &old_state)?
    } else {
        new_state
    };

    Ok((y, new_state))
}

/// Ops-based sequential implementation for prompt prefill.
///
/// Processes each timestep sequentially through the recurrence.
/// Supports both scalar and vectorized gating, and GQA (Hv > Hk).
///
/// # Arguments
/// * `q`, `k` - Shape `[B, T, Hk, Dk]`
/// * `v` - Shape `[B, T, Hv, Dv]`
/// * `g` - Shape `[B, T, Hv]` (scalar) or `[B, T, Hv, Dk]` (vectorized)
/// * `beta` - Shape `[B, T, Hv]`
/// * `state` - Optional initial state `[B, Hv, Dv, Dk]` (zeros if None)
/// * `mask` - Optional mask `[B, T]`
///
/// # Returns
/// (output `[B, T, Hv, Dv]`, final_state `[B, Hv, Dv, Dk]`)
pub fn gated_delta_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let b = q.dim(0);
    let t = q.dim(1);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);

    // Initialize state if not provided
    let mut state = if let Some(s) = state {
        s.clone()
    } else {
        ops::zeros_dtype(&[b, hv, dv, dk], q.dtype())?
    };

    // Handle GQA: repeat q, k along head dim if Hv > Hk
    let repeat_factor = hv / hk;
    let (q_rep, k_rep);
    let (q, k) = if repeat_factor > 1 {
        q_rep = ops::repeat_axis::<f32>(q.clone(), repeat_factor, 2)?;
        k_rep = ops::repeat_axis::<f32>(k.clone(), repeat_factor, 2)?;
        (&q_rep, &k_rep)
    } else {
        (q, k)
    };

    let mut ys = Vec::with_capacity(t as usize);

    for t_idx in 0..t {
        // Slice timestep t: [B, 1, H, D] -> squeeze axis 1 -> [B, H, D]
        let q_t = q.index((.., t_idx..t_idx + 1, .., ..)).squeeze_axes(&[1])?;
        let k_t = k.index((.., t_idx..t_idx + 1, .., ..)).squeeze_axes(&[1])?;
        let v_t = v.index((.., t_idx..t_idx + 1, .., ..)).squeeze_axes(&[1])?;

        let g_t = if g.ndim() == 3 {
            g.index((.., t_idx..t_idx + 1, ..)).squeeze_axes(&[1])?
        } else {
            g.index((.., t_idx..t_idx + 1, .., ..)).squeeze_axes(&[1])?
        };

        let beta_t = beta.index((.., t_idx..t_idx + 1, ..)).squeeze_axes(&[1])?;

        let mask_t = mask.map(|m| m.index((.., t_idx..t_idx + 1)).squeeze_axes(&[1]));
        let mask_t = match mask_t {
            Some(Ok(m)) => Some(m),
            Some(Err(e)) => return Err(e),
            None => None,
        };

        let (y, new_state) = gated_delta_step_ops(
            &q_t,
            &k_t,
            &v_t,
            &g_t,
            &beta_t,
            &state,
            mask_t.as_ref(),
        )?;
        state = new_state;
        ys.push(y);
    }

    // Stack outputs: Vec<[B, Hv, Dv]> -> [B, T, Hv, Dv]
    let y_refs: Vec<&Array> = ys.iter().collect();
    let y = ops::stack_axis(&y_refs, 1)?;

    Ok((y, state))
}

/// Top-level GDN update API.
///
/// Computes the gated delta network recurrence for a sequence of tokens.
/// Handles beta/gate computation from raw inputs and dispatches to the
/// appropriate implementation.
///
/// # Arguments
/// * `q`, `k` - Query/key projections, shape `[B, T, Hk, Dk]`
/// * `v` - Value projection, shape `[B, T, Hv, Dv]`
/// * `a` - Raw gating input, shape `[B, T, Hv]`
/// * `b` - Raw beta input, shape `[B, T, Hv]`
/// * `a_log` - Log decay rates, shape `[Hv]`
/// * `dt_bias` - Learnable bias, shape `[Hv]`
/// * `state` - Optional initial state `[B, Hv, Dv, Dk]`
/// * `mask` - Optional mask `[B, T]`
///
/// # Returns
/// (output `[B, T, Hv, Dv]`, final_state `[B, Hv, Dv, Dk]`)
pub fn gated_delta_update(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    // beta = sigmoid(b)
    let beta = nn::sigmoid(b)?;

    // g = compute_g(A_log, a, dt_bias)
    let g = compute_g(a_log, a, dt_bias)?;

    // Initialize state if needed
    let init_state;
    let state_ref = match state {
        Some(s) => s,
        None => {
            let b_dim = q.dim(0);
            let dk = q.dim(3);
            let hv = v.dim(2);
            let dv = v.dim(3);
            init_state = ops::zeros_dtype(&[b_dim, hv, dv, dk], q.dtype())?;
            &init_state
        }
    };

    // Use ops-based implementation
    // TODO: Add Metal kernel dispatch for inference when mlx-rs adds metal_kernel bindings
    gated_delta_ops(q, k, v, &g, &beta, Some(state_ref), mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_compute_g_shape() {
        let a_log = Array::from_slice(&[0.5f32, 1.0, 1.5], &[3]);
        let a = mlx_rs::random::normal::<f32>(&[2, 4, 3], None, None, None).unwrap();
        let dt_bias = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]);

        let g = compute_g(&a_log, &a, &dt_bias).unwrap();
        assert_eq!(g.shape(), &[2, 4, 3]);
        // g should be in (0, 1] since it's exp(-positive)
        g.eval().unwrap();
        let g_data: Vec<f32> = g.as_slice().to_vec();
        for val in &g_data {
            assert!(*val > 0.0 && *val <= 1.0, "g value {} out of range", val);
        }
    }

    #[test]
    #[serial]
    fn test_gated_delta_ops_shape() {
        let b = 2;
        let t = 4;
        let hk = 2;
        let dk = 8;
        let hv = 4;
        let dv = 8;

        let q = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let k = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let v = mlx_rs::random::normal::<f32>(&[b, t, hv, dv], None, None, None).unwrap();
        let g = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[b, t, hv], None).unwrap();
        let beta = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[b, t, hv], None).unwrap();

        let (y, state) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        assert_eq!(y.shape(), &[b, t, hv, dv]);
        assert_eq!(state.shape(), &[b, hv, dv, dk]);
    }

    #[test]
    #[serial]
    fn test_gated_delta_update_shape() {
        let b = 1;
        let t = 3;
        let hk = 2;
        let dk = 8;
        let hv = 4;
        let dv = 8;

        let q = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let k = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let v = mlx_rs::random::normal::<f32>(&[b, t, hv, dv], None, None, None).unwrap();
        let a = mlx_rs::random::normal::<f32>(&[b, t, hv], None, None, None).unwrap();
        let b_input = mlx_rs::random::normal::<f32>(&[b, t, hv], None, None, None).unwrap();
        let a_log = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[hv]);
        let dt_bias = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[hv]);

        let (y, state) =
            gated_delta_update(&q, &k, &v, &a, &b_input, &a_log, &dt_bias, None, None).unwrap();

        assert_eq!(y.shape(), &[b, t, hv, dv]);
        assert_eq!(state.shape(), &[b, hv, dv, dk]);
    }

    #[test]
    #[serial]
    fn test_state_continuity() {
        // Verify that processing in two chunks gives the same result as one chunk
        let b = 1;
        let hk = 1;
        let dk = 4;
        let hv = 2;
        let dv = 4;

        mlx_rs::random::seed(42).unwrap();

        let q = mlx_rs::random::normal::<f32>(&[b, 4, hk, dk], None, None, None).unwrap();
        let k = mlx_rs::random::normal::<f32>(&[b, 4, hk, dk], None, None, None).unwrap();
        let v = mlx_rs::random::normal::<f32>(&[b, 4, hv, dv], None, None, None).unwrap();
        let g = mlx_rs::random::uniform::<_, f32>(0.5, 1.0, &[b, 4, hv], None).unwrap();
        let beta = mlx_rs::random::uniform::<_, f32>(0.0, 0.5, &[b, 4, hv], None).unwrap();

        // Process all 4 steps at once
        let (y_full, state_full) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Process in two chunks of 2
        let q1 = q.index((.., ..2, .., ..));
        let k1 = k.index((.., ..2, .., ..));
        let v1 = v.index((.., ..2, .., ..));
        let g1 = g.index((.., ..2, ..));
        let beta1 = beta.index((.., ..2, ..));

        let (_y1, state1) = gated_delta_ops(&q1, &k1, &v1, &g1, &beta1, None, None).unwrap();

        let q2 = q.index((.., 2.., .., ..));
        let k2 = k.index((.., 2.., .., ..));
        let v2 = v.index((.., 2.., .., ..));
        let g2 = g.index((.., 2.., ..));
        let beta2 = beta.index((.., 2.., ..));

        let (_y2, state2) =
            gated_delta_ops(&q2, &k2, &v2, &g2, &beta2, Some(&state1), None).unwrap();

        // States should match
        state_full.eval().unwrap();
        state2.eval().unwrap();
        let diff = state_full.subtract(&state2).unwrap().abs().unwrap();
        let max_diff = diff.max(None).unwrap();
        max_diff.eval().unwrap();
        let max_diff_val: f32 = max_diff.item();
        assert!(
            max_diff_val < 1e-4,
            "State mismatch: max diff = {}",
            max_diff_val
        );
    }
}
