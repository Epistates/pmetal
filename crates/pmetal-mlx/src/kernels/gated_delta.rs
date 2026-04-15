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
//! - `gated_delta_ops`: Pure MLX ops sequential loop (short sequences / decode)
//! - `gated_delta_chunk_ops`: Chunkwise parallel algorithm (training prefill, T > 64)
//! - `gated_delta_update`: Top-level API that dispatches between the two paths
//!
//! The chunkwise parallel algorithm (WY factorization from FLA, ICLR 2025) splits
//! the sequence into chunks of 64 and parallelizes intra-chunk computation via
//! matrix operations, reducing sequential steps from O(T) to O(T/64).
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
//! - Ported from `mlx-lm/models/gated_delta.py` (Apple, 2025).
//! - Chunkwise algorithm: Yang et al., "Gated Linear Attention Transformers with
//!   Hardware-Efficient Training" (FLA, ICLR 2025).

use crate::array_ext::ArrayDtypeExt;
use pmetal_bridge::compat::{Array, Dtype, Exception, linalg, ops};

/// Default chunk size for the chunkwise parallel GDN algorithm.
/// Sequences longer than this use the parallel chunk path.
const DEFAULT_GDN_CHUNK_SIZE: i32 = 64;
const GDN_CHUNK_SIZE_ENV_VAR: &str = "PMETAL_GDN_CHUNK_SIZE";

fn sanitize_gdn_chunk_size(chunk_size: i32) -> i32 {
    chunk_size.clamp(16, 512)
}

fn configured_gdn_chunk_size() -> i32 {
    std::env::var(GDN_CHUNK_SIZE_ENV_VAR)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .map(sanitize_gdn_chunk_size)
        .unwrap_or(DEFAULT_GDN_CHUNK_SIZE)
}

fn resolve_chunk_size_override(chunk_size_override: Option<i32>) -> Option<i32> {
    match chunk_size_override {
        Some(value) if value <= 0 => None,
        Some(value) => Some(sanitize_gdn_chunk_size(value)),
        None => Some(configured_gdn_chunk_size()),
    }
}

/// Compute gating decay: g = exp(-exp(A_log) * softplus(a + dt_bias))
///
/// Uses the bridge's fused `compute_g` kernel (single dispatch instead of 6 ops).
///
/// # Arguments
/// * `a_log` - Log of decay rates, shape `[Hv]`
/// * `a` - Per-token gating input, shape `[B, T, Hv]`
/// * `dt_bias` - Learnable bias, shape `[Hv]`
///
/// # Returns
/// Gating decay values, shape `[B, T, Hv]`
pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    Ok(Array::fused_compute_g(a_log, a, dt_bias))
}

/// Raw (non-compiled) compute_g implementation.
///
/// Public so that callers embedding GDN inside an **outer** `mx.compile` closure
/// can inline these ops directly, allowing the outer compile to fuse them with
/// surrounding element-wise operations. Calling the compiled [`compute_g`] inside
/// another compiled closure creates a fusion barrier (the inner `Compiled` primitive
/// is opaque to the outer compile pass).
pub fn compute_g_impl(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let input_dtype = a_log.dtype();

    // Upcast to f32 for stability
    let a_log_f32 = if input_dtype != Dtype::Float32 {
        a_log.as_dtype(Dtype::Float32.as_i32())
    } else {
        a_log.clone()
    };

    // exp(A_log) gives the decay rate A
    let decay_rate = a_log_f32.exp();

    // softplus(a + dt_bias)
    let a_biased = a.add(dt_bias);
    let sp = a_biased.softplus();

    // g = exp(-A * softplus(a + dt_bias))
    let g = decay_rate.multiply(&sp).negative().exp();

    // Cast back to input dtype
    if input_dtype != Dtype::Float32 {
        Ok(g.as_dtype(input_dtype.as_i32()))
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
/// GDN step dispatch.
fn gated_delta_step_compiled(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array), Exception> {
    gated_delta_step_core_ops(q, k, v, g, beta, state)
}

fn gated_delta_step_core_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array), Exception> {
    // Decay: state = state * g
    // g can be [B, H] (scalar) or [B, H, Dk] (vectorized)
    let decayed_state = match g.ndim() {
        2 => {
            // [B, H] -> [B, H, 1, 1] for broadcasting with [B, H, Dv, Dk]
            let g_expanded = g.reshape(&[g.dim(0), g.dim(1), 1, 1]);
            state.multiply(&g_expanded)
        }
        3 => {
            // [B, H, Dk] -> [B, H, 1, Dk] for broadcasting with [B, H, Dv, Dk]
            let g_expanded = g.reshape(&[g.dim(0), g.dim(1), 1, g.dim(2)]);
            state.multiply(&g_expanded)
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
    let k_expanded = k.reshape(&[k.dim(0), k.dim(1), 1, k.dim(2)]);
    let kv_mem = decayed_state.multiply(&k_expanded).sum_axis(-1, false);

    // delta = (v - kv_mem) * beta
    // v is [B, H, Dv], beta is [B, H] -> [B, H, 1]
    let beta_expanded = beta.reshape(&[beta.dim(0), beta.dim(1), 1]);
    let delta = v.subtract(&kv_mem).multiply(&beta_expanded);

    // new_state = decayed_state + k^T * delta (outer product)
    // k_expanded: [B, H, 1, Dk], delta: [B, H, Dv] -> [B, H, Dv, 1]
    let delta_expanded = delta.reshape(&[delta.dim(0), delta.dim(1), delta.dim(2), 1]);
    let new_state = decayed_state.add(&k_expanded.multiply(&delta_expanded));

    // y = sum(new_state * q, axis=-1) -> [B, H, Dv]
    let q_expanded = q.reshape(&[q.dim(0), q.dim(1), 1, q.dim(2)]);
    let y = new_state.multiply(&q_expanded).sum_axis(-1, false);

    Ok((y, new_state))
}

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
    let (y, new_state) = gated_delta_step_compiled(q, k, v, g, beta, state)?;

    // Apply mask: if masked, keep old state
    let new_state = if let Some(mask) = mask {
        // mask is [B], expand to [B, 1, 1, 1] for broadcasting
        let mask_expanded = mask.reshape(&[mask.dim(0), 1, 1, 1]);
        ops::where_fn(&mask_expanded, &new_state, &old_state)
    } else {
        new_state
    };

    Ok((y, new_state))
}

/// Decode-specialized GDN path for `T = 1`.
///
/// This keeps the reference MLX recurrence math but removes the sequential loop,
/// timestep slicing, and output stacking overhead from the generic short-sequence path.
fn gated_delta_decode_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array), Exception> {
    let b = q.dim(0);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);

    let repeat_factor = hv / hk;
    let (q_rep, k_rep);
    let (q, k) = if repeat_factor > 1 {
        q_rep = ops::repeat_axis(q.clone(), repeat_factor, 2);
        k_rep = ops::repeat_axis(k.clone(), repeat_factor, 2);
        (&q_rep, &k_rep)
    } else {
        (q, k)
    };

    let q_t = q.reshape(&[b, hv, dk]);
    let k_t = k.reshape(&[b, hv, dk]);
    let v_t = v.reshape(&[b, hv, dv]);
    let g_t = match g.ndim() {
        3 => g.reshape(&[b, hv]),
        4 => g.reshape(&[b, hv, dk]),
        _ => {
            return Err(Exception::custom(format!(
                "Unsupported decode gating shape: {:?}",
                g.shape()
            )));
        }
    };
    let beta_t = beta.reshape(&[b, hv]);

    let (y, new_state) = gated_delta_step_compiled(&q_t, &k_t, &v_t, &g_t, &beta_t, state)?;
    Ok((y.reshape(&[b, 1, hv, dv]), new_state))
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
        ops::zeros(&[b, hv, dv, dk], q.dtype())
    };

    if t == 1 && mask.is_none() {
        return gated_delta_decode_ops(q, k, v, g, beta, &state);
    }

    // Handle GQA: repeat q, k along head dim if Hv > Hk
    let repeat_factor = hv / hk;
    let (q_rep, k_rep);
    let (q, k) = if repeat_factor > 1 {
        q_rep = ops::repeat_axis(q.clone(), repeat_factor, 2);
        k_rep = ops::repeat_axis(k.clone(), repeat_factor, 2);
        (&q_rep, &k_rep)
    } else {
        (q, k)
    };

    let b_i = b as usize;
    let hk_i = q.dim(2) as usize;
    let _hv_i = hv as usize;
    let dk_i = dk as usize;
    let dv_i = dv as usize;
    let _ = (hk_i, dk_i, dv_i); // used via dim() calls below
    let mut ys = Vec::with_capacity(t as usize);

    for t_idx in 0..t {
        // Slice timestep t: [B, 1, H, D] -> squeeze axis 1 -> [B, H, D]
        let ti = t_idx as usize;
        let ti1 = ti + 1;
        let q_h = q.dim(2) as usize;
        let q_d = q.dim(3) as usize;
        let v_h = v.dim(2) as usize;
        let v_d = v.dim(3) as usize;
        let q_t = q
            .slice(
                &[0, t_idx, 0, 0],
                &[b as i32, ti1 as i32, q_h as i32, q_d as i32],
            )
            .squeeze(1);
        let k_t = k
            .slice(
                &[0, t_idx, 0, 0],
                &[b as i32, ti1 as i32, q_h as i32, q_d as i32],
            )
            .squeeze(1);
        let v_t = v
            .slice(
                &[0, t_idx, 0, 0],
                &[b as i32, ti1 as i32, v_h as i32, v_d as i32],
            )
            .squeeze(1);

        let g_t = if g.ndim() == 3 {
            let g_h = g.dim(2) as usize;
            g.slice(&[0, t_idx, 0], &[b as i32, ti1 as i32, g_h as i32])
                .squeeze(1)
        } else {
            let g_h = g.dim(2) as usize;
            let g_d = g.dim(3) as usize;
            g.slice(
                &[0, t_idx, 0, 0],
                &[b as i32, ti1 as i32, g_h as i32, g_d as i32],
            )
            .squeeze(1)
        };

        let beta_h = beta.dim(2) as usize;
        let beta_t = beta
            .slice(&[0, t_idx, 0], &[b as i32, ti1 as i32, beta_h as i32])
            .squeeze(1);

        let mask_t = mask.map(|m| m.slice(&[0, t_idx], &[b as i32, ti1 as i32]).squeeze(1));

        let (y, new_state) =
            gated_delta_step_ops(&q_t, &k_t, &v_t, &g_t, &beta_t, &state, mask_t.as_ref())?;
        state = new_state;
        ys.push(y);
    }
    let _ = b_i; // suppress unused warning

    // Stack outputs: Vec<[B, Hv, Dv]> -> [B, T, Hv, Dv]
    let y = ops::stack_axis(&ys, 1);

    Ok((y, state))
}

// ============================================================================
// Chunkwise parallel GDN (WY factorization)
// ============================================================================

/// Compute lower-triangular decay matrix from log-space gating values.
///
/// Given `log_g` of shape `[*, C]`, computes a `[*, C, C]` lower-triangular
/// matrix where `Γ[i,j] = exp(Σ_{m=j+1}^{i} log_g[m])` for `i > j`,
/// `Γ[i,i] = 1`, and `Γ[i,j] = 0` for `i < j`.
///
/// Uses cumulative sum in log-space: `Γ[i,j] = exp(cumsum[i] - cumsum[j])`.
fn chunk_decay_matrix(log_g: &Array) -> Result<Array, Exception> {
    let cs = log_g.cumsum(-1);
    let cs_i = ops::expand_dims(&cs, -1); // [*, C, 1]
    let cs_j = ops::expand_dims(&cs, -2); // [*, 1, C]
    let log_decay = cs_i.subtract(&cs_j); // [*, C, C]
    let decay = log_decay.exp();
    Ok(ops::tril(&decay, 0)) // zero upper triangle; diagonal = exp(0) = 1
}

/// Chunkwise parallel GDN implementation.
///
/// Splits the sequence into chunks of size `chunk_size` and uses the WY
/// factorization to parallelize intra-chunk computation. Only inter-chunk
/// state propagation remains sequential (O(T/C) steps instead of O(T)).
///
/// # Algorithm (per chunk)
///
/// 1. **Decay matrix** `Γ[i,j]`: lower-triangular cumulative gating decay
/// 2. **WY system** `A = tril(diag(β) * Γ * KK^T, -1)`: captures intra-chunk
///    recurrence dependencies. Solve `δ = (I+A)^{-1} @ u` via `tri_inv`.
/// 3. **Inter-chunk output**: `y_inter = Γ_init * (Q @ S₀^T)`
/// 4. **Intra-chunk output**: `y_intra = tril(Γ * QK^T) @ δ`
/// 5. **State update**: `S_{c+1} = Γ_total * S_c + (Γ_last * δ)^T @ K`
///
/// # Arguments
/// * `q`, `k` - Shape `[B, T, Hk, Dk]` (GQA handled internally)
/// * `v` - Shape `[B, T, Hv, Dv]`
/// * `g` - Shape `[B, T, Hv]` (scalar gating, already computed)
/// * `beta` - Shape `[B, T, Hv]`
/// * `state` - Initial state `[B, Hv, Dv, Dk]`
/// * `mask` - Optional mask `[B, T]` (1=valid, 0=padded)
///
/// # Returns
/// (output `[B, T, Hv, Dv]`, final_state `[B, Hv, Dv, Dk]`)
#[allow(clippy::too_many_arguments)]
fn gated_delta_chunk_ops_impl(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
    chunk_size: i32,
) -> Result<(Array, Array), Exception> {
    let b = q.dim(0);
    let t = q.dim(1);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);
    let c = sanitize_gdn_chunk_size(chunk_size);

    // Handle GQA: repeat q, k along head dim if Hv > Hk
    let repeat_factor = hv / hk;
    let (q_rep, k_rep);
    let (q, k) = if repeat_factor > 1 {
        q_rep = ops::repeat_axis(q.clone(), repeat_factor, 2);
        k_rep = ops::repeat_axis(k.clone(), repeat_factor, 2);
        (&q_rep, &k_rep)
    } else {
        (q, k)
    };
    let h = hv;

    // Apply mask: g=1 (no decay) and beta=0 (no update) for masked positions
    let (g, beta) = if let Some(mask) = mask {
        let mask_exp = ops::expand_dims(mask, -1); // [B, T, 1]
        let ones = ops::ones(g.shape(), g.dtype());
        let zeros = ops::zeros(beta.shape(), beta.dtype());
        let g = ops::where_fn(&mask_exp, g, &ones);
        let beta = ops::where_fn(&mask_exp, beta, &zeros);
        (g, beta)
    } else {
        (g.clone(), beta.clone())
    };

    // Pad T to be divisible by C if needed
    let pad_len = (c - (t % c)) % c;
    let t_padded = t + pad_len;
    let n_chunks = t_padded / c;

    let (q, k, v, g, beta) = if pad_len > 0 {
        let q_pad = ops::zeros(&[b, pad_len, h, dk], q.dtype());
        let k_pad = ops::zeros(&[b, pad_len, h, dk], k.dtype());
        let v_pad = ops::zeros(&[b, pad_len, h, dv], v.dtype());
        let g_pad = ops::ones(&[b, pad_len, h], g.dtype());
        let beta_pad = ops::zeros(&[b, pad_len, h], beta.dtype());

        let q = ops::concatenate_axis(&[q, &q_pad], 1);
        let k = ops::concatenate_axis(&[k, &k_pad], 1);
        let v = ops::concatenate_axis(&[v, &v_pad], 1);
        let g = ops::concatenate_axis(&[&g, &g_pad], 1);
        let beta = ops::concatenate_axis(&[&beta, &beta_pad], 1);
        (q, k, v, g, beta)
    } else {
        (q.clone(), k.clone(), v.clone(), g, beta)
    };

    // Transpose to [B, H, T, D] for batched matmul
    let q = q.transpose_axes(&[0, 2, 1, 3]); // [B, H, T_padded, Dk]
    let k = k.transpose_axes(&[0, 2, 1, 3]); // [B, H, T_padded, Dk]
    let v = v.transpose_axes(&[0, 2, 1, 3]); // [B, H, T_padded, Dv]
    let g = g.transpose_axes(&[0, 2, 1]); // [B, H, T_padded]
    let beta = beta.transpose_axes(&[0, 2, 1]); // [B, H, T_padded]

    // Identity matrix for WY factorization (reused across chunks)
    let eye = ops::eye(c, Dtype::Float32); // [C, C]
    let bh = b * h;

    // ========================================================================
    // Phase 1: Precompute per-chunk data (state-independent) and collect
    //          (I+A) matrices for batched tri_inv.
    // ========================================================================
    struct ChunkPrecomp {
        q_c: Array, // [B, H, C, Dk]
        k_c: Array, // [B, H, C, Dk]
        #[allow(dead_code)] // Precomputed for chunk attention (used in tri_inv path)
        k_c_t: Array, // [B, H, Dk, C]
        beta_v: Array, // [B, H, C, Dv]
        beta_gamma_row: Array, // [B, H, 1, C]
        gamma_init: Array, // [B, H, C]
        gamma_total: Array, // [B, H]
        gamma_last: Array, // [B, H, C]
        qk_decay: Array, // [B, H, C, C]
    }

    let mut chunks: Vec<ChunkPrecomp> = Vec::with_capacity(n_chunks as usize);
    let mut i_plus_a_list: Vec<Array> = Vec::with_capacity(n_chunks as usize);

    for ci in 0..n_chunks {
        let start = ci * c;
        let end = start + c;

        // Extract chunk data via slice
        let bv = b as usize;
        let hv2 = h as usize;
        let dk_s = q.dim(3) as usize;
        let dv_s = v.dim(3) as usize;
        let c_s = c as usize;
        let q_c = q.slice(&[0, 0, start, 0], &[b, h, end, q.dim(3)]); // [B, H, C, Dk]
        let k_c = k.slice(&[0, 0, start, 0], &[b, h, end, k.dim(3)]); // [B, H, C, Dk]
        let v_c = v.slice(&[0, 0, start, 0], &[b, h, end, v.dim(3)]); // [B, H, C, Dv]
        let g_c = g.slice(&[0, 0, start], &[b, h, end]); // [B, H, C]
        let beta_c = beta.slice(&[0, 0, start], &[b, h, end]); // [B, H, C]
        let _ = (bv, hv2, dk_s, dv_s, c_s);

        // Decay matrix: clamp log result (not exp-space input) to prevent -inf/NaN in f16
        let log_g_c = g_c.log(); // [B, H, C]
        let log_g_c = ops::maximum(&log_g_c, &Array::from_f32(-30.0)); // Clamp log-space
        let cs = log_g_c.cumsum(-1); // [B, H, C]
        let decay_c = chunk_decay_matrix(&log_g_c)?; // [B, H, C, C]

        let gamma_init = cs.exp(); // [B, H, C]

        let cs_h = cs.dim(2) as usize;
        let cs_last = cs.slice(&[0, 0, c - 1], &[b, h, c]).squeeze(-1); // [B, H]
        let gamma_total = cs_last.exp(); // [B, H]

        let cs_last_exp = ops::expand_dims(&cs_last, -1); // [B, H, 1]
        let gamma_last = cs_last_exp.subtract(&cs).exp(); // [B, H, C]
        let _ = cs_h;

        // WY factorization: build (I + A) matrix
        let k_c_t = k_c.transpose_axes(&[0, 1, 3, 2]); // [B, H, Dk, C]
        let kk_t = ops::matmul(&k_c, &k_c_t); // [B, H, C, C]
        let beta_col = ops::expand_dims(&beta_c, -1); // [B, H, C, 1]
        let a_mat = ops::tril(&beta_col.multiply(&decay_c).multiply(&kk_t), -1);
        let i_plus_a = a_mat.add(&eye); // [B, H, C, C]

        // Precompute beta*v and beta*gamma_init (both state-independent)
        let beta_v = beta_col.multiply(&v_c); // [B, H, C, Dv]
        let beta_gamma = beta_c.multiply(&gamma_init); // [B, H, C]
        let beta_gamma_row = ops::expand_dims(&beta_gamma, -2); // [B, H, 1, C]

        // Precompute intra-chunk decay-weighted QK^T
        let qk_t = ops::matmul(&q_c, &k_c_t); // [B, H, C, C]
        let qk_decay = ops::tril(&decay_c.multiply(&qk_t), 0); // [B, H, C, C]

        // Flatten (I+A) to [B*H, C, C] for batched tri_inv
        i_plus_a_list.push(i_plus_a.reshape(&[bh, c, c]));

        chunks.push(ChunkPrecomp {
            q_c,
            k_c,
            k_c_t,
            beta_v,
            beta_gamma_row,
            gamma_init,
            gamma_total,
            gamma_last,
            qk_decay,
        });
    }

    // ========================================================================
    // Phase 2: Single batched tri_inv call (1 CPU sync instead of N).
    // ========================================================================
    let i_plus_a_refs: Vec<&Array> = i_plus_a_list.iter().collect();
    let batched_ipa = ops::concatenate_axis(&i_plus_a_refs, 0); // [N*B*H, C, C]
    // stop_gradient: tri_inv has no VJP in MLX. The inverse is a fixed preconditioner
    // in the WY factorization — gradients should not flow through matrix inversion.
    // This matches the FLA reference impl which computes tri_inv in torch.no_grad().
    let batched_inv = linalg::tri_inv(&batched_ipa, false).stop_gradient();

    // Split back per-chunk and precompute delta_v, t_inv_bg
    struct ChunkInvData {
        delta_v: Array,  // [B, H, C, Dv]  = T_inv @ beta_v
        t_inv_bg: Array, // [B, H, C, C]   = T_inv * beta_gamma_row
    }

    let mut inv_data: Vec<ChunkInvData> = Vec::with_capacity(n_chunks as usize);
    for ci in 0..n_chunks {
        let ci = ci as usize;
        let start = (ci as i32) * bh;
        let end = start + bh;
        let inv_c_dim = batched_inv.dim(1);
        let t_inv = batched_inv
            .slice(&[start, 0, 0], &[end, inv_c_dim, inv_c_dim])
            .reshape(&[b, h, c, c]); // [B, H, C, C]

        let delta_v = ops::matmul(&t_inv, &chunks[ci].beta_v); // [B, H, C, Dv]
        let t_inv_bg = t_inv.multiply(&chunks[ci].beta_gamma_row); // [B, H, C, C]

        inv_data.push(ChunkInvData { delta_v, t_inv_bg });
    }

    // ========================================================================
    // Phase 3: State-dependent sequential loop (N steps, no CPU syncs).
    // ========================================================================
    let mut state = state.clone(); // [B, H, Dv, Dk]
    let mut y_chunks = Vec::with_capacity(n_chunks as usize);

    for ci in 0..n_chunks as usize {
        let chunk = &chunks[ci];
        let inv = &inv_data[ci];

        // State-dependent terms
        let state_t = state.transpose_axes(&[0, 1, 3, 2]); // [B, H, Dk, Dv]
        let ks = ops::matmul(&chunk.k_c, &state_t); // [B, H, C, Dv]

        // delta = T_inv @ beta_v - T_inv_bg @ (K @ S^T)
        let delta_s = ops::matmul(&inv.t_inv_bg, &ks); // [B, H, C, Dv]
        let delta = inv.delta_v.subtract(&delta_s); // [B, H, C, Dv]

        // Inter-chunk output: y_inter = Γ_init * (Q @ S^T)
        let qs = ops::matmul(&chunk.q_c, &state_t); // [B, H, C, Dv]
        let gamma_init_exp = ops::expand_dims(&chunk.gamma_init, -1); // [B, H, C, 1]
        let y_inter = gamma_init_exp.multiply(&qs); // [B, H, C, Dv]

        // Intra-chunk output: y_intra = qk_decay @ δ
        let y_intra = ops::matmul(&chunk.qk_decay, &delta); // [B, H, C, Dv]

        y_chunks.push(y_inter.add(&y_intra)); // [B, H, C, Dv]

        // State propagation: S_{c+1} = Γ_total * S_c + (Γ_last * δ)^T @ K
        let gamma_last_exp = ops::expand_dims(&chunk.gamma_last, -1); // [B, H, C, 1]
        let delta_weighted = gamma_last_exp.multiply(&delta); // [B, H, C, Dv]
        let dw_t = delta_weighted.transpose_axes(&[0, 1, 3, 2]); // [B, H, Dv, C]
        let state_update = ops::matmul(&dw_t, &chunk.k_c); // [B, H, Dv, Dk]

        let gamma_total_exp = chunk.gamma_total.reshape(&[b, h, 1, 1]);
        state = gamma_total_exp.multiply(&state).add(&state_update);
    }

    // Concatenate chunk outputs: [B, H, T_padded, Dv]
    let y_refs: Vec<&Array> = y_chunks.iter().collect();
    let y = ops::concatenate_axis(&y_refs, 2);

    // Trim padding
    let y = if pad_len > 0 {
        y.slice(&[0, 0, 0, 0], &[b, h, t, y.dim(3)])
    } else {
        y
    };

    // Transpose back to [B, T, H, Dv]
    let y = y.transpose_axes(&[0, 2, 1, 3]);

    Ok((y, state))
}

#[cfg(test)]
fn gated_delta_chunk_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    gated_delta_chunk_ops_impl(q, k, v, g, beta, state, mask, DEFAULT_GDN_CHUNK_SIZE)
}

// ============================================================================
// Fused Metal GDN kernel (single kernel launch per layer)
// ============================================================================
//
// Fuses the entire GDN recurrence step into a single Metal dispatch.
// Each thread handles Dk/32 elements of the key dimension, with SIMD
// reductions across the 32-thread group. This eliminates ~15 separate
// MLX op kernel launches per GDN layer per token.
//
// Grid: (32, Dv, B * Hv) — one SIMD group per (batch, value_head, value_dim)
// Threadgroup: (32, 4, 1) — 32 threads = one SIMD group, 4 dv per threadgroup

/// Try the fused Metal GDN kernel. Returns None if conditions not met.
///
/// Public so that callers (e.g. compiled decode closures) can dispatch to the
/// Metal kernel directly without going through `gated_delta_update` which
/// would re-compute g/beta.
pub fn try_gdn_metal_kernel(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> Result<Option<(Array, Array)>, Exception> {
    // Metal kernel requires: no mask, Dk divisible by 32, Dk <= 256, scalar gating
    if mask.is_some() {
        return Ok(None);
    }
    let dk = q.dim(3) as usize;
    if dk % 32 != 0 || dk > 256 || dk == 0 {
        return Ok(None);
    }
    if g.ndim() != 3 {
        return Ok(None);
    }

    let t = q.dim(1);
    let (y, new_state) = Array::gdn_metal_step(q, k, v, g, beta, state, t);
    Ok(Some((y, new_state)))
}

/// Inference-only GDN dispatch with pre-computed g and beta.
///
/// Tries the fused Metal kernel first (single kernel launch per layer),
/// then falls back to the ops-based path. This is the preferred entry point
/// for compiled decode closures that have already computed g and beta.
///
/// Unlike [`gated_delta_update`], this does NOT recompute g/beta internally
/// and does NOT use a separately-compiled closure, making it safe to call
/// inside an outer `mx.compile` closure without creating fusion barriers.
pub fn gated_delta_inference_dispatch(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    // Try fused Metal kernel first (single dispatch per layer)
    if let Some(result) = try_gdn_metal_kernel(q, k, v, g, beta, state, mask)? {
        return Ok(result);
    }
    // Fallback to ops-based path
    gated_delta_ops(q, k, v, g, beta, Some(state), mask)
}

/// Replay the GDN recurrence forward from a saved `initial_state`, returning
/// only the final state (no output `y`).
///
/// This is the speculative-decoding rollback primitive for hybrid models such
/// as Qwen3.5: after a verify pass appends `T_verify` tokens but only `M < T`
/// are accepted, call this with the pre-verify state snapshot and the
/// accepted-prefix slices of `k`, `v`, `g`, `beta` to reconstruct the GDN
/// state at exactly the accepted position. Matches dflash-mlx's
/// `advance_gated_delta_states` helper but reuses the existing Metal kernel +
/// ops path instead of shipping a second kernel.
///
/// # Shapes
/// * `initial_state`: `[B, Hv, Dv, Dk]`
/// * `k`: `[B, T, Hk, Dk]` (GQA-expanded internally)
/// * `v`: `[B, T, Hv, Dv]`
/// * `g`: `[B, T, Hv]`
/// * `beta`: `[B, T, Hv]`
///
/// If `T == 0` the input state is returned unchanged.
pub fn gated_delta_state_advance(
    initial_state: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
) -> Result<Array, Exception> {
    let t = k.dim(1);
    if t == 0 {
        return Ok(initial_state.clone());
    }
    // Fast path: the dedicated state-only Metal kernel (no `q`, no `y`).
    // Preconditions mirror `try_gdn_metal_kernel`: scalar gating, Dk % 32 == 0,
    // Dk <= 256. These match every Qwen3.5 GDN layer today.
    let dk = k.dim(3) as usize;
    if g.ndim() == 3 && dk > 0 && dk % 32 == 0 && dk <= 256 {
        return Ok(Array::gdn_metal_state_update(k, v, g, beta, initial_state, t));
    }
    // Fallback: reuse the full ops-based recurrence and discard `y`. This
    // keeps GQA-expansion, vector gating, and the non-power-of-2 Dk edge
    // cases working without a second code path.
    let (_y, new_state) = gated_delta_ops(k, k, v, g, beta, Some(initial_state), None)?;
    Ok(new_state)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_dispatch(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
    training: bool,
    chunk_size_override: Option<i32>,
) -> Result<(Array, Array), Exception> {
    if !training {
        // For inference: try fused Metal kernel first (single dispatch per layer)
        if let Some(result) = try_gdn_metal_kernel(q, k, v, g, beta, state, mask)? {
            return Ok(result);
        }

        let t = q.dim(1);
        if let Some(chunk_size) = resolve_chunk_size_override(chunk_size_override)
            && t > chunk_size
        {
            return gated_delta_chunk_ops_impl(q, k, v, g, beta, state, mask, chunk_size);
        }
    }
    gated_delta_ops(q, k, v, g, beta, Some(state), mask)
}

/// Top-level GDN update API.
///
/// Computes the gated delta network recurrence for a sequence of tokens.
/// Handles beta/gate computation from raw inputs and dispatches to the
/// appropriate implementation:
/// - `T > 64`: chunkwise parallel (WY factorization) for training prefill
/// - `T <= 64`: sequential loop for decode / short sequences
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
/// * `training` - If true, forces sequential path (chunk path's `tri_inv` has no VJP
///   and produces NaN inside `value_and_grad`). If false, dispatches to the chunk path
///   for sequences longer than the configured chunk size for O(T/C) prefill.
///
/// # Returns
/// (output `[B, T, Hv, Dv]`, final_state `[B, Hv, Dv, Dk]`)
#[allow(clippy::too_many_arguments)]
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
    training: bool,
) -> Result<(Array, Array), Exception> {
    gated_delta_update_with_chunk_size_override(
        q, k, v, a, b, a_log, dt_bias, state, mask, training, None,
    )
}

/// Variant of [`gated_delta_update`] that allows benchmarking or forcing a
/// specific inference prefill chunk size.
///
/// `chunk_size_override` semantics:
/// - `None`: use the configured runtime default (`PMETAL_GDN_CHUNK_SIZE` or 64)
/// - `Some(n > 0)`: force chunked inference with size `n` (clamped to 16..=512)
/// - `Some(0)` or negative: force the sequential path even for long inference prompts
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_update_with_chunk_size_override(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
    training: bool,
    chunk_size_override: Option<i32>,
) -> Result<(Array, Array), Exception> {
    // beta = sigmoid(b)
    let beta = b.sigmoid();

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
            init_state = ops::zeros(&[b_dim, hv, dv, dk], q.dtype());
            &init_state
        }
    };

    gated_delta_dispatch(
        q,
        k,
        v,
        &g,
        &beta,
        state_ref,
        mask,
        training,
        chunk_size_override,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::random;
    use serial_test::serial;

    fn to_f32_vec(arr: &Array) -> Vec<f32> {
        let mut a = arr.clone();
        a.eval();
        let n = a.size();
        a.to_f32_vec(n).unwrap_or_default()
    }

    #[test]
    #[serial]
    fn test_compute_g_shape() {
        let a_log = Array::from_f32_slice(&[0.5f32, 1.0, 1.5], &[3]);
        let a = random::normal(&[2, 4, 3], Dtype::Float32);
        let dt_bias = Array::from_f32_slice(&[0.1f32, 0.2, 0.3], &[3]);

        let g = compute_g(&a_log, &a, &dt_bias).unwrap();
        assert_eq!(g.shape(), &[2, 4, 3]);
        // g should be in (0, 1] since it's exp(-positive)
        let g_data = to_f32_vec(&g);
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

        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        let g = random::uniform(&[b, t, hv], Dtype::Float32);
        let beta = random::uniform(&[b, t, hv], Dtype::Float32);

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

        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        let a = random::normal(&[b, t, hv], Dtype::Float32);
        let b_input = random::normal(&[b, t, hv], Dtype::Float32);
        let a_log = Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0], &[hv]);
        let dt_bias = Array::from_f32_slice(&[0.1f32, 0.2, 0.3, 0.4], &[hv]);

        let (y, state) = gated_delta_update(
            &q, &k, &v, &a, &b_input, &a_log, &dt_bias, None, None, false,
        )
        .unwrap();

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
        let t = 4;

        random::seed(42);

        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        let g = random::uniform(&[b, t, hv], Dtype::Float32);
        let beta = random::uniform(&[b, t, hv], Dtype::Float32);

        // Process all 4 steps at once
        let (_y_full, state_full) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Process in two chunks of 2
        let q1 = q.slice(&[0, 0, 0, 0], &[b, 2, hk, dk]);
        let k1 = k.slice(&[0, 0, 0, 0], &[b, 2, hk, dk]);
        let v1 = v.slice(&[0, 0, 0, 0], &[b, 2, hv, dv]);
        let g1 = g.slice(&[0, 0, 0], &[b, 2, hv]);
        let beta1 = beta.slice(&[0, 0, 0], &[b, 2, hv]);

        let (_y1, state1) = gated_delta_ops(&q1, &k1, &v1, &g1, &beta1, None, None).unwrap();

        let q2 = q.slice(&[0, 2, 0, 0], &[b, t, hk, dk]);
        let k2 = k.slice(&[0, 2, 0, 0], &[b, t, hk, dk]);
        let v2 = v.slice(&[0, 2, 0, 0], &[b, t, hv, dv]);
        let g2 = g.slice(&[0, 2, 0], &[b, t, hv]);
        let beta2 = beta.slice(&[0, 2, 0], &[b, t, hv]);

        let (_y2, state2) =
            gated_delta_ops(&q2, &k2, &v2, &g2, &beta2, Some(&state1), None).unwrap();

        // States should match
        state_full.eval();
        state2.eval();
        let diff = state_full.subtract(&state2).abs();
        let max_diff = diff.max(None);
        max_diff.eval();
        let max_diff_val: f32 = max_diff.item();
        assert!(
            max_diff_val < 1e-4,
            "State mismatch: max diff = {}",
            max_diff_val
        );
    }

    /// Helper: assert two arrays are close within tolerance, returning the max diff.
    fn assert_close(a: &Array, b: &Array, tol: f32, msg: &str) {
        let a_eval = a.clone();
        let b_eval = b.clone();
        a_eval.eval();
        b_eval.eval();
        let diff = a_eval.subtract(&b_eval).abs();
        let max_diff = diff.max(None);
        max_diff.eval();
        let max_diff_val: f32 = max_diff.item();
        assert!(
            max_diff_val < tol,
            "{}: max diff = {} (tol = {})",
            msg,
            max_diff_val,
            tol
        );
    }

    /// Helper: generate random GDN inputs for testing.
    fn random_gdn_inputs(
        b: i32,
        t: i32,
        hk: i32,
        dk: i32,
        hv: i32,
        dv: i32,
    ) -> (Array, Array, Array, Array, Array) {
        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        // g in (0.5, 1.0) to keep decay moderate and avoid numerical issues
        let g = random::uniform(&[b, t, hv], Dtype::Float32);
        // beta in (0.0, 0.5) to keep updates moderate
        let beta = random::uniform(&[b, t, hv], Dtype::Float32);
        (q, k, v, g, beta)
    }

    #[test]
    #[serial]
    fn test_chunk_vs_sequential_t128() {
        // Numerical equivalence: chunk output should match sequential for T=128
        let b = 1;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(123);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path
        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        assert_eq!(y_chunk.shape(), y_seq.shape());
        assert_eq!(state_chunk.shape(), state_seq.shape());
        assert_close(&y_chunk, &y_seq, 1e-3, "T=128 output mismatch");
        assert_close(&state_chunk, &state_seq, 1e-3, "T=128 state mismatch");
    }

    #[test]
    #[serial]
    fn test_chunk_vs_sequential_t256() {
        // Numerical equivalence: chunk output should match sequential for T=256
        let b = 1;
        let t = 256;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(456);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        assert_close(&y_chunk, &y_seq, 1e-3, "T=256 output mismatch");
        assert_close(&state_chunk, &state_seq, 1e-3, "T=256 state mismatch");
    }

    #[test]
    #[serial]
    fn test_chunk_state_continuity() {
        // Process T=256 in one call vs two calls of T=128 with state passing
        let b = 1;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;
        let t = 256;

        random::seed(789);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());

        // Process all 256 at once via chunk path
        let (_y_full, state_full) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        // Process in two halves of 128
        let q1 = q.slice(&[0, 0, 0, 0], &[b, 128, hk, dk]);
        let k1 = k.slice(&[0, 0, 0, 0], &[b, 128, hk, dk]);
        let v1 = v.slice(&[0, 0, 0, 0], &[b, 128, hv, dv]);
        let g1 = g.slice(&[0, 0, 0], &[b, 128, hv]);
        let beta1 = beta.slice(&[0, 0, 0], &[b, 128, hv]);

        let (_y1, state1) =
            gated_delta_chunk_ops(&q1, &k1, &v1, &g1, &beta1, &state_init, None).unwrap();

        let q2 = q.slice(&[0, 128, 0, 0], &[b, t, hk, dk]);
        let k2 = k.slice(&[0, 128, 0, 0], &[b, t, hk, dk]);
        let v2 = v.slice(&[0, 128, 0, 0], &[b, t, hv, dv]);
        let g2 = g.slice(&[0, 128, 0], &[b, t, hv]);
        let beta2 = beta.slice(&[0, 128, 0], &[b, t, hv]);

        let (_y2, state2) =
            gated_delta_chunk_ops(&q2, &k2, &v2, &g2, &beta2, &state1, None).unwrap();

        assert_close(
            &state_full,
            &state2,
            1e-3,
            "Chunk state continuity mismatch",
        );
    }

    #[test]
    #[serial]
    fn test_chunk_non_divisible_t() {
        // T=100 is not divisible by 64 — tests padding logic
        let b = 1;
        let t = 100;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(101);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path (will pad to 128)
        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        assert_eq!(
            y_chunk.shape(),
            &[b, t, hv, dv],
            "Output shape should be unpadded"
        );
        assert_close(&y_chunk, &y_seq, 1e-3, "T=100 output mismatch");
        assert_close(&state_chunk, &state_seq, 1e-3, "T=100 state mismatch");
    }

    #[test]
    #[serial]
    fn test_chunk_gqa() {
        // GQA: Hv=4, Hk=2 (repeat factor 2)
        let b = 1;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 4;
        let dv = 8;

        random::seed(202);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path
        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        assert_eq!(y_chunk.shape(), &[b, t, hv, dv]);
        assert_close(&y_chunk, &y_seq, 1e-3, "GQA output mismatch");
        assert_close(&state_chunk, &state_seq, 1e-3, "GQA state mismatch");
    }

    #[test]
    #[serial]
    fn test_chunk_with_mask() {
        // Masked inputs: verify chunk handles mask correctly
        let b = 1;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(303);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Mask: first 100 tokens valid, last 28 masked
        let mut mask_data = vec![1.0f32; t as usize];
        for i in 100..t as usize {
            mask_data[i] = 0.0;
        }
        let mask = Array::from_f32_slice(&mask_data, &[b, t]);

        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());

        // Chunk path with mask: state should match processing only the first 100 tokens
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, Some(&mask)).unwrap();

        // Sequential on just the first 100 tokens (no mask needed)
        let q_100 = q.slice(&[0, 0, 0, 0], &[b, 100, hk, dk]);
        let k_100 = k.slice(&[0, 0, 0, 0], &[b, 100, hk, dk]);
        let v_100 = v.slice(&[0, 0, 0, 0], &[b, 100, hv, dv]);
        let g_100 = g.slice(&[0, 0, 0], &[b, 100, hv]);
        let beta_100 = beta.slice(&[0, 0, 0], &[b, 100, hv]);

        let (_y_ref, state_ref) =
            gated_delta_ops(&q_100, &k_100, &v_100, &g_100, &beta_100, None, None).unwrap();

        assert_eq!(y_chunk.shape(), &[b, t, hv, dv]);
        // States should match since masked positions shouldn't update state
        assert_close(&state_chunk, &state_ref, 1e-3, "Masked state mismatch");
    }

    #[test]
    #[serial]
    fn test_update_dispatches_chunk() {
        // Verify gated_delta_update dispatches to chunk path for T > configured chunk size
        let b = 1;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(404);
        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        let a = random::normal(&[b, t, hv], Dtype::Float32);
        let b_input = random::normal(&[b, t, hv], Dtype::Float32);
        let a_log = Array::from_f32_slice(&[0.5f32, 1.0], &[hv]);
        let dt_bias = Array::from_f32_slice(&[0.1f32, 0.2], &[hv]);

        let (y, state) = gated_delta_update(
            &q, &k, &v, &a, &b_input, &a_log, &dt_bias, None, None, false,
        )
        .unwrap();

        assert_eq!(y.shape(), &[b, t, hv, dv]);
        assert_eq!(state.shape(), &[b, hv, dv, dk]);

        // Verify output is finite
        y.eval();
        state.eval();
    }

    #[test]
    #[serial]
    fn test_chunk_t65() {
        // Worst-case padding: T=65 → 2 chunks, second has 1 real + 63 padded tokens
        let b = 1;
        let t = 65;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(505);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path (pads to 128)
        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        assert_eq!(
            y_chunk.shape(),
            &[b, t, hv, dv],
            "Output shape should be unpadded"
        );
        assert_close(&y_chunk, &y_seq, 1e-3, "T=65 output mismatch");
        assert_close(&state_chunk, &state_seq, 1e-3, "T=65 state mismatch");
    }

    #[test]
    #[serial]
    fn test_chunk_batched() {
        // B=2 to verify batch dimension handling in chunk path
        let b = 2;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(606);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path
        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        assert_eq!(y_chunk.shape(), &[b, t, hv, dv]);
        assert_close(&y_chunk, &y_seq, 1e-3, "B=2 output mismatch");
        assert_close(&state_chunk, &state_seq, 1e-3, "B=2 state mismatch");
    }

    #[test]
    #[serial]
    fn test_chunk_nonzero_initial_state() {
        // Non-zero S₀ validates the inter-chunk contribution term (Γ_init * Q @ S^T)
        let b = 1;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(707);
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Non-zero initial state
        let state_init = random::normal(&[b, hv, dv, dk], Dtype::Float32);

        // Sequential reference with same initial state
        let (y_seq, state_seq) =
            gated_delta_ops(&q, &k, &v, &g, &beta, Some(&state_init), None).unwrap();

        // Chunk path
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        assert_eq!(y_chunk.shape(), &[b, t, hv, dv]);
        assert_close(&y_chunk, &y_seq, 1e-3, "Nonzero S₀ output mismatch");
        assert_close(&state_chunk, &state_seq, 1e-3, "Nonzero S₀ state mismatch");
    }

    #[test]
    #[serial]
    fn test_decode_specialized_matches_step_ops() {
        let b = 1;
        let t = 1;
        let hk = 2;
        let dk = 64;
        let hv = 4; // GQA: 4 value heads, 2 key heads
        let dv = 64;

        random::seed(42);
        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        let g = random::uniform(&[b, t, hv], Dtype::Float32);
        let beta = random::uniform(&[b, t, hv], Dtype::Float32);

        let state = ops::zeros(&[b, hv, dv, dk], Dtype::Float32);

        let q_rep = ops::repeat_axis(q.clone(), hv / hk, 2);
        let k_rep = ops::repeat_axis(k.clone(), hv / hk, 2);
        let q_t = q_rep.reshape(&[b, hv, dk]);
        let k_t = k_rep.reshape(&[b, hv, dk]);
        let v_t = v.reshape(&[b, hv, dv]);
        let g_t = g.reshape(&[b, hv]);
        let beta_t = beta.reshape(&[b, hv]);

        let (y_ref_step, state_ref) =
            gated_delta_step_ops(&q_t, &k_t, &v_t, &g_t, &beta_t, &state, None).unwrap();
        let y_ref = y_ref_step.reshape(&[b, t, hv, dv]);

        let (y_decode, state_decode) =
            gated_delta_decode_ops(&q, &k, &v, &g, &beta, &state).unwrap();

        assert_eq!(y_decode.shape(), y_ref.shape());
        assert_eq!(state_decode.shape(), state_ref.shape());
        assert_close(
            &y_decode,
            &y_ref,
            1e-3,
            "Decode-specialized output mismatch",
        );
        assert_close(
            &state_decode,
            &state_ref,
            1e-3,
            "Decode-specialized state mismatch",
        );
    }

    #[test]
    #[serial]
    fn test_chunk_size_override_zero_forces_sequential_path() {
        let b = 1;
        let t = 96;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(808);
        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        let a = random::normal(&[b, t, hv], Dtype::Float32);
        let b_in = random::normal(&[b, t, hv], Dtype::Float32);
        let a_log = random::normal(&[hv], Dtype::Float32).abs();
        let dt_bias = random::normal(&[hv], Dtype::Float32);

        let (y_seq, state_seq) = gated_delta_update_with_chunk_size_override(
            &q, &k, &v, &a, &b_in, &a_log, &dt_bias, None, None, true, None,
        )
        .unwrap();
        let (y_forced, state_forced) = gated_delta_update_with_chunk_size_override(
            &q,
            &k,
            &v,
            &a,
            &b_in,
            &a_log,
            &dt_bias,
            None,
            None,
            false,
            Some(0),
        )
        .unwrap();

        assert_close(&y_forced, &y_seq, 1e-3, "Forced sequential output mismatch");
        assert_close(
            &state_forced,
            &state_seq,
            1e-3,
            "Forced sequential state mismatch",
        );
    }

    #[test]
    #[serial]
    fn test_chunk_size_override_positive_matches_chunk_path() {
        let b = 1;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        random::seed(909);
        let q = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let k = random::normal(&[b, t, hk, dk], Dtype::Float32);
        let v = random::normal(&[b, t, hv, dv], Dtype::Float32);
        let a = random::normal(&[b, t, hv], Dtype::Float32);
        let b_in = random::normal(&[b, t, hv], Dtype::Float32);
        let a_log = random::normal(&[hv], Dtype::Float32).abs();
        let dt_bias = random::normal(&[hv], Dtype::Float32);
        let g = compute_g(&a_log, &a, &dt_bias).unwrap();
        let beta = b_in.sigmoid();
        let state_init = ops::zeros(&[b, hv, dv, dk], q.dtype());

        let (y_forced, state_forced) = gated_delta_update_with_chunk_size_override(
            &q,
            &k,
            &v,
            &a,
            &b_in,
            &a_log,
            &dt_bias,
            None,
            None,
            false,
            Some(32),
        )
        .unwrap();
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops_impl(&q, &k, &v, &g, &beta, &state_init, None, 32).unwrap();

        assert_close(&y_forced, &y_chunk, 1e-3, "Forced chunk output mismatch");
        assert_close(
            &state_forced,
            &state_chunk,
            1e-3,
            "Forced chunk state mismatch",
        );
    }
}
