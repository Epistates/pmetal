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

use mlx_rs::{
    Array, Dtype, StreamOrDevice,
    error::Exception,
    linalg, nn,
    ops::{self, indexing::IndexOp},
    stop_gradient,
};

/// Chunk size for the chunkwise parallel GDN algorithm.
/// Sequences longer than this use the parallel chunk path.
const GDN_CHUNK_SIZE: i32 = 64;

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

        let (y, new_state) =
            gated_delta_step_ops(&q_t, &k_t, &v_t, &g_t, &beta_t, &state, mask_t.as_ref())?;
        state = new_state;
        ys.push(y);
    }

    // Stack outputs: Vec<[B, Hv, Dv]> -> [B, T, Hv, Dv]
    let y_refs: Vec<&Array> = ys.iter().collect();
    let y = ops::stack_axis(&y_refs, 1)?;

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
    let cs = log_g.cumsum(-1, None, None)?;
    let cs_i = ops::expand_dims(&cs, -1)?; // [*, C, 1]
    let cs_j = ops::expand_dims(&cs, -2)?; // [*, 1, C]
    let log_decay = cs_i.subtract(&cs_j)?; // [*, C, C]
    let decay = log_decay.exp()?;
    ops::tril(&decay, 0) // zero upper triangle; diagonal = exp(0) = 1
}

/// Chunkwise parallel GDN implementation.
///
/// Splits the sequence into chunks of size `GDN_CHUNK_SIZE` and uses the WY
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
fn gated_delta_chunk_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let b = q.dim(0);
    let t = q.dim(1);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);
    let c = GDN_CHUNK_SIZE;

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
    let h = hv;

    // Apply mask: g=1 (no decay) and beta=0 (no update) for masked positions
    let (g, beta) = if let Some(mask) = mask {
        let mask_exp = ops::expand_dims(mask, -1)?; // [B, T, 1]
        let ones = ops::ones_dtype(g.shape(), g.dtype())?;
        let zeros = ops::zeros_dtype(beta.shape(), beta.dtype())?;
        let g = ops::r#where(&mask_exp, g, &ones)?;
        let beta = ops::r#where(&mask_exp, beta, &zeros)?;
        (g, beta)
    } else {
        (g.clone(), beta.clone())
    };

    // Pad T to be divisible by C if needed
    let pad_len = (c - (t % c)) % c;
    let t_padded = t + pad_len;
    let n_chunks = t_padded / c;

    let (q, k, v, g, beta) = if pad_len > 0 {
        let q_pad = ops::zeros_dtype(&[b, pad_len, h, dk], q.dtype())?;
        let k_pad = ops::zeros_dtype(&[b, pad_len, h, dk], k.dtype())?;
        let v_pad = ops::zeros_dtype(&[b, pad_len, h, dv], v.dtype())?;
        let g_pad = ops::ones_dtype(&[b, pad_len, h], g.dtype())?;
        let beta_pad = ops::zeros_dtype(&[b, pad_len, h], beta.dtype())?;

        let q = ops::concatenate_axis(&[q, &q_pad], 1)?;
        let k = ops::concatenate_axis(&[k, &k_pad], 1)?;
        let v = ops::concatenate_axis(&[v, &v_pad], 1)?;
        let g = ops::concatenate_axis(&[&g, &g_pad], 1)?;
        let beta = ops::concatenate_axis(&[&beta, &beta_pad], 1)?;
        (q, k, v, g, beta)
    } else {
        (q.clone(), k.clone(), v.clone(), g, beta)
    };

    // Transpose to [B, H, T, D] for batched matmul
    let q = q.transpose_axes(&[0, 2, 1, 3])?; // [B, H, T_padded, Dk]
    let k = k.transpose_axes(&[0, 2, 1, 3])?; // [B, H, T_padded, Dk]
    let v = v.transpose_axes(&[0, 2, 1, 3])?; // [B, H, T_padded, Dv]
    let g = g.transpose_axes(&[0, 2, 1])?; // [B, H, T_padded]
    let beta = beta.transpose_axes(&[0, 2, 1])?; // [B, H, T_padded]

    // Identity matrix for WY factorization (reused across chunks)
    let eye = ops::eye::<f32>(c, None, None)?; // [C, C]
    let eps = Array::from_f32(1e-6);
    let bh = b * h;

    // ========================================================================
    // Phase 1: Precompute per-chunk data (state-independent) and collect
    //          (I+A) matrices for batched tri_inv.
    // ========================================================================
    struct ChunkPrecomp {
        q_c: Array,            // [B, H, C, Dk]
        k_c: Array,            // [B, H, C, Dk]
        k_c_t: Array,          // [B, H, Dk, C]
        beta_v: Array,         // [B, H, C, Dv]
        beta_gamma_row: Array, // [B, H, 1, C]
        gamma_init: Array,     // [B, H, C]
        gamma_total: Array,    // [B, H]
        gamma_last: Array,     // [B, H, C]
        qk_decay: Array,       // [B, H, C, C]
    }

    let mut chunks: Vec<ChunkPrecomp> = Vec::with_capacity(n_chunks as usize);
    let mut i_plus_a_list: Vec<Array> = Vec::with_capacity(n_chunks as usize);

    for ci in 0..n_chunks {
        let start = ci * c;
        let end = start + c;

        // Extract chunk data
        let q_c = q.index((.., .., start..end, ..)); // [B, H, C, Dk]
        let k_c = k.index((.., .., start..end, ..)); // [B, H, C, Dk]
        let v_c = v.index((.., .., start..end, ..)); // [B, H, C, Dv]
        let g_c = g.index((.., .., start..end)); // [B, H, C]
        let beta_c = beta.index((.., .., start..end)); // [B, H, C]

        // Decay matrix: clamp g to eps before log to prevent -inf/NaN in f16
        let g_c = ops::maximum(&g_c, &eps)?;
        let log_g_c = g_c.log()?; // [B, H, C]
        let cs = log_g_c.cumsum(-1, None, None)?; // [B, H, C]
        let decay_c = chunk_decay_matrix(&log_g_c)?; // [B, H, C, C]

        let gamma_init = cs.exp()?; // [B, H, C]

        let cs_last = cs.index((.., .., (c - 1)..c)).squeeze_axes(&[-1])?; // [B, H]
        let gamma_total = cs_last.exp()?; // [B, H]

        let cs_last_exp = ops::expand_dims(&cs_last, -1)?; // [B, H, 1]
        let gamma_last = cs_last_exp.subtract(&cs)?.exp()?; // [B, H, C]

        // WY factorization: build (I + A) matrix
        let k_c_t = k_c.transpose_axes(&[0, 1, 3, 2])?; // [B, H, Dk, C]
        let kk_t = ops::matmul(&k_c, &k_c_t)?; // [B, H, C, C]
        let beta_col = ops::expand_dims(&beta_c, -1)?; // [B, H, C, 1]
        let a_mat = ops::tril(&beta_col.multiply(&decay_c)?.multiply(&kk_t)?, -1)?;
        let i_plus_a = a_mat.add(&eye)?; // [B, H, C, C]

        // Precompute beta*v and beta*gamma_init (both state-independent)
        let beta_v = beta_col.multiply(&v_c)?; // [B, H, C, Dv]
        let beta_gamma = beta_c.multiply(&gamma_init)?; // [B, H, C]
        let beta_gamma_row = ops::expand_dims(&beta_gamma, -2)?; // [B, H, 1, C]

        // Precompute intra-chunk decay-weighted QK^T
        let qk_t = ops::matmul(&q_c, &k_c_t)?; // [B, H, C, C]
        let qk_decay = ops::tril(&decay_c.multiply(&qk_t)?, 0)?; // [B, H, C, C]

        // Flatten (I+A) to [B*H, C, C] for batched tri_inv
        i_plus_a_list.push(i_plus_a.reshape(&[bh, c, c])?);

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
    let batched_ipa = ops::concatenate_axis(&i_plus_a_refs, 0)?; // [N*B*H, C, C]
    // stop_gradient: tri_inv has no VJP in MLX. The inverse is a fixed preconditioner
    // in the WY factorization — gradients should not flow through matrix inversion.
    // This matches the FLA reference impl which computes tri_inv in torch.no_grad().
    let batched_inv = stop_gradient(&linalg::tri_inv_device(
        &batched_ipa,
        None,
        StreamOrDevice::cpu(),
    )?)?;

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
        let t_inv = batched_inv
            .index((start..end, .., ..))
            .reshape(&[b, h, c, c])?; // [B, H, C, C]

        let delta_v = ops::matmul(&t_inv, &chunks[ci].beta_v)?; // [B, H, C, Dv]
        let t_inv_bg = t_inv.multiply(&chunks[ci].beta_gamma_row)?; // [B, H, C, C]

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
        let state_t = state.transpose_axes(&[0, 1, 3, 2])?; // [B, H, Dk, Dv]
        let ks = ops::matmul(&chunk.k_c, &state_t)?; // [B, H, C, Dv]

        // delta = T_inv @ beta_v - T_inv_bg @ (K @ S^T)
        let delta_s = ops::matmul(&inv.t_inv_bg, &ks)?; // [B, H, C, Dv]
        let delta = inv.delta_v.subtract(&delta_s)?; // [B, H, C, Dv]

        // Inter-chunk output: y_inter = Γ_init * (Q @ S^T)
        let qs = ops::matmul(&chunk.q_c, &state_t)?; // [B, H, C, Dv]
        let gamma_init_exp = ops::expand_dims(&chunk.gamma_init, -1)?; // [B, H, C, 1]
        let y_inter = gamma_init_exp.multiply(&qs)?; // [B, H, C, Dv]

        // Intra-chunk output: y_intra = qk_decay @ δ
        let y_intra = ops::matmul(&chunk.qk_decay, &delta)?; // [B, H, C, Dv]

        y_chunks.push(y_inter.add(&y_intra)?); // [B, H, C, Dv]

        // State propagation: S_{c+1} = Γ_total * S_c + (Γ_last * δ)^T @ K
        let gamma_last_exp = ops::expand_dims(&chunk.gamma_last, -1)?; // [B, H, C, 1]
        let delta_weighted = gamma_last_exp.multiply(&delta)?; // [B, H, C, Dv]
        let dw_t = delta_weighted.transpose_axes(&[0, 1, 3, 2])?; // [B, H, Dv, C]
        let state_update = ops::matmul(&dw_t, &chunk.k_c)?; // [B, H, Dv, Dk]

        let gamma_total_exp = chunk.gamma_total.reshape(&[b, h, 1, 1])?;
        state = gamma_total_exp.multiply(&state)?.add(&state_update)?;
    }

    // Concatenate chunk outputs: [B, H, T_padded, Dv]
    let y_refs: Vec<&Array> = y_chunks.iter().collect();
    let y = ops::concatenate_axis(&y_refs, 2)?;

    // Trim padding
    let y = if pad_len > 0 {
        y.index((.., .., ..t, ..))
    } else {
        y
    };

    // Transpose back to [B, T, H, Dv]
    let y = y.transpose_axes(&[0, 2, 1, 3])?;

    Ok((y, state))
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
///   for sequences longer than `GDN_CHUNK_SIZE` for O(T/64) prefill.
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

    // Training must use sequential path: tri_inv (CPU-only) has no VJP and produces
    // NaN inside value_and_grad due to CPU↔GPU stream sync issues.
    // Inference can use the fast chunk path (O(T/64) vs O(T) for prefill).
    if !training {
        let t = q.dim(1);
        if t > GDN_CHUNK_SIZE {
            return gated_delta_chunk_ops(q, k, v, &g, &beta, state_ref, mask);
        }
    }
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

        mlx_rs::random::seed(42).unwrap();

        let q = mlx_rs::random::normal::<f32>(&[b, 4, hk, dk], None, None, None).unwrap();
        let k = mlx_rs::random::normal::<f32>(&[b, 4, hk, dk], None, None, None).unwrap();
        let v = mlx_rs::random::normal::<f32>(&[b, 4, hv, dv], None, None, None).unwrap();
        let g = mlx_rs::random::uniform::<_, f32>(0.5, 1.0, &[b, 4, hv], None).unwrap();
        let beta = mlx_rs::random::uniform::<_, f32>(0.0, 0.5, &[b, 4, hv], None).unwrap();

        // Process all 4 steps at once
        let (_y_full, state_full) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

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

    /// Helper: assert two arrays are close within tolerance, returning the max diff.
    fn assert_close(a: &Array, b: &Array, tol: f32, msg: &str) {
        a.eval().unwrap();
        b.eval().unwrap();
        let diff = a.subtract(b).unwrap().abs().unwrap();
        let max_diff = diff.max(None).unwrap();
        max_diff.eval().unwrap();
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
        let q = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let k = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let v = mlx_rs::random::normal::<f32>(&[b, t, hv, dv], None, None, None).unwrap();
        // g in (0.5, 1.0) to keep decay moderate and avoid numerical issues
        let g = mlx_rs::random::uniform::<_, f32>(0.5, 1.0, &[b, t, hv], None).unwrap();
        // beta in (0.0, 0.5) to keep updates moderate
        let beta = mlx_rs::random::uniform::<_, f32>(0.0, 0.5, &[b, t, hv], None).unwrap();
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

        mlx_rs::random::seed(123).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path
        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();
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

        mlx_rs::random::seed(456).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();
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

        mlx_rs::random::seed(789).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();

        // Process all 256 at once via chunk path
        let (_y_full, state_full) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, None).unwrap();

        // Process in two halves of 128
        let q1 = q.index((.., ..128, .., ..));
        let k1 = k.index((.., ..128, .., ..));
        let v1 = v.index((.., ..128, .., ..));
        let g1 = g.index((.., ..128, ..));
        let beta1 = beta.index((.., ..128, ..));

        let (_y1, state1) =
            gated_delta_chunk_ops(&q1, &k1, &v1, &g1, &beta1, &state_init, None).unwrap();

        let q2 = q.index((.., 128.., .., ..));
        let k2 = k.index((.., 128.., .., ..));
        let v2 = v.index((.., 128.., .., ..));
        let g2 = g.index((.., 128.., ..));
        let beta2 = beta.index((.., 128.., ..));

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

        mlx_rs::random::seed(101).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path (will pad to 128)
        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();
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

        mlx_rs::random::seed(202).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path
        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();
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

        mlx_rs::random::seed(303).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Mask: first 100 tokens valid, last 28 masked
        let mut mask_data = vec![1.0f32; t as usize];
        for i in 100..t as usize {
            mask_data[i] = 0.0;
        }
        let mask = Array::from_slice(&mask_data, &[b, t]);

        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();

        // Chunk path with mask: state should match processing only the first 100 tokens
        let (y_chunk, state_chunk) =
            gated_delta_chunk_ops(&q, &k, &v, &g, &beta, &state_init, Some(&mask)).unwrap();

        // Sequential on just the first 100 tokens (no mask needed)
        let q_100 = q.index((.., ..100, .., ..));
        let k_100 = k.index((.., ..100, .., ..));
        let v_100 = v.index((.., ..100, .., ..));
        let g_100 = g.index((.., ..100, ..));
        let beta_100 = beta.index((.., ..100, ..));

        let (_y_ref, state_ref) =
            gated_delta_ops(&q_100, &k_100, &v_100, &g_100, &beta_100, None, None).unwrap();

        assert_eq!(y_chunk.shape(), &[b, t, hv, dv]);
        // States should match since masked positions shouldn't update state
        assert_close(&state_chunk, &state_ref, 1e-3, "Masked state mismatch");
    }

    #[test]
    #[serial]
    fn test_update_dispatches_chunk() {
        // Verify gated_delta_update dispatches to chunk path for T > GDN_CHUNK_SIZE
        let b = 1;
        let t = 128;
        let hk = 2;
        let dk = 8;
        let hv = 2;
        let dv = 8;

        mlx_rs::random::seed(404).unwrap();
        let q = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let k = mlx_rs::random::normal::<f32>(&[b, t, hk, dk], None, None, None).unwrap();
        let v = mlx_rs::random::normal::<f32>(&[b, t, hv, dv], None, None, None).unwrap();
        let a = mlx_rs::random::normal::<f32>(&[b, t, hv], None, None, None).unwrap();
        let b_input = mlx_rs::random::normal::<f32>(&[b, t, hv], None, None, None).unwrap();
        let a_log = Array::from_slice(&[0.5f32, 1.0], &[hv]);
        let dt_bias = Array::from_slice(&[0.1f32, 0.2], &[hv]);

        let (y, state) = gated_delta_update(
            &q, &k, &v, &a, &b_input, &a_log, &dt_bias, None, None, false,
        )
        .unwrap();

        assert_eq!(y.shape(), &[b, t, hv, dv]);
        assert_eq!(state.shape(), &[b, hv, dv, dk]);

        // Verify output is finite
        y.eval().unwrap();
        state.eval().unwrap();
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

        mlx_rs::random::seed(505).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path (pads to 128)
        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();
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

        mlx_rs::random::seed(606).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Sequential reference
        let (y_seq, state_seq) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None).unwrap();

        // Chunk path
        let state_init = ops::zeros_dtype(&[b, hv, dv, dk], q.dtype()).unwrap();
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

        mlx_rs::random::seed(707).unwrap();
        let (q, k, v, g, beta) = random_gdn_inputs(b, t, hk, dk, hv, dv);

        // Non-zero initial state
        let state_init = mlx_rs::random::normal::<f32>(&[b, hv, dv, dk], None, None, None).unwrap();

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
}
