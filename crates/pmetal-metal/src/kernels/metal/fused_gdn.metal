//! Fused Gated Delta Network (GDN) recurrent kernel.
//!
//! Implements the delta-rule recurrence in a single kernel dispatch:
//!   state = state * exp(g) + k^T * (beta * (v - state @ k))
//!   y = state @ q
//!
//! Ported from FLA (flash-linear-attention) `fused_recurrent_gated_delta_rule_fwd_kernel`.
//!
//! Grid: (NV, B * Hv) where NV = ceil(Dv / BV)
//! Threadgroup: 32 threads (1 SIMD group), each thread handles KEY_DIM/32 K positions.
//!
//! Key reductions (sum over K dimension) use simd_sum() within the SIMD group.
//! State [KEY_DIM/32, BV] is kept in thread-private registers — no threadgroup memory.

#include <metal_stdlib>
using namespace metal;

// Function constants for compile-time specialization.
// Set via MTLFunctionConstantValues when creating the pipeline.
constant uint FC_KEY_DIM     [[function_constant(0)]];  // Dk (64, 128, 256)
constant uint FC_VALUE_BLOCK [[function_constant(1)]];  // BV (8, 16)
constant uint FC_SCALAR_GATE [[function_constant(2)]];  // 1=scalar gate [B,T,Hv], 0=vector [B,T,Hv,Dk]

// Derived compile-time constants.
constant uint SIMD_WIDTH = 32;
constant uint K_PER_THREAD = FC_KEY_DIM / SIMD_WIDTH;

/// Runtime parameters passed via setBytes.
struct GdnRecurrentParams {
    uint batch_size;     // B
    uint num_heads;      // Hv (value heads, q/k already expanded for GQA)
    uint key_dim;        // Dk
    uint value_dim;      // Dv
    uint seq_len;        // T (1 for decode, ≤64 for short prefill)
};

/// Fused forward recurrence kernel.
///
/// Inputs (all f32):
///   q: [B, T, Hv, Dk]     — queries (GQA-expanded to Hv heads)
///   k: [B, T, Hv, Dk]     — keys (GQA-expanded to Hv heads)
///   v: [B, T, Hv, Dv]     — values
///   g: [B, T, Hv]          — scalar gating decay factor in (0,1] from compute_g()
///   beta: [B, T, Hv]       — beta gate (sigmoid-space)
///
/// In/Out:
///   state: [B, Hv, Dv, Dk] — recurrent state (read initial, write final)
///
/// Output:
///   output: [B, T, Hv, Dv] — output for each timestep
kernel void gdn_fused_recurrent_fwd(
    device const float* q          [[buffer(0)]],
    device const float* k          [[buffer(1)]],
    device const float* v          [[buffer(2)]],
    device const float* g          [[buffer(3)]],
    device const float* beta       [[buffer(4)]],
    device float* state            [[buffer(5)]],
    device float* output           [[buffer(6)]],
    constant GdnRecurrentParams& params [[buffer(7)]],
    uint2 tgid    [[threadgroup_position_in_grid]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {
    const uint v_tile = tgid.x;            // Which BV tile of the value dimension
    const uint bh_idx = tgid.y;            // batch * num_heads + head
    const uint batch  = bh_idx / params.num_heads;
    const uint head   = bh_idx % params.num_heads;

    const uint Dk = params.key_dim;
    const uint Dv = params.value_dim;
    const uint T  = params.seq_len;
    const uint Hv = params.num_heads;

    const uint v_start = v_tile * FC_VALUE_BLOCK;
    if (v_start >= Dv) return;
    const uint bv = min(FC_VALUE_BLOCK, Dv - v_start);

    // Each thread handles K_PER_THREAD consecutive K positions.
    // lane_id ∈ [0, 31], k_base = lane_id * K_PER_THREAD
    const uint k_base = lane_id * K_PER_THREAD;

    // Load initial state into registers.
    // State layout: [B, Hv, Dv, Dk], element (b,h,dv,dk) at ((b*Hv+h)*Dv+dv)*Dk+dk
    float h[8][16]; // Max K_PER_THREAD=8 (Dk=256), max BV=16
    {
        const uint s_base = ((batch * Hv + head) * Dv + v_start) * Dk + k_base;
        for (uint ki = 0; ki < K_PER_THREAD; ki++) {
            for (uint vi = 0; vi < bv; vi++) {
                h[ki][vi] = state[s_base + vi * Dk + ki];
            }
        }
    }

    // Process each timestep sequentially.
    for (uint t = 0; t < T; t++) {
        // Load q[b, t, h, k_base..k_base+K_PER_THREAD] and k[same]
        // Layout: [B, T, Hv, Dk]
        const uint qk_base = ((batch * T + t) * Hv + head) * Dk + k_base;
        float q_val[8], k_val[8];
        for (uint ki = 0; ki < K_PER_THREAD; ki++) {
            q_val[ki] = q[qk_base + ki];
            k_val[ki] = k[qk_base + ki];
        }

        // Load scalar gate and beta.
        // g layout: [B, T, Hv], beta layout: [B, T, Hv]
        const uint gb_idx = (batch * T + t) * Hv + head;
        const float g_val    = g[gb_idx];
        const float beta_val = beta[gb_idx];

        // Load v[b, t, h, v_start..v_start+bv]
        // Layout: [B, T, Hv, Dv]
        const uint v_base = ((batch * T + t) * Hv + head) * Dv + v_start;
        float v_val[16];
        for (uint vi = 0; vi < bv; vi++) {
            v_val[vi] = v[v_base + vi];
        }

        // --- Step 1: Decay state ---
        // h *= g  (g is already in exp-space from compute_g, i.e., the actual decay factor)
        for (uint ki = 0; ki < K_PER_THREAD; ki++) {
            for (uint vi = 0; vi < bv; vi++) {
                h[ki][vi] *= g_val;
            }
        }

        // --- Step 2: Compute kv_mem = sum_k(h[k,:] * k[k]) → [BV] ---
        // Each thread computes partial sum over its K_PER_THREAD rows.
        float kv_partial[16];
        for (uint vi = 0; vi < bv; vi++) {
            float acc = 0.0f;
            for (uint ki = 0; ki < K_PER_THREAD; ki++) {
                acc += h[ki][vi] * k_val[ki];
            }
            kv_partial[vi] = acc;
        }
        // SIMD reduction across the 32 lanes → all lanes get the full K-dim sum.
        float kv_mem[16];
        for (uint vi = 0; vi < bv; vi++) {
            kv_mem[vi] = simd_sum(kv_partial[vi]);
        }

        // --- Step 3: Delta = beta * (v - kv_mem) ---
        float delta[16];
        for (uint vi = 0; vi < bv; vi++) {
            delta[vi] = beta_val * (v_val[vi] - kv_mem[vi]);
        }

        // --- Step 4: Rank-1 state update ---
        // h += k[:, None] * delta[None, :]  (outer product)
        for (uint ki = 0; ki < K_PER_THREAD; ki++) {
            for (uint vi = 0; vi < bv; vi++) {
                h[ki][vi] += k_val[ki] * delta[vi];
            }
        }

        // --- Step 5: Output = sum_k(h[k,:] * q[k]) → [BV] ---
        float out_partial[16];
        for (uint vi = 0; vi < bv; vi++) {
            float acc = 0.0f;
            for (uint ki = 0; ki < K_PER_THREAD; ki++) {
                acc += h[ki][vi] * q_val[ki];
            }
            out_partial[vi] = acc;
        }
        float out_val[16];
        for (uint vi = 0; vi < bv; vi++) {
            out_val[vi] = simd_sum(out_partial[vi]);
        }

        // Write output — only lane 0 writes (all lanes hold same value after simd_sum).
        if (lane_id == 0) {
            const uint out_base = ((batch * T + t) * Hv + head) * Dv + v_start;
            for (uint vi = 0; vi < bv; vi++) {
                output[out_base + vi] = out_val[vi];
            }
        }
    }

    // Write back final state — each thread writes its own K rows.
    {
        const uint s_base = ((batch * Hv + head) * Dv + v_start) * Dk + k_base;
        for (uint ki = 0; ki < K_PER_THREAD; ki++) {
            for (uint vi = 0; vi < bv; vi++) {
                state[s_base + vi * Dk + ki] = h[ki][vi];
            }
        }
    }
}
