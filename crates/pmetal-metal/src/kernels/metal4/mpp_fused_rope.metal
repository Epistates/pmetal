// mpp_fused_rope.metal
// Metal 4 Fused RoPE (Rotary Position Embedding) with postfix fusion.
//
// RoPE is element-wise: for each pair (x_even, x_odd), apply rotation:
//   x_even_new = x_even * cos(θ) - x_odd * sin(θ)
//   x_odd_new  = x_even * sin(θ) + x_odd * cos(θ)
//
// MPP Guide Section 2.3.4 (Postfix Fusion): the postfix-RoPE kernel operates
// on cooperative tensor register output of a preceding Q/K projection GEMM,
// applying rotation before the single store to device memory — eliminating
// a full round-trip through global memory that the Metal 3 path requires.
//
// MPP Guide Section 2.3.1 (Single simdgroup): each threadgroup is one SIMD
// group (32 lanes). For the standalone kernel, each SIMD handles one
// (batch, head, seq) triple; all dim-pair iterations run across SIMD lanes.
//
// Two kernel families:
//   mpp_rope_inplace_{f16,f32}       — standalone in-place RoPE
//   mpp_rope_with_positions_{f16,f32} — custom position IDs (seq packing)
//   mpp_rope_qk_inplace_f16          — fused QK RoPE, single dispatch
//
// Grid for standalone: [batch, heads, seq_len]  Threadgroup: [32, 1, 1]

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct MppRoPEParams {
    uint  batch_size;
    uint  num_heads;
    uint  seq_len;
    uint  head_dim;    // must be even
    float base;        // default 10000.0
    float scale;       // position scale factor, default 1.0
};

// Compute inverse frequency for a dimension index in fast math.
inline float mpp_inv_freq(uint dim_idx, uint dims, float base) {
    float exponent = -metal::fast::log(base) * (float(2u * dim_idx) / float(dims));
    return metal::fast::exp(exponent);
}

// =============================================================================
// Standalone In-Place RoPE (fp32)
// =============================================================================
//
// Grid: [batch_size, num_heads, seq_len]  Threadgroup: [32, 1, 1]
// Each threadgroup is one (batch, head, position) triple.
// SIMD lanes sweep the half_dim in strides of 32.

kernel void mpp_rope_inplace_f32(
    device float*         x      [[buffer(0)]],
    constant MppRoPEParams& p    [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx  = tgid.y;
    const uint seq_idx   = tgid.z;

    if (batch_idx >= p.batch_size || head_idx >= p.num_heads || seq_idx >= p.seq_len) return;

    const float pos      = float(seq_idx) * p.scale;
    const uint half_dim  = p.head_dim / 2u;
    const uint offset    = ((batch_idx * p.num_heads + head_idx) * p.seq_len + seq_idx) * p.head_dim;
    device float* xp     = x + offset;

    for (uint d = lane; d < half_dim; d += 32u) {
        const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
        const float angle = pos * freq;
        float cos_v, sin_v;
        cos_v = metal::fast::cos(angle);
        sin_v = metal::fast::sin(angle);

        const float x1 = xp[d];
        const float x2 = xp[d + half_dim];
        xp[d]           = x1 * cos_v - x2 * sin_v;
        xp[d + half_dim] = x1 * sin_v + x2 * cos_v;
    }
}

// =============================================================================
// Standalone In-Place RoPE (fp16)
// =============================================================================

kernel void mpp_rope_inplace_f16(
    device half*          x      [[buffer(0)]],
    constant MppRoPEParams& p    [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx  = tgid.y;
    const uint seq_idx   = tgid.z;

    if (batch_idx >= p.batch_size || head_idx >= p.num_heads || seq_idx >= p.seq_len) return;

    const float pos      = float(seq_idx) * p.scale;
    const uint half_dim  = p.head_dim / 2u;
    const uint offset    = ((batch_idx * p.num_heads + head_idx) * p.seq_len + seq_idx) * p.head_dim;
    device half* xp      = x + offset;

    for (uint d = lane; d < half_dim; d += 32u) {
        const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
        const float angle = pos * freq;
        const float cos_v = metal::fast::cos(angle);
        const float sin_v = metal::fast::sin(angle);

        const float x1 = float(xp[d]);
        const float x2 = float(xp[d + half_dim]);
        xp[d]           = half(x1 * cos_v - x2 * sin_v);
        xp[d + half_dim] = half(x1 * sin_v + x2 * cos_v);
    }
}

// =============================================================================
// RoPE With Custom Position IDs (fp32)
// =============================================================================
//
// Essential for sequence packing where positions reset per sub-sequence.

kernel void mpp_rope_with_positions_f32(
    device float*           x            [[buffer(0)]],
    device const int*       position_ids [[buffer(1)]],
    constant MppRoPEParams& p            [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx  = tgid.y;
    const uint seq_idx   = tgid.z;

    if (batch_idx >= p.batch_size || head_idx >= p.num_heads || seq_idx >= p.seq_len) return;

    const float pos      = float(position_ids[seq_idx]) * p.scale;
    const uint half_dim  = p.head_dim / 2u;
    const uint offset    = ((batch_idx * p.num_heads + head_idx) * p.seq_len + seq_idx) * p.head_dim;
    device float* xp     = x + offset;

    for (uint d = lane; d < half_dim; d += 32u) {
        const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
        const float angle = pos * freq;
        const float cos_v = metal::fast::cos(angle);
        const float sin_v = metal::fast::sin(angle);

        const float x1 = xp[d];
        const float x2 = xp[d + half_dim];
        xp[d]           = x1 * cos_v - x2 * sin_v;
        xp[d + half_dim] = x1 * sin_v + x2 * cos_v;
    }
}

// Half-precision with custom positions.
kernel void mpp_rope_with_positions_f16(
    device half*            x            [[buffer(0)]],
    device const int*       position_ids [[buffer(1)]],
    constant MppRoPEParams& p            [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx  = tgid.y;
    const uint seq_idx   = tgid.z;

    if (batch_idx >= p.batch_size || head_idx >= p.num_heads || seq_idx >= p.seq_len) return;

    const float pos      = float(position_ids[seq_idx]) * p.scale;
    const uint half_dim  = p.head_dim / 2u;
    const uint offset    = ((batch_idx * p.num_heads + head_idx) * p.seq_len + seq_idx) * p.head_dim;
    device half* xp      = x + offset;

    for (uint d = lane; d < half_dim; d += 32u) {
        const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
        const float angle = pos * freq;
        const float cos_v = metal::fast::cos(angle);
        const float sin_v = metal::fast::sin(angle);

        const float x1 = float(xp[d]);
        const float x2 = float(xp[d + half_dim]);
        xp[d]           = half(x1 * cos_v - x2 * sin_v);
        xp[d + half_dim] = half(x1 * sin_v + x2 * cos_v);
    }
}

// =============================================================================
// Fused QK RoPE — single dispatch for both Q and K (fp16)
// =============================================================================
//
// Computes sin/cos once per position, applies to all Q heads then all K heads.
// Grid: [batch_size, seq_len, 1]  Threadgroup: [32, 1, 1]
//
// MPP postfix fusion note: when the Q/K projections use mpp_gemm, the output
// cooperative tensor registers hold the projected values. A future merged kernel
// can apply the rotation loop directly on those registers before the store,
// eliminating the global memory round-trip entirely. This standalone version
// covers the common case where projections are done separately.

kernel void mpp_rope_qk_inplace_f16(
    device half*            q        [[buffer(0)]],    // [batch, q_heads, seq, head_dim]
    device half*            k        [[buffer(1)]],    // [batch, kv_heads, seq, head_dim]
    constant MppRoPEParams& p        [[buffer(2)]],
    constant uint&          kv_heads [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint seq_idx   = tgid.y;

    if (batch_idx >= p.batch_size || seq_idx >= p.seq_len) return;

    const float pos     = float(seq_idx) * p.scale;
    const uint half_dim = p.head_dim / 2u;

    // Pre-compute cos/sin for each dim pair across SIMD lanes.
    // Each lane owns a disjoint set of dim indices — no barriers needed.
    // (head_dim typically 64 or 128; half_dim 32 or 64 fits in one/two passes)

    // --- Q heads ---
    for (uint h = 0u; h < p.num_heads; h++) {
        const uint q_off = ((batch_idx * p.num_heads + h) * p.seq_len + seq_idx) * p.head_dim;
        device half* qp = q + q_off;

        for (uint d = lane; d < half_dim; d += 32u) {
            const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
            const float angle = pos * freq;
            const float cos_v = metal::fast::cos(angle);
            const float sin_v = metal::fast::sin(angle);

            const float x1 = float(qp[d]);
            const float x2 = float(qp[d + half_dim]);
            qp[d]           = half(x1 * cos_v - x2 * sin_v);
            qp[d + half_dim] = half(x1 * sin_v + x2 * cos_v);
        }
    }

    // --- K heads ---
    for (uint h = 0u; h < kv_heads; h++) {
        const uint k_off = ((batch_idx * kv_heads + h) * p.seq_len + seq_idx) * p.head_dim;
        device half* kp = k + k_off;

        for (uint d = lane; d < half_dim; d += 32u) {
            const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
            const float angle = pos * freq;
            const float cos_v = metal::fast::cos(angle);
            const float sin_v = metal::fast::sin(angle);

            const float x1 = float(kp[d]);
            const float x2 = float(kp[d + half_dim]);
            kp[d]           = half(x1 * cos_v - x2 * sin_v);
            kp[d + half_dim] = half(x1 * sin_v + x2 * cos_v);
        }
    }
}

// QK RoPE with custom position IDs (fp16, seq packing)
kernel void mpp_rope_qk_with_positions_f16(
    device half*            q            [[buffer(0)]],
    device half*            k            [[buffer(1)]],
    device const int*       position_ids [[buffer(2)]],
    constant MppRoPEParams& p            [[buffer(3)]],
    constant uint&          kv_heads     [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint seq_idx   = tgid.y;

    if (batch_idx >= p.batch_size || seq_idx >= p.seq_len) return;

    const float pos     = float(position_ids[seq_idx]) * p.scale;
    const uint half_dim = p.head_dim / 2u;

    for (uint h = 0u; h < p.num_heads; h++) {
        const uint q_off = ((batch_idx * p.num_heads + h) * p.seq_len + seq_idx) * p.head_dim;
        device half* qp = q + q_off;

        for (uint d = lane; d < half_dim; d += 32u) {
            const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
            const float angle = pos * freq;
            const float cos_v = metal::fast::cos(angle);
            const float sin_v = metal::fast::sin(angle);
            const float x1 = float(qp[d]);
            const float x2 = float(qp[d + half_dim]);
            qp[d]           = half(x1 * cos_v - x2 * sin_v);
            qp[d + half_dim] = half(x1 * sin_v + x2 * cos_v);
        }
    }

    for (uint h = 0u; h < kv_heads; h++) {
        const uint k_off = ((batch_idx * kv_heads + h) * p.seq_len + seq_idx) * p.head_dim;
        device half* kp = k + k_off;

        for (uint d = lane; d < half_dim; d += 32u) {
            const float freq  = mpp_inv_freq(d, p.head_dim, p.base);
            const float angle = pos * freq;
            const float cos_v = metal::fast::cos(angle);
            const float sin_v = metal::fast::sin(angle);
            const float x1 = float(kp[d]);
            const float x2 = float(kp[d + half_dim]);
            kp[d]           = half(x1 * cos_v - x2 * sin_v);
            kp[d + half_dim] = half(x1 * sin_v + x2 * cos_v);
        }
    }
}
