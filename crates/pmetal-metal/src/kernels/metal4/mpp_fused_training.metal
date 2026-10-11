// mpp_fused_training.metal
// Metal 4 Fused AdamW Optimizer using MPP vectorized load/store.
//
// AdamW is element-wise — matmul2d does not apply. MPP benefits here come from:
// - Vectorized SIMD load/store of param + grad + moment buffers
// - Single dispatch processes all parameters (2D grid: elem × param)
// - Metal 4 command buffer lifecycle for amortized kernel launch overhead
//
// Math mirrors Metal 3 fused_training.metal exactly, which is
// torch.optim.AdamW's update in its order:
//   p = p * (1 - lr * wd)
//   m = m + (1 - beta1) * (g - m)
//   v = beta2 * v + (1 - beta2) * g^2
//   p = p - lr / (1 - beta1^t) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
//
// MPP Guide Section 2.3.4 (Postfix Fusion): simd_sum() reductions inline
// without threadgroup memory round-trips.
//
// MPP Guide Section 2.3.1 (Single simdgroup): gradient norm kernel uses
// execution_simdgroup for cross-lane reduction, avoiding threadgroup barriers.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// =============================================================================
// Shared structures (must match Rust repr(C))
// =============================================================================

/// One AdamW step's scalars, computed on the host (`AdamWScalars` in
/// fused_training.rs).
struct AdamWScalars {
    float decay;                  // 1 - lr * weight_decay
    float one_minus_beta1;        // 1 - beta1
    float beta2;
    float one_minus_beta2;        // 1 - beta2
    float step_size;              // lr / (1 - beta1^t)
    float bias_correction2_sqrt;  // sqrt(1 - beta2^t)
    float epsilon;
};

struct MppParamInfo {
    uint offset;     // element offset into flattened param/grad buffer
    uint size;       // number of elements
    uint m_offset;   // element offset into first-moment buffer
    uint v_offset;   // element offset into second-moment buffer
};

/// One element's AdamW update; `p` and `g` are already f32.
inline float adamw_element(
    float p, float g, device float* m, device float* v, uint m_idx, uint v_idx,
    constant AdamWScalars& s
) {
    const float m_val = m[m_idx] + s.one_minus_beta1 * (g - m[m_idx]);
    const float v_val = s.beta2 * v[v_idx] + s.one_minus_beta2 * g * g;
    const float denom = sqrt(v_val) / s.bias_correction2_sqrt + s.epsilon;
    m[m_idx] = m_val;
    v[v_idx] = v_val;
    return p * s.decay - s.step_size * (m_val / denom);
}

// =============================================================================
// MPP Fused AdamW
// =============================================================================
//
// Grid: [ceil(max_param_size / 32), num_params, 1]
// Threadgroup: [32, 1, 1]  — one SIMD group per threadgroup

kernel void mpp_fused_adamw_f32(
    device float*                params     [[buffer(0)]],
    device const float*          grads      [[buffer(1)]],
    device float*                m          [[buffer(2)]],
    device float*                v          [[buffer(3)]],
    device const MppParamInfo*   param_info [[buffer(4)]],
    constant AdamWScalars&       s          [[buffer(5)]],
    constant uint&               num_params [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint param_idx = tgid.y;
    if (param_idx >= num_params) return;

    const MppParamInfo info = param_info[param_idx];

    // Each threadgroup covers a 32-element SIMD stripe of this parameter.
    const uint elem_base = tgid.x * 32u + lane;
    if (elem_base >= info.size) return;

    const uint p_idx = info.offset + elem_base;
    params[p_idx] = adamw_element(
        params[p_idx], grads[p_idx], m, v,
        info.m_offset + elem_base, info.v_offset + elem_base, s);
}

// Half-precision params, fp32 moments.
kernel void mpp_fused_adamw_f16(
    device half*                 params     [[buffer(0)]],
    device const half*           grads      [[buffer(1)]],
    device float*                m          [[buffer(2)]],
    device float*                v          [[buffer(3)]],
    device const MppParamInfo*   param_info [[buffer(4)]],
    constant AdamWScalars&       s          [[buffer(5)]],
    constant uint&               num_params [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint param_idx = tgid.y;
    if (param_idx >= num_params) return;

    const MppParamInfo info = param_info[param_idx];
    const uint elem_base = tgid.x * 32u + lane;
    if (elem_base >= info.size) return;

    const uint p_idx = info.offset + elem_base;
    params[p_idx] = half(adamw_element(
        float(params[p_idx]), float(grads[p_idx]), m, v,
        info.m_offset + elem_base, info.v_offset + elem_base, s));
}

// =============================================================================
// MPP Gradient Norm + Scaling
// =============================================================================
//
// Partial reduction: each threadgroup produces one partial sum.
// Uses simd_sum() for the cross-lane reduction — no threadgroup memory.
// Grid: [ceil(total_elements / 128), 1, 1]  (4 SIMD groups per threadgroup)
// Threadgroup: [128, 1, 1]

kernel void mpp_gradient_norm_partial(
    device const float*  grads        [[buffer(0)]],
    device float*        partial_sums [[buffer(1)]],
    constant uint&       total_elems  [[buffer(2)]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  tgid   [[threadgroup_position_in_grid]],
    uint  tg_sz  [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sg_id  [[simdgroup_index_in_threadgroup]]
) {
    // Each thread handles 4 elements (float4) for bandwidth efficiency.
    float local_sum = 0.0f;
    uint base = (tgid * tg_sz + tid) * 4u;

    if (base + 3u < total_elems) {
        float4 g4 = *(device const float4*)(grads + base);
        local_sum = dot(g4, g4);
    } else {
        for (uint i = 0; i < 4u; i++) {
            uint idx = base + i;
            if (idx < total_elems) {
                float g = grads[idx];
                local_sum += g * g;
            }
        }
    }

    // SIMD-lane reduction — no threadgroup barrier needed (MPP 2.3.1)
    float sg_sum = simd_sum(local_sum);

    // Cross-simdgroup reduction via threadgroup memory.
    // Grid config: 4 simdgroups per threadgroup (tg_sz=128 / 32 = 4).
    // Xcode 26.4 SDK rejected the prior `threadgroup atomic<float>` pattern
    // on this path, so use an explicit fanin through a 4-slot scratch array.
    constexpr uint MAX_SIMDGROUPS_PER_TG = 4;
    threadgroup float sg_sums[MAX_SIMDGROUPS_PER_TG];
    if (lane == 0u) {
        sg_sums[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < MAX_SIMDGROUPS_PER_TG; ++i) {
            total += sg_sums[i];
        }
        partial_sums[tgid] = total;
    }
}

// Scale gradients by a scalar factor (for gradient clipping).
// Grid: [ceil(total_elements / 32), 1, 1]  Threadgroup: [32, 1, 1]
kernel void mpp_scale_gradients(
    device float*      grads       [[buffer(0)]],
    constant float&    scale       [[buffer(1)]],
    constant uint&     total_elems [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint base = tid * 4u;
    if (base + 3u < total_elems) {
        float4 g4 = *(device const float4*)(grads + base);
        *(device float4*)(grads + base) = g4 * scale;
    } else {
        for (uint i = base; i < min(base + 4u, total_elems); i++) {
            grads[i] *= scale;
        }
    }
}

kernel void mpp_scale_gradients_f16(
    device half*       grads       [[buffer(0)]],
    constant float&    scale       [[buffer(1)]],
    constant uint&     total_elems [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint base = tid * 4u;
    if (base + 3u < total_elems) {
        half4 g4 = *(device const half4*)(grads + base);
        *(device half4*)(grads + base) = half4(float4(g4) * scale);
    } else {
        for (uint i = base; i < min(base + 4u, total_elems); i++) {
            grads[i] = half(float(grads[i]) * scale);
        }
    }
}
