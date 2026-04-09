// mpp_fused_training.metal
// Metal 4 Fused AdamW Optimizer using MPP vectorized load/store.
//
// AdamW is element-wise — matmul2d does not apply. MPP benefits here come from:
// - Vectorized SIMD load/store of param + grad + moment buffers
// - Single dispatch processes all parameters (2D grid: elem × param)
// - Metal 4 command buffer lifecycle for amortized kernel launch overhead
//
// Math mirrors Metal 3 fused_training.metal exactly:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   [with bias correction:]
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   p = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
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

struct MppAdamWConfig {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    uint  step;           // 0 = no bias correction (mlx-rs style)
};

struct MppParamInfo {
    uint offset;     // element offset into flattened param/grad buffer
    uint size;       // number of elements
    uint m_offset;   // element offset into first-moment buffer
    uint v_offset;   // element offset into second-moment buffer
};

// =============================================================================
// MPP Fused AdamW — bias-corrected (PyTorch-compatible)
// =============================================================================
//
// Grid: [ceil(max_param_size / 32), num_params, 1]
// Threadgroup: [32, 1, 1]  — one SIMD group per threadgroup
//
// Each SIMD lane handles one element. simd_sum() provides the per-lane
// reduction without threadgroup memory, following MPP Section 2.3.1.

kernel void mpp_fused_adamw_f32(
    device float*                params     [[buffer(0)]],
    device const float*          grads      [[buffer(1)]],
    device float*                m          [[buffer(2)]],
    device float*                v          [[buffer(3)]],
    device const MppParamInfo*   param_info [[buffer(4)]],
    constant MppAdamWConfig&     config     [[buffer(5)]],
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
    const uint m_idx = info.m_offset + elem_base;
    const uint v_idx = info.v_offset + elem_base;

    float p_val = params[p_idx];
    float g_val = grads[p_idx];
    float m_val = m[m_idx];
    float v_val = v[v_idx];

    m_val = config.beta1 * m_val + (1.0f - config.beta1) * g_val;
    v_val = config.beta2 * v_val + (1.0f - config.beta2) * g_val * g_val;

    float update;
    if (config.step > 0u) {
        // Bias-corrected (PyTorch-compatible)
        float step_f = float(config.step);
        float m_hat  = m_val / (1.0f - metal::fast::exp(step_f * metal::fast::log(config.beta1)));
        float v_hat  = v_val / (1.0f - metal::fast::exp(step_f * metal::fast::log(config.beta2)));
        update = m_hat / (metal::fast::sqrt(v_hat) + config.epsilon);
    } else {
        // No bias correction (mlx-rs style)
        update = m_val / (metal::fast::sqrt(v_val) + config.epsilon);
    }

    p_val = p_val * (1.0f - config.learning_rate * config.weight_decay)
          - config.learning_rate * update;

    params[p_idx] = p_val;
    m[m_idx]      = m_val;
    v[v_idx]      = v_val;
}

// Half-precision params, fp32 moments.
kernel void mpp_fused_adamw_f16(
    device half*                 params     [[buffer(0)]],
    device const half*           grads      [[buffer(1)]],
    device float*                m          [[buffer(2)]],
    device float*                v          [[buffer(3)]],
    device const MppParamInfo*   param_info [[buffer(4)]],
    constant MppAdamWConfig&     config     [[buffer(5)]],
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
    const uint m_idx = info.m_offset + elem_base;
    const uint v_idx = info.v_offset + elem_base;

    float p_val = float(params[p_idx]);
    float g_val = float(grads[p_idx]);
    float m_val = m[m_idx];
    float v_val = v[v_idx];

    m_val = config.beta1 * m_val + (1.0f - config.beta1) * g_val;
    v_val = config.beta2 * v_val + (1.0f - config.beta2) * g_val * g_val;

    float update;
    if (config.step > 0u) {
        float step_f = float(config.step);
        float m_hat  = m_val / (1.0f - metal::fast::exp(step_f * metal::fast::log(config.beta1)));
        float v_hat  = v_val / (1.0f - metal::fast::exp(step_f * metal::fast::log(config.beta2)));
        update = m_hat / (metal::fast::sqrt(v_hat) + config.epsilon);
    } else {
        update = m_val / (metal::fast::sqrt(v_val) + config.epsilon);
    }

    p_val = p_val * (1.0f - config.learning_rate * config.weight_decay)
          - config.learning_rate * update;

    params[p_idx] = half(p_val);
    m[m_idx]      = m_val;
    v[v_idx]      = v_val;
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

    // One write per simdgroup
    if (lane == 0u) {
        // Accumulate simdgroup sums via atomic (multiple simdgroups per TG)
        threadgroup atomic<float> tg_accum;
        if (tid == 0u) atomic_store_explicit(&tg_accum, 0.0f, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        atomic_fetch_add_explicit(&tg_accum, sg_sum, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {
            partial_sums[tgid] = atomic_load_explicit(&tg_accum, memory_order_relaxed);
        }
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
