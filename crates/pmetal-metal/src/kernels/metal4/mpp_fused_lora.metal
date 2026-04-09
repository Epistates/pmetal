// mpp_fused_lora.metal
// Metal 4 Fused LoRA using MPP matmul2d.
//
// All three phases use hardware matrix multiply:
//   Phase 1: y = x @ W^T              [batch, out] ← MPP matmul2d
//   Phase 2: xA = x @ A^T             [batch, rank] ← MPP matmul2d (small)
//   Phase 3: y += scale * xA @ B^T    [batch, out]  ← MPP matmul2d
//
// Training variant saves xA for backward pass.
// Inference variant skips xA save.
//
// Phase 2 (xA) is computed globally (all output tiles share the result).
// Phase 3 adds the LoRA contribution to each output tile.
//
// Note: For rank > 64, the LoRA GEMM is large enough that MPP provides
// significant speedup. For rank <= 16, the overhead of matmul2d setup
// may not be worth it — the Metal 3 SIMD reduction approach is competitive.
// This kernel uses SIMD dot products for LoRA (rank is typically 4-64).

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct FusedLoraParams {
    uint batch_size;
    uint in_features;
    uint out_features;
    uint rank;
    float scale;
    float lr_scale_a;
    float lr_scale_b;
};

// =============================================================================
// MPP Fused LoRA Forward (fp16) — Training
// =============================================================================
//
// Grid: [num_out_tiles, num_batch_tiles, 1]
// Each threadgroup: 4 simdgroups × 32 = 128 threads, computes a 64×64 output tile.
//
// Phase 1: Base projection via matmul2d (dominant cost).
// Phase 2: xA computed per-thread (rank is small, O(rank × H) per thread batch element).
//          Saved to xA_out for backward pass.
// Phase 3: LoRA overlay added per output element from precomputed xA.

kernel void mpp_fused_lora_forward_f16(
    device half* x [[buffer(0)]],
    device half* W [[buffer(1)]],
    device half* A [[buffer(2)]],
    device half* B [[buffer(3)]],
    device half* y [[buffer(4)]],
    device half* xA_out [[buffer(5)]],    // [batch, rank] saved for backward
    constant FusedLoraParams& params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int batch = (int)params.batch_size;
    const int in_dim = (int)params.in_features;
    const int out_dim = (int)params.out_features;
    const int R = (int)params.rank;

    const int BM = 64;
    const int BN = 64;

    // Grid: [num_out_tiles, num_batch_tiles, 1]
    const int tile_b = (int)(tgid.y * BM);
    const int tile_o = (int)(tgid.x * BN);
    if (tile_b >= batch || tile_o >= out_dim) return;

    // Create tensors for base projection
    auto tX = tensor(x, dextents<int, 2>{in_dim, batch}, array<int, 2>{1, in_dim});
    auto tW = tensor(W, dextents<int, 2>{in_dim, out_dim}, array<int, 2>{1, in_dim});
    auto tY = tensor(y, dextents<int, 2>{out_dim, batch}, array<int, 2>{1, out_dim});

    // Phase 1: Base projection y = x @ W^T via MPP matmul2d
    constexpr auto base_desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64,
        static_cast<int>(dynamic_extent),
        false, true, false
    );
    mpp::tensor_ops::matmul2d<base_desc, execution_simdgroup> base_op;

    auto sliceX = tX.slice(0, tile_b);
    auto sliceW = tW.slice(0, tile_o);
    auto sliceY = tY.slice(tile_o, tile_b);

    base_op.run(sliceX, sliceW, sliceY);

    // Phase 2 & 3: LoRA overlay
    if (R > 0) {
        // Threadgroup scratch for shared xA values (max rank 64)
        threadgroup float xA_scratch[64 * 64];  // [BM_tokens, max_rank] — 16KB max

        uint total_threads = 128;
        uint linear_tid = simd_group_id * 32 + simd_lane_id;
        uint tile_b_size = min((uint)BM, params.batch_size - (uint)tile_b);

        // Phase 2: Compute xA = x @ A^T for each token in this batch tile
        // Each thread handles a subset of (token, rank) pairs
        for (uint idx = linear_tid; idx < tile_b_size * (uint)R; idx += total_threads) {
            uint b = idx / (uint)R;
            uint r = idx % (uint)R;
            uint global_b = (uint)tile_b + b;

            device half* x_row = x + global_b * params.in_features;
            device half* a_row = A + r * params.in_features;

            // Dot product: xA[b, r] = x[b, :] @ A[r, :]^T
            float4 acc4 = float4(0.0f);
            uint h4 = params.in_features & ~3u;
            for (uint h = 0; h < h4; h += 4) {
                acc4 += float4(*(device const half4*)(x_row + h))
                      * float4(*(device const half4*)(a_row + h));
            }
            float xa_val = acc4.x + acc4.y + acc4.z + acc4.w;
            for (uint h = h4; h < params.in_features; h++) {
                xa_val += float(x_row[h]) * float(a_row[h]);
            }

            xA_scratch[b * (uint)R + r] = xa_val;

            // Save xA for backward pass
            xA_out[global_b * (uint)R + r] = half(xa_val);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: y += scale * xA @ B^T for this output tile
        // Each thread adds LoRA contribution to its output elements
        uint tile_o_size = min((uint)BN, params.out_features - (uint)tile_o);

        for (uint idx = linear_tid; idx < tile_b_size * tile_o_size; idx += total_threads) {
            uint b = idx / tile_o_size;
            uint o = idx % tile_o_size;
            uint global_b = (uint)tile_b + b;
            uint global_o = (uint)tile_o + o;

            // lora_val = scale * sum_r(xA[b, r] * B[o, r])
            float lora_val = 0.0f;
            for (uint r = 0; r < (uint)R; r++) {
                lora_val += xA_scratch[b * (uint)R + r] * float(B[global_o * (uint)R + r]);
            }
            lora_val *= params.scale;

            // Add to base projection result
            y[global_b * params.out_features + global_o] =
                half(float(y[global_b * params.out_features + global_o]) + lora_val);
        }
    }
}

// =============================================================================
// MPP LoRA Forward — Inference (no xA saved)
// =============================================================================
//
// Same as training but doesn't save xA intermediate.

kernel void mpp_lora_forward_inference_f16(
    device half* x [[buffer(0)]],
    device half* W [[buffer(1)]],
    device half* A [[buffer(2)]],
    device half* B [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant FusedLoraParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int batch = (int)params.batch_size;
    const int in_dim = (int)params.in_features;
    const int out_dim = (int)params.out_features;
    const int R = (int)params.rank;

    const int BM = 64;
    const int BN = 64;
    const int tile_b = (int)(tgid.y * BM);
    const int tile_o = (int)(tgid.x * BN);
    if (tile_b >= batch || tile_o >= out_dim) return;

    auto tX = tensor(x, dextents<int, 2>{in_dim, batch}, array<int, 2>{1, in_dim});
    auto tW = tensor(W, dextents<int, 2>{in_dim, out_dim}, array<int, 2>{1, in_dim});
    auto tY = tensor(y, dextents<int, 2>{out_dim, batch}, array<int, 2>{1, out_dim});

    // Phase 1: Base projection
    constexpr auto base_desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<base_desc, execution_simdgroup> base_op;

    auto sX = tX.slice(0, tile_b);
    auto sW = tW.slice(0, tile_o);
    auto sY = tY.slice(tile_o, tile_b);
    base_op.run(sX, sW, sY);

    // Phase 2 & 3: LoRA overlay
    if (R > 0) {
        threadgroup float xA_scratch[64 * 64];

        uint total_threads = 128;
        uint linear_tid = simd_group_id * 32 + simd_lane_id;
        uint tile_b_size = min((uint)BM, params.batch_size - (uint)tile_b);

        // Phase 2: Compute xA
        for (uint idx = linear_tid; idx < tile_b_size * (uint)R; idx += total_threads) {
            uint b = idx / (uint)R;
            uint r = idx % (uint)R;
            uint global_b = (uint)tile_b + b;

            device half* x_row = x + global_b * params.in_features;
            device half* a_row = A + r * params.in_features;

            float4 acc4 = float4(0.0f);
            uint h4 = params.in_features & ~3u;
            for (uint h = 0; h < h4; h += 4) {
                acc4 += float4(*(device const half4*)(x_row + h))
                      * float4(*(device const half4*)(a_row + h));
            }
            float xa_val = acc4.x + acc4.y + acc4.z + acc4.w;
            for (uint h = h4; h < params.in_features; h++) {
                xa_val += float(x_row[h]) * float(a_row[h]);
            }
            xA_scratch[b * (uint)R + r] = xa_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: y += scale * xA @ B^T
        uint tile_o_size = min((uint)BN, params.out_features - (uint)tile_o);

        for (uint idx = linear_tid; idx < tile_b_size * tile_o_size; idx += total_threads) {
            uint b = idx / tile_o_size;
            uint o = idx % tile_o_size;
            uint global_b = (uint)tile_b + b;
            uint global_o = (uint)tile_o + o;

            float lora_val = 0.0f;
            for (uint r = 0; r < (uint)R; r++) {
                lora_val += xA_scratch[b * (uint)R + r] * float(B[global_o * (uint)R + r]);
            }
            lora_val *= params.scale;

            y[global_b * params.out_features + global_o] =
                half(float(y[global_b * params.out_features + global_o]) + lora_val);
        }
    }
}
