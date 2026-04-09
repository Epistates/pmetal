// mpp_fused_moe.metal
// Metal 4 Fused MoE Expert Forward using MPP matmul2d.
//
// Computes the full expert forward pass for dense (fp16) MoE experts:
//   gate = x @ gate_weight^T      [hidden → intermediate]
//   up   = x @ up_weight^T        [hidden → intermediate]
//   act  = silu(gate) * up        [SwiGLU — postfix fusion in registers]
//   out  = act @ down_weight^T    [intermediate → hidden]
//
// MPP Guide Section 2.3.4 (Postfix Fusion):
//   gate and up GEMMs keep results in cooperative tensor registers (rGate, rUp).
//   SwiGLU is applied element-wise in register space, producing rAct.
//   rAct is held in registers and used as the A input to the down projection
//   GEMM — no intermediate global memory write between gate/up and down.
//
// MPP Guide Section 2.3.1 (Single simdgroup):
//   execution_simdgroup is used throughout. NAX matmul2d provides hardware
//   acceleration for the 32×32 GEMM tiles.
//
// Tile dimensions: BM=32 (token), BN=32 (dim), matching the recommended
// single-simdgroup tile for NAX.
//
// For weighted residual combine (MoE scatter), a separate lightweight kernel
// adds the expert contribution scaled by the router gate weight.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct MppMoEParams {
    uint batch_size;        // number of tokens dispatched to this expert
    uint hidden_dim;        // model hidden dimension H
    uint intermediate_dim;  // expert intermediate dimension I
};

// Expert routing weight for weighted accumulation.
struct MppMoEScatterParams {
    uint num_tokens;
    uint hidden_dim;
};

inline float silu_moe(float x) {
    return x / (1.0f + metal::fast::exp(-x));
}

// =============================================================================
// MPP Fused MoE Expert Forward — gate+up SwiGLU (fp16)
// =============================================================================
//
// Produces act[batch, intermediate] = silu(gate) * up in registers.
// Output: intermediate activations held in rAct register array.
//
// Grid: [ceil(I/32), ceil(B/32), 1]  Threadgroup: [32, 1, 1]

kernel void mpp_fused_moe_gate_up_f16(
    device const half*   input         [[buffer(0)]],   // [B, H]
    device const half*   gate_weight   [[buffer(1)]],   // [I, H]
    device const half*   up_weight     [[buffer(2)]],   // [I, H]
    device half*         act_out       [[buffer(3)]],   // [B, I] — SwiGLU output
    constant MppMoEParams& params      [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_dim;
    const int I = (int)params.intermediate_dim;

    constexpr int BM = 32;
    constexpr int BN = 32;

    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    // Input:       [B, H] row-major → columns-first tensor
    auto tX  = tensor(input,       dextents<int, 2>{H, B}, array<int, 2>{1, H});
    // Weights:     [I, H] row-major → accessed as [H, I] transposed
    auto tGW = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW = tensor(up_weight,   dextents<int, 2>{H, I}, array<int, 2>{1, H});
    // Output:      [I, B] columns-first
    auto tOut = tensor(act_out,    dextents<int, 2>{I, B}, array<int, 2>{1, I});

    auto sliceX  = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);

    // MPP single-simdgroup matmul: 32×32 tile, dynamic K, A@B^T
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false, true, false
    );

    // Gate GEMM — result in registers
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> gate_op;
    auto rGate = gate_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();
    gate_op.run(sliceX, sliceGW, rGate);

    // Up GEMM — result in registers
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> up_op;
    auto rUp = up_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceUW), float>();
    up_op.run(sliceX, sliceUW, rUp);

    // Postfix SwiGLU fusion in register space
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> act_op;
    auto rAct = act_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), half>();

    for (int i = 0; i < rGate.get_capacity(); i++) {
        rAct[i] = half(silu_moe(rGate[i]) * rUp[i]);
    }

    // Single store — no threadgroup staging
    auto sliceOut = tOut.slice(tile_i, tile_b);
    rAct.store(sliceOut);
}

// =============================================================================
// MPP MoE Down Projection (fp16)
// =============================================================================
//
// Computes: expert_out[B, H] = act[B, I] @ down_weight^T
// Grid: [ceil(H/32), ceil(B/32), 1]  Threadgroup: [32, 1, 1]

kernel void mpp_fused_moe_down_f16(
    device const half*   act_in       [[buffer(0)]],   // [B, I] — SwiGLU output
    device const half*   down_weight  [[buffer(1)]],   // [H, I]
    device half*         out          [[buffer(2)]],   // [B, H]
    constant MppMoEParams& params     [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_dim;
    const int I = (int)params.intermediate_dim;

    constexpr int BM = 32;
    constexpr int BN = 32;

    const int tile_b = (int)(tgid.y * BM);
    const int tile_h = (int)(tgid.x * BN);
    if (tile_b >= B || tile_h >= H) return;

    // act_in: [B, I] row-major → [I, B] columns-first
    auto tA   = tensor(act_in,     dextents<int, 2>{I, B}, array<int, 2>{1, I});
    // down_weight: [H, I] row-major → accessed as [I, H] transposed
    auto tDW  = tensor(down_weight, dextents<int, 2>{I, H}, array<int, 2>{1, I});
    // out: [B, H] row-major → [H, B] columns-first
    auto tOut = tensor(out,         dextents<int, 2>{H, B}, array<int, 2>{1, H});

    auto sliceA  = tA.slice(0, tile_b);
    auto sliceDW = tDW.slice(0, tile_h);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false, true, false
    );

    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> down_op;
    auto rOut = down_op.template get_destination_cooperative_tensor<
        decltype(sliceA), decltype(sliceDW), half>();
    down_op.run(sliceA, sliceDW, rOut);

    auto sliceOut = tOut.slice(tile_h, tile_b);
    rOut.store(sliceOut);
}

// =============================================================================
// Weighted Residual Scatter (MoE combine)
// =============================================================================
//
// Adds expert contribution scaled by router weight to the output accumulator.
// out[token, :] += weight[token] * expert_out[local_idx, :]
//
// Element-wise — no GEMM. Single SIMD group per token, strided over hidden_dim.
// Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]

kernel void mpp_moe_weighted_scatter_f16(
    device const half*   expert_out   [[buffer(0)]],   // [num_tokens, H]
    device const float*  weights      [[buffer(1)]],   // [num_tokens]
    device half*         accum        [[buffer(2)]],   // [num_tokens, H] (read-modify-write)
    constant MppMoEScatterParams& p   [[buffer(3)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= p.num_tokens) return;

    const float w = weights[token_idx];
    const device half* src = expert_out + token_idx * p.hidden_dim;
    device half* dst       = accum      + token_idx * p.hidden_dim;

    for (uint d = lane; d < p.hidden_dim; d += 32u) {
        dst[d] = half(float(dst[d]) + w * float(src[d]));
    }
}
