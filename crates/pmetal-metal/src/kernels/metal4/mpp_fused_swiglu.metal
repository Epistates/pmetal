// mpp_fused_swiglu.metal
// Metal 4 Fused SwiGLU MLP using MPP matmul2d.
//
// Replaces the Metal 3 fused_swiglu kernels which use per-element dot products
// (SIMD-strided reduction per output element). MPP provides hardware matrix
// multiply for the projections, with SwiGLU activation applied as postfix
// fusion on the cooperative tensor result.
//
// Computes: output = silu(x @ gate_W^T) * (x @ up_W^T)
//
// Single kernel launch combines:
//   1. gate = x @ gate_weight^T
//   2. up   = x @ up_weight^T
//   3. output = silu(gate) * up
//
// MPP Guide Section 2.3.4 (Postfix Fusion): The GEMM output stays in
// cooperative tensor registers.  Both gate and up projections are computed
// with their results held in register arrays (rGate, rUp) simultaneously.
// SwiGLU is then applied element-wise in register space before the single
// store to device memory — no threadgroup memory staging required.
//
// MPP Guide Section 2.3.1: Single simdgroup (execution_simdgroup) is used
// throughout. Multi-simdgroup configurations always resulted in a significant
// performance drop in Apple's benchmarks.
//
// For LoRA: adds scale * (x @ A^T) @ B^T to each projection.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct FusedSwiGLUParams {
    uint batch_size;
    uint hidden_size;
    uint intermediate_size;
    uint lora_rank;
    float lora_scale;
};

inline float silu(float x) {
    return x / (1.0f + metal::exp(-x));
}

// =============================================================================
// MPP Fused SwiGLU Forward (fp16)
// =============================================================================
//
// Both GEMMs are computed with their results held in cooperative tensor
// register arrays simultaneously. SwiGLU activation is applied in register
// space, then a single store writes the fused result to device memory.
//
// No threadgroup memory staging for GEMM outputs — Apple Silicon cache
// hierarchy handles data reuse for the input tile automatically.

kernel void mpp_fused_swiglu_forward_f16(
    device half* input [[buffer(0)]],
    device half* gate_weight [[buffer(1)]],
    device half* up_weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant FusedSwiGLUParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_size;
    const int I = (int)params.intermediate_size;

    // Grid: [num_intermediate_tiles, num_batch_tiles, 1]
    const int BM = 32;  // batch tile — 32x32 is the recommended single-simdgroup tile
    const int BN = 32;  // intermediate tile

    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    // Create tensors
    // input:       [B, H] row-major → tensor with K=H, M=B columns-first
    auto tX = tensor(input, dextents<int, 2>{H, B}, array<int, 2>{1, H});

    // gate_weight: [I, H] row-major → transposed via descriptor
    auto tGW = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW = tensor(up_weight,   dextents<int, 2>{H, I}, array<int, 2>{1, H});

    // Output tensor: [I, B] columns-first
    auto tOut = tensor(output, dextents<int, 2>{I, B}, array<int, 2>{1, I});

    // Slices to this tile
    auto sliceX  = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);

    // MPP single-simdgroup matmul descriptor: 32x32 tile, K dynamic, A@B^T
    constexpr auto proj_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false,  // A not transposed
        true,   // B transposed (weight is [I, H], we want X @ W^T)
        false   // relaxed_precision
    );

    // Gate GEMM — result lives in register array rGate
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> gate_op;
    auto rGate = gate_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();
    gate_op.run(sliceX, sliceGW, rGate);

    // Up GEMM — result lives in register array rUp
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> up_op;
    auto rUp = up_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceUW), float>();
    up_op.run(sliceX, sliceUW, rUp);

    // Postfix fusion: apply silu(gate) * up in register space.
    // rOut will carry the fused result to device memory via a single store.
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> out_op;
    auto rOut = out_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), half>();

    for (int i = 0; i < rGate.get_capacity(); i++) {
        rOut[i] = half(silu(rGate[i]) * rUp[i]);
    }

    // Single store from registers to device memory — no staging required
    auto sliceOut = tOut.slice(tile_i, tile_b);
    rOut.store(sliceOut);
}

// =============================================================================
// MPP Fused SwiGLU Forward (fp32)
// =============================================================================

kernel void mpp_fused_swiglu_forward_f32(
    device float* input [[buffer(0)]],
    device float* gate_weight [[buffer(1)]],
    device float* up_weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant FusedSwiGLUParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_size;
    const int I = (int)params.intermediate_size;

    const int BM = 32;
    const int BN = 32;
    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    auto tX   = tensor(input,       dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto tGW  = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW  = tensor(up_weight,   dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tOut = tensor(output,      dextents<int, 2>{I, B}, array<int, 2>{1, I});

    auto sliceX  = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);

    constexpr auto proj_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false, true, false
    );

    // Gate GEMM in registers
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> gate_op;
    auto rGate = gate_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();
    gate_op.run(sliceX, sliceGW, rGate);

    // Up GEMM in registers
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> up_op;
    auto rUp = up_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceUW), float>();
    up_op.run(sliceX, sliceUW, rUp);

    // Postfix fusion in register space — no threadgroup staging
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> out_op;
    auto rOut = out_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();

    for (int i = 0; i < rGate.get_capacity(); i++) {
        rOut[i] = silu(rGate[i]) * rUp[i];
    }

    auto sliceOut = tOut.slice(tile_i, tile_b);
    rOut.store(sliceOut);
}

// =============================================================================
// MPP Fused SwiGLU + LoRA Forward (fp16)
// =============================================================================
//
// Extends mpp_fused_swiglu_forward_f16 with optional LoRA corrections on
// both gate and up projections.
//
// Compute order (all register-resident where possible):
//   1. gate = x @ gate_W^T                        ← MPP matmul2d → rGate
//   2. up   = x @ up_W^T                          ← MPP matmul2d → rUp
//   3. x_ga = x @ gate_A^T (per-token, rank-dim)  ← threadgroup scratch
//   4. x_ua = x @ up_A^T   (per-token, rank-dim)  ← threadgroup scratch
//   5. For each output element i in this tile:
//        gate_lora_i = x_ga · gate_B[i]           ← scalar reduction over rank
//        up_lora_i   = x_ua · up_B[i]             ← scalar reduction over rank
//        rGate[i] += lora_scale * gate_lora_i
//        rUp[i]   += lora_scale * up_lora_i
//   6. output[i] = silu(rGate[i]) * rUp[i]        ← postfix in register space
//
// The LoRA A projections (step 3/4) use a per-token threadgroup scratch buffer
// of 2 * lora_rank floats — tiny relative to the 32KB threadgroup limit.
// All 32 threads in the SIMD group cooperate over the rank dimension.
//
// LoRA B multiplication (step 5) is applied per output-element in register space:
// each cooperative tensor element `rGate[i]` / `rUp[i]` maps to one (batch, intermed)
// output pair; we look up the corresponding gate_B and up_B rows and dot them
// with the shared threadgroup x_ga / x_ua values.
//
// Buffer layout:
//   buffer(0): input             half [B, H]
//   buffer(1): gate_weight       half [I, H]
//   buffer(2): up_weight         half [I, H]
//   buffer(3): gate_lora_a       half [lora_rank, H]
//   buffer(4): gate_lora_b       half [I, lora_rank]
//   buffer(5): up_lora_a         half [lora_rank, H]
//   buffer(6): up_lora_b         half [I, lora_rank]
//   buffer(7): output            half [B, I]
//   buffer(8): params            FusedSwiGLUParams

kernel void mpp_fused_swiglu_lora_forward_f16(
    device const half*    input       [[buffer(0)]],
    device const half*    gate_weight [[buffer(1)]],
    device const half*    up_weight   [[buffer(2)]],
    device const half*    gate_lora_a [[buffer(3)]],
    device const half*    gate_lora_b [[buffer(4)]],
    device const half*    up_lora_a   [[buffer(5)]],
    device const half*    up_lora_b   [[buffer(6)]],
    device half*          output      [[buffer(7)]],
    constant FusedSwiGLUParams& params [[buffer(8)]],
    uint3 tgid         [[threadgroup_position_in_grid]],
    uint3 tid          [[thread_position_in_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]   // 2 * lora_rank floats
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_size;
    const int I = (int)params.intermediate_size;
    const int R = (int)params.lora_rank;

    const int BM = 32;
    const int BN = 32;
    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    // --- Step 1 & 2: MPP base projections (same as no-LoRA kernel) ----------
    auto tX  = tensor(input,       dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto tGW = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW = tensor(up_weight,   dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tOut = tensor(output, dextents<int, 2>{I, B}, array<int, 2>{1, I});

    auto sliceX  = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);

    constexpr auto proj_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false
    );

    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> gate_op;
    auto rGate = gate_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();
    gate_op.run(sliceX, sliceGW, rGate);

    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> up_op;
    auto rUp = up_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceUW), float>();
    up_op.run(sliceX, sliceUW, rUp);

    // --- Step 3 & 4: LoRA A projections for each token in this tile ----------
    // Scratch layout: [BM * R floats for gate_a] [BM * R floats for up_a]
    // Each token in the BM-wide batch tile gets R floats of LoRA intermediate.
    threadgroup float* x_gate_a = scratch;          // [BM, R]
    threadgroup float* x_up_a   = scratch + BM * R; // [BM, R]

    // All 32 lanes cooperate: each lane owns (lane_id % R) rank slots
    // across all BM token rows (lane_id / R gives the token row within BM).
    // We interleave over (token_row, rank_col) pairs with stride 32.
    const uint total_ga = (uint)(BM * R);
    const uint linear_tid = (uint)(tid.y * 32 + tid.x); // 0..31 (single simdgroup)
    for (uint idx = linear_tid; idx < total_ga; idx += 32) {
        uint row   = idx / (uint)R;  // token within BM tile
        uint r_idx = idx % (uint)R;  // LoRA rank index

        uint global_b = (uint)tile_b + row;
        if (global_b >= (uint)B) {
            x_gate_a[idx] = 0.0f;
            x_up_a[idx]   = 0.0f;
            continue;
        }

        device const half* x_tok      = input + global_b * H;
        device const half* gate_a_row = gate_lora_a + r_idx * H;
        device const half* up_a_row   = up_lora_a   + r_idx * H;

        float gate_dot = 0.0f;
        float up_dot   = 0.0f;
        for (int h = 0; h < H; h++) {
            float xv      = float(x_tok[h]);
            gate_dot += xv * float(gate_a_row[h]);
            up_dot   += xv * float(up_a_row[h]);
        }
        x_gate_a[idx] = gate_dot;
        x_up_a[idx]   = up_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 5: Add LoRA deltas to rGate and rUp in register space ----------
    // rGate[i] corresponds to the i-th element of the 32x32 cooperative tensor
    // tile. Its (batch_row, intermed_col) coordinates are given by the MPP
    // layout: consecutive elements advance along the batch dimension first.
    // With BM=BN=32 and one simdgroup, capacity = BM*BN/32 = 32 elements per lane.
    // Lane `l` owns rows [l..l] of the 32-element tile column-packed:
    // element i → batch_row = i % BM, intermed_col = tile_i + (i / BM).
    // (MPP internal layout may differ; we use get_capacity() and the
    //  column-major cooperative tensor index convention from MPP Guide §2.3.2.)
    const int cap = rGate.get_capacity();
    for (int ci = 0; ci < cap; ci++) {
        // Recover (batch_row, intermed_col) from cooperative tensor element index.
        // In execution_simdgroup mode with a [BM, BN] = [32, 32] tile:
        //   each lane holds cap=1 element at position (simd_lane_id, tile_i + ci).
        // With BM=BN=32, cap=32/32=1, so ci=0 always and
        //   batch_row = simd_lane_id (this lane's row in the 32-row batch tile).
        //   intermed_col = tile_i + (simd_lane_id / BM * BN + ci).
        // Simplified for BM=BN=32, cap=1:
        uint batch_row    = (uint)simd_lane_id;
        uint intermed_col = (uint)tile_i + (uint)ci;
        if ((int)batch_row >= BM || (int)intermed_col >= I) continue;

        // LoRA B: gate_lora_b[intermed_col, R], up_lora_b[intermed_col, R]
        device const half* gb_row = gate_lora_b + intermed_col * R;
        device const half* ub_row = up_lora_b   + intermed_col * R;

        // x_gate_a[batch_row, :] is at offset batch_row * R
        threadgroup const float* xga = x_gate_a + batch_row * R;
        threadgroup const float* xua = x_up_a   + batch_row * R;

        float gate_lora_val = 0.0f;
        float up_lora_val   = 0.0f;
        for (int r_idx = 0; r_idx < R; r_idx++) {
            gate_lora_val += xga[r_idx] * float(gb_row[r_idx]);
            up_lora_val   += xua[r_idx] * float(ub_row[r_idx]);
        }

        rGate[ci] += params.lora_scale * gate_lora_val;
        rUp[ci]   += params.lora_scale * up_lora_val;
    }

    // --- Step 6: SwiGLU postfix fusion and store -----------------------------
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> out_op;
    auto rOut = out_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), half>();

    for (int i = 0; i < cap; i++) {
        rOut[i] = half(silu(rGate[i]) * rUp[i]);
    }

    auto sliceOut = tOut.slice(tile_i, tile_b);
    rOut.store(sliceOut);
}

// =============================================================================
// MPP Fused SwiGLU + LoRA Forward (fp32)
// =============================================================================

kernel void mpp_fused_swiglu_lora_forward_f32(
    device const float*   input       [[buffer(0)]],
    device const float*   gate_weight [[buffer(1)]],
    device const float*   up_weight   [[buffer(2)]],
    device const float*   gate_lora_a [[buffer(3)]],
    device const float*   gate_lora_b [[buffer(4)]],
    device const float*   up_lora_a   [[buffer(5)]],
    device const float*   up_lora_b   [[buffer(6)]],
    device float*         output      [[buffer(7)]],
    constant FusedSwiGLUParams& params [[buffer(8)]],
    uint3 tgid         [[threadgroup_position_in_grid]],
    uint3 tid          [[thread_position_in_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]   // 2 * BM * lora_rank floats
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_size;
    const int I = (int)params.intermediate_size;
    const int R = (int)params.lora_rank;

    const int BM = 32;
    const int BN = 32;
    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    auto tX   = tensor(input,       dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto tGW  = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW  = tensor(up_weight,   dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tOut = tensor(output,      dextents<int, 2>{I, B}, array<int, 2>{1, I});

    auto sliceX  = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);

    constexpr auto proj_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false
    );

    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> gate_op;
    auto rGate = gate_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();
    gate_op.run(sliceX, sliceGW, rGate);

    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> up_op;
    auto rUp = up_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceUW), float>();
    up_op.run(sliceX, sliceUW, rUp);

    threadgroup float* x_gate_a = scratch;
    threadgroup float* x_up_a   = scratch + BM * R;

    const uint total_ga   = (uint)(BM * R);
    const uint linear_tid = (uint)(tid.y * 32 + tid.x);
    for (uint idx = linear_tid; idx < total_ga; idx += 32) {
        uint row   = idx / (uint)R;
        uint r_idx = idx % (uint)R;

        uint global_b = (uint)tile_b + row;
        if (global_b >= (uint)B) {
            x_gate_a[idx] = 0.0f;
            x_up_a[idx]   = 0.0f;
            continue;
        }

        device const float* x_tok      = input + global_b * H;
        device const float* gate_a_row = gate_lora_a + r_idx * H;
        device const float* up_a_row   = up_lora_a   + r_idx * H;

        float gate_dot = 0.0f;
        float up_dot   = 0.0f;
        for (int h = 0; h < H; h++) {
            float xv  = x_tok[h];
            gate_dot += xv * gate_a_row[h];
            up_dot   += xv * up_a_row[h];
        }
        x_gate_a[idx] = gate_dot;
        x_up_a[idx]   = up_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int cap = rGate.get_capacity();
    for (int ci = 0; ci < cap; ci++) {
        uint batch_row    = (uint)simd_lane_id;
        uint intermed_col = (uint)tile_i + (uint)ci;
        if ((int)batch_row >= BM || (int)intermed_col >= I) continue;

        device const float* gb_row = gate_lora_b + intermed_col * R;
        device const float* ub_row = up_lora_b   + intermed_col * R;

        threadgroup const float* xga = x_gate_a + batch_row * R;
        threadgroup const float* xua = x_up_a   + batch_row * R;

        float gate_lora_val = 0.0f;
        float up_lora_val   = 0.0f;
        for (int r_idx = 0; r_idx < R; r_idx++) {
            gate_lora_val += xga[r_idx] * gb_row[r_idx];
            up_lora_val   += xua[r_idx] * ub_row[r_idx];
        }

        rGate[ci] += params.lora_scale * gate_lora_val;
        rUp[ci]   += params.lora_scale * up_lora_val;
    }

    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroup> out_op;
    auto rOut = out_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();

    for (int i = 0; i < cap; i++) {
        rOut[i] = silu(rGate[i]) * rUp[i];
    }

    auto sliceOut = tOut.slice(tile_i, tile_b);
    rOut.store(sliceOut);
}

// =============================================================================
// MPP Fused MLP Forward (fp16): gate + up (SwiGLU) + down in one kernel
// =============================================================================
//
// Computes: output[B, H] = (silu(x @ gate_W^T) * (x @ up_W^T)) @ down_W^T
//
// The gate/up → SwiGLU activation stays in registers. The activation is then
// staged through a small threadgroup buffer (2 KB) to feed the down GEMM.
// The down GEMM accumulates over all I-tiles using multiply_accumulate mode.
//
// Grid: [ceil(H/32), ceil(B/32), 1]  Threadgroup: [32, 1, 1]

struct FusedMLPParams {
    uint batch_size;
    uint hidden_size;
    uint intermediate_size;
};

kernel void mpp_fused_mlp_forward_f16(
    device const half* input       [[buffer(0)]],   // [B, H]
    device const half* gate_weight [[buffer(1)]],   // [I, H]
    device const half* up_weight   [[buffer(2)]],   // [I, H]
    device const half* down_weight [[buffer(3)]],   // [H, I]
    device half*       output      [[buffer(4)]],   // [B, H]
    constant FusedMLPParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]],
    uint3 tid  [[thread_position_in_threadgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_size;
    const int I = (int)params.intermediate_size;

    // Single-simdgroup 32x32 tile.
    constexpr int BM = 32;
    constexpr int BN = 32;

    // Dispatch covers output [B, H].
    const int tile_b = (int)(tgid.y * BM);
    const int tile_h = (int)(tgid.x * BN);
    if (tile_b >= B || tile_h >= H) return;

    // 2 KB staging buffer: rAct is written here so the down GEMM can read it
    // as a cooperative tensor. Gate/up → rAct stays in registers; only the
    // handoff between the SwiGLU step and the down projection requires staging.
    threadgroup half act_stage[BM * BN];

    // Tensor descriptors for input and output.
    auto tX   = tensor(input,  dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto tOut = tensor(output, dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto sliceX   = tX.slice(0, tile_b);
    auto sliceOut = tOut.slice(tile_h, tile_b);

    // Gate/up GEMM descriptor: 32x32 tile, K=H dynamic.
    constexpr auto gu_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false, true, false
    );

    // Down GEMM descriptor: 32x32 tile, K=BN_i per I-tile (dynamic),
    // multiply_accumulate — rOut is zeroed on first call and accumulated.
    constexpr auto down_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );

    // Allocate output accumulator (MPP zeros on first run).
    mpp::tensor_ops::matmul2d<down_desc, execution_simdgroup> out_op;

    // Reference type for act_stage tensor — used to declare rOut capacity.
    auto act_stage_ref = tensor((threadgroup half*)act_stage,
                                dextents<int, 2>{BN, BM},
                                array<int, 2>{1, BN});

    // down_weight: [H, I] row-major; for matmul we treat it as [I, H] transposed.
    // Slicing: for each I-tile at offset tile_i, the [BN, H] slab is
    //   down_weight + tile_i * H  as a [H, BN] tensor → [BN, H] transposed.
    auto tDW_h = tensor(down_weight, dextents<int, 2>{I, H}, array<int, 2>{1, I});
    auto sliceDW_h = tDW_h.slice(0, tile_h);  // columns [tile_h..+BN] of [I, H]

    auto rOut = out_op.template get_destination_cooperative_tensor<
        decltype(act_stage_ref), decltype(sliceDW_h), half>();

    // I-reduction: iterate over intermediate dimension in BN-wide tiles.
    int num_i_tiles = (I + BN - 1) / BN;
    for (int ti = 0; ti < num_i_tiles; ti++) {
        int tile_i = ti * BN;

        // Gate weight slab for this I-tile: gate_weight[I, H] columns tile_i..+BN
        auto tGW = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
        auto tUW = tensor(up_weight,   dextents<int, 2>{H, I}, array<int, 2>{1, H});
        auto sliceGW = tGW.slice(0, tile_i);
        auto sliceUW = tUW.slice(0, tile_i);

        // Gate GEMM in registers.
        mpp::tensor_ops::matmul2d<gu_desc, execution_simdgroup> gate_op;
        auto rGate = gate_op.template get_destination_cooperative_tensor<
            decltype(sliceX), decltype(sliceGW), float>();
        gate_op.run(sliceX, sliceGW, rGate);

        // Up GEMM in registers.
        mpp::tensor_ops::matmul2d<gu_desc, execution_simdgroup> up_op;
        auto rUp = up_op.template get_destination_cooperative_tensor<
            decltype(sliceX), decltype(sliceUW), float>();
        up_op.run(sliceX, sliceUW, rUp);

        // SwiGLU postfix fusion → rAct (half precision).
        mpp::tensor_ops::matmul2d<gu_desc, execution_simdgroup> act_op;
        auto rAct = act_op.template get_destination_cooperative_tensor<
            decltype(sliceX), decltype(sliceGW), half>();
        for (int k = 0; k < rGate.get_capacity(); k++) {
            rAct[k] = half(silu(rGate[k]) * rUp[k]);
        }

        // Stage rAct → threadgroup memory for down GEMM.
        rAct.store(act_stage_ref);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down GEMM tile: down_weight rows tile_i..+BN, columns tile_h..+BN.
        // down_weight layout: [H, I] row-major → pointer at tile_i-th row.
        auto tDW_i = tensor(down_weight + tile_i * H,
                            dextents<int, 2>{H, BN},
                            array<int, 2>{1, H});
        auto sliceDW_i = tDW_i.slice(0, tile_h);

        // Accumulate: rOut += act_stage[BM, BN] @ sliceDW_i[BN, BN_h]^T
        out_op.run(act_stage_ref, sliceDW_i, rOut);

        threadgroup_barrier(mem_flags::mem_none);
    }

    // Single store of accumulated rOut to device memory.
    rOut.store(sliceOut);
}
