//  fused_lora.metal
//  Fused LoRA kernels for training on Apple Silicon
//
//  Implements fused forward/backward passes for Low-Rank Adaptation:
//  Forward:  y = x @ W.T + scale * (x @ A.T) @ B.T
//  Backward: Gradients for A, B, and x computed efficiently
//
//  Key optimizations:
//  - Single kernel launch for the full LoRA operation
//  - Intermediate activations kept in registers
//  - Efficient gradient accumulation using threadgroup memory
//
//  References:
//  - LoRA: https://arxiv.org/abs/2106.09685
//  - Unsloth: https://github.com/unslothai/unsloth

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Configuration and Types
// =============================================================================

/// Kernel parameters for LoRA operations
struct FusedLoraParams {
    uint batch_size;      // Batch size (B * seq_len flattened)
    uint in_features;     // Input dimension (D_in)
    uint out_features;    // Output dimension (D_out)
    uint rank;            // LoRA rank (R)
    float scale;          // LoRA scaling factor (alpha / rank)
};

// Tile sizes optimized for Apple Silicon
// These balance register pressure with parallelism
constant uint TILE_M [[function_constant(0)]];   // Batch tile
constant uint TILE_N [[function_constant(1)]];   // Output tile
constant uint TILE_K [[function_constant(2)]];   // Input/reduction tile
constant uint SIMD_SIZE = 32;

// Max tile size for static memory allocation
constant uint MAX_TILE_M = 128;

// Maximum supported LoRA rank (was 64, increased to 256)
#define MAX_LORA_RANK 256

// =============================================================================
// Utility Functions
// =============================================================================

/// Warp-level sum reduction using SIMD shuffle
inline float simd_sum(float val, uint simd_lane_id) {
    #pragma unroll
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// =============================================================================
// Vectorized Dot Product Helpers for half precision
// =============================================================================
// Metal GPU hardware has 128-bit load units. Loading half4 (8 bytes) per cycle
// instead of half (2 bytes) provides 4x load throughput. Pairing two half4
// loads fills the 128-bit bus completely.
// =============================================================================

/// Vectorized dot product: half inputs, float accumulation, strided access
inline float dot_h4(
    device const half* a,
    device const half* b,
    uint size
) {
    float4 acc = float4(0.0f);
    uint s4 = size & ~3u;
    for (uint i = 0; i < s4; i += 4) {
        acc += float4(*(device const half4*)(a + i)) * float4(*(device const half4*)(b + i));
    }
    float result = acc.x + acc.y + acc.z + acc.w;
    for (uint i = s4; i < size; i++) {
        result += float(a[i]) * float(b[i]);
    }
    return result;
}

/// Vectorized dot product with SIMD stride: half inputs, float accumulation
inline float dot_h4_simd(
    device const half* a,
    device const half* b,
    uint size,
    uint lane_id,
    uint stride
) {
    float4 acc = float4(0.0f);
    uint s4 = size & ~3u;
    for (uint i = lane_id * 4; i < s4; i += stride * 4) {
        acc += float4(*(device const half4*)(a + i)) * float4(*(device const half4*)(b + i));
    }
    float result = acc.x + acc.y + acc.z + acc.w;
    // Handle elements not covered by vectorized stride
    for (uint i = s4 + lane_id; i < size; i += stride) {
        result += float(a[i]) * float(b[i]);
    }
    return result;
}

// =============================================================================
// Fused LoRA Forward Kernel
// =============================================================================

/// Computes y = x @ W.T + scale * (x @ A.T) @ B.T
///
/// This kernel fuses the base linear and LoRA computations into a single pass,
/// keeping the intermediate (x @ A.T) in threadgroup memory for the LoRA path.
///
/// Memory layout:
/// - x: [batch_size, in_features] - input tensor
/// - W: [out_features, in_features] - base weight (frozen)
/// - A: [rank, in_features] - LoRA down projection
/// - B: [out_features, rank] - LoRA up projection
/// - y: [batch_size, out_features] - output tensor
/// - xA: [batch_size, rank] - intermediate for backward
///
/// Thread organization:
/// - Grid: [ceil(batch/TILE_M), ceil(out_features/TILE_N), 1]
/// - Threadgroup: [TILE_N, TILE_M/SIMD_SIZE, 1]
kernel void fused_lora_forward(
    device const half* x [[buffer(0)]],           // [batch, in_features]
    device const half* W [[buffer(1)]],           // [out_features, in_features]
    device const half* A [[buffer(2)]],           // [rank, in_features]
    device const half* B [[buffer(3)]],           // [out_features, rank]
    device half* y [[buffer(4)]],                 // [batch, out_features]
    device half* xA [[buffer(5)]],                // [batch, rank] - intermediate for backward
    constant FusedLoraParams& params [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],  // host allocates tile_m * rank floats
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Tile position
    const uint batch_start = tgid.x * TILE_M;
    const uint out_start = tgid.y * TILE_N;

    // Thread assignment within tile
    const uint local_m = simd_group_id;
    const uint local_n = simd_lane_id;

    // Bounds check
    const uint batch_idx = batch_start + local_m;
    const uint out_idx = out_start + local_n;

    if (batch_idx >= params.batch_size || out_idx >= params.out_features) {
        return;
    }

    // Accumulators
    float acc_base = 0.0f;
    float acc_lora = 0.0f;

    // Dynamic threadgroup memory for intermediate xA values
    // Size is tile_m * rank floats, allocated by the host via setThreadgroupMemoryLength
    threadgroup float* xA_tile = scratch;

    // -------------------------------------------------------------------------
    // Phase 1: Compute x @ W.T (base linear) — vectorized with half4 loads
    // -------------------------------------------------------------------------
    {
        device const half* x_row = x + batch_idx * params.in_features;
        device const half* w_row = W + out_idx * params.in_features;
        float4 base_acc4 = float4(0.0f);
        uint k4 = params.in_features & ~3u;
        for (uint k = 0; k < k4; k += 4) {
            base_acc4 += float4(*(device const half4*)(x_row + k)) *
                         float4(*(device const half4*)(w_row + k));
        }
        acc_base = base_acc4.x + base_acc4.y + base_acc4.z + base_acc4.w;
        for (uint k = k4; k < params.in_features; k++) {
            acc_base += float(x_row[k]) * float(w_row[k]);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: Compute x @ A.T (LoRA down projection) — vectorized with half4
    // Only compute once per batch element, stored in threadgroup memory
    // -------------------------------------------------------------------------
    {
        device const half* x_row = x + batch_idx * params.in_features;
        for (uint r = 0; r < params.rank; r++) {
            device const half* a_row = A + r * params.in_features;
            float4 acc_a4 = float4(0.0f);
            uint k4 = params.in_features & ~3u;

            // SIMD-strided vectorized reduction
            for (uint k = simd_lane_id * 4; k < k4; k += SIMD_SIZE * 4) {
                acc_a4 += float4(*(device const half4*)(x_row + k)) *
                          float4(*(device const half4*)(a_row + k));
            }
            float acc_a = acc_a4.x + acc_a4.y + acc_a4.z + acc_a4.w;

            // Scalar remainder
            for (uint k = k4 + simd_lane_id; k < params.in_features; k += SIMD_SIZE) {
                acc_a += float(x_row[k]) * float(a_row[k]);
            }

            // SIMD reduction
            acc_a = simd_sum(acc_a, simd_lane_id);

            // Store result (first lane writes)
            if (simd_lane_id == 0) {
                xA_tile[local_m * params.rank + r] = acc_a;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Save xA to global memory for backward pass
    if (local_n < params.rank) {
        xA[batch_idx * params.rank + local_n] = half(xA_tile[local_m * params.rank + local_n]);
    }

    // -------------------------------------------------------------------------
    // Phase 3: Compute (x @ A.T) @ B.T (LoRA up projection)
    // -------------------------------------------------------------------------
    for (uint r = 0; r < params.rank; r++) {
        float xa_val = xA_tile[local_m * params.rank + r];
        float b_val = float(B[out_idx * params.rank + r]);
        acc_lora += xa_val * b_val;
    }

    // -------------------------------------------------------------------------
    // Phase 4: Combine and write output
    // -------------------------------------------------------------------------
    float result = acc_base + params.scale * acc_lora;
    y[batch_idx * params.out_features + out_idx] = half(result);
}

// =============================================================================
// Fused LoRA Backward Kernel - Compute dA and dB
// =============================================================================

/// Backward pass for LoRA parameters.
///
/// Computes:
/// - dA = scale * (x.T @ (dY @ B.T))  -> [rank, in_features]
/// - dB = scale * (xA.T @ dY)         -> [out_features, rank]
///
/// Memory layout:
/// - dY: [batch_size, out_features] - upstream gradient
/// - x: [batch_size, in_features] - input (saved from forward)
/// - xA: [batch_size, rank] - intermediate from forward
/// - B: [out_features, rank] - LoRA up projection
/// - dA: [rank, in_features] - gradient for A
/// - dB: [out_features, rank] - gradient for B
kernel void fused_lora_backward_ab(
    device const half* dY [[buffer(0)]],          // [batch, out_features]
    device const half* x [[buffer(1)]],           // [batch, in_features]
    device const half* xA [[buffer(2)]],          // [batch, rank]
    device const half* B [[buffer(3)]],           // [out_features, rank]
    device half* dA [[buffer(4)]],                // [rank, in_features]
    device half* dB [[buffer(5)]],                // [out_features, rank]
    constant FusedLoraParams& params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // Part A: Compute dB = scale * (xA.T @ dY)
    // dB[out, r] = scale * sum_b(xA[b, r] * dY[b, out])
    // -------------------------------------------------------------------------
    // Grid: [ceil(out_features/TILE_N), ceil(rank/TILE_K), 1]

    const uint out_idx = tgid.x * TILE_N + tid.x;
    const uint rank_idx = tgid.y * TILE_K + tid.y;

    if (out_idx < params.out_features && rank_idx < params.rank) {
        float acc = 0.0f;

        // Accumulate over batch — note: non-contiguous access pattern
        // (strided by rank/out_features), so vectorization applies to batch dim
        for (uint b = 0; b < params.batch_size; b++) {
            float xa_val = float(xA[b * params.rank + rank_idx]);
            float dy_val = float(dY[b * params.out_features + out_idx]);
            acc += xa_val * dy_val;
        }

        dB[out_idx * params.rank + rank_idx] = half(params.scale * acc);
    }
}

/// Backward pass for dA gradient.
///
/// Computes dA[r, k] = scale * sum_b( x[b, k] * (dY[b, :] @ B[:, r]) )
///
/// Optimizations:
/// - B rows (B[0..r-1, rank_idx]) are staged into threadgroup memory once so all
///   batch iterations read from L1 rather than device memory.
/// - dY rows are loaded with half4 vectorized loads (4x throughput).
/// - A SIMD group of 32 lanes cooperates: each lane takes a strided slice of the
///   out_features reduction, then simd_sum() merges the 32 partial sums.
/// - Only lane 0 writes the result.
///
/// Grid: [ceil(in_features/TILE_K), ceil(rank/TILE_M), 1]
/// Threadgroup: [SIMD_SIZE, TILE_M, 1]  (one SIMD group per rank row in the tile)
///
/// Host must allocate TILE_M * MAX_LORA_RANK * sizeof(float) bytes for scratch.
/// At TILE_M=4 and MAX_LORA_RANK=256: 4KB, well within the 32KB limit.
kernel void fused_lora_backward_a(
    device const half* dY [[buffer(0)]],          // [batch, out_features]
    device const half* x [[buffer(1)]],           // [batch, in_features]
    device const half* B [[buffer(2)]],           // [out_features, rank]
    device half* dA [[buffer(3)]],                // [rank, in_features]
    constant FusedLoraParams& params [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],  // TILE_M * out_features floats for B staging
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Grid: [ceil(in_features/TILE_K), ceil(rank/TILE_M), 1]
    const uint in_idx   = tgid.x * TILE_K + tid.x;
    const uint rank_idx = tgid.y * TILE_M + simd_group_id;

    const uint total_threads = TILE_M * SIMD_SIZE;
    const uint linear_tid    = simd_group_id * SIMD_SIZE + simd_lane_id;

    // -------------------------------------------------------------------------
    // Stage B column: B[:, rank_idx] into scratch[simd_group_id * out_features + o]
    // Each SIMD group stages the B column it will use for the reduction,
    // using vectorized half4 loads.
    // Threads with rank_idx out-of-range skip the write but still reach the barrier.
    // -------------------------------------------------------------------------
    // scratch layout: [TILE_M][out_features] — float
    threadgroup float* B_col = scratch + simd_group_id * params.out_features;

    if (rank_idx < params.rank) {
        uint s4 = params.out_features & ~3u;
        for (uint o = simd_lane_id * 4; o < s4; o += SIMD_SIZE * 4) {
            // B[o, rank_idx] = B[o * rank + rank_idx]: strided, can't half4-load directly.
            // Load 4 consecutive half scalars individually but unrolled.
            B_col[o + 0] = float(B[(o + 0) * params.rank + rank_idx]);
            B_col[o + 1] = float(B[(o + 1) * params.rank + rank_idx]);
            B_col[o + 2] = float(B[(o + 2) * params.rank + rank_idx]);
            B_col[o + 3] = float(B[(o + 3) * params.rank + rank_idx]);
        }
        for (uint o = s4 + simd_lane_id; o < params.out_features; o += SIMD_SIZE) {
            B_col[o] = float(B[o * params.rank + rank_idx]);
        }
    }

    // All threads must reach the barrier — guard the WRITE, not the barrier itself.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (in_idx >= params.in_features || rank_idx >= params.rank) {
        return;
    }

    // -------------------------------------------------------------------------
    // dA[rank_idx, in_idx] = scale * sum_b( x[b, in_idx] * dot(dY[b], B_col) )
    // Each lane takes a strided slice of out_features for the dot product.
    // -------------------------------------------------------------------------
    float acc = 0.0f;

    for (uint b = 0; b < params.batch_size; b++) {
        device const half* dY_row = dY + b * params.out_features;

        // Vectorized dot: dY_row * B_col, lane-strided with half4 loads on dY_row.
        float4 dot_acc4 = float4(0.0f);
        uint s4 = params.out_features & ~3u;
        for (uint o = simd_lane_id * 4; o < s4; o += SIMD_SIZE * 4) {
            float4 dy4 = float4(*(device const half4*)(dY_row + o));
            float4 b4  = float4(B_col[o], B_col[o+1], B_col[o+2], B_col[o+3]);
            dot_acc4  += dy4 * b4;
        }
        float dy_b = dot_acc4.x + dot_acc4.y + dot_acc4.z + dot_acc4.w;
        for (uint o = s4 + simd_lane_id; o < params.out_features; o += SIMD_SIZE) {
            dy_b += float(dY_row[o]) * B_col[o];
        }

        // SIMD reduction: all 32 lanes contribute their partial dot sum.
        dy_b = simd_sum(dy_b);

        // Only lane 0 has the full dot; multiply by x[b, in_idx] and accumulate.
        if (simd_lane_id == 0) {
            float x_val = float(x[b * params.in_features + in_idx]);
            acc += x_val * dy_b;
        }
    }

    // Write result (only lane 0 holds the accumulated value).
    if (simd_lane_id == 0) {
        dA[rank_idx * params.in_features + in_idx] = half(params.scale * acc);
    }
}

// =============================================================================
// Fused LoRA Backward Kernel - Compute dX
// =============================================================================

/// Backward pass for input gradient.
///
/// Computes dX = dY @ W + scale * (dY @ B) @ A
///
/// Memory layout:
/// - dY: [batch_size, out_features] - upstream gradient
/// - W: [out_features, in_features] - base weight
/// - A: [rank, in_features] - LoRA down projection
/// - B: [out_features, rank] - LoRA up projection
/// - dX: [batch_size, in_features] - input gradient
kernel void fused_lora_backward_x(
    device const half* dY [[buffer(0)]],          // [batch, out_features]
    device const half* W [[buffer(1)]],           // [out_features, in_features]
    device const half* A [[buffer(2)]],           // [rank, in_features]
    device const half* B [[buffer(3)]],           // [out_features, rank]
    device half* dX [[buffer(4)]],                // [batch, in_features]
    constant FusedLoraParams& params [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],  // host allocates tile_m * rank floats
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Grid: [ceil(batch/TILE_M), ceil(in_features/TILE_K), 1]

    const uint batch_idx = tgid.x * TILE_M + simd_group_id;
    const uint in_idx = tgid.y * TILE_K + simd_lane_id;

    if (batch_idx >= params.batch_size || in_idx >= params.in_features) {
        return;
    }

    // Dynamic threadgroup memory for intermediate dY @ B
    // Size is tile_m * rank floats, allocated by the host via setThreadgroupMemoryLength
    threadgroup float* dyB_tile = scratch;

    // -------------------------------------------------------------------------
    // Phase 1: Compute dY @ W (base gradient)
    // W is [out_features, in_features], accessed as W[o * in_features + in_idx]
    // which is strided, so we can't vectorize on the contiguous dim here.
    // -------------------------------------------------------------------------
    float acc_base = 0.0f;
    for (uint o = 0; o < params.out_features; o++) {
        float dy_val = float(dY[batch_idx * params.out_features + o]);
        float w_val = float(W[o * params.in_features + in_idx]);
        acc_base += dy_val * w_val;
    }

    // -------------------------------------------------------------------------
    // Phase 2: Compute dY @ B (LoRA intermediate)
    // B is [out_features, rank], accessed as B[o * rank + r] — strided.
    // -------------------------------------------------------------------------
    for (uint r = 0; r < params.rank; r++) {
        float acc = 0.0f;

        for (uint o = simd_lane_id; o < params.out_features; o += SIMD_SIZE) {
            float dy_val = float(dY[batch_idx * params.out_features + o]);
            float b_val = float(B[o * params.rank + r]);
            acc += dy_val * b_val;
        }

        // SIMD reduction
        acc = simd_sum(acc, simd_lane_id);

        if (simd_lane_id == 0) {
            dyB_tile[simd_group_id * params.rank + r] = acc;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 3: Compute (dY @ B) @ A (LoRA gradient)
    // -------------------------------------------------------------------------
    float acc_lora = 0.0f;
    for (uint r = 0; r < params.rank; r++) {
        float dyb_val = dyB_tile[simd_group_id * params.rank + r];
        float a_val = float(A[r * params.in_features + in_idx]);
        acc_lora += dyb_val * a_val;
    }

    // -------------------------------------------------------------------------
    // Phase 4: Combine and write output
    // -------------------------------------------------------------------------
    float result = acc_base + params.scale * acc_lora;
    dX[batch_idx * params.in_features + in_idx] = half(result);
}

// =============================================================================
// Optimized Single-Layer LoRA Forward (for inference)
// =============================================================================

/// Lightweight LoRA forward for inference (no xA saved).
/// Optimized for throughput when gradients are not needed.
kernel void lora_forward_inference(
    device const half* x [[buffer(0)]],           // [batch, in_features]
    device const half* W [[buffer(1)]],           // [out_features, in_features]
    device const half* A [[buffer(2)]],           // [rank, in_features]
    device const half* B [[buffer(3)]],           // [out_features, rank]
    device half* y [[buffer(4)]],                 // [batch, out_features]
    constant FusedLoraParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Simple grid: each thread handles one output element
    const uint batch_idx = tgid.x;
    const uint out_idx = tgid.y * 32 + tid;

    if (batch_idx >= params.batch_size || out_idx >= params.out_features) {
        return;
    }

    // Base linear — vectorized with half4 loads
    device const half* x_row = x + batch_idx * params.in_features;
    device const half* w_row = W + out_idx * params.in_features;
    float acc = dot_h4(x_row, w_row, params.in_features);

    // LoRA contribution — vectorized x @ A^T per rank
    float lora_acc = 0.0f;
    for (uint r = 0; r < params.rank; r++) {
        device const half* a_row = A + r * params.in_features;
        float xa = dot_h4(x_row, a_row, params.in_features);
        lora_acc += xa * float(B[out_idx * params.rank + r]);
    }

    y[batch_idx * params.out_features + out_idx] = half(acc + params.scale * lora_acc);
}

// =============================================================================
// Fused LoRA + RMSNorm Forward (future optimization)
// =============================================================================

// Placeholder for fused attention LoRA kernels (QKV projection)
// These would fuse Q, K, V LoRA projections into a single kernel pass

// Placeholder for fused MLP LoRA kernels (gate/up/down projections with SwiGLU)
// These would fuse all three projections similar to unsloth's LoRA_MLP
