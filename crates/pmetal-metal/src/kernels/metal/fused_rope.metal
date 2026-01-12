// =============================================================================
// FUSED RoPE (ROTARY POSITION EMBEDDING) KERNEL
// =============================================================================
//
// Provides high-performance rotary position embeddings with:
// - In-place rotation (modifies input directly)
// - Custom position ID support (essential for sequence packing)
// - Fused sin/cos computation
// - Optional QK fusion (apply RoPE to both Q and K in single pass)
//
// Benefits:
// - No intermediate tensor allocations
// - Single kernel launch for QK RoPE
// - Optimized for Apple Silicon
//
// =============================================================================

#include <metal_stdlib>
using namespace metal;

#define SIMD_SIZE 32
#define THREADS_PER_HEAD 64

/// Parameters for RoPE kernel
struct RoPEParams {
    uint batch_size;       // Number of batches
    uint num_heads;        // Number of attention heads
    uint seq_len;          // Sequence length
    uint head_dim;         // Head dimension (must be even)
    float base;            // Base frequency (default 10000)
    float scale;           // Position scale factor
};

/// Compute inverse frequency for a dimension index.
inline float inv_freq(uint dim_idx, uint dims, float base) {
    // inv_freq = 1 / (base^(2i/dims))
    return 1.0f / pow(base, float(2 * dim_idx) / float(dims));
}

// =============================================================================
// IN-PLACE RoPE WITH SEQUENTIAL POSITIONS
// =============================================================================

/// Apply RoPE in-place with sequential positions [0, 1, 2, ...].
///
/// Input shape: [batch, heads, seq_len, head_dim]
/// The head_dim is split in half for rotation.
kernel void rope_inplace(
    device float* x [[buffer(0)]],              // [batch, heads, seq_len, head_dim]
    constant RoPEParams& params [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx = tgid.y;
    const uint seq_idx = tgid.z;

    if (batch_idx >= params.batch_size ||
        head_idx >= params.num_heads ||
        seq_idx >= params.seq_len) {
        return;
    }

    // Compute position (scaled)
    const float pos = float(seq_idx) * params.scale;
    const uint half_dim = params.head_dim / 2;

    // Pointer to this token's head embedding
    const uint offset = ((batch_idx * params.num_heads + head_idx) * params.seq_len + seq_idx) * params.head_dim;
    device float* x_ptr = x + offset;

    // Each thread handles one or more dimension pairs
    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        // Compute angle: pos * inv_freq(d)
        const float freq = inv_freq(d, params.head_dim, params.base);
        const float angle = pos * freq;
        const float cos_val = cos(angle);
        const float sin_val = sin(angle);

        // Load dimension pair (non-traditional: first half, second half)
        const float x1 = x_ptr[d];
        const float x2 = x_ptr[d + half_dim];

        // Apply rotation:
        // x1' = x1 * cos - x2 * sin
        // x2' = x1 * sin + x2 * cos
        x_ptr[d] = x1 * cos_val - x2 * sin_val;
        x_ptr[d + half_dim] = x1 * sin_val + x2 * cos_val;
    }
}

/// Half-precision version.
kernel void rope_inplace_f16(
    device half* x [[buffer(0)]],
    constant RoPEParams& params [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx = tgid.y;
    const uint seq_idx = tgid.z;

    if (batch_idx >= params.batch_size ||
        head_idx >= params.num_heads ||
        seq_idx >= params.seq_len) {
        return;
    }

    const float pos = float(seq_idx) * params.scale;
    const uint half_dim = params.head_dim / 2;
    const uint offset = ((batch_idx * params.num_heads + head_idx) * params.seq_len + seq_idx) * params.head_dim;
    device half* x_ptr = x + offset;

    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        const float freq = inv_freq(d, params.head_dim, params.base);
        const float angle = pos * freq;
        const float cos_val = cos(angle);
        const float sin_val = sin(angle);

        // Use fp32 for computation
        const float x1 = float(x_ptr[d]);
        const float x2 = float(x_ptr[d + half_dim]);

        x_ptr[d] = half(x1 * cos_val - x2 * sin_val);
        x_ptr[d + half_dim] = half(x1 * sin_val + x2 * cos_val);
    }
}

// =============================================================================
// RoPE WITH CUSTOM POSITION IDS (FOR SEQUENCE PACKING)
// =============================================================================

/// Apply RoPE with explicit position IDs.
///
/// Essential for sequence packing where multiple sequences are concatenated
/// and position IDs reset for each sequence.
///
/// Input:
/// - x: [batch, heads, seq_len, head_dim]
/// - position_ids: [seq_len] - custom position indices
kernel void rope_with_positions(
    device float* x [[buffer(0)]],              // [batch, heads, seq_len, head_dim]
    device const int* position_ids [[buffer(1)]],  // [seq_len]
    constant RoPEParams& params [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx = tgid.y;
    const uint seq_idx = tgid.z;

    if (batch_idx >= params.batch_size ||
        head_idx >= params.num_heads ||
        seq_idx >= params.seq_len) {
        return;
    }

    // Get position from position_ids (handles sequence packing)
    const float pos = float(position_ids[seq_idx]) * params.scale;
    const uint half_dim = params.head_dim / 2;
    const uint offset = ((batch_idx * params.num_heads + head_idx) * params.seq_len + seq_idx) * params.head_dim;
    device float* x_ptr = x + offset;

    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        const float freq = inv_freq(d, params.head_dim, params.base);
        const float angle = pos * freq;
        const float cos_val = cos(angle);
        const float sin_val = sin(angle);

        const float x1 = x_ptr[d];
        const float x2 = x_ptr[d + half_dim];

        x_ptr[d] = x1 * cos_val - x2 * sin_val;
        x_ptr[d + half_dim] = x1 * sin_val + x2 * cos_val;
    }
}

/// Half-precision version with position IDs.
kernel void rope_with_positions_f16(
    device half* x [[buffer(0)]],
    device const int* position_ids [[buffer(1)]],
    constant RoPEParams& params [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx = tgid.y;
    const uint seq_idx = tgid.z;

    if (batch_idx >= params.batch_size ||
        head_idx >= params.num_heads ||
        seq_idx >= params.seq_len) {
        return;
    }

    const float pos = float(position_ids[seq_idx]) * params.scale;
    const uint half_dim = params.head_dim / 2;
    const uint offset = ((batch_idx * params.num_heads + head_idx) * params.seq_len + seq_idx) * params.head_dim;
    device half* x_ptr = x + offset;

    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        const float freq = inv_freq(d, params.head_dim, params.base);
        const float angle = pos * freq;
        const float cos_val = cos(angle);
        const float sin_val = sin(angle);

        const float x1 = float(x_ptr[d]);
        const float x2 = float(x_ptr[d + half_dim]);

        x_ptr[d] = half(x1 * cos_val - x2 * sin_val);
        x_ptr[d + half_dim] = half(x1 * sin_val + x2 * cos_val);
    }
}

// =============================================================================
// FUSED QK RoPE (APPLY TO BOTH Q AND K IN SINGLE KERNEL)
// =============================================================================

/// Apply RoPE to both Q and K tensors in a single kernel launch.
///
/// This is more efficient than launching two separate kernels as it:
/// 1. Computes sin/cos once per position
/// 2. Amortizes kernel launch overhead
///
/// Inputs:
/// - q: [batch, q_heads, seq_len, head_dim]
/// - k: [batch, kv_heads, seq_len, head_dim]
kernel void rope_qk_inplace(
    device float* q [[buffer(0)]],              // [batch, q_heads, seq_len, head_dim]
    device float* k [[buffer(1)]],              // [batch, kv_heads, seq_len, head_dim]
    constant RoPEParams& params [[buffer(2)]],
    constant uint& kv_heads [[buffer(3)]],      // Number of KV heads
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint seq_idx = tgid.y;

    if (batch_idx >= params.batch_size || seq_idx >= params.seq_len) {
        return;
    }

    const float pos = float(seq_idx) * params.scale;
    const uint half_dim = params.head_dim / 2;

    // Precompute sin/cos for this position (shared across heads)
    // Use threadgroup memory to share across threads
    threadgroup float cos_cache[128];  // Max head_dim/2 = 128
    threadgroup float sin_cache[128];

    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        const float freq = inv_freq(d, params.head_dim, params.base);
        const float angle = pos * freq;
        cos_cache[d] = cos(angle);
        sin_cache[d] = sin(angle);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply to all Q heads
    for (uint head = 0; head < params.num_heads; head++) {
        const uint q_offset = ((batch_idx * params.num_heads + head) * params.seq_len + seq_idx) * params.head_dim;
        device float* q_ptr = q + q_offset;

        for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
            const float x1 = q_ptr[d];
            const float x2 = q_ptr[d + half_dim];

            q_ptr[d] = x1 * cos_cache[d] - x2 * sin_cache[d];
            q_ptr[d + half_dim] = x1 * sin_cache[d] + x2 * cos_cache[d];
        }
    }

    // Apply to all K heads (may be fewer due to GQA)
    for (uint head = 0; head < kv_heads; head++) {
        const uint k_offset = ((batch_idx * kv_heads + head) * params.seq_len + seq_idx) * params.head_dim;
        device float* k_ptr = k + k_offset;

        for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
            const float x1 = k_ptr[d];
            const float x2 = k_ptr[d + half_dim];

            k_ptr[d] = x1 * cos_cache[d] - x2 * sin_cache[d];
            k_ptr[d + half_dim] = x1 * sin_cache[d] + x2 * cos_cache[d];
        }
    }
}

/// Fused QK RoPE with custom position IDs.
kernel void rope_qk_with_positions(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const int* position_ids [[buffer(2)]],
    constant RoPEParams& params [[buffer(3)]],
    constant uint& kv_heads [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint seq_idx = tgid.y;

    if (batch_idx >= params.batch_size || seq_idx >= params.seq_len) {
        return;
    }

    const float pos = float(position_ids[seq_idx]) * params.scale;
    const uint half_dim = params.head_dim / 2;

    threadgroup float cos_cache[128];
    threadgroup float sin_cache[128];

    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        const float freq = inv_freq(d, params.head_dim, params.base);
        const float angle = pos * freq;
        cos_cache[d] = cos(angle);
        sin_cache[d] = sin(angle);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < params.num_heads; head++) {
        const uint q_offset = ((batch_idx * params.num_heads + head) * params.seq_len + seq_idx) * params.head_dim;
        device float* q_ptr = q + q_offset;

        for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
            const float x1 = q_ptr[d];
            const float x2 = q_ptr[d + half_dim];

            q_ptr[d] = x1 * cos_cache[d] - x2 * sin_cache[d];
            q_ptr[d + half_dim] = x1 * sin_cache[d] + x2 * cos_cache[d];
        }
    }

    for (uint head = 0; head < kv_heads; head++) {
        const uint k_offset = ((batch_idx * kv_heads + head) * params.seq_len + seq_idx) * params.head_dim;
        device float* k_ptr = k + k_offset;

        for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
            const float x1 = k_ptr[d];
            const float x2 = k_ptr[d + half_dim];

            k_ptr[d] = x1 * cos_cache[d] - x2 * sin_cache[d];
            k_ptr[d + half_dim] = x1 * sin_cache[d] + x2 * cos_cache[d];
        }
    }
}

// =============================================================================
// PRECOMPUTED COS/SIN CACHE (FOR BATCHED INFERENCE)
// =============================================================================

/// Precompute cos/sin cache for a range of positions.
///
/// Output:
/// - cos_cache: [max_seq_len, head_dim/2]
/// - sin_cache: [max_seq_len, head_dim/2]
kernel void compute_rope_cache(
    device float* cos_cache [[buffer(0)]],
    device float* sin_cache [[buffer(1)]],
    constant uint& max_seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& base [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint pos_idx = tgid.x;
    const uint half_dim = head_dim / 2;

    if (pos_idx >= max_seq_len) return;

    const float pos = float(pos_idx) * scale;
    const uint offset = pos_idx * half_dim;

    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        const float freq = inv_freq(d, head_dim, base);
        const float angle = pos * freq;

        cos_cache[offset + d] = cos(angle);
        sin_cache[offset + d] = sin(angle);
    }
}

/// Apply RoPE using precomputed cache.
kernel void rope_with_cache(
    device float* x [[buffer(0)]],                  // [batch, heads, seq_len, head_dim]
    device const float* cos_cache [[buffer(1)]],    // [max_seq_len, head_dim/2]
    device const float* sin_cache [[buffer(2)]],    // [max_seq_len, head_dim/2]
    constant RoPEParams& params [[buffer(3)]],
    constant uint& offset [[buffer(4)]],            // Position offset for KV cache
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.x;
    const uint head_idx = tgid.y;
    const uint seq_idx = tgid.z;

    if (batch_idx >= params.batch_size ||
        head_idx >= params.num_heads ||
        seq_idx >= params.seq_len) {
        return;
    }

    const uint half_dim = params.head_dim / 2;
    const uint pos = seq_idx + offset;
    const uint cache_offset = pos * half_dim;
    const uint x_offset = ((batch_idx * params.num_heads + head_idx) * params.seq_len + seq_idx) * params.head_dim;
    device float* x_ptr = x + x_offset;

    for (uint d = tid; d < half_dim; d += THREADS_PER_HEAD) {
        const float cos_val = cos_cache[cache_offset + d];
        const float sin_val = sin_cache[cache_offset + d];

        const float x1 = x_ptr[d];
        const float x2 = x_ptr[d + half_dim];

        x_ptr[d] = x1 * cos_val - x2 * sin_val;
        x_ptr[d + half_dim] = x1 * sin_val + x2 * cos_val;
    }
}
