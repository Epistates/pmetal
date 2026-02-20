//  fused_merge.metal
//  Fused model merging kernels for Apple Silicon
//
//  These kernels enable efficient model merging by fusing:
//  1. Task vector computation (tensor - base)
//  2. Magnitude-based sparsification
//  3. Sign consensus for TIES merging
//  4. Weighted sum with scaling
//
//  Key insight: Instead of multiple MLX ops with intermediate tensors,
//  fuse operations into single dispatches to eliminate memory traffic.
//
//  Performance target: 3-5x speedup over sequential MLX operations

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Configuration
// =============================================================================

// Maximum number of models supported by fused TIES kernel
// Limited by threadgroup memory for sparse_task_vectors array
constant uint MAX_MODELS = 16;

// =============================================================================
// Merge Configuration Structures
// =============================================================================

/// Metadata for batched tensor processing
struct TensorInfo {
    uint offset;            // Offset into the flattened buffer
    uint size;              // Number of elements in this tensor
    float density;          // Sparsification density (0.0-1.0)
    float threshold;        // Computed magnitude threshold
};

/// Global merge configuration
struct MergeConfig {
    uint num_tensors;       // Number of tensors in batch
    uint total_elements;    // Total elements across all tensors
    float epsilon;          // Numerical stability constant
    uint _pad;              // Padding for alignment
};

/// TIES merge configuration
struct TiesConfig {
    uint num_models;        // Number of models being merged
    uint elements_per_model; // Elements per model tensor
    float lambda;           // Global scaling factor
    float epsilon;          // Numerical stability constant
};

// =============================================================================
// Basic Operations
// =============================================================================

/// Compute absolute value (magnitude) for each element
kernel void fused_compute_magnitudes(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const TensorInfo* tensor_info [[buffer(2)]],
    constant MergeConfig& config [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.total_elements) return;
    output[tid] = abs(input[tid]);
}

/// Apply sparsification mask based on pre-computed thresholds
/// output[i] = |input[i]| >= threshold ? input[i] : 0
kernel void fused_apply_sparsification(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const TensorInfo* tensor_info [[buffer(2)]],
    constant MergeConfig& config [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.total_elements) return;

    // Find which tensor this element belongs to
    uint tensor_idx = 0;
    for (uint i = 0; i < config.num_tensors; i++) {
        if (tid >= tensor_info[i].offset &&
            tid < tensor_info[i].offset + tensor_info[i].size) {
            tensor_idx = i;
            break;
        }
    }

    float val = input[tid];
    float threshold = tensor_info[tensor_idx].threshold;

    // Apply threshold mask
    output[tid] = (abs(val) >= threshold) ? val : 0.0f;
}

/// Compute task vectors: output = fine_tuned - base
kernel void fused_task_vectors(
    device const float* fine_tuned [[buffer(0)]],
    device const float* base [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_elements) return;
    output[tid] = fine_tuned[tid] - base[tid];
}

/// Compute weighted sum: output = base + lambda * sum(weight[i] * tensor[i])
kernel void fused_weighted_sum(
    device const float* tensors [[buffer(0)]],     // [num_models, elements]
    device const float* weights [[buffer(1)]],     // [num_models]
    device const float* base [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_models [[buffer(4)]],
    constant uint& total_elements [[buffer(5)]],
    constant float& lambda [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_elements) return;

    // Accumulate weighted sum across models
    float sum = 0.0f;
    for (uint m = 0; m < num_models; m++) {
        sum += weights[m] * tensors[m * total_elements + tid];
    }

    // Apply lambda scaling and add to base
    output[tid] = base[tid] + lambda * sum;
}

// =============================================================================
// Sign Consensus
// =============================================================================

/// Compute sign consensus for TIES merging
/// consensus[i] = sign(sum(weight[j] * sign(tensor[j][i])))
kernel void fused_sign_consensus(
    device const float* tensors [[buffer(0)]],     // [num_models, elements]
    device const float* weights [[buffer(1)]],     // [num_models]
    device float* consensus [[buffer(2)]],         // [elements]
    constant uint& num_models [[buffer(3)]],
    constant uint& total_elements [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_elements) return;

    // Compute weighted sign sum
    float weighted_sign_sum = 0.0f;
    for (uint m = 0; m < num_models; m++) {
        float val = tensors[m * total_elements + tid];
        float sign_val = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        weighted_sign_sum += weights[m] * sign_val;
    }

    // Output is the sign of the weighted sum
    consensus[tid] = (weighted_sign_sum > 0.0f) ? 1.0f :
                     ((weighted_sign_sum < 0.0f) ? -1.0f : 0.0f);
}

// =============================================================================
// Fused TIES Merge
// =============================================================================

/// Fused TIES merge kernel - combines all TIES operations in single pass
///
/// For each element:
/// 1. Compute task vectors (tensor - base) for each model
/// 2. Apply magnitude threshold (sparsification)
/// 3. Compute sign consensus
/// 4. Apply consensus mask and weighted sum
/// 5. Scale by lambda and add to base
///
/// Input layout:
/// - tensors: [num_models, elements_per_model] - fine-tuned models
/// - base: [elements_per_model] - base model
/// - weights: [num_models] - per-model weights
/// - thresholds: [num_models] - pre-computed magnitude thresholds
///
/// Output: [elements_per_model] - merged result
kernel void fused_ties_merge(
    device const float* tensors [[buffer(0)]],     // [num_models, elements]
    device const float* base [[buffer(1)]],        // [elements]
    device const float* weights [[buffer(2)]],     // [num_models]
    device const float* thresholds [[buffer(3)]],  // [num_models]
    device float* output [[buffer(4)]],            // [elements]
    constant TiesConfig& config [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.elements_per_model) return;

    const uint num_models = config.num_models;
    const uint elements = config.elements_per_model;
    const float base_val = base[tid];

    // Step 1-2: Compute sparse task vectors and weighted sign sum
    float weighted_sign_sum = 0.0f;
    float sparse_task_vectors[16];  // Support up to 16 models

    for (uint m = 0; m < num_models && m < 16; m++) {
        float model_val = tensors[m * elements + tid];
        float task_val = model_val - base_val;
        float abs_task = abs(task_val);
        float threshold = thresholds[m];

        // Sparsify by magnitude
        float sparse_val = (abs_task >= threshold) ? task_val : 0.0f;
        sparse_task_vectors[m] = sparse_val;

        // Accumulate weighted sign for consensus
        if (sparse_val != 0.0f) {
            float sign_val = (sparse_val > 0.0f) ? 1.0f : -1.0f;
            weighted_sign_sum += weights[m] * sign_val;
        }
    }

    // Step 3: Compute consensus sign
    float consensus = (weighted_sign_sum > 0.0f) ? 1.0f :
                      ((weighted_sign_sum < 0.0f) ? -1.0f : 0.0f);

    // Step 4: Apply consensus mask and compute weighted sum
    float weighted_sum = 0.0f;
    for (uint m = 0; m < num_models && m < 16; m++) {
        float sparse_val = sparse_task_vectors[m];

        // Only include if sign matches consensus
        if (sparse_val != 0.0f) {
            float sign_val = (sparse_val > 0.0f) ? 1.0f : -1.0f;
            if (sign_val == consensus) {
                weighted_sum += weights[m] * sparse_val;
            }
        }
    }

    // Step 5: Apply lambda scaling and add to base
    output[tid] = base_val + config.lambda * weighted_sum;
}

/// Fused TIES merge with online threshold computation
/// Uses reservoir sampling to estimate threshold on-the-fly
kernel void fused_ties_merge_online(
    device const float* tensors [[buffer(0)]],     // [num_models, elements]
    device const float* base [[buffer(1)]],        // [elements]
    device const float* weights [[buffer(2)]],     // [num_models]
    device const float* densities [[buffer(3)]],   // [num_models]
    device float* output [[buffer(4)]],            // [elements]
    device float* scratch [[buffer(5)]],           // Scratch space for thresholds
    constant TiesConfig& config [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    // This kernel processes in two phases:
    // Phase 1: Estimate thresholds using partial histogram
    // Phase 2: Apply fused TIES merge with computed thresholds

    // For simplicity in this version, assume thresholds are pre-computed
    // and passed in the scratch buffer

    if (tid >= config.elements_per_model) return;

    const uint num_models = config.num_models;
    const uint elements = config.elements_per_model;
    const float base_val = base[tid];

    float weighted_sign_sum = 0.0f;
    float sparse_values[16];

    for (uint m = 0; m < num_models && m < 16; m++) {
        float model_val = tensors[m * elements + tid];
        float task_val = model_val - base_val;
        float abs_task = abs(task_val);
        float threshold = scratch[m];  // Pre-computed threshold

        float sparse_val = (abs_task >= threshold) ? task_val : 0.0f;
        sparse_values[m] = sparse_val;

        if (sparse_val != 0.0f) {
            float sign_val = (sparse_val > 0.0f) ? 1.0f : -1.0f;
            weighted_sign_sum += weights[m] * sign_val;
        }
    }

    float consensus = (weighted_sign_sum > 0.0f) ? 1.0f :
                      ((weighted_sign_sum < 0.0f) ? -1.0f : 0.0f);

    float weighted_sum = 0.0f;
    for (uint m = 0; m < num_models && m < 16; m++) {
        float sparse_val = sparse_values[m];
        if (sparse_val != 0.0f) {
            float sign_val = (sparse_val > 0.0f) ? 1.0f : -1.0f;
            if (sign_val == consensus) {
                weighted_sum += weights[m] * sparse_val;
            }
        }
    }

    output[tid] = base_val + config.lambda * weighted_sum;
}

// =============================================================================
// Partial Threshold Estimation
// =============================================================================

/// Compute partial samples for threshold estimation using reservoir sampling
/// Each threadgroup samples magnitudes for histogram-based threshold estimation
kernel void fused_partial_threshold(
    device const float* magnitudes [[buffer(0)]],
    device float* partial_samples [[buffer(1)]],
    device const TensorInfo* tensor_info [[buffer(2)]],
    constant MergeConfig& config [[buffer(3)]],
    constant uint& samples_per_tensor [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one tensor
    if (tgid >= config.num_tensors) return;

    TensorInfo info = tensor_info[tgid];
    uint tensor_offset = info.offset;
    uint tensor_size = info.size;

    // Threadgroup-local sample buffer
    threadgroup float local_samples[256];
    threadgroup uint sample_count;

    if (lid == 0) {
        sample_count = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread contributes samples
    uint stride = tensor_size / min(tg_size, tensor_size);
    for (uint i = lid; i < samples_per_tensor && i * stride < tensor_size; i += tg_size) {
        uint idx = tensor_offset + (i * stride);
        if (idx < tensor_offset + tensor_size) {
            local_samples[i % 256] = magnitudes[idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write samples to global memory
    uint output_offset = tgid * samples_per_tensor;
    for (uint i = lid; i < samples_per_tensor; i += tg_size) {
        if (output_offset + i < config.num_tensors * samples_per_tensor) {
            partial_samples[output_offset + i] = local_samples[i % 256];
        }
    }
}

// =============================================================================
// DARE (Drop And REscale) Operations
// =============================================================================

/// Murmur3 finalization hash for deterministic per-element pseudo-randomness.
/// Provides good avalanche effect: every output bit depends on every input bit.
inline uint hash_murmur3(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

/// DARE sparsification: randomly drop elements and rescale
/// Uses deterministic "random" based on element index and seed
kernel void fused_dare_sparsify(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& density [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    constant uint& seed [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_elements) return;

    // Handle edge cases
    if (density >= 1.0f) {
        output[tid] = input[tid];
        return;
    }
    if (density <= 0.0f) {
        output[tid] = 0.0f;
        return;
    }

    // Use Murmur3 finalization hash for better avalanche effect (MED-M5 fix).
    // The previous 2-round hash had poor bit mixing and visible patterns.
    float rand = float(hash_murmur3(tid ^ seed)) / float(0xFFFFFFFFu);

    // Keep with probability density, rescale by 1/density
    if (rand < density) {
        output[tid] = input[tid] / density;
    } else {
        output[tid] = 0.0f;
    }
}

// =============================================================================
// Linear Merge (Simple Weighted Average)
// =============================================================================

/// Simple weighted average merge: output = sum(weight[i] * tensor[i])
kernel void fused_linear_merge(
    device const float* tensors [[buffer(0)]],     // [num_models, elements]
    device const float* weights [[buffer(1)]],     // [num_models]
    device float* output [[buffer(2)]],            // [elements]
    constant uint& num_models [[buffer(3)]],
    constant uint& total_elements [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_elements) return;

    float sum = 0.0f;
    for (uint m = 0; m < num_models; m++) {
        sum += weights[m] * tensors[m * total_elements + tid];
    }

    output[tid] = sum;
}

// =============================================================================
// SLERP (Spherical Linear Interpolation)
// =============================================================================

/// SLERP merge for two models
/// slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
/// where omega = arccos(dot(a,b) / (|a| * |b|))
///
/// This kernel computes the interpolation; dot product is pre-computed
kernel void fused_slerp_merge(
    device const float* tensor_a [[buffer(0)]],
    device const float* tensor_b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& t [[buffer(3)]],              // Interpolation factor
    constant float& omega [[buffer(4)]],          // Pre-computed angle
    constant float& sin_omega [[buffer(5)]],      // Pre-computed sin(omega)
    constant uint& total_elements [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_elements) return;

    // Handle degenerate case (parallel vectors)
    if (sin_omega < 1e-6) {
        // Fall back to linear interpolation
        output[tid] = (1.0f - t) * tensor_a[tid] + t * tensor_b[tid];
        return;
    }

    float coeff_a = sin((1.0f - t) * omega) / sin_omega;
    float coeff_b = sin(t * omega) / sin_omega;

    output[tid] = coeff_a * tensor_a[tid] + coeff_b * tensor_b[tid];
}

/// Compute dot product and norms for SLERP (partial reduction)
kernel void fused_slerp_dot_norm(
    device const float* tensor_a [[buffer(0)]],
    device const float* tensor_b [[buffer(1)]],
    device float* partial_results [[buffer(2)]],  // [3 * num_threadgroups]: dot, norm_a, norm_b
    constant uint& total_elements [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_dot[256];
    threadgroup float shared_norm_a[256];
    threadgroup float shared_norm_b[256];

    float local_dot = 0.0f;
    float local_norm_a = 0.0f;
    float local_norm_b = 0.0f;

    // Each thread accumulates multiple elements
    uint start = tgid * tg_size * 4 + tid;
    for (uint i = 0; i < 4; i++) {
        uint idx = start + i * tg_size;
        if (idx < total_elements) {
            float a = tensor_a[idx];
            float b = tensor_b[idx];
            local_dot += a * b;
            local_norm_a += a * a;
            local_norm_b += b * b;
        }
    }

    shared_dot[tid] = local_dot;
    shared_norm_a[tid] = local_norm_a;
    shared_norm_b[tid] = local_norm_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_dot[tid] += shared_dot[tid + s];
            shared_norm_a[tid] += shared_norm_a[tid + s];
            shared_norm_b[tid] += shared_norm_b[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes results
    if (tid == 0) {
        partial_results[tgid * 3 + 0] = shared_dot[0];
        partial_results[tgid * 3 + 1] = shared_norm_a[0];
        partial_results[tgid * 3 + 2] = shared_norm_b[0];
    }
}
