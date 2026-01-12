//
// mHC (Manifold-Constrained Hyper-Connections) Metal Kernels
//
// Based on arXiv:2512.24880 - DeepSeek's mHC paper
//
// These kernels implement:
// 1. Compute mappings: Fused RMSNorm + projection for H̃^pre, H̃^post, H̃^res
// 2. Sinkhorn-Knopp: Iterative doubly stochastic projection
// 3. Apply pre: Aggregate streams using H^pre weights
// 4. Apply post+res: Fused post-mapping and residual merge
//

#include <metal_stdlib>
using namespace metal;

// Maximum expansion rate (n)
constant uint MAX_N = 16;

// Constants for Sinkhorn iteration
constant uint SINKHORN_ITERS = 20;
constant float EPSILON = 1e-8f;

//==============================================================================
// Kernel 1: Compute Mappings
//
// Fuses: x̃_l' = RMSNorm(flatten(x_l))
//        H̃ = α · (x̃' · φ) + b
//
// Inputs:
//   x_l: [batch, n, C] - Input tensor (bfloat16)
//   phi: [nC, n² + 2n] - Projection matrices (tfloat32)
//   bias: [n² + 2n] - Biases (float32)
//   alpha: [3] - Gating factors (float32)
//
// Outputs:
//   H_tilde: [batch, n² + 2n] - Unconstrained mappings (float32)
//==============================================================================

kernel void compute_mappings(
    device const half* x_l         [[buffer(0)]],
    device const float* phi        [[buffer(1)]],
    device const float* bias       [[buffer(2)]],
    device const float* alpha      [[buffer(3)]],
    device float* H_tilde          [[buffer(4)]],
    constant uint& batch_size      [[buffer(5)]],
    constant uint& n               [[buffer(6)]],
    constant uint& C               [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    uint nC = n * C;
    uint output_dim = n * n + 2 * n;

    // Each thread handles one batch element
    for (uint batch_idx = tid; batch_idx < batch_size; batch_idx += threads) {
        uint x_offset = batch_idx * nC;
        uint out_offset = batch_idx * output_dim;

        // Step 1: Compute RMS norm
        float sum_sq = 0.0f;
        for (uint i = 0; i < nC; i++) {
            float val = float(x_l[x_offset + i]);
            sum_sq += val * val;
        }
        float rms_inv = rsqrt(sum_sq / float(nC) + EPSILON);

        // Step 2: Compute projections with fused norm
        // H̃^pre (first n outputs)
        for (uint j = 0; j < n; j++) {
            float acc = 0.0f;
            for (uint i = 0; i < nC; i++) {
                float x_norm = float(x_l[x_offset + i]) * rms_inv;
                acc += x_norm * phi[i * output_dim + j];
            }
            H_tilde[out_offset + j] = alpha[0] * acc + bias[j];
        }

        // H̃^post (next n outputs)
        for (uint j = 0; j < n; j++) {
            float acc = 0.0f;
            for (uint i = 0; i < nC; i++) {
                float x_norm = float(x_l[x_offset + i]) * rms_inv;
                acc += x_norm * phi[i * output_dim + n + j];
            }
            H_tilde[out_offset + n + j] = alpha[1] * acc + bias[n + j];
        }

        // H̃^res (remaining n² outputs)
        for (uint j = 0; j < n * n; j++) {
            float acc = 0.0f;
            for (uint i = 0; i < nC; i++) {
                float x_norm = float(x_l[x_offset + i]) * rms_inv;
                acc += x_norm * phi[i * output_dim + 2 * n + j];
            }
            H_tilde[out_offset + 2 * n + j] = alpha[2] * acc + bias[2 * n + j];
        }
    }
}

//==============================================================================
// Kernel 2: Apply Constraints (Sigmoid + Sinkhorn-Knopp)
//
// H^pre = σ(H̃^pre)
// H^post = 2σ(H̃^post)
// H^res = Sinkhorn-Knopp(H̃^res)
//==============================================================================

kernel void apply_constraints(
    device float* H_tilde          [[buffer(0)]],  // [batch, n² + 2n] in/out
    device float* H_pre            [[buffer(1)]],  // [batch, n] output
    device float* H_post           [[buffer(2)]],  // [batch, n] output
    device float* H_res            [[buffer(3)]],  // [batch, n, n] output
    constant uint& batch_size      [[buffer(4)]],
    constant uint& n               [[buffer(5)]],
    constant uint& sinkhorn_iters  [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    uint output_dim = n * n + 2 * n;

    for (uint batch_idx = tid; batch_idx < batch_size; batch_idx += threads) {
        uint h_offset = batch_idx * output_dim;
        uint pre_offset = batch_idx * n;
        uint post_offset = batch_idx * n;
        uint res_offset = batch_idx * n * n;

        // Apply sigmoid to H^pre
        for (uint i = 0; i < n; i++) {
            float val = H_tilde[h_offset + i];
            H_pre[pre_offset + i] = 1.0f / (1.0f + exp(-val));
        }

        // Apply scaled sigmoid to H^post
        for (uint i = 0; i < n; i++) {
            float val = H_tilde[h_offset + n + i];
            H_post[post_offset + i] = 2.0f / (1.0f + exp(-val));
        }

        // Sinkhorn-Knopp for H^res
        // Thread-local storage for n×n matrix
        float M[MAX_N * MAX_N];

        // Exponentiate
        for (uint i = 0; i < n; i++) {
            for (uint j = 0; j < n; j++) {
                M[i * n + j] = exp(H_tilde[h_offset + 2 * n + i * n + j]);
            }
        }

        // Iterate
        for (uint t = 0; t < sinkhorn_iters; t++) {
            // Row normalization
            for (uint i = 0; i < n; i++) {
                float row_sum = EPSILON;
                for (uint j = 0; j < n; j++) {
                    row_sum += M[i * n + j];
                }
                float inv_sum = 1.0f / row_sum;
                for (uint j = 0; j < n; j++) {
                    M[i * n + j] *= inv_sum;
                }
            }

            // Column normalization
            for (uint j = 0; j < n; j++) {
                float col_sum = EPSILON;
                for (uint i = 0; i < n; i++) {
                    col_sum += M[i * n + j];
                }
                float inv_sum = 1.0f / col_sum;
                for (uint i = 0; i < n; i++) {
                    M[i * n + j] *= inv_sum;
                }
            }
        }

        // Write output
        for (uint i = 0; i < n; i++) {
            for (uint j = 0; j < n; j++) {
                H_res[res_offset + i * n + j] = M[i * n + j];
            }
        }
    }
}

//==============================================================================
// Kernel 3: Apply Pre-Mapping
//
// h_in = H^pre @ x_l  (aggregate streams)
//
// Inputs:
//   x_l: [batch, n, C]
//   H_pre: [batch, n]
//
// Output:
//   h_in: [batch, C]
//==============================================================================

kernel void apply_pre_mapping(
    device const half* x_l         [[buffer(0)]],  // [batch, n, C]
    device const float* H_pre      [[buffer(1)]],  // [batch, n]
    device half* h_in              [[buffer(2)]],  // [batch, C]
    constant uint& batch_size      [[buffer(3)]],
    constant uint& n               [[buffer(4)]],
    constant uint& C               [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 threads [[threads_per_grid]]
) {
    uint batch_idx = tid.x;
    uint c_start = tid.y * 4;  // Process 4 elements at a time

    if (batch_idx >= batch_size) return;

    uint x_offset = batch_idx * n * C;
    uint pre_offset = batch_idx * n;
    uint out_offset = batch_idx * C;

    // Process 4 channels at a time for better memory access
    for (uint c = c_start; c < C && c < c_start + 4; c++) {
        float sum = 0.0f;
        for (uint s = 0; s < n; s++) {
            sum += H_pre[pre_offset + s] * float(x_l[x_offset + s * C + c]);
        }
        h_in[out_offset + c] = half(sum);
    }
}

//==============================================================================
// Kernel 4: Apply Post+Res Mapping (Fused)
//
// x_{l+1} = H^res @ x_l + H^post^T @ h_out
//
// This kernel fuses the residual mixing and post-mapping for efficiency.
//
// Inputs:
//   x_l: [batch, n, C]
//   h_out: [batch, C]
//   H_res: [batch, n, n]
//   H_post: [batch, n]
//
// Output:
//   x_out: [batch, n, C]
//==============================================================================

kernel void apply_post_res_mapping(
    device const half* x_l         [[buffer(0)]],  // [batch, n, C]
    device const half* h_out       [[buffer(1)]],  // [batch, C]
    device const float* H_res      [[buffer(2)]],  // [batch, n, n]
    device const float* H_post     [[buffer(3)]],  // [batch, n]
    device half* x_out             [[buffer(4)]],  // [batch, n, C]
    constant uint& batch_size      [[buffer(5)]],
    constant uint& n               [[buffer(6)]],
    constant uint& C               [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 threads [[threads_per_grid]]
) {
    uint batch_idx = tid.x;
    uint stream_idx = tid.y;
    uint c_start = tid.z * 4;  // Process 4 channels

    if (batch_idx >= batch_size || stream_idx >= n) return;

    uint x_offset = batch_idx * n * C;
    uint hout_offset = batch_idx * C;
    uint res_offset = batch_idx * n * n;
    uint post_offset = batch_idx * n;
    uint out_offset = batch_idx * n * C;

    // Load H^post weight for this stream
    float h_post_val = H_post[post_offset + stream_idx];

    // Process channels
    for (uint c = c_start; c < C && c < c_start + 4; c++) {
        // H^res @ x_l component
        float res_val = 0.0f;
        for (uint j = 0; j < n; j++) {
            res_val += H_res[res_offset + stream_idx * n + j] *
                       float(x_l[x_offset + j * C + c]);
        }

        // H^post^T @ h_out component
        float post_val = h_post_val * float(h_out[hout_offset + c]);

        x_out[out_offset + stream_idx * C + c] = half(res_val + post_val);
    }
}

//==============================================================================
// Kernel 5: Sinkhorn-Knopp Backward
//
// Computes gradient through Sinkhorn iterations by recomputing forward
// and backpropagating through each iteration.
//==============================================================================

kernel void sinkhorn_backward(
    device const float* H_tilde    [[buffer(0)]],  // [batch, n, n] input
    device const float* grad_H_res [[buffer(1)]],  // [batch, n, n] upstream grad
    device float* grad_H_tilde     [[buffer(2)]],  // [batch, n, n] output grad
    constant uint& batch_size      [[buffer(3)]],
    constant uint& n               [[buffer(4)]],
    constant uint& sinkhorn_iters  [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    for (uint batch_idx = tid; batch_idx < batch_size; batch_idx += threads) {
        uint offset = batch_idx * n * n;

        // Storage for forward pass intermediates and gradients
        float M[MAX_N * MAX_N];
        float intermediates[SINKHORN_ITERS + 1][MAX_N * MAX_N];
        float grad[MAX_N * MAX_N];

        // Forward pass with storage
        for (uint i = 0; i < n * n; i++) {
            M[i] = exp(H_tilde[offset + i]);
            intermediates[0][i] = M[i];
        }

        for (uint t = 0; t < sinkhorn_iters; t++) {
            // Row normalization
            for (uint i = 0; i < n; i++) {
                float row_sum = EPSILON;
                for (uint j = 0; j < n; j++) {
                    row_sum += M[i * n + j];
                }
                float inv_sum = 1.0f / row_sum;
                for (uint j = 0; j < n; j++) {
                    M[i * n + j] *= inv_sum;
                }
            }

            // Column normalization
            for (uint j = 0; j < n; j++) {
                float col_sum = EPSILON;
                for (uint i = 0; i < n; i++) {
                    col_sum += M[i * n + j];
                }
                float inv_sum = 1.0f / col_sum;
                for (uint i = 0; i < n; i++) {
                    M[i * n + j] *= inv_sum;
                }
            }

            // Store intermediate
            for (uint i = 0; i < n * n; i++) {
                intermediates[t + 1][i] = M[i];
            }
        }

        // Initialize gradient from upstream
        for (uint i = 0; i < n * n; i++) {
            grad[i] = grad_H_res[offset + i];
        }

        // Backward pass through iterations
        for (int t = sinkhorn_iters - 1; t >= 0; t--) {
            // Recompute row-normalized intermediate
            float M_row[MAX_N * MAX_N];
            for (uint i = 0; i < n * n; i++) {
                M_row[i] = intermediates[t][i];
            }
            for (uint i = 0; i < n; i++) {
                float row_sum = EPSILON;
                for (uint j = 0; j < n; j++) {
                    row_sum += M_row[i * n + j];
                }
                float inv_sum = 1.0f / row_sum;
                for (uint j = 0; j < n; j++) {
                    M_row[i * n + j] *= inv_sum;
                }
            }

            // Backward through column norm
            float grad_M_row[MAX_N * MAX_N];
            for (uint j = 0; j < n; j++) {
                float col_sum = EPSILON;
                for (uint i = 0; i < n; i++) {
                    col_sum += M_row[i * n + j];
                }
                float inv_sum = 1.0f / col_sum;
                float inv_sum_sq = inv_sum * inv_sum;

                for (uint i = 0; i < n; i++) {
                    float g = grad[i * n + j];
                    grad_M_row[i * n + j] = g * inv_sum;

                    // Contribution from normalization
                    for (uint k = 0; k < n; k++) {
                        grad_M_row[k * n + j] -= g * M_row[k * n + j] * M_row[i * n + j] * inv_sum_sq;
                    }
                }
            }

            // Backward through row norm
            float grad_prev[MAX_N * MAX_N];
            for (uint i = 0; i < n; i++) {
                float row_sum = EPSILON;
                for (uint j = 0; j < n; j++) {
                    row_sum += intermediates[t][i * n + j];
                }
                float inv_sum = 1.0f / row_sum;
                float inv_sum_sq = inv_sum * inv_sum;

                for (uint j = 0; j < n; j++) {
                    float g = grad_M_row[i * n + j];
                    grad_prev[i * n + j] = g * inv_sum;

                    for (uint k = 0; k < n; k++) {
                        grad_prev[i * n + k] -= g * intermediates[t][i * n + k] *
                                                intermediates[t][i * n + j] * inv_sum_sq;
                    }
                }
            }

            for (uint i = 0; i < n * n; i++) {
                grad[i] = grad_prev[i];
            }
        }

        // Backward through exp
        for (uint i = 0; i < n * n; i++) {
            grad_H_tilde[offset + i] = intermediates[0][i] * grad[i];
        }
    }
}

//==============================================================================
// Kernel 6: Compute Amax Gain Magnitude (for monitoring)
//
// Computes the maximum absolute row sum (forward gain) and column sum
// (backward gain) for stability monitoring.
//==============================================================================

kernel void compute_amax_gain(
    device const float* H_res      [[buffer(0)]],  // [batch, n, n]
    device float* forward_gain     [[buffer(1)]],  // [batch]
    device float* backward_gain    [[buffer(2)]],  // [batch]
    constant uint& batch_size      [[buffer(3)]],
    constant uint& n               [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    for (uint batch_idx = tid; batch_idx < batch_size; batch_idx += threads) {
        uint offset = batch_idx * n * n;

        // Forward gain: max absolute row sum
        float max_row = 0.0f;
        for (uint i = 0; i < n; i++) {
            float row_sum = 0.0f;
            for (uint j = 0; j < n; j++) {
                row_sum += abs(H_res[offset + i * n + j]);
            }
            max_row = max(max_row, row_sum);
        }
        forward_gain[batch_idx] = max_row;

        // Backward gain: max absolute column sum
        float max_col = 0.0f;
        for (uint j = 0; j < n; j++) {
            float col_sum = 0.0f;
            for (uint i = 0; i < n; i++) {
                col_sum += abs(H_res[offset + i * n + j]);
            }
            max_col = max(max_col, col_sum);
        }
        backward_gain[batch_idx] = max_col;
    }
}

//==============================================================================
// Kernel 7: Expand to Streams
//
// Replicates single-stream input to n-stream format.
//==============================================================================

kernel void expand_to_streams(
    device const half* x_in        [[buffer(0)]],  // [batch, C]
    device half* x_out             [[buffer(1)]],  // [batch, n, C]
    constant uint& batch_size      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& C               [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint c_idx = tid.y;

    if (batch_idx >= batch_size || c_idx >= C) return;

    half val = x_in[batch_idx * C + c_idx];

    for (uint s = 0; s < n; s++) {
        x_out[batch_idx * n * C + s * C + c_idx] = val;
    }
}

//==============================================================================
// Kernel 8: Collapse Streams
//
// Collapses n-stream format back to single stream (average mode).
//==============================================================================

kernel void collapse_streams(
    device const half* x_in        [[buffer(0)]],  // [batch, n, C]
    device half* x_out             [[buffer(1)]],  // [batch, C]
    constant uint& batch_size      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& C               [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint c_idx = tid.y;

    if (batch_idx >= batch_size || c_idx >= C) return;

    float sum = 0.0f;
    for (uint s = 0; s < n; s++) {
        sum += float(x_in[batch_idx * n * C + s * C + c_idx]);
    }

    x_out[batch_idx * C + c_idx] = half(sum / float(n));
}
