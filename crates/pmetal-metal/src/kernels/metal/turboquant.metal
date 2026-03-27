#include <metal_stdlib>
using namespace metal;

struct TurboQuantTransformParams {
    uint dim;
};

kernel void turboquant_apply_rows(
    const device float* input [[buffer(0)]],
    const device float* matrix [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant TurboQuantTransformParams& params [[buffer(3)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    const uint dim = params.dim;
    const uint row_base = row_idx * dim;

    for (uint index = tid; index < dim; index += threads_per_group) {
        shared_input[index] = input[row_base + index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint out_dim = tid; out_dim < dim; out_dim += threads_per_group) {
        const device float* matrix_row = matrix + out_dim * dim;
        float acc = 0.0f;
        for (uint index = 0; index < dim; ++index) {
            acc += matrix_row[index] * shared_input[index];
        }
        output[row_base + out_dim] = acc;
    }
}
