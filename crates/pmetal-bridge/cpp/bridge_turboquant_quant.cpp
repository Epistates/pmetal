// TurboQuant quantize-and-pack and dequantize kernels.
//
// A row is a vector the caller has already rotated (`rotated = x · Πᵀ`, one
// ordinary matmul). Because the rotation is orthogonal, the row's norm is the
// norm of x. The kernel
//
//   1. takes the L2 norm of the row,
//   2. divides by it, which puts the row on the unit sphere,
//   3. replaces each coordinate by the index of its nearest Lloyd-Max
//      centroid (a binary search over the codebook's midpoints),
//   4. packs the indices, `bits` each, into little-endian 32-bit words, and
//   5. stores, as the row's scale, the norm times the least-squares gain of
//      the reconstruction (see below).
//
// The packed layout is a plain bitstream: coordinate `i` occupies bits
// `[i * bits, (i + 1) * bits)` of the row, counted from bit 0 of word 0, and an
// index may straddle two words. A row takes `ceil(dim * bits / 32)` words. The
// attention kernels read the same layout; `bridge_turboquant_attention.cpp`
// and the Rust oracle in the tests spell it out the same way.
//
// A row whose norm is zero, or whose squared norm is not finite, is stored as
// the zero vector (norm 0, all indices 0), which is also what it decodes to.

#include "bridge_internal.h"

static const char* TURBOQUANT_QUANTIZE_PACK_SOURCE = R"(
    constexpr uint kDim = (uint)DIM;
    constexpr uint kBits = (uint)BITS;
    constexpr uint kWords = (kDim * kBits + 31u) / 32u;
    constexpr uint kLevels = (1u << kBits) - 1u;   // number of thresholds

    threadgroup uchar lds_index[kDim];

    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    const device float* x = rotated + (ulong)row * kDim;

    float sum_sq = 0.0f;
    for (uint i = lane; i < kDim; i += 32u) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    bool ok = isfinite(sum_sq) && sum_sq > 1e-30f;
    float norm = ok ? sqrt(sum_sq) : 0.0f;
    float inv = ok ? 1.0f / norm : 0.0f;

    // Quantize every coordinate, and accumulate <y, y_hat> and |y_hat|^2.
    float dot = 0.0f;
    float centroid_sq = 0.0f;
    for (uint i = lane; i < kDim; i += 32u) {
        float y = x[i] * inv;
        uint lo = 0u;
        uint hi = kLevels;
        for (uint step = 0u; step < kBits; ++step) {
            uint mid = (lo + hi) >> 1u;
            if (y > thresholds[mid]) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        lds_index[i] = (uchar)lo;
        float c = codebook[lo];
        dot += y * c;
        centroid_sq += c * c;
    }
    dot = simd_sum(dot);
    centroid_sq = simd_sum(centroid_sq);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint w = lane; w < kWords; w += 32u) {
        uint word = 0u;
        uint first = (w * 32u) / kBits;
        uint last = min((w * 32u + 31u) / kBits, kDim - 1u);
        for (uint i = first; i <= last; ++i) {
            uint idx = (uint)lds_index[i];
            int shift = (int)(i * kBits) - (int)(w * 32u);
            word |= shift >= 0 ? (idx << (uint)shift) : (idx >> (uint)(-shift));
        }
        packed[(ulong)row * kWords + w] = word;
    }
    if (lane == 0u) {
        // The stored scale is |x| times the least-squares gain of the
        // reconstruction, <y, y_hat> / |y_hat|^2: a Lloyd-Max codebook
        // shrinks every vector a little, and the largest coordinates (the
        // ones an aligned query weighs most) the most, which biases the
        // logits of exactly the keys attention is about.
        float gain = (centroid_sq > 0.0f && dot > 0.0f) ? dot / centroid_sq : 1.0f;
        scales[row] = norm * gain;
    }
)";

static const char* TURBOQUANT_DEQUANTIZE_SOURCE = R"(
    constexpr uint kDim = (uint)DIM;
    constexpr uint kBits = (uint)BITS;
    constexpr uint kWords = (kDim * kBits + 31u) / 32u;
    constexpr uint kMask = (1u << kBits) - 1u;

    uint i = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint bit = i * kBits;
    uint wi = bit >> 5u;
    uint off = bit & 31u;
    const device uint* words = packed + (ulong)row * kWords;
    uint x = words[wi] >> off;
    if (off + kBits > 32u) {
        x |= words[wi + 1u] << (32u - off);
    }
    out[(ulong)row * kDim + i] = scales[row] * codebook[x & kMask];
)";

static mlx::core::fast::CustomKernelFunction& get_turboquant_quantize_pack_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_quantize_pack",
        {"rotated", "thresholds", "codebook"},
        {"packed", "scales"},
        TURBOQUANT_QUANTIZE_PACK_SOURCE,
        "",
        true,
        false);
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_dequantize_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_dequantize",
        {"packed", "scales", "codebook"},
        {"out"},
        TURBOQUANT_DEQUANTIZE_SOURCE,
        "",
        true,
        false);
    return kernel;
}

extern "C" {

int mlx_inline_turboquant_quantize_pack(
    mlx_inline_array*       out_packed,
    mlx_inline_array*       out_scales,
    const mlx_inline_array* rotated,
    const mlx_inline_array* thresholds,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                bits,
    uint32_t                n_rows)
{
    using namespace mlx::core;
    try {
        if (dim == 0 || bits == 0 || bits > 8 || n_rows == 0) {
            throw std::invalid_argument("[turboquant_quantize_pack] dim, bits or rows out of range");
        }
        const array& rot = as_arr(rotated);
        const array& thr = as_arr(thresholds);
        const array& cb = as_arr(codebook);
        if (rot.dtype() != float32 || thr.dtype() != float32 || cb.dtype() != float32) {
            throw std::invalid_argument("[turboquant_quantize_pack] rows, thresholds and codebook must be f32");
        }
        if (rot.size() != (size_t)n_rows * dim || thr.size() != ((size_t)1 << bits) - 1 ||
            cb.size() != ((size_t)1 << bits)) {
            throw std::invalid_argument("[turboquant_quantize_pack] input sizes do not match dim, bits and rows");
        }
        const int words = (int)((dim * bits + 31u) / 32u);
        auto outputs = get_turboquant_quantize_pack_kernel()(
            {rot, thr, cb},
            {{(int)n_rows, words}, {(int)n_rows}},
            {uint32, float32},
            {32, (int)n_rows, 1},
            {32, 1, 1},
            {{"DIM", (int)dim}, {"BITS", (int)bits}},
            std::nullopt, false, {});
        new (out_packed->buf) array(outputs[0]);
        new (out_scales->buf) array(outputs[1]);
        pmetal_bridge_clear_error_internal();
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("turboquant_quantize_pack", e.what());
    } catch (...) {
        pmetal_bridge_set_last_error("turboquant_quantize_pack", "unknown C++ exception");
    }
    bridge_placeholder(out_packed);
    bridge_placeholder(out_scales);
    return 1;
}

int mlx_inline_turboquant_dequantize(
    mlx_inline_array*       out,
    const mlx_inline_array* packed,
    const mlx_inline_array* scales,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                bits,
    uint32_t                n_rows)
{
    using namespace mlx::core;
    try {
        if (dim == 0 || bits == 0 || bits > 8 || n_rows == 0) {
            throw std::invalid_argument("[turboquant_dequantize] dim, bits or rows out of range");
        }
        const array& pk = as_arr(packed);
        const array& nm = as_arr(scales);
        const array& cb = as_arr(codebook);
        const size_t words = (dim * bits + 31u) / 32u;
        if (pk.dtype() != uint32 || nm.dtype() != float32 || cb.dtype() != float32) {
            throw std::invalid_argument("[turboquant_dequantize] expects u32 words, f32 scales and codebook");
        }
        if (pk.size() != (size_t)n_rows * words || nm.size() != n_rows || cb.size() != ((size_t)1 << bits)) {
            throw std::invalid_argument("[turboquant_dequantize] input sizes do not match dim, bits and rows");
        }
        auto outputs = get_turboquant_dequantize_kernel()(
            {pk, nm, cb},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)dim, (int)n_rows, 1},
            {(int)std::min(dim, 128u), 1, 1},
            {{"DIM", (int)dim}, {"BITS", (int)bits}},
            std::nullopt, false, {});
        new (out->buf) array(outputs[0]);
        pmetal_bridge_clear_error_internal();
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("turboquant_dequantize", e.what());
    } catch (...) {
        pmetal_bridge_set_last_error("turboquant_dequantize", "unknown C++ exception");
    }
    bridge_placeholder(out);
    return 1;
}

} // extern "C"
