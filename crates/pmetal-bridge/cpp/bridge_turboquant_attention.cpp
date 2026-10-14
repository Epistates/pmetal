// TurboQuant decode attention: one query token per row, over the compressed
// cold store plus an uncompressed tail, in two kernels.
//
// What the cache holds
//   cold    packed keys and values: per token, `ceil(D * bits / 32)` words of a
//           bitstream of Lloyd-Max centroid indices (the layout of
//           `bridge_turboquant_quant.cpp`) and one f32 scale (its norm times
//           the gain of its reconstruction). Keys and values are stored in
//           the *rotated* basis: a stored row is `Π x / |x|`.
//   dense   the first few tokens (attention sinks) and the most recent ones,
//           uncompressed, in the model's dtype and the original basis. The
//           recent ones live in a ring.
//
// The score of a cold token is `|k| * <Π q, ŷ_k>` and the value it adds is
// `|v| * ŷ_v` in the rotated basis, so the queries are rotated once per step
// and no key or value is ever rotated back. The accumulator of the cold blocks
// stays in the rotated basis and is rotated back once, after the merge.
//
// Pass 1 (`attention_partials`)
//   One threadgroup is one simdgroup: it owns one run of `tokens_per_block`
//   tokens of one KV head, and attends to them for every query head that
//   shares the KV head (GQA), so a token is read and unpacked once for all of
//   them. A tile is 32 tokens. Scoring puts a token on each lane: the lane
//   unpacks its key row and accumulates one dot product per query head. The
//   tile's weights are then broadcast and the values accumulated with a
//   dimension slice on each lane, the way MLX's own vector SDPA does. Softmax
//   is online, in base 2: the query is pre-multiplied by `scale * log2(e)`.
//   Output: per (query row, block) the unnormalised accumulator, the running
//   max and the running sum.
//
// Pass 2 (`attention_merge`)
//   One threadgroup per query row merges that row's blocks with the usual
//   log-sum-exp rescale, rotates the cold part of the accumulator back with
//   `Π`, adds the dense part, divides by the normaliser (which includes an
//   optional per-head sink logit, as gpt-oss uses) and writes the output.
//
// Everything that changes from step to step is in the int32 `params` array,
// never in a template argument: a template argument is compiled into the
// kernel, and a value that moves every decode step would compile a new Metal
// library for each token.
//
//   params[0]  cold tokens            params[6]  blocks over the cold store
//   params[1]  sink tokens            params[7]  attention scale * log2(e), f32 bits
//   params[2]  ring capacity          params[8]  cold store capacity (tokens)
//   params[3]  ring start             params[9]  dense store capacity (tokens)
//   params[4]  ring length            params[10] total blocks
//   params[5]  tokens per block       params[11] 1 when the merge has sink logits
//                                     params[12] first cold token attended to
//                                     params[13] first dense token attended to
//
// The last two let a layer attend to a suffix of the sequence (a sliding
// window, a chunk of chunked attention) with no change in how the blocks are
// laid out: a segment's blocks tile `[first, count)`.
//
// Dense slot of logical dense token j: j < sinks ? j : sinks + (start + j - sinks) % cap.

#include "bridge_internal.h"

#include <map>
#include <mutex>
#include <string>

static const char* TURBOQUANT_ATTENTION_PARTIALS_SOURCE = R"(
    constexpr uint kD = (uint)DIM;
    constexpr uint kD4 = kD / 4u;
    constexpr uint kDv = (uint)VDIM;
    constexpr uint kBk = (uint)KBITS;
    constexpr uint kBv = (uint)VBITS;
    constexpr uint kG = (uint)GROUPS;
    constexpr uint kHkv = (uint)KV_HEADS;
    constexpr uint kWk = (kD * kBk + 31u) / 32u;
    constexpr uint kWv = (kDv * kBv + 31u) / 32u;
    constexpr uint kVpl = (kDv + 31u) / 32u;
    constexpr uint kMaskK = (1u << kBk) - 1u;
    constexpr uint kMaskV = (1u << kBv) - 1u;

    // The queries live in threadgroup memory as halves: every lane reads all of them
    // for each token, so the bytes read (not the precision, 1e-3 relative on a
    // logit next to a quantization error of 1e-1) are what the key pass pays for.
    threadgroup half4 lds_q[kG * kD4];
    threadgroup float lds_cbk[1u << kBk];
    threadgroup float lds_cbv[1u << kBv];

    const int n_cold = params[0];
    const int n_sink = params[1];
    const int ring_cap = params[2];
    const int ring_start = params[3];
    const int ring_len = params[4];
    const int tokens_per_block = params[5];
    const int blocks_cold = params[6];
    const float qscale = as_type<float>(params[7]);
    const uint cap_cold = (uint)params[8];
    const uint cap_dense = (uint)params[9];
    const uint n_blocks = (uint)params[10];

    const uint blk = threadgroup_position_in_grid.x;
    const uint kv = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint batch = kv / kHkv;
    const uint hk = kv % kHkv;
    const uint r0 = batch * (kHkv * kG) + hk * kG;

    const bool cold = (int)blk < blocks_cold;
    const int n_dense = n_sink + ring_len;
    const int seg_blk = cold ? (int)blk : (int)blk - blocks_cold;
    const int t_begin = (cold ? params[12] : params[13]) + seg_blk * tokens_per_block;
    const int t_end = min(t_begin + tokens_per_block, cold ? n_cold : n_dense);

    if (cold) {
        const device float4* q4 = (const device float4*)(q_rot + (ulong)r0 * kD);
        for (uint i = lane; i < kG * kD4; i += 32u) {
            lds_q[i] = half4(clamp(q4[i] * qscale, -65504.0f, 65504.0f));
        }
        for (uint i = lane; i < (1u << kBk); i += 32u) {
            lds_cbk[i] = cb_k[i];
        }
        for (uint i = lane; i < (1u << kBv); i += 32u) {
            lds_cbv[i] = cb_v[i];
        }
    } else {
        for (uint i = lane; i < kG * kD4; i += 32u) {
            const ulong b = (ulong)r0 * kD + (ulong)i * 4u;
            lds_q[i] = half4(clamp(float4(float(q[b]), float(q[b + 1u]), float(q[b + 2u]), float(q[b + 3u])) * qscale, -65504.0f, 65504.0f));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m[kG];
    float l[kG];
    float acc[kG][kVpl];
    for (uint g = 0u; g < kG; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        for (uint i = 0u; i < kVpl; ++i) {
            acc[g][i] = 0.0f;
        }
    }

    for (int t0 = t_begin; t0 < t_end; t0 += 32) {
        const int t = t0 + (int)lane;
        const bool valid = t < t_end;

        // Score one token per lane, for every query head of the KV head.
        float s[kG];
        for (uint g = 0u; g < kG; ++g) {
            s[g] = 0.0f;
        }
        float vscale = 1.0f;
        if (valid) {
            if (cold) {
                const ulong tok = (ulong)kv * cap_cold + (uint)t;
                const device uint* row = k_packed + tok * kWk;
                // The row's words in registers, loaded 16 bytes at a time; the
                // loops below have compile-time bounds, so every index into
                // them is static after unrolling.
                uint wv[kWk];
                const device packed_uint4* row4 = (const device packed_uint4*)row;
                #pragma clang loop unroll(full)
                for (uint w = 0u; w + 4u <= kWk; w += 4u) {
                    const packed_uint4 v4 = row4[w / 4u];
                    wv[w] = v4[0];
                    wv[w + 1u] = v4[1];
                    wv[w + 2u] = v4[2];
                    wv[w + 3u] = v4[3];
                }
                #pragma clang loop unroll(full)
                for (uint w = (kWk / 4u) * 4u; w < kWk; ++w) {
                    wv[w] = row[w];
                }
                #pragma clang loop unroll(full)
                for (uint i4 = 0u; i4 < kD4; ++i4) {
                    float c[4];
                    #pragma clang loop unroll(full)
                    for (uint k = 0u; k < 4u; ++k) {
                        const uint bit = (i4 * 4u + k) * kBk;
                        const uint wi = bit >> 5u;
                        const uint off = bit & 31u;
                        uint x = wv[wi] >> off;
                        if (off + kBk > 32u) {
                            x |= wv[wi + 1u] << (32u - off);
                        }
                        c[k] = lds_cbk[x & kMaskK];
                    }
                    const float4 cv = float4(c[0], c[1], c[2], c[3]);
                    for (uint g = 0u; g < kG; ++g) {
                        s[g] += dot(float4(lds_q[g * kD4 + i4]), cv);
                    }
                }
                const float kn = k_scale[tok];
                for (uint g = 0u; g < kG; ++g) {
                    s[g] *= kn;
                }
                vscale = v_scale[tok];
            } else {
                const uint slot = t < n_sink
                    ? (uint)t
                    : (uint)n_sink + (uint)((ring_start + (t - n_sink)) % ring_cap);
                auto kr = dense_k + ((ulong)kv * cap_dense + slot) * kD;
                for (uint i4 = 0u; i4 < kD4; ++i4) {
                    const float4 cv = float4(
                        float(kr[i4 * 4u]), float(kr[i4 * 4u + 1u]),
                        float(kr[i4 * 4u + 2u]), float(kr[i4 * 4u + 3u]));
                    for (uint g = 0u; g < kG; ++g) {
                        s[g] += dot(float4(lds_q[g * kD4 + i4]), cv);
                    }
                }
            }
        }

        // Online softmax over the tile, in base 2.
        float pv[kG];
        for (uint g = 0u; g < kG; ++g) {
            const float sv = valid ? s[g] : -INFINITY;
            const float new_m = max(m[g], simd_max(sv));
            const float alpha = (new_m == -INFINITY) ? 1.0f : exp2(m[g] - new_m);
            const float p = (valid && new_m != -INFINITY) ? exp2(sv - new_m) : 0.0f;
            l[g] = l[g] * alpha + simd_sum(p);
            m[g] = new_m;
            for (uint i = 0u; i < kVpl; ++i) {
                acc[g][i] *= alpha;
            }
            pv[g] = p * vscale;
        }

        // Accumulate the tile's values, a slice of the dimensions per lane.
        const int tile = min(32, t_end - t0);
        for (int j = 0; j < tile; ++j) {
            float p[kG];
            for (uint g = 0u; g < kG; ++g) {
                p[g] = simd_broadcast(pv[g], (ushort)j);
            }
            const int tj = t0 + j;
            if (cold) {
                const ulong tok = (ulong)kv * cap_cold + (uint)tj;
                const device uint* vrow = v_packed + tok * kWv;
                for (uint i = 0u; i < kVpl; ++i) {
                    const uint d = lane * kVpl + i;
                    if (d < kDv) {
                        const uint bit = d * kBv;
                        const uint wi = bit >> 5u;
                        const uint off = bit & 31u;
                        uint x = vrow[wi] >> off;
                        if (off + kBv > 32u) {
                            x |= vrow[wi + 1u] << (32u - off);
                        }
                        const float c = lds_cbv[x & kMaskV];
                        for (uint g = 0u; g < kG; ++g) {
                            acc[g][i] = fma(p[g], c, acc[g][i]);
                        }
                    }
                }
            } else {
                const uint slot = tj < n_sink
                    ? (uint)tj
                    : (uint)n_sink + (uint)((ring_start + (tj - n_sink)) % ring_cap);
                auto vr = dense_v + ((ulong)kv * cap_dense + slot) * kDv;
                for (uint i = 0u; i < kVpl; ++i) {
                    const uint d = lane * kVpl + i;
                    if (d < kDv) {
                        const float c = float(vr[d]);
                        for (uint g = 0u; g < kG; ++g) {
                            acc[g][i] = fma(p[g], c, acc[g][i]);
                        }
                    }
                }
            }
        }
    }

    for (uint g = 0u; g < kG; ++g) {
        const ulong cell = (ulong)(r0 + g) * n_blocks + blk;
        if (lane == 0u) {
            part_m[cell] = m[g];
            part_l[cell] = l[g];
        }
        for (uint i = 0u; i < kVpl; ++i) {
            const uint d = lane * kVpl + i;
            if (d < kDv) {
                part_acc[cell * kDv + d] = acc[g][i];
            }
        }
    }
)";

static const char* TURBOQUANT_ATTENTION_MERGE_SOURCE = R"(
    constexpr uint kDv = (uint)VDIM;
    constexpr uint kHq = (uint)Q_HEADS;
    constexpr uint kThreads = 128u;

    threadgroup float lds_cold[kDv];
    threadgroup float lds_dense[kDv];
    threadgroup float lds_max[kThreads / 32u];

    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint simd_id = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n_blocks = (uint)params[10];
    const uint blocks_cold = (uint)params[6];
    const bool has_sink = params[11] != 0;
    const float log2e = 1.4426950408889634f;

    float mx = -INFINITY;
    for (uint b = tid; b < n_blocks; b += kThreads) {
        mx = max(mx, part_m[(ulong)row * n_blocks + b]);
    }
    float sink_l2 = -INFINITY;
    if (has_sink) {
        sink_l2 = sinks[row % kHq] * log2e;
    }
    mx = simd_max(mx);
    if (lane == 0u) {
        lds_max[simd_id] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_all = sink_l2;
    for (uint i = 0u; i < kThreads / 32u; ++i) {
        m_all = max(m_all, lds_max[i]);
    }

    // Normaliser. Every thread computes it; it is a handful of blocks.
    float denom = 0.0f;
    if (m_all != -INFINITY) {
        for (uint b = 0u; b < n_blocks; ++b) {
            const ulong cell = (ulong)row * n_blocks + b;
            denom += exp2(part_m[cell] - m_all) * part_l[cell];
        }
        if (has_sink) {
            denom += exp2(sink_l2 - m_all);
        }
    }

    for (uint d = tid; d < kDv; d += kThreads) {
        float cold_acc = 0.0f;
        float dense_acc = 0.0f;
        if (m_all != -INFINITY) {
            for (uint b = 0u; b < n_blocks; ++b) {
                const ulong cell = (ulong)row * n_blocks + b;
                const float w = exp2(part_m[cell] - m_all);
                const float a = part_acc[cell * kDv + d];
                if (b < blocks_cold) {
                    cold_acc = fma(w, a, cold_acc);
                } else {
                    dense_acc = fma(w, a, dense_acc);
                }
            }
        }
        lds_cold[d] = cold_acc;
        lds_dense[d] = dense_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < kDv; d += kThreads) {
        float rotated_back = 0.0f;
        if (blocks_cold > 0u) {
            for (uint j = 0u; j < kDv; ++j) {
                rotated_back = fma(lds_cold[j], inv_rot[(ulong)j * kDv + d], rotated_back);
            }
        }
        tq_store(out + (ulong)row * kDv + d,
                 denom > 0.0f ? (rotated_back + lds_dense[d]) / denom : 0.0f);
    }
)";

// The merge writes in the queries' dtype, which is not known to the source.
static const char* TURBOQUANT_ATTENTION_HEADER = R"(
    template <typename T, typename U>
    METAL_FUNC void tq_store(device T* destination, U value) {
        *destination = static_cast<T>(value);
    }
)";

static std::string dtype_tag(mlx::core::Dtype dt) {
    using namespace mlx::core;
    if (dt == float32) return "f32";
    if (dt == float16) return "f16";
    if (dt == bfloat16) return "bf16";
    throw std::invalid_argument("[turboquant_attention] only f32, f16 and bf16 inputs are supported");
}


static mlx::core::fast::CustomKernelFunction& get_attention_partials_kernel(const std::string& tag) {
    static std::mutex mutex;
    static std::map<std::string, mlx::core::fast::CustomKernelFunction> kernels;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = kernels.find(tag);
    if (it == kernels.end()) {
        it = kernels.emplace(tag, mlx::core::fast::metal_kernel(
            "turboquant_attention_partials_" + tag,
            {"q", "q_rot", "k_packed", "k_scale", "v_packed", "v_scale",
             "cb_k", "cb_v", "dense_k", "dense_v", "params"},
            {"part_acc", "part_m", "part_l"},
            TURBOQUANT_ATTENTION_PARTIALS_SOURCE,
            "",
            true,
            false)).first;
    }
    return it->second;
}

static mlx::core::fast::CustomKernelFunction& get_attention_merge_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_merge",
        {"part_acc", "part_m", "part_l", "inv_rot", "sinks", "params"},
        {"out"},
        TURBOQUANT_ATTENTION_MERGE_SOURCE,
        TURBOQUANT_ATTENTION_HEADER,
        true,
        false);
    return kernel;
}

extern "C" {

int mlx_inline_turboquant_attention_partials(
    mlx_inline_array*       out_acc,
    mlx_inline_array*       out_m,
    mlx_inline_array*       out_l,
    const mlx_inline_array* q,
    const mlx_inline_array* q_rot,
    const mlx_inline_array* k_packed,
    const mlx_inline_array* k_scale,
    const mlx_inline_array* v_packed,
    const mlx_inline_array* v_scale,
    const mlx_inline_array* cb_k,
    const mlx_inline_array* cb_v,
    const mlx_inline_array* dense_k,
    const mlx_inline_array* dense_v,
    const mlx_inline_array* params,
    uint32_t                dim,
    uint32_t                value_dim,
    uint32_t                key_bits,
    uint32_t                value_bits,
    uint32_t                kv_heads,
    uint32_t                groups,
    uint32_t                n_rows,
    uint32_t                n_kv_rows,
    uint32_t                n_blocks)
{
    using namespace mlx::core;
    try {
        if (dim == 0 || value_dim == 0 || key_bits == 0 || key_bits > 8 || value_bits == 0 ||
            value_bits > 8 || kv_heads == 0 || groups == 0 || n_rows == 0 || n_kv_rows == 0 ||
            n_blocks == 0) {
            throw std::invalid_argument("[turboquant_attention_partials] a size is zero or out of range");
        }
        if (n_rows != n_kv_rows / kv_heads * kv_heads * groups || n_kv_rows % kv_heads != 0) {
            throw std::invalid_argument("[turboquant_attention_partials] row counts disagree with heads");
        }
        if (dim % 4 != 0) {
            throw std::invalid_argument("[turboquant_attention_partials] the key dimension must be a multiple of 4");
        }
        const size_t lds_bytes =
            4 * ((size_t)groups * dim + ((size_t)1 << key_bits) + ((size_t)1 << value_bits));
        if (lds_bytes > 32 * 1024) {
            throw std::invalid_argument(
                "[turboquant_attention_partials] the queries of one KV head do not fit in threadgroup memory");
        }
        const array& qa = as_arr(q);
        const array& da = as_arr(dense_k);
        const std::string tag = dtype_tag(qa.dtype()) + "_" + dtype_tag(da.dtype()) + "_" +
                                dtype_tag(as_arr(dense_v).dtype());
        auto outputs = get_attention_partials_kernel(tag)(
            {qa, as_arr(q_rot), as_arr(k_packed), as_arr(k_scale), as_arr(v_packed), as_arr(v_scale),
             as_arr(cb_k), as_arr(cb_v), da, as_arr(dense_v), as_arr(params)},
            {{(int)n_rows, (int)n_blocks, (int)value_dim}, {(int)n_rows, (int)n_blocks}, {(int)n_rows, (int)n_blocks}},
            {float32, float32, float32},
            {32 * (int)n_blocks, (int)n_kv_rows, 1},
            {32, 1, 1},
            {{"DIM", (int)dim}, {"VDIM", (int)value_dim}, {"KBITS", (int)key_bits},
             {"VBITS", (int)value_bits}, {"GROUPS", (int)groups}, {"KV_HEADS", (int)kv_heads}},
            std::nullopt, false, {});
        new (out_acc->buf) array(outputs[0]);
        new (out_m->buf) array(outputs[1]);
        new (out_l->buf) array(outputs[2]);
        pmetal_bridge_clear_error_internal();
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("turboquant_attention_partials", e.what());
    } catch (...) {
        pmetal_bridge_set_last_error("turboquant_attention_partials", "unknown C++ exception");
    }
    bridge_placeholder(out_acc);
    bridge_placeholder(out_m);
    bridge_placeholder(out_l);
    return 1;
}

int mlx_inline_turboquant_attention_merge(
    mlx_inline_array*       out,
    const mlx_inline_array* part_acc,
    const mlx_inline_array* part_m,
    const mlx_inline_array* part_l,
    const mlx_inline_array* inv_rot,
    const mlx_inline_array* sinks,
    const mlx_inline_array* params,
    uint32_t                value_dim,
    uint32_t                q_heads,
    uint32_t                n_rows,
    uint32_t                n_blocks,
    int                     out_dtype)
{
    using namespace mlx::core;
    try {
        if (value_dim == 0 || q_heads == 0 || n_rows == 0 || n_blocks == 0) {
            throw std::invalid_argument("[turboquant_attention_merge] a size is zero");
        }
        const Dtype dt = dtype_from_int(out_dtype);
        (void)dtype_tag(dt);
        auto outputs = get_attention_merge_kernel()(
            {as_arr(part_acc), as_arr(part_m), as_arr(part_l), as_arr(inv_rot), as_arr(sinks), as_arr(params)},
            {{(int)n_rows, (int)value_dim}},
            {dt},
            {128 * (int)n_rows, 1, 1},
            {128, 1, 1},
            {{"VDIM", (int)value_dim}, {"Q_HEADS", (int)q_heads}},
            std::nullopt, false, {});
        new (out->buf) array(outputs[0]);
        pmetal_bridge_clear_error_internal();
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("turboquant_attention_merge", e.what());
    } catch (...) {
        pmetal_bridge_set_last_error("turboquant_attention_merge", "unknown C++ exception");
    }
    bridge_placeholder(out);
    return 1;
}

} // extern "C"
