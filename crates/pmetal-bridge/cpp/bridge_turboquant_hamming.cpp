// TurboQuant 1-bit Hamming pre-filter kernels (Phase F skip-list).
//
// At very long context (>= 32K cold history) the dot-product score kernels
// are bandwidth-bound on the cold key store. A 1-bit sign-hash pre-filter
// approximates the angular distance between the rotated query and each
// cached rotated key via XOR + popcount: sign-hash Hamming distance is
// monotonically related to angular distance, which monotonically tracks
// dot-product magnitude on unit-normalized inputs. So picking the M slots
// with the smallest Hamming distance yields a high-recall candidate set,
// and exact attention only runs on those M.

#include "bridge_turboquant_internal.h"

// XOR + popcount Hamming distances between a per-q-row query sign hash and
// a [kv_rows, S, packed_dim] cache of key sign hashes.
//
//   query_signs:    [N, packed_dim] uint32        (N = q_rows = batch*q_heads)
//   key_signs:      [kv_rows, S, packed_dim] uint32  (kv_rows = batch*kv_heads)
//   out_distances:  [N, S] uint32                  (Hamming distance, 0..D)
//
// `groups` (= q_heads / kv_heads) lets a single kv row serve `groups`
// consecutive query rows for GQA. Non-GQA models pass groups=1, in which
// case row → kv_row is identity and behaviour matches the original kernel.
//
// 1 thread per (row, slot). packed_dim is small (4 for D=128, 8 for D=256)
// so the inner loop fits in registers; no shared memory needed.
static const char* TURBOQUANT_HAMMING_DISTANCES_SOURCE = R"(
    uint slot = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (row >= n_rows || slot >= n_seq) return;

    uint kv_row = row / groups;
    uint key_base = (kv_row * n_seq + slot) * packed_dim;
    uint qry_base = row * packed_dim;
    uint dist = 0u;
    for (uint w = 0u; w < packed_dim; ++w) {
        uint xor_w = key_signs[key_base + w] ^ query_signs[qry_base + w];
        dist += popcount(xor_w);
    }
    out_distances[row * n_seq + slot] = dist;
)";

static mlx::core::fast::CustomKernelFunction& get_turboquant_hamming_distances_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_hamming_distances",
        {"query_signs", "key_signs"},
        {"out_distances"},
        TURBOQUANT_HAMMING_DISTANCES_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

extern "C" {

int mlx_inline_turboquant_hamming_distances(
    mlx_inline_array*       out,
    const mlx_inline_array* query_signs,
    const mlx_inline_array* key_signs,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                groups)
{
    using namespace mlx::core;

    if (packed_dim == 0 || n_rows == 0 || n_seq == 0 || groups == 0) return 1;
    if ((n_rows % groups) != 0) return 1;

    try {
        auto& kernel = get_turboquant_hamming_distances_kernel();
        // Threadgroup x-extent is capped at 256 so we don't exceed Metal's
        // 1024-thread tg limit for very long sequences (S can reach 100K+).
        int tg_x = (int)std::min(n_seq, 256u);
        auto outputs = kernel(
            {as_arr(query_signs), as_arr(key_signs)},
            {{(int)n_rows, (int)n_seq}},
            {uint32},
            {(int)n_seq, (int)n_rows, 1},
            {tg_x, 1, 1},
            {{"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows},
             {"n_seq", (int)n_seq},
             {"groups", (int)groups}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("turboquant_hamming_distances", e.what());
        return 1;
    } catch (...) {
        pmetal_bridge_set_last_error("turboquant_hamming_distances", "unknown C++ exception");
        return 1;
    }
}

}  // extern "C"
