//  dw_gemm.metal
//  Tiled fp32 SGEMM for weight gradient accumulation in ANE training.
//
//  Computes: C = alpha * A @ B^T + beta * C
//
//  All 7 per-layer dW GEMMs (+ the embedding gradient GEMM) are encoded into a
//  single Metal command buffer, executing on the GPU while the ANE handles dx
//  propagation.
//
//  Tile: BM = 64, BN = 64, BK = 16. Four simdgroups (128 threads) in a 2 x 2
//  grid, each owning a 32 x 32 sub-tile held as sixteen 8 x 8 fragments and
//  fed by `simdgroup_multiply_accumulate`. A and B are both stored
//  K-contiguous (A is [M, K], B is [N, K]), so both tiles are staged as
//  [row][k] with a padded row stride, and B^T fragments are read through the
//  transposed index. The next k-block is fetched into registers while the
//  current one is multiplied.
//
//  Fragments live in registers as per-lane float2 halves and are wrapped in
//  `simdgroup_float8x8` only around each multiply-accumulate. Holding them as
//  `simdgroup_float8x8` arrays loaded with `simdgroup_load` runs the same
//  arithmetic about seven times slower.
//
//  Edge tiles: every load is zero-filled outside [M, K] / [N, K] and every
//  store is guarded, so M, N and K may be anything. With beta == 0 the old
//  contents of C are never read (a fresh buffer may hold anything).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 16;
// Row stride of the staged tiles, padded past BK so the fragment loads of
// consecutive rows land in different threadgroup memory banks.
constant uint LD = BK + 4;
constant uint THREADS = 128;
// Staged A + B, reused for the BM x BN output tile in the epilogue.
constant uint SMEM_FLOATS = (2 * BM * LD > BM * BN) ? (2 * BM * LD) : (BM * BN);

struct DwGemmParams {
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
};

// A BM x BK (or BN x BK) tile is BM * BK / 4 float4 chunks, split evenly over
// the threadgroup.
constant uint CHUNKS_PER_ROW = BK / 4;
constant uint CHUNKS_PER_THREAD = (BM * CHUNKS_PER_ROW) / THREADS;

/// Fetch this thread's chunks of the tile of a row-major [total_rows, K]
/// matrix at (`row0`, `k0`) into registers, zero-filled out of range.
inline void fetch_tile(
    thread float4 (&regs)[CHUNKS_PER_THREAD],
    device const float* src,
    uint total_rows,
    uint row0,
    uint k0,
    uint K,
    uint tid
) {
    const bool vec = (K & 3u) == 0u;
    #pragma unroll
    for (uint n = 0; n < CHUNKS_PER_THREAD; n++) {
        const uint i = tid + n * THREADS;
        const uint r = i / CHUNKS_PER_ROW;
        const uint c = (i % CHUNKS_PER_ROW) * 4u;
        const uint gr = row0 + r;
        const uint gk = k0 + c;
        float4 v = float4(0.0f);
        if (gr < total_rows) {
            if (vec) {
                if (gk < K) {
                    v = *reinterpret_cast<device const float4*>(src + gr * K + gk);
                }
            } else {
                const device float* p = src + gr * K;
                if (gk + 0 < K) v.x = p[gk + 0];
                if (gk + 1 < K) v.y = p[gk + 1];
                if (gk + 2 < K) v.z = p[gk + 2];
                if (gk + 3 < K) v.w = p[gk + 3];
            }
        }
        regs[n] = v;
    }
}

/// Write this thread's fetched chunks into the staged tile (row stride LD).
inline void stage_tile(
    threadgroup float* dst,
    thread const float4 (&regs)[CHUNKS_PER_THREAD],
    uint tid
) {
    #pragma unroll
    for (uint n = 0; n < CHUNKS_PER_THREAD; n++) {
        const uint i = tid + n * THREADS;
        const uint r = i / CHUNKS_PER_ROW;
        const uint c = (i % CHUNKS_PER_ROW) * 4u;
        *reinterpret_cast<threadgroup float4*>(dst + r * LD + c) = regs[n];
    }
}

/// One lane's coordinates inside an 8 x 8 simdgroup fragment: the lane owns
/// the two elements (row, col) and (row, col + 1).
inline short2 frag_coord(ushort lane) {
    const short qid = lane / 4;
    const short row = (qid & 4) + ((lane / 2) % 4);
    const short col = (qid & 2) * 2 + (lane % 2) * 2;
    return short2(col, row);
}

/// D = A * B + C on lane-resident fragment halves.
inline float2 frag_mma(float2 a, float2 b, float2 c) {
    simdgroup_float8x8 am, bm, cm, dm;
    reinterpret_cast<thread float2&>(am.thread_elements()) = a;
    reinterpret_cast<thread float2&>(bm.thread_elements()) = b;
    reinterpret_cast<thread float2&>(cm.thread_elements()) = c;
    simdgroup_multiply_accumulate(dm, am, bm, cm);
    return reinterpret_cast<thread float2&>(dm.thread_elements());
}

kernel void dw_gemm_accum(
    device const float* A         [[buffer(0)]],   // [M, K] row-major
    device const float* B         [[buffer(1)]],   // [N, K] row-major (transposed on read)
    device float*       C         [[buffer(2)]],   // [M, N] row-major, read-modify-write
    constant DwGemmParams& params [[buffer(3)]],
    uint2  tg_id [[threadgroup_position_in_grid]],
    uint   tid   [[thread_index_in_threadgroup]],
    ushort sg    [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]
) {
    threadgroup float smem[SMEM_FLOATS];
    threadgroup float* As = smem;
    threadgroup float* Bs = smem + BM * LD;

    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    // grid.x walks N tiles, grid.y walks M tiles.
    const uint tile_row = tg_id.y * BM;
    const uint tile_col = tg_id.x * BN;

    // This simdgroup's 32 x 32 sub-tile inside the 64 x 64 tile.
    const short sg_row = (sg >> 1) * 32;
    const short sg_col = (sg & 1) * 32;

    const short2 rc = frag_coord(lane);  // (col, row) within a fragment
    // Offsets of this lane's two A elements (m = sg_row + rc.y, k = rc.x + {0,1})
    // and B^T elements (k = rc.y, n = sg_col + rc.x + {0,1}) in the staged tiles.
    const short a_off = (sg_row + rc.y) * LD + rc.x;
    const short b_off = (sg_col + rc.x) * LD + rc.y;

    float2 acc[4][4];
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = float2(0.0f);
        }
    }

    // The next k-block is fetched into registers while the current one is
    // multiplied, so the global-memory latency hides behind the MMAs.
    float4 a_regs[CHUNKS_PER_THREAD];
    float4 b_regs[CHUNKS_PER_THREAD];
    fetch_tile(a_regs, A, M, tile_row, 0, K, tid);
    fetch_tile(b_regs, B, N, tile_col, 0, K, tid);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        stage_tile(As, a_regs, tid);
        stage_tile(Bs, b_regs, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < K) {
            fetch_tile(a_regs, A, M, tile_row, k0 + BK, K, tid);
            fetch_tile(b_regs, B, N, tile_col, k0 + BK, K, tid);
        }

        #pragma unroll
        for (uint kk = 0; kk < BK; kk += 8) {
            float2 a[4];
            float2 b[4];
            #pragma unroll
            for (uint i = 0; i < 4; i++) {
                const threadgroup float* p = As + a_off + (i * 8) * LD + kk;
                a[i] = float2(p[0], p[1]);
            }
            #pragma unroll
            for (uint j = 0; j < 4; j++) {
                const threadgroup float* p = Bs + b_off + (j * 8) * LD + kk;
                b[j] = float2(p[0], p[LD]);
            }
            #pragma unroll
            for (uint i = 0; i < 4; i++) {
                #pragma unroll
                for (uint j = 0; j < 4; j++) {
                    const uint jj = (i & 1u) ? (3u - j) : j;
                    acc[i][jj] = frag_mma(a[i], b[jj], acc[i][jj]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Epilogue: C = alpha * acc + beta * C, through threadgroup memory ----
    threadgroup float* Cs = smem;
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            threadgroup float* p = Cs + (sg_row + i * 8 + rc.y) * BN + sg_col + j * 8 + rc.x;
            p[0] = acc[i][j].x;
            p[1] = acc[i][j].y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float alpha = params.alpha;
    const float beta  = params.beta;
    const bool vec_n = (N & 3u) == 0u;

    for (uint i = tid; i < BM * (BN / 4); i += THREADS) {
        const uint r = i / (BN / 4);
        const uint c = (i % (BN / 4)) * 4u;
        const uint gr = tile_row + r;
        const uint gc = tile_col + c;
        if (gr >= M || gc >= N) continue;

        const float4 v = *reinterpret_cast<threadgroup const float4*>(Cs + r * BN + c);
        device float* out = C + gr * N + gc;
        if (vec_n) {
            float4 result = alpha * v;
            if (beta != 0.0f) {
                result += beta * *reinterpret_cast<device const float4*>(out);
            }
            *reinterpret_cast<device float4*>(out) = result;
        } else {
            const float lane_v[4] = {v.x, v.y, v.z, v.w};
            for (uint e = 0; e < 4 && gc + e < N; e++) {
                float result = alpha * lane_v[e];
                if (beta != 0.0f) {
                    result += beta * out[e];
                }
                out[e] = result;
            }
        }
    }
}
