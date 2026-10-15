// Cut cross-entropy, fused: an LM head's per-token logsumexp and target logit
// straight from the hidden states, with the [tokens, vocab] logits never in
// device memory, forward or backward.
//
// Wijmans et al., "Cut Your Losses in Large-Vocabulary Language Models"
// (arXiv 2411.09009) and Apple's reference implementation
// (github.com/apple/ml-cross-entropy). The kernels below follow its structure
// on Metal's simdgroup matrices, tiles of 64 tokens by 64 vocabulary rows,
// with 8-token groups as the unit of gradient filtering:
//
//   forward   one threadgroup per token tile and vocabulary range. Each tile
//             of logits comes out of the head GEMM in f32 and is rounded once
//             to the head's dtype (as the model's own head rounds it), then
//             bias, scale and softcap run in f32, followed by an online
//             logsumexp per token, the target logit (read out of the tile
//             that holds it) and each 8-token group's largest logit in the
//             tile.
//   backward  the softmax is negligible almost everywhere. A group whose
//             largest logit in a tile sits more than ln(1/eps) under the
//             smallest logsumexp among its tokens has nothing in that tile
//             above `eps` (the paper's gradient filtering, decided from the
//             forward's record rather than after recomputing). The groups
//             that survive are recomputed and their logit gradient written to
//             a scratch one vocabulary chunk wide. Two block-sparse GEMMs
//             reduce it into the hidden states' and the head's gradients,
//             leaving out every group whose recomputed softmax is under `eps`
//             after all.
//
// The whole thing is one `custom_vjp`, so autograd sees a differentiable op:
// gradients reach the hidden states, the head (and through it any adapter
// folded into it), and the bias. None of the Metal kernels has a VJP of its
// own, so a backward that is itself being differentiated (a second
// derivative) is built from MLX ops instead.

#include "bridge_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mx = mlx::core;

namespace {

// Rows and columns of every tile, threads per threadgroup (four simdgroups,
// two by two over the tile), and tokens per filtering group, which is one
// 8-row simdgroup matrix.
constexpr int kTile = 64;
constexpr int kThreads = 128;
constexpr int kGroup = 8;
// Depth of one staged slice of the head GEMM, as MLX's own GEMM stages a
// 64x64 bf16 tile on this class of GPU. Deeper slices take more threadgroup
// memory, which costs occupancy.
constexpr int kDepth = 16;
// Threadgroups the forward aims for: enough to fill the GPU several times
// over, so each token tile's vocabulary is split into ranges.
constexpr int kForwardGroups = 2048;

const char* kHeader = R"METAL(
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

#define CCE_TILE 64
#define CCE_THREADS 128
#define CCE_UNROLL _Pragma("clang loop unroll(full)")

typedef simdgroup_matrix<float, 8, 8> cce_mat;

// Where a lane's two elements of an 8x8 simdgroup matrix sit: (column, row),
// the second one column to the right.
inline short2 cce_coord(uint lane) {
  const short qid = short(lane / 4);
  const short row = (qid & 4) + short((lane / 2) % 4);
  const short col = (qid & 2) * 2 + short(lane % 2) * 2;
  return short2(col, row);
}

// Eight consecutive elements into threadgroup memory: one vector copy from
// device memory, element by element from constant memory (where MLX puts an
// input of fewer than eight elements).
template <typename T>
struct cce_vec8 {
  T v[8];
};
template <typename T>
inline void cce_copy8(threadgroup T* d, const device T* s) {
  *((threadgroup cce_vec8<T>*)d) = *((const device cce_vec8<T>*)s);
}
template <typename T>
inline void cce_copy8(threadgroup T* d, const constant T* s) {
  CCE_UNROLL
  for (int e = 0; e < 8; ++e) {
    d[e] = s[e];
  }
}

// Copy a ROWS x COLS block of a row-major matrix (leading dimension `ld`) into
// threadgroup memory (leading dimension LD), zero outside `rows` x `cols`.
template <typename T, int ROWS, int COLS, int LD, typename Src>
inline void cce_stage(threadgroup T* dst, Src src, int ld, int rows, int cols, uint tid) {
  constexpr int VEC = 8;
  constexpr int PER_ROW = COLS / VEC;
  constexpr int ROWS_PER_PASS = CCE_THREADS / PER_ROW;
  const int c = (int(tid) % PER_ROW) * VEC;
  CCE_UNROLL
  for (int r = int(tid) / PER_ROW; r < ROWS; r += ROWS_PER_PASS) {
    threadgroup T* d = dst + r * LD + c;
    if (r < rows && c + VEC <= cols) {
      cce_copy8(d, src + size_t(r) * ld + c);
    } else {
      CCE_UNROLL
      for (int e = 0; e < VEC; ++e) {
        d[e] = (r < rows && c + e < cols) ? T(src[size_t(r) * ld + c + e]) : T(0);
      }
    }
  }
}

// acc[i][j] += A * B over DEPTH, for the 32x32 block of a 64x64 output that
// simdgroup (wm, wn) owns: rows wm*32 + i*8, columns wn*32 + j*8. A(r, k) is
// a[r * A_R + k * A_K] and B(k, n) is b[k * B_K + n * B_N]. Row fragment i
// is left out when bit i of `rows` is clear, and the 8-deep step s when bit
// s of `steps` is clear; both masks are the same across a simdgroup, and
// only a MASKED instance reads them (the branches cost the full product).
template <int DEPTH, int A_R, int A_K, int B_K, int B_N, bool MASKED = false, typename TA,
          typename TB>
inline void cce_mma(thread cce_mat (&acc)[4][4], const threadgroup TA* a,
                    const threadgroup TB* b, short wm, short wn, short2 lc,
                    uint rows = 0xF, uint steps = 0xFF) {
  CCE_UNROLL
  for (short kk = 0; kk < DEPTH; kk += 8) {
    if (MASKED && (steps & (1u << (kk / 8))) == 0) {
      continue;
    }
    cce_mat am[4];
    cce_mat bm[4];
    CCE_UNROLL
    for (short i = 0; i < 4; ++i) {
      if (MASKED && (rows & (1u << i)) == 0) {
        continue;
      }
      const int r = wm * 32 + i * 8 + lc.y;
      const int k = kk + lc.x;
      am[i].thread_elements()[0] = float(a[r * A_R + k * A_K]);
      am[i].thread_elements()[1] = float(a[r * A_R + (k + 1) * A_K]);
    }
    CCE_UNROLL
    for (short j = 0; j < 4; ++j) {
      const int k = kk + lc.y;
      const int n = wn * 32 + j * 8 + lc.x;
      bm[j].thread_elements()[0] = float(b[k * B_K + n * B_N]);
      bm[j].thread_elements()[1] = float(b[k * B_K + (n + 1) * B_N]);
    }
    CCE_UNROLL
    for (short i = 0; i < 4; ++i) {
      if (MASKED && (rows & (1u << i)) == 0) {
        continue;
      }
      CCE_UNROLL
      for (short j = 0; j < 4; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
      }
    }
  }
}

// One 64x64 tile of the head GEMM, x[64, K] * w[64, K]^T, in f32, for the row
// fragments `rows` selects (when MASKED).
template <typename TX, typename TW, int BK, bool MASKED = false, typename PX, typename PW>
inline void cce_logits(thread cce_mat (&acc)[4][4], PX x, PW w, int K,
                       int rows_x, int rows_w, threadgroup TX* xs,
                       threadgroup TW* ws, uint tid, short wm, short wn,
                       short2 lc, uint rows = 0xF) {
  constexpr int LDX = BK + 16 / sizeof(TX);
  constexpr int LDW = BK + 16 / sizeof(TW);
  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    CCE_UNROLL
    for (short j = 0; j < 4; ++j) {
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cce_stage<TX, CCE_TILE, BK, LDX>(xs, x + k0, K, rows_x, K - k0, tid);
    cce_stage<TW, CCE_TILE, BK, LDW>(ws, w + k0, K, rows_w, K - k0, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cce_mma<BK, LDX, 1, 1, LDW, MASKED>(acc, xs, ws, wm, wn, lc, rows);
  }
}

// The logit the loss sees, from the GEMM's f32 accumulator: rounded to the
// head's output dtype as the model's head produces it, plus the bias (added in
// that dtype, as the model adds it), times `scale`, softcapped. `dz` is its
// derivative with respect to the rounded head output.
template <typename TO, bool BIAS, bool SOFTCAP>
inline float cce_logit(float acc, float bias, float scale, float cap, thread float& dz) {
  float z = float(TO(acc));
  if (BIAS) {
    z = float(TO(z + bias));
  }
  z *= scale;
  dz = scale;
  if (SOFTCAP) {
    const float t = precise::tanh(z / cap);
    z = cap * t;
    dz = scale * (1.0f - t * t);
  }
  return z;
}

// Fold a second (max, sum of exp) pair into the first.
inline void cce_merge(thread float& m, thread float& s, float om, float os) {
  const float mn = max(m, om);
  s = s * fast::exp(m - mn) + os * fast::exp(om - mn);
  m = mn;
}

// The live 8-token groups of a 64-token tile in a [groups, tiles] mask, one
// bit each.
template <typename P>
inline uint cce_groups(P live, int tm, int tiles, int tile) {
  uint mask = 0;
  CCE_UNROLL
  for (int s = 0; s < 8; ++s) {
    mask |= live[(tm * 8 + s) * tiles + tile] ? (1u << s) : 0u;
  }
  return mask;
}
)METAL";

// Inputs: x [N, K], w [V, K], targets [N], bias [V] (or one ignored element),
// dims {N, V, K, TV, tiles per split}, fparams {scale, cap}.
// Outputs: part_m, part_s [2 * splits, N] (each simdgroup column's running
// max and sum of exp), target [N] (zero where no column matches), gmax
// [8 * TM, TV] (each 8-token group's largest logit in each vocabulary tile,
// +inf where the tile holds one of the group's targets).
const char* kForwardSource = R"METAL(
  const int N = dims[0];
  const int V = dims[1];
  const int K = dims[2];
  const int TV = dims[3];
  const int TPS = dims[4];
  const float scale = fparams[0];
  const float cap = fparams[1];

  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const short wm = short(sg / 2);
  const short wn = short(sg % 2);
  const short2 lc = cce_coord(lane);
  const int tm = int(threadgroup_position_in_grid.x);
  const int split = int(threadgroup_position_in_grid.y);
  const int m0 = tm * CCE_TILE;
  const int rows_x = min(CCE_TILE, N - m0);

  constexpr int LDX = BK + 16 / sizeof(TX);
  constexpr int LDW = BK + 16 / sizeof(TW);
  threadgroup TX xs[CCE_TILE * LDX];
  threadgroup TW ws[CCE_TILE * LDW];
  threadgroup float gred[2][4][2];

  int row[4];
  int tgt[4];
  float run_m[4];
  float run_s[4];
  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    row[i] = m0 + wm * 32 + i * 8 + lc.y;
    tgt[i] = row[i] < N ? int(targets[row[i]]) : -1;
    run_m[i] = -FLT_MAX;
    run_s[i] = 0.0f;
  }

  const int tv_end = min(TV, (split + 1) * TPS);
  for (int tv = split * TPS; tv < tv_end; ++tv) {
    const int v0 = tv * CCE_TILE;
    cce_mat acc[4][4];
    cce_logits<TX, TW, BK>(acc, x + size_t(m0) * K, w + size_t(v0) * K, K,
                           rows_x, min(CCE_TILE, V - v0), xs, ws, tid, wm, wn, lc);
    CCE_UNROLL
    for (short i = 0; i < 4; ++i) {
      float z[8];
      float tmax = -INFINITY;
      bool hit = false;
      CCE_UNROLL
      for (short j = 0; j < 4; ++j) {
        CCE_UNROLL
        for (short e = 0; e < 2; ++e) {
          const int col = v0 + wn * 32 + j * 8 + lc.x + e;
          float dz;
          const float v = col < V
              ? cce_logit<TO, BIAS, SOFTCAP>(acc[i][j].thread_elements()[e],
                                             BIAS ? float(bias[col]) : 0.0f,
                                             scale, cap, dz)
              : -INFINITY;
          z[j * 2 + e] = v;
          tmax = max(tmax, v);
          if (col == tgt[i]) {
            target[row[i]] = v;
            hit = true;
          }
        }
      }
      const float gm = simd_max(tmax);
      const bool ghit = simd_any(hit);
      if (lane == 0) {
        gred[wm][i][wn] = ghit ? INFINITY : gm;
      }
      const float mn = max(run_m[i], tmax);
      float s = run_s[i] * fast::exp(run_m[i] - mn);
      CCE_UNROLL
      for (short k = 0; k < 8; ++k) {
        s += fast::exp(z[k] - mn);
      }
      run_m[i] = mn;
      run_s[i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
      gmax[size_t(tm * 8 + int(tid)) * TV + tv] =
          max(gred[tid / 4][tid % 4][0], gred[tid / 4][tid % 4][1]);
    }
  }

  // The four lanes that hold a row's columns differ in lane bits 0 and 3.
  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    cce_merge(run_m[i], run_s[i], simd_shuffle_xor(run_m[i], 1),
              simd_shuffle_xor(run_s[i], 1));
    cce_merge(run_m[i], run_s[i], simd_shuffle_xor(run_m[i], 8),
              simd_shuffle_xor(run_s[i], 8));
    if ((lane & 9) == 0 && row[i] < N) {
      const size_t p = size_t(split * 2 + wn) * N + row[i];
      part_m[p] = run_m[i];
      part_s[p] = run_s[i];
    }
  }
)METAL";

// The logit gradient of one 64x64 tile of a vocabulary chunk, into `dz`, for
// the tile's live groups.
//
// Inputs: x, w, targets, bias as the forward; lse, glse, gt [N] (the
// cotangents of each token's logsumexp and target logit); bound [8 * TM, TVC]
// (the forward's verdict per group and tile of this chunk); dims {N, V, K,
// TVC, first tile of the chunk}; fparams {scale, cap, eps}.
// Outputs: dz [64 * TM, 64 * TVC] in the head's dtype, written for the
// groups the bound keeps; live [8 * TM, TVC] (whether anything in the group's
// tile reaches `eps`, or it holds a target).
const char* kGradSource = R"METAL(
  const int N = dims[0];
  const int V = dims[1];
  const int K = dims[2];
  const int TVC = dims[3];
  const int tv0 = dims[4];
  const int SCW = TVC * CCE_TILE;
  const float scale = fparams[0];
  const float cap = fparams[1];
  const float eps = fparams[2];

  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const short wm = short(sg / 2);
  const short wn = short(sg % 2);
  const short2 lc = cce_coord(lane);
  const int tm = int(threadgroup_position_in_grid.x);
  const int tl = int(threadgroup_position_in_grid.y);
  threadgroup bool lred[2][4][2];

  const uint groups = cce_groups(bound, tm, TVC, tl);
  if (groups == 0) {
    if (tid < 8) {
      live[(tm * 8 + int(tid)) * TVC + tl] = false;
    }
    return;
  }
  const uint rows = (groups >> (wm * 4)) & 0xF;

  const int m0 = tm * CCE_TILE;
  const int v0 = (tv0 + tl) * CCE_TILE;
  constexpr int LDX = BK + 16 / sizeof(TX);
  constexpr int LDW = BK + 16 / sizeof(TW);
  threadgroup TX xs[CCE_TILE * LDX];
  threadgroup TW ws[CCE_TILE * LDW];

  cce_mat acc[4][4];
  // The same branch for the whole threadgroup: the GEMM holds barriers.
  if (groups == 0xFF) {
    cce_logits<TX, TW, BK>(acc, x + size_t(m0) * K, w + size_t(v0) * K, K,
                           min(CCE_TILE, N - m0), min(CCE_TILE, V - v0), xs, ws,
                           tid, wm, wn, lc);
  } else {
    cce_logits<TX, TW, BK, true>(acc, x + size_t(m0) * K, w + size_t(v0) * K, K,
                                 min(CCE_TILE, N - m0), min(CCE_TILE, V - v0), xs,
                                 ws, tid, wm, wn, lc, rows);
  }

  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    bool any_live = false;
    const int r = m0 + wm * 32 + i * 8 + lc.y;
    device TO* out = dz + size_t(r) * SCW + tl * CCE_TILE + wn * 32 + lc.x;
    if ((rows & (1u << i)) == 0) {
      // A group the bound drops: zeros, so the tile's scratch is all
      // defined (the reductions skip these rows anyway).
      CCE_UNROLL
      for (short j = 0; j < 4; ++j) {
        out[j * 8] = TO(0);
        out[j * 8 + 1] = TO(0);
      }
    } else {
      const bool in = r < N;
      const float l = in ? lse[r] : 0.0f;
      const float gl = in ? glse[r] : 0.0f;
      const float gtr = in ? gt[r] : 0.0f;
      const int t = in ? int(targets[r]) : -1;
      CCE_UNROLL
      for (short j = 0; j < 4; ++j) {
        CCE_UNROLL
        for (short e = 0; e < 2; ++e) {
          const int col = v0 + wn * 32 + j * 8 + lc.x + e;
          float d = 0.0f;
          if (col < V) {
            float dlogit;
            const float z = cce_logit<TO, BIAS, SOFTCAP>(
                acc[i][j].thread_elements()[e], BIAS ? float(bias[col]) : 0.0f,
                scale, cap, dlogit);
            const float p = fast::exp(z - l);
            const bool is_t = col == t;
            any_live |= (gl != 0.0f && p >= eps) || (is_t && gtr != 0.0f);
            d = (gl * p + (is_t ? gtr : 0.0f)) * dlogit;
          }
          out[j * 8 + e] = TO(d);
        }
      }
      any_live = simd_any(any_live);
    }
    if (lane == 0) {
      lred[wm][i][wn] = any_live;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8) {
    live[(tm * 8 + int(tid)) * TVC + tl] = lred[tid / 4][tid % 4][0] || lred[tid / 4][tid % 4][1];
  }
)METAL";

// The hidden states' gradient over one vocabulary chunk, added to the
// previous chunks': dx[64-token tile, 64-column K slice] += dz * w over the
// chunk's live groups in that token tile.
//
// Inputs: dz, w, live as above, prev [N, K] f32 (or one ignored element),
// dims {N, V, K, TVC, first tile of the chunk}. Output: dx [N, K] f32.
const char* kGradXSource = R"METAL(
  const int N = dims[0];
  const int V = dims[1];
  const int K = dims[2];
  const int TVC = dims[3];
  const int tv0 = dims[4];
  const int SCW = TVC * CCE_TILE;

  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const short wm = short(sg / 2);
  const short wn = short(sg % 2);
  const short2 lc = cce_coord(lane);
  const int k0 = int(threadgroup_position_in_grid.x) * CCE_TILE;
  const int tm = int(threadgroup_position_in_grid.y);
  const int m0 = tm * CCE_TILE;

  constexpr int D = 32;
  constexpr int LDA = D + 16 / sizeof(TO);
  constexpr int LDB = CCE_TILE + 16 / sizeof(TW);
  threadgroup TO as[CCE_TILE * LDA];
  threadgroup TW bs[D * LDB];

  cce_mat acc[4][4];
  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    CCE_UNROLL
    for (short j = 0; j < 4; ++j) {
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  for (int tl = 0; tl < TVC; ++tl) {
    const uint groups = cce_groups(live, tm, TVC, tl);
    if (groups == 0) {
      continue;
    }
    const uint rows = (groups >> (wm * 4)) & 0xF;
    const int v0 = (tv0 + tl) * CCE_TILE;
    CCE_UNROLL
    for (int kk = 0; kk < CCE_TILE; kk += D) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      cce_stage<TO, CCE_TILE, D, LDA>(as, dz + size_t(m0) * SCW + tl * CCE_TILE + kk,
                                      SCW, CCE_TILE, D, tid);
      cce_stage<TW, D, CCE_TILE, LDB>(bs, w + size_t(v0 + kk) * K + k0, K,
                                      V - v0 - kk, K - k0, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (rows == 0xF) {
        cce_mma<D, LDA, 1, LDB, 1>(acc, as, bs, wm, wn, lc);
      } else {
        cce_mma<D, LDA, 1, LDB, 1, true>(acc, as, bs, wm, wn, lc, rows);
      }
    }
  }
  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    const int r = m0 + wm * 32 + i * 8 + lc.y;
    if (r >= N) {
      continue;
    }
    CCE_UNROLL
    for (short j = 0; j < 4; ++j) {
      CCE_UNROLL
      for (short e = 0; e < 2; ++e) {
        const int c = k0 + wn * 32 + j * 8 + lc.x + e;
        if (c < K) {
          const size_t p = size_t(r) * K + c;
          dx[p] = acc[i][j].thread_elements()[e] + (PREV ? prev[p] : 0.0f);
        }
      }
    }
  }
)METAL";

// The head's gradient for one vocabulary chunk: dw[64-row vocabulary tile,
// 64-column K slice] = dz^T * x over the chunk's live groups in that
// vocabulary tile.
//
// Inputs: dz, x, live as above, dims {N, V, K, TVC, first tile of the chunk}.
// Output: dw [rows of the chunk, K] in the head's dtype.
const char* kGradWSource = R"METAL(
  const int N = dims[0];
  const int V = dims[1];
  const int K = dims[2];
  const int TVC = dims[3];
  const int tv0 = dims[4];
  const int SCW = TVC * CCE_TILE;
  const int TM = (N + CCE_TILE - 1) / CCE_TILE;

  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const short wm = short(sg / 2);
  const short wn = short(sg % 2);
  const short2 lc = cce_coord(lane);
  const int k0 = int(threadgroup_position_in_grid.x) * CCE_TILE;
  const int tl = int(threadgroup_position_in_grid.y);

  constexpr int D = 32;
  constexpr int LDA = CCE_TILE + 16 / sizeof(TO);
  constexpr int LDB = CCE_TILE + 16 / sizeof(TX);
  threadgroup TO as[D * LDA];
  threadgroup TX bs[D * LDB];

  cce_mat acc[4][4];
  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    CCE_UNROLL
    for (short j = 0; j < 4; ++j) {
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  for (int tm = 0; tm < TM; ++tm) {
    const uint groups = cce_groups(live, tm, TVC, tl);
    if (groups == 0) {
      continue;
    }
    const int m0 = tm * CCE_TILE;
    CCE_UNROLL
    for (int kk = 0; kk < CCE_TILE; kk += D) {
      // The tokens of this slice are its 8-deep steps, one group each.
      const uint steps = (groups >> (kk / 8)) & ((1u << (D / 8)) - 1);
      if (steps == 0) {
        continue;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      cce_stage<TO, D, CCE_TILE, LDA>(as, dz + size_t(m0 + kk) * SCW + tl * CCE_TILE,
                                      SCW, D, CCE_TILE, tid);
      cce_stage<TX, D, CCE_TILE, LDB>(bs, x + size_t(m0 + kk) * K + k0, K,
                                      N - m0 - kk, K - k0, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // A(r, k) = dz[token k, column r]: the staged block read transposed.
      if (steps == 0xF) {
        cce_mma<D, 1, LDA, LDB, 1>(acc, as, bs, wm, wn, lc);
      } else {
        cce_mma<D, 1, LDA, LDB, 1, true>(acc, as, bs, wm, wn, lc, 0xF, steps);
      }
    }
  }
  const int rows = min(SCW, V - tv0 * CCE_TILE);
  CCE_UNROLL
  for (short i = 0; i < 4; ++i) {
    const int r = tl * CCE_TILE + wm * 32 + i * 8 + lc.y;
    if (r >= rows) {
      continue;
    }
    CCE_UNROLL
    for (short j = 0; j < 4; ++j) {
      CCE_UNROLL
      for (short e = 0; e < 2; ++e) {
        const int c = k0 + wn * 32 + j * 8 + lc.x + e;
        if (c < K) {
          dw[size_t(r) * K + c] = TW(acc[i][j].thread_elements()[e]);
        }
      }
    }
  }
)METAL";

mx::fast::CustomKernelFunction& forward_kernel() {
  static auto kernel = mx::fast::metal_kernel(
      "pmetal_cce_forward",
      {"x", "w", "targets", "bias", "dims", "fparams"},
      {"part_m", "part_s", "target", "gmax"},
      kForwardSource,
      kHeader);
  return kernel;
}

mx::fast::CustomKernelFunction& grad_kernel() {
  static auto kernel = mx::fast::metal_kernel(
      "pmetal_cce_grad",
      {"x", "w", "targets", "bias", "lse", "glse", "gt", "bound", "dims", "fparams"},
      {"dz", "live"},
      kGradSource,
      kHeader);
  return kernel;
}

mx::fast::CustomKernelFunction& grad_x_kernel() {
  static auto kernel = mx::fast::metal_kernel(
      "pmetal_cce_grad_x",
      {"dz", "w", "live", "prev", "dims"},
      {"dx"},
      kGradXSource,
      kHeader);
  return kernel;
}

mx::fast::CustomKernelFunction& grad_w_kernel() {
  static auto kernel = mx::fast::metal_kernel(
      "pmetal_cce_grad_w",
      {"dz", "x", "live", "dims"},
      {"dw"},
      kGradWSource,
      kHeader);
  return kernel;
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// What the kernels need to know beyond their array inputs.
struct Spec {
  float scale;
  float softcap;
  float eps;
  // Vocabulary columns per chunk of the backward's scratch.
  int chunk;
  bool has_bias;
  // Which gradients the caller's autograd will ask for, judged from whether
  // the hidden states and the head were tracers. Only the memory schedule
  // depends on it: each gradient is built either way and evaluated if asked.
  bool want_dx;
  bool want_dw;
};

mx::Dtype head_dtype(const array& x, const array& w) {
  return mx::promote_types(x.dtype(), w.dtype());
}

// The forward: [lse, target logit, gmax].
std::vector<array> forward(const array& x, const array& w, const array& targets,
                           const array& bias, const Spec& spec) {
  const int N = x.shape(0);
  const int K = x.shape(1);
  const int V = w.shape(0);
  const int TM = ceil_div(N, kTile);
  const int TV = ceil_div(V, kTile);
  const int want_splits = std::clamp(ceil_div(kForwardGroups, TM), 1, TV);
  const int tps = ceil_div(TV, want_splits);
  const int splits = ceil_div(TV, tps);

  const array dims = array({N, V, K, TV, tps}, mx::int32);
  const array fparams = array({spec.scale, spec.softcap}, mx::float32);
  auto outs = forward_kernel()(
      {x, w, targets, bias, dims, fparams},
      {{2 * splits, N}, {2 * splits, N}, {N}, {TM * kGroup, TV}},
      {mx::float32, mx::float32, mx::float32, mx::float32},
      {TM * kThreads, splits, 1},
      {kThreads, 1, 1},
      {{"TX", x.dtype()},
       {"TW", w.dtype()},
       {"TO", head_dtype(x, w)},
       {"BK", kDepth},
       {"BIAS", spec.has_bias},
       {"SOFTCAP", spec.softcap > 0.0f}},
      0.0f,
      false,
      {});
  const array& part_m = outs[0];
  const array& part_s = outs[1];
  const array m = mx::max(part_m, 0);
  const array lse = mx::add(
      m, mx::log(mx::sum(mx::multiply(part_s, mx::exp(mx::subtract(part_m, m))), 0)));
  return {lse, outs[2], outs[3]};
}

// The logit transform of the kernels' `cce_logit`, on a whole array.
array transform(array z, const array* bias, const Spec& spec, mx::Dtype out) {
  z = mx::astype(z, out);
  if (bias != nullptr) {
    z = mx::add(z, mx::astype(*bias, out));
  }
  z = mx::astype(z, mx::float32);
  if (spec.scale != 1.0f) {
    z = mx::multiply(z, array(spec.scale));
  }
  if (spec.softcap > 0.0f) {
    z = mx::multiply(mx::tanh(mx::divide(z, array(spec.softcap))), array(spec.softcap));
  }
  return z;
}

// The backward built from MLX ops, a vocabulary chunk at a time, unfiltered:
// for a second derivative, which the Metal kernels can't give.
std::vector<array> backward_ops(const array& x, const array& w, const array& targets,
                                const array* bias, const array& lse, const array& glse,
                                const array& gt, const Spec& spec) {
  const int N = x.shape(0);
  const int K = x.shape(1);
  const int V = w.shape(0);
  const mx::Dtype out = head_dtype(x, w);
  array dx = mx::zeros({N, K}, mx::float32);
  std::vector<array> dws;
  std::vector<array> dbs;
  for (int v0 = 0; v0 < V; v0 += spec.chunk) {
    const int v1 = std::min(V, v0 + spec.chunk);
    const array wc = mx::slice(w, {v0, 0}, {v1, K});
    std::optional<array> bc;
    if (bias != nullptr) {
      bc = mx::slice(*bias, {v0}, {v1});
    }
    const array z = transform(mx::matmul(x, mx::transpose(wc)), bc ? &*bc : nullptr, spec, out);
    // d(softcap(scale * r)) / dr
    array dlogit = array(spec.scale);
    if (spec.softcap > 0.0f) {
      const array t = mx::divide(z, array(spec.softcap));
      dlogit = mx::multiply(array(spec.scale), mx::subtract(array(1.0f), mx::square(t)));
    }
    const array p = mx::exp(mx::subtract(z, mx::expand_dims(lse, 1)));
    const array cols = mx::expand_dims(mx::arange(v0, v1, mx::int32), 0);
    const array onehot = mx::astype(mx::equal(mx::expand_dims(targets, 1), cols), mx::float32);
    const array d = mx::multiply(
        mx::add(mx::multiply(mx::expand_dims(glse, 1), p),
                mx::multiply(mx::expand_dims(gt, 1), onehot)),
        dlogit);
    dx = mx::add(dx, mx::matmul(d, mx::astype(wc, mx::float32)));
    dws.push_back(
        mx::astype(mx::matmul(mx::transpose(d), mx::astype(x, mx::float32)), w.dtype()));
    if (bias != nullptr) {
      dbs.push_back(mx::sum(d, 0));
    }
  }
  std::vector<array> grads = {mx::astype(dx, x.dtype()), mx::concatenate(dws, 0)};
  if (bias != nullptr) {
    grads.push_back(mx::astype(mx::concatenate(dbs, 0), bias->dtype()));
  }
  return grads;
}

std::vector<array> backward(const array& x, const array& w, const array& targets,
                            const array& bias, const array& lse, const array& gmax,
                            const array& glse, const array& gt, const Spec& spec) {
  const int N = x.shape(0);
  const int K = x.shape(1);
  const int V = w.shape(0);
  const int TM = ceil_div(N, kTile);
  const int TV = ceil_div(V, kTile);
  const mx::Dtype out = head_dtype(x, w);

  // The bound, per group and tile. A token whose logsumexp has no cotangent
  // takes no part; a group's +inf (a target in the tile) minus +inf is NaN,
  // which `less` answers false, so such a group stays live.
  const float inf = std::numeric_limits<float>::infinity();
  const float ln_eps = spec.eps > 0.0f ? std::log(spec.eps) : -inf;
  array lse_eff = mx::where(mx::not_equal(glse, array(0.0f)), lse, array(inf));
  lse_eff = mx::pad(lse_eff, {{0, TM * kTile - N}}, array(inf));
  const array group_min = mx::min(mx::reshape(lse_eff, {TM * kGroup, kGroup}), 1, true);
  const array bound = mx::logical_not(mx::less(mx::subtract(gmax, group_min), array(ln_eps)));

  const array fparams = array({spec.scale, spec.softcap, spec.eps}, mx::float32);
  const std::vector<std::pair<std::string, mx::fast::TemplateArg>> grad_targs = {
      {"TX", x.dtype()},
      {"TW", w.dtype()},
      {"TO", out},
      {"BK", kDepth},
      {"BIAS", spec.has_bias},
      {"SOFTCAP", spec.softcap > 0.0f}};

  const int tvc_max = std::clamp(spec.chunk / kTile, 1, TV);
  std::optional<array> dx;
  std::optional<array> dw_chunk;
  // Each chunk's rows land in place: the previous `dw` has no other reader,
  // so `slice_update` donates its buffer rather than copying the whole head.
  array dw = mx::zeros({V, K}, w.dtype());
  std::vector<array> dbs;
  const array dummy = array({0.0f}, mx::float32);
  for (int c0 = 0; c0 < TV; c0 += tvc_max) {
    const int tvc = std::min(tvc_max, TV - c0);
    const int rows = std::min(tvc * kTile, V - c0 * kTile);
    array bound_c = mx::slice(bound, {0, c0}, {TM * kGroup, c0 + tvc});
    // This chunk's scratch only once the previous chunk's has been read, so
    // one is alive at a time. MLX would otherwise run the chunks' gradient
    // kernels side by side, with every scratch alive at once.
    std::vector<array> after;
    if (dx && spec.want_dx) {
      after.push_back(*dx);
    }
    if (dw_chunk && spec.want_dw) {
      after.push_back(*dw_chunk);
    }
    if (!after.empty()) {
      bound_c = mx::depends({bound_c}, after)[0];
    }
    const array dims = array({N, V, K, tvc, c0}, mx::int32);
    auto grad = grad_kernel()(
        {x, w, targets, bias, lse, glse, gt, bound_c, dims, fparams},
        {{TM * kTile, tvc * kTile}, {TM * kGroup, tvc}},
        {out, mx::bool_},
        {TM * kThreads, tvc, 1},
        {kThreads, 1, 1},
        grad_targs,
        std::nullopt,
        false,
        {});
    const array& dz = grad[0];
    const array& live = grad[1];

    auto gx = grad_x_kernel()(
        {dz, w, live, dx ? *dx : dummy, dims},
        {{N, K}},
        {mx::float32},
        {ceil_div(K, kTile) * kThreads, TM, 1},
        {kThreads, 1, 1},
        {{"TO", out}, {"TW", w.dtype()}, {"PREV", dx.has_value()}},
        std::nullopt,
        false,
        {});
    dx = gx[0];

    auto gw = grad_w_kernel()(
        {dz, x, live, dims},
        {{rows, K}},
        {w.dtype()},
        {ceil_div(K, kTile) * kThreads, tvc, 1},
        {kThreads, 1, 1},
        {{"TO", out}, {"TX", x.dtype()}, {"TW", w.dtype()}},
        std::nullopt,
        false,
        {});
    dw_chunk = gw[0];
    dw = mx::slice_update(dw, *dw_chunk, {c0 * kTile, 0}, {c0 * kTile + rows, K});

    if (spec.has_bias) {
      // Column sums over the live groups; a dead group's scratch may never
      // have been written.
      const array sums = mx::sum(
          mx::astype(mx::reshape(dz, {TM * kGroup, kGroup, tvc, kTile}), mx::float32), 1);
      const array kept = mx::where(mx::expand_dims(live, 2), sums, array(0.0f));
      dbs.push_back(mx::slice(mx::reshape(mx::sum(kept, 0), {tvc * kTile}), {0}, {rows}));
    }
  }
  std::vector<array> grads = {mx::astype(*dx, x.dtype()), dw};
  if (spec.has_bias) {
    grads.push_back(mx::astype(mx::concatenate(dbs, 0), bias.dtype()));
  }
  return grads;
}

// Why the fused path can't take these inputs, or nullptr.
const char* unsupported(const array& x, const array& w, const array* bias) {
  if (!mx::metal::is_available() || mx::default_device().type != mx::Device::gpu) {
    return "no Metal GPU on the default device";
  }
  const auto ok = [](mx::Dtype t) {
    return t == mx::float32 || t == mx::float16 || t == mx::bfloat16;
  };
  if (!ok(x.dtype()) || !ok(w.dtype()) || (bias != nullptr && !ok(bias->dtype()))) {
    return "hidden states or head not in f32, f16 or bf16";
  }
  if (x.ndim() != 2 || w.ndim() != 2 || x.shape(1) != w.shape(1) || x.shape(0) == 0 ||
      w.shape(0) == 0 || x.shape(1) == 0) {
    return "hidden states and head don't multiply";
  }
  if (bias != nullptr && (bias->ndim() != 1 || bias->shape(0) != w.shape(0))) {
    return "bias doesn't match the head";
  }
  return nullptr;
}

// ── Cross-entropy over logits that already exist ─────────────────────────────
//
// A model without the fused head hands the loss its `[tokens, vocab]` logits.
// What the loss needs of a row is three numbers, and a threadgroup per row
// reads the row once for all of them, in f32 whatever the logits' dtype:
//
//   lse     the row's logsumexp (an online maximum and sum of exponentials)
//   target  the logit at the row's label, read as it is stored
//   zsum    the sum of the row's logits (label smoothing needs their mean)
//
// The loss is arithmetic on those [tokens] vectors, so label smoothing, the
// z-loss and the ignore mask all stay in ordinary ops, and the one custom
// gradient is the logits': `glse * softmax + gt * onehot + gzsum`, written
// once, in the logits' dtype. Built from MLX ops instead, the same loss holds
// an f32 copy of the logits and another of their gradient, and runs 2.5 times
// as long.

constexpr int kRowThreads = 256;

// Inputs: z [N, V]; targets [N] (int32); dims {N, V}.
// Outputs: lse, target, zsum, each [N] f32.
const char* kRowStatsSource = R"METAL(
  const int V = dims[1];
  const int row = int(threadgroup_position_in_grid.x);
  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  // `auto`: MLX passes an input of fewer than eight elements in constant
  // memory, not device memory.
  auto zr = z + size_t(row) * V;

  float run_m = -INFINITY;
  float run_s = 0.0f;
  float run_sum = 0.0f;
  for (int j = int(tid); j < V; j += ROW_THREADS) {
    const float a = float(zr[j]);
    run_sum += a;
    if (a > run_m) {
      run_s = run_s * fast::exp(run_m - a) + 1.0f;
      run_m = a;
    } else if (a > -INFINITY) {
      run_s += fast::exp(a - run_m);
    }
  }

  threadgroup float part_m[ROW_THREADS / 32];
  threadgroup float part_s[ROW_THREADS / 32];
  threadgroup float part_sum[ROW_THREADS / 32];
  const float sg_m = simd_max(run_m);
  const float sg_s = simd_sum(run_m > -INFINITY ? run_s * fast::exp(run_m - sg_m) : 0.0f);
  const float sg_sum = simd_sum(run_sum);
  if (lane == 0) {
    part_m[sg] = sg_m;
    part_s[sg] = sg_s;
    part_sum[sg] = sg_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const bool live = lane < ROW_THREADS / 32;
    const float pm = live ? part_m[lane] : -INFINITY;
    const float ps = live ? part_s[lane] : 0.0f;
    const float pu = live ? part_sum[lane] : 0.0f;
    const float m = simd_max(pm);
    const float s = simd_sum(pm > -INFINITY ? ps * fast::exp(pm - m) : 0.0f);
    const float u = simd_sum(pu);
    if (lane == 0) {
      const int t = targets[row];
      lse[row] = m + log(s);
      zsum[row] = u;
      target[row] = (t >= 0 && t < V) ? float(zr[t]) : 0.0f;
    }
  }
)METAL";

// Inputs: z [N, V]; targets [N]; lse, glse, gt, gzsum [N] (f32: the rows'
// logsumexp and the cotangents of the three statistics); dims {N, V}.
// Output: dz [N, V] in z's dtype.
const char* kRowGradSource = R"METAL(
  const int V = dims[1];
  const int row = int(threadgroup_position_in_grid.x);
  const uint tid = thread_index_in_threadgroup;
  auto zr = z + size_t(row) * V;
  device TZ* dr = dz + size_t(row) * V;
  const float l = lse[row];
  const float g_lse = glse[row];
  const float g_target = gt[row];
  const float g_sum = gzsum[row];
  const int t = targets[row];
  for (int j = int(tid); j < V; j += ROW_THREADS) {
    const float p = fast::exp(float(zr[j]) - l);
    dr[j] = TZ(g_lse * p + g_sum + (j == t ? g_target : 0.0f));
  }
)METAL";

mx::fast::CustomKernelFunction& row_stats_kernel() {
  static auto kernel = mx::fast::metal_kernel(
      "pmetal_ce_row_stats",
      {"z", "targets", "dims"},
      {"lse", "target", "zsum"},
      kRowStatsSource,
      "");
  return kernel;
}

mx::fast::CustomKernelFunction& row_grad_kernel() {
  static auto kernel = mx::fast::metal_kernel(
      "pmetal_ce_row_grad",
      {"z", "targets", "lse", "glse", "gt", "gzsum", "dims"},
      {"dz"},
      kRowGradSource,
      "");
  return kernel;
}

std::vector<std::pair<std::string, mx::fast::TemplateArg>> row_targs(const array& z) {
  return {{"TZ", z.dtype()}, {"ROW_THREADS", kRowThreads}};
}

// [lse, target logit, sum of logits] per row of z [N, V].
std::vector<array> row_stats(const array& z, const array& targets) {
  const int N = z.shape(0);
  const int V = z.shape(1);
  const array dims = array({N, V}, mx::int32);
  return row_stats_kernel()(
      {z, targets, dims},
      {{N}, {N}, {N}},
      {mx::float32, mx::float32, mx::float32},
      {N * kRowThreads, 1, 1},
      {kRowThreads, 1, 1},
      row_targs(z),
      0.0f,
      false,
      {});
}

// The same statistics from MLX ops, for a device without Metal or a dtype the
// kernels don't take. Differentiable as it is.
std::vector<array> row_stats_ops(const array& z, const array& targets) {
  const array zf = mx::astype(z, mx::float32);
  const array idx = mx::expand_dims(targets, 1);
  const array safe = mx::clip(idx, std::optional<array>(array(0)),
                              std::optional<array>(array(z.shape(1) - 1)));
  const array inside = mx::logical_and(mx::greater_equal(idx, array(0)),
                                       mx::less(idx, array(z.shape(1))));
  const array picked = mx::squeeze(mx::take_along_axis(zf, safe, 1), 1);
  const array target = mx::where(mx::squeeze(inside, 1), picked, array(0.0f));
  return {mx::logsumexp(zf, 1), target, mx::sum(zf, 1)};
}

// d/dz of `glse * lse + gt * z[target] + gzsum * sum(z)`.
array row_grad(const array& z, const array& targets, const array& lse, const array& glse,
               const array& gt, const array& gzsum, bool nested) {
  const int N = z.shape(0);
  const int V = z.shape(1);
  if (nested) {
    const array p = mx::exp(mx::subtract(mx::astype(z, mx::float32), mx::expand_dims(lse, 1)));
    const array cols = mx::expand_dims(mx::arange(0, V, mx::int32), 0);
    const array onehot = mx::astype(mx::equal(mx::expand_dims(targets, 1), cols), mx::float32);
    const array d = mx::add(
        mx::add(mx::multiply(mx::expand_dims(glse, 1), p),
                mx::multiply(mx::expand_dims(gt, 1), onehot)),
        mx::expand_dims(gzsum, 1));
    return mx::astype(d, z.dtype());
  }
  const array dims = array({N, V}, mx::int32);
  return row_grad_kernel()(
      {z, targets, lse, glse, gt, gzsum, dims},
      {{N, V}},
      {z.dtype()},
      {N * kRowThreads, 1, 1},
      {kRowThreads, 1, 1},
      row_targs(z),
      0.0f,
      false,
      {})[0];
}

} // namespace

extern "C" {

int mlx_inline_cce_fused(mlx_inline_array* dst_lse, mlx_inline_array* dst_target,
                         const mlx_inline_array* hidden, const mlx_inline_array* weight,
                         const mlx_inline_array* targets, const mlx_inline_array* bias,
                         float logit_scale, float softcap, float filter_eps,
                         int vocab_chunk, bool want_dx, bool want_dw,
                         const char** reason) {
  *reason = nullptr;
  int status = 0;
  try {
    const array& x = as_arr(hidden);
    const array& w = as_arr(weight);
    const array* b = bias != nullptr ? &as_arr(bias) : nullptr;
    if (const char* why = unsupported(x, w, b)) {
      *reason = why;
      bridge_placeholder(dst_lse);
      bridge_placeholder(dst_target);
      pmetal_runtime_clear_error_internal();
      return 1;
    }
    // The kernels read one target per token.
    const array& targets_in = as_arr(targets);
    if (targets_in.size() != static_cast<size_t>(x.shape(0))) {
      throw std::invalid_argument(
          "[cce_fused] " + std::to_string(targets_in.size()) + " targets for " +
          std::to_string(x.shape(0)) + " tokens");
    }
    const Spec spec{logit_scale, softcap > 0.0f ? softcap : 0.0f, std::max(filter_eps, 0.0f),
                    std::max(vocab_chunk, kTile), b != nullptr, want_dx, want_dw};
    const array t = mx::astype(as_arr(targets), mx::int32);
    // A one-element stand-in when there's no bias; the kernels never read it.
    const array bias_or_dummy = b != nullptr ? *b : array({0.0f}, w.dtype());

    auto fwd = [t, bias_or_dummy, spec](const std::vector<array>& primals) {
      return forward(primals[0], primals[1], t,
                     spec.has_bias ? primals[2] : bias_or_dummy, spec);
    };
    auto vjp = [t, bias_or_dummy, spec](const std::vector<array>& primals,
                                        const std::vector<array>& cotangents,
                                        const std::vector<array>& outputs) {
      const array& x = primals[0];
      const array& w = primals[1];
      const array* b = spec.has_bias ? &primals[2] : nullptr;
      const array glse = mx::astype(cotangents[0], mx::float32);
      const array gt = mx::astype(cotangents[1], mx::float32);
      // A second derivative: this backward runs inside a nested
      // `value_and_grad`, so the gradient being built is itself under
      // differentiation, and the Metal kernels have no VJP. (The cotangents
      // can't tell: MLX marks them as tracers in a first derivative too.)
      if (pmetal_runtime_grad_depth() > 1) {
        return backward_ops(x, w, t, b, outputs[0], glse, gt, spec);
      }
      return backward(x, w, t, b != nullptr ? *b : bias_or_dummy, outputs[0], outputs[2],
                      glse, gt, spec);
    };
    std::vector<array> primals = {x, w};
    if (b != nullptr) {
      primals.push_back(*b);
    }
    auto outs = mx::custom_vjp(fwd, vjp)(primals);
    new (dst_lse->buf) array(outs[0]);
    new (dst_target->buf) array(outs[1]);
    pmetal_runtime_clear_error_internal();
  } catch (const std::exception& e) {
    pmetal_runtime_set_last_error("cce_fused", e.what());
    bridge_placeholder(dst_lse);
    bridge_placeholder(dst_target);
    status = 2;
  } catch (...) {
    pmetal_runtime_set_last_error("cce_fused", "unknown C++ exception");
    bridge_placeholder(dst_lse);
    bridge_placeholder(dst_target);
    status = 2;
  }
  return status;
}

// Each row of `logits` [N, V]: its logsumexp, the logit at its target and the
// sum of its logits, all f32 [N]. Differentiable with respect to the logits.
// Returns 0, or 2 with the error on the channel.
int mlx_inline_ce_row_stats(mlx_inline_array* dst_lse, mlx_inline_array* dst_target,
                            mlx_inline_array* dst_zsum, const mlx_inline_array* logits,
                            const mlx_inline_array* targets) {
  int status = 0;
  try {
    const array& z = as_arr(logits);
    if (z.ndim() != 2 || z.shape(0) == 0 || z.shape(1) == 0) {
      throw std::invalid_argument("[ce_row_stats] logits must be a non-empty [tokens, vocab] matrix");
    }
    if (as_arr(targets).size() != static_cast<size_t>(z.shape(0))) {
      throw std::invalid_argument(
          "[ce_row_stats] " + std::to_string(as_arr(targets).size()) + " targets for " +
          std::to_string(z.shape(0)) + " tokens");
    }
    const array t = mx::astype(as_arr(targets), mx::int32);
    const auto kernel_dtype = [](mx::Dtype d) {
      return d == mx::float32 || d == mx::float16 || d == mx::bfloat16;
    };
    std::vector<array> outs;
    if (!mx::metal::is_available() || mx::default_device().type != mx::Device::gpu ||
        !kernel_dtype(z.dtype())) {
      outs = row_stats_ops(z, t);
    } else {
      auto fwd = [t](const std::vector<array>& primals) { return row_stats(primals[0], t); };
      auto vjp = [t](const std::vector<array>& primals, const std::vector<array>& cotangents,
                     const std::vector<array>& outputs) {
        // A second derivative runs this backward inside a nested
        // `value_and_grad`, and the Metal kernel has no VJP.
        return std::vector<array>{row_grad(
            primals[0], t, outputs[0], mx::astype(cotangents[0], mx::float32),
            mx::astype(cotangents[1], mx::float32), mx::astype(cotangents[2], mx::float32),
            pmetal_runtime_grad_depth() > 1)};
      };
      outs = mx::custom_vjp(fwd, vjp)({z});
    }
    new (dst_lse->buf) array(outs[0]);
    new (dst_target->buf) array(outs[1]);
    new (dst_zsum->buf) array(outs[2]);
    pmetal_runtime_clear_error_internal();
  } catch (const std::exception& e) {
    pmetal_runtime_set_last_error("ce_row_stats", e.what());
    bridge_placeholder(dst_lse);
    bridge_placeholder(dst_target);
    bridge_placeholder(dst_zsum);
    status = 2;
  } catch (...) {
    pmetal_runtime_set_last_error("ce_row_stats", "unknown C++ exception");
    bridge_placeholder(dst_lse);
    bridge_placeholder(dst_target);
    bridge_placeholder(dst_zsum);
    status = 2;
  }
  return status;
}

} // extern "C"
