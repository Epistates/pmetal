//! Accelerate.framework wrappers for CPU-side vector operations.
//!
//! Provides vDSP-accelerated routines used across both the standard training
//! path (gradient norm, clipping) and the ANE hybrid path (RMSNorm, softmax,
//! cross-entropy, Adam optimizer, embedding, matrix multiply).
//!
//! All channel-first operations use `[D, S]` layout where D = channels (dim)
//! and S = spatial (sequence length). Each channel is a contiguous vector of
//! length S, enabling stride-1 vDSP calls.

#[cfg(target_os = "macos")]
mod ffi {
    unsafe extern "C" {
        // === Existing bindings ===

        /// Sum of element-squared: result = sum(data[i]^2)
        pub fn vDSP_svesq(a: *const f32, ia: isize, c: *mut f32, n: usize);

        /// Vector-scalar multiply: C[i] = A[i] * B
        pub fn vDSP_vsmul(
            a: *const f32,
            ia: isize,
            b: *const f32,
            c: *mut f32,
            ic: isize,
            n: usize,
        );

        // === New vDSP bindings ===

        /// Vector add: C[i] = A[i] + B[i]
        pub fn vDSP_vadd(
            a: *const f32,
            ia: isize,
            b: *const f32,
            ib: isize,
            c: *mut f32,
            ic: isize,
            n: usize,
        );

        /// Vector element-wise multiply: C[i] = A[i] * B[i]
        pub fn vDSP_vmul(
            a: *const f32,
            ia: isize,
            b: *const f32,
            ib: isize,
            c: *mut f32,
            ic: isize,
            n: usize,
        );

        /// Vector subtract: C[i] = B[i] - A[i]  (NOTE: vDSP_vsub subtracts A from B!)
        pub fn vDSP_vsub(
            a: *const f32,
            ia: isize,
            b: *const f32,
            ib: isize,
            c: *mut f32,
            ic: isize,
            n: usize,
        );

        /// Vector + scalar add: C[i] = A[i] + B
        pub fn vDSP_vsadd(
            a: *const f32,
            ia: isize,
            b: *const f32,
            c: *mut f32,
            ic: isize,
            n: usize,
        );

        /// Vector * scalar + scalar add: C[i] = A[i] * B + D
        pub fn vDSP_vsmsa(
            a: *const f32,
            ia: isize,
            b: *const f32,
            c: *const f32,
            d: *mut f32,
            id: isize,
            n: usize,
        );

        /// Vector scalar multiply-add: C[i] = A[i] * B + C[i]
        pub fn vDSP_vsma(
            a: *const f32,
            ia: isize,
            b: *const f32,
            c: *const f32,
            ic: isize,
            d: *mut f32,
            id: isize,
            n: usize,
        );

        /// Sum of elements: result = sum(data[i])
        pub fn vDSP_sve(a: *const f32, ia: isize, c: *mut f32, n: usize);

        /// Maximum value: result = max(data[i])
        pub fn vDSP_maxv(a: *const f32, ia: isize, c: *mut f32, n: usize);

        /// Matrix transpose: dst = src^T
        /// vDSP_mtrans(A, IA, C, IC, M, N) transposes M×N → N×M
        pub fn vDSP_mtrans(a: *const f32, ia: isize, c: *mut f32, ic: isize, m: usize, n: usize);

        /// Vectorized exp: dst[i] = exp(src[i])
        pub fn vvexpf(dst: *mut f32, src: *const f32, n: *const i32);

        /// Vectorized reciprocal sqrt: dst[i] = 1/sqrt(src[i])
        pub fn vvrsqrtf(dst: *mut f32, src: *const f32, n: *const i32);

        /// BLAS single-precision general matrix multiply
        pub fn cblas_sgemm(
            order: i32,   // CblasRowMajor=101
            trans_a: i32, // CblasNoTrans=111, CblasTrans=112
            trans_b: i32,
            m: i32, // rows of op(A) and C
            n: i32, // cols of op(B) and C
            k: i32, // cols of op(A) / rows of op(B)
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

// CBLAS constants
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

// ============================================================================
// Existing operations
// ============================================================================

/// Compute the sum of squares of a float slice using vDSP.
pub fn sum_of_squares(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(target_os = "macos")]
    {
        let mut result: f32 = 0.0;
        unsafe {
            ffi::vDSP_svesq(data.as_ptr(), 1, &mut result, data.len());
        }
        result
    }

    #[cfg(not(target_os = "macos"))]
    {
        data.iter().map(|x| x * x).sum()
    }
}

/// Scale a float slice in-place using vDSP.
pub fn scale_inplace(data: &mut [f32], scale: f32) {
    if data.is_empty() {
        return;
    }

    #[cfg(target_os = "macos")]
    {
        unsafe {
            ffi::vDSP_vsmul(data.as_ptr(), 1, &scale, data.as_mut_ptr(), 1, data.len());
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        for x in data.iter_mut() {
            *x *= scale;
        }
    }
}

// ============================================================================
// New vDSP operations for ANE hybrid training
// ============================================================================

/// Vector add: `out[i] = a[i] + b[i]`.
pub fn vadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, out.len());
    if n == 0 {
        return;
    }

    #[cfg(target_os = "macos")]
    unsafe {
        ffi::vDSP_vadd(a.as_ptr(), 1, b.as_ptr(), 1, out.as_mut_ptr(), 1, n);
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..n {
            out[i] = a[i] + b[i];
        }
    }
}

/// Vector element-wise multiply: `out[i] = a[i] * b[i]`.
pub fn vmul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, out.len());
    if n == 0 {
        return;
    }

    #[cfg(target_os = "macos")]
    unsafe {
        ffi::vDSP_vmul(a.as_ptr(), 1, b.as_ptr(), 1, out.as_mut_ptr(), 1, n);
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..n {
            out[i] = a[i] * b[i];
        }
    }
}

/// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`.
///
/// Wraps `cblas_sgemm` with row-major layout.
/// - `m`: rows of op(A) and C
/// - `n`: cols of op(B) and C
/// - `k`: cols of op(A) = rows of op(B)
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) {
    debug_assert_eq!(c.len(), m * n);

    #[cfg(target_os = "macos")]
    {
        let ta = if trans_a { CBLAS_TRANS } else { CBLAS_NO_TRANS };
        let tb = if trans_b { CBLAS_TRANS } else { CBLAS_NO_TRANS };
        let lda = if trans_a { m as i32 } else { k as i32 };
        let ldb = if trans_b { k as i32 } else { n as i32 };
        let ldc = n as i32;
        unsafe {
            ffi::cblas_sgemm(
                CBLAS_ROW_MAJOR,
                ta,
                tb,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                ldc,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Naive fallback
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            for val in c.iter_mut() {
                *val *= beta;
            }
        }
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_val = if trans_a { a[p * m + i] } else { a[i * k + p] };
                    let b_val = if trans_b { b[j * k + p] } else { b[p * n + j] };
                    sum += a_val * b_val;
                }
                c[i * n + j] += alpha * sum;
            }
        }
    }
}

/// Matrix transpose: `dst` = `src`^T.
///
/// `src` is `rows × cols` row-major, `dst` is `cols × rows` row-major.
pub fn matrix_transpose(dst: &mut [f32], src: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);

    #[cfg(target_os = "macos")]
    unsafe {
        // vDSP_mtrans(A, IA, C, IC, M, N) transposes M cols × N rows
        // For a rows×cols matrix: M=cols, N=rows
        ffi::vDSP_mtrans(src.as_ptr(), 1, dst.as_mut_ptr(), 1, cols, rows);
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
}

/// RMSNorm on channel-first `[D, S]` layout.
///
/// For each sequence position, computes:
/// `out[d*S+t] = w[d] * x[d*S+t] / sqrt(mean(x[:,t]^2) + eps)`
///
/// This matches the ANE reference `rmsnorm()` implementation using vDSP.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], dim: usize, seq: usize) {
    debug_assert_eq!(x.len(), dim * seq);
    debug_assert_eq!(out.len(), dim * seq);
    debug_assert_eq!(w.len(), dim);

    #[cfg(target_os = "macos")]
    {
        let mut tmp = vec![0.0f32; seq];
        let mut ss = vec![0.0f32; seq];

        // Accumulate x^2 across channels
        for i in 0..dim {
            let row = &x[i * seq..(i + 1) * seq];
            unsafe {
                ffi::vDSP_vmul(row.as_ptr(), 1, row.as_ptr(), 1, tmp.as_mut_ptr(), 1, seq);
                ffi::vDSP_vadd(tmp.as_ptr(), 1, ss.as_ptr(), 1, ss.as_mut_ptr(), 1, seq);
            }
        }

        // ss = ss / dim + eps
        let inv_d = 1.0f32 / dim as f32;
        let eps = 1e-5f32;
        unsafe {
            ffi::vDSP_vsmsa(ss.as_ptr(), 1, &inv_d, &eps, ss.as_mut_ptr(), 1, seq);
        }

        // ss = 1/sqrt(ss)
        let n = seq as i32;
        unsafe {
            ffi::vvrsqrtf(ss.as_mut_ptr(), ss.as_ptr(), &n);
        }

        // out[d,:] = x[d,:] * ss * w[d]
        for i in 0..dim {
            let row_x = &x[i * seq..(i + 1) * seq];
            let row_out = &mut out[i * seq..(i + 1) * seq];
            unsafe {
                ffi::vDSP_vmul(
                    row_x.as_ptr(),
                    1,
                    ss.as_ptr(),
                    1,
                    row_out.as_mut_ptr(),
                    1,
                    seq,
                );
                ffi::vDSP_vsmul(row_out.as_ptr(), 1, &w[i], row_out.as_mut_ptr(), 1, seq);
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        for t in 0..seq {
            let mut ss = 0.0f32;
            for d in 0..dim {
                let v = x[d * seq + t];
                ss += v * v;
            }
            ss = 1.0 / (ss / dim as f32 + 1e-5).sqrt();
            for d in 0..dim {
                out[d * seq + t] = w[d] * x[d * seq + t] * ss;
            }
        }
    }
}

/// RMSNorm backward on channel-first `[D, S]` layout.
///
/// Computes `dx` (gradient w.r.t. input) and accumulates into `dw` (gradient w.r.t. weights).
/// Matches the ANE reference `rmsnorm_bwd()` implementation.
pub fn rmsnorm_backward(
    dx: &mut [f32],
    dw: &mut [f32],
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    dim: usize,
    seq: usize,
) {
    debug_assert_eq!(x.len(), dim * seq);
    debug_assert_eq!(dy.len(), dim * seq);
    debug_assert_eq!(dx.len(), dim * seq);
    debug_assert_eq!(w.len(), dim);
    debug_assert_eq!(dw.len(), dim);

    #[cfg(target_os = "macos")]
    {
        let mut tmp = vec![0.0f32; seq];
        let mut ss = vec![0.0f32; seq];

        // Compute variance: ss = sum(x^2) / dim + eps
        for i in 0..dim {
            let row = &x[i * seq..(i + 1) * seq];
            unsafe {
                ffi::vDSP_vmul(row.as_ptr(), 1, row.as_ptr(), 1, tmp.as_mut_ptr(), 1, seq);
                ffi::vDSP_vadd(tmp.as_ptr(), 1, ss.as_ptr(), 1, ss.as_mut_ptr(), 1, seq);
            }
        }
        let inv_d = 1.0f32 / dim as f32;
        let eps = 1e-5f32;
        unsafe {
            ffi::vDSP_vsmsa(ss.as_ptr(), 1, &inv_d, &eps, ss.as_mut_ptr(), 1, seq);
        }

        // rrms = 1/sqrt(variance)
        let mut rrms = vec![0.0f32; seq];
        let n = seq as i32;
        unsafe {
            ffi::vvrsqrtf(rrms.as_mut_ptr(), ss.as_ptr(), &n);
        }

        // dot = sum_d(dy[d,:] * x[d,:] * w[d])
        let mut dot = vec![0.0f32; seq];
        for i in 0..dim {
            unsafe {
                ffi::vDSP_vmul(
                    dy[i * seq..].as_ptr(),
                    1,
                    x[i * seq..].as_ptr(),
                    1,
                    tmp.as_mut_ptr(),
                    1,
                    seq,
                );
                ffi::vDSP_vsma(
                    tmp.as_ptr(),
                    1,
                    &w[i],
                    dot.as_ptr(),
                    1,
                    dot.as_mut_ptr(),
                    1,
                    seq,
                );
            }
        }

        // ss = rrms^2 / dim (for the correction term)
        unsafe {
            ffi::vDSP_vmul(rrms.as_ptr(), 1, rrms.as_ptr(), 1, ss.as_mut_ptr(), 1, seq);
            ffi::vDSP_vsmul(ss.as_ptr(), 1, &inv_d, ss.as_mut_ptr(), 1, seq);
        }

        // dot = dot * ss (correction factor)
        unsafe {
            ffi::vDSP_vmul(dot.as_ptr(), 1, ss.as_ptr(), 1, dot.as_mut_ptr(), 1, seq);
        }

        // dx[d,:] = w[d] * (dy[d,:] - x[d,:] * dot) * rrms
        // dw[d] += sum_t(dy[d,t] * x[d,t] * rrms[t])
        for i in 0..dim {
            unsafe {
                // tmp = x[d,:] * dot
                ffi::vDSP_vmul(
                    x[i * seq..].as_ptr(),
                    1,
                    dot.as_ptr(),
                    1,
                    tmp.as_mut_ptr(),
                    1,
                    seq,
                );
                // tmp = dy[d,:] - tmp  (note: vDSP_vsub does B - A)
                ffi::vDSP_vsub(
                    tmp.as_ptr(),
                    1,
                    dy[i * seq..].as_ptr(),
                    1,
                    tmp.as_mut_ptr(),
                    1,
                    seq,
                );
                // tmp = tmp * rrms
                ffi::vDSP_vmul(tmp.as_ptr(), 1, rrms.as_ptr(), 1, tmp.as_mut_ptr(), 1, seq);
                // dx[d,:] = tmp * w[d]
                ffi::vDSP_vsmul(tmp.as_ptr(), 1, &w[i], dx[i * seq..].as_mut_ptr(), 1, seq);

                // dw accumulation: tmp2 = dy[d,:] * x[d,:] * rrms
                ffi::vDSP_vmul(
                    dy[i * seq..].as_ptr(),
                    1,
                    x[i * seq..].as_ptr(),
                    1,
                    tmp.as_mut_ptr(),
                    1,
                    seq,
                );
                ffi::vDSP_vmul(tmp.as_ptr(), 1, rrms.as_ptr(), 1, tmp.as_mut_ptr(), 1, seq);
                let mut s: f32 = 0.0;
                ffi::vDSP_sve(tmp.as_ptr(), 1, &mut s, seq);
                dw[i] += s;
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        for t in 0..seq {
            let mut ss = 0.0f32;
            for d in 0..dim {
                let v = x[d * seq + t];
                ss += v * v;
            }
            let variance = ss / dim as f32 + 1e-5;
            let rrms = 1.0 / variance.sqrt();
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += dy[d * seq + t] * x[d * seq + t] * w[d];
            }
            let correction = dot * rrms * rrms / dim as f32;
            for d in 0..dim {
                dx[d * seq + t] = w[d] * (dy[d * seq + t] - x[d * seq + t] * correction) * rrms;
                dw[d] += dy[d * seq + t] * x[d * seq + t] * rrms;
            }
        }
    }
}

/// Cross-entropy loss with gradient computation.
///
/// Operates on channel-first `[V, S]` layout (vocab × sequence).
/// Returns mean loss and writes `dlogits = (softmax(logits) - one_hot(targets)) / S`.
///
/// Matches the ANE reference `cross_entropy_loss()` using vDSP softmax.
pub fn cross_entropy_loss(
    dlogits: &mut [f32],
    logits: &[f32],
    targets: &[u16],
    vocab: usize,
    seq: usize,
) -> f32 {
    debug_assert_eq!(logits.len(), vocab * seq);
    debug_assert_eq!(dlogits.len(), vocab * seq);
    debug_assert_eq!(targets.len(), seq);

    #[cfg(target_os = "macos")]
    {
        // Transpose [V,S] → [S,V] for contiguous per-position softmax
        let mut buf = vec![0.0f32; seq * vocab];
        unsafe {
            ffi::vDSP_mtrans(logits.as_ptr(), 1, buf.as_mut_ptr(), 1, seq, vocab);
        }

        let mut total_loss = 0.0f32;
        let inv_s = 1.0f32 / seq as f32;

        for t in 0..seq {
            let row = &mut buf[t * vocab..(t + 1) * vocab];

            unsafe {
                // max for numerical stability
                let mut maxv: f32 = 0.0;
                ffi::vDSP_maxv(row.as_ptr(), 1, &mut maxv, vocab);

                // subtract max
                let neg_max = -maxv;
                ffi::vDSP_vsadd(row.as_ptr(), 1, &neg_max, row.as_mut_ptr(), 1, vocab);

                // exp
                let n = vocab as i32;
                ffi::vvexpf(row.as_mut_ptr(), row.as_ptr(), &n);

                // sum and normalize
                let mut sum: f32 = 0.0;
                ffi::vDSP_sve(row.as_ptr(), 1, &mut sum, vocab);
                let inv_sum = 1.0 / sum;
                ffi::vDSP_vsmul(row.as_ptr(), 1, &inv_sum, row.as_mut_ptr(), 1, vocab);
            }

            // loss
            let tgt = targets[t] as usize;
            total_loss -= (row[tgt] + 1e-10).ln();

            // gradient: softmax - one_hot, scaled by 1/S
            row[tgt] -= 1.0;
            unsafe {
                ffi::vDSP_vsmul(row.as_ptr(), 1, &inv_s, row.as_mut_ptr(), 1, vocab);
            }
        }

        // Transpose back [S,V] → [V,S]
        unsafe {
            ffi::vDSP_mtrans(buf.as_ptr(), 1, dlogits.as_mut_ptr(), 1, vocab, seq);
        }

        total_loss / seq as f32
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut total_loss = 0.0f32;
        let inv_s = 1.0 / seq as f32;

        for t in 0..seq {
            // Find max for stability
            let mut maxv = f32::NEG_INFINITY;
            for v in 0..vocab {
                maxv = maxv.max(logits[v * seq + t]);
            }

            // Compute softmax
            let mut sum = 0.0f32;
            for v in 0..vocab {
                let e = (logits[v * seq + t] - maxv).exp();
                dlogits[v * seq + t] = e;
                sum += e;
            }
            for v in 0..vocab {
                dlogits[v * seq + t] /= sum;
            }

            // Loss
            let tgt = targets[t] as usize;
            total_loss -= (dlogits[tgt * seq + t] + 1e-10).ln();

            // Gradient
            dlogits[tgt * seq + t] -= 1.0;
            for v in 0..vocab {
                dlogits[v * seq + t] *= inv_s;
            }
        }

        total_loss / seq as f32
    }
}

/// Softmax in-place on channel-first `[D, S]` layout.
///
/// Computes softmax along the channel dimension for each sequence position.
pub fn softmax_inplace(data: &mut [f32], dim: usize, seq: usize) {
    debug_assert_eq!(data.len(), dim * seq);

    for t in 0..seq {
        // Find max
        let mut maxv = f32::NEG_INFINITY;
        for d in 0..dim {
            maxv = maxv.max(data[d * seq + t]);
        }

        // Exp and sum
        let mut sum = 0.0f32;
        for d in 0..dim {
            let e = (data[d * seq + t] - maxv).exp();
            data[d * seq + t] = e;
            sum += e;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for d in 0..dim {
            data[d * seq + t] *= inv_sum;
        }
    }
}

/// Adam optimizer update.
///
/// Updates weights `w` using gradient `g` with moment estimates `m` and `v`.
/// `t` is the 1-based step count for bias correction.
#[allow(clippy::too_many_arguments)]
pub fn adam_update(
    w: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
) {
    let n = w.len();
    debug_assert_eq!(n, g.len());
    debug_assert_eq!(n, m.len());
    debug_assert_eq!(n, v.len());

    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);

    for i in 0..n {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let mh = m[i] / bc1;
        let vh = v[i] / bc2;
        w[i] -= lr * mh / (vh.sqrt() + eps);
    }
}

/// SiLU activation in-place: `x[i] = x[i] * sigmoid(x[i])`.
///
/// Uses vDSP for vectorized exp computation on macOS.
pub fn silu_inplace(data: &mut [f32]) {
    let n = data.len();
    if n == 0 {
        return;
    }

    #[cfg(target_os = "macos")]
    {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // silu(x) = x * sigmoid(x)
        let mut neg = vec![0.0f32; n];
        let minus_one = -1.0f32;
        unsafe {
            ffi::vDSP_vsmul(data.as_ptr(), 1, &minus_one, neg.as_mut_ptr(), 1, n);
        }
        let ni = n as i32;
        unsafe {
            ffi::vvexpf(neg.as_mut_ptr(), neg.as_ptr(), &ni);
        }
        let one = 1.0f32;
        unsafe {
            ffi::vDSP_vsadd(neg.as_ptr(), 1, &one, neg.as_mut_ptr(), 1, n);
        }
        // neg[i] = 1 + exp(-x[i]), so sigmoid = 1/neg
        // silu = x / neg
        for i in 0..n {
            data[i] /= neg[i];
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        for x in data.iter_mut() {
            let sig = 1.0 / (1.0 + (-*x).exp());
            *x *= sig;
        }
    }
}

/// Embedding lookup: token_ids → x `[dim, seq]` (channel-first).
///
/// `embed` is `[vocab, dim]` row-major. Transposes from row-major embedding
/// to channel-first output layout.
pub fn embed_lookup(x: &mut [f32], embed: &[f32], tokens: &[u16], dim: usize, seq: usize) {
    debug_assert_eq!(x.len(), dim * seq);
    debug_assert_eq!(tokens.len(), seq);

    for t in 0..seq {
        let tok = tokens[t] as usize;
        for d in 0..dim {
            x[d * seq + t] = embed[tok * dim + d];
        }
    }
}

/// Embedding backward: accumulate gradients into `d_embed`.
///
/// `dx` is channel-first `[dim, seq]`, `d_embed` is `[vocab, dim]` row-major.
pub fn embed_backward(d_embed: &mut [f32], dx: &[f32], tokens: &[u16], dim: usize, seq: usize) {
    debug_assert_eq!(dx.len(), dim * seq);
    debug_assert_eq!(tokens.len(), seq);

    for t in 0..seq {
        let tok = tokens[t] as usize;
        for d in 0..dim {
            d_embed[tok * dim + d] += dx[d * seq + t];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_of_squares() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = sum_of_squares(&data);
        assert!((result - 30.0).abs() < 1e-6, "Expected 30.0, got {result}");
    }

    #[test]
    fn test_sum_of_squares_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(sum_of_squares(&data), 0.0);
    }

    #[test]
    fn test_scale_inplace() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        scale_inplace(&mut data, 0.5);
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 1.5).abs() < 1e-6);
        assert!((data[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        scale_inplace(&mut data, 2.0);
    }

    #[test]
    fn test_sum_of_squares_large() {
        let data: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
        let expected: f32 = data.iter().map(|x| x * x).sum();
        let result = sum_of_squares(&data);
        assert!(
            (result - expected).abs() / expected < 1e-5,
            "Expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_sum_of_squares_negative_values() {
        let data = vec![-1.0f32, -2.0, -3.0, -4.0];
        let result = sum_of_squares(&data);
        assert!((result - 30.0).abs() < 1e-6, "Expected 30.0, got {result}");
    }

    #[test]
    fn test_sum_of_squares_single_element() {
        let data = vec![7.0f32];
        let result = sum_of_squares(&data);
        assert!((result - 49.0).abs() < 1e-6, "Expected 49.0, got {result}");
    }

    #[test]
    fn test_sum_of_squares_very_large() {
        let n = 1_048_576;
        let data: Vec<f32> = vec![1.0; n];
        let result = sum_of_squares(&data);
        assert!(
            (result - n as f32).abs() / (n as f32) < 1e-5,
            "Expected {n}, got {result}"
        );
    }

    #[test]
    fn test_scale_inplace_negative_values() {
        let mut data = vec![-1.0f32, -2.0, -3.0, -4.0];
        scale_inplace(&mut data, -2.0);
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
        assert!((data[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_inplace_single_element() {
        let mut data = vec![5.0f32];
        scale_inplace(&mut data, 3.0);
        assert!(
            (data[0] - 15.0).abs() < 1e-6,
            "Expected 15.0, got {}",
            data[0]
        );
    }

    #[test]
    fn test_scale_inplace_very_large() {
        let n = 1_048_576;
        let mut data: Vec<f32> = vec![2.0; n];
        scale_inplace(&mut data, 0.5);
        for (i, val) in data.iter().enumerate().take(10) {
            assert!(
                (*val - 1.0).abs() < 1e-6,
                "Mismatch at index {i}: expected 1.0, got {val}"
            );
        }
        assert!((data[n - 1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vadd() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 4];
        vadd(&a, &b, &mut out);
        assert!((out[0] - 11.0).abs() < 1e-6);
        assert!((out[3] - 44.0).abs() < 1e-6);
    }

    #[test]
    fn test_vmul() {
        let a = vec![2.0f32, 3.0, 4.0, 5.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 4];
        vmul(&a, &b, &mut out);
        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[3] - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_basic() {
        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        gemm(&a, &b, &mut c, 2, 2, 3, 1.0, 0.0, false, false);
        assert!((c[0] - 58.0).abs() < 1e-4); // 1*7+2*9+3*11
        assert!((c[1] - 64.0).abs() < 1e-4); // 1*8+2*10+3*12
        assert!((c[2] - 139.0).abs() < 1e-4); // 4*7+5*9+6*11
        assert!((c[3] - 154.0).abs() < 1e-4); // 4*8+5*10+6*12
    }

    #[test]
    fn test_matrix_transpose() {
        // 2x3 → 3x2
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 6];
        matrix_transpose(&mut dst, &src, 2, 3);
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_rmsnorm_basic() {
        let dim = 4;
        let seq = 2;
        // Channel-first: x[d*seq + t]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4 channels, 2 positions
        let w = vec![1.0, 1.0, 1.0, 1.0]; // identity weights
        let mut out = vec![0.0f32; dim * seq];
        rmsnorm(&mut out, &x, &w, dim, seq);

        // Verify: for position 0: values are 1,3,5,7
        // rms = sqrt((1+9+25+49)/4 + 1e-5) = sqrt(21.00001) ≈ 4.58258
        // out = val / rms
        let rms0 = ((1.0 + 9.0 + 25.0 + 49.0) / 4.0 + 1e-5f32).sqrt();
        assert!((out[0] - 1.0 / rms0).abs() < 1e-4);
        assert!((out[2] - 3.0 / rms0).abs() < 1e-4);
    }

    #[test]
    fn test_cross_entropy_basic() {
        let vocab = 4;
        let seq = 1;
        // One position, 4 vocab items
        let logits = vec![1.0, 2.0, 3.0, 4.0]; // [V, S=1] channel-first
        let targets = vec![2u16]; // target is vocab index 2
        let mut dlogits = vec![0.0f32; vocab * seq];

        let loss = cross_entropy_loss(&mut dlogits, &logits, &targets, vocab, seq);

        // Softmax of [1,2,3,4]: sm = exp(x-4)/sum
        let max_v = 4.0f32;
        let exps: Vec<f32> = logits.iter().map(|x| (x - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let sm: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let expected_loss = -(sm[2] + 1e-10).ln();

        assert!(
            (loss - expected_loss).abs() < 1e-4,
            "loss: {loss} vs {expected_loss}"
        );
    }

    #[test]
    fn test_adam_update() {
        let mut w = vec![1.0f32, 2.0, 3.0];
        let g = vec![0.1, 0.2, 0.3];
        let mut m = vec![0.0f32; 3];
        let mut v = vec![0.0f32; 3];

        adam_update(&mut w, &g, &mut m, &mut v, 1, 0.001, 0.9, 0.999, 1e-8);

        // After 1 step, weights should have decreased
        assert!(w[0] < 1.0);
        assert!(w[1] < 2.0);
        assert!(w[2] < 3.0);
    }

    #[test]
    fn test_silu_inplace() {
        let mut data = vec![0.0f32, 1.0, -1.0, 2.0, -2.0];
        silu_inplace(&mut data);

        // silu(0) = 0 * 0.5 = 0
        assert!((data[0] - 0.0).abs() < 1e-5);
        // silu(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((data[1] - 0.7311).abs() < 1e-3);
        // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert!((data[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_embed_lookup_roundtrip() {
        let vocab = 3;
        let dim = 2;
        let seq = 3;
        // embed: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        let embed = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let tokens = vec![2u16, 0, 1];
        let mut x = vec![0.0f32; dim * seq];

        embed_lookup(&mut x, &embed, &tokens, dim, seq);

        // Token 2 at position 0: x[0*3+0]=0.5, x[1*3+0]=0.6
        assert!((x[0] - 0.5).abs() < 1e-6);
        assert!((x[3] - 0.6).abs() < 1e-6);
        // Token 0 at position 1: x[0*3+1]=0.1, x[1*3+1]=0.2
        assert!((x[1] - 0.1).abs() < 1e-6);
        assert!((x[4] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_embed_backward() {
        let dim = 2;
        let seq = 2;
        let vocab = 3;
        let dx = vec![1.0, 2.0, 3.0, 4.0]; // [dim=2, seq=2] channel-first
        let tokens = vec![1u16, 1]; // both positions point to token 1
        let mut d_embed = vec![0.0f32; vocab * dim];

        embed_backward(&mut d_embed, &dx, &tokens, dim, seq);

        // Token 1: d_embed[1*2+0] += dx[0*2+0] + dx[0*2+1] = 1+2 = 3
        //          d_embed[1*2+1] += dx[1*2+0] + dx[1*2+1] = 3+4 = 7
        assert!((d_embed[2] - 3.0).abs() < 1e-6);
        assert!((d_embed[3] - 7.0).abs() < 1e-6);
    }
}
