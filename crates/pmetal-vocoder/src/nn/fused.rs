//! A fused Metal kernel for the anti-aliased activation.
//!
//! [`Activation1d`](super::Activation1d) is upsample, Snake, low-pass and
//! decimate. Run as separate ops it is about twenty tiny kernels per call on
//! tensors that fit in cache, so the GPU spends its time launching and
//! synchronising instead of computing; BigVGAN-v2 calls it 108 times per
//! forward and it dominates synthesis time. This kernel does all four steps
//! in one launch, reading the input once and writing the output once, as the
//! reference's CUDA kernel does.
//!
//! The kernel has no gradient. It is used for inference only, after
//! [`Activation1d::prepare_inference`](super::Activation1d::prepare_inference).

use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::Array;
use pmetal_runtime::{MetalKernel, TemplateArg};
use std::sync::OnceLock;

/// Output samples one threadgroup produces per channel lane.
const TILE: i32 = 16;
/// Channel lanes per threadgroup (fewer for narrow tensors).
const MAX_LANES: i32 = 32;

/// The kernel body.
///
/// A threadgroup owns a tile of `TILE` consecutive output samples for `LANES`
/// channels. It first computes the activated, interpolated signal the tile's
/// low-pass window reads, once per sample, into threadgroup memory:
///
/// ```text
/// u[n] = sum_t fu[n + PAD_L_U - t*UP] * x[clamp(t - PAD_U)]    (interpolation)
/// z[n] = u[n] + sin^2(alpha u[n]) / (beta + eps)               (Snake)
/// ```
///
/// then each thread forms one output, `out[j] = sum_i fd[i] z[clamp(j*DOWN + i - PAD_L_D)]`.
/// Computing `z` once per tile rather than once per output saves about
/// `KD / DOWN` times the `sin` evaluations, which are the expensive part. The
/// interpolation gain `UP` is folded into `fu`; padded threads past the end of
/// the tensor take part in the barrier but write nothing.
const SOURCE: &str = r#"
    constexpr int ZLEN = TILE * DOWN + KD - DOWN;
    threadgroup float z_tile[ZLEN * LANES];

    const int lane = (int)thread_position_in_threadgroup.x;
    const int row = (int)thread_position_in_threadgroup.y;
    const int c = (int)thread_position_in_grid.x;
    const int j0 = (int)threadgroup_position_in_grid.y * TILE;
    const uint b = thread_position_in_grid.z;
    const int len = x_shape[1];
    const int chans = x_shape[2];
    const int up_len = len * UP;
    const int out_len = (up_len + PAD_L_D + PAD_R_D - KD) / DOWN + 1;

    const int cr = min(c, chans - 1);
    const size_t base = (size_t)b * (size_t)len * (size_t)chans + (size_t)cr;
    const float a = alpha[cr];
    const float inv = inv_den[cr];

    for (int zi = row; zi < ZLEN; zi += TILE) {
        int n = j0 * DOWN - PAD_L_D + zi;
        n = min(max(n, 0), up_len - 1);
        const int shifted = n + PAD_L_U;
        const int t_hi = min(shifted / UP, len + 2 * PAD_U - 1);
        int t_lo = shifted - (KU - 1);
        t_lo = t_lo <= 0 ? 0 : (t_lo + UP - 1) / UP;
        float u = 0.0f;
        for (int t = t_lo; t <= t_hi; ++t) {
            const int src = min(max(t - PAD_U, 0), len - 1);
            u += fu[shifted - t * UP] * (float)x[base + (size_t)src * (size_t)chans];
        }
        const float s = metal::precise::sin(a * u);
        z_tile[zi * LANES + lane] = u + s * s * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int j = j0 + row;
    if (c < chans && j < out_len) {
        float acc = 0.0f;
        for (int i = 0; i < KD; ++i) {
            acc += fd[i] * z_tile[(row * DOWN + i) * LANES + lane];
        }
        out[((size_t)b * (size_t)out_len + (size_t)j) * (size_t)chans + (size_t)c] =
            static_cast<T>(acc);
    }
"#;

fn kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "bigvgan_anti_alias_activation",
            &["x", "alpha", "inv_den", "fu", "fd"],
            &["out"],
            SOURCE,
            "",
        )
        .expect("the kernel source has no NUL bytes")
    })
}

/// Geometry of the resamplers around the activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub up_ratio: i32,
    pub down_ratio: i32,
    pub up_kernel: i32,
    pub down_kernel: i32,
    /// Replicate padding applied to the input before interpolating.
    pub up_pad: i32,
    /// Offset of the first kept interpolated sample.
    pub up_pad_left: i32,
    pub down_pad_left: i32,
    pub down_pad_right: i32,
}

impl Geometry {
    /// Whether the kernel's threadgroup tile fits Metal's 32 KiB threadgroup
    /// memory for `channels` channels (it does for every ratio BigVGAN uses;
    /// larger decimation ratios fall back to the op path).
    pub fn fits_threadgroup_memory(&self, channels: i32) -> bool {
        let lanes = channels.clamp(1, MAX_LANES);
        let tile_samples = TILE * self.down_ratio + self.down_kernel - self.down_ratio;
        i64::from(tile_samples) * i64::from(lanes) * 4 <= 30_000
    }

    /// Output length for `len` input samples.
    pub fn out_len(&self, len: i32) -> i32 {
        (len * self.up_ratio + self.down_pad_left + self.down_pad_right - self.down_kernel)
            / self.down_ratio
            + 1
    }
}

/// Per-channel and filter constants the kernel reads.
#[derive(Debug)]
pub(crate) struct FusedParams {
    pub geometry: Geometry,
    /// `alpha` (exponentiated if stored in log scale), float32 `[C]`.
    pub alpha: Array,
    /// `1 / (beta + eps)`, float32 `[C]`.
    pub inv_den: Array,
    /// Interpolation taps times the up ratio, float32 `[KU]`.
    pub up_taps: Array,
    /// Low-pass taps, float32 `[KD]`.
    pub down_taps: Array,
}

/// Run the fused activation on `x` (`[batch, time, channels]`).
pub(crate) fn anti_alias_activation(x: &Array, p: &FusedParams) -> Result<Array> {
    let g = &p.geometry;
    let (batch, len, channels) = (x.dim(0), x.dim(1), x.dim(2));
    let out_len = g.out_len(len);
    if out_len < 1 {
        return Err(VocoderError::ShapeMismatch {
            expected: vec![-1, -1, channels],
            actual: x.shape().to_vec(),
        });
    }
    // Threads along channels first: neighbouring lanes read neighbouring
    // addresses of a channels-last tensor. The grid is rounded up to whole
    // threadgroups because the kernel synchronises within one.
    let lanes = channels.clamp(1, MAX_LANES);
    let template = [
        ("T", TemplateArg::Dtype(x.dtype())),
        ("UP", TemplateArg::Int(g.up_ratio)),
        ("DOWN", TemplateArg::Int(g.down_ratio)),
        ("KU", TemplateArg::Int(g.up_kernel)),
        ("KD", TemplateArg::Int(g.down_kernel)),
        ("PAD_U", TemplateArg::Int(g.up_pad)),
        ("PAD_L_U", TemplateArg::Int(g.up_pad_left)),
        ("PAD_L_D", TemplateArg::Int(g.down_pad_left)),
        ("PAD_R_D", TemplateArg::Int(g.down_pad_right)),
        ("TILE", TemplateArg::Int(TILE)),
        ("LANES", TemplateArg::Int(lanes)),
    ];
    let grid = [
        (channels + lanes - 1) / lanes * lanes,
        (out_len + TILE - 1) / TILE * TILE,
        batch,
    ];
    let mut out = kernel()
        .call(
            &[x, &p.alpha, &p.inv_den, &p.up_taps, &p.down_taps],
            &[(&[batch, out_len, channels], x.dtype())],
            grid,
            [lanes, TILE, 1],
            &template,
        )
        .map_err(|e| VocoderError::Mlx(e.into()))?;
    Ok(out.remove(0))
}
