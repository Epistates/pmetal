//! Pillow-exact image resampling.
//!
//! Every HuggingFace image processor resamples through Pillow — the fast
//! torchvision backends reimplement its filters rather than the other way round
//! — so Pillow's output *is* the definition of a released VLM's pixel input.
//! Matching it is not a nicety: a vision tower sees whatever the resampler
//! produced, and a systematically different resize is a silent, permanent
//! distribution shift on every image the model ever sees.
//!
//! The `image` crate's `imageops::resize` gets the kernel, the half-pixel
//! alignment, and the downscale support scaling right, but still differs from
//! Pillow by up to 14/255 at hard edges, because it samples
//! vertical-before-horizontal through an *unclamped* f32 intermediate where
//! Pillow goes horizontal-first through a clamped 8-bit one. Bicubic overshoot
//! at an edge therefore survives one implementation's first pass and is clipped
//! in the other's. `fast_image_resize` measures no closer (15/255). The gap is
//! structural, so this module reproduces Pillow's pipeline instead:
//!
//!   1. horizontal pass, then vertical (never the reverse),
//!   2. **22-bit fixed-point** taps and accumulation, not floating point,
//!   3. clamp to `[0, 255]` *between* the passes, not only at the end.
//!
//! The result is bit-exact against Pillow.
//! `crates/pmetal-data/tests/pillow_resample_parity.rs` asserts that by byte
//! comparison over both filters and a grid of scale ratios — up, down,
//! single-axis, >20x, and coprime sizes where no output centre lands on an input
//! centre.

use image::RgbImage;

/// Bits of fractional precision in the fixed-point accumulator.
///
/// Pillow's `PRECISION_BITS = 32 - 8 - 2`: eight bits for the sample and two of
/// headroom for a cubic filter's overshoot, inside a 32-bit accumulator. We
/// accumulate in `i64` — the wider type cannot change the result, since Pillow's
/// budget already rules out overflow, but it removes any chance of a debug-build
/// panic if a future filter has fatter lobes.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Half an output unit, added before the shift so the truncation rounds.
const ROUND_BIAS: i64 = 1 << (PRECISION_BITS - 1);

/// Resampling filters, defined exactly as Pillow defines them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleFilter {
    /// Triangle / linear interpolation. `PIL.Image.BILINEAR`.
    Bilinear,
    /// Cubic convolution with `a = -0.5` — the Catmull-Rom spline.
    /// `PIL.Image.BICUBIC`, and what every Gemma / SigLIP / CLIP processor uses.
    Bicubic,
}

impl ResampleFilter {
    /// Kernel half-width in input pixels, before downscale stretching.
    fn support(self) -> f64 {
        match self {
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
        }
    }

    /// Filter weight at `x` (in units of the stretched kernel).
    fn kernel(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Self::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Self::Bicubic => {
                // Pillow's `bicubic_filter`, written in the same Horner form so
                // the floating-point rounding matches term for term.
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

/// Precomputed filter taps for one axis: Pillow's `precompute_coeffs`.
struct AxisCoeffs {
    /// `(first input index, tap count)` per output pixel.
    bounds: Vec<(usize, usize)>,
    /// Fixed-point taps, `ksize` per output pixel (unused slots stay zero).
    taps: Vec<i64>,
    /// Stride into `taps`.
    ksize: usize,
}

impl AxisCoeffs {
    /// Taps for output pixel `out`.
    fn taps_for(&self, out: usize) -> (usize, &[i64]) {
        let (start, count) = self.bounds[out];
        let base = out * self.ksize;
        (start, &self.taps[base..base + count])
    }
}

/// Build one axis' filter taps.
///
/// Pillow maps output pixel centres to input coordinates as
/// `center = (out + 0.5) · scale`, stretches the kernel by `max(1, scale)` when
/// downscaling (so the filter averages rather than aliases), normalises the taps
/// to sum to 1, and only then quantises to fixed point — rounding away from
/// zero, which is why the sign test below is on the normalised weight.
fn precompute_coeffs(in_size: usize, out_size: usize, filter: ResampleFilter) -> AxisCoeffs {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = filter.support() * filter_scale;
    let ksize = support.ceil() as usize * 2 + 1;
    let inv_scale = 1.0 / filter_scale;

    let mut bounds = Vec::with_capacity(out_size);
    let mut taps = vec![0i64; out_size * ksize];
    let mut weights = vec![0f64; ksize];

    for out in 0..out_size {
        let center = (out as f64 + 0.5) * scale;
        // C's `(int)` truncates toward zero. Both bounds are clamped into range
        // immediately after, which makes truncation and floor agree here.
        let first = ((center - support + 0.5) as isize).clamp(0, in_size as isize) as usize;
        let last = ((center + support + 0.5) as isize).clamp(0, in_size as isize) as usize;
        let count = last.saturating_sub(first).min(ksize);

        let mut sum = 0.0;
        for (i, weight) in weights.iter_mut().take(count).enumerate() {
            *weight = filter.kernel(((first + i) as f64 - center + 0.5) * inv_scale);
            sum += *weight;
        }

        let base = out * ksize;
        for (i, &weight) in weights.iter().take(count).enumerate() {
            let normalised = if sum != 0.0 { weight / sum } else { 0.0 };
            let scaled = normalised * (1i64 << PRECISION_BITS) as f64;
            // Round half away from zero, as Pillow's `normalize_coeffs_8bpc`.
            taps[base + i] = (if normalised < 0.0 {
                scaled - 0.5
            } else {
                scaled + 0.5
            }) as i64;
        }
        bounds.push((first, count));
    }

    AxisCoeffs {
        bounds,
        taps,
        ksize,
    }
}

/// Shift the fixed-point accumulator back down and clamp, as Pillow's `clip8`.
///
/// Pillow spells the clamp as a lookup table sized for the maximum overshoot a
/// cubic filter can produce; the arithmetic is a plain saturating shift.
fn clip8(acc: i64) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resample along x: `[src_h, src_w, 3]` → `[src_h, out_w, 3]`, interleaved RGB.
fn horizontal_pass(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    out_w: usize,
    c: &AxisCoeffs,
) -> Vec<u8> {
    let mut out = vec![0u8; out_w * src_h * 3];
    for y in 0..src_h {
        let row = &src[y * src_w * 3..(y + 1) * src_w * 3];
        for x in 0..out_w {
            let (first, taps) = c.taps_for(x);
            let dst = (y * out_w + x) * 3;
            for ch in 0..3 {
                let mut acc = ROUND_BIAS;
                for (i, &tap) in taps.iter().enumerate() {
                    acc += row[(first + i) * 3 + ch] as i64 * tap;
                }
                out[dst + ch] = clip8(acc);
            }
        }
    }
    out
}

/// Resample along y: `[src_h, width, 3]` → `[out_h, width, 3]`, interleaved RGB.
fn vertical_pass(src: &[u8], width: usize, out_h: usize, c: &AxisCoeffs) -> Vec<u8> {
    let mut out = vec![0u8; width * out_h * 3];
    for y in 0..out_h {
        let (first, taps) = c.taps_for(y);
        for x in 0..width {
            let dst = (y * width + x) * 3;
            for ch in 0..3 {
                let mut acc = ROUND_BIAS;
                for (i, &tap) in taps.iter().enumerate() {
                    acc += src[((first + i) * width + x) * 3 + ch] as i64 * tap;
                }
                out[dst + ch] = clip8(acc);
            }
        }
    }
    out
}

/// Resize an RGB8 image, bit-exactly reproducing
/// `PIL.Image.resize((width, height), resample=filter)`.
///
/// Each axis whose size is unchanged is skipped, as Pillow does — the taps for a
/// 1:1 axis are an identity kernel, so this is a shortcut rather than a
/// behavioural difference.
pub fn resize_rgb8(src: &RgbImage, width: u32, height: u32, filter: ResampleFilter) -> RgbImage {
    let (src_w, src_h) = (src.width() as usize, src.height() as usize);
    let (out_w, out_h) = (width as usize, height as usize);
    if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
        return RgbImage::new(width, height);
    }
    if (out_w, out_h) == (src_w, src_h) {
        return src.clone();
    }

    let mut buffer = src.as_raw().clone();
    let mut buffer_w = src_w;
    if out_w != src_w {
        let coeffs = precompute_coeffs(src_w, out_w, filter);
        buffer = horizontal_pass(&buffer, buffer_w, src_h, out_w, &coeffs);
        buffer_w = out_w;
    }
    if out_h != src_h {
        let coeffs = precompute_coeffs(src_h, out_h, filter);
        buffer = vertical_pass(&buffer, buffer_w, out_h, &coeffs);
    }

    RgbImage::from_raw(width, height, buffer).expect("resampled buffer matches the output size")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 11) as u8, (x * 3 + y * 5) as u8])
        })
    }

    #[test]
    fn identity_size_is_a_passthrough() {
        let src = ramp(9, 5);
        let out = resize_rgb8(&src, 9, 5, ResampleFilter::Bicubic);
        assert_eq!(out.as_raw(), src.as_raw());
    }

    /// A one-axis resize must leave the other axis untouched, which is what the
    /// per-axis skip buys — and a cheap check that the passes are not transposed.
    #[test]
    fn single_axis_resize_preserves_the_other_axis() {
        let src = ramp(12, 6);
        let out = resize_rgb8(&src, 4, 6, ResampleFilter::Bilinear);
        assert_eq!((out.width(), out.height()), (4, 6));
        let out = resize_rgb8(&src, 12, 3, ResampleFilter::Bilinear);
        assert_eq!((out.width(), out.height()), (12, 3));
    }

    /// Taps must sum to one within fixed-point rounding, or the image gains or
    /// loses brightness. Checked at a downscale (stretched kernel) and an
    /// upscale (kernel at unit width).
    #[test]
    fn taps_are_normalised() {
        for (in_size, out_size) in [(100, 37), (37, 100), (16, 16)] {
            let coeffs = precompute_coeffs(in_size, out_size, ResampleFilter::Bicubic);
            for out in 0..out_size {
                let (_, taps) = coeffs.taps_for(out);
                let sum: i64 = taps.iter().sum();
                let one = 1i64 << PRECISION_BITS;
                assert!(
                    (sum - one).abs() <= taps.len() as i64,
                    "{in_size}->{out_size} out {out}: taps sum to {sum}, want {one}"
                );
            }
        }
    }

    /// A constant image must survive any resize unchanged: the taps sum to one,
    /// so every output sample is the same constant. Catches sign errors and
    /// bound off-by-ones that a smooth-ramp test would hide.
    #[test]
    fn flat_field_is_preserved() {
        let src = RgbImage::from_pixel(23, 17, image::Rgb([200, 100, 37]));
        for (w, h) in [(7, 5), (61, 43), (23, 5), (7, 17)] {
            for filter in [ResampleFilter::Bilinear, ResampleFilter::Bicubic] {
                let out = resize_rgb8(&src, w, h, filter);
                for px in out.pixels() {
                    assert_eq!(px.0, [200, 100, 37], "{w}x{h} {filter:?}");
                }
            }
        }
    }

    /// Bicubic overshoot at a hard edge must clamp rather than wrap: the whole
    /// reason this module exists is that the clamp lands between the passes.
    #[test]
    fn hard_edge_overshoot_stays_in_range() {
        let src = RgbImage::from_fn(32, 32, |x, y| {
            if x < 16 && y < 16 {
                image::Rgb([0, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            }
        });
        let out = resize_rgb8(&src, 12, 12, ResampleFilter::Bicubic);
        // `u8` cannot represent an out-of-range value, so the assertion that
        // matters is that the extremes are still present and untouched.
        assert!(out.pixels().any(|p| p.0 == [0, 0, 0]));
        assert!(out.pixels().any(|p| p.0 == [255, 255, 255]));
    }
}
