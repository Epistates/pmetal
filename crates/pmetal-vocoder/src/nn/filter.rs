//! Kaiser-windowed sinc low-pass filters.
//!
//! BigVGAN band-limits its periodic activations: the signal is upsampled,
//! passed through the nonlinearity, then low-pass filtered and downsampled so
//! the harmonics the activation creates cannot fold back as aliasing. Both
//! resamplers use the same windowed-sinc design (Kaiser, with the window shape
//! chosen from the attenuation the transition band calls for), reproduced here
//! tap for tap.

use crate::error::{Result, VocoderError};
use std::f64::consts::PI;

/// Zeroth-order modified Bessel function of the first kind, `I0(x)`.
///
/// Power series `sum_k ((x/2)^k / k!)^2`; it converges for every finite `x`
/// and the Kaiser windows here have `beta` of at most a few tens.
pub fn bessel_i0(x: f64) -> f64 {
    let half_sq = (x / 2.0) * (x / 2.0);
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    loop {
        term *= half_sq / (k * k);
        sum += term;
        if term < sum * 1e-17 {
            return sum;
        }
        k += 1.0;
    }
}

/// Normalised sinc, `sin(pi x) / (pi x)`, with `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// A symmetric (non-periodic) Kaiser window of `len` samples.
pub fn kaiser_window(len: usize, beta: f64) -> Vec<f64> {
    if len == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    let half = (len - 1) as f64 / 2.0;
    (0..len)
        .map(|n| {
            let r = (n as f64 - half) / half;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// The Kaiser `beta` for a stop-band attenuation of `a` dB.
fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

/// A Kaiser-windowed sinc low-pass filter with unit DC gain.
///
/// * `cutoff` - cutoff frequency as a fraction of the sample rate (`0 < cutoff <= 0.5`).
/// * `half_width` - transition half-width, same units; it sets the window's `beta`.
/// * `kernel_size` - number of taps. Even sizes centre the filter between
///   samples (a half-sample delay), odd sizes on a sample.
///
/// Normalising the taps to sum to one keeps a constant input constant; without
/// it the filter would leak a little of the DC component.
pub fn kaiser_sinc_filter(cutoff: f64, half_width: f64, kernel_size: usize) -> Result<Vec<f32>> {
    if kernel_size == 0 {
        return Err(VocoderError::Config(
            "filter kernel_size must be positive".into(),
        ));
    }
    if !(cutoff > 0.0 && cutoff <= 0.5) {
        return Err(VocoderError::Config(format!(
            "filter cutoff must be in (0, 0.5], got {cutoff}"
        )));
    }
    let even = kernel_size % 2 == 0;
    let half_size = (kernel_size / 2) as f64;

    // Window shape from the transition width: A = 2.285 (N-1) pi df + 7.95.
    let delta_f = 4.0 * half_width;
    let attenuation = 2.285 * (half_size - 1.0) * PI * delta_f + 7.95;
    let window = kaiser_window(kernel_size, kaiser_beta(attenuation));

    let taps: Vec<f64> = (0..kernel_size)
        .map(|n| {
            let time = if even {
                n as f64 - half_size + 0.5
            } else {
                n as f64 - half_size
            };
            2.0 * cutoff * window[n] * sinc(2.0 * cutoff * time)
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    Ok(taps.into_iter().map(|t| (t / sum) as f32).collect())
}

/// Frequency-sampling FIR design (`scipy.signal.firwin2`) for a linear-phase
/// filter of `numtaps` taps with a Hamming window.
///
/// `freq` (from 0 to 1, where 1 is Nyquist) and `gain` describe the desired
/// piecewise-linear magnitude response. Only the type I (odd `numtaps`) and
/// type II (even `numtaps`, which must have zero gain at Nyquist) cases are
/// supported, with strictly increasing `freq`.
pub fn firwin2(numtaps: usize, freq: &[f64], gain: &[f64]) -> Result<Vec<f32>> {
    let err = |msg: &str| Err(VocoderError::Config(format!("firwin2: {msg}")));
    if numtaps < 2 || freq.len() != gain.len() || freq.len() < 2 {
        return err("needs at least two taps and matching freq/gain lists");
    }
    if freq[0] != 0.0 || freq[freq.len() - 1] != 1.0 {
        return err("freq must start at 0 and end at 1");
    }
    if freq.windows(2).any(|w| w[1] <= w[0]) {
        return err("freq must be strictly increasing");
    }
    if numtaps % 2 == 0 && gain[gain.len() - 1] != 0.0 {
        return err("an even-length filter must have zero gain at Nyquist");
    }

    // Frequency grid: 1 + 2^ceil(log2(numtaps)) points on [0, 1].
    let nfreqs = 1 + (1usize << (numtaps as f64).log2().ceil() as u32);
    let interp = |x: f64| -> f64 {
        let i = freq.partition_point(|&f| f <= x).clamp(1, freq.len() - 1);
        let (x0, x1) = (freq[i - 1], freq[i]);
        let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
        gain[i - 1] + t * (gain[i] - gain[i - 1])
    };
    let grid: Vec<f64> = (0..nfreqs)
        .map(|k| k as f64 / (nfreqs - 1) as f64)
        .collect();
    // Linear phase: delay the response by (numtaps - 1) / 2 samples so the
    // first `numtaps` samples of the inverse transform are the filter.
    let delay = (numtaps - 1) as f64 / 2.0;
    let spectrum: Vec<(f64, f64)> = grid
        .iter()
        .map(|&x| {
            let magnitude = interp(x);
            let phase = -delay * PI * x;
            (magnitude * phase.cos(), magnitude * phase.sin())
        })
        .collect();

    // Inverse real DFT of length n = 2 (nfreqs - 1); only n < numtaps is needed.
    let n = 2 * (nfreqs - 1);
    let hamming = |i: usize| 0.54 - 0.46 * (2.0 * PI * i as f64 / (numtaps - 1) as f64).cos();
    Ok((0..numtaps)
        .map(|i| {
            let mut acc =
                spectrum[0].0 + spectrum[nfreqs - 1].0 * if i % 2 == 0 { 1.0 } else { -1.0 };
            for (k, &(re, im)) in spectrum.iter().enumerate().take(nfreqs - 1).skip(1) {
                let angle = 2.0 * PI * ((k * i) % n) as f64 / n as f64;
                acc += 2.0 * (re * angle.cos() - im * angle.sin());
            }
            (acc / n as f64 * hamming(i)) as f32
        })
        .collect())
}

/// The low-pass filter `nnAudio` builds for downsampling in a CQT: pass band
/// up to `band_center / (1 + transition)`, stop band from
/// `band_center * (1 + transition)` (frequencies relative to Nyquist).
pub fn cqt_lowpass_filter(
    band_center: f64,
    kernel_length: usize,
    transition_bandwidth: f64,
) -> Result<Vec<f32>> {
    let pass_max = band_center / (1.0 + transition_bandwidth);
    let stop_min = band_center * (1.0 + transition_bandwidth);
    firwin2(
        kernel_length,
        &[0.0, pass_max, stop_min, 1.0],
        &[1.0, 1.0, 0.0, 0.0],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_i0_known_values() {
        // Abramowitz & Stegun table 9.8.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-14);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_442).abs() < 1e-11);
    }

    #[test]
    fn filter_has_unit_dc_gain_and_is_symmetric() {
        for (cutoff, hw, k) in [(0.25, 0.3, 12), (0.25, 0.3, 11), (1.0 / 6.0, 0.2, 18)] {
            let f = kaiser_sinc_filter(cutoff, hw, k).unwrap();
            let sum: f32 = f.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
            for i in 0..k / 2 {
                assert!((f[i] - f[k - 1 - i]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn rejects_invalid_cutoff() {
        assert!(kaiser_sinc_filter(0.0, 0.3, 12).is_err());
        assert!(kaiser_sinc_filter(0.6, 0.3, 12).is_err());
    }
}
