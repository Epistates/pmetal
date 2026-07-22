//! YARN (Yet-another-RoPE-extensioN) frequency + scale math, shared across
//! architectures whose reference is mlx-lm's `rope_utils.YarnRoPE`.
//!
//! DeepSeek-V3 and GPT-OSS both extend context with YARN and compute the exact
//! same per-dimension inverse frequencies and embedding `mscale`; only the RoPE
//! *application* differs (DeepSeek uses interleaved/`traditional=true` and folds
//! `mscale²` into the softmax scale; GPT-OSS uses split-half/`traditional=false`
//! and scales q/k by `mscale` before rotation). The freq/scale derivation here
//! is byte-for-byte the mlx-lm formula so both callers stay parity-exact.
//!
//! Callers feed [`YarnRope::inv_freq`] to
//! [`pmetal_mlx::kernels::rope::apply_rope_with_freqs`] and apply
//! [`YarnRope::mscale`] to the input themselves (the two archs disagree on
//! whether/where mscale lands, so it is not baked in here).

use pmetal_bridge::compat::Array;

/// YARN attention/length scale: `0.1 * mscale * ln(scale) + 1` (1.0 for scale ≤ 1).
///
/// Mirrors mlx-lm `YarnRoPE.yarn_get_mscale`.
pub fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn yarn_find_correction_dim(num_rotations: f32, dim: i32, base: f32, max_pos: i32) -> f32 {
    (dim as f32 * (max_pos as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

fn yarn_find_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: i32,
    base: f32,
    max_pos: i32,
) -> (f32, f32) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_pos).floor();
    let high = yarn_find_correction_dim(high_rot, dim, base, max_pos).ceil();
    (low.max(0.0), high.min((dim - 1) as f32))
}

/// Precomputed YARN rotary state: per-dimension inverse frequencies (consumed by
/// `apply_rope_with_freqs`) plus the embedding mscale applied to q/k before
/// rotation.
#[derive(Debug, Clone)]
pub struct YarnRope {
    /// `[dim/2]` inverse frequencies (angular), one per rotary pair.
    pub inv_freq: Array,
    /// Embedding scale applied to q/k before rotation (`1.0` = no-op).
    pub mscale: f32,
}

/// Build the YARN per-dimension inverse frequencies and embedding mscale.
///
/// `scaling_factor == 1` collapses to standard RoPE (freq_inter == freq_extra).
/// Mirrors mlx-lm `YarnRoPE.__init__` exactly (correction range in *dimension*
/// space, linear ramp over the dim index).
#[allow(clippy::too_many_arguments)]
pub fn build_yarn_rope(
    dim: i32,
    base: f32,
    scaling_factor: f32,
    original_max_pos: i32,
    beta_fast: f32,
    beta_slow: f32,
    mscale: f32,
    mscale_all_dim: f32,
) -> YarnRope {
    let half = (dim / 2) as usize;
    let (low, high) = yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_pos);
    let denom = if (high - low).abs() < f32::EPSILON {
        0.001 // prevent singularity (mlx yarn_linear_ramp_mask)
    } else {
        high - low
    };
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (2 * i) as f32 / dim as f32;
        let freq_extra = base.powf(exponent);
        let freq_inter = scaling_factor * freq_extra;
        let ramp = (((i as f32) - low) / denom).clamp(0.0, 1.0);
        let freq_mask = 1.0 - ramp;
        let freqs =
            (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask));
        inv_freq.push(1.0 / freqs);
    }
    let emb_mscale =
        yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim);
    YarnRope {
        inv_freq: Array::from_slice(&inv_freq, &[half as i32]),
        mscale: emb_mscale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yarn_factor_one_is_plain_inv_freq() {
        // scaling_factor == 1 collapses to standard RoPE: inv_freq[i] =
        // base^(-2i/dim), and mscale == 1 (yarn_get_mscale returns 1 for
        // scale ≤ 1).
        let dim = 8;
        let base = 10000.0_f32;
        let yarn = build_yarn_rope(dim, base, 1.0, 4096, 32.0, 1.0, 1.0, 0.0);
        assert!((yarn.mscale - 1.0).abs() < 1e-6, "mscale {}", yarn.mscale);

        let mut inv = yarn.inv_freq.clone();
        let got = inv.to_f32_vec((dim / 2) as usize).expect("inv_freq vec");
        for (i, g) in got.iter().enumerate() {
            let expected = base.powf(-((2 * i) as f32) / dim as f32);
            assert!(
                (g - expected).abs() < 1e-4,
                "inv_freq[{i}] {g} != plain {expected}"
            );
        }
    }

    #[test]
    fn yarn_mscale_matches_reference_formula() {
        // GPT-OSS defaults (mscale=1, mscale_all_dim=0): the embedding scale is
        // yarn_get_mscale(factor, 1) / yarn_get_mscale(factor, 0)
        //   = (0.1·ln(factor) + 1) / 1.
        let factor = 32.0_f32;
        let yarn = build_yarn_rope(64, 150000.0, factor, 4096, 32.0, 1.0, 1.0, 0.0);
        let expected = 0.1 * factor.ln() + 1.0;
        assert!(
            (yarn.mscale - expected).abs() < 1e-6,
            "mscale {} != {expected}",
            yarn.mscale
        );
    }

    #[test]
    fn yarn_scaling_stretches_high_index_frequencies() {
        // With factor > 1, the highest-index (lowest-frequency) pair is fully
        // interpolated: inv_freq shrinks by ~1/factor vs the unscaled value.
        let dim = 8;
        let base = 10000.0_f32;
        let plain = build_yarn_rope(dim, base, 1.0, 4096, 32.0, 1.0, 1.0, 0.0);
        let scaled = build_yarn_rope(dim, base, 32.0, 4096, 32.0, 1.0, 1.0, 0.0);
        let mut p = plain.inv_freq.clone();
        let mut s = scaled.inv_freq.clone();
        let pv = p.to_f32_vec((dim / 2) as usize).expect("vec");
        let sv = s.to_f32_vec((dim / 2) as usize).expect("vec");
        let last = pv.len() - 1;
        assert!(
            sv[last] < pv[last],
            "scaled inv_freq[{last}]={} should be below plain {}",
            sv[last],
            pv[last]
        );
    }
}
