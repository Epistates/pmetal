//! The Beta-density Lloyd-Max codebooks against an independent f64 evaluation.

#![cfg(test)]

use super::beta_codebook;

/// Independent f64 evaluation of a codebook on the density of one coordinate
/// of a uniform point on the unit sphere in R^dim, f(x) ∝ (1 - x²)^((dim-3)/2).
///
/// Returns `(distortion, worst_centroid_residual)`: the expected squared
/// error of nearest-centroid quantization relative to the coordinate's
/// variance 1/dim, and the largest gap between a centroid and the mean of its
/// own cell, relative to the coordinate's standard deviation. A Lloyd-Max
/// codebook has a zero residual.
fn codebook_report(codebook: &[f32], dim: usize) -> (f64, f64) {
    let alpha = (dim as f64 - 3.0) / 2.0;
    // Midpoint rule on 4M points over the part of [-1, 1] that carries mass.
    let half_width = (14.0 / (dim as f64).sqrt()).min(1.0);
    let n = 1usize << 22;
    let step = 2.0 * half_width / n as f64;
    let mut mass = vec![0.0f64; codebook.len()];
    let mut first = vec![0.0f64; codebook.len()];
    let mut second = vec![0.0f64; codebook.len()];
    let mut total = 0.0;
    let mut k = 0usize;
    for i in 0..n {
        let x = -half_width + (i as f64 + 0.5) * step;
        while k + 1 < codebook.len()
            && (x - f64::from(codebook[k])).abs() > (x - f64::from(codebook[k + 1])).abs()
        {
            k += 1;
        }
        let w = (1.0 - x * x).max(0.0).powf(alpha);
        total += w;
        mass[k] += w;
        first[k] += w * x;
        second[k] += w * x * x;
    }
    let variance = 1.0 / dim as f64;
    let mut distortion = 0.0;
    let mut residual = 0.0f64;
    for (j, &c) in codebook.iter().enumerate() {
        let c = f64::from(c);
        distortion += second[j] - 2.0 * c * first[j] + c * c * mass[j];
        if mass[j] > 0.0 {
            residual = residual.max((c - first[j] / mass[j]).abs());
        }
    }
    (distortion / total / variance, residual / variance.sqrt())
}

#[test]
fn codebooks_are_lloyd_max_fixed_points() {
    for &dim in &[32usize, 64, 128, 256] {
        for bits in 1..=8u8 {
            let codebook = beta_codebook(dim, bits);
            let (_, residual) = codebook_report(&codebook, dim);
            assert!(
                residual < 2e-4,
                "dim {dim} bits {bits}: a centroid sits {residual:.5} standard deviations \
                 from the mean of its own cell, so the codebook is not a Lloyd-Max fixed point"
            );
        }
    }
}

#[test]
fn codebooks_are_symmetric_and_sorted() {
    for &dim in &[64usize, 128, 192] {
        for bits in 1..=8u8 {
            let codebook = beta_codebook(dim, bits);
            assert_eq!(codebook.len(), 1usize << bits);
            assert!(codebook.windows(2).all(|w| w[0] < w[1]));
            let n = codebook.len();
            for i in 0..n / 2 {
                assert_eq!(codebook[i], -codebook[n - 1 - i], "dim {dim} bits {bits}");
            }
        }
    }
}

#[test]
fn codebook_distortion_matches_gaussian_optimum_in_high_dimension() {
    // Normalised distortion of the optimal b-bit quantizer for a unit
    // Gaussian (Max, 1960; Lloyd-Max tables). The coordinate of a uniform
    // point on a high-dimensional sphere is Gaussian to a very close degree.
    let optimum = [
        (1u8, 0.3633802),
        (2, 0.1175016),
        (3, 0.0345448),
        (4, 0.0095007),
        (5, 0.0024906),
        (6, 0.0006424),
    ];
    for (bits, expected) in optimum {
        let (distortion, _) = codebook_report(&beta_codebook(4096, bits), 4096);
        let rel = (distortion - expected).abs() / expected;
        assert!(
            rel < 0.005,
            "bits {bits}: distortion {distortion:.7} vs optimum {expected:.7}"
        );
    }
    // Each extra bit divides the optimal distortion by about 4, in the head
    // dimensions models use as well.
    for &dim in &[64usize, 128, 256] {
        let mut previous = codebook_report(&beta_codebook(dim, 6), dim).0;
        for bits in 7..=8u8 {
            let d = codebook_report(&beta_codebook(dim, bits), dim).0;
            assert!(
                d < previous / 3.9,
                "dim {dim}: {bits} bits distortion {d:.7} after {previous:.7}"
            );
            previous = d;
        }
    }
}

#[test]
fn building_a_codebook_is_cheap() {
    // Every `TurboQuantState` builds one codebook per bit width up to the
    // largest it uses, at model load.
    let start = std::time::Instant::now();
    for bits in 1..=8u8 {
        let _ = super::math::build_beta_codebook(131, bits);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(3),
        "eight codebooks took {elapsed:?}"
    );
}
