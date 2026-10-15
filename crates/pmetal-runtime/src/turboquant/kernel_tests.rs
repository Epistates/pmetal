//! The quantize-and-pack and dequantize kernels against a plain f64 oracle
//! that spells the packed layout out independently.

#![cfg(test)]

use super::quantizer::{Quantizer, words_per_row};
use super::reference_tests::{Gauss, to_host};
use crate::InlineArray;

/// Indices of one packed row, read out of the little-endian bitstream the
/// layout specifies: coordinate `i` is bits `[i * bits, (i + 1) * bits)`.
fn unpack_row(words: &[u32], dim: usize, bits: u8) -> Vec<u32> {
    let bits = usize::from(bits);
    (0..dim)
        .map(|i| {
            let mut value = 0u32;
            for b in 0..bits {
                let pos = i * bits + b;
                let bit = (words[pos / 32] >> (pos % 32)) & 1;
                value |= bit << b;
            }
            value
        })
        .collect()
}

fn gaussian_rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut g = Gauss::new(seed);
    (0..n * dim).map(|_| g.normal() as f32).collect()
}

fn packed_words(array: &InlineArray) -> Vec<u32> {
    array.eval();
    array.as_slice::<u32>().to_vec()
}

#[test]
fn quantize_pack_agrees_with_the_oracle_for_every_width_and_head_dim() {
    for &dim in &[32usize, 64, 80, 96, 128, 192, 256] {
        for bits in 1..=8u8 {
            let q = Quantizer::new(dim, bits);
            let n = 17usize;
            let rows = gaussian_rows(n, dim, 1000 + dim as u64 + u64::from(bits));
            let input = InlineArray::from_f32_slice(&rows, &[n as i32, dim as i32]);
            let encoded = q.encode(&input).expect("encode");
            let words = words_per_row(dim, bits);
            assert_eq!(encoded.packed.shape(), &[n as i32, words as i32]);
            assert_eq!(encoded.scales.shape(), &[n as i32]);

            let rotated = to_host(&q.rotate(&input));
            let packed = packed_words(&encoded.packed);
            let scales = to_host(&encoded.scales);
            let codebook = q.codebook();
            let thresholds: Vec<f64> = codebook
                .windows(2)
                .map(|w| 0.5 * (f64::from(w[0]) + f64::from(w[1])))
                .collect();
            let mut near_threshold_flips = 0usize;
            for row in 0..n {
                let r = &rotated[row * dim..(row + 1) * dim];
                let norm: f64 = r.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
                let got = unpack_row(&packed[row * words..(row + 1) * words], dim, bits);
                // The stored scale is the norm times the least-squares gain of
                // the reconstruction the stored indices describe.
                let (mut dot, mut centroid_sq) = (0.0f64, 0.0f64);
                for (i, &idx) in got.iter().enumerate() {
                    let c = f64::from(codebook[idx as usize]);
                    dot += f64::from(r[i]) / norm * c;
                    centroid_sq += c * c;
                }
                let want_scale = norm * dot / centroid_sq;
                assert!(
                    (f64::from(scales[row]) - want_scale).abs() <= 1e-5 * want_scale,
                    "dim {dim} bits {bits} row {row}: scale {} vs {want_scale}",
                    scales[row]
                );
                for (i, &idx) in got.iter().enumerate() {
                    let y = f64::from(r[i]) / norm;
                    let want = thresholds.iter().filter(|&&t| y > t).count() as u32;
                    if idx != want {
                        // Only a coordinate within rounding of a boundary may
                        // fall on the other side of it.
                        let nearest = thresholds
                            .iter()
                            .map(|t| (y - t).abs())
                            .fold(f64::INFINITY, f64::min);
                        assert!(
                            nearest < 1e-5 && idx.abs_diff(want) == 1,
                            "dim {dim} bits {bits} row {row} coord {i}: index {idx}, oracle {want}, \
                             distance to a threshold {nearest}"
                        );
                        near_threshold_flips += 1;
                    }
                }
            }
            assert!(
                near_threshold_flips <= n * dim / 500 + 1,
                "dim {dim} bits {bits}: {near_threshold_flips} boundary flips"
            );
            // Padding bits past the last coordinate stay zero.
            let used = dim * usize::from(bits);
            if used % 32 != 0 {
                for row in 0..n {
                    let last = packed[row * words + words - 1];
                    assert_eq!(
                        last >> (used % 32),
                        0,
                        "dim {dim} bits {bits}: dirty padding"
                    );
                }
            }
        }
    }
}

#[test]
fn dequantize_is_norm_times_codebook_entry() {
    for &dim in &[64usize, 96, 128] {
        for bits in [1u8, 2, 3, 4, 5, 8] {
            let q = Quantizer::new(dim, bits);
            let n = 9usize;
            let rows = gaussian_rows(n, dim, 77 + u64::from(bits));
            let encoded = q
                .encode(&InlineArray::from_f32_slice(&rows, &[n as i32, dim as i32]))
                .expect("encode");
            let rotated_back = to_host(&q.decode_rotated(&encoded).expect("dequantize"));
            let packed = packed_words(&encoded.packed);
            let scales = to_host(&encoded.scales);
            let words = words_per_row(dim, bits);
            for row in 0..n {
                let idx = unpack_row(&packed[row * words..(row + 1) * words], dim, bits);
                for i in 0..dim {
                    let want = scales[row] * q.codebook()[idx[i] as usize];
                    assert_eq!(
                        rotated_back[row * dim + i],
                        want,
                        "dim {dim} bits {bits} row {row} coord {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn round_trip_error_follows_the_lloyd_max_distortion() {
    // The expected squared error of a b-bit quantizer relative to the squared
    // norm is the codebook's distortion (the paper's Theorem 1 bounds it by
    // sqrt(3) * pi / 2 * 4^-b). Measure it through the whole path: rotate,
    // quantize, pack, unpack, rotate back.
    let dim = 128usize;
    let n = 512usize;
    let rows = gaussian_rows(n, dim, 5);
    let input = InlineArray::from_f32_slice(&rows, &[n as i32, dim as i32]);
    let optimum = [
        (1u8, 0.36086),
        (2, 0.11600),
        (3, 0.033966),
        (4, 0.0093148),
        (5, 0.0024515),
        (6, 0.00063),
        (8, 0.0000403),
    ];
    for (bits, distortion) in optimum {
        let q = Quantizer::new(dim, bits);
        let decoded = to_host(
            &q.decode(&q.encode(&input).expect("encode"))
                .expect("decode"),
        );
        let mut err = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in rows.iter().zip(&decoded) {
            err += f64::from(a - b).powi(2);
            den += f64::from(*a).powi(2);
        }
        let measured = err / den;
        assert!(
            (measured / distortion - 1.0).abs() < 0.08,
            "bits {bits}: relative squared error {measured:.7}, codebook distortion {distortion:.7}"
        );
        let paper_bound = 3f64.sqrt() * std::f64::consts::PI / 2.0 / 4f64.powi(i32::from(bits));
        assert!(
            measured <= paper_bound,
            "bits {bits}: {measured} above the paper's bound"
        );
    }
}

#[test]
fn the_stored_scale_removes_the_per_row_error_in_the_reconstructions_length() {
    // A Lloyd-Max reconstruction is shorter than its vector by an amount that
    // varies from row to row (rows with coordinates in the tails lose the
    // most), and the logit of a key is scaled by it. The least-squares gain
    // the kernel folds into the stored scale removes the part of it that
    // varies: a key's inner product with itself, in units of its own squared
    // length, scatters far less than with the plain norm.
    for &bits in &[3u8, 4, 5, 6] {
        let (dim, n) = (128usize, 2000usize);
        let rows = gaussian_rows(n, dim, 17);
        let q = Quantizer::new(dim, bits);
        let encoded = q
            .encode(&InlineArray::from_f32_slice(&rows, &[n as i32, dim as i32]))
            .expect("encode");
        let with_gain = to_host(&q.decode(&encoded).expect("decode"));
        let exact_scales: Vec<f32> = rows
            .chunks(dim)
            .map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        let plain = super::quantizer::Encoded {
            packed: encoded.packed.clone(),
            scales: InlineArray::from_f32_slice(&exact_scales, &[n as i32]),
        };
        let without_gain = to_host(&q.decode(&plain).expect("decode"));
        let scatter = |decoded: &[f32]| -> f64 {
            let ratios: Vec<f64> = rows
                .chunks(dim)
                .zip(decoded.chunks(dim))
                .map(|(k, k_hat)| {
                    let dot: f64 = k
                        .iter()
                        .zip(k_hat)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                    let sq: f64 = k.iter().map(|a| f64::from(*a).powi(2)).sum();
                    dot / sq
                })
                .collect();
            let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
            (ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64).sqrt()
        };
        let (a, b) = (scatter(&with_gain), scatter(&without_gain));
        assert!(
            a < 0.7 * b,
            "{bits} bits: <k, k_hat> / |k|^2 scatters by {a:.5} with the gain, {b:.5} without"
        );
    }
}

#[test]
fn degenerate_rows_are_stored_as_zero_vectors() {
    let dim = 64usize;
    let mut rows = gaussian_rows(4, dim, 9);
    for v in &mut rows[dim..2 * dim] {
        *v = 0.0;
    }
    rows[2 * dim + 3] = f32::NAN;
    rows[3 * dim + 5] = f32::INFINITY;
    let q = Quantizer::new(dim, 4);
    let encoded = q
        .encode(&InlineArray::from_f32_slice(&rows, &[4, dim as i32]))
        .expect("encode");
    let scales = to_host(&encoded.scales);
    assert!(scales[0] > 0.0);
    assert_eq!(&scales[1..], &[0.0, 0.0, 0.0]);
    let decoded = to_host(&q.decode(&encoded).expect("decode"));
    assert!(decoded[..dim].iter().any(|v| *v != 0.0));
    assert!(
        decoded[dim..].iter().all(|v| *v == 0.0),
        "zero rows must decode to zero"
    );
}

#[test]
fn rotation_matrices_are_orthogonal() {
    for &dim in &[64usize, 80, 96, 128, 192, 256] {
        let pi = super::math::rotation_matrix(dim);
        for a in (0..dim).step_by(7) {
            for b in (0..dim).step_by(5) {
                let dot: f64 = (0..dim)
                    .map(|k| f64::from(pi[a * dim + k]) * f64::from(pi[b * dim + k]))
                    .sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 2e-6, "dim {dim} rows {a},{b}: {dot}");
            }
        }
    }
}
