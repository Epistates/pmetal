//! The TurboQuant quantizer for one tensor: a random rotation, a Lloyd-Max
//! codebook and the bit-packed layout the kernels read.
//!
//! A vector `x` of `dim` coordinates is stored as one f32 scale and `dim`
//! centroid indices of `bits` bits each, packed into a little-endian
//! bitstream of `ceil(dim * bits / 32)` words:
//!
//! ```text
//! y     = Π x / |x|                Π is a fixed Haar-random orthogonal matrix
//! idx   = nearest Lloyd-Max centroid of each coordinate of y
//! gain  = <y, c> / <c, c>          c = codebook[idx], the least-squares gain
//! scale = |x| · gain
//! x̂     = scale · Πᵀ · codebook[idx]
//! ```
//!
//! The gain is not in the paper. A Lloyd-Max reconstruction is a little
//! shorter than its vector, by an amount that varies from vector to vector
//! (the more, the further out its largest coordinates lie), and the logit of
//! a key is scaled by it; the gain absorbs that, at no cost in storage (it is
//! folded into the norm), and the encoder gets it from sums it already forms.
//!
//! After the rotation every coordinate of `y` follows the same density
//! (Beta, close to Gaussian) whatever `x` was, so one codebook fits every
//! vector and nothing is calibrated on data. The mean-squared error is within
//! a constant of the information-theoretic bound (arXiv:2504.19874, Theorem 1),
//! and `<q, x̂>` is `|x| <Π q, ŷ>`: queries are rotated once per step and the
//! stored rows never are.
//!
//! The paper's second stage, a one-bit QJL sketch of the residual that makes
//! the inner-product estimate unbiased, is deliberately not part of this
//! type. The estimate it removes the bias of feeds a softmax, where the
//! variance it adds costs more accuracy than the bias it removes (see the
//! tests in `quality_tests`).

use std::sync::Arc;

use crate::InlineArray;
use crate::compat::Dtype;

use super::kernels;
use super::math::{beta_codebook, rotation_matrix};

/// A quantizer for tensors of `dim` coordinates at `bits` bits per coordinate.
#[derive(Debug)]
pub struct Quantizer {
    dim: usize,
    bits: u8,
    /// `Πᵀ`, `[dim, dim]` f32: `x · Πᵀ` rotates every row of `x`.
    rotation_t: InlineArray,
    /// `Π`, `[dim, dim]` f32: `r · Π` rotates every row of `r` back.
    rotation: InlineArray,
    /// The `2^bits` centroids, ascending, f32.
    codebook: InlineArray,
    /// The `2^bits - 1` midpoints between neighbouring centroids, f32.
    thresholds: InlineArray,
    codebook_host: Arc<Vec<f32>>,
}

/// A batch of quantized rows: `packed [N, words] u32` and `scales [N] f32`.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub(super) packed: InlineArray,
    pub(super) scales: InlineArray,
}

impl Quantizer {
    pub(super) fn new(dim: usize, bits: u8) -> Self {
        assert!(
            (1..=8).contains(&bits),
            "TurboQuant bits must be in 1..=8, got {bits}"
        );
        assert!(dim >= 2, "TurboQuant needs at least 2 coordinates");
        let rotation_host = rotation_matrix(dim);
        let codebook_host = beta_codebook(dim, bits);
        let thresholds_host: Vec<f32> = codebook_host
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        let rotation = InlineArray::from_f32_slice(&rotation_host, &[dim as i32, dim as i32]);
        let rotation_t = rotation.transpose_axes(&[1, 0]);
        let codebook = InlineArray::from_f32_slice(&codebook_host, &[codebook_host.len() as i32]);
        let thresholds =
            InlineArray::from_f32_slice(&thresholds_host, &[thresholds_host.len() as i32]);
        // Materialise the constants now so the first decode step does not
        // carry the upload.
        rotation.eval();
        rotation_t.eval();
        codebook.eval();
        thresholds.eval();
        Self {
            dim,
            bits,
            rotation_t,
            rotation,
            codebook,
            thresholds,
            codebook_host,
        }
    }

    /// Coordinates per vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bits per coordinate.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// 32-bit words one packed vector occupies.
    pub fn words(&self) -> usize {
        words_per_row(self.dim, self.bits)
    }

    /// The centroids, ascending.
    pub fn codebook(&self) -> &[f32] {
        &self.codebook_host
    }

    pub(super) fn codebook_array(&self) -> &InlineArray {
        &self.codebook
    }

    /// `Π`, the matrix that takes a rotated row back to the original basis
    /// (`r · Π`).
    pub(super) fn rotation_array(&self) -> &InlineArray {
        &self.rotation
    }

    /// Rotate the rows of `x` (`[N, dim]`, any float dtype) into the basis
    /// the packed rows live in; the result is f32.
    pub(super) fn rotate(&self, x: &InlineArray) -> InlineArray {
        let x = if x.dtype_raw() == Dtype::Float32.as_i32() {
            x.clone()
        } else {
            x.as_dtype(Dtype::Float32.as_i32())
        };
        x.matmul(&self.rotation_t)
    }

    /// Quantize `rows` (`[N, dim]`, any float dtype).
    pub(super) fn encode(&self, rows: &InlineArray) -> Result<Encoded, String> {
        let n_rows = rows.dim(0) as usize;
        debug_assert_eq!(rows.dim(1) as usize, self.dim);
        let rotated = self.rotate(rows);
        let (packed, scales) = kernels::quantize_pack(
            &rotated,
            &self.thresholds,
            &self.codebook,
            self.dim,
            self.bits,
            n_rows,
        )
        .map_err(|err| err.to_string())?;
        Ok(Encoded { packed, scales })
    }

    /// The stored rows back as vectors, `[N, dim]` f32 in the original basis.
    pub(super) fn decode(&self, encoded: &Encoded) -> Result<InlineArray, String> {
        Ok(self.decode_rotated(encoded)?.matmul(&self.rotation))
    }

    /// The stored rows as `|x| · codebook[idx]`, `[N, dim]` f32, in the
    /// rotated basis.
    pub(super) fn decode_rotated(&self, encoded: &Encoded) -> Result<InlineArray, String> {
        let n_rows = encoded.scales.dim(0) as usize;
        kernels::dequantize(
            &encoded.packed,
            &encoded.scales,
            &self.codebook,
            self.dim,
            self.bits,
            n_rows,
        )
        .map_err(|err| err.to_string())
    }
}

/// 32-bit words one packed row of `dim` coordinates at `bits` bits takes.
pub fn words_per_row(dim: usize, bits: u8) -> usize {
    (dim * usize::from(bits)).div_ceil(32)
}
