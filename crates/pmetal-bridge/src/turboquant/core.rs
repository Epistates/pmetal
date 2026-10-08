//! Per-dimension TurboQuant core: rotation, QJL projection, and codebooks.
//!
//! A `TurboQuantCore` is the immutable static data that drives a single
//! `(dim, max_mse_bits)` slot of a [`super::TurboQuantState`]. Construction is
//! expensive (QR decomposition + Lloyd-Max) and deterministic; share via `Arc`
//! across every layer that uses the same shape.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::InlineArray;
use crate::compat::Dtype;

use super::TURBOQUANT_SEED;
use super::dim_uses_fwht;
use super::math::{
    beta_codebook, generate_rademacher_signs, generate_random_orthogonal,
    generate_random_projection, inline_array_to_f32_vec, matmul_rows, matrix_to_inline_array,
    signed_fwht_forward, transpose_square_matrix,
};

/// Per-dimension core: rotation, QJL projection matrix, and codebooks.
///
/// Expensive to construct (random QR decomposition + Lloyd-Max), but
/// can be shared cheaply via `Arc` across heads and layers.
#[derive(Debug)]
pub struct TurboQuantCore {
    /// Number of dimensions this core handles.
    pub(super) dim: usize,
    /// Row-major [dim × dim] orthogonal rotation matrix Π (f32).
    rotation: Vec<f32>,
    /// Row-major [dim × dim] inverse (= transpose) of rotation.
    inverse_rotation: Vec<f32>,
    /// Row-major [dim × dim] Gaussian random projection J for QJL.
    qjl_projection: Vec<f32>,
    /// Row-major [dim × dim] transpose of qjl_projection.
    inverse_qjl_projection: Vec<f32>,
    /// Optional signed-FWHT rotation signs for power-of-two dims.
    wht_left_signs: Option<Vec<f32>>,
    /// Optional signed-FWHT rotation signs for power-of-two dims.
    wht_right_signs: Option<Vec<f32>>,
    /// Optional signed-FWHT QJL signs for power-of-two dims.
    qjl_wht_left_signs: Option<Vec<f32>>,
    /// Optional signed-FWHT QJL signs for power-of-two dims.
    qjl_wht_right_signs: Option<Vec<f32>>,
    /// `codebooks[b]` holds the 2^b sorted centroids for `b`-bit quantisation.
    /// Index 0 is unused (0-bit is a degenerate case).
    codebooks: Vec<Vec<f32>>,
    /// InlineArray view of the rotation matrix for GPU-accelerated matmul.
    rotation_arr: Option<InlineArray>,
    /// Bfloat16 view of the rotation matrix for bf16 output reconstruction.
    rotation_arr_bf16: Option<InlineArray>,
    /// InlineArray view of the inverse rotation matrix.
    pub(super) inverse_rotation_arr: Option<InlineArray>,
    /// InlineArray view of the QJL projection matrix.
    qjl_arr: Option<InlineArray>,
    /// InlineArray view of the inverse QJL projection matrix.
    pub(super) inverse_qjl_arr: Option<InlineArray>,
    /// Horizontally-stacked `[inverse_rotation | inverse_qjl]` of shape
    /// `[dim, 2*dim]`. Lets the hot-path attention kernel do one fused
    /// matmul for (query_rot, query_proj) instead of two separate
    /// dispatches — saves one dispatch per layer per decode step.
    stacked_inv_rot_qjl_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT rotation signs for D=256 experiments.
    wht_left_signs_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT rotation signs for D=256 experiments.
    wht_right_signs_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT QJL signs for D=256 experiments.
    qjl_wht_left_signs_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT QJL signs for D=256 experiments.
    qjl_wht_right_signs_arr: Option<InlineArray>,
    /// GPU-side codebook arrays: `codebook_arrs[b]` is a 1-D f32 InlineArray
    /// holding 2^b centroids for b-bit quantisation.  Indexed as codebooks.
    codebook_arrs: Vec<Option<InlineArray>>,
    /// 256-entry zero-padded codebook arrays for use by fullbyte score
    /// kernels that load all 256 centroids into threadgroup memory regardless
    /// of `bits`. `codebook_arrs_padded_256[b]` has the first `2^b` entries
    /// matching `codebook_arrs[b]`, the rest zeros. Built lazily-eagerly at
    /// construction for every `b` where the regular codebook exists.
    codebook_arrs_padded_256: Vec<Option<InlineArray>>,
}

impl TurboQuantCore {
    pub(super) fn new(dim: usize, max_mse_bits: u8) -> Self {
        let mut rng = StdRng::seed_from_u64(TURBOQUANT_SEED ^ ((dim as u64) << 32));

        // Pow2 dims (every transformer head_dim worth optimizing for) skip the
        // four [d×d] dense matrices entirely — signed-FWHT replaces them at
        // O(d log d) compute. At d=256 that's a ~1 MB allocation saved per
        // core, plus four GPU upload+eval()s that no longer happen.
        let use_fwht = dim_uses_fwht(dim);
        let (rotation, inverse_rotation, qjl_projection, inverse_qjl_projection) = if use_fwht {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        } else {
            let rot = generate_random_orthogonal(dim, &mut rng);
            let inv_rot = transpose_square_matrix(&rot, dim);
            let qjl = generate_random_projection(dim, &mut rng);
            let inv_qjl = transpose_square_matrix(&qjl, dim);
            (rot, inv_rot, qjl, inv_qjl)
        };
        let (wht_left_signs, wht_right_signs, qjl_wht_left_signs, qjl_wht_right_signs) = if use_fwht
        {
            let mut wht_rng = StdRng::seed_from_u64(TURBOQUANT_SEED ^ 0x5748_5400 ^ dim as u64);
            (
                Some(generate_rademacher_signs(dim, &mut wht_rng)),
                Some(generate_rademacher_signs(dim, &mut wht_rng)),
                Some(generate_rademacher_signs(dim, &mut wht_rng)),
                Some(generate_rademacher_signs(dim, &mut wht_rng)),
            )
        } else {
            (None, None, None, None)
        };

        let mut codebooks = vec![Vec::new(); usize::from(max_mse_bits) + 1];
        for bits in 1..=max_mse_bits {
            codebooks[usize::from(bits)] = (*beta_codebook(dim, bits)).clone();
        }

        // Build InlineArray GPU matrices for the dense fallback path. With
        // FWHT default we skip these for pow2 dim — the rotate_*_array hot
        // paths return early via the FWHT branch and never read these arrays.
        let (rotation_arr, rotation_arr_bf16, inverse_rotation_arr, qjl_arr, inverse_qjl_arr) =
            if use_fwht {
                (None, None, None, None, None)
            } else {
                let rot_arr = matrix_to_inline_array(&rotation, dim);
                let rot_arr_bf16 = rot_arr.as_ref().map(|arr| {
                    let cast = arr.as_dtype(Dtype::Bfloat16.as_i32());
                    cast.eval();
                    cast
                });
                let inv_rot_arr = matrix_to_inline_array(&inverse_rotation, dim);
                let qjl_arr_local = matrix_to_inline_array(&qjl_projection, dim);
                let inv_qjl_arr = matrix_to_inline_array(&inverse_qjl_projection, dim);
                (
                    rot_arr,
                    rot_arr_bf16,
                    inv_rot_arr,
                    qjl_arr_local,
                    inv_qjl_arr,
                )
            };
        // Stacked [inv_rot | inv_qjl] is only useful as a fused matmul for the
        // dense path; FWHT decomposes the same operation across two FWHT calls
        // already. Skip building it under FWHT.
        let stacked_inv_rot_qjl_arr = if use_fwht {
            None
        } else if let (Some(rot), Some(qjl)) = (&inverse_rotation_arr, &inverse_qjl_arr) {
            let stacked = crate::compat::ops::concatenate_axis(&[rot, qjl], -1);
            stacked.eval();
            Some(stacked)
        } else {
            None
        };
        let signs_to_inline_array = |signs: &Option<Vec<f32>>| {
            if dim_uses_fwht(dim) {
                signs.as_ref().map(|values| {
                    let arr = InlineArray::from_f32_slice(values, &[dim as i32]);
                    arr.eval();
                    arr
                })
            } else {
                None
            }
        };
        let wht_left_signs_arr = signs_to_inline_array(&wht_left_signs);
        let wht_right_signs_arr = signs_to_inline_array(&wht_right_signs);
        let qjl_wht_left_signs_arr = signs_to_inline_array(&qjl_wht_left_signs);
        let qjl_wht_right_signs_arr = signs_to_inline_array(&qjl_wht_right_signs);

        // GPU-side codebooks: each is a tiny 1-D f32 array (16 elements for 4-bit).
        let codebook_arrs: Vec<Option<InlineArray>> = codebooks
            .iter()
            .map(|cb| {
                if cb.is_empty() {
                    None
                } else {
                    let arr = InlineArray::from_f32_slice(cb, &[cb.len() as i32]);
                    arr.eval();
                    Some(arr)
                }
            })
            .collect();

        // 256-padded variants for fullbyte score kernels (see field doc).
        // `b == 8` reuses the regular codebook (already 256 entries); `b < 8`
        // gets a zero-padded copy. Cost: ≤8 * 1KB = 8KB per core.
        let codebook_arrs_padded_256: Vec<Option<InlineArray>> = codebooks
            .iter()
            .map(|cb| {
                if cb.is_empty() {
                    None
                } else if cb.len() >= 256 {
                    let arr = InlineArray::from_f32_slice(cb, &[cb.len() as i32]);
                    arr.eval();
                    Some(arr)
                } else {
                    let mut padded = cb.clone();
                    padded.resize(256, 0.0f32);
                    let arr = InlineArray::from_f32_slice(&padded, &[256i32]);
                    arr.eval();
                    Some(arr)
                }
            })
            .collect();

        Self {
            dim,
            rotation,
            inverse_rotation,
            qjl_projection,
            inverse_qjl_projection,
            wht_left_signs,
            wht_right_signs,
            qjl_wht_left_signs,
            qjl_wht_right_signs,
            codebooks,
            rotation_arr,
            rotation_arr_bf16,
            inverse_rotation_arr,
            qjl_arr,
            inverse_qjl_arr,
            stacked_inv_rot_qjl_arr,
            wht_left_signs_arr,
            wht_right_signs_arr,
            qjl_wht_left_signs_arr,
            qjl_wht_right_signs_arr,
            codebook_arrs,
            codebook_arrs_padded_256,
        }
    }

    pub(super) fn codebook(&self, bits: u8) -> &[f32] {
        &self.codebooks[usize::from(bits)]
    }

    pub(super) fn codebook_arr(&self, bits: u8) -> Option<&InlineArray> {
        self.codebook_arrs.get(usize::from(bits))?.as_ref()
    }

    /// Codebook arr zero-padded to 256 entries for fullbyte score kernels
    /// that hardcode `kKeyCentroids = 256u`. Encoder still uses
    /// `codebook_arr(bits)` because `gpu_quantize_mse` reads
    /// `n_centroids = cb_arr.shape()[0]` and we want exactly `2^bits`
    /// candidate centroids during nearest-neighbour search. The padded
    /// view is read-only by the score kernel; out-of-range entries are
    /// never indexed because all stored u8 indices are in `[0, 2^bits)`.
    pub(super) fn codebook_arr_padded_256(&self, bits: u8) -> Option<&InlineArray> {
        self.codebook_arrs_padded_256
            .get(usize::from(bits))?
            .as_ref()
    }

    /// GPU-native nearest-centroid quantisation via fused Metal kernel.
    ///
    /// `rotated`: `[N, D]` f32 — already normalised and rotated.
    /// Returns `[N, D]` uint32 indices on success.
    ///
    /// The fused kernel eliminates the `[N, D, C]` intermediate tensor that
    /// the old expand_dims+subtract+square+argmin chain allocated.  Falls back
    /// to the ops-based path if Metal is unavailable or n_centroids > 16.
    pub(super) fn gpu_quantize_mse(&self, rotated: &InlineArray, bits: u8) -> Option<InlineArray> {
        let cb_arr = self.codebook_arr(bits)?;
        let shape = rotated.shape();
        let ndim = shape.len();
        // Compute flat N = product of all dimensions except last.
        let n_rows: i32 = shape[..ndim - 1].iter().product();
        let dim = shape[ndim - 1] as u32;
        let n_centroids = cb_arr.shape()[0] as u32;

        // Reshape to [N, D] for the kernel (kernel expects exactly 2-D input).
        // The caller may pass higher-rank tensors like [B, H, S, D].
        let flat = if ndim == 2 {
            // Already [N, D] — no copy.
            None
        } else {
            Some(rotated.reshape(&[n_rows, dim as i32]))
        };
        let input_2d = flat.as_ref().unwrap_or(rotated);

        if let Some(indices_2d) =
            InlineArray::turboquant_encode(input_2d, cb_arr, dim, n_centroids, n_rows as u32)
        {
            // Reshape back to original leading dims + D.
            if ndim == 2 {
                return Some(indices_2d);
            }
            let mut out_shape: Vec<i32> = shape[..ndim - 1].to_vec();
            out_shape.push(dim as i32);
            return Some(indices_2d.reshape(&out_shape));
        }

        // Ops fallback — original expand_dims+argmin path.
        let cb_arr = self.codebook_arr(bits)?;
        let expanded = rotated.expand_dims(-1);
        let diffs = expanded.subtract(cb_arr);
        let sq = diffs.multiply(&diffs);
        Some(sq.argmin(-1))
    }

    /// GPU-native codebook reconstruction via fused Metal kernel.
    ///
    /// `indices`: `[..., D]` uint32.  Returns `[..., D]` f32 of centroid values.
    ///
    /// The fused kernel eliminates the flatten→take→reshape round-trip.
    /// Falls back to the ops-based take_axis path if Metal is unavailable.
    pub(super) fn gpu_reconstruct_mse(
        &self,
        indices: &InlineArray,
        bits: u8,
    ) -> Option<InlineArray> {
        let cb_arr = self.codebook_arr(bits)?;
        let shape = indices.shape();
        let ndim = shape.len();
        let n_rows: i32 = shape[..ndim - 1].iter().product();
        let dim = shape[ndim - 1] as u32;
        let n_centroids = cb_arr.shape()[0] as u32;

        let flat = if ndim == 2 {
            None
        } else {
            Some(indices.reshape(&[n_rows, dim as i32]))
        };
        let indices_2d = flat.as_ref().unwrap_or(indices);

        if let Some(recon_2d) =
            InlineArray::turboquant_decode(indices_2d, cb_arr, dim, n_centroids, n_rows as u32)
        {
            if ndim == 2 {
                return Some(recon_2d);
            }
            let mut out_shape: Vec<i32> = shape[..ndim - 1].to_vec();
            out_shape.push(dim as i32);
            return Some(recon_2d.reshape(&out_shape));
        }

        // Ops fallback — original take_axis+reshape path.
        let cb_arr = self.codebook_arr(bits)?;
        let orig_shape: Vec<i32> = shape.to_vec();
        let n: i32 = orig_shape.iter().product();
        let flat_idx = indices.reshape(&[n]);
        let gathered = cb_arr.take_axis(&flat_idx, 0);
        Some(gathered.reshape(&orig_shape))
    }

    /// Rotate input rows: output = input · Π^T  (each row left-multiplied by Π).
    pub(super) fn rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        if dim_uses_fwht(self.dim) {
            if let (Some(pre), Some(post)) = (&self.wht_right_signs, &self.wht_left_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
        self.apply_transform(input, &self.rotation, &self.rotation_arr)
    }

    /// Inverse-rotate: output = input · Π.
    pub(super) fn inverse_rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        if dim_uses_fwht(self.dim) {
            if let (Some(pre), Some(post)) = (&self.wht_left_signs, &self.wht_right_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
        self.apply_transform(input, &self.inverse_rotation, &self.inverse_rotation_arr)
    }

    /// Project via Gaussian matrix J for QJL.
    pub(super) fn project_rows(&self, input: &[f32]) -> Vec<f32> {
        if dim_uses_fwht(self.dim) {
            if let (Some(pre), Some(post)) = (&self.qjl_wht_right_signs, &self.qjl_wht_left_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
        self.apply_transform(input, &self.qjl_projection, &self.qjl_arr)
    }

    /// Inverse-project via J^T.
    pub(super) fn inverse_project_rows(&self, input: &[f32]) -> Vec<f32> {
        if dim_uses_fwht(self.dim) {
            if let (Some(pre), Some(post)) = (&self.qjl_wht_left_signs, &self.qjl_wht_right_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
        self.apply_transform(input, &self.inverse_qjl_projection, &self.inverse_qjl_arr)
    }

    /// Apply a [dim × dim] linear transform to a batch of row vectors.
    ///
    /// Tries GPU matmul via InlineArray first; falls back to CPU on any failure.
    fn apply_transform(
        &self,
        input: &[f32],
        matrix_cpu: &[f32],
        matrix_gpu: &Option<InlineArray>,
    ) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        if let Some(m_arr) = matrix_gpu {
            if let Some(result) = try_gpu_matmul_rows(input, self.dim, m_arr) {
                return result;
            }
        }

        // CPU fallback — avoids GPU round-trip for tiny inputs.
        matmul_rows(matrix_cpu, self.dim, input)
    }
    /// Experimental signed-FWHT rotation path for power-of-two dims.
    pub(super) fn rotate_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.wht_right_signs.as_ref()?,
            self.wht_left_signs.as_ref()?,
            &self.wht_left_signs_arr,
            &self.wht_right_signs_arr,
        )
    }

    /// Experimental inverse signed-FWHT rotation path for power-of-two dims.
    pub(super) fn inverse_rotate_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.wht_left_signs.as_ref()?,
            self.wht_right_signs.as_ref()?,
            &self.wht_right_signs_arr,
            &self.wht_left_signs_arr,
        )
    }

    /// Experimental signed-FWHT QJL projection path for power-of-two dims.
    pub(super) fn project_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.qjl_wht_right_signs.as_ref()?,
            self.qjl_wht_left_signs.as_ref()?,
            &self.qjl_wht_left_signs_arr,
            &self.qjl_wht_right_signs_arr,
        )
    }

    /// Experimental inverse signed-FWHT QJL projection path for power-of-two dims.
    pub(super) fn inverse_project_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.qjl_wht_left_signs.as_ref()?,
            self.qjl_wht_right_signs.as_ref()?,
            &self.qjl_wht_right_signs_arr,
            &self.qjl_wht_left_signs_arr,
        )
    }

    fn apply_signed_fwht_rows(
        &self,
        input_rows: &InlineArray,
        pre_signs_cpu: &[f32],
        post_signs_cpu: &[f32],
        post_signs_gpu: &Option<InlineArray>,
        pre_signs_gpu: &Option<InlineArray>,
    ) -> Option<InlineArray> {
        if !self.dim.is_power_of_two()
            || input_rows.ndim() != 2
            || input_rows.dim(1) != self.dim as i32
        {
            return None;
        }
        let n_rows = input_rows.dim(0) as usize;

        if let (Some(post_gpu), Some(pre_gpu)) = (post_signs_gpu.as_ref(), pre_signs_gpu.as_ref()) {
            if let Some(out) = InlineArray::turboquant_signed_fwht_pow2_rows(
                input_rows,
                post_gpu,
                pre_gpu,
                self.dim as u32,
                n_rows as u32,
            ) {
                return Some(out);
            }
        }

        let mut output = inline_array_to_f32_vec(input_rows, n_rows * self.dim)?;
        for row in output.chunks_mut(self.dim) {
            signed_fwht_forward(row, post_signs_cpu, pre_signs_cpu);
        }
        Some(InlineArray::from_f32_slice(
            &output,
            &[n_rows as i32, self.dim as i32],
        ))
    }

    fn apply_array_transform_rows(
        &self,
        input: &InlineArray,
        matrix_gpu: &Option<InlineArray>,
        wht_impl: impl FnOnce(&Self, &InlineArray) -> Option<InlineArray>,
    ) -> Option<InlineArray> {
        let shape = input.shape();
        if shape.is_empty() || *shape.last()? != self.dim as i32 {
            return None;
        }
        let ndim = shape.len();
        let n_rows: i32 = if ndim == 1 {
            1
        } else {
            shape[..ndim - 1].iter().product()
        };
        let input_rows = if ndim == 2 {
            input.clone()
        } else {
            input.reshape(&[n_rows, self.dim as i32])
        };
        let output_rows = if dim_uses_fwht(self.dim) {
            let input_rows_f32 = if input_rows.dtype_raw() == Dtype::Float32.as_i32() {
                input_rows
            } else {
                input_rows.as_dtype(Dtype::Float32.as_i32())
            };
            wht_impl(self, &input_rows_f32)?
        } else {
            let matrix = matrix_gpu.as_ref()?;
            input_rows.matmul(matrix)
        };
        if ndim == 2 {
            Some(output_rows)
        } else {
            Some(output_rows.reshape(shape))
        }
    }

    pub(super) fn rotate_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.inverse_rotation_arr, |core, rows| {
            core.rotate_rows_wht(rows)
        })
    }

    /// Fused rotate + project: computes `input @ [inv_rotation | inv_qjl]`
    /// as a single [N, 2*dim] matmul and splits the result back into
    /// `(query_rot, query_proj)` of shape [N, dim] each. Saves one op
    /// dispatch per decode step compared to calling `rotate_array` and
    /// `project_array` separately. Returns `None` if the stacked matrix
    /// wasn't built at construction (falls back to separate paths in
    /// the caller), or if the FWHT fast path is enabled (that path
    /// stays unchanged).
    pub(super) fn rotate_and_project_array(
        &self,
        input: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        if dim_uses_fwht(self.dim) {
            return None;
        }
        let stacked = self.stacked_inv_rot_qjl_arr.as_ref()?;
        let shape = input.shape();
        let ndim = shape.len();
        if ndim == 0 {
            return None;
        }
        let last_dim = shape[ndim - 1];
        if last_dim as usize != self.dim {
            return None;
        }
        let n: i32 = shape[..ndim - 1].iter().product();
        let input_2d = input.reshape(&[n, last_dim]);
        let fused = input_2d.matmul(stacked); // [n, 2*dim]
        let dim_i32 = self.dim as i32;
        let rot = fused.slice(&[0, 0], &[n, dim_i32]);
        let proj = fused.slice(&[0, dim_i32], &[n, 2 * dim_i32]);
        Some((rot, proj))
    }

    pub(super) fn inverse_rotate_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.rotation_arr, |core, rows| {
            core.inverse_rotate_rows_wht(rows)
        })
    }

    pub(super) fn project_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.inverse_qjl_arr, |core, rows| {
            core.project_rows_wht(rows)
        })
    }

    pub(super) fn inverse_project_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.qjl_arr, |core, rows| {
            core.inverse_project_rows_wht(rows)
        })
    }

    pub(super) fn inverse_rotate_output_array(
        &self,
        input: &InlineArray,
        output_dtype: i32,
    ) -> Option<InlineArray> {
        if dim_uses_fwht(self.dim) {
            let output = self.inverse_rotate_array(input)?;
            if output_dtype == Dtype::Bfloat16.as_i32() {
                Some(output.as_dtype(output_dtype))
            } else {
                Some(output)
            }
        } else if output_dtype == Dtype::Bfloat16.as_i32() {
            let input_bf16 = input.as_dtype(output_dtype);
            let rotation_bf16 = self.rotation_arr_bf16.as_ref()?;
            Some(input_bf16.matmul(rotation_bf16))
        } else {
            let rotation = self.rotation_arr.as_ref()?;
            Some(input.matmul(rotation))
        }
    }
}

/// Attempt a GPU matrix-multiply: input [N, dim] × matrix [dim, dim] → [N, dim].
///
/// Input is uploaded to GPU as f32, matmul is performed in f32, result is
/// copied back to a Vec<f32>.  Returns `None` if shape or dtype conversion fails.
fn try_gpu_matmul_rows(input: &[f32], dim: usize, matrix: &InlineArray) -> Option<Vec<f32>> {
    let n = input.len() / dim;
    if n == 0 || dim == 0 {
        return None;
    }

    // Upload input [N, dim] as f32, multiply by pre-built matrix [dim, dim].
    let input_arr = InlineArray::from_f32_slice(input, &[n as i32, dim as i32]);
    let result_arr = input_arr.matmul(matrix);
    inline_array_to_f32_vec(&result_arr, n * dim)
}
