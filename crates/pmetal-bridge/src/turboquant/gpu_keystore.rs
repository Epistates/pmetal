//! GPU-resident TurboQuant K/V stores (Uniform + Mixed precision paths).
//!
//! The Uniform stores ([`GpuKeyStore`], [`GpuValueStore`]) hold the scalar
//! quantised state for layers configured with a single bit-width. The Mixed
//! stores ([`GpuMixedKeyStore`], [`GpuMixedValueStore`]) split each row into
//! a regular sub-vector + an outlier sub-vector with independent codebooks.
//!
//! All tensors live on the GPU and grow along axis 2 (T) as new tokens are
//! appended. The score and dequantize kernels read these stores in-place; no
//! CPU round-trip occurs in the hot path.

use crate::InlineArray;


/// GPU-resident quantised key data for the Uniform (non-outlier) path.
///
/// All tensors live entirely on the GPU — no CPU round-trips during normal
/// operation.  Shape convention (accumulated over T steps):
///   indices:         [B, H, T, D]  uint8   — codebook index per coordinate
///   indices_t:       [B, H, D, T]  uint8   — score-friendly transposed view
///   q8_keybytes_t:   [B, H, D, T]  uint8   — q8-only packed index/sign view
///   q8_keybytes_seq: [B, H, T, D]  uint8   — q8 seq-major key shadow
///   q8_fullbyte_seq: [B, H, T, D]  uint8   — q8 seq-major pure-256-centroid key shadow
///   q8_kvbytes_seq:  [B, H, T, D]  uint16  — D256 q8 seq-major packed {key,value}
///   q8_slot_scales_seq: [B, H, T, 4]  f32  — [key_norm, residual_norm, value_norm, key_slot_scale]
///   norms:           [B, H, T, 1]  f32     — optional L2 norm before unit-sphere normalise
///   qjl_signs:       [B, H, T, ceil(D/32)]  uint32 packed sign words
///   qjl_signs_t:     [B, H, ceil(D/32), T]  uint32 transposed sign-word view
///   residual_norms:  [B, H, T, 1]  f32     — optional unscaled residual L2 norm
///   key_slot_scale:  [B, H, T, 1]  f32     — per-row codebook scaling factor
///                                           (max(|rotated|) / centroid_max).
///                                           Populated only when q8_slot_scales_seq is None;
///                                           otherwise read from component 3 of the pack.
#[derive(Debug, Clone)]
pub(super) struct GpuKeyStore {
    pub(super) indices: InlineArray,
    pub(super) indices_t: Option<InlineArray>,
    pub(super) q8_keybytes_t: Option<InlineArray>,
    pub(super) q8_keybytes_seq: Option<InlineArray>,
    pub(super) q8_fullbyte_seq: Option<InlineArray>,
    pub(super) q8_kvbytes_seq: Option<InlineArray>,
    pub(super) q8_slot_scales_seq: Option<InlineArray>,
    pub(super) norms: Option<InlineArray>,
    pub(super) qjl_signs: InlineArray,
    pub(super) qjl_signs_t: Option<InlineArray>,
    pub(super) residual_norms: Option<InlineArray>,
    pub(super) key_slot_scale: Option<InlineArray>,
}

impl GpuKeyStore {
    /// Concatenate a new step's GPU arrays along the T (axis 2) dimension.
    pub(super) fn append(&mut self, new: GpuKeyStore) {
        self.indices = self.indices.kv_cache_append(&new.indices, 2);
        self.indices_t = match (self.indices_t.take(), new.indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.q8_keybytes_t = match (self.q8_keybytes_t.take(), new.q8_keybytes_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.q8_keybytes_seq = match (self.q8_keybytes_seq.take(), new.q8_keybytes_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_fullbyte_seq = match (self.q8_fullbyte_seq.take(), new.q8_fullbyte_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_kvbytes_seq = match (self.q8_kvbytes_seq.take(), new.q8_kvbytes_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_slot_scales_seq = match (self.q8_slot_scales_seq.take(), new.q8_slot_scales_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.norms = match (self.norms.take(), new.norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            _ => None,
        };
        self.qjl_signs = self.qjl_signs.kv_cache_append(&new.qjl_signs, 2);
        self.qjl_signs_t = match (self.qjl_signs_t.take(), new.qjl_signs_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.residual_norms = match (self.residual_norms.take(), new.residual_norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            _ => None,
        };
        self.key_slot_scale = match (self.key_slot_scale.take(), new.key_slot_scale) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            _ => None,
        };
    }

    pub(super) fn cache_seq_capacity(&self) -> i32 {
        self.q8_kvbytes_seq
            .as_ref()
            .map(|arr| arr.dim(2))
            .or_else(|| self.q8_keybytes_seq.as_ref().map(|arr| arr.dim(2)))
            .or_else(|| self.indices_t.as_ref().map(|arr| arr.dim(3)))
            .unwrap_or_else(|| self.indices.dim(2))
    }

    pub(super) fn indices_t_array(&self) -> InlineArray {
        self.indices_t
            .clone()
            .unwrap_or_else(|| self.indices.transpose_axes(&[0, 1, 3, 2]))
    }

    pub(super) fn qjl_signs_t_array(&self) -> InlineArray {
        self.qjl_signs_t
            .clone()
            .unwrap_or_else(|| self.qjl_signs.transpose_axes(&[0, 1, 3, 2]))
    }

    pub(super) fn qjl_words(&self) -> i32 {
        self.qjl_signs_t
            .as_ref()
            .map(|arr| arr.dim(2))
            .unwrap_or_else(|| self.qjl_signs.dim(3))
    }

    pub(super) fn slot_scale_component_array(&self, component: i32) -> Option<InlineArray> {
        let slot_scales = self.q8_slot_scales_seq.as_ref()?;
        Some(slot_scales.slice(
            &[0, 0, 0, component],
            &[
                slot_scales.dim(0),
                slot_scales.dim(1),
                slot_scales.dim(2),
                component + 1,
            ],
        ))
    }

    pub(super) fn key_norms_array(&self) -> Option<InlineArray> {
        self.norms
            .clone()
            .or_else(|| self.slot_scale_component_array(0))
    }

    pub(super) fn residual_norms_array(&self) -> Option<InlineArray> {
        self.residual_norms
            .clone()
            .or_else(|| self.slot_scale_component_array(1))
    }

    /// Per-row codebook scaling factor (`max(|rotated|) / centroid_max`).
    ///
    /// Stored in `key_slot_scale` for the standard (non-q8-shadow) path; for
    /// the q8 seq-shadow path it lives at component 3 of `q8_slot_scales_seq`.
    pub(super) fn key_slot_scale_array(&self) -> Option<InlineArray> {
        self.key_slot_scale
            .clone()
            .or_else(|| self.slot_scale_component_array(3))
    }

    pub(super) fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        out.push(&mut self.indices);
        if let Some(indices_t) = self.indices_t.as_mut() {
            out.push(indices_t);
        }
        if let Some(q8_keybytes_t) = self.q8_keybytes_t.as_mut() {
            out.push(q8_keybytes_t);
        }
        if let Some(q8_keybytes_seq) = self.q8_keybytes_seq.as_mut() {
            out.push(q8_keybytes_seq);
        }
        if let Some(q8_fullbyte_seq) = self.q8_fullbyte_seq.as_mut() {
            out.push(q8_fullbyte_seq);
        }
        if let Some(q8_kvbytes_seq) = self.q8_kvbytes_seq.as_mut() {
            out.push(q8_kvbytes_seq);
        }
        if let Some(q8_slot_scales_seq) = self.q8_slot_scales_seq.as_mut() {
            out.push(q8_slot_scales_seq);
        }
        if let Some(norms) = self.norms.as_mut() {
            out.push(norms);
        }
        out.push(&mut self.qjl_signs);
        if let Some(qjl_signs_t) = self.qjl_signs_t.as_mut() {
            out.push(qjl_signs_t);
        }
        if let Some(residual_norms) = self.residual_norms.as_mut() {
            out.push(residual_norms);
        }
        if let Some(key_slot_scale) = self.key_slot_scale.as_mut() {
            out.push(key_slot_scale);
        }
    }
}

/// GPU-resident quantised value data for the Uniform path.
///
///   indices:  [B, H, T, D]  uint8
///   indices_t:[B, H, D, T]  uint8
///   norms:    [B, H, T, 1]  f32
#[derive(Debug, Clone)]
pub(super) struct GpuValueStore {
    pub(super) indices: Option<InlineArray>,
    pub(super) indices_t: Option<InlineArray>,
    pub(super) norms: Option<InlineArray>,
    pub(super) d256_rot_values_seq: Option<InlineArray>,
}

impl GpuValueStore {
    pub(super) fn append(&mut self, new: GpuValueStore) {
        self.indices = match (self.indices.take(), new.indices) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.indices_t = match (self.indices_t.take(), new.indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.norms = match (self.norms.take(), new.norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.d256_rot_values_seq = match (self.d256_rot_values_seq.take(), new.d256_rot_values_seq)
        {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
    }

    pub(super) fn indices_t_array(&self) -> Option<InlineArray> {
        self.indices_t.clone().or_else(|| {
            self.indices
                .as_ref()
                .map(|arr| arr.transpose_axes(&[0, 1, 3, 2]))
        })
    }

    pub(super) fn norms_array(&self) -> Option<InlineArray> {
        self.norms.clone()
    }

    pub(super) fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        if let Some(indices) = self.indices.as_mut() {
            out.push(indices);
        }
        if let Some(indices_t) = self.indices_t.as_mut() {
            out.push(indices_t);
        }
        if let Some(norms) = self.norms.as_mut() {
            out.push(norms);
        }
        if let Some(d256_rot_values_seq) = self.d256_rot_values_seq.as_mut() {
            out.push(d256_rot_values_seq);
        }
    }
}

//
// Layout pinned for upcoming Mixed attention kernels:
//   regular_indices    — [B, H, T, D_reg]               u8
//   regular_indices_t  — [B, H, D_reg, T]               u8   (S-innermost)
//   regular_qjl_signs  — [B, H, T, ceil(D_reg/32)]      u32
//   regular_qjl_signs_t — [B, H, ceil(D_reg/32), T]     u32  (S-innermost)
//   regular_norms      — [B, H, T, 1]                   f32
//   regular_residual_norms — [B, H, T, 1]               f32
//   regular_src_dim    — [B, H, T, D_reg]               u8   (scatter-back map)
//
// Outlier mirrors: same shapes with D_out instead of D_reg.
//
// Scatter tables (`*_src_dim`) hold the original-D positions of each
// sub-vector slot, sorted ascending. They are written once per token at
// encode time and gathered by the attention kernel to scatter regular and
// outlier contributions back into the [B, H, D_total] output. K and V
// stores have *independent* scatter tables — `select_outlier_mask` runs on
// each tensor's row magnitudes separately.

#[derive(Debug, Clone)]
pub(super) struct GpuMixedKeyStore {
    pub(super) regular_indices: InlineArray,
    pub(super) regular_indices_t: Option<InlineArray>,
    pub(super) regular_qjl_signs: InlineArray,
    pub(super) regular_qjl_signs_t: Option<InlineArray>,
    pub(super) regular_norms: InlineArray,
    pub(super) regular_residual_norms: InlineArray,
    pub(super) regular_slot_scale: InlineArray,
    pub(super) regular_src_dim: InlineArray,
    pub(super) outlier_indices: InlineArray,
    pub(super) outlier_indices_t: Option<InlineArray>,
    pub(super) outlier_qjl_signs: InlineArray,
    pub(super) outlier_qjl_signs_t: Option<InlineArray>,
    pub(super) outlier_norms: InlineArray,
    pub(super) outlier_residual_norms: InlineArray,
    pub(super) outlier_slot_scale: InlineArray,
    pub(super) outlier_src_dim: InlineArray,
}

impl GpuMixedKeyStore {
    pub(super) fn append(&mut self, new: GpuMixedKeyStore) {
        self.regular_indices = self.regular_indices.kv_cache_append(&new.regular_indices, 2);
        self.regular_indices_t = match (self.regular_indices_t.take(), new.regular_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.regular_qjl_signs = self
            .regular_qjl_signs
            .kv_cache_append(&new.regular_qjl_signs, 2);
        self.regular_qjl_signs_t = match (self.regular_qjl_signs_t.take(), new.regular_qjl_signs_t)
        {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.regular_norms = self.regular_norms.kv_cache_append(&new.regular_norms, 2);
        self.regular_residual_norms = self
            .regular_residual_norms
            .kv_cache_append(&new.regular_residual_norms, 2);
        self.regular_slot_scale = self
            .regular_slot_scale
            .kv_cache_append(&new.regular_slot_scale, 2);
        self.regular_src_dim = self.regular_src_dim.kv_cache_append(&new.regular_src_dim, 2);
        self.outlier_indices = self.outlier_indices.kv_cache_append(&new.outlier_indices, 2);
        self.outlier_indices_t = match (self.outlier_indices_t.take(), new.outlier_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.outlier_qjl_signs = self
            .outlier_qjl_signs
            .kv_cache_append(&new.outlier_qjl_signs, 2);
        self.outlier_qjl_signs_t = match (self.outlier_qjl_signs_t.take(), new.outlier_qjl_signs_t)
        {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.outlier_norms = self.outlier_norms.kv_cache_append(&new.outlier_norms, 2);
        self.outlier_residual_norms = self
            .outlier_residual_norms
            .kv_cache_append(&new.outlier_residual_norms, 2);
        self.outlier_slot_scale = self
            .outlier_slot_scale
            .kv_cache_append(&new.outlier_slot_scale, 2);
        self.outlier_src_dim = self.outlier_src_dim.kv_cache_append(&new.outlier_src_dim, 2);
    }

    pub(super) fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        out.push(&mut self.regular_indices);
        if let Some(indices_t) = self.regular_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.regular_qjl_signs);
        if let Some(signs_t) = self.regular_qjl_signs_t.as_mut() {
            out.push(signs_t);
        }
        out.push(&mut self.regular_norms);
        out.push(&mut self.regular_residual_norms);
        out.push(&mut self.regular_slot_scale);
        out.push(&mut self.regular_src_dim);
        out.push(&mut self.outlier_indices);
        if let Some(indices_t) = self.outlier_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.outlier_qjl_signs);
        if let Some(signs_t) = self.outlier_qjl_signs_t.as_mut() {
            out.push(signs_t);
        }
        out.push(&mut self.outlier_norms);
        out.push(&mut self.outlier_residual_norms);
        out.push(&mut self.outlier_slot_scale);
        out.push(&mut self.outlier_src_dim);
    }
}

#[derive(Debug, Clone)]
pub(super) struct GpuMixedValueStore {
    pub(super) regular_indices: InlineArray,
    pub(super) regular_indices_t: Option<InlineArray>,
    pub(super) regular_norms: InlineArray,
    pub(super) regular_src_dim: InlineArray,
    pub(super) outlier_indices: InlineArray,
    pub(super) outlier_indices_t: Option<InlineArray>,
    pub(super) outlier_norms: InlineArray,
    pub(super) outlier_src_dim: InlineArray,
}

impl GpuMixedValueStore {
    pub(super) fn append(&mut self, new: GpuMixedValueStore) {
        self.regular_indices = self.regular_indices.kv_cache_append(&new.regular_indices, 2);
        self.regular_indices_t = match (self.regular_indices_t.take(), new.regular_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.regular_norms = self.regular_norms.kv_cache_append(&new.regular_norms, 2);
        self.regular_src_dim = self.regular_src_dim.kv_cache_append(&new.regular_src_dim, 2);
        self.outlier_indices = self.outlier_indices.kv_cache_append(&new.outlier_indices, 2);
        self.outlier_indices_t = match (self.outlier_indices_t.take(), new.outlier_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.outlier_norms = self.outlier_norms.kv_cache_append(&new.outlier_norms, 2);
        self.outlier_src_dim = self.outlier_src_dim.kv_cache_append(&new.outlier_src_dim, 2);
    }

    pub(super) fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        out.push(&mut self.regular_indices);
        if let Some(indices_t) = self.regular_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.regular_norms);
        out.push(&mut self.regular_src_dim);
        out.push(&mut self.outlier_indices);
        if let Some(indices_t) = self.outlier_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.outlier_norms);
        out.push(&mut self.outlier_src_dim);
    }
}

