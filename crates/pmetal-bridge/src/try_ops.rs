//! Checked (`try_*`) variants of the most exception-prone `InlineArray` ops.
//!
//! The legacy ops in [`InlineArray`] have void-returning signatures for
//! ABI stability — an exception in the C++ layer produces a sentinel
//! `array(0.0f)` alongside a thread-local error record (see
//! [`crate::error`]). The `try_*` methods on this file's `impl InlineArray`
//! block compose the op with [`check_last_error`] so callers can propagate
//! the failure with `?` instead of checking after every call:
//!
//! ```ignore
//! use pmetal_bridge::{InlineArray, BridgeResult};
//! fn attn_forward(q: &InlineArray, k: &InlineArray, v: &InlineArray) -> BridgeResult<InlineArray> {
//!     q.try_matmul(k)?.try_softmax(-1)?.try_matmul(v)
//! }
//! ```
//!
//! The checked variants are purely additive — they call the same underlying
//! FFI function as their legacy counterparts; only the error-read + return
//! shape differs. Zero existing callsite needs to change.
//!
//! ## Coverage
//!
//! These are the ops flagged by the April 2026 bridge audit as
//! shape-sensitive or otherwise prone to raising `std::invalid_argument` /
//! `std::runtime_error` from MLX. The list is deliberately narrow to keep
//! the checked surface focused; other ops can adopt a `try_*` sibling
//! later as their callers need them.

use crate::InlineArray;
use crate::error::{BridgeResult, check_last_error};

impl InlineArray {
    // ── Binary / reduction math ──────────────────────────────────────────

    /// Checked matmul. Propagates MLX shape mismatches as
    /// [`crate::BridgeError::CxxException`] instead of silently returning a
    /// zeros scalar.
    pub fn try_matmul(&self, other: &Self) -> BridgeResult<Self> {
        let out = self.matmul(other);
        check_last_error()?;
        Ok(out)
    }

    /// Checked softmax. The `axis` argument is validated by MLX; an
    /// out-of-range value surfaces as `CxxException`.
    pub fn try_softmax(&self, axis: i32) -> BridgeResult<Self> {
        let out = self.softmax(axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked precise (fp32-internal) softmax.
    pub fn try_softmax_precise(&self, axis: i32) -> BridgeResult<Self> {
        let out = self.softmax_precise(axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked reshape — the most common shape-fail site per the v0.5.0
    /// sampling + MoE test-abort findings.
    pub fn try_reshape(&self, shape: &[i32]) -> BridgeResult<Self> {
        let out = self.reshape(shape);
        check_last_error()?;
        Ok(out)
    }

    // ── Fast ops / attention ─────────────────────────────────────────────

    /// Checked fused RMS norm.
    pub fn try_rms_norm(&self, weight: Option<&Self>, eps: f32) -> BridgeResult<Self> {
        let out = self.rms_norm(weight, eps);
        check_last_error()?;
        Ok(out)
    }

    /// Checked fused scaled-dot-product attention.
    pub fn try_sdpa(&self, k: &Self, v: &Self, scale: f32, mask_mode: &str) -> BridgeResult<Self> {
        let out = self.sdpa(k, v, scale, mask_mode);
        check_last_error()?;
        Ok(out)
    }

    /// Checked fused SDPA with optional explicit mask array.
    pub fn try_sdpa_with_mask(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Self>,
    ) -> BridgeResult<Self> {
        let out = self.sdpa_with_mask(k, v, scale, mask);
        check_last_error()?;
        Ok(out)
    }

    // ── Gather / quantized paths ─────────────────────────────────────────

    /// Checked gather-matmul (batched expert dispatch for MoE). This op
    /// sits on the MoE hot-path that was aborting on shape mismatches in
    /// qwen3_moe dispatch tests (see `project_v050_remaining`).
    pub fn try_gather_mm(
        &self,
        other: &Self,
        lhs_indices: Option<&Self>,
        rhs_indices: Option<&Self>,
        sorted: bool,
    ) -> BridgeResult<Self> {
        let out = self.gather_mm(other, lhs_indices, rhs_indices, sorted);
        check_last_error()?;
        Ok(out)
    }

    /// Checked dequantize (group-size + bits validated on the C++ side).
    pub fn try_dequantize(
        &self,
        scales: &Self,
        biases: &Self,
        group_size: i32,
        bits: i32,
    ) -> BridgeResult<Self> {
        let out = self.dequantize(scales, biases, group_size, bits);
        check_last_error()?;
        Ok(out)
    }

    /// Checked quantized matmul.
    pub fn try_quantized_matmul(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
    ) -> BridgeResult<Self> {
        let out = self.quantized_matmul(w, scales, biases, transpose, group_size, bits);
        check_last_error()?;
        Ok(out)
    }

    /// Checked gather-quantized-matmul. The argument count mirrors the
    /// underlying FFI; the `too_many_arguments` lint is suppressed because
    /// splitting into a builder would just relocate the noise.
    #[allow(clippy::too_many_arguments)]
    pub fn try_gather_qmm(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        lhs_indices: Option<&Self>,
        rhs_indices: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        sorted: bool,
    ) -> BridgeResult<Self> {
        let out = self.gather_qmm(
            w,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
            transpose,
            group_size,
            bits,
            sorted,
        );
        check_last_error()?;
        Ok(out)
    }

    // ── Composition / shuffle ops ────────────────────────────────────────

    /// Checked 2-way concatenate (most callers use the 2-arg form).
    pub fn try_concatenate_2(&self, other: &Self, axis: i32) -> BridgeResult<Self> {
        let out = self.concatenate_2(other, axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked split — returns `Vec<Self>` of `indices.len() + 1` entries.
    ///
    /// On C++ failure every output slot is still placement-new'd with a
    /// zero sentinel (see `bridge.cpp`), so dropping the returned `Vec`
    /// is safe even in the error path.
    pub fn try_split(&self, indices: &[i32], axis: i32) -> BridgeResult<Vec<Self>> {
        let out = self.split(indices, axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked cross-entropy loss.
    pub fn try_cross_entropy(&self, targets: &Self, axis: i32) -> BridgeResult<Self> {
        let out = self.cross_entropy(targets, axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked sparse cross-entropy (integer-class targets).
    pub fn try_cross_entropy_sparse(&self, indices: &Self, axis: i32) -> BridgeResult<Self> {
        let out = self.cross_entropy_sparse(indices, axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked log-softmax along `axis`.
    pub fn try_log_softmax(&self, axis: i32) -> BridgeResult<Self> {
        let out = self.log_softmax(axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked clip — bounds may be `None` for one-sided clipping.
    pub fn try_clip(&self, lo: Option<&Self>, hi: Option<&Self>) -> BridgeResult<Self> {
        let out = self.clip(lo, hi);
        check_last_error()?;
        Ok(out)
    }

    /// Checked layer norm.
    pub fn try_layer_norm(
        &self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        eps: f32,
    ) -> BridgeResult<Self> {
        let out = self.layer_norm(weight, bias, eps);
        check_last_error()?;
        Ok(out)
    }

    /// Checked constant-pad. `pad_widths_flat` must have length `2 * ndim`.
    pub fn try_pad_constant(
        &self,
        pad_widths_flat: &[i32],
        fill_value: f32,
    ) -> BridgeResult<Self> {
        let out = self.pad_constant(pad_widths_flat, fill_value);
        check_last_error()?;
        Ok(out)
    }

    /// Checked leaky-ReLU activation.
    pub fn try_leaky_relu(&self, neg_slope: f32) -> BridgeResult<Self> {
        let out = self.leaky_relu(neg_slope);
        check_last_error()?;
        Ok(out)
    }

    /// Checked addmm (`c + a @ b`).
    pub fn try_addmm(c: &Self, a: &Self, b: &Self) -> BridgeResult<Self> {
        let out = Self::addmm(c, a, b);
        check_last_error()?;
        Ok(out)
    }

    /// Checked 1-D convolution.
    pub fn try_conv1d(
        &self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> BridgeResult<Self> {
        let out = self.conv1d(weight, stride, padding, dilation, groups);
        check_last_error()?;
        Ok(out)
    }

    /// Checked 2-D convolution (NHWC, MLX standard).
    #[allow(clippy::too_many_arguments)]
    pub fn try_conv2d(
        &self,
        weight: &Self,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        dil_h: i32,
        dil_w: i32,
        groups: i32,
    ) -> BridgeResult<Self> {
        let out = self.conv2d(weight, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
        check_last_error()?;
        Ok(out)
    }

    /// Checked tri-inverse (lower or upper triangular). `use_cpu=true` forces
    /// CPU dispatch (matching mlx-lm's WY-factorization usage).
    pub fn try_tri_inv(&self, upper: bool, use_cpu: bool) -> BridgeResult<Self> {
        let out = self.tri_inv(upper, use_cpu);
        check_last_error()?;
        Ok(out)
    }

    /// Checked SVD — returns `(U, S, Vt)`.
    pub fn try_svd(&self) -> BridgeResult<(Self, Self, Self)> {
        let out = self.svd();
        check_last_error()?;
        Ok(out)
    }

    // ── Shape ops ───────────────────────────────────────────────────────

    /// Checked single-axis squeeze. Out-of-range axis surfaces as
    /// `BridgeError::CxxException`.
    pub fn try_squeeze(&self, axis: i32) -> BridgeResult<Self> {
        let out = self.squeeze(axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked single-axis expand_dims.
    pub fn try_expand_dims(&self, axis: i32) -> BridgeResult<Self> {
        let out = self.expand_dims(axis);
        check_last_error()?;
        Ok(out)
    }

    /// Checked transpose. Permutation must contain each axis index exactly once.
    pub fn try_transpose_axes(&self, axes: &[i32]) -> BridgeResult<Self> {
        let out = self.transpose_axes(axes);
        check_last_error()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BridgeError;

    #[test]
    fn try_matmul_reports_shape_mismatch() {
        // 2×3 and 4×5 are not multipliable; MLX throws std::invalid_argument.
        let a = InlineArray::from_f32_slice(&[1.0; 6], &[2, 3]);
        let b = InlineArray::from_f32_slice(&[1.0; 20], &[4, 5]);

        let err = a
            .try_matmul(&b)
            .expect_err("incompatible shapes should surface a BridgeError");

        match err {
            BridgeError::CxxException(msg) => {
                assert!(msg.contains("[matmul]"), "expected op tag, got: {msg}");
            }
            BridgeError::Unknown(msg) => panic!("expected CxxException, got Unknown: {msg}"),
        }

        // The error slot must have been cleared on read.
        crate::error::clear_last_error();
    }

    #[test]
    fn try_matmul_success_is_silent() {
        let a = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = InlineArray::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let prod = a.try_matmul(&b).expect("2x2 × 2x2 should succeed");
        assert_eq!(prod.shape(), &[2, 2]);
    }

    #[test]
    fn try_reshape_reports_element_count_mismatch() {
        let a = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // 4 elems can't be reshaped to [3,2].
        let err = a
            .try_reshape(&[3, 2])
            .expect_err("wrong element count should surface BridgeError");
        if let BridgeError::CxxException(msg) = err {
            assert!(msg.contains("[reshape]"));
        } else {
            panic!("expected CxxException");
        }
    }

    #[test]
    fn try_softmax_out_of_range_axis() {
        let a = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let err = a
            .try_softmax(42)
            .expect_err("axis=42 on a 2-D tensor should fail");
        if let BridgeError::CxxException(msg) = err {
            assert!(msg.contains("[softmax]"));
        } else {
            panic!("expected CxxException");
        }
    }
}
