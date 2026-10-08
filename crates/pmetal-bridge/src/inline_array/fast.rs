//! Fused neural-net ops: rms_norm, rope*, sdpa*, conv*, layer_norm,
//! linalg (tri_inv, svd, addmm), and other "fast" math primitives
//! (clip, log_softmax, cross_entropy, pad, split).
//!
//! These wrap high-throughput Metal kernels that encode full sub-graphs in a
//! single dispatch, avoiding op-by-op FFI overhead.
//!
//! ## Error handling
//!
//! All methods in this module have infallible-looking signatures for ABI
//! stability, but every wrapped op can fail on the C++ side — an unsupported
//! dtype, shape mismatch, or an `eps` the kernel rejects will set the
//! thread-local error slot and return a scalar-zero sentinel tensor. The
//! error then silently propagates through subsequent shape-indexed ops
//! until something blows up with a confusing panic.
//!
//! Two paths keep this debuggable:
//!
//! 1. Call [`crate::check_last_error`] (or the [`crate::try_ops`] variants
//!    like [`crate::InlineArray::try_rms_norm`]) after each fast op to surface
//!    the failure at the site where it actually happened.
//! 2. Leave [`crate::set_error_log_mode`] at its default (on in debug builds)
//!    so the first caught exception prints to stderr — useful during bring-up
//!    when callers haven't wired up explicit checks yet.

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

impl InlineArray {
    // ── Fast ops ─────────────────────────────────────────────────────────

    pub fn rms_norm(&self, weight: Option<&Self>, eps: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rms_norm(
                dst.as_mut_ptr(),
                &self.raw,
                weight.map_or(std::ptr::null(), |w| &w.raw),
                eps,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn rope(&self, dims: i32, traditional: bool, base: f32, scale: f32, offset: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rope(
                dst.as_mut_ptr(),
                &self.raw,
                dims,
                traditional,
                base,
                scale,
                offset,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// RoPE with an explicit inverse-frequency array. Pass the full
    /// `head_dim` as `dims` and an `[rotated_dims / 2]`-sized `freqs`
    /// array; non-rotated dimensions can be padded with `f32::INFINITY`
    /// so `mx.fast.rope` skips them. This mirrors the reference
    /// `ProportionalRoPE` — used by Gemma 4 full-attention layers.
    pub fn rope_with_freqs(
        &self,
        dims: i32,
        traditional: bool,
        scale: f32,
        offset: i32,
        freqs: &Self,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rope_with_freqs(
                dst.as_mut_ptr(),
                &self.raw,
                dims,
                traditional,
                scale,
                offset,
                &freqs.raw,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Per-position RoPE: applies an array of int32 offsets (one per
    /// token) instead of a single scalar offset. Required for tree
    /// verify where each tree node has its own depth, not a
    /// contiguous position sequence. `offset_arr` must be a 1-D int32
    /// InlineArray of length `seq_len`.
    pub fn rope_with_pos_ids(
        &self,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset_arr: &Self,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rope_with_pos_ids(
                dst.as_mut_ptr(),
                &self.raw,
                dims,
                traditional,
                base,
                scale,
                &offset_arr.raw,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn sdpa(&self, k: &Self, v: &Self, scale: f32, mask_mode: &str) -> Self {
        let c = std::ffi::CString::new(mask_mode).unwrap();
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sdpa(
                dst.as_mut_ptr(),
                &self.raw,
                &k.raw,
                &v.raw,
                scale,
                c.as_ptr(),
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// SDPA with optional mask array. Pass `None` for no mask.
    #[inline]
    pub fn sdpa_with_mask(&self, k: &Self, v: &Self, scale: f32, mask: Option<&Self>) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let mask_ptr = mask
            .map(|m| &m.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_sdpa_with_mask(dst.as_mut_ptr(), &self.raw, &k.raw, &v.raw, scale, mask_ptr);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn split(&self, indices: &[i32], axis: i32) -> Vec<Self> {
        let n = indices.len() + 1;
        let mut bufs: Vec<MaybeUninit<RawBuf>> = (0..n).map(|_| MaybeUninit::uninit()).collect();
        unsafe {
            mlx_inline_split(
                &self.raw,
                indices.as_ptr(),
                indices.len() as i32,
                axis,
                bufs.as_mut_ptr() as *mut RawBuf,
            );
            bufs.into_iter()
                .map(|b| Self {
                    raw: b.assume_init(),
                })
                .collect()
        }
    }

    pub fn conv1d(
        &self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_conv1d(
                dst.as_mut_ptr(),
                &self.raw,
                &weight.raw,
                stride,
                padding,
                dilation,
                groups,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Linalg / loss helpers ───────────────────────────────────────────

    pub fn tri_inv(&self, upper: bool, use_cpu: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tri_inv(dst.as_mut_ptr(), &self.raw, upper, use_cpu);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Singular Value Decomposition — returns `(U, S, Vt)`.
    ///
    /// Economy/thin SVD: `U` is `[m, k]`, `S` is `[k]`, `Vt` is `[k, n]`
    /// where `k = min(m, n)`.  Always runs on the CPU stream.
    pub fn svd(&self) -> (Self, Self, Self) {
        let mut u = MaybeUninit::<RawBuf>::uninit();
        let mut s = MaybeUninit::<RawBuf>::uninit();
        let mut vt = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_svd(u.as_mut_ptr(), s.as_mut_ptr(), vt.as_mut_ptr(), &self.raw);
            (
                Self {
                    raw: u.assume_init(),
                },
                Self {
                    raw: s.assume_init(),
                },
                Self {
                    raw: vt.assume_init(),
                },
            )
        }
    }

    /// Clip values to [lo, hi]. Either bound can be None.
    pub fn clip(&self, lo: Option<&Self>, hi: Option<&Self>) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let lo_ptr = lo
            .map(|x| &x.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let hi_ptr = hi
            .map(|x| &x.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_clip(dst.as_mut_ptr(), &self.raw, lo_ptr, hi_ptr);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn log_softmax(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_log_softmax(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Cross-entropy: -sum(targets * log_softmax(self), axis).
    pub fn cross_entropy(&self, targets: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_cross_entropy(dst.as_mut_ptr(), &self.raw, &targets.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sparse cross-entropy for integer class-index targets.
    ///
    /// `self` holds logits of shape `[..., V, ...]`. `indices` holds int class
    /// indices whose shape equals `self.shape` with `axis` removed. Returns
    /// per-position NLL in nats with that same reduced shape.
    ///
    /// Computes `logsumexp(logits, axis) - logits[indices]` inside a single
    /// bridge call — the full `[..., V]` log-softmax tensor never materializes,
    /// so memory scales with `logits.size() / V` rather than `logits.size()`.
    pub fn cross_entropy_sparse(&self, indices: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_cross_entropy_sparse(dst.as_mut_ptr(), &self.raw, &indices.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Layers ──────────────────────────────────────────────────────────

    pub fn layer_norm(&self, weight: Option<&Self>, bias: Option<&Self>, eps: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let w_ptr = weight
            .map(|w| &w.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let b_ptr = bias
            .map(|b| &b.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_layer_norm(dst.as_mut_ptr(), &self.raw, w_ptr, b_ptr, eps);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// `c + a @ b` (addmm).
    pub fn addmm(c: &Self, a: &Self, b: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_addmm(dst.as_mut_ptr(), &c.raw, &a.raw, &b.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// 2-D convolution (NHWC format, MLX standard).
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        weight: &Self,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        dil_h: i32,
        dil_w: i32,
        groups: i32,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_conv2d(
                dst.as_mut_ptr(),
                &self.raw,
                &weight.raw,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
                groups,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Pad every axis with `fill_value`. `pad_widths_flat` is
    /// `[before_0, after_0, before_1, after_1, ...]`, one pair per axis.
    pub fn pad_constant(&self, pad_widths_flat: &[i32], fill_value: f32) -> Self {
        self.pad(pad_widths_flat, PadMode::Constant, fill_value)
    }

    /// Pad every axis in `mode`, like `mx.pad`. `pad_widths_flat` is
    /// `[before_0, after_0, before_1, after_1, ...]`, one pair per axis;
    /// `fill_value` is cast to the array's dtype and used by
    /// [`PadMode::Constant`] only.
    pub fn pad(&self, pad_widths_flat: &[i32], mode: PadMode, fill_value: f32) -> Self {
        debug_assert_eq!(pad_widths_flat.len(), 2 * self.ndim() as usize);
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_pad(
                dst.as_mut_ptr(),
                &self.raw,
                pad_widths_flat.as_ptr(),
                (pad_widths_flat.len() / 2) as i32,
                fill_value,
                mode.as_c_str().as_ptr(),
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }
}

/// How [`InlineArray::pad`] fills the new elements: the four modes `mx.pad`
/// has, with NumPy's meaning for each. For `[1, 2, 3]` padded by two on the
/// left:
///
/// | mode        | result            |
/// |-------------|-------------------|
/// | `Constant`  | `[c, c, 1, 2, 3]` |
/// | `Edge`      | `[1, 1, 1, 2, 3]` |
/// | `Reflect`   | `[3, 2, 1, 2, 3]` |
/// | `Symmetric` | `[2, 1, 1, 2, 3]` |
///
/// NumPy's other modes (`wrap`, `linear_ramp`, `maximum`, `mean`, `median`,
/// `minimum`, `empty`) have no MLX counterpart, and parsing one of those names
/// is an error rather than a silent constant pad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PadMode {
    /// A constant value (the pad call's `fill_value`).
    #[default]
    Constant,
    /// The edge element repeated.
    Edge,
    /// Mirror image that excludes the edge element.
    Reflect,
    /// Mirror image that includes the edge element.
    Symmetric,
}

impl PadMode {
    /// The mode's name in MLX and NumPy.
    pub fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Edge => "edge",
            Self::Reflect => "reflect",
            Self::Symmetric => "symmetric",
        }
    }

    fn as_c_str(self) -> &'static std::ffi::CStr {
        match self {
            Self::Constant => c"constant",
            Self::Edge => c"edge",
            Self::Reflect => c"reflect",
            Self::Symmetric => c"symmetric",
        }
    }
}

impl std::str::FromStr for PadMode {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "constant" => Ok(Self::Constant),
            "edge" => Ok(Self::Edge),
            "reflect" => Ok(Self::Reflect),
            "symmetric" => Ok(Self::Symmetric),
            other => Err(format!(
                "pad mode `{other}` is not supported; MLX pads in `constant`, `edge`, \
                 `reflect` or `symmetric` mode"
            )),
        }
    }
}
