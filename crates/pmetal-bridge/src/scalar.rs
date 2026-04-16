//! Scalar-dtype safety helpers for [`InlineArray`].
//!
//! ## The footgun
//!
//! [`InlineArray::from_f32`] always produces a 0-D **f32** array, regardless
//! of the surrounding model's dtype. In mixed-precision code (bf16 / f16
//! weights), the naïve pattern
//!
//! ```ignore
//! let scale = InlineArray::from_f32(inv_scale);     // f32
//! let result = x.multiply(&scale);                  // bf16 × f32 → promotes!
//! ```
//!
//! silently promotes `x` to f32 for the duration of the op, then casts the
//! result back — a measured **~5× slowdown** per scalar-bearing op in
//! the Qwen3.5 and Gemma 4 forward passes (see
//! `feedback_native_bridge_dtype_rules`). Forgetting
//! `.as_dtype(weights.model_dtype)` has no compiler warning and no runtime
//! error — it's a pure-perf landmine.
//!
//! ## The fix in this module
//!
//! Two ergonomic additions that make the correct thing the easy thing:
//!
//! * [`InlineArray::scalar_like`] — one-shot scalar construction with the
//!   dtype copied from a peer array. Replaces `from_f32(v).as_dtype(d)`.
//! * `[*]_scalar` methods (`mul_scalar`, `add_scalar`, `sub_scalar`,
//!   `div_scalar`) — compose `scalar_like` with the binary op so callers
//!   never touch dtype plumbing for one-off scalar ops.
//!
//! Both are additive; none of the existing `from_f32` / `as_dtype` / binary
//! op surface is affected, so nothing has to migrate atomically. New code
//! and hot paths can switch incrementally.
//!
//! ## Example
//!
//! ```ignore
//! // Before (footgun):
//! let half    = InlineArray::from_f32(0.5);                // f32 — promotes x!
//! let result  = x.multiply(&half);
//!
//! // Before (verbose but correct):
//! let half    = InlineArray::from_f32(0.5).as_dtype(x.dtype_raw());
//! let result  = x.multiply(&half);
//!
//! // After (safe + terse):
//! let result  = x.mul_scalar(0.5);
//! ```
//!
//! ## Scope
//!
//! These helpers cover the four pointwise ops that account for the vast
//! majority of scalar-bearing sites in the native model code:
//! multiplication, addition, subtraction, division. `x.sub_scalar(v)`
//! computes `x − v`; use `InlineArray::scalar_like(v, &x).subtract(&x)` for
//! `v − x`. Non-arithmetic scalar uses (e.g. `pow`, `maximum(x, 0)`,
//! comparison) are out of scope here and retain their existing APIs.

use crate::InlineArray;

impl InlineArray {
    /// Creates a 0-D scalar array carrying `value` with the same dtype as
    /// `peer`.
    ///
    /// Preferred over `InlineArray::from_f32(value).as_dtype(peer.dtype_raw())`
    /// because it can't be invoked without the dtype, eliminating the
    /// "forgot to cast" footgun.
    #[inline]
    pub fn scalar_like(value: f32, peer: &Self) -> Self {
        InlineArray::from_f32(value).as_dtype(peer.dtype_raw())
    }

    /// `self * value`, with the scalar cast to `self`'s dtype before the
    /// multiply (no dtype promotion).
    #[inline]
    pub fn mul_scalar(&self, value: f32) -> Self {
        self.multiply(&Self::scalar_like(value, self))
    }

    /// `self + value`, with the scalar cast to `self`'s dtype.
    #[inline]
    pub fn add_scalar(&self, value: f32) -> Self {
        self.add(&Self::scalar_like(value, self))
    }

    /// `self - value`, with the scalar cast to `self`'s dtype.
    ///
    /// For `value - self` use
    /// `InlineArray::scalar_like(value, self).subtract(self)`.
    #[inline]
    pub fn sub_scalar(&self, value: f32) -> Self {
        self.subtract(&Self::scalar_like(value, self))
    }

    /// `self / value`, with the scalar cast to `self`'s dtype.
    #[inline]
    pub fn div_scalar(&self, value: f32) -> Self {
        self.divide(&Self::scalar_like(value, self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::Dtype;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn scalar_like_matches_peer_dtype_float32() {
        let peer = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        assert_eq!(peer.dtype(), Dtype::Float32);

        let s = InlineArray::scalar_like(3.5, &peer);
        assert_eq!(s.dtype(), Dtype::Float32);
    }

    #[test]
    fn scalar_like_matches_peer_dtype_bfloat16() {
        // Build a bf16 peer by casting.
        let peer_f32 = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        let peer_bf16 = peer_f32.as_dtype(Dtype::Bfloat16.as_i32());
        assert_eq!(peer_bf16.dtype(), Dtype::Bfloat16);

        let s = InlineArray::scalar_like(3.5, &peer_bf16);
        assert_eq!(
            s.dtype(),
            Dtype::Bfloat16,
            "scalar_like must adopt the peer's non-f32 dtype"
        );
    }

    #[test]
    fn mul_scalar_preserves_dtype() {
        let x_f32 = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let x_bf16 = x_f32.as_dtype(Dtype::Bfloat16.as_i32());
        let y = x_bf16.mul_scalar(2.0);
        assert_eq!(
            y.dtype(),
            Dtype::Bfloat16,
            "mul_scalar must keep the array's dtype (no f32 promotion)"
        );
    }

    #[test]
    fn add_sub_mul_div_scalar_arithmetic() {
        let x = InlineArray::from_f32_slice(&[10.0, 20.0, 30.0], &[3]);

        let mut mul = x.mul_scalar(0.5);
        let mut add = x.add_scalar(1.0);
        let mut sub = x.sub_scalar(2.0);
        let mut div = x.div_scalar(2.0);

        let mul_vec = mul.to_f32_vec(3).expect("mul_scalar eval");
        let add_vec = add.to_f32_vec(3).expect("add_scalar eval");
        let sub_vec = sub.to_f32_vec(3).expect("sub_scalar eval");
        let div_vec = div.to_f32_vec(3).expect("div_scalar eval");

        for (got, want) in mul_vec.iter().zip([5.0, 10.0, 15.0]) {
            assert!(approx_eq(*got, want, 1e-5), "mul: {got} != {want}");
        }
        for (got, want) in add_vec.iter().zip([11.0, 21.0, 31.0]) {
            assert!(approx_eq(*got, want, 1e-5), "add: {got} != {want}");
        }
        for (got, want) in sub_vec.iter().zip([8.0, 18.0, 28.0]) {
            assert!(approx_eq(*got, want, 1e-5), "sub: {got} != {want}");
        }
        for (got, want) in div_vec.iter().zip([5.0, 10.0, 15.0]) {
            assert!(approx_eq(*got, want, 1e-5), "div: {got} != {want}");
        }
    }
}
