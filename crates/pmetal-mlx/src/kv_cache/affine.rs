//! One layer of the affine-quantized KV cache.
//!
//! Keys and values are stored as MLX affine groups: `group_size` consecutive
//! channels of a token share a scale and a bias, and each channel is a
//! `bits`-bit integer, `x ≈ q · scale + bias`. The packed words, scales and
//! biases live in buffers written in place and grown by doubling (the
//! [`QuantizedTuple`] layout the native engines use), so appending a token
//! costs the token, not the cache.
//!
//! # Reading
//!
//! - A decode step attends straight to the packed rows with
//!   [`quantized_sdpa_kv`]: MLX dequantizes inside its `quantized_matmul`
//!   kernels and the cache is never expanded.
//! - [`AffineLayer::update`] returns dense keys and values for the callers
//!   that need them (a prefill chunk, an architecture with no fast path).
//!   The tokens it was just given come back exact: only history is
//!   dequantized, so a prompt's own attention is not degraded by quantizing
//!   it, and the first generated token sees the logits an unquantized cache
//!   would give.
//!
//! # Error
//!
//! A step is `(max - min) / (2^bits - 1)` over the element's group. A plain
//! min-max map is off by at most half a step; MLX aligns the larger edge of
//! the group to a level, which moves the levels by up to half a step and
//! clips what falls past the last one, so an element is within one step.

use std::fmt;

use pmetal_runtime::compat::{Array, Dtype, Exception, ops};
use pmetal_runtime::decode::quantized_sdpa_kv;
use pmetal_runtime::native_common::kv_cache::GrowthPolicy;
use pmetal_runtime::qwen3_native::QuantizedTuple;

/// Bit widths MLX's affine quantizer packs.
pub(super) const SUPPORTED_BITS: [u8; 6] = [2, 3, 4, 5, 6, 8];

/// Whether `bits` is a width MLX's affine quantizer packs.
pub(super) fn bits_supported(bits: u8) -> bool {
    SUPPORTED_BITS.contains(&bits)
}

/// 32-bit words one quantized row of `dim` channels takes.
pub(super) fn packed_words(dim: usize, bits: u8) -> usize {
    (dim * usize::from(bits)).div_ceil(32)
}

/// One side (keys or values) of a layer.
#[derive(Clone)]
struct Side {
    bits: u8,
    tuple: Option<QuantizedTuple>,
}

impl Side {
    fn nbytes(&self) -> usize {
        self.tuple.as_ref().map_or(0, |t| {
            t.packed.nbytes() + t.scales.nbytes() + t.biases.nbytes()
        })
    }

    /// Quantize `x` `[B, H, S, D]` and write it at slots `at..at + S`,
    /// growing the buffers first.
    fn append(&mut self, x: &Array, at: usize, group_size: usize) {
        let (b, h, s, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
        let (packed, scales, biases) = x.quantize_weights(group_size as i32, i32::from(self.bits));
        QuantizedTuple::ensure_capacity(
            &mut self.tuple,
            GrowthPolicy::AmortizedChunked,
            b,
            h,
            (at + s as usize) as i32,
            packed_words(d as usize, self.bits) as i32,
            (d as usize / group_size) as i32,
            Dtype::Uint32.as_i32(),
            scales.dtype_raw(),
        );
        let tuple = self.tuple.as_mut().expect("ensure_capacity allocated");
        let start = [0, 0, at as i32, 0];
        let stop = |last: i32| [b, h, at as i32 + s, last];
        tuple.packed = tuple
            .packed
            .slice_set(&packed, &start, &stop(packed.dim(3)));
        tuple.scales = tuple
            .scales
            .slice_set(&scales, &start, &stop(scales.dim(3)));
        tuple.biases = tuple
            .biases
            .slice_set(&biases, &start, &stop(biases.dim(3)));
    }

    /// Slots `from..to` of the three buffers.
    fn rows(&self, from: usize, to: usize) -> (Array, Array, Array) {
        let t = self.tuple.as_ref().expect("rows of an empty side");
        let cut = |a: &Array| {
            a.slice(
                &[0, 0, from as i32, 0],
                &[a.dim(0), a.dim(1), to as i32, a.dim(3)],
            )
        };
        (cut(&t.packed), cut(&t.scales), cut(&t.biases))
    }

    /// Tokens `from..to` back as floats in `dtype`.
    fn dequantize(&self, from: usize, to: usize, group_size: usize, dtype: i32) -> Array {
        let (packed, scales, biases) = self.rows(from, to);
        let dense = packed.dequantize(&scales, &biases, group_size as i32, i32::from(self.bits));
        if dense.dtype_raw() == dtype {
            dense
        } else {
            dense.as_dtype(dtype)
        }
    }
}

/// One layer's quantized keys and values.
#[derive(Clone)]
pub(super) struct AffineLayer {
    group_size: usize,
    keys: Side,
    values: Side,
    /// Tokens held.
    offset: usize,
}

impl fmt::Debug for AffineLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffineLayer")
            .field("key_bits", &self.keys.bits)
            .field("value_bits", &self.values.bits)
            .field("group_size", &self.group_size)
            .field("offset", &self.offset)
            .finish()
    }
}

impl AffineLayer {
    pub(super) fn new(key_bits: u8, value_bits: u8, group_size: usize) -> Self {
        assert!(
            bits_supported(key_bits) && bits_supported(value_bits),
            "KV cache quantization supports {SUPPORTED_BITS:?} bits, got {key_bits} and {value_bits}"
        );
        Self {
            group_size,
            keys: Side {
                bits: key_bits,
                tuple: None,
            },
            values: Side {
                bits: value_bits,
                tuple: None,
            },
            offset: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.offset
    }

    pub(super) fn is_empty(&self) -> bool {
        self.offset == 0
    }

    pub(super) fn nbytes(&self) -> usize {
        self.keys.nbytes() + self.values.nbytes()
    }

    pub(super) fn reset(&mut self) {
        self.keys.tuple = None;
        self.values.tuple = None;
        self.offset = 0;
    }

    /// Drop the newest `n` tokens; their slots are overwritten by the next
    /// call. Returns how many were dropped.
    pub(super) fn rollback(&mut self, n: usize) -> usize {
        let n = n.min(self.offset);
        self.offset -= n;
        if self.offset == 0 {
            self.reset();
        }
        n
    }

    fn check(&self, keys: &Array, values: &Array) -> Result<(), Exception> {
        if keys.ndim() != 4 || values.ndim() != 4 || keys.dim(2) != values.dim(2) {
            return Err(Exception::custom(format!(
                "quantized KV cache: expected [B, heads, seq, dim] keys {:?} and values {:?} of equal length",
                keys.shape(),
                values.shape()
            )));
        }
        for (what, dim) in [("key", keys.dim(3)), ("value", values.dim(3))] {
            if dim as usize % self.group_size != 0 {
                return Err(Exception::custom(format!(
                    "quantized KV cache: the {what} head dimension {dim} is not a multiple of the group size {}",
                    self.group_size
                )));
            }
        }
        if let Some(held) = &self.keys.tuple
            && (held.packed.dim(0), held.packed.dim(1)) != (keys.dim(0), keys.dim(1))
        {
            return Err(Exception::custom(format!(
                "quantized KV cache holds [{}, {}, *, *]; cannot append {:?}",
                held.packed.dim(0),
                held.packed.dim(1),
                keys.shape()
            )));
        }
        Ok(())
    }

    /// Quantize and store `keys` `[B, H, S, Dk]` and `values`.
    fn append(&mut self, keys: &Array, values: &Array) {
        self.keys.append(keys, self.offset, self.group_size);
        self.values.append(values, self.offset, self.group_size);
        self.offset += keys.dim(2) as usize;
    }

    /// Store the new tokens and return every key and value: the history
    /// dequantized, the new tokens exact.
    pub(super) fn update(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.check(keys, values)?;
        let before = self.offset;
        self.append(keys, values);
        if before == 0 {
            return Ok((keys.clone(), values.clone()));
        }
        let join = |side: &Side, new: &Array| {
            let old = side.dequantize(0, before, self.group_size, new.dtype_raw());
            ops::concatenate_axis(&[&old, new], 2)
        };
        Ok((join(&self.keys, keys), join(&self.values, values)))
    }

    /// Everything held, dequantized, in `dtype`.
    #[cfg(test)]
    pub(super) fn dequantized(&self, dtype: i32) -> Option<(Array, Array)> {
        (self.offset > 0).then(|| {
            (
                self.keys.dequantize(0, self.offset, self.group_size, dtype),
                self.values
                    .dequantize(0, self.offset, self.group_size, dtype),
            )
        })
    }

    /// Attention of single-token `queries` `[B, Hq, 1, Dk]` over the cache
    /// after storing `keys` and `values`, computed on the packed rows.
    /// `window` limits the keys to the newest ones.
    pub(super) fn attend(
        &mut self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
        window: Option<usize>,
    ) -> Result<Array, Exception> {
        self.check(keys, values)?;
        if queries.dim(0) != keys.dim(0) || queries.dim(1) % keys.dim(1) != 0 {
            return Err(Exception::custom(format!(
                "quantized KV cache: queries {:?} do not fit keys {:?}",
                queries.shape(),
                keys.shape()
            )));
        }
        self.append(keys, values);
        let from = window.map_or(0, |w| self.offset.saturating_sub(w));
        let k = self.keys.rows(from, self.offset);
        let v = self.values.rows(from, self.offset);
        Ok(quantized_sdpa_kv(
            queries,
            (&k.0, &k.1, &k.2),
            (&v.0, &v.1, &v.2),
            scale,
            queries.dim(2),
            queries.dim(1),
            keys.dim(1),
            self.group_size as i32,
            i32::from(self.keys.bits),
            i32::from(self.values.bits),
            None,
        ))
    }
}
