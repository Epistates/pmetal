//! Quantized KV cache - MLX-LM parity implementation.

use mlx_rs::{
    Array, Dtype,
    error::Exception,
    ops::concatenate_axis,
    ops::indexing::IndexOp,
    ops::{dequantize, quantize},
};

/// Quantized representation of cached K/V tensors.
#[derive(Debug, Clone)]
struct QuantizedTensor {
    /// Quantized data (packed integers).
    data: Array,
    /// Scale factors per group.
    scales: Array,
    /// Bias/zero-points per group.
    biases: Array,
}

/// Quantized KV cache that stores keys/values in lower precision.
///
/// This implementation matches MLX-LM's `QuantizedKVCache` for full parity.
/// It reduces memory usage significantly while maintaining acceptable quality:
/// - 8-bit: ~50% memory reduction
/// - 4-bit: ~75% memory reduction
///
/// # Quantization Scheme
///
/// Uses block-wise quantization with configurable group size:
/// - Each group of `group_size` elements shares a scale and bias
/// - Values are quantized as: `quantized = round((value - bias) / scale)`
/// - Dequantized as: `value = quantized * scale + bias`
///
/// # Note
///
/// Requires MLX to have quantize/dequantize operations available.
/// Falls back to standard cache if quantization fails.
#[derive(Debug)]
pub struct QuantizedKVCache {
    /// Quantized keys.
    keys: Option<QuantizedTensor>,
    /// Quantized values.
    values: Option<QuantizedTensor>,
    /// Total offset (tokens seen).
    offset: usize,
    /// Number of bits for key quantization.
    pub(crate) bits: u8,
    /// Number of bits for value quantization.
    value_bits: u8,
    /// Group size for quantization.
    pub(crate) group_size: usize,
    /// Allocation step size.
    #[allow(dead_code)] // Stored for future pre-allocation strategy
    step: usize,
    /// Original dtype for dequantization.
    dtype: Dtype,
}

impl QuantizedKVCache {
    /// Create a new quantized KV cache.
    ///
    /// # Arguments
    /// * `bits` - Number of bits (2, 4, or 8)
    /// * `group_size` - Group size for quantization (default: 64)
    pub fn new(bits: u8, group_size: usize) -> Self {
        assert!(
            bits == 2 || bits == 4 || bits == 8,
            "bits must be 2, 4, or 8"
        );
        Self {
            keys: None,
            values: None,
            offset: 0,
            bits,
            value_bits: bits,
            group_size,
            step: 256,
            dtype: Dtype::Float16,
        }
    }

    /// Create a new quantized KV cache with asymmetric key/value bit widths.
    ///
    /// # Arguments
    /// * `key_bits` - Number of bits for keys (2, 4, or 8)
    /// * `value_bits` - Number of bits for values (2, 4, or 8)
    /// * `group_size` - Group size for quantization (default: 64)
    pub fn new_asymmetric(key_bits: u8, value_bits: u8, group_size: usize) -> Self {
        assert!(
            key_bits == 2 || key_bits == 4 || key_bits == 8,
            "key_bits must be 2, 4, or 8"
        );
        assert!(
            value_bits == 2 || value_bits == 4 || value_bits == 8,
            "value_bits must be 2, 4, or 8"
        );
        Self {
            keys: None,
            values: None,
            offset: 0,
            bits: key_bits,
            value_bits,
            group_size,
            step: 256,
            dtype: Dtype::Float16,
        }
    }

    /// Get the current length.
    pub fn len(&self) -> usize {
        self.offset
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Get RoPE offset.
    pub fn rope_offset(&self) -> i32 {
        self.offset as i32
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }

    /// Pack a 4-D float tensor `[B, H, S, D]` into a [`QuantizedTensor`].
    ///
    /// MLX's `quantize` operates on 2-D matrices where groups are formed along
    /// the last dimension.  The strategy is:
    ///
    /// 1. Cast to float32 (quantize requires a floating-point input).
    /// 2. Reshape `[B, H, S, D]` → `[B*H*S, D]` so the last axis is the head
    ///    dimension, which is where the group structure lives.
    /// 3. Invoke `mlx_rs::ops::quantize`, which returns
    ///    `(w_q [rows, D/el_per_int], scales [rows, D/group_size], biases [rows, D/group_size])`.
    /// 4. Reshape each of those 2-D results back to 4-D so they can be
    ///    concatenated across the sequence dimension later.
    ///
    /// # Panics / Errors
    ///
    /// Returns an `Exception` if MLX rejects the shapes (e.g., `D` not
    /// divisible by `group_size`).  The caller is responsible for ensuring the
    /// head dimension satisfies this constraint or for padding prior to calling.
    fn quantize_with_bits(&self, tensor: &Array, bits: u8) -> Result<QuantizedTensor, Exception> {
        let shape = tensor.shape();
        let batch = shape[0] as usize;
        let heads = shape[1] as usize;
        let seq = shape[2] as usize;
        let dim = shape[3] as usize;

        // MLX quantize requires a float32 input.
        let float_tensor = tensor.as_type::<f32>()?;

        // Collapse the three leading dimensions into one so the last axis is
        // the head dimension that we want to quantize over.
        let rows = (batch * heads * seq) as i32;
        let flat = float_tensor.reshape(&[rows, dim as i32])?;

        // mlx_rs::ops::quantize returns (w_q, scales, biases).
        //   w_q    : [rows, dim * bits / 32]   (packed u32)
        //   scales : [rows, dim / group_size]
        //   biases : [rows, dim / group_size]
        let group_size_i32 = self.group_size as i32;
        let bits_i32 = bits as i32;
        let (w_q, scales_2d, biases_2d) = quantize(&flat, group_size_i32, bits_i32)?;

        // Reshape packed data back to [B, H, S, packed_dim].
        let packed_dim = w_q.dim(1);
        let data = w_q.reshape(&[batch as i32, heads as i32, seq as i32, packed_dim])?;

        // Reshape scales/biases back to [B, H, S, num_groups].
        let num_groups = scales_2d.dim(1);
        let scales = scales_2d.reshape(&[batch as i32, heads as i32, seq as i32, num_groups])?;
        let biases = biases_2d.reshape(&[batch as i32, heads as i32, seq as i32, num_groups])?;

        Ok(QuantizedTensor {
            data,
            scales,
            biases,
        })
    }

    /// Unpack a [`QuantizedTensor`] back into a float tensor `[B, H, S, D]`.
    ///
    /// This is the exact inverse of [`Self::quantize_with_bits`]:
    ///
    /// 1. Flatten the 4-D packed data and metadata to 2-D.
    /// 2. Invoke `mlx_rs::ops::dequantize`, which reconstructs a float32
    ///    matrix `[rows, D]`.
    /// 3. Reshape back to `[B, H, S, D]` and cast to the original dtype that
    ///    was fed in (stored in `self.dtype`).
    fn dequantize_with_bits(
        &self,
        qtensor: &QuantizedTensor,
        bits: u8,
    ) -> Result<Array, Exception> {
        let shape = qtensor.data.shape();
        let batch = shape[0] as usize;
        let heads = shape[1] as usize;
        let seq = shape[2] as usize;
        // packed_dim = D * bits / 32 => D = packed_dim * 32 / bits
        let packed_dim = shape[3] as usize;
        let el_per_int = 32usize / bits as usize;
        let dim = packed_dim * el_per_int;

        let rows = (batch * heads * seq) as i32;

        // Flatten to 2D for the MLX op.
        let flat_data = qtensor.data.reshape(&[rows, packed_dim as i32])?;
        let flat_scales = qtensor.scales.reshape(&[rows, qtensor.scales.dim(3)])?;
        let flat_biases = qtensor.biases.reshape(&[rows, qtensor.biases.dim(3)])?;

        // dequantize returns a float32 array [rows, dim].
        let group_size_i32 = self.group_size as i32;
        let bits_i32 = bits as i32;
        let flat_float = dequantize(
            &flat_data,
            &flat_scales,
            &flat_biases,
            group_size_i32,
            bits_i32,
        )?;

        // Restore 4-D layout [B, H, S, D] and cast to the original dtype.
        let out_4d = flat_float.reshape(&[batch as i32, heads as i32, seq as i32, dim as i32])?;

        // Cast back to the dtype we received (typically f16 / bf16).
        out_4d.as_dtype(self.dtype)
    }

    /// Update cache with new keys and values.
    pub fn update_and_fetch(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.dtype = keys.dtype();
        let num_steps = keys.dim(2) as usize;

        // Quantize new keys/values (using per-tensor bit widths for asymmetric support)
        let q_keys = self.quantize_with_bits(keys, self.bits)?;
        let q_values = self.quantize_with_bits(values, self.value_bits)?;

        if self.keys.is_none() {
            self.keys = Some(q_keys);
            self.values = Some(q_values);
        } else {
            // Concatenate quantized tensors
            let existing_k = self.keys.as_ref().unwrap();
            let existing_v = self.values.as_ref().unwrap();

            self.keys = Some(QuantizedTensor {
                data: concatenate_axis(&[&existing_k.data, &q_keys.data], 2)?,
                scales: concatenate_axis(&[&existing_k.scales, &q_keys.scales], 2)?,
                biases: concatenate_axis(&[&existing_k.biases, &q_keys.biases], 2)?,
            });
            self.values = Some(QuantizedTensor {
                data: concatenate_axis(&[&existing_v.data, &q_values.data], 2)?,
                scales: concatenate_axis(&[&existing_v.scales, &q_values.scales], 2)?,
                biases: concatenate_axis(&[&existing_v.biases, &q_values.biases], 2)?,
            });
        }

        self.offset += num_steps;

        // Dequantize for attention computation (using per-tensor bit widths)
        let dk = self.dequantize_with_bits(self.keys.as_ref().unwrap(), self.bits)?;
        let dv = self.dequantize_with_bits(self.values.as_ref().unwrap(), self.value_bits)?;

        Ok((dk, dv))
    }

    /// Check if trimmable.
    pub fn is_trimmable(&self) -> bool {
        true
    }

    /// Trim n tokens.
    pub fn trim(&mut self, n: usize) -> usize {
        let trimmed = n.min(self.offset);
        self.offset -= trimmed;
        trimmed
    }

    /// Roll back (discard) the last `n` tokens from the cache.
    pub fn rollback(&mut self, n: usize) {
        let new_offset = self.offset.saturating_sub(n);
        if new_offset == 0 {
            self.keys = None;
            self.values = None;
        } else if let (Some(k), Some(v)) = (&mut self.keys, &mut self.values) {
            *k = QuantizedTensor {
                data: k.data.index((.., .., ..new_offset as i32, ..)),
                scales: k.scales.index((.., .., ..new_offset as i32, ..)),
                biases: k.biases.index((.., .., ..new_offset as i32, ..)),
            };
            *v = QuantizedTensor {
                data: v.data.index((.., .., ..new_offset as i32, ..)),
                scales: v.scales.index((.., .., ..new_offset as i32, ..)),
                biases: v.biases.index((.., .., ..new_offset as i32, ..)),
            };
        }
        self.offset = new_offset;
    }

    /// Estimated memory usage.
    pub fn memory_usage(&self) -> usize {
        if let Some(ref k) = self.keys {
            let k_elements: usize = k.data.shape().iter().map(|&d| d as usize).product();
            let s_elements: usize = k.scales.shape().iter().map(|&d| d as usize).product();
            // data uses 4 bytes (u32), scales/biases use 2 bytes each (f16)
            let k_bytes = k_elements * 4 + s_elements * 4;
            k_bytes * 2 // K + V
        } else {
            0
        }
    }
}

/// Convenience function to create a quantized KV cache.
pub fn create_quantized_cache(bits: u8, group_size: usize) -> QuantizedKVCache {
    QuantizedKVCache::new(bits, group_size)
}
