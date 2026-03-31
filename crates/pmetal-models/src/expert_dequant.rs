//! Expert weight dequantization: raw bytes to typed Metal buffers.
//!
//! This module bridges the I/O layer (which produces raw `Vec<u8>` from `pread`)
//! and the GPU kernel layer (which consumes typed [`MetalBuffer`]s). Given a
//! flat byte slice and an [`ExpertRecord`] that describes the layout, each of
//! the nine components is sliced, reinterpreted as the correct element type, and
//! uploaded into a Metal shared buffer.
//!
//! # Type mapping
//!
//! | Component      | Element type | Bytes/elem | Description                      |
//! |----------------|--------------|------------|----------------------------------|
//! | `*_weight`     | `u32`        | 4          | Packed 4-bit or 2-bit quant data |
//! | `*_scales`     | `u16`        | 2          | bfloat16 affine scale            |
//! | `*_biases`     | `u16`        | 2          | bfloat16 affine zero-point bias  |
//!
//! All values are stored little-endian in the packed file, matching the ARM/x86
//! host byte order on Apple Silicon. The conversion uses `chunks_exact` +
//! `from_le_bytes` to guarantee correctness without relying on unsafe transmutes
//! or additional crate dependencies.
//!
//! # Performance
//!
//! For T=1 decode (top-k=4-8 experts), each expert record is roughly 7 MB for
//! Qwen3.5-397B at 4-bit. The copy into the Metal shared buffer is a single
//! `newBufferWithBytes:length:options:` call, which DMA-copies directly into
//! unified memory. The `chunks_exact` conversion is a tight scalar loop — for
//! the weight components (u32) this is ~500 K iterations and completes in well
//! under 1 ms on M2+ hardware. Scales/biases are ~128 K iterations each.
//!
//! If this ever becomes a bottleneck, the weight component can be converted with
//! a single `std::slice::from_raw_parts` cast (alignment is guaranteed by the
//! packing format: all components start at their natural alignment boundary since
//! the layout is computed with 4-byte weight blocks followed by 2-byte
//! scale/bias blocks, each sized to a multiple of their element size).

use std::mem;
use std::sync::Arc;

use pmetal_metal::{
    ExpertWeightBuffers,
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
};

use crate::expert_layout::{ExpertComponent, ExpertRecord};

// ============================================================================
// Public API
// ============================================================================

/// Parse raw expert bytes (from `pread`) into typed Metal buffers.
///
/// The `raw` slice must cover at least [`ExpertRecord::total_size`] bytes starting
/// from the beginning of the expert record (i.e. it should be exactly the bytes
/// returned by [`crate::expert_io::ExpertOffloadContext::read_experts`] for a
/// single expert).
///
/// # Errors
///
/// Returns a descriptive `String` error if:
/// - `raw` is shorter than `record.total_size()`
/// - Any component's byte range extends beyond `raw`
/// - A component's byte count is not divisible by the element size (corrupt layout)
/// - Metal buffer creation fails (out of GPU memory)
///
/// # Example
///
/// ```ignore
/// let ctx = MetalContext::global();
/// let layout = ExpertPackLayout::load(&packed_dir);
/// let raw_bytes = offload_ctx.read_experts(layer_idx, &[expert_id]);
/// let buffers = parse_expert_weights(&raw_bytes[0], &layout.record, &ctx);
/// fused_expert.forward_single_expert(&input, &buffers, &output, &scratch);
/// ```
pub fn parse_expert_weights(
    raw: &[u8],
    record: &ExpertRecord,
    ctx: &Arc<MetalContext>,
) -> Result<ExpertWeightBuffers, String> {
    let required = record.total_size();
    if raw.len() < required {
        return Err(format!(
            "expert_dequant: raw buffer too short: got {} bytes, need {} bytes (record.total_size)",
            raw.len(),
            required,
        ));
    }

    Ok(ExpertWeightBuffers {
        gate_weights: extract_u32(raw, &record.gate_weight, ctx, "gate_weight")?,
        gate_scales: extract_u16(raw, &record.gate_scales, ctx, "gate_scales")?,
        gate_biases: extract_u16(raw, &record.gate_biases, ctx, "gate_biases")?,
        up_weights: extract_u32(raw, &record.up_weight, ctx, "up_weight")?,
        up_scales: extract_u16(raw, &record.up_scales, ctx, "up_scales")?,
        up_biases: extract_u16(raw, &record.up_biases, ctx, "up_biases")?,
        down_weights: extract_u32(raw, &record.down_weight, ctx, "down_weight")?,
        down_scales: extract_u16(raw, &record.down_scales, ctx, "down_scales")?,
        down_biases: extract_u16(raw, &record.down_biases, ctx, "down_biases")?,
    })
}

// ============================================================================
// Component extraction helpers
// ============================================================================

/// Slice `raw` at `comp.offset..comp.offset+comp.size`, convert each 4-byte
/// chunk to a little-endian `u32`, and upload into a Metal shared buffer.
fn extract_u32(
    raw: &[u8],
    comp: &ExpertComponent,
    ctx: &Arc<MetalContext>,
    name: &'static str,
) -> Result<MetalBuffer<u32>, String> {
    let bytes = component_bytes(raw, comp, name)?;
    validate_alignment(bytes.len(), mem::size_of::<u32>(), name)?;

    // Convert LE bytes → u32 elements.
    //
    // Using chunks_exact rather than a transmute cast because:
    // 1. No additional `unsafe` block needed here (we already have the
    //    `#![allow(unsafe_code)]` gate in fused_moe.rs but this file
    //    should stay safe).
    // 2. The compiler optimises this loop to a vectorised NEON load on
    //    Apple Silicon when the target bytes are already in LE order —
    //    which they always are on ARM. In practice it compiles to a plain
    //    `LDR` sequence with no byte-swap instructions.
    // 3. Correctness is guaranteed regardless of host endianness (though
    //    pmetal only targets macOS/ARM, being explicit is free).
    let elements: Vec<u32> = bytes
        .chunks_exact(mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    MetalBuffer::from_slice(ctx, &elements, BufferUsage::Shared)
        .map_err(|e| format!("expert_dequant: failed to create MetalBuffer<u32> for {name}: {e}"))
}

/// Slice `raw` at `comp.offset..comp.offset+comp.size`, convert each 2-byte
/// chunk to a little-endian `u16` (raw bfloat16 bits), and upload into a Metal
/// shared buffer.
fn extract_u16(
    raw: &[u8],
    comp: &ExpertComponent,
    ctx: &Arc<MetalContext>,
    name: &'static str,
) -> Result<MetalBuffer<u16>, String> {
    let bytes = component_bytes(raw, comp, name)?;
    validate_alignment(bytes.len(), mem::size_of::<u16>(), name)?;

    let elements: Vec<u16> = bytes
        .chunks_exact(mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    MetalBuffer::from_slice(ctx, &elements, BufferUsage::Shared)
        .map_err(|e| format!("expert_dequant: failed to create MetalBuffer<u16> for {name}: {e}"))
}

// ============================================================================
// Internal utilities
// ============================================================================

/// Return the byte slice for a single component, validating bounds.
#[inline]
fn component_bytes<'a>(
    raw: &'a [u8],
    comp: &ExpertComponent,
    name: &'static str,
) -> Result<&'a [u8], String> {
    let end = comp.offset.checked_add(comp.size).ok_or_else(|| {
        format!(
            "expert_dequant: component '{name}' offset+size overflow \
             (offset={}, size={})",
            comp.offset, comp.size,
        )
    })?;

    raw.get(comp.offset..end).ok_or_else(|| {
        format!(
            "expert_dequant: component '{name}' out of bounds: \
             offset={}, size={}, raw.len()={}",
            comp.offset,
            comp.size,
            raw.len(),
        )
    })
}

/// Validate that `byte_len` is an exact multiple of `elem_size`.
///
/// A mismatch indicates a corrupt or mismatched layout descriptor.
#[inline]
fn validate_alignment(byte_len: usize, elem_size: usize, name: &'static str) -> Result<(), String> {
    if byte_len % elem_size != 0 {
        return Err(format!(
            "expert_dequant: component '{name}' byte count ({byte_len}) is not \
             divisible by element size ({elem_size}); layout descriptor may be corrupt",
        ));
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_layout::{ExpertPackLayout, PackedBits};

    /// Build a synthetic raw expert buffer filled with recognisable byte patterns.
    ///
    /// Weight bytes cycle 0x00..=0xFF; scale/bias bytes use a distinct pattern
    /// (0x3F80 = 1.0 in bf16, repeated) so we can verify no cross-component
    /// contamination.
    fn make_synthetic_raw(record: &crate::expert_layout::ExpertRecord) -> Vec<u8> {
        let total = record.total_size();
        let mut raw = vec![0u8; total];

        // Fill weight components with a counting pattern (LE u32 = 0, 1, 2, …)
        for comp in [&record.gate_weight, &record.up_weight, &record.down_weight] {
            for (i, chunk) in raw[comp.offset..comp.offset + comp.size]
                .chunks_exact_mut(4)
                .enumerate()
            {
                chunk.copy_from_slice(&(i as u32).to_le_bytes());
            }
        }

        // Fill scale/bias components with bf16 1.0 = 0x3F80
        for comp in [
            &record.gate_scales,
            &record.gate_biases,
            &record.up_scales,
            &record.up_biases,
            &record.down_scales,
            &record.down_biases,
        ] {
            for chunk in raw[comp.offset..comp.offset + comp.size].chunks_exact_mut(2) {
                chunk.copy_from_slice(&0x3F80u16.to_le_bytes());
            }
        }

        raw
    }

    /// Smoke-test: parse a small synthetic expert and verify element counts and
    /// spot-check values. Requires a Metal device (macOS only).
    #[test]
    #[cfg(target_os = "macos")]
    fn test_parse_expert_weights_counts() {
        let ctx = MetalContext::global().unwrap();

        // Small synthetic config: hidden=256, intermediate=128, group=64, 4-bit
        let record = ExpertRecord::compute(256, 128, 64, PackedBits::Four);
        let raw = make_synthetic_raw(&record);

        let buffers =
            parse_expert_weights(&raw, &record, &ctx).expect("parse_expert_weights should succeed");

        // gate_weight: [128, 256/8] = [128, 32] → 4096 u32 elements
        assert_eq!(
            buffers.gate_weights.len(),
            record.gate_weight.size / mem::size_of::<u32>(),
            "gate_weights element count mismatch"
        );

        // gate_scales: [128, 256/64] = [128, 4] → 512 u16 elements
        assert_eq!(
            buffers.gate_scales.len(),
            record.gate_scales.size / mem::size_of::<u16>(),
            "gate_scales element count mismatch"
        );

        // Spot-check: first weight element should be 0 (counting pattern)
        assert_eq!(
            buffers.gate_weights.as_slice()[0],
            0u32,
            "gate_weights[0] should be 0"
        );
        // Second element should be 1
        assert_eq!(
            buffers.gate_weights.as_slice()[1],
            1u32,
            "gate_weights[1] should be 1"
        );

        // Spot-check: all scale elements should be 0x3F80 (bf16 1.0)
        assert!(
            buffers.gate_scales.as_slice().iter().all(|&v| v == 0x3F80),
            "all gate_scales should be bf16 1.0 (0x3F80)"
        );
        assert!(
            buffers.down_biases.as_slice().iter().all(|&v| v == 0x3F80),
            "all down_biases should be bf16 1.0 (0x3F80)"
        );
    }

    /// Verify that a short raw buffer is rejected with a clear error.
    #[test]
    #[cfg(target_os = "macos")]
    fn test_parse_expert_weights_short_buffer() {
        let ctx = MetalContext::global().unwrap();
        let record = ExpertRecord::compute(256, 128, 64, PackedBits::Four);
        let short = vec![0u8; record.total_size() - 1];

        let result = parse_expert_weights(&short, &record, &ctx);
        assert!(result.is_err(), "should fail on short buffer");
        let msg = result.err().unwrap();
        assert!(
            msg.contains("raw buffer too short"),
            "error message should mention short buffer: {msg}"
        );
    }

    /// Verify that a misaligned component size is caught.
    ///
    /// This tests the internal `validate_alignment` helper by constructing an
    /// `ExpertComponent` with an odd byte count for a u32 component.
    #[test]
    fn test_validate_alignment_rejects_odd_u32() {
        let result = validate_alignment(7, mem::size_of::<u32>(), "test_component");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("not divisible"), "message: {msg}");
    }

    /// Verify that component_bytes catches an out-of-bounds component.
    #[test]
    fn test_component_bytes_out_of_bounds() {
        let raw = vec![0u8; 16];
        let comp = ExpertComponent {
            offset: 10,
            size: 10, // 10+10=20 > 16
            shape: vec![5, 2],
        };
        let result = component_bytes(&raw, &comp, "test");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("out of bounds"), "message: {msg}");
    }

    /// Verify that component_bytes catches offset+size overflow.
    #[test]
    fn test_component_bytes_overflow() {
        let raw = vec![0u8; 16];
        let comp = ExpertComponent {
            offset: usize::MAX,
            size: 1,
            shape: vec![],
        };
        let result = component_bytes(&raw, &comp, "overflow_test");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("overflow"), "message: {msg}");
    }

    /// Cross-check element counts against the Qwen3.5-style config used in
    /// `expert_layout` tests (hidden=4096, intermediate=1024, group=64, 4-bit).
    ///
    /// This is a pure arithmetic test — no Metal device needed.
    #[test]
    fn test_element_counts_qwen3_5_config() {
        let record = ExpertRecord::compute(4096, 1024, 64, PackedBits::Four);

        // gate_weight: [1024, 4096/8] * 4 bytes = 2,097,152 bytes → 524,288 u32
        assert_eq!(record.gate_weight.size, 1024 * 512 * 4);
        assert_eq!(record.gate_weight.size / mem::size_of::<u32>(), 524_288);

        // gate_scales: [1024, 4096/64] * 2 bytes = 131,072 bytes → 65,536 u16
        assert_eq!(record.gate_scales.size, 1024 * 64 * 2);
        assert_eq!(record.gate_scales.size / mem::size_of::<u16>(), 65_536);

        // down_weight: [4096, 1024/8] * 4 bytes = [4096, 128] * 4 = 2,097,152 bytes → 524,288 u32
        assert_eq!(record.down_weight.size, 4096 * 128 * 4);
        assert_eq!(record.down_weight.size / mem::size_of::<u32>(), 524_288);

        // down_scales: [4096, 1024/64] * 2 bytes = [4096, 16] * 2 = 131,072 bytes → 65,536 u16
        assert_eq!(record.down_scales.size, 4096 * 16 * 2);
        assert_eq!(record.down_scales.size / mem::size_of::<u16>(), 65_536);

        // Total = 3 * gate_w + 3 * gate_s + 3 * gate_b (same as gate for up, different for down)
        // Verified separately by expert_layout tests.
        assert_eq!(record.total_size(), 7_077_888);
    }
}
