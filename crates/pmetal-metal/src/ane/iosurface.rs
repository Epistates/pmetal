//! IOSurface zero-copy data transfer for ANE.
//!
//! IOSurfaces serve as the shared-memory interface between CPU and ANE.
//! All surfaces use a flat 1D layout with channel-first `[C, S]` data.
//! Data is stored as fp16 on the surface, with NEON-accelerated conversion
//! to/from f32 during read/write operations.

use std::ffi::c_void;

use crate::error::{MetalError, Result};
use crate::neon_convert::{f16_to_f32_bulk, f32_to_f16_bulk};

// IOSurface.framework FFI (linked at build time when ane feature is active)
#[allow(dead_code)]
mod ffi {
    use std::ffi::c_void;

    // IOSurface lock flags
    pub const K_IO_SURFACE_LOCK_READ_ONLY: u32 = 1;

    // CoreFoundation string constants (resolved at runtime via CFString)
    // We use the raw symbol names from IOSurface.framework
    unsafe extern "C" {
        // IOSurface creation and management
        pub fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
        pub fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
        pub fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;

        // CoreFoundation
        pub fn CFRelease(cf: *const c_void);
    }
}

/// An IOSurface for ANE data transfer.
///
/// Wraps an `IOSurfaceRef` with RAII cleanup. Data is stored as fp16 on the
/// surface with NEON-accelerated f32↔fp16 conversion during read/write.
pub struct IoSurface {
    /// The underlying IOSurfaceRef (CFRetained).
    surface: *mut c_void,
    /// Total size in bytes.
    size_bytes: usize,
}

// SAFETY: IOSurface is a kernel-backed shared memory object, safe to send between threads.
unsafe impl Send for IoSurface {}
unsafe impl Sync for IoSurface {}

impl IoSurface {
    /// Create a new IOSurface with the given size in bytes.
    ///
    /// Uses a flat 1D layout: width=bytes, height=1, bytesPerElement=1, pixelFormat=0.
    pub fn new(size_bytes: usize) -> Result<Self> {
        let surface = unsafe { create_surface(size_bytes) };
        if surface.is_null() {
            return Err(MetalError::IoSurfaceCreation {
                size: size_bytes,
                reason: "IOSurfaceCreate returned NULL".into(),
            });
        }
        Ok(Self {
            surface,
            size_bytes,
        })
    }

    /// Create an IOSurface sized for `channels * spatial` fp16 elements.
    pub fn for_tensor(channels: usize, spatial: usize) -> Result<Self> {
        Self::new(channels * spatial * 2) // 2 bytes per fp16
    }

    /// Create an IOSurface sized for `channels * spatial` fp32 elements.
    ///
    /// Used by the dynamic weight pipeline where activations and weights are
    /// packed as fp32 in the spatial dimension (MIL handles fp32→fp16 cast).
    pub fn for_tensor_f32(channels: usize, spatial: usize) -> Result<Self> {
        Self::new(channels * spatial * 4) // 4 bytes per fp32
    }

    /// Get the raw IOSurfaceRef pointer (for use with ANE APIs).
    pub fn as_ptr(&self) -> *mut c_void {
        self.surface
    }

    /// Size of this surface in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Write f32 data as fp16 to the surface.
    ///
    /// Converts `data` (f32, channel-first `[C, S]`) to fp16 and writes to the surface.
    /// Uses NEON 8-wide vectorized conversion.
    pub fn write_f32_as_fp16(&self, data: &[f32], channels: usize, spatial: usize) {
        let n = channels * spatial;
        debug_assert!(n * 2 <= self.size_bytes);

        unsafe {
            ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
            let base = ffi::IOSurfaceGetBaseAddress(self.surface) as *mut u16;
            let dst = std::slice::from_raw_parts_mut(base, n);
            f32_to_f16_bulk(data, dst);
            ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
        }
    }

    /// Write f32 data as fp16 at a channel offset within the surface.
    ///
    /// Writes starting at `ch_offset * spatial` elements into the surface.
    pub fn write_f32_as_fp16_at(
        &self,
        ch_offset: usize,
        data: &[f32],
        channels: usize,
        spatial: usize,
    ) {
        let n = channels * spatial;
        let offset = ch_offset * spatial;
        debug_assert!((offset + n) * 2 <= self.size_bytes);

        unsafe {
            ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
            let base = ffi::IOSurfaceGetBaseAddress(self.surface) as *mut u16;
            let dst = std::slice::from_raw_parts_mut(base.add(offset), n);
            f32_to_f16_bulk(data, dst);
            ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
        }
    }

    /// Read fp16 data from the surface as f32.
    ///
    /// Reads `channels * spatial` elements starting at `ch_offset` channels.
    /// Uses NEON 8-wide vectorized conversion.
    pub fn read_fp16_as_f32(
        &self,
        dst: &mut [f32],
        ch_offset: usize,
        channels: usize,
        spatial: usize,
    ) {
        let n = channels * spatial;
        let offset = ch_offset * spatial;
        debug_assert_eq!(dst.len(), n);
        debug_assert!((offset + n) * 2 <= self.size_bytes);

        unsafe {
            ffi::IOSurfaceLock(
                self.surface,
                ffi::K_IO_SURFACE_LOCK_READ_ONLY,
                std::ptr::null_mut(),
            );
            let base = ffi::IOSurfaceGetBaseAddress(self.surface) as *const u16;
            let src = std::slice::from_raw_parts(base.add(offset), n);
            f16_to_f32_bulk(src, dst);
            ffi::IOSurfaceUnlock(
                self.surface,
                ffi::K_IO_SURFACE_LOCK_READ_ONLY,
                std::ptr::null_mut(),
            );
        }
    }

    /// Write f32 data at a channel offset within an fp32 IOSurface.
    ///
    /// Writes `channels * spatial` f32 elements starting at `ch_offset * spatial`.
    /// No dtype conversion — used for writing activations into fp32 surfaces
    /// (e.g. concatenating Q, K, V for the attention kernel).
    pub fn write_f32_at(
        &self,
        ch_offset: usize,
        data: &[f32],
        channels: usize,
        spatial: usize,
    ) {
        let n = channels * spatial;
        let offset = ch_offset * spatial;
        debug_assert_eq!(data.len(), n);
        debug_assert!((offset + n) * 4 <= self.size_bytes);

        unsafe {
            ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
            let base = ffi::IOSurfaceGetBaseAddress(self.surface) as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), base.add(offset), n);
            ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
        }
    }

    /// Write packed fp32 data for the dynamic weight pipeline.
    ///
    /// Packs activations and weight columns into a single IOSurface using
    /// per-channel interleaved layout: `surface[ch][0:seq] = act[ch][0:seq]`,
    /// then `surface[ch][seq:seq+wc] = weight_row[ch][0:wc]` for each weight.
    ///
    /// Layout: `[1, IC, 1, SEQ + total_weight_cols]` fp32
    /// - `sp[0:seq]` = activations `[ic, seq]`
    /// - `sp[seq:seq+w0_cols]` = weight0 row per channel
    /// - `sp[seq+w0_cols:seq+w0_cols+w1_cols]` = weight1 row per channel
    /// - etc.
    ///
    /// `weights` is a slice of `(data, cols)` pairs. Each weight matrix is
    /// `[ic, cols]` row-major f32.
    pub fn write_packed_f32(
        &self,
        act: &[f32],
        weights: &[(&[f32], usize)],
        ic: usize,
        seq: usize,
    ) {
        let total_weight_cols: usize = weights.iter().map(|(_, c)| *c).sum();
        let sp = seq + total_weight_cols;
        debug_assert!(ic * sp * 4 <= self.size_bytes);
        debug_assert_eq!(act.len(), ic * seq);

        unsafe {
            ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
            let base = ffi::IOSurfaceGetBaseAddress(self.surface) as *mut f32;

            for ch in 0..ic {
                // Copy activation row: act[ch*seq .. ch*seq + seq]
                std::ptr::copy_nonoverlapping(act.as_ptr().add(ch * seq), base.add(ch * sp), seq);

                // Copy weight rows at spatial offsets
                let mut w_off = seq;
                for &(w_data, w_cols) in weights {
                    debug_assert_eq!(w_data.len(), ic * w_cols);
                    std::ptr::copy_nonoverlapping(
                        w_data.as_ptr().add(ch * w_cols),
                        base.add(ch * sp + w_off),
                        w_cols,
                    );
                    w_off += w_cols;
                }
            }

            ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
        }
    }

    /// Write packed fp32 data with multiple activation streams.
    ///
    /// Like `write_packed_f32` but supports multiple activation buffers packed
    /// sequentially in the spatial dimension before the weight columns.
    ///
    /// Layout: `[1, IC, 1, act0_cols + act1_cols + ... + total_weight_cols]`
    /// `acts` is `[(data, cols)]`, `weights` is `[(data, cols)]`.
    pub fn write_packed_f32_multi(
        &self,
        acts: &[(&[f32], usize)],
        weights: &[(&[f32], usize)],
        ic: usize,
    ) {
        let total_act_cols: usize = acts.iter().map(|(_, c)| *c).sum();
        let total_weight_cols: usize = weights.iter().map(|(_, c)| *c).sum();
        let sp = total_act_cols + total_weight_cols;
        debug_assert!(ic * sp * 4 <= self.size_bytes);

        unsafe {
            ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
            let base = ffi::IOSurfaceGetBaseAddress(self.surface) as *mut f32;

            for ch in 0..ic {
                let mut off = 0;

                // Copy activation rows
                for &(a_data, a_cols) in acts {
                    debug_assert_eq!(a_data.len(), ic * a_cols);
                    std::ptr::copy_nonoverlapping(
                        a_data.as_ptr().add(ch * a_cols),
                        base.add(ch * sp + off),
                        a_cols,
                    );
                    off += a_cols;
                }

                // Copy weight rows
                for &(w_data, w_cols) in weights {
                    debug_assert_eq!(w_data.len(), ic * w_cols);
                    std::ptr::copy_nonoverlapping(
                        w_data.as_ptr().add(ch * w_cols),
                        base.add(ch * sp + off),
                        w_cols,
                    );
                    off += w_cols;
                }
            }

            ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
        }
    }

    /// Read fp32 data directly from an fp32 IOSurface.
    ///
    /// Reads `channels * spatial` fp32 elements starting at `ch_offset`.
    /// No dtype conversion — used with dynamic pipeline fp32 output surfaces.
    pub fn read_f32(&self, dst: &mut [f32], ch_offset: usize, channels: usize, spatial: usize) {
        let n = channels * spatial;
        let offset = ch_offset * spatial;
        debug_assert_eq!(dst.len(), n);
        debug_assert!((offset + n) * 4 <= self.size_bytes);

        unsafe {
            ffi::IOSurfaceLock(
                self.surface,
                ffi::K_IO_SURFACE_LOCK_READ_ONLY,
                std::ptr::null_mut(),
            );
            let base = ffi::IOSurfaceGetBaseAddress(self.surface) as *const f32;
            std::ptr::copy_nonoverlapping(base.add(offset), dst.as_mut_ptr(), n);
            ffi::IOSurfaceUnlock(
                self.surface,
                ffi::K_IO_SURFACE_LOCK_READ_ONLY,
                std::ptr::null_mut(),
            );
        }
    }

    /// Copy fp16 data directly between IOSurfaces (avoids f32 round-trip).
    ///
    /// Copies `channels * spatial` fp16 elements from `src` (at `src_ch_offset`)
    /// to `self` (at `dst_ch_offset`).
    pub fn copy_from(
        &self,
        dst_ch_offset: usize,
        src: &IoSurface,
        src_ch_offset: usize,
        channels: usize,
        spatial: usize,
    ) {
        let n = channels * spatial;
        let dst_off = dst_ch_offset * spatial;
        let src_off = src_ch_offset * spatial;
        let bytes = n * 2; // fp16 = 2 bytes

        debug_assert!((dst_off + n) * 2 <= self.size_bytes);
        debug_assert!((src_off + n) * 2 <= src.size_bytes);

        unsafe {
            ffi::IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
            ffi::IOSurfaceLock(
                src.surface,
                ffi::K_IO_SURFACE_LOCK_READ_ONLY,
                std::ptr::null_mut(),
            );

            let dst_base = ffi::IOSurfaceGetBaseAddress(self.surface) as *mut u8;
            let src_base = ffi::IOSurfaceGetBaseAddress(src.surface) as *const u8;

            std::ptr::copy_nonoverlapping(
                src_base.add(src_off * 2),
                dst_base.add(dst_off * 2),
                bytes,
            );

            ffi::IOSurfaceUnlock(
                src.surface,
                ffi::K_IO_SURFACE_LOCK_READ_ONLY,
                std::ptr::null_mut(),
            );
            ffi::IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
        }
    }
}

impl Drop for IoSurface {
    fn drop(&mut self) {
        if !self.surface.is_null() {
            unsafe {
                ffi::CFRelease(self.surface as *const c_void);
            }
        }
    }
}

/// Create an IOSurface with flat 1D layout via CoreFoundation dictionary.
unsafe fn create_surface(size_bytes: usize) -> *mut c_void {
    use objc2::msg_send;
    use objc2_foundation::{NSNumber, NSString};

    // Build the properties dictionary using Foundation types
    // Keys are the IOSurface property constants
    let keys = [
        NSString::from_str("IOSurfaceWidth"),
        NSString::from_str("IOSurfaceHeight"),
        NSString::from_str("IOSurfaceBytesPerElement"),
        NSString::from_str("IOSurfaceBytesPerRow"),
        NSString::from_str("IOSurfaceAllocSize"),
        NSString::from_str("IOSurfacePixelFormat"),
    ];
    let values = [
        NSNumber::new_usize(size_bytes),
        NSNumber::new_usize(1),
        NSNumber::new_usize(1),
        NSNumber::new_usize(size_bytes),
        NSNumber::new_usize(size_bytes),
        NSNumber::new_usize(0),
    ];

    let dict_cls = objc2::runtime::AnyClass::get(c"NSMutableDictionary").unwrap();
    let dict: *mut objc2::runtime::AnyObject = msg_send![dict_cls, new];
    for (k, v) in keys.iter().zip(values.iter()) {
        let _: () = msg_send![dict, setObject: &**v, forKey: &**k];
    }

    unsafe { ffi::IOSurfaceCreate(dict as *const c_void) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iosurface_create() {
        let surface = IoSurface::new(1024);
        match surface {
            Ok(s) => {
                assert_eq!(s.size_bytes(), 1024);
                assert!(!s.as_ptr().is_null());
            }
            Err(_) => {
                // May fail on non-macOS or restricted environments
            }
        }
    }

    #[test]
    fn test_iosurface_write_read_roundtrip() {
        let channels = 4;
        let spatial = 8;
        let surface = match IoSurface::for_tensor(channels, spatial) {
            Ok(s) => s,
            Err(_) => return, // Skip on unsupported platforms
        };

        // Write known data
        let data: Vec<f32> = (0..channels * spatial).map(|i| i as f32 * 0.1).collect();
        surface.write_f32_as_fp16(&data, channels, spatial);

        // Read it back
        let mut out = vec![0.0f32; channels * spatial];
        surface.read_fp16_as_f32(&mut out, 0, channels, spatial);

        // Verify within fp16 precision
        for (i, (orig, read)) in data.iter().zip(out.iter()).enumerate() {
            let expected = half::f16::from_f32(*orig).to_f32();
            assert!(
                (read - expected).abs() < 0.01,
                "Mismatch at {i}: orig={orig}, read={read}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_iosurface_f32_write_read() {
        let channels = 4;
        let spatial = 8;
        let surface = match IoSurface::for_tensor_f32(channels, spatial) {
            Ok(s) => s,
            Err(_) => return,
        };

        // Write known f32 data using packed write (no weights)
        let data: Vec<f32> = (0..channels * spatial).map(|i| i as f32 * 0.5).collect();
        surface.write_packed_f32(&data, &[], channels, spatial);

        // Read it back
        let mut out = vec![0.0f32; channels * spatial];
        surface.read_f32(&mut out, 0, channels, spatial);

        for (i, (orig, read)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                (read - orig).abs() < 1e-6,
                "Mismatch at {i}: orig={orig}, read={read}"
            );
        }
    }

    #[test]
    fn test_iosurface_packed_f32_with_weights() {
        let ic = 4;
        let seq = 8;
        let oc = 3;
        let sp = seq + oc;
        let surface = match IoSurface::for_tensor_f32(ic, sp) {
            Ok(s) => s,
            Err(_) => return,
        };

        let act: Vec<f32> = (0..ic * seq).map(|i| i as f32).collect();
        let weight: Vec<f32> = (0..ic * oc).map(|i| 100.0 + i as f32).collect();

        surface.write_packed_f32(&act, &[(&weight, oc)], ic, seq);

        // Read back the full spatial extent and verify layout
        let mut out = vec![0.0f32; ic * sp];
        surface.read_f32(&mut out, 0, ic, sp);

        // Verify per-channel: out[ch*sp .. ch*sp+seq] == act, out[ch*sp+seq .. ch*sp+sp] == weight
        for ch in 0..ic {
            for t in 0..seq {
                assert!(
                    (out[ch * sp + t] - act[ch * seq + t]).abs() < 1e-6,
                    "Act mismatch at ch={ch}, t={t}"
                );
            }
            for w in 0..oc {
                assert!(
                    (out[ch * sp + seq + w] - weight[ch * oc + w]).abs() < 1e-6,
                    "Weight mismatch at ch={ch}, w={w}"
                );
            }
        }
    }

    #[test]
    fn test_iosurface_copy() {
        let channels = 4;
        let spatial = 8;
        let src = match IoSurface::for_tensor(channels, spatial) {
            Ok(s) => s,
            Err(_) => return,
        };
        let dst = match IoSurface::for_tensor(channels, spatial) {
            Ok(s) => s,
            Err(_) => return,
        };

        let data: Vec<f32> = (0..channels * spatial).map(|i| i as f32).collect();
        src.write_f32_as_fp16(&data, channels, spatial);

        // Copy src → dst
        dst.copy_from(0, &src, 0, channels, spatial);

        // Read from dst
        let mut out = vec![0.0f32; channels * spatial];
        dst.read_fp16_as_f32(&mut out, 0, channels, spatial);

        for (i, (orig, read)) in data.iter().zip(out.iter()).enumerate() {
            let expected = half::f16::from_f32(*orig).to_f32();
            assert!(
                (read - expected).abs() < 0.1,
                "Mismatch at {i}: orig={orig}, read={read}"
            );
        }
    }
}
