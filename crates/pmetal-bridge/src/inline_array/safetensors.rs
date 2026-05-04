//! Bulk safetensors shard loader and global MLX random-seed setter.
//!
//! `load_safetensors_shard` parses a `.safetensors` file once and returns every
//! tensor as an `InlineArray`, falling back to a per-tensor path if the batched
//! C++ loader is unavailable.

use std::mem::MaybeUninit;

use memmap2::Mmap;
use safetensors::{Dtype as SafeDtype, SafeTensors};

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

// ── Batch safetensors loader ──────────────────────────────────────────────

/// Load all arrays from a safetensors shard in a single parse.
///
/// This is substantially faster than calling `InlineArray::load_safetensors`
/// per key because the file is parsed exactly once.  A typical model shard
/// has ~300 tensors; `MAX_ENTRIES` (2048) comfortably covers any realistic
/// shard.
///
/// Returns `None` on I/O or parse error.  Individual key allocation failures
/// (malformed UTF-8 key) are silently skipped.
pub fn load_safetensors_shard(path: &str) -> Option<Vec<(String, InlineArray)>> {
    const MAX_ENTRIES: usize = 2048;

    let c_path = std::ffi::CString::new(path).ok()?;

    // Allocate key-pointer buffer.  C++ will strdup into each slot.
    let mut key_ptrs: Vec<*mut std::ffi::c_char> = vec![std::ptr::null_mut(); MAX_ENTRIES];

    // Allocate uninitialised array slots.  C++ does placement new into each
    // occupied slot; only the first `count` slots are initialised.
    let mut arr_slots: Vec<MaybeUninit<RawBuf>> = (0..MAX_ENTRIES)
        .map(|_| MaybeUninit::<RawBuf>::uninit())
        .collect();

    let count = unsafe {
        mlx_inline_load_safetensors_all(
            c_path.as_ptr(),
            key_ptrs.as_mut_ptr(),
            // Cast *mut MaybeUninit<RawBuf> → *mut RawBuf.  This is safe
            // because MaybeUninit<T> has the same layout as T.
            arr_slots.as_mut_ptr() as *mut RawBuf,
            MAX_ENTRIES as i32,
        )
    };

    if count < 0 {
        // Fallback: recover tensor names from the safetensors header, then load
        // each tensor through the single-key bridge path. This preserves a
        // correct native load path even when the batched C++ loader fails.
        return load_safetensors_shard_fallback(path);
    }

    let count = count as usize;

    // Convert the count valid slots into owned InlineArrays + String keys.
    // We must adopt each initialised array slot so its destructor runs on drop.
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: C++ placement-new'd into slots [0, count).
        let array = InlineArray {
            raw: unsafe { arr_slots[i].assume_init() },
        };

        // key_ptrs[i] is a strdup'd C string.  Convert to Rust String and
        // free the C allocation immediately — the String owns the data.
        let key = unsafe {
            let s = std::ffi::CStr::from_ptr(key_ptrs[i])
                .to_string_lossy()
                .into_owned();
            // Free the strdup allocation.
            libc_free(key_ptrs[i] as *mut std::ffi::c_void);
            s
        };

        result.push((key, array));
    }

    Some(result)
}

fn load_safetensors_shard_fallback(path: &str) -> Option<Vec<(String, InlineArray)>> {
    let mapped = map_safetensors_file(path)?;
    let tensors = SafeTensors::deserialize(&mapped).ok()?;
    let names = tensors.names();
    let mut result = Vec::with_capacity(names.len());
    for key in names {
        let tensor = tensors.tensor(key).ok()?;
        let array = inline_array_from_tensor_view(&tensor)?;
        result.push((key.to_string(), array));
    }
    Some(result)
}

fn map_safetensors_file(path: &str) -> Option<Mmap> {
    let file = std::fs::File::open(path).ok()?;
    unsafe { Mmap::map(&file).ok() }
}

fn as_typed_slice<T>(data: &[u8]) -> Option<&[T]> {
    // SAFETY: We only accept fully aligned, remainder-free views.
    let (prefix, values, suffix) = unsafe { data.align_to::<T>() };
    if prefix.is_empty() && suffix.is_empty() {
        Some(values)
    } else {
        None
    }
}

fn shape_to_i32(shape: &[usize]) -> Option<Vec<i32>> {
    shape.iter().map(|&dim| i32::try_from(dim).ok()).collect()
}

fn inline_array_from_tensor_view(
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Option<InlineArray> {
    let shape = shape_to_i32(tensor.shape())?;
    match tensor.dtype() {
        SafeDtype::F32 => Some(InlineArray::from_f32_slice(
            as_typed_slice::<f32>(tensor.data())?,
            &shape,
        )),
        SafeDtype::I32 => Some(InlineArray::from_i32_slice_shaped(
            as_typed_slice::<i32>(tensor.data())?,
            &shape,
        )),
        SafeDtype::U32 => Some(InlineArray::from_u32_slice(
            as_typed_slice::<u32>(tensor.data())?,
            &shape,
        )),
        SafeDtype::U8 => Some(InlineArray::from_u8_slice(
            as_typed_slice::<u8>(tensor.data())?,
            &shape,
        )),
        SafeDtype::F16 => Some(InlineArray::from_u16_bits_slice(
            as_typed_slice::<u16>(tensor.data())?,
            &shape,
            9,
        )),
        SafeDtype::BF16 => Some(InlineArray::from_u16_bits_slice(
            as_typed_slice::<u16>(tensor.data())?,
            &shape,
            11,
        )),
        SafeDtype::F8_E4M3 => Some(InlineArray::from_u8_slice(
            as_typed_slice::<u8>(tensor.data())?,
            &shape,
        )),
        SafeDtype::I64 => {
            let values = as_typed_slice::<i64>(tensor.data())?;
            let cast: Vec<i32> = values.iter().map(|&value| value as i32).collect();
            Some(InlineArray::from_i32_slice_shaped(&cast, &shape))
        }
        _ => None,
    }
}

/// Thin wrapper around libc free so we can call it without a libc dependency.
/// `strdup` allocates with the C allocator; we must free with the same.
unsafe fn libc_free(ptr: *mut std::ffi::c_void) {
    unsafe extern "C" {
        fn free(ptr: *mut std::ffi::c_void);
    }
    unsafe { free(ptr) }
}

// ── Single-key loader (impl on InlineArray) ──────────────────────────────

impl InlineArray {
    /// Load a single array from a safetensors file by key name.
    /// Uses pmetal-bridge's MLX instance (not mlx-rs) — critical for avoiding
    /// dual-allocator interference.
    pub fn load_safetensors(path: &str, key: &str) -> Option<Self> {
        let c_path = std::ffi::CString::new(path).ok()?;
        let c_key = std::ffi::CString::new(key).ok()?;
        let mut dst = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            if mlx_inline_load_safetensors_key(dst.as_mut_ptr(), c_path.as_ptr(), c_key.as_ptr())
                == 0
            {
                Some(Self {
                    raw: dst.assume_init(),
                })
            } else {
                let mapped = map_safetensors_file(path)?;
                let tensors = SafeTensors::deserialize(&mapped).ok()?;
                let tensor = tensors.tensor(key).ok()?;
                inline_array_from_tensor_view(&tensor)
            }
        }
    }
}

// ── Random seed (global, not per-array) ───────────────────────────────────

/// Set the global MLX random seed for reproducibility.
pub fn random_seed(seed: u64) {
    unsafe { mlx_inline_random_seed(seed) }
}
