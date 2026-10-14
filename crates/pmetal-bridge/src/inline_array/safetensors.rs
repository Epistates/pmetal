//! Safetensors I/O and the global MLX random-seed setter.
//!
//! [`try_load_safetensors_shard`] parses a `.safetensors` file once and returns
//! every tensor as an `InlineArray`, falling back to a per-tensor path if the
//! batched C++ loader fails. `InlineArray::save_safetensors` writes a file
//! atomically.

use std::mem::MaybeUninit;

use memmap2::Mmap;
use safetensors::{Dtype as SafeDtype, SafeTensors};

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;
use crate::error::{BridgeError, BridgeResult, check_last_error};

/// A load failure, naming the file.
fn load_error(path: &str, why: impl std::fmt::Display) -> BridgeError {
    BridgeError::CxxException(format!("[load_safetensors] {path}: {why}"))
}

// ── Batch safetensors loader ──────────────────────────────────────────────

/// Load all arrays from a safetensors shard in a single parse.
///
/// This is substantially faster than calling `InlineArray::load_safetensors`
/// per key because the file is parsed exactly once.
///
/// The buffers are sized from the shard's own header rather than a fixed cap.
/// ⚠️ A cap that is too small used to drop tensors **silently**: the C++ loader
/// stopped at `max_entries` and returned that count, which looks exactly like a
/// complete load. `mlx-community/gemma-4-e4b-it-4bit` ships 2770 tensors in one
/// file (text tower plus the audio and vision towers), so the old cap of 2048
/// lost 722 of them, `model.norm.weight` among them, and the loader reported a
/// missing weight for a key that was sitting in the file.
///
/// Returns `None` on I/O or parse error; [`try_load_safetensors_shard`] says
/// why.
pub fn load_safetensors_shard(path: &str) -> Option<Vec<(String, InlineArray)>> {
    try_load_safetensors_shard(path).ok()
}

/// [`load_safetensors_shard`], with the reason a load failed.
///
/// # Errors
/// The file can't be opened or its header can't be read, MLX's loader rejects
/// it and the per-tensor fallback can't read some tensor either (an
/// unsupported dtype, a misaligned tensor, an `i64` value that doesn't fit the
/// `i32` the fallback holds it in). The message carries MLX's reason and the
/// fallback's.
pub fn try_load_safetensors_shard(path: &str) -> BridgeResult<Vec<(String, InlineArray)>> {
    let c_path = std::ffi::CString::new(path)
        .map_err(|_| load_error(path, "the path contains a NUL byte"))?;
    // The header says how many tensors there are, so no cap can cut one off.
    let entries = safetensors_entry_count(path).map_err(|why| load_error(path, why))?;
    if entries == 0 {
        return Ok(Vec::new());
    }

    // Allocate key-pointer buffer.  C++ will strdup into each slot.
    let mut key_ptrs: Vec<*mut std::ffi::c_char> = vec![std::ptr::null_mut(); entries];

    // Allocate uninitialised array slots.  C++ does placement new into each
    // occupied slot; only the first `count` slots are initialised.
    let mut arr_slots: Vec<MaybeUninit<RawBuf>> = (0..entries)
        .map(|_| MaybeUninit::<RawBuf>::uninit())
        .collect();
    let capacity = i32::try_from(entries)
        .map_err(|_| load_error(path, format!("{entries} tensors are more than i32::MAX")))?;

    // A failure is handled below (the per-tensor fallback), so a strict build
    // must not panic at the op.
    let count = crate::expect_bridge_errors(|| unsafe {
        mlx_inline_load_safetensors_all(
            c_path.as_ptr(),
            key_ptrs.as_mut_ptr(),
            // Cast *mut MaybeUninit<RawBuf> → *mut RawBuf.  This is safe
            // because MaybeUninit<T> has the same layout as T.
            arr_slots.as_mut_ptr() as *mut RawBuf,
            capacity,
        )
    });

    if count < 0 {
        // Fallback: recover tensor names from the safetensors header, then load
        // each tensor from the mapped file. This keeps a load working even when
        // the batched C++ loader fails (an FP8 payload, say).
        let mlx_says = check_last_error()
            .err()
            .map_or_else(|| "MLX's loader failed".to_string(), |e| e.to_string());
        return load_safetensors_shard_fallback(path).map_err(|fallback_says| {
            load_error(
                path,
                format!("{mlx_says}; per-tensor fallback: {fallback_says}"),
            )
        });
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

    Ok(result)
}

/// How many tensors a shard holds, from its header alone.
///
/// `SafeTensors::deserialize` parses the JSON header and borrows the rest, so
/// this does not read any tensor data.
fn safetensors_entry_count(path: &str) -> Result<usize, String> {
    let mapped = map_safetensors_file(path)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|e| format!("unreadable safetensors header: {e}"))?;
    Ok(tensors.names().len())
}

fn load_safetensors_shard_fallback(path: &str) -> Result<Vec<(String, InlineArray)>, String> {
    let mapped = map_safetensors_file(path)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|e| format!("unreadable safetensors header: {e}"))?;
    let names = tensors.names();
    let mut result = Vec::with_capacity(names.len());
    for key in names {
        let tensor = tensors
            .tensor(key)
            .map_err(|e| format!("tensor `{key}`: {e}"))?;
        let array = inline_array_from_tensor_view(&tensor)
            .map_err(|why| format!("tensor `{key}`: {why}"))?;
        result.push((key.to_string(), array));
    }
    Ok(result)
}

fn map_safetensors_file(path: &str) -> Result<Mmap, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot open: {e}"))?;
    // SAFETY: the map is read through shared references while it lives and
    // copied into MLX arrays before it is dropped; a file truncated underneath
    // us is the usual mmap hazard, accepted for a model file being loaded.
    unsafe { Mmap::map(&file) }.map_err(|e| format!("cannot map: {e}"))
}

fn as_typed_slice<T>(data: &[u8]) -> Result<&[T], String> {
    // SAFETY: We only accept fully aligned, remainder-free views.
    let (prefix, values, suffix) = unsafe { data.align_to::<T>() };
    if prefix.is_empty() && suffix.is_empty() {
        Ok(values)
    } else {
        Err(format!(
            "the data is not aligned to {} bytes or not a whole number of elements",
            std::mem::size_of::<T>()
        ))
    }
}

fn shape_to_i32(shape: &[usize]) -> Result<Vec<i32>, String> {
    shape
        .iter()
        .map(|&dim| i32::try_from(dim).map_err(|_| format!("dimension {dim} exceeds i32::MAX")))
        .collect()
}

fn inline_array_from_tensor_view(
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Result<InlineArray, String> {
    let shape = shape_to_i32(tensor.shape())?;
    Ok(match tensor.dtype() {
        SafeDtype::F32 => {
            InlineArray::from_f32_slice(as_typed_slice::<f32>(tensor.data())?, &shape)
        }
        SafeDtype::I32 => {
            InlineArray::from_i32_slice_shaped(as_typed_slice::<i32>(tensor.data())?, &shape)
        }
        SafeDtype::U32 => {
            InlineArray::from_u32_slice(as_typed_slice::<u32>(tensor.data())?, &shape)
        }
        SafeDtype::U8 => InlineArray::from_u8_slice(as_typed_slice::<u8>(tensor.data())?, &shape),
        SafeDtype::F16 => {
            InlineArray::from_u16_bits_slice(as_typed_slice::<u16>(tensor.data())?, &shape, 9)
        }
        SafeDtype::BF16 => {
            InlineArray::from_u16_bits_slice(as_typed_slice::<u16>(tensor.data())?, &shape, 11)
        }
        // MLX holds an FP8 payload in a uint8 array (see `to_fp8`).
        SafeDtype::F8_E4M3 => {
            InlineArray::from_u8_slice(as_typed_slice::<u8>(tensor.data())?, &shape)
        }
        SafeDtype::I64 => {
            let values = as_typed_slice::<i64>(tensor.data())?;
            let narrowed: Result<Vec<i32>, String> = values
                .iter()
                .map(|&value| {
                    i32::try_from(value)
                        .map_err(|_| format!("the i64 value {value} does not fit an i32"))
                })
                .collect();
            InlineArray::from_i32_slice_shaped(&narrowed?, &shape)
        }
        other => return Err(format!("the dtype {other:?} is not supported")),
    })
}

/// Thin wrapper around libc free so we can call it without a libc dependency.
/// `strdup` allocates with the C allocator; we must free with the same.
unsafe fn libc_free(ptr: *mut std::ffi::c_void) {
    unsafe extern "C" {
        fn free(ptr: *mut std::ffi::c_void);
    }
    unsafe { free(ptr) }
}

// ── Single-key loader and the saver (impl on InlineArray) ────────────────

impl InlineArray {
    /// Write `entries` to `path` as one safetensors file, all or nothing.
    ///
    /// The bytes go to `<path>.partial`, are forced to disk, and are renamed
    /// over `path`, so an interrupted save or a full disk leaves the previous
    /// file as it was rather than a torn one; the partial file is removed on
    /// failure. `path` is used exactly as given (any extension). Arrays are
    /// evaluated as part of the save.
    ///
    /// # Errors
    /// A duplicate or NUL-containing name, an empty array (safetensors can't
    /// hold one), an unwritable path or a failed write.
    pub fn save_safetensors(path: &str, entries: &[(&str, &InlineArray)]) -> BridgeResult<()> {
        let invalid = |what: String| {
            Err(BridgeError::CxxException(format!(
                "[save_safetensors] {what}"
            )))
        };
        let Ok(c_path) = std::ffi::CString::new(path) else {
            return invalid(format!("path {path:?} contains a NUL byte"));
        };
        let mut c_keys = Vec::with_capacity(entries.len());
        for (key, _) in entries {
            match std::ffi::CString::new(*key) {
                Ok(c_key) => c_keys.push(c_key),
                Err(_) => return invalid(format!("tensor name {key:?} contains a NUL byte")),
            }
        }
        let key_ptrs: Vec<*const std::ffi::c_char> = c_keys.iter().map(|k| k.as_ptr()).collect();
        // The arrays as a contiguous run of buffers, copied bitwise: the
        // callee copies each array (a reference-count bump) and the
        // originals outlive the call.
        let raw_arrays: Vec<RawBuf> = entries.iter().map(|(_, a)| a.raw).collect();
        let count = i32::try_from(entries.len()).map_err(|_| {
            BridgeError::CxxException("[save_safetensors] more than i32::MAX tensors".into())
        })?;
        // The failure comes back as the `Err` below, so a strict build must
        // not panic at the op.
        crate::expect_bridge_errors(|| {
            // SAFETY: `c_path` and every key are NUL-terminated and outlive
            // the call; `raw_arrays` holds `count` initialised buffers.
            unsafe {
                mlx_inline_save_safetensors(
                    c_path.as_ptr(),
                    key_ptrs.as_ptr(),
                    raw_arrays.as_ptr(),
                    count,
                );
            }
            check_last_error()
        })
    }

    /// Load a single array from a safetensors file by key name.
    /// Loads through the bridge's own MLX instance, so the array shares its
    /// allocator with every other bridge op. `None` if the file or the key
    /// can't be read; [`Self::try_load_safetensors`] says why.
    pub fn load_safetensors(path: &str, key: &str) -> Option<Self> {
        Self::try_load_safetensors(path, key).ok()
    }

    /// [`Self::load_safetensors`], with the reason a load failed.
    ///
    /// # Errors
    /// The file can't be opened or parsed, the key isn't in it, or its dtype
    /// is one neither MLX's loader nor the fallback reads.
    pub fn try_load_safetensors(path: &str, key: &str) -> BridgeResult<Self> {
        let c_path = std::ffi::CString::new(path)
            .map_err(|_| load_error(path, "the path contains a NUL byte"))?;
        let c_key = std::ffi::CString::new(key)
            .map_err(|_| load_error(path, format!("the key {key:?} contains a NUL byte")))?;
        let mut dst = std::mem::MaybeUninit::<RawBuf>::uninit();
        // A failure is handled below (the per-tensor fallback), so a strict
        // build must not panic at the op.
        let status = crate::expect_bridge_errors(|| unsafe {
            mlx_inline_load_safetensors_key(dst.as_mut_ptr(), c_path.as_ptr(), c_key.as_ptr())
        });
        if status == 0 {
            // SAFETY: a zero status means the callee placement-new'd the array.
            return Ok(Self {
                raw: unsafe { dst.assume_init() },
            });
        }
        let mlx_says = check_last_error()
            .err()
            .map_or_else(|| "MLX's loader failed".to_string(), |e| e.to_string());
        let fallback = || -> Result<Self, String> {
            let mapped = map_safetensors_file(path)?;
            let tensors = SafeTensors::deserialize(&mapped)
                .map_err(|e| format!("unreadable safetensors header: {e}"))?;
            let tensor = tensors
                .tensor(key)
                .map_err(|e| format!("tensor `{key}`: {e}"))?;
            inline_array_from_tensor_view(&tensor)
        };
        fallback()
            .map_err(|why| load_error(path, format!("{mlx_says}; per-tensor fallback: {why}")))
    }
}

// ── Random seed (global, not per-array) ───────────────────────────────────

/// Set the global MLX random seed for reproducibility.
pub fn random_seed(seed: u64) {
    unsafe { mlx_inline_random_seed(seed) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⚠️ The regression this exists for: the loader used to size its buffers
    /// to a fixed 2048 and the C++ side stopped there, returning a count that
    /// looked like a complete load. `mlx-community/gemma-4-e4b-it-4bit` holds
    /// 2770 tensors in one file, so 722 went missing and the Gemma 4 loader
    /// reported `missing weight model.norm.weight` for a key that was in the
    /// file all along.
    #[test]
    fn a_shard_with_more_tensors_than_the_old_cap_loads_every_one() {
        const ENTRIES: usize = 2770;

        let dir = std::env::temp_dir().join("pmetal-safetensors-cap-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("many.safetensors");
        let path_str = path.to_str().expect("utf-8 path");

        let arrays: Vec<InlineArray> = (0..ENTRIES)
            .map(|i| InlineArray::from_f32_slice(&[i as f32], &[1]))
            .collect();
        let names: Vec<String> = (0..ENTRIES).map(|i| format!("tensor.{i}.weight")).collect();
        let entries: Vec<(&str, &InlineArray)> = names
            .iter()
            .map(String::as_str)
            .zip(arrays.iter())
            .collect();
        InlineArray::save_safetensors(path_str, &entries).expect("shard saves");

        let loaded = load_safetensors_shard(path_str).expect("shard loads");
        assert_eq!(loaded.len(), ENTRIES, "every tensor comes back");
        assert!(
            loaded.iter().any(|(k, _)| k == "tensor.2769.weight"),
            "the last tensor is present, not truncated away"
        );

        let _ = std::fs::remove_file(&path);
    }
}
