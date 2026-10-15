//! Reading what a safetensors file declares about itself, from its header
//! alone: how many tensors, of which element types, and whether the file is
//! the length that header promises.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use pmetal_core::{PMetalError, Result};

/// The largest header the format allows (the reference implementation's limit).
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Parameters and bytes of one element type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DtypeTotals {
    /// Elements, summed over the tensors of this type.
    pub elements: u64,
    /// Bytes of tensor data.
    pub bytes: u64,
}

/// What a safetensors header says a file holds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SafetensorsSummary {
    /// Number of tensors.
    pub tensors: usize,
    /// Tensor data, in total, by element type as written in the header
    /// (`BF16`, `F32`, `U32`, ...).
    pub by_dtype: BTreeMap<String, DtypeTotals>,
    /// Size of the file in bytes.
    pub file_bytes: u64,
}

impl SafetensorsSummary {
    /// Bytes of tensor data.
    pub fn data_bytes(&self) -> u64 {
        self.by_dtype.values().map(|t| t.bytes).sum()
    }

    /// Elements across all tensors. Packed integer types (quantized weights
    /// stored as `U32`) count their storage words, not the weights inside.
    pub fn elements(&self) -> u64 {
        self.by_dtype.values().map(|t| t.elements).sum()
    }

    /// Add another file's totals (a sharded checkpoint).
    pub fn merge(&mut self, other: &Self) {
        self.tensors += other.tensors;
        self.file_bytes += other.file_bytes;
        for (dtype, totals) in &other.by_dtype {
            let slot = self.by_dtype.entry(dtype.clone()).or_default();
            slot.elements += totals.elements;
            slot.bytes += totals.bytes;
        }
    }
}

/// Bits per element of a safetensors dtype name.
pub fn dtype_bits(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F64" | "I64" | "U64" | "C64" => 64,
        "F32" | "I32" | "U32" => 32,
        "F16" | "BF16" | "I16" | "U16" => 16,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 8,
        "F6_E2M3" | "F6_E3M2" => 6,
        "F4" => 4,
        _ => return None,
    })
}

fn bad(path: &Path, what: impl std::fmt::Display) -> PMetalError {
    PMetalError::Hub(format!("{}: {what}", path.display()))
}

/// Summarize a safetensors file from its header, checking that the header is
/// well formed and that the file is exactly as long as it declares. A file cut
/// short by an interrupted transfer fails this, whatever its tail holds.
pub fn summarize_safetensors(path: &Path) -> Result<SafetensorsSummary> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .map_err(|_| bad(path, "too short to be a safetensors file"))?;
    let header_len = u64::from_le_bytes(len);
    if header_len > MAX_HEADER_BYTES || header_len + 8 > file_bytes {
        return Err(bad(
            path,
            "the header length is wrong; not a safetensors file, or cut short",
        ));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    summarize_header(&header, file_bytes, path)
}

fn summarize_header(header: &[u8], file_bytes: u64, path: &Path) -> Result<SafetensorsSummary> {
    let value: serde_json::Value = serde_json::from_slice(header)
        .map_err(|e| bad(path, format!("header is not JSON: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| bad(path, "header is not a JSON object"))?;
    let data_len = file_bytes - 8 - header.len() as u64;

    let mut summary = SafetensorsSummary {
        file_bytes,
        ..Default::default()
    };
    let mut end_of_data = 0u64;
    for (name, tensor) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = tensor
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| bad(path, format!("tensor `{name}` has no dtype")))?;
        let bits = dtype_bits(dtype).ok_or_else(|| {
            bad(
                path,
                format!("tensor `{name}` has the unknown dtype {dtype}"),
            )
        })?;
        let elements = tensor
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| bad(path, format!("tensor `{name}` has no shape")))?
            .iter()
            .try_fold(1u64, |acc, d| d.as_u64().and_then(|d| acc.checked_mul(d)))
            .ok_or_else(|| bad(path, format!("tensor `{name}` has an invalid shape")))?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .filter(|o| o.len() == 2)
            .and_then(|o| Some((o[0].as_u64()?, o[1].as_u64()?)))
            .ok_or_else(|| bad(path, format!("tensor `{name}` has invalid data_offsets")))?;
        let bytes = elements
            .checked_mul(bits)
            .map(|b| b.div_ceil(8))
            .ok_or_else(|| bad(path, format!("tensor `{name}` has an invalid shape")))?;
        if offsets.1 < offsets.0 || offsets.1 - offsets.0 != bytes {
            return Err(bad(
                path,
                format!(
                    "tensor `{name}` declares {bytes} bytes but its offsets span {}",
                    offsets.1.saturating_sub(offsets.0)
                ),
            ));
        }
        end_of_data = end_of_data.max(offsets.1);
        let slot = summary.by_dtype.entry(dtype.to_owned()).or_default();
        slot.elements += elements;
        slot.bytes += bytes;
        summary.tensors += 1;
    }
    if end_of_data != data_len {
        return Err(bad(
            path,
            format!(
                "the header accounts for {end_of_data} bytes of data but the file holds {data_len}: \
                 the transfer was cut short or the file is damaged"
            ),
        ));
    }
    Ok(summary)
}

/// Write a safetensors file for tests: `(name, dtype, shape)` tensors, each
/// filled with zeros.
#[cfg(test)]
pub(crate) fn write_test_safetensors(path: &Path, tensors: &[(&str, &str, &[u64])]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape) in tensors {
        let bytes = (shape.iter().product::<u64>() * dtype_bits(dtype).unwrap()).div_ceil(8);
        header.insert(
            (*name).to_owned(),
            serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes] }),
        );
        offset += bytes;
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(vec![0u8; offset as usize]);
    std::fs::write(path, out).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_well_formed_file_is_summarized_by_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.safetensors");
        write_test_safetensors(
            &path,
            &[
                ("a", "BF16", &[4, 8]),
                ("b", "BF16", &[2]),
                ("c", "U32", &[16, 4]),
                ("d", "F32", &[3]),
            ],
        );
        let s = summarize_safetensors(&path).unwrap();
        assert_eq!(s.tensors, 4);
        assert_eq!(
            s.by_dtype["BF16"],
            DtypeTotals {
                elements: 34,
                bytes: 68
            }
        );
        assert_eq!(
            s.by_dtype["U32"],
            DtypeTotals {
                elements: 64,
                bytes: 256
            }
        );
        assert_eq!(s.data_bytes(), 68 + 256 + 12);
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.safetensors");
        write_test_safetensors(&path, &[("a", "F16", &[1000])]);
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 10]).unwrap();
        let err = summarize_safetensors(&path).unwrap_err().to_string();
        assert!(err.contains("cut short"), "{err}");
    }

    #[test]
    fn trailing_garbage_is_rejected_too() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.safetensors");
        write_test_safetensors(&path, &[("a", "F16", &[10])]);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend([0u8; 7]);
        std::fs::write(&path, bytes).unwrap();
        assert!(summarize_safetensors(&path).is_err());
    }

    #[test]
    fn non_safetensors_input_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.safetensors");
        for junk in [&b""[..], b"short", &[0xff; 64][..], b"\x02\0\0\0\0\0\0\0{}"] {
            std::fs::write(&path, junk).unwrap();
            // "{}" is a valid empty header: an empty file with no tensors.
            let _ = summarize_safetensors(&path);
        }
        std::fs::write(&path, [0xff; 64]).unwrap();
        assert!(summarize_safetensors(&path).is_err());
    }

    #[test]
    fn dtype_widths_cover_the_quantization_formats() {
        for (dtype, bits) in [
            ("BF16", 16),
            ("F8_E4M3", 8),
            ("F4", 4),
            ("U32", 32),
            ("I64", 64),
        ] {
            assert_eq!(dtype_bits(dtype), Some(bits), "{dtype}");
        }
        assert_eq!(dtype_bits("X9"), None);
    }
}
