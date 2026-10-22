//! Importance Matrix (IMatrix) implementation.
//!
//! IMatrix stores the importance of each weight matrix in the model,
//! usually calculated as the accumulated squared activations of the inputs
//! to that weight matrix during a calibration pass.
//!
//! # Binary Format (legacy `.dat` format)
//!
//! The legacy layout, as the reference imatrix tool writes and reads it:
//!
//! **Header:**
//! - `n_entries: i32` - Number of tensor entries
//!
//! **Per-entry data (repeated n_entries times):**
//! - `len: i32` - Length of tensor name string
//! - `name: [u8; len]` - Tensor name (not null-terminated)
//! - `ncall: i32` - Number of chunks processed
//! - `nval: i32` - Total number of values
//! - `values: [f32; nval]` - Activation squared sums
//!
//! **Footer:**
//! - `m_last_chunk: i32` - Final chunk count
//! - `dataset_len: i32` - Dataset filename length
//! - `dataset: [u8; dataset_len]` - Dataset filename
//!
//! # GGUF format
//!
//! Current imatrix tools write a GGUF file (`general.type = "imatrix"`) with,
//! per weight matrix `<name>`, an `F32` tensor `<name>.in_sum2` of the summed
//! squared input activations (`n_per_row` values, times the number of experts
//! for a mixture-of-experts matrix) and `<name>.counts` with the number of
//! tokens accumulated (one per expert); `imatrix.chunk_count` and
//! `imatrix.datasets` describe the run. [`IMatrix::load`] reads either format.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use pmetal_core::{PMetalError, Result};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Seek, Write};

/// Maximum allowed tensor name length (prevents DoS from malicious files).
pub const MAX_NAME_LENGTH: i32 = 4096;

/// Maximum allowed number of values per tensor (prevents OOM from malicious files).
/// 100 million elements = ~400MB per tensor, reasonable upper bound.
pub const MAX_VALUES_COUNT: i32 = 100_000_000;

/// Maximum allowed number of entries in an imatrix file.
pub const MAX_ENTRIES: i32 = 100_000;

/// Importance Matrix data.
#[derive(Debug, Clone, Default)]
pub struct IMatrix {
    /// Map of tensor name to importance data.
    /// The data represents the accumulated squared activations (Hessian approximation).
    pub data: HashMap<String, Vec<f32>>,
    /// Number of calibration calls per tensor.
    pub ncalls: HashMap<String, i32>,
    /// Dataset filename (from footer, if available).
    pub dataset_name: Option<String>,
    /// Last chunk count (from footer, if available).
    pub last_chunk: Option<i32>,
}

impl IMatrix {
    /// Create a new empty IMatrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load an importance matrix from a file, in either the GGUF format or the
    /// legacy `.dat` format (told apart by the file's first four bytes).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened
    /// - File is corrupted or has invalid format
    /// - Values exceed safety limits (DoS protection)
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(PMetalError::Io)?;
        let file_size = file.metadata().map_err(PMetalError::Io)?.len();
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        let is_gguf =
            reader.read_exact(&mut magic).is_ok() && u32::from_le_bytes(magic) == crate::GGUF_MAGIC;
        reader.rewind().map_err(PMetalError::Io)?;
        if is_gguf {
            drop(reader);
            return Self::load_gguf(path);
        }
        Self::load_from_reader(&mut reader, file_size)
    }

    /// Load an importance matrix from a GGUF file as written by current
    /// imatrix tools.
    ///
    /// Each matrix's `in_sum2` is divided by its `counts` (per expert for a
    /// stacked mixture-of-experts matrix) so the stored values are mean
    /// squared activations, with `ncall` 1.
    pub fn load_gguf(path: &std::path::Path) -> Result<Self> {
        use crate::{GgmlType, GgufModel, MetadataValue};

        let model = GgufModel::open(path)
            .map_err(|e| PMetalError::Serialization(format!("imatrix GGUF: {e}")))?;
        let to_f32 = |name: &str| -> Result<Vec<f32>> {
            let info = model.tensor_info(name).ok_or_else(|| {
                PMetalError::InvalidArgument(format!("imatrix GGUF has no tensor {name}"))
            })?;
            if info.dtype != GgmlType::F32 {
                return Err(PMetalError::InvalidArgument(format!(
                    "imatrix tensor {name} is {:?}, expected F32",
                    info.dtype
                )));
            }
            let bytes = model
                .tensor_data(name)
                .map_err(|e| PMetalError::Serialization(format!("imatrix GGUF: {e}")))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect())
        };

        let mut out = Self::new();
        let names: Vec<String> = model.tensor_names().map(String::from).collect();
        for tensor in &names {
            let Some(name) = tensor
                .strip_suffix(".in_sum2")
                .or_else(|| tensor.strip_suffix(".sums"))
            else {
                continue;
            };
            let sums = to_f32(tensor)?;
            let counts = to_f32(&format!("{name}.counts"))?;
            if counts.is_empty() || sums.len() % counts.len() != 0 {
                return Err(PMetalError::InvalidArgument(format!(
                    "imatrix tensor {tensor} has {} values for {} counts",
                    sums.len(),
                    counts.len()
                )));
            }
            let per_expert = sums.len() / counts.len();
            let mut values = Vec::with_capacity(sums.len());
            for (e, count) in counts.iter().enumerate() {
                let inv = if *count > 0.0 { 1.0 / count } else { 0.0 };
                values.extend(
                    sums[e * per_expert..(e + 1) * per_expert]
                        .iter()
                        .map(|v| v * inv),
                );
            }
            out.insert(name.to_string(), values, 1);
        }
        out.last_chunk = match model.get_metadata("imatrix.chunk_count") {
            Some(MetadataValue::Uint32(v)) => i32::try_from(*v).ok(),
            Some(MetadataValue::Int32(v)) => Some(*v),
            _ => None,
        };
        out.dataset_name = match model.get_metadata("imatrix.datasets") {
            Some(MetadataValue::Array(items)) => items.iter().find_map(|v| match v {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            }),
            _ => None,
        };
        Ok(out)
    }

    /// Load IMatrix from a reader.
    pub fn load_from_reader<R: Read + Seek>(reader: &mut R, file_size: u64) -> Result<Self> {
        let mut data = HashMap::new();
        let mut ncalls = HashMap::new();

        // Read number of entries (header)
        let n_entries = reader.read_i32::<LittleEndian>().map_err(PMetalError::Io)?;

        // Validate entry count
        if !(0..=MAX_ENTRIES).contains(&n_entries) {
            return Err(PMetalError::InvalidArgument(format!(
                "Invalid imatrix entry count: {} (max: {})",
                n_entries, MAX_ENTRIES
            )));
        }

        // Read each entry
        for entry_idx in 0..n_entries {
            // Read name length
            let name_len = reader.read_i32::<LittleEndian>().map_err(PMetalError::Io)?;

            // Validate name length
            if name_len <= 0 || name_len > MAX_NAME_LENGTH {
                return Err(PMetalError::InvalidArgument(format!(
                    "Invalid tensor name length at entry {}: {} (max: {})",
                    entry_idx, name_len, MAX_NAME_LENGTH
                )));
            }

            // Read name
            let mut name_bytes = vec![0u8; name_len as usize];
            reader
                .read_exact(&mut name_bytes)
                .map_err(PMetalError::Io)?;
            let name = String::from_utf8(name_bytes).map_err(|e| {
                PMetalError::Serialization(format!(
                    "Invalid UTF-8 in tensor name at entry {}: {}",
                    entry_idx, e
                ))
            })?;

            // Read ncall (number of calibration chunks processed)
            let ncall = reader.read_i32::<LittleEndian>().map_err(PMetalError::Io)?;

            // Read nval (number of values)
            let nval = reader.read_i32::<LittleEndian>().map_err(PMetalError::Io)?;

            // Validate value count
            if !(0..=MAX_VALUES_COUNT).contains(&nval) {
                return Err(PMetalError::InvalidArgument(format!(
                    "Invalid value count for tensor '{}': {} (max: {})",
                    name, nval, MAX_VALUES_COUNT
                )));
            }

            // Check we have enough bytes remaining (early bounds check)
            let values_size = (nval as u64) * 4; // f32 = 4 bytes
            let current_pos = reader.stream_position().map_err(PMetalError::Io)?;
            if current_pos + values_size > file_size {
                return Err(PMetalError::InvalidArgument(format!(
                    "Truncated file: tensor '{}' expects {} bytes but only {} remain",
                    name,
                    values_size,
                    file_size.saturating_sub(current_pos)
                )));
            }

            // Read values
            let mut values = vec![0.0f32; nval as usize];
            reader
                .read_f32_into::<LittleEndian>(&mut values)
                .map_err(PMetalError::Io)?;

            data.insert(name.clone(), values);
            ncalls.insert(name, ncall);
        }

        // Try to read footer (optional - may not exist in all versions)
        let mut last_chunk = None;
        let mut dataset_name = None;

        if let Ok(chunk) = reader.read_i32::<LittleEndian>() {
            last_chunk = Some(chunk);

            if let Ok(dataset_len) = reader.read_i32::<LittleEndian>() {
                if dataset_len > 0 && dataset_len <= MAX_NAME_LENGTH {
                    let mut dataset_bytes = vec![0u8; dataset_len as usize];
                    if reader.read_exact(&mut dataset_bytes).is_ok() {
                        dataset_name = String::from_utf8(dataset_bytes).ok();
                    }
                }
            }
        }

        Ok(Self {
            data,
            ncalls,
            dataset_name,
            last_chunk,
        })
    }

    /// Save IMatrix to a file (legacy `.dat` format), atomically.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let dir = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => std::path::Path::new("."),
        };
        let temp = tempfile::NamedTempFile::new_in(dir).map_err(PMetalError::Io)?;
        let mut writer = BufWriter::new(temp);
        self.save_to_writer(&mut writer)?;
        let temp = writer
            .into_inner()
            .map_err(|e| PMetalError::Io(e.into_error()))?;
        temp.persist(path).map_err(|e| PMetalError::Io(e.error))?;
        Ok(())
    }

    /// Save IMatrix to a writer.
    pub fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write number of entries
        writer
            .write_i32::<LittleEndian>(self.data.len() as i32)
            .map_err(PMetalError::Io)?;

        // Write each entry, in name order so the output is reproducible.
        let mut entries: Vec<_> = self.data.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, values) in entries {
            let name_bytes = name.as_bytes();

            // Write name length
            writer
                .write_i32::<LittleEndian>(name_bytes.len() as i32)
                .map_err(PMetalError::Io)?;

            // Write name
            writer.write_all(name_bytes).map_err(PMetalError::Io)?;

            // Write ncall
            let ncall = self.ncalls.get(name).copied().unwrap_or(1);
            writer
                .write_i32::<LittleEndian>(ncall)
                .map_err(PMetalError::Io)?;

            // Write nval
            writer
                .write_i32::<LittleEndian>(values.len() as i32)
                .map_err(PMetalError::Io)?;

            // Write values
            for &v in values {
                writer
                    .write_f32::<LittleEndian>(v)
                    .map_err(PMetalError::Io)?;
            }
        }

        // Write footer
        let last_chunk = self.last_chunk.unwrap_or(0);
        writer
            .write_i32::<LittleEndian>(last_chunk)
            .map_err(PMetalError::Io)?;

        let dataset = self.dataset_name.as_deref().unwrap_or("");
        let dataset_bytes = dataset.as_bytes();
        writer
            .write_i32::<LittleEndian>(dataset_bytes.len() as i32)
            .map_err(PMetalError::Io)?;
        writer.write_all(dataset_bytes).map_err(PMetalError::Io)?;

        writer.flush().map_err(PMetalError::Io)?;
        Ok(())
    }

    /// Add or update importance data for a tensor.
    pub fn insert(&mut self, name: String, values: Vec<f32>, ncall: i32) {
        self.ncalls.insert(name.clone(), ncall);
        self.data.insert(name, values);
    }

    /// Get the number of tensors in this imatrix.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if this imatrix is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The importance of each input channel of `name`: the mean squared
    /// activation, which is what the quantizers weight the error by
    /// ([`crate::quantize::quantize_tensor`]). The legacy format stores sums
    /// over `ncall` chunks, so they are divided by it; a stacked
    /// mixture-of-experts matrix has one run of channels per expert.
    pub fn importance(&self, name: &str) -> Option<Vec<f32>> {
        let values = self.data.get(name)?;
        let ncall = self.ncalls.get(name).copied().unwrap_or(1).max(1) as f32;
        Some(values.iter().map(|v| v / ncall).collect())
    }

    /// Get the total importance score for a tensor.
    pub fn total_importance(&self, name: &str) -> Option<f32> {
        self.data.get(name).map(|values| values.iter().sum())
    }

    /// Get the mean importance score for a tensor.
    pub fn mean_importance(&self, name: &str) -> Option<f32> {
        self.data.get(name).map(|values| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_imatrix() -> IMatrix {
        let mut imatrix = IMatrix::new();
        imatrix.insert("layer.0.weight".to_string(), vec![1.0, 2.0, 3.0, 4.0], 10);
        imatrix.insert("layer.1.weight".to_string(), vec![5.0, 6.0, 7.0], 20);
        imatrix.dataset_name = Some("test_dataset.txt".to_string());
        imatrix.last_chunk = Some(100);
        imatrix
    }

    #[test]
    fn test_save_load_roundtrip() {
        let original = create_test_imatrix();

        // Save to buffer
        let mut buffer = Vec::new();
        original.save_to_writer(&mut buffer).unwrap();

        // Load from buffer
        let mut cursor = Cursor::new(&buffer);
        let loaded = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64).unwrap();

        // Verify data
        assert_eq!(original.data.len(), loaded.data.len());
        for (name, values) in &original.data {
            assert_eq!(values, loaded.data.get(name).unwrap());
        }
        assert_eq!(original.ncalls, loaded.ncalls);
        assert_eq!(original.dataset_name, loaded.dataset_name);
        assert_eq!(original.last_chunk, loaded.last_chunk);
    }

    #[test]
    fn test_bounds_validation_name_length() {
        // Create a buffer with invalid name length
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(1).unwrap(); // 1 entry
        buffer.write_i32::<LittleEndian>(-1).unwrap(); // Invalid negative length

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid tensor name length")
        );
    }

    #[test]
    fn test_bounds_validation_name_too_long() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(1).unwrap(); // 1 entry
        buffer
            .write_i32::<LittleEndian>(MAX_NAME_LENGTH + 1)
            .unwrap(); // Too long

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
    }

    #[test]
    fn test_bounds_validation_value_count() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(1).unwrap(); // 1 entry
        buffer.write_i32::<LittleEndian>(4).unwrap(); // name length
        buffer.write_all(b"test").unwrap(); // name
        buffer.write_i32::<LittleEndian>(1).unwrap(); // ncall
        buffer.write_i32::<LittleEndian>(-1).unwrap(); // Invalid negative nval

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid value count")
        );
    }

    #[test]
    fn test_bounds_validation_value_count_too_large() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(1).unwrap(); // 1 entry
        buffer.write_i32::<LittleEndian>(4).unwrap(); // name length
        buffer.write_all(b"test").unwrap(); // name
        buffer.write_i32::<LittleEndian>(1).unwrap(); // ncall
        buffer
            .write_i32::<LittleEndian>(MAX_VALUES_COUNT + 1)
            .unwrap(); // Too large

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_file_detection() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(1).unwrap(); // 1 entry
        buffer.write_i32::<LittleEndian>(4).unwrap(); // name length
        buffer.write_all(b"test").unwrap(); // name
        buffer.write_i32::<LittleEndian>(1).unwrap(); // ncall
        buffer.write_i32::<LittleEndian>(1000).unwrap(); // nval = 1000 (but no values follow)

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Truncated file"));
    }

    #[test]
    fn test_invalid_utf8_name() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(1).unwrap(); // 1 entry
        buffer.write_i32::<LittleEndian>(4).unwrap(); // name length
        buffer.write_all(&[0xFF, 0xFE, 0x00, 0x01]).unwrap(); // Invalid UTF-8

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid UTF-8"));
    }

    #[test]
    fn test_empty_imatrix() {
        let imatrix = IMatrix::new();
        assert!(imatrix.is_empty());
        assert_eq!(imatrix.len(), 0);
    }

    #[test]
    fn test_importance_calculations() {
        let imatrix = create_test_imatrix();

        // Total importance
        assert_eq!(imatrix.total_importance("layer.0.weight"), Some(10.0));
        assert_eq!(imatrix.total_importance("layer.1.weight"), Some(18.0));
        assert_eq!(imatrix.total_importance("nonexistent"), None);

        // Mean importance
        assert_eq!(imatrix.mean_importance("layer.0.weight"), Some(2.5));
        assert_eq!(imatrix.mean_importance("layer.1.weight"), Some(6.0));
    }

    #[test]
    fn test_entry_count_validation() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(-1).unwrap(); // Invalid negative entry count

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid imatrix entry count")
        );
    }

    #[test]
    fn test_entry_count_too_large() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(MAX_ENTRIES + 1).unwrap(); // Too many entries

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
    }

    #[test]
    fn importance_divides_legacy_sums_by_the_call_count() {
        let imatrix = create_test_imatrix();
        assert_eq!(
            imatrix.importance("layer.0.weight"),
            Some(vec![0.1, 0.2, 0.3, 0.4])
        );
        assert!(imatrix.importance("nonexistent").is_none());
    }

    #[test]
    fn saving_is_deterministic_and_round_trips_through_a_file() {
        let mut imatrix = IMatrix::new();
        for i in 0..20 {
            imatrix.insert(format!("blk.{i}.w"), vec![i as f32; 3], 4);
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        imatrix.save_to_writer(&mut a).unwrap();
        imatrix.clone().save_to_writer(&mut b).unwrap();
        assert_eq!(a, b);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.dat");
        imatrix.save(&path).unwrap();
        let loaded = IMatrix::load(&path).unwrap();
        assert_eq!(loaded.data, imatrix.data);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn gguf_imatrix_is_loaded_and_normalised_per_expert() {
        use crate::{GgmlType, GgufBuilder, MetadataValue};
        let mut b = GgufBuilder::with_model("imatrix", "m");
        b.add_metadata("general.type", MetadataValue::String("imatrix".into()));
        b.add_u32("imatrix.chunk_count", 7);
        b.add_string_array("imatrix.datasets", vec!["calib.txt".into()]);
        // A dense matrix: 4 channels, 8 tokens.
        b.add_f32_tensor(
            "blk.0.attn_q.weight.in_sum2",
            vec![1, 4],
            vec![8.0, 16.0, 0.0, 4.0],
        );
        b.add_f32_tensor("blk.0.attn_q.weight.counts", vec![1, 1], vec![8.0]);
        // A stacked-experts matrix: 2 experts x 3 channels, 2 and 4 tokens.
        b.add_f32_tensor(
            "blk.0.ffn_up_exps.weight.in_sum2",
            vec![2, 3],
            vec![2.0, 4.0, 6.0, 4.0, 8.0, 12.0],
        );
        b.add_f32_tensor(
            "blk.0.ffn_up_exps.weight.counts",
            vec![1, 2],
            vec![2.0, 4.0],
        );
        b.add_raw_tensor("other", vec![1], GgmlType::F32, vec![0; 4]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("imatrix.gguf");
        b.write_file(&path).unwrap();

        let m = IMatrix::load(&path).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(
            m.importance("blk.0.attn_q.weight"),
            Some(vec![1.0, 2.0, 0.0, 0.5])
        );
        assert_eq!(
            m.importance("blk.0.ffn_up_exps.weight"),
            Some(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        );
        assert_eq!(m.last_chunk, Some(7));
        assert_eq!(m.dataset_name.as_deref(), Some("calib.txt"));
    }
}
