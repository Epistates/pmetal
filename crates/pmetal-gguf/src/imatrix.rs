//! Importance Matrix (IMatrix) implementation.
//!
//! IMatrix stores the importance of each weight matrix in the model,
//! usually calculated as the accumulated squared activations of the inputs
//! to that weight matrix during a calibration pass.
//!
//! # Binary Format (llama.cpp legacy .dat format)
//!
//! Based on llama.cpp `save_imatrix_legacy()` and `load_imatrix_legacy()`:
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
//! # References
//!
//! - [llama.cpp imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/imatrix.cpp)

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

    /// Load IMatrix from a file (llama.cpp legacy .dat format).
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

        Self::load_from_reader(&mut reader, file_size)
    }

    /// Load IMatrix from a reader.
    pub fn load_from_reader<R: Read + Seek>(reader: &mut R, file_size: u64) -> Result<Self> {
        let mut data = HashMap::new();
        let mut ncalls = HashMap::new();

        // Read number of entries (header)
        let n_entries = reader
            .read_i32::<LittleEndian>()
            .map_err(PMetalError::Io)?;

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
            let name_len = reader
                .read_i32::<LittleEndian>()
                .map_err(PMetalError::Io)?;

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
            let ncall = reader
                .read_i32::<LittleEndian>()
                .map_err(PMetalError::Io)?;

            // Read nval (number of values)
            let nval = reader
                .read_i32::<LittleEndian>()
                .map_err(PMetalError::Io)?;

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

    /// Save IMatrix to a file (llama.cpp legacy .dat format).
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let file = std::fs::File::create(path).map_err(PMetalError::Io)?;
        let mut writer = BufWriter::new(file);

        self.save_to_writer(&mut writer)
    }

    /// Save IMatrix to a writer.
    pub fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write number of entries
        writer
            .write_i32::<LittleEndian>(self.data.len() as i32)
            .map_err(PMetalError::Io)?;

        // Write each entry
        for (name, values) in &self.data {
            let name_bytes = name.as_bytes();

            // Write name length
            writer
                .write_i32::<LittleEndian>(name_bytes.len() as i32)
                .map_err(PMetalError::Io)?;

            // Write name
            writer
                .write_all(name_bytes)
                .map_err(PMetalError::Io)?;

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
        writer
            .write_all(dataset_bytes)
            .map_err(PMetalError::Io)?;

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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid tensor name length"));
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid value count"));
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid imatrix entry count"));
    }

    #[test]
    fn test_entry_count_too_large() {
        let mut buffer = Vec::new();
        buffer.write_i32::<LittleEndian>(MAX_ENTRIES + 1).unwrap(); // Too many entries

        let mut cursor = Cursor::new(&buffer);
        let result = IMatrix::load_from_reader(&mut cursor, buffer.len() as u64);
        assert!(result.is_err());
    }
}
