//! GGUF file reader for loading quantized models.
//!
//! Based on the GGUF v3 specification and provides support for reading
//! GGUF files produced by llama.cpp, Ollama, and other GGML-based tools.
//!
//! # Supported Versions
//!
//! - GGUF v1 (legacy)
//! - GGUF v2
//! - GGUF v3 (current)
//!
//! # Example
//!
//! ```ignore
//! use pmetal_gguf::reader::GgufContent;
//!
//! let content = GgufContent::from_file("model.gguf")?;
//!
//! // Read metadata
//! if let Some(arch) = content.get_metadata("general.architecture") {
//!     println!("Architecture: {:?}", arch);
//! }
//!
//! // Read tensor data
//! let mut file = std::fs::File::open("model.gguf")?;
//! let data = content.read_tensor_data(&mut file, "blk.0.attn_q.weight")?;
//! ```

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::{GgmlType, MetadataValue, TensorInfo, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};

/// GGUF file version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufVersion {
    /// Version 1 (legacy format).
    V1,
    /// Version 2.
    V2,
    /// Version 3 (current).
    V3,
}

impl std::fmt::Display for GgufVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V1 => write!(f, "v1"),
            Self::V2 => write!(f, "v2"),
            Self::V3 => write!(f, "v3"),
        }
    }
}

/// Maximum allowed string length in GGUF metadata (10 MB).
/// This prevents memory exhaustion attacks from malicious GGUF files.
pub const MAX_STRING_LENGTH: usize = 10 * 1024 * 1024;

/// Maximum allowed array length in GGUF metadata (1 million elements).
/// This prevents memory exhaustion attacks from malicious GGUF files.
pub const MAX_ARRAY_LENGTH: usize = 1_000_000;

/// Maximum allowed tensor count (100,000 tensors).
/// Modern LLMs typically have < 10,000 tensors.
pub const MAX_TENSOR_COUNT: usize = 100_000;

/// Maximum allowed metadata count (100,000 entries).
pub const MAX_METADATA_COUNT: usize = 100_000;

/// Error type for GGUF reading.
#[derive(Debug, thiserror::Error)]
pub enum GgufReadError {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid GGUF magic number.
    #[error("Invalid GGUF magic: expected 0x46554747, got 0x{0:08X}")]
    InvalidMagic(u32),
    /// Unsupported GGUF version.
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    /// Invalid metadata value type.
    #[error("Invalid metadata value type: {0}")]
    InvalidValueType(u32),
    /// Invalid tensor dtype.
    #[error("Invalid tensor dtype: {0}")]
    InvalidDtype(u32),
    /// Tensor not found.
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    /// Invalid UTF-8 in string.
    #[error("Invalid UTF-8 in string")]
    InvalidUtf8,
    /// String too large.
    #[error("String too large: {size} bytes exceeds maximum of {max} bytes")]
    StringTooLarge {
        /// Actual size in bytes.
        size: usize,
        /// Maximum allowed size.
        max: usize,
    },
    /// Array too large.
    #[error("Array too large: {len} elements exceeds maximum of {max} elements")]
    ArrayTooLarge {
        /// Actual length.
        len: usize,
        /// Maximum allowed length.
        max: usize,
    },
    /// Too many tensors.
    #[error("Too many tensors: {count} exceeds maximum of {max}")]
    TooManyTensors {
        /// Actual count.
        count: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// Too many metadata entries.
    #[error("Too many metadata entries: {count} exceeds maximum of {max}")]
    TooManyMetadata {
        /// Actual count.
        count: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// Integer overflow in size calculation.
    #[error("Integer overflow in tensor size calculation")]
    IntegerOverflow,
    /// Tensor size exceeds system limits.
    #[error("Tensor too large: {n_elements} elements of type {dtype:?}")]
    TensorTooLarge {
        /// Number of elements.
        n_elements: u64,
        /// Data type.
        dtype: String,
    },
}

/// GGUF file content (header + tensor metadata).
///
/// This struct contains all the metadata and tensor information from a GGUF file,
/// but not the actual tensor data. Use `read_tensor_data` to lazily load tensor data.
#[derive(Debug)]
pub struct GgufContent {
    /// GGUF version.
    pub version: GgufVersion,
    /// Metadata key-value pairs.
    pub metadata: HashMap<String, MetadataValue>,
    /// Tensor information.
    pub tensor_infos: HashMap<String, TensorInfo>,
    /// Offset to tensor data section.
    pub tensor_data_offset: u64,
}

impl GgufContent {
    /// Read GGUF content from a reader.
    ///
    /// This reads the header, metadata, and tensor info but not the actual tensor data.
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self, GgufReadError> {
        // Read and validate magic
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC && magic != 0x47475546 {
            // Also accept big-endian "GGUF"
            return Err(GgufReadError::InvalidMagic(magic));
        }

        // Read version
        let version_num = reader.read_u32::<LittleEndian>()?;
        let version = match version_num {
            1 => GgufVersion::V1,
            2 => GgufVersion::V2,
            3 => GgufVersion::V3,
            v => return Err(GgufReadError::UnsupportedVersion(v)),
        };

        // Read counts (V1 uses u32, V2/V3 use u64) with validation
        let (tensor_count, metadata_kv_count) = match version {
            GgufVersion::V1 => {
                let tc = reader.read_u32::<LittleEndian>()? as usize;
                let mc = reader.read_u32::<LittleEndian>()? as usize;
                (tc, mc)
            }
            GgufVersion::V2 | GgufVersion::V3 => {
                let tc64 = reader.read_u64::<LittleEndian>()?;
                let mc64 = reader.read_u64::<LittleEndian>()?;
                // Validate counts before cast to prevent overflow on 32-bit systems
                if tc64 > MAX_TENSOR_COUNT as u64 {
                    return Err(GgufReadError::TooManyTensors {
                        count: tc64 as usize,
                        max: MAX_TENSOR_COUNT,
                    });
                }
                if mc64 > MAX_METADATA_COUNT as u64 {
                    return Err(GgufReadError::TooManyMetadata {
                        count: mc64 as usize,
                        max: MAX_METADATA_COUNT,
                    });
                }
                (tc64 as usize, mc64 as usize)
            }
        };

        // Additional validation for V1 (already cast, but check limits)
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(GgufReadError::TooManyTensors {
                count: tensor_count,
                max: MAX_TENSOR_COUNT,
            });
        }
        if metadata_kv_count > MAX_METADATA_COUNT {
            return Err(GgufReadError::TooManyMetadata {
                count: metadata_kv_count,
                max: MAX_METADATA_COUNT,
            });
        }

        // Read metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_string(reader, &version)?;
            let value_type_num = reader.read_u32::<LittleEndian>()?;
            let value = read_value(reader, value_type_num, &version)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensor_infos = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(reader, &version)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;

            // Read dimensions (stored in reverse order in GGUF)
            let mut dims: Vec<u64> = match version {
                GgufVersion::V1 => (0..n_dims)
                    .map(|_| reader.read_u32::<LittleEndian>().map(|d| d as u64))
                    .collect::<Result<Vec<_>, _>>()?,
                GgufVersion::V2 | GgufVersion::V3 => (0..n_dims)
                    .map(|_| reader.read_u64::<LittleEndian>())
                    .collect::<Result<Vec<_>, _>>()?,
            };
            dims.reverse(); // GGUF stores column-major, we want row-major

            let dtype_num = reader.read_u32::<LittleEndian>()?;
            let dtype = GgmlType::try_from(dtype_num)
                .map_err(|_| GgufReadError::InvalidDtype(dtype_num))?;

            let offset = reader.read_u64::<LittleEndian>()?;

            tensor_infos.insert(
                name.clone(),
                TensorInfo {
                    name,
                    n_dimensions: n_dims,
                    dimensions: dims,
                    dtype,
                    offset,
                },
            );
        }

        // Calculate tensor data offset with alignment
        let position = reader.stream_position()?;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                MetadataValue::Uint32(a) => Some(*a as u64),
                MetadataValue::Uint64(a) => Some(*a),
                _ => None,
            })
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u64);

        let tensor_data_offset = ((position + alignment - 1) / alignment) * alignment;

        Ok(Self {
            version,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    /// Read GGUF content from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, GgufReadError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::read(&mut reader)
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Get tensor info by name.
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.get(name)
    }

    /// Get the model architecture from metadata.
    pub fn architecture(&self) -> Option<&str> {
        match self.metadata.get("general.architecture")? {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_infos.keys().map(String::as_str)
    }

    /// Get the total number of tensors.
    pub fn num_tensors(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Read raw tensor data from file.
    ///
    /// This seeks to the tensor's position in the file and reads the raw bytes.
    pub fn read_tensor_data<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<Vec<u8>, GgufReadError> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| GgufReadError::TensorNotFound(name.to_string()))?;

        // Use checked arithmetic to prevent integer overflow attacks
        let byte_size = info.byte_size_checked().map_err(|e| match e {
            crate::TensorSizeError::ElementCountOverflow
            | crate::TensorSizeError::ByteSizeOverflow => GgufReadError::IntegerOverflow,
            crate::TensorSizeError::ElementCountTooLarge(n) => GgufReadError::TensorTooLarge {
                n_elements: n,
                dtype: format!("{:?}", info.dtype),
            },
        })?;

        let mut data = vec![0u8; byte_size];

        reader.seek(SeekFrom::Start(self.tensor_data_offset + info.offset))?;
        reader.read_exact(&mut data)?;

        Ok(data)
    }

    /// Read all tensors from file as raw bytes.
    ///
    /// Returns a HashMap mapping tensor names to their raw byte data.
    pub fn read_all_tensors<R: Read + Seek>(
        &self,
        reader: &mut R,
    ) -> Result<HashMap<String, Vec<u8>>, GgufReadError> {
        let mut tensors = HashMap::with_capacity(self.tensor_infos.len());

        for name in self.tensor_infos.keys() {
            let data = self.read_tensor_data(reader, name)?;
            tensors.insert(name.clone(), data);
        }

        Ok(tensors)
    }
}

/// Read a GGUF string with size validation.
fn read_string<R: Read>(reader: &mut R, version: &GgufVersion) -> Result<String, GgufReadError> {
    let len = match version {
        GgufVersion::V1 => reader.read_u32::<LittleEndian>()? as usize,
        GgufVersion::V2 | GgufVersion::V3 => {
            let len64 = reader.read_u64::<LittleEndian>()?;
            // Validate length before cast to prevent overflow on 32-bit systems
            if len64 > MAX_STRING_LENGTH as u64 {
                return Err(GgufReadError::StringTooLarge {
                    size: len64 as usize,
                    max: MAX_STRING_LENGTH,
                });
            }
            len64 as usize
        }
    };

    // Validate string length to prevent memory exhaustion
    if len > MAX_STRING_LENGTH {
        return Err(GgufReadError::StringTooLarge {
            size: len,
            max: MAX_STRING_LENGTH,
        });
    }

    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;

    // Remove trailing nulls (GGUF spec says non-null terminated but some files have them)
    while bytes.last() == Some(&0) {
        bytes.pop();
    }

    String::from_utf8(bytes).map_err(|_| GgufReadError::InvalidUtf8)
}

/// Read a metadata value with size validation.
fn read_value<R: Read>(
    reader: &mut R,
    value_type: u32,
    version: &GgufVersion,
) -> Result<MetadataValue, GgufReadError> {
    let value = match value_type {
        0 => MetadataValue::Uint8(reader.read_u8()?),
        1 => MetadataValue::Int8(reader.read_i8()?),
        2 => MetadataValue::Uint16(reader.read_u16::<LittleEndian>()?),
        3 => MetadataValue::Int16(reader.read_i16::<LittleEndian>()?),
        4 => MetadataValue::Uint32(reader.read_u32::<LittleEndian>()?),
        5 => MetadataValue::Int32(reader.read_i32::<LittleEndian>()?),
        6 => MetadataValue::Float32(reader.read_f32::<LittleEndian>()?),
        7 => MetadataValue::Bool(reader.read_u8()? != 0),
        8 => MetadataValue::String(read_string(reader, version)?),
        9 => {
            // Array: type + length + elements
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let len = match version {
                GgufVersion::V1 => reader.read_u32::<LittleEndian>()? as usize,
                GgufVersion::V2 | GgufVersion::V3 => {
                    let len64 = reader.read_u64::<LittleEndian>()?;
                    // Validate length before cast to prevent overflow on 32-bit systems
                    if len64 > MAX_ARRAY_LENGTH as u64 {
                        return Err(GgufReadError::ArrayTooLarge {
                            len: len64 as usize,
                            max: MAX_ARRAY_LENGTH,
                        });
                    }
                    len64 as usize
                }
            };

            // Validate array length to prevent memory exhaustion
            if len > MAX_ARRAY_LENGTH {
                return Err(GgufReadError::ArrayTooLarge {
                    len,
                    max: MAX_ARRAY_LENGTH,
                });
            }

            let elements: Vec<MetadataValue> = (0..len)
                .map(|_| read_value(reader, elem_type, version))
                .collect::<Result<_, _>>()?;
            MetadataValue::Array(elements)
        }
        10 => MetadataValue::Uint64(reader.read_u64::<LittleEndian>()?),
        11 => MetadataValue::Int64(reader.read_i64::<LittleEndian>()?),
        12 => MetadataValue::Float64(reader.read_f64::<LittleEndian>()?),
        t => return Err(GgufReadError::InvalidValueType(t)),
    };
    Ok(value)
}

impl TryFrom<u32> for GgmlType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            19 => Ok(Self::Iq1S),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::Iq1M),
            30 => Ok(Self::Bf16),
            34 => Ok(Self::Tq1_0),
            35 => Ok(Self::Tq2_0),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_conversion() {
        assert_eq!(GgmlType::try_from(0), Ok(GgmlType::F32));
        assert_eq!(GgmlType::try_from(1), Ok(GgmlType::F16));
        assert_eq!(GgmlType::try_from(2), Ok(GgmlType::Q4_0));
        assert_eq!(GgmlType::try_from(8), Ok(GgmlType::Q8_0));
        assert_eq!(GgmlType::try_from(30), Ok(GgmlType::Bf16));
        assert!(GgmlType::try_from(255).is_err());
    }

    #[test]
    fn test_version_display() {
        assert_eq!(GgufVersion::V1.to_string(), "v1");
        assert_eq!(GgufVersion::V2.to_string(), "v2");
        assert_eq!(GgufVersion::V3.to_string(), "v3");
    }
}
