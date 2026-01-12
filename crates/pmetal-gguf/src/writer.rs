//! GGUF file writer.

use crate::{
    GgmlType, MetadataValue, MetadataValueType, TensorInfo, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC,
    GGUF_VERSION,
};
use pmetal_core::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::BTreeMap;
use std::io::{Seek, Write};

/// Builder for creating GGUF files.
#[derive(Debug)]
pub struct GgufBuilder {
    /// Metadata key-value pairs.
    metadata: BTreeMap<String, MetadataValue>,
    /// Tensor information and data.
    tensors: Vec<(TensorInfo, Vec<u8>)>,
    /// Alignment for tensor data.
    alignment: u32,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    /// Create a new GGUF builder.
    pub fn new() -> Self {
        let mut builder = Self {
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
        };
        // Set default alignment metadata
        builder.add_metadata(
            crate::keys::GENERAL_ALIGNMENT,
            MetadataValue::Uint32(GGUF_DEFAULT_ALIGNMENT),
        );
        // Set default quantization version (2 is current standard)
        builder.add_metadata(
            crate::keys::GENERAL_QUANTIZATION_VERSION,
            MetadataValue::Uint32(2),
        );
        builder
    }

    /// Create a builder with basic model info.
    pub fn with_model(architecture: &str, name: &str) -> Self {
        let mut builder = Self::new();
        builder.add_metadata(
            crate::keys::GENERAL_ARCHITECTURE,
            MetadataValue::String(architecture.to_string()),
        );
        builder.add_metadata(
            crate::keys::GENERAL_NAME,
            MetadataValue::String(name.to_string()),
        );
        builder
    }

    /// Set the alignment for tensor data.
    pub fn alignment(mut self, alignment: u32) -> Self {
        self.alignment = alignment;
        self.metadata.insert(
            crate::keys::GENERAL_ALIGNMENT.to_string(),
            MetadataValue::Uint32(alignment),
        );
        self
    }

    /// Add a metadata key-value pair.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: MetadataValue) -> &mut Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Add a string metadata value.
    pub fn add_string(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.add_metadata(key, MetadataValue::String(value.into()))
    }

    /// Add a u32 metadata value.
    pub fn add_u32(&mut self, key: impl Into<String>, value: u32) -> &mut Self {
        self.add_metadata(key, MetadataValue::Uint32(value))
    }

    /// Add a u64 metadata value.
    pub fn add_u64(&mut self, key: impl Into<String>, value: u64) -> &mut Self {
        self.add_metadata(key, MetadataValue::Uint64(value))
    }

    /// Add a f32 metadata value.
    pub fn add_f32(&mut self, key: impl Into<String>, value: f32) -> &mut Self {
        self.add_metadata(key, MetadataValue::Float32(value))
    }

    /// Add a bool metadata value.
    pub fn add_bool(&mut self, key: impl Into<String>, value: bool) -> &mut Self {
        self.add_metadata(key, MetadataValue::Bool(value))
    }

    /// Add an array of strings metadata value.
    pub fn add_string_array(&mut self, key: impl Into<String>, values: Vec<String>) -> &mut Self {
        let arr: Vec<MetadataValue> = values.into_iter().map(MetadataValue::String).collect();
        self.add_metadata(key, MetadataValue::Array(arr))
    }

    /// Add an array of f32 metadata value.
    pub fn add_f32_array(&mut self, key: impl Into<String>, values: Vec<f32>) -> &mut Self {
        let arr: Vec<MetadataValue> = values.into_iter().map(MetadataValue::Float32).collect();
        self.add_metadata(key, MetadataValue::Array(arr))
    }

    /// Add an array of i32 metadata value.
    pub fn add_i32_array(&mut self, key: impl Into<String>, values: Vec<i32>) -> &mut Self {
        let arr: Vec<MetadataValue> = values.into_iter().map(MetadataValue::Int32).collect();
        self.add_metadata(key, MetadataValue::Array(arr))
    }

    /// Add a tensor with f32 data.
    pub fn add_f32_tensor(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<u64>,
        data: Vec<f32>,
    ) -> &mut Self {
        let info = TensorInfo::new(name, dimensions, GgmlType::F32);
        // Convert f32 to bytes (little-endian)
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors.push((info, bytes));
        self
    }

    /// Add a tensor with f16 data.
    pub fn add_f16_tensor(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<u64>,
        data: Vec<half::f16>,
    ) -> &mut Self {
        let info = TensorInfo::new(name, dimensions, GgmlType::F16);
        // Convert f16 to bytes (little-endian)
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.tensors.push((info, bytes));
        self
    }

    /// Add a tensor with bf16 data.
    pub fn add_bf16_tensor(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<u64>,
        data: Vec<half::bf16>,
    ) -> &mut Self {
        let info = TensorInfo::new(name, dimensions, GgmlType::Bf16);
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.tensors.push((info, bytes));
        self
    }

    /// Add a tensor with raw bytes (pre-quantized or custom format).
    pub fn add_raw_tensor(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<u64>,
        dtype: GgmlType,
        data: Vec<u8>,
    ) -> &mut Self {
        let info = TensorInfo::new(name, dimensions, dtype);
        self.tensors.push((info, data));
        self
    }

    /// Build and write the GGUF file to the given writer.
    pub fn write<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        // Write header
        self.write_header(writer)?;

        // Write metadata
        for (key, value) in &self.metadata {
            self.write_string(writer, key)?;
            self.write_metadata_value(writer, value)?;
        }

        // Calculate tensor data offsets
        let tensor_infos = self.calculate_tensor_offsets();

        // Write tensor infos
        for info in &tensor_infos {
            self.write_tensor_info(writer, info)?;
        }

        // Pad to alignment before tensor data
        let current_pos = writer.stream_position().map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
        })?;
        let padding = align_offset(current_pos, self.alignment as u64) - current_pos;
        for _ in 0..padding {
            writer.write_u8(0).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        }

        // Write tensor data
        for (i, (_, data)) in self.tensors.iter().enumerate() {
            writer.write_all(data).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;

            // Pad to alignment (except for last tensor)
            if i < self.tensors.len() - 1 {
                let current_pos = writer.stream_position().map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
                })?;
                let padding = align_offset(current_pos, self.alignment as u64) - current_pos;
                for _ in 0..padding {
                    writer.write_u8(0).map_err(|e| {
                        pmetal_core::PMetalError::Io(std::io::Error::new(
                            e.kind(),
                            e.to_string(),
                        ))
                    })?;
                }
            }
        }

        Ok(())
    }

    /// Build to a byte vector.
    pub fn build_to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        self.write(&mut buffer)?;
        Ok(buffer.into_inner())
    }

    /// Write the GGUF header.
    fn write_header<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Magic
        writer.write_u32::<LittleEndian>(GGUF_MAGIC).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
        })?;
        // Version
        writer
            .write_u32::<LittleEndian>(GGUF_VERSION)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        // Tensor count
        writer
            .write_u64::<LittleEndian>(self.tensors.len() as u64)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        // Metadata KV count
        writer
            .write_u64::<LittleEndian>(self.metadata.len() as u64)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        Ok(())
    }

    /// Write a GGUF string (length-prefixed).
    fn write_string<W: Write>(&self, writer: &mut W, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        writer
            .write_u64::<LittleEndian>(bytes.len() as u64)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        writer.write_all(bytes).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
        })?;
        Ok(())
    }

    /// Write a metadata value.
    fn write_metadata_value<W: Write>(&self, writer: &mut W, value: &MetadataValue) -> Result<()> {
        // Write type
        writer
            .write_u32::<LittleEndian>(value.value_type() as u32)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;

        // Write value
        match value {
            MetadataValue::Uint8(v) => writer.write_u8(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int8(v) => writer.write_i8(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Uint16(v) => writer.write_u16::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int16(v) => writer.write_i16::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Uint32(v) => writer.write_u32::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int32(v) => writer.write_i32::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Float32(v) => writer.write_f32::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Bool(v) => writer.write_u8(if *v { 1 } else { 0 }).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::String(s) => {
                self.write_string(writer, s)?;
            }
            MetadataValue::Uint64(v) => writer.write_u64::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int64(v) => writer.write_i64::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Float64(v) => writer.write_f64::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Array(arr) => {
                // Write array element type
                let elem_type = arr
                    .first()
                    .map(|v| v.value_type())
                    .unwrap_or(MetadataValueType::Uint8);
                writer
                    .write_u32::<LittleEndian>(elem_type as u32)
                    .map_err(|e| {
                        pmetal_core::PMetalError::Io(std::io::Error::new(
                            e.kind(),
                            e.to_string(),
                        ))
                    })?;
                // Write array length
                writer
                    .write_u64::<LittleEndian>(arr.len() as u64)
                    .map_err(|e| {
                        pmetal_core::PMetalError::Io(std::io::Error::new(
                            e.kind(),
                            e.to_string(),
                        ))
                    })?;
                // Write array elements (without type prefix)
                for elem in arr {
                    self.write_metadata_value_data(writer, elem)?;
                }
            }
        }
        Ok(())
    }

    /// Write just the data portion of a metadata value (no type prefix).
    fn write_metadata_value_data<W: Write>(
        &self,
        writer: &mut W,
        value: &MetadataValue,
    ) -> Result<()> {
        match value {
            MetadataValue::Uint8(v) => writer.write_u8(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int8(v) => writer.write_i8(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Uint16(v) => writer.write_u16::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int16(v) => writer.write_i16::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Uint32(v) => writer.write_u32::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int32(v) => writer.write_i32::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Float32(v) => writer.write_f32::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Bool(v) => writer.write_u8(if *v { 1 } else { 0 }).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::String(s) => {
                self.write_string(writer, s)?;
            }
            MetadataValue::Uint64(v) => writer.write_u64::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Int64(v) => writer.write_i64::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Float64(v) => writer.write_f64::<LittleEndian>(*v).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?,
            MetadataValue::Array(_) => {
                // Nested arrays - write full value with type
                self.write_metadata_value(writer, value)?;
            }
        }
        Ok(())
    }

    /// Write tensor info.
    fn write_tensor_info<W: Write>(&self, writer: &mut W, info: &TensorInfo) -> Result<()> {
        // Name
        self.write_string(writer, &info.name)?;
        // Number of dimensions
        writer
            .write_u32::<LittleEndian>(info.n_dimensions)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        // Dimensions
        for dim in &info.dimensions {
            writer.write_u64::<LittleEndian>(*dim).map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        }
        // Type
        writer
            .write_u32::<LittleEndian>(info.dtype as u32)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        // Offset
        writer
            .write_u64::<LittleEndian>(info.offset)
            .map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
            })?;
        Ok(())
    }

    /// Calculate tensor offsets.
    fn calculate_tensor_offsets(&self) -> Vec<TensorInfo> {
        let mut infos = Vec::with_capacity(self.tensors.len());
        let mut offset: u64 = 0;

        for (info, data) in &self.tensors {
            let mut new_info = info.clone();
            new_info.offset = offset;
            infos.push(new_info);

            // Next offset is current + data size, aligned
            offset += data.len() as u64;
            offset = align_offset(offset, self.alignment as u64);
        }

        infos
    }
}

/// Calculate aligned offset.
fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset + (alignment - (offset % alignment)) % alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_header() {
        let builder = GgufBuilder::with_model("llama", "test-model");
        let bytes = builder.build_to_bytes().unwrap();

        // Check magic number
        assert_eq!(&bytes[0..4], &GGUF_MAGIC.to_le_bytes());
        // Check version
        assert_eq!(&bytes[4..8], &GGUF_VERSION.to_le_bytes());
    }

    #[test]
    fn test_gguf_metadata() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_u32("test.value", 42);
        builder.add_string("test.string", "hello");

        let bytes = builder.build_to_bytes().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_gguf_tensor() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_f32_tensor("test.weight", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let bytes = builder.build_to_bytes().unwrap();
        assert!(!bytes.is_empty());

        // Verify header shows 1 tensor
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tensor_count, 1);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_string_array() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_string_array(
            "tokenizer.ggml.tokens",
            vec!["hello".to_string(), "world".to_string()],
        );

        let bytes = builder.build_to_bytes().unwrap();
        assert!(!bytes.is_empty());
    }
}
