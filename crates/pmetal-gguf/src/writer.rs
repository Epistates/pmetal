//! GGUF file writer.

use crate::{
    GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GgmlType, MetadataValue, MetadataValueType,
    TensorInfo,
};
use byteorder::{LittleEndian, WriteBytesExt};
use pmetal_core::{PMetalError, Result};
use std::collections::{BTreeMap, HashSet};
use std::io::{Seek, Write};

use crate::reader::{MAX_ARRAY_LENGTH, MAX_STRING_LENGTH};

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
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
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
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
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
        self.validate()?;

        // Write header
        self.write_header(writer)?;

        // Write metadata
        for (key, value) in &self.metadata {
            self.write_string(writer, key)?;
            self.write_metadata_value(writer, value)?;
        }

        // Calculate tensor data offsets
        let tensor_infos = self.calculate_tensor_offsets()?;

        // Write tensor infos
        for info in &tensor_infos {
            self.write_tensor_info(writer, info)?;
        }

        // Pad to alignment before tensor data
        let current_pos = writer.stream_position().map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
        })?;
        let aligned = align_offset_checked(current_pos, self.alignment as u64)?;
        write_padding(writer, aligned - current_pos)?;

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
                let aligned = align_offset_checked(current_pos, self.alignment as u64)?;
                write_padding(writer, aligned - current_pos)?;
            }
        }

        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.alignment == 0 || self.alignment > 1_048_576 {
            return Err(PMetalError::InvalidArgument(format!(
                "GGUF alignment must be in 1..=1048576, got {}",
                self.alignment
            )));
        }

        for (key, value) in &self.metadata {
            if key.len() > MAX_STRING_LENGTH {
                return Err(PMetalError::InvalidArgument(format!(
                    "GGUF metadata key {} exceeds {} bytes",
                    key, MAX_STRING_LENGTH
                )));
            }
            validate_metadata_value(value)?;
        }

        let mut tensor_names = HashSet::with_capacity(self.tensors.len());
        for (info, data) in &self.tensors {
            if !tensor_names.insert(info.name.as_str()) {
                return Err(PMetalError::InvalidArgument(format!(
                    "Duplicate GGUF tensor name: {}",
                    info.name
                )));
            }
            if info.name.len() > MAX_STRING_LENGTH {
                return Err(PMetalError::InvalidArgument(format!(
                    "GGUF tensor name {} exceeds {} bytes",
                    info.name, MAX_STRING_LENGTH
                )));
            }
            if info.n_dimensions as usize != info.dimensions.len() {
                return Err(PMetalError::InvalidArgument(format!(
                    "GGUF tensor {} declares {} dimensions but stores {}",
                    info.name,
                    info.n_dimensions,
                    info.dimensions.len()
                )));
            }
            if info.n_dimensions > crate::reader::MAX_TENSOR_DIMS {
                return Err(PMetalError::InvalidArgument(format!(
                    "GGUF tensor {} has {} dimensions, exceeding maximum of {}",
                    info.name,
                    info.n_dimensions,
                    crate::reader::MAX_TENSOR_DIMS
                )));
            }
            let expected = info.byte_size_checked().map_err(|e| {
                PMetalError::InvalidArgument(format!(
                    "GGUF tensor {} has invalid dimensions or dtype: {}",
                    info.name, e
                ))
            })?;
            if expected != data.len() {
                return Err(PMetalError::InvalidArgument(format!(
                    "GGUF tensor {} has {} bytes but {:?} shape {:?} requires {} bytes",
                    info.name,
                    data.len(),
                    info.dtype,
                    info.dimensions,
                    expected
                )));
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
                validate_array_elements(arr)?;
                // Write array element type
                let elem_type = arr
                    .first()
                    .map(|v| v.value_type())
                    .unwrap_or(MetadataValueType::Uint8);
                writer
                    .write_u32::<LittleEndian>(elem_type as u32)
                    .map_err(|e| {
                        pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
                    })?;
                // Write array length
                writer
                    .write_u64::<LittleEndian>(arr.len() as u64)
                    .map_err(|e| {
                        pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
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
            MetadataValue::Array(inner) => {
                validate_array_elements(inner)?;
                // Nested arrays — write array header then element data (no per-element type prefix).
                // The GGUF spec requires array elements to omit individual type codes; the element
                // type is given once in the outer array header.
                let elem_type = inner.first().map(|v| v.value_type() as u32).unwrap_or(0); // empty array, type doesn't matter
                writer.write_u32::<LittleEndian>(elem_type).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
                })?;
                writer
                    .write_u64::<LittleEndian>(inner.len() as u64)
                    .map_err(|e| {
                        pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
                    })?;
                for elem in inner {
                    self.write_metadata_value_data(writer, elem)?;
                }
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
        // GGUF stores dimensions with the fastest-changing axis first. The
        // public builder accepts conventional row-major shapes, matching the
        // reader's reversed `TensorInfo::dimensions`.
        for dim in info.dimensions.iter().rev() {
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
        writer.write_u64::<LittleEndian>(info.offset).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
        })?;
        Ok(())
    }

    /// Calculate tensor offsets.
    fn calculate_tensor_offsets(&self) -> Result<Vec<TensorInfo>> {
        let mut infos = Vec::with_capacity(self.tensors.len());
        let mut offset: u64 = 0;

        for (info, data) in &self.tensors {
            let mut new_info = info.clone();
            new_info.offset = offset;
            infos.push(new_info);

            // Next offset is current + data size, aligned
            let len = u64::try_from(data.len()).map_err(|_| {
                PMetalError::InvalidArgument(format!(
                    "GGUF tensor {} byte length exceeds u64",
                    info.name
                ))
            })?;
            offset = offset.checked_add(len).ok_or_else(|| {
                PMetalError::InvalidArgument("GGUF tensor offsets overflow u64".into())
            })?;
            offset = align_offset_checked(offset, self.alignment as u64)?;
        }

        Ok(infos)
    }
}

fn validate_metadata_value(value: &MetadataValue) -> Result<()> {
    match value {
        MetadataValue::String(s) if s.len() > MAX_STRING_LENGTH => {
            Err(PMetalError::InvalidArgument(format!(
                "GGUF string metadata value exceeds {} bytes",
                MAX_STRING_LENGTH
            )))
        }
        MetadataValue::Array(arr) => {
            validate_array_elements(arr)?;
            for elem in arr {
                validate_metadata_value(elem)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_array_elements(arr: &[MetadataValue]) -> Result<()> {
    if arr.len() > MAX_ARRAY_LENGTH {
        return Err(PMetalError::InvalidArgument(format!(
            "GGUF metadata array has {} elements, exceeding {}",
            arr.len(),
            MAX_ARRAY_LENGTH
        )));
    }
    if let Some(first) = arr.first() {
        let expected = first.value_type();
        if arr.iter().any(|elem| elem.value_type() != expected) {
            return Err(PMetalError::InvalidArgument(
                "GGUF metadata arrays must contain a single value type".into(),
            ));
        }
    }
    Ok(())
}

fn write_padding<W: Write>(writer: &mut W, len: u64) -> Result<()> {
    const ZEROES: [u8; 1024] = [0; 1024];
    let mut remaining = len;
    while remaining > 0 {
        let chunk = remaining.min(ZEROES.len() as u64) as usize;
        writer.write_all(&ZEROES[..chunk]).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(e.kind(), e.to_string()))
        })?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn align_offset_checked(offset: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(PMetalError::InvalidArgument(
            "GGUF alignment must be greater than zero".into(),
        ));
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        Ok(offset)
    } else {
        offset
            .checked_add(alignment - remainder)
            .ok_or_else(|| PMetalError::InvalidArgument("GGUF aligned offset overflows u64".into()))
    }
}

/// Calculate aligned offset.
#[cfg(test)]
fn align_offset(offset: u64, alignment: u64) -> u64 {
    align_offset_checked(offset, alignment).expect("valid GGUF alignment and offset")
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
        builder.add_f32_tensor(
            "test.weight",
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let bytes = builder.build_to_bytes().unwrap();
        assert!(!bytes.is_empty());

        // Verify header shows 1 tensor
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tensor_count, 1);
    }

    #[test]
    fn tensor_dimensions_round_trip_in_row_major_order() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_f32_tensor(
            "test.weight",
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let bytes = builder.build_to_bytes().unwrap();
        let mut cursor = std::io::Cursor::new(bytes);
        let content = crate::GgufContent::read(&mut cursor).unwrap();
        let info = content.get_tensor_info("test.weight").unwrap();
        assert_eq!(info.dimensions, vec![2, 3]);
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

    #[test]
    fn zero_alignment_is_rejected() {
        let builder = GgufBuilder::with_model("llama", "test-model").alignment(0);
        assert!(builder.build_to_bytes().is_err());
    }

    #[test]
    fn heterogeneous_metadata_array_is_rejected() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_metadata(
            "bad.array",
            MetadataValue::Array(vec![
                MetadataValue::Uint32(1),
                MetadataValue::String("two".into()),
            ]),
        );
        assert!(builder.build_to_bytes().is_err());
    }

    #[test]
    fn tensor_byte_size_mismatch_is_rejected() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_raw_tensor("bad.weight", vec![2], GgmlType::F32, vec![0; 4]);
        assert!(builder.build_to_bytes().is_err());
    }

    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_f32_tensor("dup.weight", vec![1], vec![1.0]);
        builder.add_f32_tensor("dup.weight", vec![1], vec![2.0]);
        assert!(builder.build_to_bytes().is_err());
    }
}
