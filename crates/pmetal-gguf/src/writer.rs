//! GGUF file writer.
//!
//! Two ways to write a file, both producing exactly the same bytes for the
//! same content:
//!
//! - [`GgufBuilder`] collects metadata and tensors in memory and writes them
//!   out ([`GgufBuilder::write`], [`GgufBuilder::write_file`]). Convenient for
//!   small files and tests.
//! - [`GgufStreamWriter`] takes the metadata and the tensor *declarations*
//!   (name, shape, type) first and then the tensor data one tensor at a time,
//!   so a multi-gigabyte model is quantized and written tensor by tensor
//!   instead of being held in memory twice. [`GgufFileWriter`] is the same
//!   writing to a file atomically.
//!
//! # Conformance
//!
//! The files follow the GGUF version 3 layout and the constraints the
//! reference loader enforces: keys of at most 65,535 bytes, tensor names of at
//! most 63 bytes (the reference stores them in a 64-byte buffer), at most four
//! dimensions, quantized rows that are whole blocks, tensor offsets that are
//! multiples of `general.alignment`, and `general.alignment` /
//! `general.quantization_version` written as `uint32`. Anything else is
//! refused here rather than producing a file another tool cannot open.
//!
//! # Atomicity
//!
//! [`GgufFileWriter`] and [`GgufBuilder::write_file`] write to a temporary
//! file in the destination directory, sync it, and rename it over the
//! destination, so a crash or an error never leaves a half-written model
//! under the final name.

use crate::reader::{MAX_ALIGNMENT, MAX_ARRAY_LENGTH, MAX_STRING_LENGTH};
use crate::{
    GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GgmlType, MetadataValue, MetadataValueType,
    TensorInfo,
};
use byteorder::{LittleEndian, WriteBytesExt};
use pmetal_core::{PMetalError, Result};
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Longest metadata key the specification allows.
pub const MAX_KEY_LENGTH: usize = 65_535;

/// Longest tensor name the reference loader accepts (it stores names in a
/// 64-byte buffer, terminator included).
pub const MAX_TENSOR_NAME_LENGTH: usize = 63;

/// Most dimensions the reference loader accepts for a tensor.
pub const MAX_WRITE_TENSOR_DIMS: usize = 4;

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
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        let infos: Vec<TensorInfo> = self.tensors.iter().map(|(i, _)| i.clone()).collect();
        let mut stream = GgufStreamWriter::new(writer, &self.metadata, infos, self.alignment)?;
        for (info, data) in &self.tensors {
            stream.write_tensor(&info.name, data)?;
        }
        stream.finish()?;
        Ok(())
    }

    /// Write the file to `path` atomically: the destination either keeps its
    /// previous content or holds the complete new file.
    pub fn write_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let infos: Vec<TensorInfo> = self.tensors.iter().map(|(i, _)| i.clone()).collect();
        let mut file = GgufFileWriter::create(path, &self.metadata, infos, self.alignment)?;
        for (info, data) in &self.tensors {
            file.write_tensor(&info.name, data)?;
        }
        file.finish()
    }

    /// A streaming, atomic writer for a file with this builder's metadata and
    /// alignment and the tensors `tensors` declares, which replaces the
    /// builder's own tensors (a model too large to hold in memory declares
    /// each tensor's type and shape up front and hands over its data one tensor
    /// at a time, in declaration order: [`GgufFileWriter::write_tensor`]).
    pub fn file_writer(
        &self,
        path: impl AsRef<Path>,
        tensors: Vec<TensorInfo>,
    ) -> Result<GgufFileWriter> {
        GgufFileWriter::create(path, &self.metadata, tensors, self.alignment)
    }

    /// Build to a byte vector.
    pub fn build_to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.write(&mut buffer)?;
        Ok(buffer)
    }
}

/// Counts the bytes written through it, so padding needs no `Seek`.
struct CountingWriter<W> {
    inner: W,
    written: u64,
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Writes a GGUF file whose tensors are declared up front and supplied one at
/// a time, in declaration order.
///
/// The header is written by [`GgufStreamWriter::new`]; each
/// [`write_tensor`](Self::write_tensor) checks the data against its
/// declaration and writes it followed by the padding to the next aligned
/// offset; [`finish`](Self::finish) verifies that every declared tensor was
/// supplied.
pub struct GgufStreamWriter<W: Write> {
    out: CountingWriter<W>,
    /// `(name, byte length)` of the tensors still to come, in order.
    pending: Vec<(String, u64)>,
    next: usize,
    alignment: u64,
}

impl<W: Write> GgufStreamWriter<W> {
    /// Validate the declarations and write the header (metadata, tensor infos
    /// and the padding before the data section).
    ///
    /// `metadata` should carry `general.architecture`. `general.alignment`
    /// is added when `alignment` is not the default of 32 and the map does
    /// not already say so (when it does, the two must agree), and
    /// `general.quantization_version` (2, the current format) is added when
    /// any tensor is quantized and the map has none, as the specification
    /// requires.
    pub fn new(
        writer: W,
        metadata: &BTreeMap<String, MetadataValue>,
        mut tensors: Vec<TensorInfo>,
        alignment: u32,
    ) -> Result<Self> {
        validate_alignment(alignment)?;
        let mut metadata = metadata.clone();
        if alignment != GGUF_DEFAULT_ALIGNMENT {
            metadata
                .entry(crate::keys::GENERAL_ALIGNMENT.to_string())
                .or_insert(MetadataValue::Uint32(alignment));
        }
        if tensors.iter().any(|t| t.dtype.is_quantized()) {
            metadata
                .entry(crate::keys::GENERAL_QUANTIZATION_VERSION.to_string())
                .or_insert(MetadataValue::Uint32(2));
        }
        let metadata = &metadata;
        match metadata.get(crate::keys::GENERAL_ALIGNMENT) {
            None | Some(MetadataValue::Uint32(_)) => {}
            Some(other) => {
                return Err(PMetalError::InvalidArgument(format!(
                    "general.alignment must be a uint32, got {:?}",
                    other.value_type()
                )));
            }
        }
        if let Some(MetadataValue::Uint32(declared)) = metadata.get(crate::keys::GENERAL_ALIGNMENT)
        {
            if *declared != alignment {
                return Err(PMetalError::InvalidArgument(format!(
                    "general.alignment is {declared} but the data is aligned to {alignment}"
                )));
            }
        }
        for (key, value) in metadata {
            if key.len() > MAX_KEY_LENGTH {
                return Err(PMetalError::InvalidArgument(format!(
                    "GGUF metadata key of {} bytes exceeds the {MAX_KEY_LENGTH}-byte limit",
                    key.len()
                )));
            }
            validate_metadata_value(key, value)?;
        }

        // Assign offsets: each tensor starts at an aligned offset.
        let mut names = HashSet::with_capacity(tensors.len());
        let mut pending = Vec::with_capacity(tensors.len());
        let mut offset = 0u64;
        for info in &mut tensors {
            let size = validate_tensor(info)?;
            if !names.insert(info.name.clone()) {
                return Err(PMetalError::InvalidArgument(format!(
                    "Duplicate GGUF tensor name: {}",
                    info.name
                )));
            }
            info.offset = offset;
            pending.push((info.name.clone(), size));
            offset = align_offset_checked(
                offset.checked_add(size).ok_or_else(|| {
                    PMetalError::InvalidArgument("GGUF tensor offsets overflow u64".into())
                })?,
                u64::from(alignment),
            )?;
        }

        let mut out = CountingWriter {
            inner: writer,
            written: 0,
        };
        out.write_u32::<LittleEndian>(GGUF_MAGIC)?;
        out.write_u32::<LittleEndian>(GGUF_VERSION)?;
        out.write_u64::<LittleEndian>(tensors.len() as u64)?;
        out.write_u64::<LittleEndian>(metadata.len() as u64)?;
        for (key, value) in metadata {
            write_string(&mut out, key)?;
            out.write_u32::<LittleEndian>(value.value_type() as u32)?;
            write_value_payload(&mut out, value)?;
        }
        for info in &tensors {
            write_tensor_info(&mut out, info)?;
        }
        pad_to_alignment(&mut out, u64::from(alignment))?;

        Ok(Self {
            out,
            pending,
            next: 0,
            alignment: u64::from(alignment),
        })
    }

    /// Write the next tensor's data. `name` must be the next declared tensor
    /// and `data` exactly its size.
    pub fn write_tensor(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let Some((expected, size)) = self.pending.get(self.next) else {
            return Err(PMetalError::InvalidArgument(format!(
                "tensor {name} was not declared"
            )));
        };
        if expected != name {
            return Err(PMetalError::InvalidArgument(format!(
                "tensors must be written in declaration order: expected {expected}, got {name}"
            )));
        }
        if data.len() as u64 != *size {
            return Err(PMetalError::InvalidArgument(format!(
                "GGUF tensor {name} has {} bytes but its declaration requires {size}",
                data.len()
            )));
        }
        self.out.write_all(data)?;
        pad_to_alignment(&mut self.out, self.alignment)?;
        self.next += 1;
        Ok(())
    }

    /// Finish the file: every declared tensor must have been written.
    pub fn finish(mut self) -> Result<W> {
        if let Some((name, _)) = self.pending.get(self.next) {
            return Err(PMetalError::InvalidArgument(format!(
                "tensor {name} (and {} more) declared but never written",
                self.pending.len() - self.next - 1
            )));
        }
        self.out.flush()?;
        Ok(self.out.inner)
    }
}

/// A [`GgufStreamWriter`] that writes to a file atomically.
///
/// The data goes to a temporary file in the destination's directory.
/// [`finish`](Self::finish) syncs it and renames it over the destination;
/// dropping the writer without finishing removes the temporary file.
pub struct GgufFileWriter {
    stream: GgufStreamWriter<std::io::BufWriter<tempfile::NamedTempFile>>,
    dest: PathBuf,
}

impl GgufFileWriter {
    /// Create the temporary file and write the header; see
    /// [`GgufStreamWriter::new`].
    pub fn create(
        dest: impl AsRef<Path>,
        metadata: &BTreeMap<String, MetadataValue>,
        tensors: Vec<TensorInfo>,
        alignment: u32,
    ) -> Result<Self> {
        let dest = dest.as_ref().to_path_buf();
        let dir = match dest.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let temp = tempfile::Builder::new()
            .prefix(".gguf-")
            .suffix(".tmp")
            .tempfile_in(dir)?;
        // 1 MiB: tensor data is written in large pieces already.
        let writer = std::io::BufWriter::with_capacity(1 << 20, temp);
        let stream = GgufStreamWriter::new(writer, metadata, tensors, alignment)?;
        Ok(Self { stream, dest })
    }

    /// See [`GgufStreamWriter::write_tensor`].
    pub fn write_tensor(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.stream.write_tensor(name, data)
    }

    /// Sync the file and move it into place.
    pub fn finish(self) -> Result<()> {
        let writer = self.stream.finish()?;
        let temp = writer
            .into_inner()
            .map_err(|e| PMetalError::Io(e.into_error()))?;
        temp.as_file().sync_all()?;
        temp.persist(&self.dest)
            .map_err(|e| PMetalError::Io(e.error))?;
        Ok(())
    }
}

fn validate_alignment(alignment: u32) -> Result<()> {
    if alignment == 0 || u64::from(alignment) > MAX_ALIGNMENT || !alignment.is_power_of_two() {
        return Err(PMetalError::InvalidArgument(format!(
            "GGUF alignment must be a power of two in 1..={MAX_ALIGNMENT}, got {alignment}"
        )));
    }
    Ok(())
}

/// Check one tensor declaration against what the reference loader accepts and
/// return the byte size its data must have.
fn validate_tensor(info: &TensorInfo) -> Result<u64> {
    if info.name.is_empty() || info.name.len() > MAX_TENSOR_NAME_LENGTH {
        return Err(PMetalError::InvalidArgument(format!(
            "GGUF tensor name {:?} must be 1 to {MAX_TENSOR_NAME_LENGTH} bytes, got {}",
            info.name,
            info.name.len()
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
    if info.dimensions.len() > MAX_WRITE_TENSOR_DIMS {
        return Err(PMetalError::InvalidArgument(format!(
            "GGUF tensor {} has {} dimensions; the format allows at most {MAX_WRITE_TENSOR_DIMS}",
            info.name,
            info.dimensions.len()
        )));
    }
    let block_size = info.dtype.block_size() as u64;
    let n_elements = info.n_elements_checked().map_err(|e| {
        PMetalError::InvalidArgument(format!(
            "GGUF tensor {} has invalid dimensions: {e}",
            info.name
        ))
    })?;
    if block_size > 1 && n_elements > 0 {
        let row = info.dimensions.last().copied().unwrap_or(1);
        if row % block_size != 0 {
            return Err(PMetalError::InvalidArgument(format!(
                "GGUF tensor {} has rows of {row} elements, not a multiple of the {:?} block size {block_size}",
                info.name, info.dtype
            )));
        }
    }
    let size = info.byte_size_checked().map_err(|e| {
        PMetalError::InvalidArgument(format!(
            "GGUF tensor {} has invalid dimensions or dtype: {e}",
            info.name
        ))
    })?;
    u64::try_from(size).map_err(|_| {
        PMetalError::InvalidArgument(format!("GGUF tensor {} byte length exceeds u64", info.name))
    })
}

fn validate_metadata_value(key: &str, value: &MetadataValue) -> Result<()> {
    match value {
        MetadataValue::String(s) if s.len() > MAX_STRING_LENGTH => {
            Err(PMetalError::InvalidArgument(format!(
                "GGUF string value of {key} exceeds {MAX_STRING_LENGTH} bytes"
            )))
        }
        MetadataValue::Array(arr) => {
            if arr.len() > MAX_ARRAY_LENGTH {
                return Err(PMetalError::InvalidArgument(format!(
                    "GGUF array {key} has {} elements, exceeding {MAX_ARRAY_LENGTH}",
                    arr.len()
                )));
            }
            if let Some(first) = arr.first() {
                let expected = first.value_type();
                if arr.iter().any(|elem| elem.value_type() != expected) {
                    return Err(PMetalError::InvalidArgument(format!(
                        "GGUF metadata array {key} must contain a single value type"
                    )));
                }
            }
            arr.iter().try_for_each(|e| validate_metadata_value(key, e))
        }
        _ => Ok(()),
    }
}

fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<()> {
    writer.write_u64::<LittleEndian>(s.len() as u64)?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

/// Write a value without its type tag (the tag is written once by the caller:
/// per entry for a key/value pair, per array for its elements).
fn write_value_payload<W: Write>(writer: &mut W, value: &MetadataValue) -> Result<()> {
    match value {
        MetadataValue::Uint8(v) => writer.write_u8(*v)?,
        MetadataValue::Int8(v) => writer.write_i8(*v)?,
        MetadataValue::Uint16(v) => writer.write_u16::<LittleEndian>(*v)?,
        MetadataValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
        MetadataValue::Uint32(v) => writer.write_u32::<LittleEndian>(*v)?,
        MetadataValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
        MetadataValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
        MetadataValue::Bool(v) => writer.write_u8(u8::from(*v))?,
        MetadataValue::String(s) => write_string(writer, s)?,
        MetadataValue::Uint64(v) => writer.write_u64::<LittleEndian>(*v)?,
        MetadataValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
        MetadataValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
        MetadataValue::Array(arr) => {
            // The element type is given once, in the array header; elements
            // carry no tag of their own. An empty array's type is irrelevant.
            let elem_type = arr
                .first()
                .map_or(MetadataValueType::Uint8, MetadataValue::value_type);
            writer.write_u32::<LittleEndian>(elem_type as u32)?;
            writer.write_u64::<LittleEndian>(arr.len() as u64)?;
            for elem in arr {
                write_value_payload(writer, elem)?;
            }
        }
    }
    Ok(())
}

fn write_tensor_info<W: Write>(writer: &mut W, info: &TensorInfo) -> Result<()> {
    write_string(writer, &info.name)?;
    writer.write_u32::<LittleEndian>(info.n_dimensions)?;
    // GGUF stores the fastest-changing axis first. The public builder accepts
    // conventional row-major shapes, matching the reader's reversed
    // `TensorInfo::dimensions`.
    for dim in info.dimensions.iter().rev() {
        writer.write_u64::<LittleEndian>(*dim)?;
    }
    writer.write_u32::<LittleEndian>(info.dtype as u32)?;
    writer.write_u64::<LittleEndian>(info.offset)?;
    Ok(())
}

fn pad_to_alignment<W: Write>(out: &mut CountingWriter<W>, alignment: u64) -> Result<()> {
    const ZEROES: [u8; 1024] = [0; 1024];
    let mut remaining = align_offset_checked(out.written, alignment)? - out.written;
    while remaining > 0 {
        let chunk = remaining.min(ZEROES.len() as u64) as usize;
        out.write_all(&ZEROES[..chunk])?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufContent;
    use std::io::Cursor;

    fn read(bytes: Vec<u8>) -> GgufContent {
        GgufContent::read(&mut Cursor::new(bytes)).unwrap()
    }

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
    fn test_gguf_tensor() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder.add_f32_tensor(
            "test.weight",
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let bytes = builder.build_to_bytes().unwrap();

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

        let content = read(builder.build_to_bytes().unwrap());
        let info = content.get_tensor_info("test.weight").unwrap();
        assert_eq!(info.dimensions, vec![2, 3]);
    }

    #[test]
    fn metadata_and_tensor_data_round_trip() {
        let mut builder = GgufBuilder::with_model("llama", "test-model");
        builder
            .add_u32("llama.block_count", 4)
            .add_u64("big", 1 << 40)
            .add_f32("f", 0.5)
            .add_bool("flag", true)
            .add_string("s", "héllo")
            .add_string_array("tokens", vec!["a".into(), "".into(), "ccc".into()])
            .add_i32_array("types", vec![1, -2, 3])
            .add_f32_array("scores", vec![0.0, -1.5])
            .add_metadata(
                "nested",
                MetadataValue::Array(vec![
                    MetadataValue::Array(vec![MetadataValue::Uint16(1), MetadataValue::Uint16(2)]),
                    MetadataValue::Array(vec![]),
                    MetadataValue::Array(vec![MetadataValue::Uint16(3)]),
                ]),
            )
            .add_metadata("small", MetadataValue::Int8(-3))
            .add_metadata("empty", MetadataValue::Array(vec![]));
        builder.add_f32_tensor("a", vec![3], vec![1.0, 2.0, 3.0]);
        builder.add_raw_tensor("b", vec![2, 32], GgmlType::Q8_0, vec![7u8; 2 * 34]);

        let bytes = builder.build_to_bytes().unwrap();
        let content = read(bytes.clone());
        for (key, value) in &builder.metadata {
            assert_eq!(content.get_metadata(key), Some(value), "{key}");
        }
        let mut cursor = Cursor::new(bytes);
        assert_eq!(
            content.read_tensor_data(&mut cursor, "a").unwrap(),
            [1.0f32, 2.0, 3.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            content.read_tensor_data(&mut cursor, "b").unwrap(),
            vec![7u8; 68]
        );
    }

    #[test]
    fn tensor_data_starts_aligned_and_every_tensor_is_padded() {
        for alignment in [1u32, 8, 32, 64, 4096] {
            let mut builder = GgufBuilder::with_model("llama", "m").alignment(alignment);
            builder.add_f32_tensor("a", vec![3], vec![1.0, 2.0, 3.0]);
            builder.add_f32_tensor("b", vec![5], vec![4.0; 5]);
            let bytes = builder.build_to_bytes().unwrap();
            let content = read(bytes.clone());
            assert_eq!(content.alignment, u64::from(alignment));
            assert_eq!(content.tensor_data_offset % u64::from(alignment), 0);
            for name in ["a", "b"] {
                assert_eq!(
                    content.get_tensor_info(name).unwrap().offset % u64::from(alignment),
                    0
                );
            }
            // The file ends on an aligned boundary, like the reference writer's.
            assert_eq!(bytes.len() as u64 % u64::from(alignment), 0);
        }
    }

    #[test]
    fn zero_or_non_power_of_two_alignment_is_rejected() {
        for bad in [0u32, 3, 48] {
            let builder = GgufBuilder::with_model("llama", "test-model").alignment(bad);
            assert!(builder.build_to_bytes().is_err(), "alignment {bad}");
        }
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

    #[test]
    fn reference_loader_limits_are_enforced() {
        // A 64-byte name does not fit the reference loader's 64-byte buffer.
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_f32_tensor("n".repeat(64), vec![1], vec![0.0]);
        assert!(builder.build_to_bytes().is_err());
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_f32_tensor("n".repeat(63), vec![1], vec![0.0]);
        assert!(builder.build_to_bytes().is_ok());

        // Empty name.
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_f32_tensor("", vec![1], vec![0.0]);
        assert!(builder.build_to_bytes().is_err());

        // More than four dimensions.
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_f32_tensor("t", vec![1, 1, 1, 1, 1], vec![0.0]);
        assert!(builder.build_to_bytes().is_err());

        // A key over 65,535 bytes.
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_u32("k".repeat(MAX_KEY_LENGTH + 1), 1);
        assert!(builder.build_to_bytes().is_err());

        // Quantized rows that are not whole blocks, even though the total is.
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_raw_tensor("q", vec![4, 40], GgmlType::Q4_0, vec![0u8; 5 * 18]);
        assert!(builder.build_to_bytes().is_err());
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_raw_tensor("q", vec![4, 64], GgmlType::Q4_0, vec![0u8; 8 * 18]);
        assert!(builder.build_to_bytes().is_ok());

        // general.alignment must be a uint32.
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_metadata(crate::keys::GENERAL_ALIGNMENT, MetadataValue::Uint64(32));
        assert!(builder.build_to_bytes().is_err());
    }

    #[test]
    fn stream_writer_matches_the_builder_byte_for_byte() {
        let mut builder = GgufBuilder::with_model("llama", "m");
        builder.add_u32("llama.block_count", 2);
        builder.add_f32_tensor("a", vec![3], vec![1.0, 2.0, 3.0]);
        builder.add_raw_tensor("b", vec![1, 32], GgmlType::Q8_0, vec![9u8; 34]);
        let expected = builder.build_to_bytes().unwrap();

        let infos = vec![
            TensorInfo::new("a", vec![3], GgmlType::F32),
            TensorInfo::new("b", vec![1, 32], GgmlType::Q8_0),
        ];
        let mut out = Vec::new();
        let mut stream = GgufStreamWriter::new(&mut out, &builder.metadata, infos, 32).unwrap();
        stream
            .write_tensor(
                "a",
                &[1.0f32, 2.0, 3.0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        stream.write_tensor("b", &[9u8; 34]).unwrap();
        stream.finish().unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn stream_writer_enforces_declaration_order_and_completeness() {
        let infos = || {
            vec![
                TensorInfo::new("a", vec![1], GgmlType::F32),
                TensorInfo::new("b", vec![1], GgmlType::F32),
            ]
        };
        let meta = BTreeMap::new();

        // Out of order.
        let mut s = GgufStreamWriter::new(Vec::new(), &meta, infos(), 32).unwrap();
        assert!(s.write_tensor("b", &[0; 4]).is_err());

        // Wrong size.
        let mut s = GgufStreamWriter::new(Vec::new(), &meta, infos(), 32).unwrap();
        assert!(s.write_tensor("a", &[0; 3]).is_err());

        // Undeclared tensor after all declared ones were written.
        let mut s = GgufStreamWriter::new(Vec::new(), &meta, infos(), 32).unwrap();
        s.write_tensor("a", &[0; 4]).unwrap();
        s.write_tensor("b", &[0; 4]).unwrap();
        assert!(s.write_tensor("c", &[0; 4]).is_err());

        // Finishing with a tensor missing.
        let mut s = GgufStreamWriter::new(Vec::new(), &meta, infos(), 32).unwrap();
        s.write_tensor("a", &[0; 4]).unwrap();
        assert!(s.finish().is_err());
    }

    #[test]
    fn file_writer_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        std::fs::write(&dest, b"previous content").unwrap();

        // A writer dropped without finishing leaves the destination alone and
        // no temporary file behind.
        {
            let infos = vec![TensorInfo::new("a", vec![1], GgmlType::F32)];
            let mut w = GgufFileWriter::create(&dest, &BTreeMap::new(), infos, 32).unwrap();
            w.write_tensor("a", &[0; 4]).unwrap();
        }
        assert_eq!(std::fs::read(&dest).unwrap(), b"previous content");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);

        // A failed build (validation error) also leaves it alone.
        let mut bad = GgufBuilder::with_model("llama", "m");
        bad.add_raw_tensor("x", vec![2], GgmlType::F32, vec![0; 4]);
        assert!(bad.write_file(&dest).is_err());
        assert_eq!(std::fs::read(&dest).unwrap(), b"previous content");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);

        // A successful write replaces it with the complete file.
        let mut good = GgufBuilder::with_model("llama", "m");
        good.add_f32_tensor("a", vec![2], vec![1.0, 2.0]);
        good.write_file(&dest).unwrap();
        let content = GgufContent::from_file(&dest).unwrap();
        assert_eq!(content.num_tensors(), 1);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
