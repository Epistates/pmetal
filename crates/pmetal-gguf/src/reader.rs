//! GGUF file reader.
//!
//! Implements the GGUF specification (versions 1 to 3; version 3 is current)
//! as published with ggml: a little-endian header, a metadata key/value
//! section, the tensor-info section and, after padding to
//! `general.alignment`, the tensor data.
//!
//! # Hostile input
//!
//! A GGUF file is untrusted input, and its header contains counts and lengths
//! that would otherwise decide how much memory the reader allocates. Every
//! such field is checked against the number of bytes the file actually has
//! left before anything is allocated, so a few-hundred-byte file cannot ask
//! for gigabytes. Every tensor's data range is checked against the file size
//! and the alignment rules once the header is parsed, which means a truncated
//! download is rejected when the file is opened, not when a late tensor is
//! first read.
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

use crate::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GgmlType, MetadataValue, TensorInfo};

/// GGUF file version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufVersion {
    /// Version 1 (legacy format, 32-bit counts and lengths).
    V1,
    /// Version 2 (64-bit counts and lengths).
    V2,
    /// Version 3 (current; adds big-endian files, which are not supported).
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

/// Maximum allowed string length in GGUF metadata (1 GiB).
///
/// A string is additionally limited to the bytes left in the file, so this
/// only bounds what a *valid* but enormous file may carry (a full tokenizer
/// JSON runs to tens of megabytes).
pub const MAX_STRING_LENGTH: usize = 1 << 30;

/// Maximum allowed array length in GGUF metadata (16 million elements).
///
/// An array is additionally limited to what the remaining bytes can hold.
pub const MAX_ARRAY_LENGTH: usize = 16 * 1024 * 1024;

/// Maximum allowed tensor count (100,000 tensors).
/// Modern LLMs typically have < 10,000 tensors.
pub const MAX_TENSOR_COUNT: usize = 100_000;

/// Maximum allowed metadata count (100,000 entries).
pub const MAX_METADATA_COUNT: usize = 100_000;

/// Maximum allowed tensor rank.
///
/// GGUF model tensors are normally rank <= 4. Keeping a modest upper bound
/// prevents malicious headers from forcing excessive dimension reads.
pub const MAX_TENSOR_DIMS: u32 = 16;

/// Maximum nesting depth for GGUF metadata arrays.
///
/// Prevents stack overflow from maliciously crafted files with deeply
/// nested array-of-array structures.
pub const MAX_ARRAY_DEPTH: u8 = 8;

/// Largest accepted `general.alignment`.
pub const MAX_ALIGNMENT: u64 = 1 << 20;

/// Error type for GGUF reading.
#[derive(Debug, thiserror::Error)]
pub enum GgufReadError {
    /// IO error (including a file that ends in the middle of a field).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid GGUF magic number.
    #[error("Invalid GGUF magic: expected 0x46554747, got 0x{0:08X}")]
    InvalidMagic(u32),
    /// Unsupported GGUF version.
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    /// The file is a big-endian GGUF, which this reader does not decode.
    #[error("Big-endian GGUF files are not supported")]
    BigEndian,
    /// Invalid metadata value type.
    #[error("Invalid metadata value type: {0}")]
    InvalidValueType(u32),
    /// Invalid tensor dtype.
    #[error("Invalid tensor dtype: {0}")]
    InvalidDtype(u32),
    /// Invalid alignment value.
    #[error("Invalid alignment: {0} (must be a power of two in 1..=1048576)")]
    InvalidAlignment(u64),
    /// `general.alignment` is present but is not a `uint32`.
    #[error("general.alignment must be a uint32, found {0}")]
    InvalidAlignmentType(String),
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
    /// Tensor has too many dimensions.
    #[error("Tensor {name} has {n_dims} dimensions, exceeding maximum of {max}")]
    TooManyTensorDimensions {
        /// Tensor name.
        name: String,
        /// Actual dimension count.
        n_dims: u32,
        /// Maximum allowed dimensions.
        max: u32,
    },
    /// A count or length in the header promises more data than the file has.
    #[error("{what} of {claimed} exceeds the {remaining} bytes remaining in the file")]
    ExceedsFile {
        /// The header field.
        what: &'static str,
        /// What the field claims.
        claimed: u64,
        /// Bytes left in the file at that point.
        remaining: u64,
    },
    /// Duplicate metadata key.
    #[error("Duplicate GGUF metadata key: {0}")]
    DuplicateMetadataKey(String),
    /// Duplicate tensor name.
    #[error("Duplicate GGUF tensor name: {0}")]
    DuplicateTensorName(String),
    /// Metadata array nesting depth exceeded the safety limit.
    #[error("Metadata array nesting depth exceeded limit of {max}")]
    ArrayNestingTooDeep {
        /// Maximum allowed nesting depth.
        max: u8,
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
    /// A quantized tensor's rows are not a whole number of blocks.
    #[error(
        "Tensor {name} has rows of {row_len} elements, not a multiple of the {dtype:?} block size {block_size}"
    )]
    RowNotBlockAligned {
        /// Tensor name.
        name: String,
        /// Tensor type.
        dtype: GgmlType,
        /// Elements per block.
        block_size: usize,
        /// Elements in the tensor's innermost dimension.
        row_len: u64,
    },
    /// A tensor's data offset is not a multiple of `general.alignment`.
    #[error("Tensor {name} has offset {offset}, not a multiple of the alignment {alignment}")]
    MisalignedTensor {
        /// Tensor name.
        name: String,
        /// Offset relative to the start of the tensor data.
        offset: u64,
        /// The file's alignment.
        alignment: u64,
    },
    /// A split model's shards are missing or disagree with each other.
    #[error("Split GGUF model: {0}")]
    Split(String),
    /// A tensor's data lies (partly) beyond the end of the file.
    #[error("Tensor {name} ends at byte {end}, past the end of the {file_len}-byte file")]
    TensorOutOfBounds {
        /// Tensor name.
        name: String,
        /// Absolute end of the tensor's data.
        end: u64,
        /// File length.
        file_len: u64,
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
    /// The file's `general.alignment` (default 32).
    pub alignment: u64,
    /// Total length of the file in bytes.
    pub file_len: u64,
}

impl Default for GgufContent {
    /// An empty version-3 file: no metadata, no tensors.
    fn default() -> Self {
        Self {
            version: GgufVersion::V3,
            metadata: HashMap::new(),
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            alignment: u64::from(GGUF_DEFAULT_ALIGNMENT),
            file_len: 0,
        }
    }
}

/// A reader limited to the bytes the file really has, so that header fields
/// can be checked against what remains.
struct Bounded<'a, R> {
    inner: &'a mut R,
    remaining: u64,
}

impl<R: Read> Read for Bounded<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let cap = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        let take = buf.len().min(cap);
        let n = self.inner.read(&mut buf[..take])?;
        self.remaining -= n as u64;
        Ok(n)
    }
}

impl GgufContent {
    /// Read GGUF content from the beginning of a stream.
    ///
    /// This reads the header, metadata, and tensor info but not the actual tensor data.
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self, GgufReadError> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;

        let mut bounded = Bounded {
            inner: reader,
            remaining: file_len,
        };
        let parsed = parse_header(&mut bounded, file_len)?;
        Ok(Self {
            version: parsed.version,
            metadata: parsed.metadata,
            tensor_infos: parsed.tensor_infos,
            tensor_data_offset: parsed.tensor_data_offset,
            alignment: parsed.alignment,
            file_len,
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

    /// Get all tensor names (in no particular order).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_infos.keys().map(String::as_str)
    }

    /// The tensors in the order their data is laid out in the file, which is
    /// the order to read them in for sequential disk access.
    pub fn tensors_by_offset(&self) -> Vec<&TensorInfo> {
        let mut infos: Vec<&TensorInfo> = self.tensor_infos.values().collect();
        infos.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
        infos
    }

    /// Get the total number of tensors.
    pub fn num_tensors(&self) -> usize {
        self.tensor_infos.len()
    }

    /// The absolute byte range `start..end` of a tensor's data in the file.
    pub fn tensor_data_range(&self, name: &str) -> Result<std::ops::Range<u64>, GgufReadError> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| GgufReadError::TensorNotFound(name.to_string()))?;
        tensor_range(self.tensor_data_offset, info)
    }

    /// Read raw tensor data from file.
    ///
    /// This seeks to the tensor's position in the file and reads the raw bytes.
    /// The tensor's extent was validated against the file size when the header
    /// was parsed, so the allocation is bounded by the file.
    pub fn read_tensor_data<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<Vec<u8>, GgufReadError> {
        let range = self.tensor_data_range(name)?;
        if range.end > self.file_len {
            return Err(GgufReadError::TensorOutOfBounds {
                name: name.to_string(),
                end: range.end,
                file_len: self.file_len,
            });
        }
        let len =
            usize::try_from(range.end - range.start).map_err(|_| GgufReadError::IntegerOverflow)?;
        let mut data = vec![0u8; len];
        reader.seek(SeekFrom::Start(range.start))?;
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

        for info in self.tensors_by_offset() {
            let data = self.read_tensor_data(reader, &info.name)?;
            tensors.insert(info.name.clone(), data);
        }

        Ok(tensors)
    }
}

struct ParsedHeader {
    version: GgufVersion,
    metadata: HashMap<String, MetadataValue>,
    tensor_infos: HashMap<String, TensorInfo>,
    tensor_data_offset: u64,
    alignment: u64,
}

fn parse_header<R: Read>(
    reader: &mut Bounded<'_, R>,
    file_len: u64,
) -> Result<ParsedHeader, GgufReadError> {
    let magic = reader.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(GgufReadError::InvalidMagic(magic));
    }

    let version_num = reader.read_u32::<LittleEndian>()?;
    let version = match version_num {
        1 => GgufVersion::V1,
        2 => GgufVersion::V2,
        3 => GgufVersion::V3,
        // A big-endian file stores the version byte-swapped, so the low half
        // of what we read as little endian is zero.
        v if v & 0xffff == 0 && v != 0 => return Err(GgufReadError::BigEndian),
        v => return Err(GgufReadError::UnsupportedVersion(v)),
    };

    let (tensor_count, metadata_kv_count) = match version {
        GgufVersion::V1 => (
            u64::from(reader.read_u32::<LittleEndian>()?),
            u64::from(reader.read_u32::<LittleEndian>()?),
        ),
        GgufVersion::V2 | GgufVersion::V3 => (
            reader.read_u64::<LittleEndian>()?,
            reader.read_u64::<LittleEndian>()?,
        ),
    };
    if tensor_count > MAX_TENSOR_COUNT as u64 {
        return Err(GgufReadError::TooManyTensors {
            count: usize::try_from(tensor_count).unwrap_or(usize::MAX),
            max: MAX_TENSOR_COUNT,
        });
    }
    if metadata_kv_count > MAX_METADATA_COUNT as u64 {
        return Err(GgufReadError::TooManyMetadata {
            count: usize::try_from(metadata_kv_count).unwrap_or(usize::MAX),
            max: MAX_METADATA_COUNT,
        });
    }
    // The smallest possible entries: a key (length + empty) and a type, and a
    // name, rank, type and offset. Counts that cannot fit in the file are
    // rejected before anything is reserved for them.
    let len_size = string_len_size(version);
    check_fits(
        "metadata entry count",
        metadata_kv_count,
        len_size + 4 + 1,
        reader.remaining,
    )?;
    check_fits(
        "tensor count",
        tensor_count,
        len_size + 4 + 4 + 8,
        reader.remaining,
    )?;

    let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        let key = read_string(reader, version)?;
        let value_type = reader.read_u32::<LittleEndian>()?;
        let value = read_value(reader, value_type, version, 0)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GgufReadError::DuplicateMetadataKey(key));
        }
    }

    let mut tensor_infos = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(reader, version)?;
        let n_dims = reader.read_u32::<LittleEndian>()?;
        if n_dims > MAX_TENSOR_DIMS {
            return Err(GgufReadError::TooManyTensorDimensions {
                name,
                n_dims,
                max: MAX_TENSOR_DIMS,
            });
        }

        // GGUF stores the fastest-varying dimension first; the in-memory
        // convention here is row-major, so reverse.
        let mut dims = (0..n_dims)
            .map(|_| match version {
                GgufVersion::V1 => reader.read_u32::<LittleEndian>().map(u64::from),
                GgufVersion::V2 | GgufVersion::V3 => reader.read_u64::<LittleEndian>(),
            })
            .collect::<Result<Vec<u64>, _>>()?;
        dims.reverse();

        let dtype_num = reader.read_u32::<LittleEndian>()?;
        let dtype =
            GgmlType::try_from(dtype_num).map_err(|_| GgufReadError::InvalidDtype(dtype_num))?;
        let offset = reader.read_u64::<LittleEndian>()?;

        let info = TensorInfo {
            name: name.clone(),
            n_dimensions: n_dims,
            dimensions: dims,
            dtype,
            offset,
        };
        validate_tensor_shape(&info)?;
        if tensor_infos.insert(name.clone(), info).is_some() {
            return Err(GgufReadError::DuplicateTensorName(name));
        }
    }

    let alignment = match metadata.get(crate::keys::GENERAL_ALIGNMENT) {
        None => u64::from(GGUF_DEFAULT_ALIGNMENT),
        Some(MetadataValue::Uint32(a)) => u64::from(*a),
        Some(other) => {
            return Err(GgufReadError::InvalidAlignmentType(format!(
                "{:?}",
                other.value_type()
            )));
        }
    };
    if alignment == 0 || alignment > MAX_ALIGNMENT || !alignment.is_power_of_two() {
        return Err(GgufReadError::InvalidAlignment(alignment));
    }

    // Everything consumed so far is the header; the tensor data starts at the
    // next multiple of the alignment.
    let header_len = file_len - reader.remaining;
    let tensor_data_offset = align_offset_checked(header_len, alignment)?;

    for info in tensor_infos.values() {
        if info.offset % alignment != 0 {
            return Err(GgufReadError::MisalignedTensor {
                name: info.name.clone(),
                offset: info.offset,
                alignment,
            });
        }
        let range = tensor_range(tensor_data_offset, info)?;
        if range.end > file_len {
            return Err(GgufReadError::TensorOutOfBounds {
                name: info.name.clone(),
                end: range.end,
                file_len,
            });
        }
    }

    Ok(ParsedHeader {
        version,
        metadata,
        tensor_infos,
        tensor_data_offset,
        alignment,
    })
}

/// Fail unless `count` items of at least `min_size` bytes fit in `remaining`.
fn check_fits(
    what: &'static str,
    count: u64,
    min_size: u64,
    remaining: u64,
) -> Result<(), GgufReadError> {
    match count.checked_mul(min_size) {
        Some(total) if total <= remaining => Ok(()),
        _ => Err(GgufReadError::ExceedsFile {
            what,
            claimed: count,
            remaining,
        }),
    }
}

/// Width of the length prefix of a string, and of an array count.
fn string_len_size(version: GgufVersion) -> u64 {
    match version {
        GgufVersion::V1 => 4,
        GgufVersion::V2 | GgufVersion::V3 => 8,
    }
}

fn align_offset_checked(offset: u64, alignment: u64) -> Result<u64, GgufReadError> {
    let remainder = offset % alignment;
    if remainder == 0 {
        Ok(offset)
    } else {
        offset
            .checked_add(alignment - remainder)
            .ok_or(GgufReadError::IntegerOverflow)
    }
}

/// The absolute byte range of a tensor's data given where the data section
/// starts.
fn tensor_range(
    tensor_data_offset: u64,
    info: &TensorInfo,
) -> Result<std::ops::Range<u64>, GgufReadError> {
    let size = tensor_byte_size(info)?;
    let start = tensor_data_offset
        .checked_add(info.offset)
        .ok_or(GgufReadError::IntegerOverflow)?;
    let end = start
        .checked_add(size)
        .ok_or(GgufReadError::IntegerOverflow)?;
    Ok(start..end)
}

fn tensor_byte_size(info: &TensorInfo) -> Result<u64, GgufReadError> {
    let size = info.byte_size_checked().map_err(|e| match e {
        crate::TensorSizeError::ElementCountOverflow | crate::TensorSizeError::ByteSizeOverflow => {
            GgufReadError::IntegerOverflow
        }
        crate::TensorSizeError::ElementCountTooLarge(n) => GgufReadError::TensorTooLarge {
            n_elements: n,
            dtype: format!("{:?}", info.dtype),
        },
    })?;
    u64::try_from(size).map_err(|_| GgufReadError::IntegerOverflow)
}

/// Check that a tensor's element count and byte size are computable and that
/// its rows are whole blocks.
fn validate_tensor_shape(info: &TensorInfo) -> Result<(), GgufReadError> {
    tensor_byte_size(info)?;
    let block_size = info.dtype.block_size();
    if block_size > 1 {
        let row_len = info.dimensions.last().copied().unwrap_or(1);
        if row_len % block_size as u64 != 0 && info.n_elements_checked().is_ok_and(|n| n > 0) {
            return Err(GgufReadError::RowNotBlockAligned {
                name: info.name.clone(),
                dtype: info.dtype,
                block_size,
                row_len,
            });
        }
    }
    Ok(())
}

/// Read a GGUF string, limited by the bytes left in the file.
fn read_string<R: Read>(
    reader: &mut Bounded<'_, R>,
    version: GgufVersion,
) -> Result<String, GgufReadError> {
    let len = match version {
        GgufVersion::V1 => u64::from(reader.read_u32::<LittleEndian>()?),
        GgufVersion::V2 | GgufVersion::V3 => reader.read_u64::<LittleEndian>()?,
    };
    if len > reader.remaining {
        return Err(GgufReadError::ExceedsFile {
            what: "string length",
            claimed: len,
            remaining: reader.remaining,
        });
    }
    if len > MAX_STRING_LENGTH as u64 {
        return Err(GgufReadError::StringTooLarge {
            size: usize::try_from(len).unwrap_or(usize::MAX),
            max: MAX_STRING_LENGTH,
        });
    }

    // `len` is no larger than the bytes the file still has, so this
    // allocation cannot exceed the file's size.
    let mut bytes = vec![0u8; len as usize];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| GgufReadError::InvalidUtf8)
}

/// The fewest bytes one element of a metadata type occupies in the file, used
/// to bound an array's length by what the file has left.
fn min_encoded_size(value_type: u32, version: GgufVersion) -> Result<u64, GgufReadError> {
    Ok(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        // length prefix, empty string
        8 => string_len_size(version),
        // element type + count
        9 => 4 + string_len_size(version),
        t => return Err(GgufReadError::InvalidValueType(t)),
    })
}

/// Read a metadata value with size validation.
///
/// `depth` tracks the current array nesting depth and must be `0` at the
/// top level. Returns [`GgufReadError::ArrayNestingTooDeep`] if the depth
/// exceeds [`MAX_ARRAY_DEPTH`].
fn read_value<R: Read>(
    reader: &mut Bounded<'_, R>,
    value_type: u32,
    version: GgufVersion,
    depth: u8,
) -> Result<MetadataValue, GgufReadError> {
    let value = match value_type {
        0 => MetadataValue::Uint8(reader.read_u8()?),
        1 => MetadataValue::Int8(reader.read_i8()?),
        2 => MetadataValue::Uint16(reader.read_u16::<LittleEndian>()?),
        3 => MetadataValue::Int16(reader.read_i16::<LittleEndian>()?),
        4 => MetadataValue::Uint32(reader.read_u32::<LittleEndian>()?),
        5 => MetadataValue::Int32(reader.read_i32::<LittleEndian>()?),
        6 => MetadataValue::Float32(reader.read_f32::<LittleEndian>()?),
        7 => match reader.read_u8()? {
            0 => MetadataValue::Bool(false),
            1 => MetadataValue::Bool(true),
            // The spec defines a bool as 0 or 1.
            _ => return Err(GgufReadError::InvalidValueType(7)),
        },
        8 => MetadataValue::String(read_string(reader, version)?),
        9 => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(GgufReadError::ArrayNestingTooDeep {
                    max: MAX_ARRAY_DEPTH,
                });
            }

            let elem_type = reader.read_u32::<LittleEndian>()?;
            let len = match version {
                GgufVersion::V1 => u64::from(reader.read_u32::<LittleEndian>()?),
                GgufVersion::V2 | GgufVersion::V3 => reader.read_u64::<LittleEndian>()?,
            };
            let min_size = min_encoded_size(elem_type, version)?;
            if len > MAX_ARRAY_LENGTH as u64 {
                return Err(GgufReadError::ArrayTooLarge {
                    len: usize::try_from(len).unwrap_or(usize::MAX),
                    max: MAX_ARRAY_LENGTH,
                });
            }
            check_fits("array length", len, min_size, reader.remaining)?;

            let mut elements = Vec::with_capacity(len as usize);
            for _ in 0..len {
                elements.push(read_value(reader, elem_type, version, depth + 1)?);
            }
            MetadataValue::Array(elements)
        }
        10 => MetadataValue::Uint64(reader.read_u64::<LittleEndian>()?),
        11 => MetadataValue::Int64(reader.read_i64::<LittleEndian>()?),
        12 => MetadataValue::Float64(reader.read_f64::<LittleEndian>()?),
        t => return Err(GgufReadError::InvalidValueType(t)),
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Cursor;

    fn write_string_v3(bytes: &mut Vec<u8>, s: &str) {
        bytes.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        bytes.extend_from_slice(s.as_bytes());
    }

    fn header_v3(tensors: u64, kvs: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        bytes.write_u32::<LittleEndian>(3).unwrap();
        bytes.write_u64::<LittleEndian>(tensors).unwrap();
        bytes.write_u64::<LittleEndian>(kvs).unwrap();
        bytes
    }

    fn read(bytes: Vec<u8>) -> Result<GgufContent, GgufReadError> {
        GgufContent::read(&mut Cursor::new(bytes))
    }

    /// A file with one 2x3 f32 tensor and a little metadata.
    fn small_valid_file() -> Vec<u8> {
        let mut builder = crate::GgufBuilder::with_model("llama", "tiny");
        builder.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "bb".into()]);
        builder.add_f32_tensor("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        builder.build_to_bytes().unwrap()
    }

    #[test]
    fn test_ggml_type_conversion() {
        assert_eq!(GgmlType::try_from(0), Ok(GgmlType::F32));
        assert_eq!(GgmlType::try_from(1), Ok(GgmlType::F16));
        assert_eq!(GgmlType::try_from(2), Ok(GgmlType::Q4_0));
        assert_eq!(GgmlType::try_from(8), Ok(GgmlType::Q8_0));
        assert_eq!(GgmlType::try_from(30), Ok(GgmlType::Bf16));
        assert_eq!(GgmlType::try_from(39), Ok(GgmlType::Mxfp4));
        assert_eq!(GgmlType::try_from(40), Ok(GgmlType::Nvfp4));
        assert_eq!(GgmlType::try_from(41), Ok(GgmlType::Q1_0));
        assert!(GgmlType::try_from(255).is_err());
    }

    #[test]
    fn test_version_display() {
        assert_eq!(GgufVersion::V1.to_string(), "v1");
        assert_eq!(GgufVersion::V2.to_string(), "v2");
        assert_eq!(GgufVersion::V3.to_string(), "v3");
    }

    #[test]
    fn duplicate_metadata_keys_are_rejected() {
        let mut bytes = header_v3(0, 2);
        write_string_v3(&mut bytes, "dup");
        bytes.write_u32::<LittleEndian>(4).unwrap();
        bytes.write_u32::<LittleEndian>(1).unwrap();
        write_string_v3(&mut bytes, "dup");
        bytes.write_u32::<LittleEndian>(4).unwrap();
        bytes.write_u32::<LittleEndian>(2).unwrap();

        let err = read(bytes).unwrap_err();
        assert!(matches!(err, GgufReadError::DuplicateMetadataKey(key) if key == "dup"));
    }

    #[test]
    fn excessive_tensor_rank_is_rejected_before_dimension_allocation() {
        let mut bytes = header_v3(1, 0);
        write_string_v3(&mut bytes, "bad.weight");
        bytes
            .write_u32::<LittleEndian>(MAX_TENSOR_DIMS + 1)
            .unwrap();
        bytes.resize(bytes.len() + 64, 0);

        let err = read(bytes).unwrap_err();
        assert!(
            matches!(err, GgufReadError::TooManyTensorDimensions { name, .. } if name == "bad.weight")
        );
    }

    #[test]
    fn counts_larger_than_the_file_are_rejected_without_allocating() {
        // A 24-byte file claiming the maximum number of tensors.
        let err = read(header_v3(MAX_TENSOR_COUNT as u64, 0)).unwrap_err();
        assert!(matches!(
            err,
            GgufReadError::ExceedsFile {
                what: "tensor count",
                ..
            }
        ));
        let err = read(header_v3(0, MAX_METADATA_COUNT as u64)).unwrap_err();
        assert!(matches!(
            err,
            GgufReadError::ExceedsFile {
                what: "metadata entry count",
                ..
            }
        ));
    }

    #[test]
    fn string_longer_than_the_file_is_rejected_without_allocating() {
        let mut bytes = header_v3(0, 1);
        bytes.write_u64::<LittleEndian>(1 << 29).unwrap(); // key claims 512 MiB
        bytes.extend_from_slice(b"short");
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::ExceedsFile {
                what: "string length",
                ..
            }
        ));
    }

    #[test]
    fn array_longer_than_the_file_is_rejected_without_allocating() {
        let mut bytes = header_v3(0, 1);
        write_string_v3(&mut bytes, "k");
        bytes.write_u32::<LittleEndian>(9).unwrap(); // array
        bytes.write_u32::<LittleEndian>(10).unwrap(); // of u64
        bytes.write_u64::<LittleEndian>(10_000_000).unwrap(); // 80 MB, from a 60-byte file
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::ExceedsFile {
                what: "array length",
                ..
            }
        ));
    }

    #[test]
    fn nested_arrays_are_bounded_too() {
        // Array of 1,000,000 arrays, each header 12 bytes: needs 12 MB.
        let mut bytes = header_v3(0, 1);
        write_string_v3(&mut bytes, "k");
        bytes.write_u32::<LittleEndian>(9).unwrap();
        bytes.write_u32::<LittleEndian>(9).unwrap();
        bytes.write_u64::<LittleEndian>(1_000_000).unwrap();
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::ExceedsFile { .. }
        ));
    }

    #[test]
    fn unknown_array_element_type_is_rejected_even_when_empty() {
        let mut bytes = header_v3(0, 1);
        write_string_v3(&mut bytes, "k");
        bytes.write_u32::<LittleEndian>(9).unwrap();
        bytes.write_u32::<LittleEndian>(99).unwrap();
        bytes.write_u64::<LittleEndian>(0).unwrap();
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::InvalidValueType(99)
        ));
    }

    #[test]
    fn deep_array_nesting_is_rejected() {
        let mut bytes = header_v3(0, 1);
        write_string_v3(&mut bytes, "k");
        bytes.write_u32::<LittleEndian>(9).unwrap();
        for _ in 0..=MAX_ARRAY_DEPTH {
            bytes.write_u32::<LittleEndian>(9).unwrap();
            bytes.write_u64::<LittleEndian>(1).unwrap();
        }
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::ArrayNestingTooDeep { .. }
        ));
    }

    #[test]
    fn bool_other_than_zero_or_one_is_rejected() {
        let mut bytes = header_v3(0, 1);
        write_string_v3(&mut bytes, "k");
        bytes.write_u32::<LittleEndian>(7).unwrap();
        bytes.push(2);
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::InvalidValueType(7)
        ));
    }

    #[test]
    fn big_endian_files_are_named_as_such() {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        bytes.extend_from_slice(&3u32.to_be_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        assert!(matches!(read(bytes).unwrap_err(), GgufReadError::BigEndian));
    }

    #[test]
    fn strings_are_not_trimmed() {
        let mut builder = crate::GgufBuilder::with_model("llama", "tiny");
        builder.add_string("odd", "a\0");
        builder.add_string("tokenizer.chat_template", "x\n");
        let content = read(builder.build_to_bytes().unwrap()).unwrap();
        assert_eq!(
            content.get_metadata("odd"),
            Some(&MetadataValue::String("a\0".into()))
        );
        assert_eq!(
            content.get_metadata("tokenizer.chat_template"),
            Some(&MetadataValue::String("x\n".into()))
        );
    }

    /// Replace the first occurrence of `needle` in `bytes`.
    fn patch(bytes: &mut [u8], needle: &[u8], with: &[u8]) {
        assert_eq!(needle.len(), with.len());
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("needle present");
        bytes[at..at + needle.len()].copy_from_slice(with);
    }

    #[test]
    fn alignment_must_be_a_uint32_power_of_two() {
        let valid = small_valid_file();
        assert_eq!(read(valid.clone()).unwrap().alignment, 32);

        // 48 is not a power of two. Rewrite the value that follows the key.
        let mut bytes = valid.clone();
        let key = b"general.alignment";
        let at = bytes.windows(key.len()).position(|w| w == key).unwrap();
        let value_at = at + key.len() + 4; // skip the type
        bytes[value_at..value_at + 4].copy_from_slice(&48u32.to_le_bytes());
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::InvalidAlignment(48)
        ));

        // Zero alignment.
        let mut bytes = valid.clone();
        bytes[value_at..value_at + 4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::InvalidAlignment(0)
        ));

        // A uint64 alignment is not what the specification defines.
        let mut builder = crate::GgufBuilder::with_model("llama", "tiny");
        builder.add_metadata(crate::keys::GENERAL_ALIGNMENT, MetadataValue::Uint64(32));
        builder.add_f32_tensor("w", vec![2], vec![1.0, 2.0]);
        assert!(matches!(
            read(builder.build_to_bytes().unwrap()).unwrap_err(),
            GgufReadError::InvalidAlignmentType(_)
        ));
    }

    #[test]
    fn tensor_extending_past_the_file_is_rejected_at_open() {
        // Claim 2^40 elements for a 24-byte tensor: a reader that trusted the
        // header would try to allocate 4 TiB.
        let mut bytes = small_valid_file();
        patch(&mut bytes, &3u64.to_le_bytes(), &(1u64 << 40).to_le_bytes());
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::TensorOutOfBounds { .. }
        ));
    }

    /// A file holding one tensor, with `data_len` bytes of data after the
    /// header (padded to 32) and the given shape, type and offset in its info.
    fn file_with_tensor(ne: &[u64], dtype: u32, offset: u64, data_len: usize) -> Vec<u8> {
        let mut bytes = header_v3(1, 0);
        write_string_v3(&mut bytes, "t.weight");
        bytes.write_u32::<LittleEndian>(ne.len() as u32).unwrap();
        for &d in ne {
            bytes.write_u64::<LittleEndian>(d).unwrap();
        }
        bytes.write_u32::<LittleEndian>(dtype).unwrap();
        bytes.write_u64::<LittleEndian>(offset).unwrap();
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.resize(bytes.len() + data_len, 0);
        bytes
    }

    #[test]
    fn misaligned_tensor_offset_is_rejected() {
        let bytes = file_with_tensor(&[4], 0, 8, 64);
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::MisalignedTensor { offset: 8, .. }
        ));
        assert!(read(file_with_tensor(&[4], 0, 32, 64)).is_ok());
    }

    #[test]
    fn quantized_rows_must_be_whole_blocks() {
        // ne0 = 40 is not a multiple of the Q4_0 block size, even though
        // 40 * 4 elements is 5 blocks in total.
        let bytes = file_with_tensor(&[40, 4], 2, 0, 4096);
        assert!(matches!(
            read(bytes).unwrap_err(),
            GgufReadError::RowNotBlockAligned { row_len: 40, .. }
        ));
        assert!(read(file_with_tensor(&[64, 4], 2, 0, 4096)).is_ok());
    }

    #[test]
    fn every_truncation_of_a_valid_file_is_an_error() {
        let valid = small_valid_file();
        assert!(read(valid.clone()).is_ok());
        for len in 0..valid.len() {
            assert!(
                read(valid[..len].to_vec()).is_err(),
                "a {len}-byte prefix of a {}-byte file was accepted",
                valid.len()
            );
        }
    }

    #[test]
    fn corrupting_any_byte_never_panics() {
        let valid = small_valid_file();
        for at in 0..valid.len() {
            for flip in [0xffu8, 0x80, 0x01] {
                let mut bytes = valid.clone();
                bytes[at] ^= flip;
                let _ = read(bytes);
            }
        }
    }

    #[test]
    fn tensors_by_offset_follows_the_file_layout() {
        let mut builder = crate::GgufBuilder::with_model("llama", "tiny");
        builder.add_f32_tensor("z", vec![4], vec![0.0; 4]);
        builder.add_f32_tensor("a", vec![4], vec![0.0; 4]);
        let content = read(builder.build_to_bytes().unwrap()).unwrap();
        let order: Vec<&str> = content
            .tensors_by_offset()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(order, ["z", "a"]);
    }
}
