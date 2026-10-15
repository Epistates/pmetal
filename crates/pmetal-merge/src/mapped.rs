//! Memory-mapped safetensors files.
//!
//! A [`MappedFile`] maps one `.safetensors` file read-only and indexes its
//! header, so a tensor is a byte range of the mapping: nothing is read until
//! it is asked for, and the operating system pages data in and out as the
//! merge walks the model. Peak memory for a merge is therefore a few tensors,
//! not the checkpoints.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::{MergeError, Result};

/// Header entry of one tensor in a safetensors file.
#[derive(Debug, Clone)]
pub(crate) struct TensorMeta {
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    /// Absolute byte offset within the mapping (past the header).
    pub(crate) data_start: usize,
    pub(crate) data_end: usize,
}

/// A memory-mapped safetensors file with its tensor locations indexed.
pub(crate) struct MappedFile {
    mmap: Mmap,
    /// Ordered by file offset, for sequential access.
    pub(crate) tensors: Vec<(String, TensorMeta)>,
    index: HashMap<String, usize>,
    pub(crate) metadata: Option<HashMap<String, String>>,
}

impl MappedFile {
    /// Map `path` and parse its header, validating every tensor's byte range.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            MergeError::ModelLoad(format!("failed to open {}: {}", path.display(), e))
        })?;

        // SAFETY: the file is opened read-only and the map is dropped with
        // this handle. Like any memory-mapped input, it must not be truncated
        // by another process while a merge reads it.
        #[allow(unsafe_code)]
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            MergeError::ModelLoad(format!("failed to mmap {}: {}", path.display(), e))
        })?;

        let (header_len, metadata_view) = SafeTensors::read_metadata(&mmap).map_err(|e| {
            MergeError::ModelLoad(format!(
                "failed to parse safetensors header in {}: {}",
                path.display(),
                e
            ))
        })?;

        // 8-byte little-endian header length, the JSON header, then the data.
        let data_offset = 8 + header_len;

        let mut tensors: Vec<(String, TensorMeta)> = metadata_view
            .tensors()
            .into_iter()
            .map(|(name, info)| {
                let (start, end) = info.data_offsets;
                (
                    name,
                    TensorMeta {
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        data_start: data_offset + start,
                        data_end: data_offset + end,
                    },
                )
            })
            .collect();
        tensors.sort_by_key(|(_, m)| m.data_start);

        let index = tensors
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();

        Ok(Self {
            mmap,
            tensors,
            index,
            metadata: metadata_view.metadata().clone(),
        })
    }

    pub(crate) fn tensor_meta(&self, name: &str) -> Result<&TensorMeta> {
        self.index
            .get(name)
            .map(|&i| &self.tensors[i].1)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))
    }

    /// The raw little-endian bytes of tensor `name`. They start at an
    /// arbitrary byte offset, so they must be decoded without assuming
    /// alignment.
    pub(crate) fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let meta = self.tensor_meta(name)?;
        Ok(&self.mmap[meta.data_start..meta.data_end])
    }

    /// The bytes of the tensor described by `meta`, which must come from
    /// this file.
    pub(crate) fn slice(&self, meta: &TensorMeta) -> &[u8] {
        &self.mmap[meta.data_start..meta.data_end]
    }

    pub(crate) fn advise_sequential(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Sequential);
    }

    pub(crate) fn advise_dontneed(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Normal);
    }
}
