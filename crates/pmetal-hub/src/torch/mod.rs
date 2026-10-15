//! Reading PyTorch checkpoints (`pytorch_model.bin`, `.pt`, `.pth`) without
//! running anything.
//!
//! A checkpoint is a ZIP archive (stored, uncompressed; ZIP64 for the big
//! shards) holding a Python pickle that describes the tensors and one raw
//! storage file per tensor. [`TorchArchive`] maps the file, runs the pickle
//! through a restricted interpreter (it knows a short
//! allow-list of globals and refuses everything else, so opening an untrusted
//! file cannot execute code), and hands out each tensor as a slice of the
//! mapping. Nothing is copied until a caller does.
//!
//! [`convert_pytorch_dir`] turns a model directory's pickles into safetensors
//! files next to them, so every loader that reads safetensors reads models
//! that were only ever published as pickles.
//!
//! # Safety
//!
//! Mapping a file is `unsafe` in general: if another process truncates it
//! while mapped, reading a page past the new end raises `SIGBUS`. As with any
//! mapped model file, do not modify it while it is open.

mod convert;
mod pickle;
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub mod testing;
#[cfg(test)]
mod tests;
mod zip;

use std::path::{Path, PathBuf};

use memmap2::Mmap;
use pmetal_core::{PMetalError, Result};

pub use convert::{convert_pytorch_dir, write_safetensors};
pub use pickle::TorchDtype;

use pickle::Obj;

/// Most tensors a checkpoint may declare.
const MAX_TENSORS: usize = 1_000_000;

fn bad(path: &Path, what: impl std::fmt::Display) -> PMetalError {
    PMetalError::Serialization(format!("{}: {what}", path.display()))
}

/// One tensor of a checkpoint: a view of the mapped file.
#[derive(Debug, Clone)]
pub struct TorchTensor<'a> {
    /// The name in the state dict (`model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Element type.
    pub dtype: TorchDtype,
    /// Shape.
    pub shape: Vec<usize>,
    /// The elements, little-endian, row-major and contiguous.
    pub data: &'a [u8],
}

impl TorchTensor<'_> {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

struct Entry {
    group: Option<String>,
    name: String,
    dtype: TorchDtype,
    shape: Vec<usize>,
    /// Byte range of the tensor's elements in the mapped file.
    range: std::ops::Range<usize>,
}

/// A memory-mapped PyTorch checkpoint.
pub struct TorchArchive {
    path: PathBuf,
    map: Mmap,
    entries: Vec<Entry>,
}

impl std::fmt::Debug for TorchArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TorchArchive")
            .field("path", &self.path)
            .field("tensors", &self.entries.len())
            .finish()
    }
}

/// The nested dicts a trainer may have wrapped the weights in, in the order
/// they are looked for when the top level holds no tensors.
const WRAPPER_KEYS: &[&str] = &["state_dict", "model", "generator", "module"];

impl TorchArchive {
    /// Map and index a checkpoint. Only a short allow-list of pickle globals
    /// is understood; a file that needs anything else is an error.
    #[allow(unsafe_code)]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        // SAFETY: the mapping is read-only; the documented contract is that
        // the file is not modified while open. Every range handed out below
        // is checked against the mapped length.
        let map = unsafe { Mmap::map(&file)? };

        let archive = zip::StoredZip::parse(&map).map_err(|e| bad(path, e))?;
        let (root_dir, pickle_bytes) = archive
            .find_suffix("data.pkl")
            .ok_or_else(|| bad(path, "no data.pkl: not a PyTorch checkpoint (the legacy pre-1.6 format is not supported)"))?;
        if let Some(order) = archive.entry(&format!("{root_dir}/byteorder")) {
            if order != b"little" {
                return Err(bad(path, "the checkpoint is big-endian"));
            }
        }
        let root = pickle::load(pickle_bytes).map_err(|e| bad(path, e))?;

        let mut entries = Vec::new();
        let mut collect = |group: Option<&str>, dict: &Obj| -> Result<()> {
            for (name, value) in dict.dict_items().unwrap_or_default() {
                let (Some(name), Obj::Tensor(t)) = (name.as_str(), value) else {
                    // Non-tensor entries (`_metadata`, step counters) are not weights.
                    continue;
                };
                if entries.len() >= MAX_TENSORS {
                    return Err(bad(path, "implausibly many tensors"));
                }
                let width = t.storage.dtype.size();
                let numel = t
                    .shape
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| {
                        bad(path, format!("tensor `{name}` has an overflowing shape"))
                    })?;
                // Only densely packed row-major tensors are what a state
                // dict holds; a view with other strides is not one.
                let mut expected = 1usize;
                for (&dim, &stride) in t.shape.iter().zip(&t.stride).rev() {
                    if dim > 1 && stride != expected {
                        return Err(bad(
                            path,
                            format!("tensor `{name}` is not stored contiguously"),
                        ));
                    }
                    expected = expected.saturating_mul(dim);
                }
                let storage = format!("{root_dir}/data/{}", t.storage.key);
                let storage_range = archive.entry_range(&storage).ok_or_else(|| {
                    bad(
                        path,
                        format!("tensor `{name}`: storage `{storage}` is missing"),
                    )
                })?;
                let start = t
                    .offset
                    .checked_mul(width)
                    .zip(numel.checked_mul(width))
                    .and_then(|(s, len)| Some((s, s.checked_add(len)?)))
                    .filter(|&(_, end)| end <= storage_range.1)
                    .ok_or_else(|| {
                        bad(
                            path,
                            format!("tensor `{name}` runs past the end of its storage"),
                        )
                    })?;
                entries.push(Entry {
                    group: group.map(str::to_owned),
                    name: name.to_owned(),
                    dtype: t.storage.dtype,
                    shape: t.shape,
                    range: storage_range.0 + start.0..storage_range.0 + start.1,
                });
            }
            Ok(())
        };

        // Tensors at the top level, and in each nested dict (a trainer's
        // checkpoint wraps `{"generator": {...}, "mpd": {...}}`).
        collect(None, &root)?;
        for (key, value) in root.dict_items().unwrap_or_default() {
            if let (Some(key), Obj::Dict(_)) = (key.as_str(), &value) {
                collect(Some(key), &value)?;
            }
        }
        if entries.is_empty() {
            return Err(bad(path, "the checkpoint holds no tensors"));
        }
        Ok(Self {
            path: path.to_path_buf(),
            map,
            entries,
        })
    }

    /// The nested dicts that hold tensors (`generator`, `state_dict`, ...).
    pub fn groups(&self) -> Vec<&str> {
        let mut groups: Vec<&str> = self
            .entries
            .iter()
            .filter_map(|e| e.group.as_deref())
            .collect();
        groups.dedup();
        groups.sort_unstable();
        groups.dedup();
        groups
    }

    /// The tensors of one nested dict (`Some("generator")`), or, with `None`,
    /// the weights: the tensors at the top level if there are any, else those
    /// of the first of `state_dict`, `model`, `generator`, `module` present.
    pub fn tensors(&self, group: Option<&str>) -> Result<Vec<TorchTensor<'_>>> {
        let chosen: Option<Option<&str>> = match group {
            Some(g) => Some(Some(g)),
            None if self.entries.iter().any(|e| e.group.is_none()) => Some(None),
            None => WRAPPER_KEYS
                .iter()
                .find(|k| self.entries.iter().any(|e| e.group.as_deref() == Some(**k)))
                .map(|k| Some(*k)),
        };
        let Some(chosen) = chosen else {
            return Err(bad(
                &self.path,
                "no weights found: the top level holds no tensors and no state_dict/model/generator entry does",
            ));
        };
        let tensors: Vec<TorchTensor<'_>> = self
            .entries
            .iter()
            .filter(|e| e.group.as_deref() == chosen)
            .map(|e| TorchTensor {
                name: e.name.clone(),
                dtype: e.dtype,
                shape: e.shape.clone(),
                data: &self.map[e.range.clone()],
            })
            .collect();
        if tensors.is_empty() {
            return Err(bad(
                &self.path,
                format!(
                    "the checkpoint has no `{}` entry with tensors",
                    group.unwrap_or("top-level")
                ),
            ));
        }
        Ok(tensors)
    }
}
