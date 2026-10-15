//! A GGUF model on disk: one file or a split set, with memory-mapped tensor data.
//!
//! [`GgufContent`] parses a single file's header; [`GgufModel`] is what a
//! loader wants. It opens the file, maps it, and hands out tensor bytes as
//! slices of the mapping, so nothing is read until a tensor is used and
//! decoding can run straight from the page cache into the destination buffer
//! ([`crate::dequant::dequantize_into`]).
//!
//! # Split models
//!
//! Large checkpoints are published as a set of files named
//! `<name>-00001-of-00003.gguf`, each a complete GGUF file whose metadata
//! carries `split.no` (zero-based `uint16`), `split.count` (`uint16`) and
//! `split.tensors.count` (`int32`, the total across shards). The first shard
//! holds the model's metadata (and possibly some tensors); the others hold
//! tensors. Opening any shard of a set opens the whole set and presents one
//! tensor namespace; [`GgufModel::metadata_content`] is the first shard.
//!
//! # Safety
//!
//! Mapping a file is `unsafe` in general: if another process truncates it
//! while mapped, reading a page past the new end raises `SIGBUS`. As with any
//! mapped model file, do not modify it while it is open.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::reader::{GgufContent, GgufReadError};
use crate::{MetadataValue, TensorInfo};

/// Metadata keys of the split convention.
const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

struct Shard {
    path: PathBuf,
    content: GgufContent,
    map: Mmap,
}

/// A GGUF model (one file or a split set) with memory-mapped tensor data.
pub struct GgufModel {
    /// In `split.no` order; never empty.
    shards: Vec<Shard>,
    /// Tensor name to the index of the shard that holds it.
    owner: HashMap<String, usize>,
}

impl std::fmt::Debug for GgufModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufModel")
            .field(
                "shards",
                &self.shards.iter().map(|s| &s.path).collect::<Vec<_>>(),
            )
            .field("tensors", &self.owner.len())
            .finish()
    }
}

#[allow(unsafe_code)]
fn open_shard(path: &Path) -> Result<Shard, GgufReadError> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the mapping is read-only; the documented contract is that the
    // file is not modified while open. Every range handed out is checked
    // against the mapped length.
    let map = unsafe { Mmap::map(&file)? };
    let content = GgufContent::read(&mut std::io::Cursor::new(&map[..]))?;
    Ok(Shard {
        path: path.to_path_buf(),
        content,
        map,
    })
}

/// `(split.no, split.count, split.tensors.count)` if the file declares them.
fn split_keys(content: &GgufContent) -> Result<Option<(u64, u64, Option<i64>)>, GgufReadError> {
    let as_u = |key: &str| -> Result<Option<u64>, GgufReadError> {
        match content.get_metadata(key) {
            None => Ok(None),
            Some(MetadataValue::Uint16(v)) => Ok(Some(u64::from(*v))),
            Some(MetadataValue::Uint32(v)) => Ok(Some(u64::from(*v))),
            Some(MetadataValue::Uint64(v)) => Ok(Some(*v)),
            Some(MetadataValue::Int32(v)) if *v >= 0 => Ok(Some(*v as u64)),
            Some(other) => Err(GgufReadError::Split(format!(
                "{key} must be an unsigned integer, found {:?}",
                other.value_type()
            ))),
        }
    };
    let (Some(no), Some(count)) = (as_u(SPLIT_NO)?, as_u(SPLIT_COUNT)?) else {
        return Ok(None);
    };
    let total = match content.get_metadata(SPLIT_TENSORS_COUNT) {
        Some(MetadataValue::Int32(v)) => Some(i64::from(*v)),
        Some(MetadataValue::Uint32(v)) => Some(i64::from(*v)),
        Some(MetadataValue::Int64(v)) => Some(*v),
        _ => None,
    };
    Ok(Some((no, count, total)))
}

/// Split `<prefix>-<no>-of-<count>.gguf` into `(prefix, no, count)` as written
/// (the numbers keep their zero padding).
fn split_file_parts(path: &Path) -> Option<(&str, &str, &str)> {
    let stem = path.file_name()?.to_str()?.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (prefix, no) = head.rsplit_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (no.len() == count.len() && digits(no) && digits(count)).then_some((prefix, no, count))
}

/// Split `dir/<prefix>-<no>-of-<count>.gguf` into its prefix and number width.
fn split_file_name(path: &Path) -> Option<(String, usize)> {
    let (prefix, _, count) = split_file_parts(path)?;
    Some((prefix.to_string(), count.len()))
}

/// If `path` names a shard of a split model (`<name>-00002-of-00005.gguf`),
/// the path of the set's first shard in the same directory; `None` for a file
/// that is not named like a shard. Directory scans use this to treat a split
/// model as the one model it is.
pub fn first_shard_path(path: &Path) -> Option<PathBuf> {
    let (prefix, no, count) = split_file_parts(path)?;
    let first = format!("{prefix}-{:0>width$}-of-{count}.gguf", 1, width = no.len());
    Some(path.with_file_name(first))
}

impl GgufModel {
    /// Open a model. Passing any shard of a split set opens the whole set.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufReadError> {
        let path = path.as_ref();
        let first = open_shard(path)?;

        let Some((no, count, _)) = split_keys(&first.content)? else {
            return Self::from_shards(vec![first]);
        };
        if count <= 1 {
            return Self::from_shards(vec![first]);
        }
        if no >= count {
            return Err(GgufReadError::Split(format!(
                "{} is shard {no} of {count}",
                path.display()
            )));
        }
        let (prefix, width) = split_file_name(path).ok_or_else(|| {
            GgufReadError::Split(format!(
                "{} declares {count} shards but is not named <name>-NNNNN-of-NNNNN.gguf",
                path.display()
            ))
        })?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));

        let mut opened = Some(first);
        let mut shards = Vec::with_capacity(count as usize);
        for i in 0..count {
            if i == no {
                shards.extend(opened.take());
                continue;
            }
            let sibling = dir.join(format!(
                "{prefix}-{:0width$}-of-{count:0width$}.gguf",
                i + 1
            ));
            shards.push(open_shard(&sibling).map_err(|e| match e {
                GgufReadError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
                    GgufReadError::Split(format!("shard {} is missing", sibling.display()))
                }
                other => other,
            })?);
        }
        Self::from_shards(shards)
    }

    /// Check a set of shards for consistency and index their tensors.
    fn from_shards(mut shards: Vec<Shard>) -> Result<Self, GgufReadError> {
        if shards.len() > 1 {
            let count = shards.len() as u64;
            let mut declared_total = None;
            for (i, shard) in shards.iter().enumerate() {
                let (no, shard_count, total) = split_keys(&shard.content)?.ok_or_else(|| {
                    GgufReadError::Split(format!(
                        "{} is part of a split set but has no split.* metadata",
                        shard.path.display()
                    ))
                })?;
                if no != i as u64 || shard_count != count {
                    return Err(GgufReadError::Split(format!(
                        "{} declares shard {no} of {shard_count}, expected {i} of {count}",
                        shard.path.display()
                    )));
                }
                declared_total = declared_total.or(total);
            }
            if let Some(total) = declared_total {
                let actual: usize = shards.iter().map(|s| s.content.num_tensors()).sum();
                if i64::try_from(actual).ok() != Some(total) {
                    return Err(GgufReadError::Split(format!(
                        "the shards hold {actual} tensors but split.tensors.count says {total}"
                    )));
                }
            }
        }

        let mut owner = HashMap::new();
        for (i, shard) in shards.iter().enumerate() {
            for name in shard.content.tensor_names() {
                if owner.insert(name.to_string(), i).is_some() {
                    return Err(GgufReadError::DuplicateTensorName(name.to_string()));
                }
            }
        }
        shards.shrink_to_fit();
        Ok(Self { shards, owner })
    }

    /// The first shard's header: the model's metadata, and with a single file
    /// the whole content. `architecture()`, `tokenizer_data()` and
    /// [`crate::config::config_from_gguf`] read from this.
    pub fn metadata_content(&self) -> &GgufContent {
        &self.shards[0].content
    }

    /// Look up a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata_content().get_metadata(key)
    }

    /// The model architecture (`general.architecture`).
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_content().architecture()
    }

    /// Number of files in the model.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Paths of the files, in shard order.
    pub fn shard_paths(&self) -> impl Iterator<Item = &Path> {
        self.shards.iter().map(|s| s.path.as_path())
    }

    /// Number of tensors across all shards.
    pub fn num_tensors(&self) -> usize {
        self.owner.len()
    }

    /// All tensor names (in no particular order).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.owner.keys().map(String::as_str)
    }

    /// Tensor information by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let shard = *self.owner.get(name)?;
        self.shards[shard].content.get_tensor_info(name)
    }

    /// All tensors in on-disk order: by shard, then by offset. Reading in this
    /// order is sequential on disk.
    pub fn tensors_in_disk_order(&self) -> Vec<&TensorInfo> {
        self.shards
            .iter()
            .flat_map(|s| s.content.tensors_by_offset())
            .collect()
    }

    /// The raw bytes of a tensor, borrowed from the mapping.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], GgufReadError> {
        let shard = *self
            .owner
            .get(name)
            .ok_or_else(|| GgufReadError::TensorNotFound(name.to_string()))?;
        let shard = &self.shards[shard];
        let range = shard.content.tensor_data_range(name)?;
        let (start, end) = (
            usize::try_from(range.start).map_err(|_| GgufReadError::IntegerOverflow)?,
            usize::try_from(range.end).map_err(|_| GgufReadError::IntegerOverflow)?,
        );
        shard
            .map
            .get(start..end)
            .ok_or_else(|| GgufReadError::TensorOutOfBounds {
                name: name.to_string(),
                end: range.end,
                file_len: shard.map.len() as u64,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufBuilder;

    fn write_shard(
        dir: &Path,
        name: &str,
        split: Option<(u16, u16, i32)>,
        tensors: &[(&str, f32)],
    ) -> PathBuf {
        let mut b = GgufBuilder::with_model("llama", "m");
        if let Some((no, count, total)) = split {
            b.add_metadata(SPLIT_NO, MetadataValue::Uint16(no));
            b.add_metadata(SPLIT_COUNT, MetadataValue::Uint16(count));
            b.add_metadata(SPLIT_TENSORS_COUNT, MetadataValue::Int32(total));
        }
        for (n, v) in tensors {
            b.add_f32_tensor(*n, vec![4], vec![*v; 4]);
        }
        let path = dir.join(name);
        std::fs::write(&path, b.build_to_bytes().unwrap()).unwrap();
        path
    }

    #[test]
    fn single_file_serves_tensor_bytes_from_the_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_shard(dir.path(), "m.gguf", None, &[("a", 1.0), ("b", 2.0)]);
        let model = GgufModel::open(&path).unwrap();
        assert_eq!(model.num_shards(), 1);
        assert_eq!(model.num_tensors(), 2);
        assert_eq!(model.architecture(), Some("llama"));
        let b: Vec<f32> = model
            .tensor_data("b")
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(b, vec![2.0; 4]);
        assert!(matches!(
            model.tensor_data("nope"),
            Err(GgufReadError::TensorNotFound(_))
        ));
    }

    #[test]
    fn split_set_opens_from_any_shard_with_one_namespace() {
        let dir = tempfile::tempdir().unwrap();
        // The first shard carries metadata only, as published splits often do.
        let first = write_shard(dir.path(), "m-00001-of-00003.gguf", Some((0, 3, 3)), &[]);
        let second = write_shard(
            dir.path(),
            "m-00002-of-00003.gguf",
            Some((1, 3, 3)),
            &[("x", 3.0), ("y", 4.0)],
        );
        let _third = write_shard(
            dir.path(),
            "m-00003-of-00003.gguf",
            Some((2, 3, 3)),
            &[("z", 5.0)],
        );

        for path in [&first, &second] {
            let model = GgufModel::open(path).unwrap();
            assert_eq!(model.num_shards(), 3);
            assert_eq!(model.num_tensors(), 3);
            assert_eq!(model.architecture(), Some("llama"));
            for (name, value) in [("x", 3.0f32), ("y", 4.0), ("z", 5.0)] {
                let bytes = model.tensor_data(name).unwrap();
                assert_eq!(&bytes[..4], &value.to_le_bytes());
            }
            assert_eq!(model.tensors_in_disk_order().len(), 3);
        }
    }

    #[test]
    fn missing_or_inconsistent_shards_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let first = write_shard(dir.path(), "m-00001-of-00002.gguf", Some((0, 2, 1)), &[]);
        // Second shard absent.
        assert!(matches!(
            GgufModel::open(&first),
            Err(GgufReadError::Split(msg)) if msg.contains("missing")
        ));

        // Present, but the declared tensor total disagrees with what is stored.
        write_shard(
            dir.path(),
            "m-00002-of-00002.gguf",
            Some((1, 2, 1)),
            &[("a", 1.0), ("b", 2.0)],
        );
        assert!(matches!(
            GgufModel::open(&first),
            Err(GgufReadError::Split(msg)) if msg.contains("split.tensors.count")
        ));
    }

    #[test]
    fn a_tensor_in_two_shards_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let first = write_shard(
            dir.path(),
            "m-00001-of-00002.gguf",
            Some((0, 2, 2)),
            &[("a", 1.0)],
        );
        write_shard(
            dir.path(),
            "m-00002-of-00002.gguf",
            Some((1, 2, 2)),
            &[("a", 2.0)],
        );
        assert!(matches!(
            GgufModel::open(&first),
            Err(GgufReadError::DuplicateTensorName(_))
        ));
    }

    #[test]
    fn the_first_shard_is_found_from_any_shard() {
        let p = Path::new("/x/Model-UD-Q4-00003-of-00005.gguf");
        assert_eq!(
            first_shard_path(p),
            Some(PathBuf::from("/x/Model-UD-Q4-00001-of-00005.gguf"))
        );
        assert_eq!(first_shard_path(Path::new("/x/model.gguf")), None);
        assert_eq!(first_shard_path(Path::new("/x/m-1-of-0003.gguf")), None);
    }

    #[test]
    fn shard_file_names_are_parsed() {
        assert_eq!(
            split_file_name(Path::new("/x/Model-UD-Q4-00002-of-00005.gguf")),
            Some(("Model-UD-Q4".to_string(), 5))
        );
        assert_eq!(split_file_name(Path::new("model.gguf")), None);
        assert_eq!(split_file_name(Path::new("m-1-of-0003.gguf")), None);
    }
}
