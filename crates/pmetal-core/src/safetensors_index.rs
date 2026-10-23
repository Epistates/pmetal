//! The `model.safetensors.index.json` of a sharded checkpoint, read and written
//! in one place.
//!
//! An index maps every tensor name to the shard file that holds it. Reading one
//! is reading untrusted input: a crafted index can name `../../victim` and send
//! a loader out of the model directory. [`ShardIndex::parse`] refuses every
//! shard name that is not a plain relative path under the directory, so no
//! caller has to remember to.
//!
//! Writing is atomic: the index is written to a sibling file and renamed over
//! the target, so a reader never sees half an index.
//!
//! The safetensors *header* of a shard is read with the `safetensors` crate
//! (`SafeTensors::read_metadata`), not here; this module holds the pieces that
//! crate does not: the index file, the path rule and the dtype tag table.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The index file of a sharded safetensors checkpoint.
pub const INDEX_FILE: &str = "model.safetensors.index.json";

/// Whether `path` is a file path a repository (or an index file) may name:
/// relative, `/`-separated, no empty, `.` or `..` component, no backslash or
/// NUL. Anything else could point out of the directory it is joined onto.
pub fn is_safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains(['\\', '\0'])
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

/// The bits one element of a safetensors dtype tag takes (`"BF16"` is 16,
/// `"F4"` is 4), or `None` for a tag this table does not know.
pub fn dtype_bits(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F64" | "I64" | "U64" | "C64" => 64,
        "F32" | "I32" | "U32" => 32,
        "F16" | "BF16" | "I16" | "U16" => 16,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => 8,
        "F6_E2M3" | "F6_E3M2" => 6,
        "F4" => 4,
        _ => return None,
    })
}

/// Why an index could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// The file could not be read, or written.
    #[error("{}: {source}", path.display())]
    Io {
        /// The index file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The text is not JSON.
    #[error("weight index is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The JSON is not an index: no `weight_map` object, or an entry that is
    /// not a tensor name and a shard file name.
    #[error("{0}")]
    Malformed(String),
    /// A shard name that leaves the model directory.
    #[error("weight index names a shard outside the model directory: {0:?}")]
    ShardEscapes(String),
}

/// A parsed index: which shard holds each tensor.
///
/// Tensor names are in name order, so everything derived from an index (the
/// shard list, a rewritten index file) is the same on every run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShardIndex {
    /// The index's `metadata` object (`total_size`, ...), kept as written.
    pub metadata: serde_json::Map<String, serde_json::Value>,
    /// Tensor name to the shard file that holds it, relative to the model
    /// directory. Every value passed [`is_safe_relative_path`] when the index
    /// was parsed.
    pub weight_map: BTreeMap<String, String>,
}

impl ShardIndex {
    /// An index over `weight_map`, with `metadata.total_size` set to
    /// `total_size` bytes.
    pub fn new(weight_map: BTreeMap<String, String>, total_size: u64) -> Self {
        let mut metadata = serde_json::Map::new();
        metadata.insert("total_size".into(), total_size.into());
        Self {
            metadata,
            weight_map,
        }
    }

    /// Parse an index, refusing one that names a shard outside the model
    /// directory. An index with an empty `weight_map` parses; check
    /// [`shards`](Self::shards) if a caller needs at least one.
    pub fn parse(json: &[u8]) -> Result<Self, IndexError> {
        let value: serde_json::Value = serde_json::from_slice(json)?;
        let map = value
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| IndexError::Malformed("weight index has no `weight_map`".into()))?;
        let mut weight_map = BTreeMap::new();
        for (tensor, shard) in map {
            let shard = shard.as_str().ok_or_else(|| {
                IndexError::Malformed(format!(
                    "weight index: the shard of `{tensor}` is not a file name"
                ))
            })?;
            if !is_safe_relative_path(shard) {
                return Err(IndexError::ShardEscapes(shard.to_owned()));
            }
            weight_map.insert(tensor.clone(), shard.to_owned());
        }
        let metadata = value
            .get("metadata")
            .and_then(|m| m.as_object())
            .cloned()
            .unwrap_or_default();
        Ok(Self {
            metadata,
            weight_map,
        })
    }

    /// Read and parse the index file at `path`.
    pub fn read(path: &Path) -> Result<Self, IndexError> {
        let bytes = std::fs::read(path).map_err(|source| IndexError::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&bytes)
    }

    /// The shard files the index names, each once, in name order.
    pub fn shards(&self) -> Vec<&str> {
        let mut shards: Vec<&str> = self.weight_map.values().map(String::as_str).collect();
        shards.sort_unstable();
        shards.dedup();
        shards
    }

    /// The shard that holds `tensor`.
    pub fn shard_of(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }

    /// The index as the JSON `transformers` writes, pretty-printed.
    pub fn to_json(&self) -> String {
        let value = serde_json::json!({
            "metadata": self.metadata,
            "weight_map": self.weight_map,
        });
        serde_json::to_string_pretty(&value).expect("a map of strings serializes")
    }

    /// Write the index to `path` atomically: to a sibling `.partial` file,
    /// synced, then renamed over `path`. A failed write removes the partial
    /// file and leaves `path` as it was.
    pub fn write(&self, path: &Path) -> Result<(), IndexError> {
        let mut partial = path.as_os_str().to_owned();
        partial.push(".partial");
        let partial = PathBuf::from(partial);
        let result = (|| -> std::io::Result<()> {
            let mut file = std::fs::File::create(&partial)?;
            file.write_all(self.to_json().as_bytes())?;
            file.sync_all()?;
            std::fs::rename(&partial, path)
        })();
        if let Err(source) = result {
            let _ = std::fs::remove_file(&partial);
            return Err(IndexError::Io {
                path: path.to_owned(),
                source,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> ShardIndex {
        let map = [
            ("b.weight", "model-00002-of-00002.safetensors"),
            ("a.weight", "model-00001-of-00002.safetensors"),
            ("a.bias", "model-00001-of-00002.safetensors"),
        ]
        .into_iter()
        .map(|(t, s)| (t.to_owned(), s.to_owned()))
        .collect();
        ShardIndex::new(map, 1234)
    }

    #[test]
    fn paths_that_leave_the_directory_are_unsafe() {
        for ok in ["model.safetensors", "text/model-00001.safetensors", "a.b/c"] {
            assert!(is_safe_relative_path(ok), "{ok}");
        }
        for bad in [
            "",
            "/etc/passwd",
            "../x",
            "a/../../x",
            "a//b",
            "./a",
            "a/.",
            "a\\b",
            "a\0b",
        ] {
            assert!(!is_safe_relative_path(bad), "{bad:?}");
        }
    }

    #[test]
    fn a_crafted_index_is_refused_whatever_it_names() {
        for shard in ["../../victim", "/etc/passwd", "a/../../b", "x\\y", ""] {
            let json = serde_json::json!({ "weight_map": { "t": shard } }).to_string();
            let err = ShardIndex::parse(json.as_bytes()).unwrap_err();
            assert!(
                matches!(err, IndexError::ShardEscapes(_)),
                "{shard:?}: {err}"
            );
            assert!(err.to_string().contains("outside"), "{err}");
        }
    }

    #[test]
    fn what_is_not_an_index_is_an_error() {
        assert!(matches!(
            ShardIndex::parse(b"not json"),
            Err(IndexError::Json(_))
        ));
        assert!(matches!(
            ShardIndex::parse(b"{}"),
            Err(IndexError::Malformed(_))
        ));
        assert!(matches!(
            ShardIndex::parse(br#"{"weight_map": []}"#),
            Err(IndexError::Malformed(_))
        ));
        assert!(matches!(
            ShardIndex::parse(br#"{"weight_map": {"t": 3}}"#),
            Err(IndexError::Malformed(_))
        ));
    }

    #[test]
    fn shards_are_listed_once_in_name_order() {
        let shards = ShardIndex::parse(br#"{"weight_map":{"a":"s2","b":"s1","c":"s2"}}"#)
            .unwrap()
            .shards()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(shards, ["s1", "s2"]);
        assert_eq!(
            index().shard_of("a.bias"),
            Some("model-00001-of-00002.safetensors")
        );
        assert_eq!(index().shard_of("missing"), None);
    }

    #[test]
    fn an_index_round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(INDEX_FILE);
        let written = index();
        written.write(&path).unwrap();
        let read = ShardIndex::read(&path).unwrap();
        assert_eq!(read, written);
        assert_eq!(read.metadata["total_size"], 1234);
        // No partial file is left behind, and the text is the same on every
        // write.
        assert!(!dir.path().join(format!("{INDEX_FILE}.partial")).exists());
        let first = std::fs::read(&path).unwrap();
        written.write(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), first);
    }

    #[test]
    fn a_failed_write_leaves_the_old_index_and_no_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(INDEX_FILE);
        index().write(&path).unwrap();
        let before = std::fs::read(&path).unwrap();
        // The target is a directory: the rename fails.
        let blocked = dir.path().join("blocked");
        std::fs::create_dir(&blocked).unwrap();
        let err = index().write(&blocked).unwrap_err();
        assert!(matches!(err, IndexError::Io { .. }), "{err}");
        assert!(!dir.path().join("blocked.partial").exists());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn a_missing_index_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = ShardIndex::read(&dir.path().join(INDEX_FILE)).unwrap_err();
        assert!(err.to_string().contains(INDEX_FILE), "{err}");
    }

    #[test]
    fn dtype_tags_give_their_widths() {
        assert_eq!(dtype_bits("BF16"), Some(16));
        assert_eq!(dtype_bits("F8_E4M3"), Some(8));
        assert_eq!(dtype_bits("F4"), Some(4));
        assert_eq!(dtype_bits("F8_E4M3FNUZ"), Some(8));
        assert_eq!(dtype_bits("NOPE"), None);
    }
}
