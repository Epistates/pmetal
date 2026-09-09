use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use serde::{Serialize, de::DeserializeOwned};
use tracing::{debug, warn};

fn load_map<V>(path: &Path) -> HashMap<String, V>
where
    V: DeserializeOwned,
{
    let Ok(contents) = fs::read_to_string(path) else {
        return HashMap::new();
    };

    match serde_json::from_str::<HashMap<String, V>>(&contents) {
        Ok(cache) => cache,
        Err(error) => {
            warn!(
                "Failed to parse persistent kernel cache {}: {}",
                path.display(),
                error
            );
            HashMap::new()
        }
    }
}

fn write_map<V>(path: &Path, map: &HashMap<String, V>)
where
    V: Serialize,
{
    let Some(parent) = path.parent() else {
        return;
    };
    if let Err(error) = fs::create_dir_all(parent) {
        warn!(
            "Failed to create persistent kernel cache directory {}: {}",
            parent.display(),
            error
        );
        return;
    }

    let tmp_path = path.with_extension("json.tmp");
    match serde_json::to_string_pretty(map) {
        Ok(json) => {
            if let Err(error) = fs::write(&tmp_path, json) {
                warn!(
                    "Failed to write persistent kernel cache {}: {}",
                    tmp_path.display(),
                    error
                );
                return;
            }
            if let Err(error) = fs::rename(&tmp_path, path) {
                warn!(
                    "Failed to finalize persistent kernel cache {}: {}",
                    path.display(),
                    error
                );
                let _ = fs::remove_file(&tmp_path);
            }
        }
        Err(error) => {
            warn!(
                "Failed to serialize persistent kernel cache {}: {}",
                path.display(),
                error
            );
        }
    }
}

/// Bump whenever the *meaning* of a cached choice changes: a new selection
/// criterion, a fixed backend, an added or removed candidate.
///
/// A cached entry records a conclusion, not the reasoning that produced it, so
/// entries written by older logic are not merely stale — they are wrong, and
/// they are sticky: the cache short-circuits benchmarking entirely, so a fix to
/// the selection rule has no effect on any machine that already ran the old
/// one. Epoch 2 is the first release whose attention-backend selection judges a
/// candidate against the reference's own scale instead of an absolute 0.1;
/// epoch-1 files accepted backends that returned all zeros for models whose
/// activations sit far below unit scale.
#[cfg_attr(
    test,
    expect(dead_code, reason = "tests never touch the on-disk cache")
)]
const CACHE_EPOCH: u32 = 2;

fn default_cache_path(filename: &str) -> Option<PathBuf> {
    #[cfg(test)]
    {
        let _ = filename;
        None
    }

    #[cfg(not(test))]
    {
        dirs::cache_dir().map(|dir| {
            dir.join("pmetal")
                .join(format!("mlx-kernels-v{CACHE_EPOCH}"))
                .join(filename)
        })
    }
}

pub(crate) struct PersistentChoiceCache<V> {
    path: Option<PathBuf>,
    map: Mutex<HashMap<String, V>>,
}

impl<V> PersistentChoiceCache<V>
where
    V: Clone + Serialize + DeserializeOwned,
{
    pub(crate) fn new(filename: &str) -> Self {
        Self::with_optional_path(default_cache_path(filename))
    }

    fn with_optional_path(path: Option<PathBuf>) -> Self {
        let map = path.as_deref().map(load_map::<V>).unwrap_or_default();

        if let Some(path) = &path {
            debug!(
                "Loaded {} entries from persistent kernel cache {}",
                map.len(),
                path.display()
            );
        }

        Self {
            path,
            map: Mutex::new(map),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_path(path: PathBuf) -> Self {
        Self::with_optional_path(Some(path))
    }

    pub(crate) fn get(&self, key: &str) -> Option<V> {
        self.map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .cloned()
    }

    pub(crate) fn insert(&self, key: String, value: V) {
        let snapshot = {
            let mut map = self
                .map
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            map.insert(key, value);
            map.clone()
        };

        if let Some(path) = &self.path {
            write_map(path, &snapshot);
        }
    }

    #[cfg(test)]
    pub(crate) fn clear(&self) {
        self.map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();

        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    enum Choice {
        Fast,
        Precise,
    }

    #[test]
    fn test_persistent_choice_cache_roundtrip() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("choices.json");

        let cache = PersistentChoiceCache::<Choice>::with_path(path.clone());
        assert_eq!(cache.get("a"), None);
        cache.insert("a".to_string(), Choice::Fast);
        assert_eq!(cache.get("a"), Some(Choice::Fast));

        let cache = PersistentChoiceCache::<Choice>::with_path(path.clone());
        assert_eq!(cache.get("a"), Some(Choice::Fast));

        cache.clear();
        assert!(!path.exists());
    }
}
