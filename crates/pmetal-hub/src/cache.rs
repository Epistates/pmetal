//! Local cache management.

use pmetal_core::Result;
use std::path::PathBuf;

/// Get the HuggingFace home directory, resolved the way the Python
/// `huggingface_hub` library does it.
///
/// Resolution order:
/// 1. `HF_HOME`
/// 2. `$XDG_CACHE_HOME/huggingface`
/// 3. `~/.cache/huggingface` — default on all platforms
fn hf_home() -> PathBuf {
    if let Some(home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(home);
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("huggingface");
    }
    dirs::home_dir()
        .unwrap_or_default()
        .join(".cache")
        .join("huggingface")
}

/// Get the HuggingFace hub cache directory.
///
/// Every download goes through a client pointed at this directory, so the
/// cache lookups here and the downloads agree by construction.
///
/// Resolution order:
/// 1. `HF_HUB_CACHE` — direct override for the hub cache directory
/// 2. `HUGGINGFACE_HUB_CACHE` — its legacy name
/// 3. `<HF home>/hub`
pub fn cache_dir() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("hub"))
}

/// Get the HuggingFace datasets cache directory.
///
/// Resolution order:
/// 1. `HF_DATASETS_CACHE` — direct override
/// 2. `<HF home>/datasets`
pub fn datasets_cache_dir() -> PathBuf {
    std::env::var_os("HF_DATASETS_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("datasets"))
}

/// Get the pmetal-specific cache directory (for non-HF local state).
pub fn pmetal_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .map(|p| p.join("pmetal"))
        .unwrap_or_else(|| PathBuf::from(".cache/pmetal"))
}

/// Evict a single model from the local HuggingFace hub cache.
///
/// Removes the cache directory for `repo_id` (e.g., `"Qwen/Qwen3-0.6B"`).
/// The directory is resolved as `<cache_dir>/models--<org>--<name>`, matching
/// the layout used by `huggingface_hub` and the `hf-hub` crate.
///
/// Returns `Ok(())` if the directory did not exist (idempotent).
pub fn evict_model(repo_id: &str) -> Result<()> {
    // Convert "org/name" → "models--org--name"
    let dir_name = format!("models--{}", repo_id.replace('/', "--"));
    let model_cache_dir = cache_dir().join(dir_name);
    if model_cache_dir.exists() {
        std::fs::remove_dir_all(&model_cache_dir)?;
    }
    Ok(())
}

/// Check if a model is already cached locally and return its path.
///
/// Looks for `<cache_dir>/models--<org>--<name>/snapshots/<hash>/config.json`
/// which is the HF hub cache layout. Returns the snapshot directory if found.
/// This is a fast local-only check — no network calls.
pub fn find_cached_model(repo_id: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", repo_id.replace('/', "--"));
    let model_cache_dir = cache_dir().join(dir_name);
    let snapshots_dir = model_cache_dir.join("snapshots");

    if !snapshots_dir.is_dir() {
        return None;
    }

    // Find the most recent snapshot that contains config.json
    let mut snapshots: Vec<_> = std::fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    // Sort by modification time descending (most recent first)
    snapshots.sort_by(|a, b| {
        let t_a = a
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let t_b = b
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        t_b.cmp(&t_a)
    });

    for snap in snapshots {
        let snap_path = snap.path();
        // A valid model snapshot must have config.json
        if snap_path.join("config.json").exists() && shards_complete(&snap_path) {
            return Some(snap_path);
        }
    }

    None
}

/// Whether every shard a snapshot's `model.safetensors.index.json` names is
/// present. A snapshot without an index has nothing to check.
///
/// ⚠️ A download interrupted after the index but before the last shard (a
/// 429 from the Hub does it) leaves a snapshot with `config.json` that is
/// missing weights. Treating that as cached made every retry return early, so
/// the gap could never be filled, and the load then failed on the missing
/// shard. A missing blob behind a snapshot symlink reads as absent here.
fn shards_complete(snapshot: &std::path::Path) -> bool {
    let Ok(index) = std::fs::read_to_string(snapshot.join("model.safetensors.index.json")) else {
        return true;
    };
    let Ok(index) = serde_json::from_str::<serde_json::Value>(&index) else {
        return false;
    };
    let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) else {
        return false;
    };
    weight_map
        .values()
        .filter_map(|shard| shard.as_str())
        .all(|shard| snapshot.join(shard).exists())
}

/// Check if a dataset is already cached locally and return its path.
///
/// Same as `find_cached_model` but for datasets.
pub fn find_cached_dataset(repo_id: &str) -> Option<PathBuf> {
    let dir_name = format!("datasets--{}", repo_id.replace('/', "--"));
    let cache_root = cache_dir();
    let dataset_cache_dir = cache_root.join(&dir_name);
    let snapshots_dir = dataset_cache_dir.join("snapshots");

    if !snapshots_dir.is_dir() {
        return None;
    }

    let mut snapshots: Vec<_> = std::fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    snapshots.sort_by(|a, b| {
        let t_a = a
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let t_b = b
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        t_b.cmp(&t_a)
    });

    for snap in snapshots {
        let snap_path = snap.path();
        // A valid dataset snapshot should have at least one file
        if std::fs::read_dir(&snap_path)
            .ok()
            .map(|rd| rd.count() > 0)
            .unwrap_or(false)
        {
            return Some(snap_path);
        }
    }

    None
}

/// Clear the model cache.
pub fn clear_cache() -> Result<()> {
    let cache = cache_dir();
    if cache.exists() {
        std::fs::remove_dir_all(&cache)?;
    }
    Ok(())
}

/// Get cache size in bytes.
pub fn cache_size() -> Result<u64> {
    let cache = cache_dir();
    if !cache.exists() {
        return Ok(0);
    }

    let mut size = 0u64;
    for entry in walkdir::WalkDir::new(&cache).into_iter().flatten() {
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                size += metadata.len();
            }
        }
    }
    Ok(size)
}

// Note: walkdir dependency would need to be added to Cargo.toml

#[cfg(test)]
mod tests {
    use super::shards_complete;

    fn snapshot(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pmetal-hub-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a.w": "model-00001-of-00002.safetensors",
                               "b.w": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        dir
    }

    /// The shape a 429 left behind: the index and the first shard, with the
    /// second shard's snapshot symlink pointing at a blob that never arrived.
    #[test]
    fn a_snapshot_missing_a_shard_is_not_cached() {
        let dir = snapshot("partial");
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), b"x").unwrap();
        std::os::unix::fs::symlink(
            dir.join("never-downloaded-blob"),
            dir.join("model-00002-of-00002.safetensors"),
        )
        .unwrap();
        assert!(!shards_complete(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_snapshot_with_every_shard_is_cached() {
        let dir = snapshot("complete");
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), b"x").unwrap();
        assert!(shards_complete(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_single_file_snapshot_has_nothing_to_check() {
        let dir = snapshot("single");
        std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
        assert!(shards_complete(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
