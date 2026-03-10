//! Local cache management.

use pmetal_core::Result;
use std::path::PathBuf;

/// Get the HuggingFace hub cache directory.
///
/// Respects `HF_HOME` and `HF_HUB_CACHE` environment variables, matching
/// the behavior of the `hf-hub` crate and the Python `huggingface_hub` library.
///
/// Resolution order:
/// 1. `HF_HUB_CACHE` — direct override for the hub cache directory
/// 2. `HF_HOME/hub` — if `HF_HOME` is set
/// 3. `~/.cache/huggingface/hub` — default on all platforms
pub fn cache_dir() -> PathBuf {
    // HF_HUB_CACHE takes highest priority (direct path to hub cache)
    if let Ok(hub_cache) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(hub_cache);
    }
    // Use hf-hub crate's Cache::from_env() which handles HF_HOME
    hf_hub::Cache::from_env().path().clone()
}

/// Get the HuggingFace datasets cache directory.
///
/// Resolution order:
/// 1. `HF_DATASETS_CACHE` — direct override
/// 2. `HF_HOME/datasets` — if `HF_HOME` is set
/// 3. `~/.cache/huggingface/datasets` — default on all platforms
pub fn datasets_cache_dir() -> PathBuf {
    if let Ok(ds_cache) = std::env::var("HF_DATASETS_CACHE") {
        return PathBuf::from(ds_cache);
    }
    // hf_hub::Cache path points to <root>/hub, go up one level and add datasets
    let mut path = hf_hub::Cache::from_env().path().clone();
    path.pop(); // remove "hub"
    path.push("datasets");
    path
}

/// Get the pmetal-specific cache directory (for non-HF local state).
pub fn pmetal_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .map(|p| p.join("pmetal"))
        .unwrap_or_else(|| PathBuf::from(".cache/pmetal"))
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
