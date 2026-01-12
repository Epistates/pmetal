//! Local cache management.

use pmetal_core::Result;
use std::path::PathBuf;

/// Get the default cache directory.
pub fn cache_dir() -> PathBuf {
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
    for entry in walkdir::WalkDir::new(&cache) {
        if let Ok(entry) = entry {
            if entry.file_type().is_file() {
                if let Ok(metadata) = entry.metadata() {
                    size += metadata.len();
                }
            }
        }
    }
    Ok(size)
}

// Note: walkdir dependency would need to be added to Cargo.toml
