//! The crate's error type and adapter-file IO.

use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;

use pmetal_bridge::compat::{Array, Exception};

/// Error type for LoRA operations.
#[derive(Debug, thiserror::Error)]
pub enum LoraError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// IO error.
    #[error("IO error: {0}")]
    Io(String),
    /// Shape mismatch error.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    /// Invalid state error.
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Load a full safetensors shard into a map of tensor name to array.
pub fn load_safetensors_map(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, LoraError> {
    let path = path.as_ref();
    let path_str = path.to_str().ok_or_else(|| {
        LoraError::Mlx(Exception::custom(format!(
            "non-UTF-8 safetensors path: {}",
            path.display()
        )))
    })?;
    let entries =
        pmetal_bridge::inline_array::load_safetensors_shard(path_str).ok_or_else(|| {
            LoraError::Mlx(Exception::custom(format!(
                "failed to load safetensors shard: {}",
                path.display()
            )))
        })?;
    Ok(entries.into_iter().collect())
}

/// Save a named array map to a safetensors file.
pub fn save_safetensors_map<K>(
    path: impl AsRef<Path>,
    params: &HashMap<K, Array>,
) -> Result<(), LoraError>
where
    K: AsRef<str> + Eq + Hash,
{
    let path = path.as_ref();
    let path_str = path.to_str().ok_or_else(|| {
        LoraError::Mlx(Exception::custom(format!(
            "non-UTF-8 safetensors path: {}",
            path.display()
        )))
    })?;
    let entries: Vec<(&str, &Array)> = params
        .iter()
        .map(|(key, value)| (key.as_ref(), value))
        .collect();
    Array::save_safetensors(path_str, &entries);
    Ok(())
}
