//! Model and dataset path resolution.
//!
//! Provides canonical `resolve_model_path()` used by all consumers (CLI, GUI,
//! Python bindings, orchestrator) to resolve a HuggingFace model ID or local
//! path to a directory on disk, downloading if necessary.

use pmetal_core::SecretString;
use std::path::{Path, PathBuf};

use crate::download::{DownloadOptions, download_model_with};
use crate::select::WeightFormat;

/// Check if a string looks like a HuggingFace model ID (e.g., `"org/model"`)
/// rather than a local filesystem path.
pub fn is_hf_id(s: &str) -> bool {
    !s.starts_with('/') && !s.starts_with('.') && s.contains('/')
}

/// Split a trailing `:QUANT` selector off a model reference:
/// `"org/model-GGUF:Q4_K_M"` is `("org/model-GGUF", Some("Q4_K_M"))`. A string
/// without one, or whose tail looks like a path, is returned whole.
pub fn split_quant_selector(reference: &str) -> (&str, Option<&str>) {
    match reference.rsplit_once(':') {
        Some((id, quant))
            if !quant.is_empty()
                && !quant.contains(['/', '\\'])
                && is_hf_id(id)
                && !id.contains(':') =>
        {
            (id, Some(quant))
        }
        _ => (reference, None),
    }
}

/// Resolve a model identifier to a local directory path.
///
/// If `model_id` looks like a HuggingFace ID and the path does not already
/// exist on disk, the model is downloaded via [`crate::download_model`] using
/// the supplied `revision` and `token`. Otherwise the string is treated as a
/// local path and the optional arguments are ignored.
///
/// A `:QUANT` suffix (`org/model-GGUF:Q4_K_M`) selects one GGUF quantization
/// of a multi-quant repository.
pub async fn resolve_model_path(
    model_id: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> pmetal_core::Result<PathBuf> {
    resolve_model_path_with(model_id, &DownloadOptions::new(revision, token)).await
}

/// [`resolve_model_path`] with explicit [`DownloadOptions`].
pub async fn resolve_model_path_with(
    model_id: &str,
    options: &DownloadOptions,
) -> pmetal_core::Result<PathBuf> {
    if Path::new(model_id).exists() {
        return Ok(PathBuf::from(model_id));
    }
    let (id, quant) = split_quant_selector(model_id);
    if !is_hf_id(id) {
        return Ok(PathBuf::from(model_id));
    }
    let mut options = options.clone();
    if let Some(quant) = quant {
        options.gguf_quant = Some(quant.to_owned());
        if options.weights == WeightFormat::Auto {
            options.weights = WeightFormat::Gguf;
        }
    }
    download_model_with(id, &options).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quant_selector_is_split_off_a_hub_id() {
        assert_eq!(
            split_quant_selector("unsloth/Qwen3-0.6B-GGUF:Q4_K_M"),
            ("unsloth/Qwen3-0.6B-GGUF", Some("Q4_K_M"))
        );
        assert_eq!(split_quant_selector("org/model"), ("org/model", None));
        // Not selectors: a path with a colon, an empty tail, a bare name.
        assert_eq!(split_quant_selector("org/model:"), ("org/model:", None));
        assert_eq!(split_quant_selector("./a/b:c/d"), ("./a/b:c/d", None));
        assert_eq!(split_quant_selector("model:Q4"), ("model:Q4", None));
        assert_eq!(split_quant_selector("a:b/c:d"), ("a:b/c:d", None));
    }

    #[test]
    fn hub_ids_are_told_from_paths() {
        assert!(is_hf_id("Qwen/Qwen3-0.6B"));
        assert!(!is_hf_id("/abs/path"));
        assert!(!is_hf_id("./rel/path"));
        assert!(!is_hf_id("plain"));
    }

    #[tokio::test]
    async fn an_existing_directory_wins_over_a_hub_id() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("org").join("model");
        std::fs::create_dir_all(&nested).unwrap();
        let resolved = resolve_model_path(nested.to_str().unwrap(), None, None)
            .await
            .unwrap();
        assert_eq!(resolved, nested);
    }
}
