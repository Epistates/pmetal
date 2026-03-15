//! Model and dataset downloading from HuggingFace Hub.

use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use pmetal_core::{Result, SecretString};
use std::path::PathBuf;

/// Build API with optional token authentication.
fn build_api(token: Option<&SecretString>) -> Result<Api> {
    let mut builder = ApiBuilder::from_env();

    if let Some(secret) = token {
        builder = builder.with_token(Some(secret.expose_secret().to_string()));
    }

    builder
        .build()
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))
}

/// Files to skip during full-repo download (large binaries, metadata, etc.).
const SKIP_EXTENSIONS: &[&str] = &[
    ".bin",     // PyTorch weights — we use safetensors
    ".pt",      // PyTorch checkpoint
    ".pth",     // PyTorch checkpoint
    ".onnx",    // ONNX export
    ".ot",      // ONNX export
    ".msgpack", // Flax weights
    ".h5",      // TensorFlow/Keras weights
    ".pb",      // TensorFlow protobuf
    ".tflite",  // TensorFlow Lite
    ".gguf",    // GGUF quantized (download separately if needed)
    ".zip",     // Archives
    ".tar",     // Archives
    ".gz",      // Archives (except .json.gz which is handled)
];

/// Files to skip by exact name.
const SKIP_FILES: &[&str] = &[
    ".gitattributes",
    "flax_model.msgpack",
    "tf_model.h5",
    "pytorch_model.bin",
    "rust_model.ot",
];

/// Dataset files worth downloading for local consumption.
const DATASET_EXTENSIONS: &[&str] = &[
    ".parquet",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".arrow",
];

/// Download a model from HuggingFace Hub.
///
/// Lists all files in the repo via `info()` and downloads everything needed:
/// configs, tokenizer files, safetensors weights, and any other small metadata.
/// Skips PyTorch `.bin`, ONNX, TensorFlow, and other non-safetensors weight formats.
///
/// # Arguments
/// * `model_id` - Model identifier (e.g., "meta-llama/Llama-3.2-1B")
/// * `revision` - Optional revision/branch (e.g., "main")
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_model(
    model_id: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<PathBuf> {
    let api = build_api(token)?;

    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        )),
        None => api.model(model_id.to_string()),
    };

    // List all files in the repository
    let repo_info = repo
        .info()
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(format!("Failed to list repo files: {e}")))?;

    let all_files: Vec<&str> = repo_info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .collect();

    tracing::info!(
        "Found {} files in {}, downloading...",
        all_files.len(),
        model_id
    );

    // Partition files into those we want and those we skip
    let mut model_dir: Option<PathBuf> = None;
    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut failures = Vec::new();

    for filename in &all_files {
        // Skip files by extension
        if SKIP_EXTENSIONS.iter().any(|ext| filename.ends_with(ext)) {
            tracing::debug!("Skipping {} (excluded format)", filename);
            skipped += 1;
            continue;
        }

        // Skip files by exact name
        #[allow(clippy::manual_contains)]
        if SKIP_FILES.iter().any(|f| *filename == *f) {
            tracing::debug!("Skipping {} (excluded file)", filename);
            skipped += 1;
            continue;
        }

        // Skip directories / hidden files
        if filename.starts_with('.') {
            skipped += 1;
            continue;
        }

        // Download the file
        match repo.get(filename).await {
            Ok(path) => {
                tracing::info!("  {}", filename);
                downloaded += 1;
                // Capture model directory from the first downloaded file
                if model_dir.is_none() {
                    model_dir = path.parent().map(PathBuf::from);
                }
            }
            Err(e) => {
                failures.push(format!("{filename}: {e}"));
            }
        }
    }

    tracing::info!(
        "Downloaded {} files, skipped {} ({})",
        downloaded,
        skipped,
        model_id
    );

    if downloaded == 0 {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "No files downloaded for {}",
            model_id
        )));
    }

    if !failures.is_empty() {
        let preview = failures
            .into_iter()
            .take(10)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(pmetal_core::PMetalError::Hub(format!(
            "Download incomplete for {}: {}",
            model_id, preview
        )));
    }

    model_dir.ok_or_else(|| pmetal_core::PMetalError::Hub(format!("No files downloaded for {}", model_id)))
}

/// Download a specific file from a model repository.
///
/// # Arguments
/// * `model_id` - Model identifier
/// * `filename` - File to download
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_file(
    model_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<PathBuf> {
    let api = build_api(token)?;

    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        )),
        None => api.model(model_id.to_string()),
    };

    repo.get(filename)
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))
}

/// Download all safetensors files for a model.
///
/// # Arguments
/// * `model_id` - Model identifier
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_safetensors(
    model_id: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<Vec<PathBuf>> {
    let api = build_api(token)?;

    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        )),
        None => api.model(model_id.to_string()),
    };

    // Try to get model.safetensors first (single file models)
    if let Ok(path) = repo.get("model.safetensors").await {
        return Ok(vec![path]);
    }

    // Otherwise, get the model.safetensors.index.json for sharded models
    let index_path = repo
        .get("model.safetensors.index.json")
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))?;

    // Parse index to get shard filenames
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)
        .map_err(|e| pmetal_core::PMetalError::Serialization(e.to_string()))?;

    let mut paths = Vec::new();
    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
        let mut filenames: std::collections::HashSet<String> = std::collections::HashSet::new();
        for filename in weight_map.values() {
            if let Some(f) = filename.as_str() {
                filenames.insert(f.to_string());
            }
        }

        for filename in filenames {
            let path = repo
                .get(&filename)
                .await
                .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))?;
            paths.push(path);
        }
    }

    if paths.is_empty() {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "No safetensor shards found in index for {}",
            model_id
        )));
    }

    Ok(paths)
}

/// Download a dataset from HuggingFace Hub.
///
/// Returns the path to the dataset directory containing downloaded files.
///
/// # Arguments
/// * `dataset_id` - Dataset identifier (e.g., "tatsu-lab/alpaca")
/// * `split` - Optional dataset split name; when set, files whose path
///   contains this substring are also downloaded regardless of extension.
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_dataset(
    dataset_id: &str,
    split: Option<&str>,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<PathBuf> {
    let api = build_api(token)?;

    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            dataset_id.to_string(),
            RepoType::Dataset,
            rev.to_string(),
        )),
        None => api.repo(Repo::new(dataset_id.to_string(), RepoType::Dataset)),
    };

    let repo_info = repo
        .info()
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(format!("Failed to list repo files: {e}")))?;

    let mut downloaded_paths = Vec::new();
    let mut failures = Vec::new();

    for sibling in &repo_info.siblings {
        let filename = sibling.rfilename.as_str();
        if filename.starts_with('.') {
            continue;
        }

        let include = filename == "README.md"
            || DATASET_EXTENSIONS.iter().any(|ext| filename.ends_with(ext))
            || split.is_some_and(|s| filename.contains(s));
        if !include {
            continue;
        }

        match repo.get(filename).await {
            Ok(path) => downloaded_paths.push(path),
            Err(err) => failures.push(format!("{filename}: {err}")),
        }
    }

    if downloaded_paths.is_empty() {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "No dataset files downloaded for {}",
            dataset_id
        )));
    }

    if !failures.is_empty() {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "Dataset download incomplete for {}: {}",
            dataset_id,
            failures.into_iter().take(10).collect::<Vec<_>>().join(", ")
        )));
    }

    Ok(downloaded_paths[0]
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".")))
}

/// Download dataset Parquet files from HuggingFace Hub.
///
/// Most HuggingFace datasets are stored as Parquet files in the `data/` directory.
/// This function downloads all Parquet files for a given split.
///
/// # Arguments
/// * `dataset_id` - Dataset identifier (e.g., "tatsu-lab/alpaca")
/// * `split` - Dataset split (e.g., "train", "test")
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
///
/// # Returns
/// Vector of paths to downloaded Parquet files
pub async fn download_dataset_parquet(
    dataset_id: &str,
    split: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<Vec<PathBuf>> {
    let api = build_api(token)?;

    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            dataset_id.to_string(),
            RepoType::Dataset,
            rev.to_string(),
        )),
        None => api.repo(Repo::new(dataset_id.to_string(), RepoType::Dataset)),
    };

    let mut paths = Vec::new();

    // Common patterns for parquet files in HuggingFace datasets:
    // 1. data/{split}-00000-of-00001.parquet (sharded)
    // 2. data/{split}.parquet (single file)
    // 3. default/{split}/0000.parquet (auto-converted format)
    // 4. {split}/train-00000-of-00001.parquet (split directory)

    // Try single file patterns
    let single_patterns = [
        format!("data/{}.parquet", split),
        format!("default/{}/0000.parquet", split),
        format!("{}/{}-00000-of-00001.parquet", split, split),
        format!("{}.parquet", split),
    ];

    for pattern in &single_patterns {
        if let Ok(path) = repo.get(pattern).await {
            paths.push(path);
            return Ok(paths);
        }
    }

    // Try numbered shards in common locations
    let shard_prefixes = [
        format!("data/{}", split),
        format!("default/{}", split),
        split.to_string(),
    ];

    for prefix in &shard_prefixes {
        for i in 0..100 {
            // Try common shard naming patterns
            let patterns = [
                format!("{}-{:05}-of-{:05}.parquet", prefix, i, 1),
                format!("{}/{:04}.parquet", prefix, i),
                format!("{}-{:05}.parquet", prefix, i),
            ];

            let mut found_in_batch = false;
            for pattern in &patterns {
                if let Ok(path) = repo.get(pattern).await {
                    paths.push(path);
                    found_in_batch = true;
                    break;
                }
            }

            // For numbered shards, stop after finding some and then getting no more
            if i > 0 && !found_in_batch && !paths.is_empty() {
                return Ok(paths);
            }
        }

        if !paths.is_empty() {
            return Ok(paths);
        }
    }

    // Try using the parquet conversion branch which HuggingFace provides
    let parquet_repo = api.repo(Repo::with_revision(
        dataset_id.to_string(),
        RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    ));

    // Try common paths in the parquet branch
    let parquet_patterns = [
        format!("default/{}/0000.parquet", split),
        format!("{}/0000.parquet", split),
        format!("default/{split}-00000-of-00001.parquet"),
    ];

    for pattern in &parquet_patterns {
        if let Ok(path) = parquet_repo.get(pattern).await {
            paths.push(path);
            return Ok(paths);
        }
    }

    if paths.is_empty() {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "No parquet files found for split '{}' in dataset '{}'. \
            Try checking the dataset page for available files.",
            split, dataset_id
        )));
    }

    Ok(paths)
}

/// Download a specific file from a dataset repository.
///
/// # Arguments
/// * `dataset_id` - Dataset identifier
/// * `filename` - File to download
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_dataset_file(
    dataset_id: &str,
    filename: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<PathBuf> {
    let api = build_api(token)?;

    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            dataset_id.to_string(),
            RepoType::Dataset,
            rev.to_string(),
        )),
        None => api.repo(Repo::new(dataset_id.to_string(), RepoType::Dataset)),
    };

    repo.get(filename)
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))
}
