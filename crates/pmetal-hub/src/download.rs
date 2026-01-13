//! Model and dataset downloading from HuggingFace Hub.

use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use pmetal_core::{Result, SecretString};
use std::path::PathBuf;

/// Build API with optional token authentication.
fn build_api(token: Option<&SecretString>) -> Result<Api> {
    let mut builder = ApiBuilder::new();

    if let Some(secret) = token {
        builder = builder.with_token(Some(secret.expose_secret().to_string()));
    }

    builder
        .build()
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))
}

/// Download a model from HuggingFace Hub.
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

    // Download config.json to get the model path
    let config_path = repo
        .get("config.json")
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))?;

    // Return the parent directory (model cache location)
    Ok(config_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".")))
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

    Ok(paths)
}

/// Download a dataset from HuggingFace Hub.
///
/// Returns the path to the dataset directory containing downloaded files.
///
/// # Arguments
/// * `dataset_id` - Dataset identifier (e.g., "tatsu-lab/alpaca")
/// * `_split` - Dataset split (currently unused, reserved for future use)
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_dataset(
    dataset_id: &str,
    _split: Option<&str>,
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

    // Try to get the README first to determine the dataset path
    let readme_path = repo
        .get("README.md")
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))?;

    // Return the parent directory (dataset cache location)
    Ok(readme_path
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
    // 3. {split}/train-00000-of-00001.parquet (split directory)

    // Try single file first
    let single_file = format!("data/{}.parquet", split);
    if let Ok(path) = repo.get(&single_file).await {
        paths.push(path);
        return Ok(paths);
    }

    // Try numbered shards (up to 100 shards)
    for i in 0..100 {
        // We need to try common shard counts
        for total in [1, 2, 4, 5, 10, 20, 50, 100] {
            let exact_file = format!("data/{}-{:05}-of-{:05}.parquet", split, i, total);
            if let Ok(path) = repo.get(&exact_file).await {
                paths.push(path);
                break;
            }
        }

        // Stop if we found no more shards after finding at least one
        if i > 0 && paths.len() == i {
            break;
        }
    }

    if paths.is_empty() {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "No parquet files found for split '{}' in dataset '{}'",
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
