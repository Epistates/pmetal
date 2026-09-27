//! Model and dataset downloading from HuggingFace Hub.

use hf_hub::{
    HFClient, HFError, HFRepository, HFResult, RepoType, RepoTypeDataset, RepoTypeModel, split_id,
};
use pmetal_core::{Result, SecretString};
use std::path::{Path, PathBuf};

/// Build a client with optional token authentication.
///
/// The cache directory is pmetal's own resolution, so a download lands where
/// [`crate::cache::find_cached_model`] looks for it.
fn build_client(token: Option<&SecretString>) -> Result<HFClient> {
    let mut builder = HFClient::builder().cache_dir(crate::cache::cache_dir());

    if let Some(secret) = token {
        builder = builder.token(secret.expose_secret());
    }

    builder.build().map_err(hub_error)
}

fn hub_error(e: HFError) -> pmetal_core::PMetalError {
    pmetal_core::PMetalError::Hub(e.to_string())
}

/// A repository handle pinned to one revision (`None` is the default branch).
struct Repo<T: RepoType> {
    handle: HFRepository<T>,
    revision: Option<String>,
}

impl<T: RepoType> Repo<T> {
    /// Download `filename` into the cache and return its snapshot path.
    async fn get(&self, filename: &str) -> HFResult<PathBuf> {
        self.handle
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
            .await
    }
}

fn model_repo(client: &HFClient, id: &str, revision: Option<&str>) -> Repo<RepoTypeModel> {
    let (owner, name) = split_id(id);
    Repo {
        handle: client.model(owner, name),
        revision: revision.map(str::to_owned),
    }
}

fn dataset_repo(client: &HFClient, id: &str, revision: Option<&str>) -> Repo<RepoTypeDataset> {
    let (owner, name) = split_id(id);
    Repo {
        handle: client.dataset(owner, name),
        revision: revision.map(str::to_owned),
    }
}

/// The snapshot directory a downloaded repo file lives under.
///
/// `path` ends in the repo-relative `filename`, which can be nested
/// (`data/train-00000.parquet`), so its parent is not always the snapshot root.
fn snapshot_root(path: &Path, filename: &str) -> Option<PathBuf> {
    path.ancestors()
        .nth(Path::new(filename).components().count())
        .map(PathBuf::from)
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
    ".parquet", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".arrow",
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
    // Fast path: check local cache first (no network call)
    if revision.is_none() {
        if let Some(cached_path) = crate::cache::find_cached_model(model_id) {
            tracing::info!(
                "Model {} found in cache: {}",
                model_id,
                cached_path.display()
            );
            return Ok(cached_path);
        }
    }

    let client = build_client(token)?;
    let repo = model_repo(&client, model_id, revision);

    // List all files in the repository
    let repo_info = repo
        .handle
        .info()
        .maybe_revision(repo.revision.clone())
        .send()
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(format!("Failed to list repo files: {e}")))?;

    let all_files: Vec<&str> = repo_info
        .siblings
        .iter()
        .flatten()
        .map(|s| s.rfilename.as_str())
        .collect();

    tracing::info!(
        "Found {} files in {}, downloading...",
        all_files.len(),
        model_id
    );

    // Check if this is a GGUF-only repo (no safetensors files)
    let has_safetensors = all_files
        .iter()
        .any(|f| f.ends_with(".safetensors") || f.ends_with(".safetensors.index.json"));
    let gguf_files: Vec<&str> = all_files
        .iter()
        .filter(|f| f.ends_with(".gguf"))
        .copied()
        .collect();
    let is_gguf_only = !has_safetensors && !gguf_files.is_empty();

    if is_gguf_only {
        tracing::info!(
            "{} is a GGUF-only repo ({} variants found), selecting best quantization...",
            model_id,
            gguf_files.len()
        );
    }

    // For GGUF-only repos, pick the best single GGUF file
    let selected_gguf = if is_gguf_only {
        Some(select_best_gguf(&gguf_files))
    } else {
        None
    };

    // Partition files into those we want and those we skip
    let mut model_dir: Option<PathBuf> = None;
    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut failures = Vec::new();

    for filename in &all_files {
        // For GGUF-only repos: download the selected GGUF + metadata files
        // For safetensors repos: skip all GGUF files (current behavior)
        let is_gguf = filename.ends_with(".gguf");
        if is_gguf {
            if let Some(ref selected) = selected_gguf {
                if *filename != selected.as_str() {
                    tracing::debug!("Skipping {} (not selected GGUF variant)", filename);
                    skipped += 1;
                    continue;
                }
                // Fall through to download the selected GGUF
            } else {
                tracing::debug!("Skipping {} (safetensors available)", filename);
                skipped += 1;
                continue;
            }
        }

        // Skip files by extension (but not GGUF if we're in GGUF-only mode)
        if !is_gguf && SKIP_EXTENSIONS.iter().any(|ext| filename.ends_with(ext)) {
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
                    model_dir = snapshot_root(&path, filename);
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
        let preview = failures.into_iter().take(10).collect::<Vec<_>>().join(", ");
        return Err(pmetal_core::PMetalError::Hub(format!(
            "Download incomplete for {}: {}",
            model_id, preview
        )));
    }

    let dir = model_dir.ok_or_else(|| {
        pmetal_core::PMetalError::Hub(format!("No files downloaded for {}", model_id))
    })?;

    // For GGUF-only downloads: generate config.json from GGUF metadata
    // if no config.json exists in the downloaded directory
    if is_gguf_only && !dir.join("config.json").exists() {
        if let Some(ref gguf_name) = selected_gguf {
            let gguf_path = dir.join(gguf_name);
            match pmetal_gguf::GgufContent::from_file(&gguf_path) {
                Ok(content) => {
                    if let Some(config_path) =
                        pmetal_gguf::config::write_config_from_gguf(&content, &dir)
                    {
                        tracing::info!(
                            "Generated config.json from GGUF metadata: {}",
                            config_path.display()
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("Could not read GGUF metadata for config generation: {e}");
                }
            }
        }
    }

    Ok(dir)
}

/// Select the best GGUF file from a list of variants.
///
/// Preference order (best balance of quality vs size):
/// Q4_K_M > Q5_K_M > Q4_K_S > Q5_K_S > Q6_K > Q3_K_M > Q8_0 > first available
fn select_best_gguf(gguf_files: &[&str]) -> String {
    let preferences = [
        "q4_k_m", "Q4_K_M", "q5_k_m", "Q5_K_M", "q4_k_s", "Q4_K_S", "q5_k_s", "Q5_K_S", "q6_k",
        "Q6_K", "q3_k_m", "Q3_K_M", "q8_0", "Q8_0", "f16", "F16",
    ];

    for pref in &preferences {
        if let Some(f) = gguf_files.iter().find(|f| f.contains(pref)) {
            return f.to_string();
        }
    }

    // Fallback: pick the smallest GGUF file by name (likely lowest quant)
    gguf_files
        .first()
        .map(|f| f.to_string())
        .unwrap_or_default()
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
    let client = build_client(token)?;
    model_repo(&client, model_id, revision)
        .get(filename)
        .await
        .map_err(hub_error)
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
    let client = build_client(token)?;
    let repo = model_repo(&client, model_id, revision);

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
    // Fast path: check local cache first (no network call)
    if revision.is_none() {
        if let Some(cached_path) = crate::cache::find_cached_dataset(dataset_id) {
            tracing::info!(
                "Dataset {} found in cache: {}",
                dataset_id,
                cached_path.display()
            );
            return Ok(cached_path);
        }
    }

    let client = build_client(token)?;
    let repo = dataset_repo(&client, dataset_id, revision);

    let repo_info = repo
        .handle
        .info()
        .maybe_revision(repo.revision.clone())
        .send()
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(format!("Failed to list repo files: {e}")))?;

    let mut dataset_dir: Option<PathBuf> = None;
    let mut data_files_downloaded = 0usize;
    let mut data_failures = Vec::new();

    for sibling in repo_info.siblings.iter().flatten() {
        let filename = sibling.rfilename.as_str();
        if filename.starts_with('.') {
            continue;
        }

        let is_data_file = DATASET_EXTENSIONS.iter().any(|ext| filename.ends_with(ext))
            || split.is_some_and(|s| filename.contains(s));
        let is_readme = filename == "README.md";

        if !is_data_file && !is_readme {
            continue;
        }

        match repo.get(filename).await {
            Ok(path) => {
                if is_data_file {
                    data_files_downloaded += 1;
                }
                if dataset_dir.is_none() {
                    dataset_dir = snapshot_root(&path, filename);
                }
            }
            Err(err) => {
                if is_data_file {
                    data_failures.push(format!("{filename}: {err}"));
                } else {
                    // Non-fatal: README.md may be gated or absent on some repos.
                    tracing::warn!("Could not download {filename} for {dataset_id}: {err}");
                }
            }
        }
    }

    if data_files_downloaded == 0 {
        if !data_failures.is_empty() {
            return Err(pmetal_core::PMetalError::Hub(format!(
                "Dataset download incomplete for {}: {}",
                dataset_id,
                data_failures
                    .into_iter()
                    .take(10)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        return Err(pmetal_core::PMetalError::Hub(format!(
            "No dataset files downloaded for {}",
            dataset_id
        )));
    }

    if !data_failures.is_empty() {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "Dataset download incomplete for {}: {}",
            dataset_id,
            data_failures
                .into_iter()
                .take(10)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    dataset_dir.ok_or_else(|| {
        pmetal_core::PMetalError::Hub(format!("No dataset files downloaded for {dataset_id}"))
    })
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
    let client = build_client(token)?;
    let repo = dataset_repo(&client, dataset_id, revision);

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
    let parquet_repo = dataset_repo(&client, dataset_id, Some("refs/convert/parquet"));

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
    let client = build_client(token)?;
    dataset_repo(&client, dataset_id, revision)
        .get(filename)
        .await
        .map_err(hub_error)
}

#[cfg(test)]
mod tests {
    use super::snapshot_root;
    use std::path::Path;

    #[test]
    fn a_nested_file_resolves_to_its_snapshot_root() {
        let snapshot = Path::new("/hub/datasets--o--n/snapshots/abc");
        let file = snapshot.join("data/train-00000-of-00001.parquet");
        assert_eq!(
            snapshot_root(&file, "data/train-00000-of-00001.parquet").as_deref(),
            Some(snapshot)
        );
        assert_eq!(
            snapshot_root(&snapshot.join("config.json"), "config.json").as_deref(),
            Some(snapshot)
        );
    }

    /// Hits the Hub. `tatsu-lab/alpaca` keeps its data under `data/`.
    #[tokio::test]
    #[ignore = "network"]
    async fn a_dataset_with_nested_files_downloads_to_its_snapshot_root() {
        let dir = super::download_dataset("tatsu-lab/alpaca", None, Some("main"), None)
            .await
            .unwrap();
        assert!(dir.join("data").is_dir(), "{}", dir.display());
    }
}
