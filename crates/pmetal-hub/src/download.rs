//! Model downloads from the HuggingFace Hub.
//!
//! A download resolves the revision to one commit, lists that commit's files,
//! chooses the ones the model needs (see [`crate::select`]), fetches exactly
//! those concurrently into the standard cache, and checks what arrived. The
//! snapshot directory it returns is complete and from a single commit, or the
//! call failed.

use std::path::{Path, PathBuf};

use hf_hub::progress::Progress;
use pmetal_core::{PMetalError, Result, SecretString};

use crate::cache::{find_cached_model_in, is_commit_sha, write_ref};
use crate::client::HubOptions;
use crate::progress::{CallbackHandler, ProgressCallback};
use crate::repo::{RepoHandle, is_network_unavailable};
use crate::select::{
    ModelPlan, RepoFile, SelectedWeights, WeightFormat, index_files_to_read, plan_model,
    shards_in_index,
};
use crate::verify::verify_model_dir;

/// How thoroughly to check downloaded files against what the Hub listed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Verification {
    /// Trust the transfer.
    None,
    /// Every file is present and exactly the size the Hub lists. Cheap.
    #[default]
    Size,
    /// As `Size`, and large files are hashed (SHA-256) against the digest the
    /// Hub lists. Reads every byte back from disk.
    Sha256,
}

/// How to download a model.
#[derive(Clone, Default)]
pub struct DownloadOptions {
    /// Connection settings: token, endpoint, cache directory, offline mode.
    pub hub: HubOptions,
    /// Branch, tag or commit to fetch. `None` takes what is already cached,
    /// and the default branch if nothing is; name `"main"` to check the Hub
    /// for a newer commit.
    pub revision: Option<String>,
    /// Which weight format to fetch.
    pub weights: WeightFormat,
    /// For GGUF weights, the quantization to fetch (`Q4_K_M`, `Q8_0`, ...);
    /// `None` picks a sensible default.
    pub gguf_quant: Option<String>,
    /// Checks to run on the downloaded files.
    pub verify: Verification,
    /// Files fetched at the same time (eight when `None`).
    pub max_workers: Option<usize>,
    /// Called with progress snapshots (bytes, rate, ETA) while files transfer.
    pub progress: Option<ProgressCallback>,
}

impl std::fmt::Debug for DownloadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadOptions")
            .field("hub", &self.hub)
            .field("revision", &self.revision)
            .field("weights", &self.weights)
            .field("gguf_quant", &self.gguf_quant)
            .field("verify", &self.verify)
            .field("max_workers", &self.max_workers)
            .field("progress", &self.progress.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

impl DownloadOptions {
    /// Options from the positional arguments of the plain download functions.
    pub fn new(revision: Option<&str>, token: Option<&SecretString>) -> Self {
        Self {
            hub: HubOptions::from_token(token),
            revision: revision.map(str::to_owned),
            ..Self::default()
        }
    }

    /// Whether a cached snapshot can be served without asking the Hub: the
    /// caller wants no particular weights or quantization, and no particular
    /// branch (or one that is a commit, which never changes).
    fn cache_may_answer(&self) -> bool {
        self.weights == WeightFormat::Auto
            && self.gguf_quant.is_none()
            && self.revision.as_deref().is_none_or(is_commit_sha)
    }
}

/// The most a weight index (`*.index.json`) may be; real ones are well under a
/// few megabytes.
const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;

/// What a model download would fetch: the commit, the files, their size.
#[derive(Debug, Clone)]
pub struct DownloadPlan {
    /// The repository.
    pub repo_id: String,
    /// The commit the revision resolved to.
    pub commit: String,
    /// The files chosen.
    pub model: ModelPlan,
}

/// Download a model from HuggingFace Hub.
///
/// Fetches what is needed to run the model: configs and tokenizer files plus
/// one set of weights (safetensors; or one GGUF quantization when that is all
/// the repository has). Everything comes from the same commit. See
/// [`download_model_with`] to choose the format, quantization, endpoint and
/// more.
///
/// # Arguments
/// * `model_id` - Model identifier (e.g., "meta-llama/Llama-3.2-1B")
/// * `revision` - Optional revision/branch (e.g., "main"); `None` serves a
///   cached copy without contacting the Hub
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_model(
    model_id: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<PathBuf> {
    download_model_with(model_id, &DownloadOptions::new(revision, token)).await
}

/// Download a model with explicit [`DownloadOptions`]; the snapshot directory.
///
/// With `options.hub.offline` (or `HF_HUB_OFFLINE`) nothing touches the
/// network: the cache answers or the call fails. Online, a cached copy is
/// served without a request when no revision is named; a named branch is
/// resolved first, and if the Hub cannot be reached the copy last cached for
/// that name is used.
///
/// A model published only as PyTorch pickles is converted to safetensors in
/// the snapshot directory (see [`crate::torch`]), so the directory returned
/// always holds weights that load.
pub async fn download_model_with(model_id: &str, options: &DownloadOptions) -> Result<PathBuf> {
    let snapshot = fetch_model(model_id, options).await?;
    ensure_loadable(snapshot).await
}

/// If `snapshot` holds PyTorch pickles and no safetensors, write the
/// safetensors (a no-op when they are already there).
async fn ensure_loadable(snapshot: PathBuf) -> Result<PathBuf> {
    if verify_model_dir(&snapshot).format != Some(WeightFormat::PyTorch) {
        return Ok(snapshot);
    }
    tracing::info!(
        "Converting PyTorch weights in {} to safetensors",
        snapshot.display()
    );
    let dir = snapshot.clone();
    tokio::task::spawn_blocking(move || crate::torch::convert_pytorch_dir(&dir))
        .await
        .map_err(|e| PMetalError::Hub(format!("the weight conversion was interrupted: {e}")))??;
    let report = verify_model_dir(&snapshot);
    if report.is_complete() {
        Ok(snapshot)
    } else {
        Err(PMetalError::Hub(format!(
            "converted the PyTorch weights in {} but the result is not a complete model: {}",
            snapshot.display(),
            report.problems.join("; ")
        )))
    }
}

async fn fetch_model(model_id: &str, options: &DownloadOptions) -> Result<PathBuf> {
    crate::repo::validate_repo_id(model_id)?;
    let cache_root = options.hub.resolved_cache_dir();
    let revision = options.revision.as_deref();

    if options.hub.is_offline() {
        return cached_or_offline_error(&cache_root, model_id, revision, options);
    }
    if options.cache_may_answer() {
        if let Some(path) = find_cached_model_in(&cache_root, model_id, revision) {
            tracing::info!("Model {model_id} found in cache: {}", path.display());
            return Ok(path);
        }
    }

    let client = options.hub.client()?;
    let repo = RepoHandle::model(&client, model_id)?;
    let plan = match plan_for(&repo, options).await {
        Ok(plan) => plan,
        Err(PlanError::Unreachable(e)) => {
            // The Hub cannot be reached: the last copy cached for this name
            // beats failing.
            if let Some(path) = find_cached_model_in(&cache_root, model_id, revision) {
                tracing::warn!("The Hub is unreachable ({e}); using the cached copy of {model_id}");
                return Ok(path);
            }
            return Err(e);
        }
        Err(PlanError::Failed(e)) => return Err(e),
    };

    tracing::info!(
        "Downloading {} files ({}) of {model_id} at {}",
        plan.model.files.len(),
        crate::fit::format_bytes(plan.model.total_bytes()),
        &plan.commit[..plan.commit.len().min(8)],
    );
    let progress = options
        .progress
        .clone()
        .map(|cb| Progress::new(CallbackHandler::new(cb)));
    let paths: Vec<String> = plan.model.files.iter().map(|f| f.path.clone()).collect();
    let snapshot = repo
        .download_snapshot(&plan.commit, &paths, options.max_workers, progress)
        .await?;

    // Name the revision in the cache, as the Hub's own tools do, so the next
    // lookup (including an offline one) finds this commit by name.
    let ref_name = match revision {
        Some(r) if is_commit_sha(r) => None,
        Some(r) => Some(r),
        None => Some("main"),
    };
    if let Some(name) = ref_name {
        if let Err(e) = write_ref(&cache_root, &repo.folder(), name, &plan.commit) {
            tracing::warn!("Could not record {name} for {model_id}: {e}");
        }
    }

    check_downloaded(&snapshot, &plan.model.files, options.verify)?;
    if let SelectedWeights::Gguf { .. } = plan.model.weights {
        ensure_gguf_config(&snapshot, &plan.model);
    }
    let report = verify_model_dir(&snapshot);
    if !report.is_complete() {
        return Err(PMetalError::Hub(format!(
            "downloaded {model_id} but the result is not a complete model: {}",
            report.problems.join("; ")
        )));
    }
    Ok(snapshot)
}

/// Work out what a download of `model_id` would fetch, without fetching it:
/// the resolved commit, the files and their total size. Contacts the Hub (the
/// listing and the weight indexes, a few kilobytes).
pub async fn plan_model_download(
    model_id: &str,
    options: &DownloadOptions,
) -> Result<DownloadPlan> {
    if options.hub.is_offline() {
        return Err(PMetalError::Hub(
            "cannot plan a download in offline mode: the file listing comes from the Hub".into(),
        ));
    }
    let client = options.hub.client()?;
    let repo = RepoHandle::model(&client, model_id)?;
    plan_for(&repo, options).await.map_err(|e| match e {
        PlanError::Unreachable(e) | PlanError::Failed(e) => e,
    })
}

enum PlanError {
    /// The Hub could not be reached or is overloaded.
    Unreachable(PMetalError),
    /// The Hub answered and the answer is a failure (no such repository, no
    /// access, no usable weights).
    Failed(PMetalError),
}

async fn plan_for(
    repo: &RepoHandle,
    options: &DownloadOptions,
) -> std::result::Result<DownloadPlan, PlanError> {
    let classify = |e: hf_hub::HFError| {
        if is_network_unavailable(&e) {
            PlanError::Unreachable(repo.error(&e))
        } else {
            PlanError::Failed(repo.error(&e))
        }
    };
    let commit = repo
        .resolve_commit(options.revision.as_deref())
        .await
        .map_err(classify)?;
    let files = repo.list_files(&commit).await.map_err(classify)?;

    // Which shards belong to the model is written in the weight indexes.
    let mut indexes = std::collections::HashMap::new();
    for path in index_files_to_read(&files, options.weights) {
        let bytes = repo
            .read_file(&commit, &path, MAX_INDEX_BYTES)
            .await
            .map_err(PlanError::Failed)?;
        let shards = shards_in_index(&bytes)
            .map_err(|e| PlanError::Failed(PMetalError::Hub(format!("{path}: {e}"))))?;
        indexes.insert(path, shards);
    }
    let model = plan_model(
        &files,
        &indexes,
        options.weights,
        options.gguf_quant.as_deref(),
    )
    .map_err(|e| PlanError::Failed(PMetalError::Hub(format!("{}: {e}", repo.id))))?;
    Ok(DownloadPlan {
        repo_id: repo.id.clone(),
        commit,
        model,
    })
}

fn cached_or_offline_error(
    cache_root: &Path,
    model_id: &str,
    revision: Option<&str>,
    options: &DownloadOptions,
) -> Result<PathBuf> {
    // Offline, the cache must hold the weights asked for: a quantization or
    // format named explicitly is checked against what the snapshot contains.
    let found = find_cached_model_in(cache_root, model_id, revision).filter(|snap| {
        let report = verify_model_dir(snap);
        let format_ok = match (options.weights, report.format) {
            (WeightFormat::Auto, _) => true,
            (wanted, got) => got == Some(wanted),
        };
        let quant_ok = options.gguf_quant.as_deref().is_none_or(|q| {
            report.weight_files.iter().any(|f| {
                f.file_name()
                    .and_then(|n| n.to_str())
                    .and_then(crate::select::gguf_quant_label)
                    .is_some_and(|label| label.eq_ignore_ascii_case(q.trim()))
            })
        });
        format_ok && quant_ok
    });
    found.ok_or_else(|| {
        PMetalError::Hub(format!(
            "{model_id}{} is not in the local cache ({}) and offline mode is on: download it \
             once with network access",
            revision.map(|r| format!("@{r}")).unwrap_or_default(),
            cache_root.display()
        ))
    })
}

/// Check every planned file arrived with the size (and, if asked, the hash)
/// the Hub listed.
fn check_downloaded(snapshot: &Path, planned: &[RepoFile], verify: Verification) -> Result<()> {
    if verify == Verification::None {
        return Ok(());
    }
    let mut problems = Vec::new();
    for file in planned {
        let path = snapshot.join(&file.path);
        match std::fs::metadata(&path) {
            Err(_) => problems.push(format!("{} is missing", file.path)),
            Ok(meta) if meta.len() != file.size => problems.push(format!(
                "{} is {} bytes, the Hub lists {}",
                file.path,
                meta.len(),
                file.size
            )),
            Ok(_) => {
                if verify == Verification::Sha256 {
                    if let Some(expected) = &file.sha256 {
                        match sha256_of(&path) {
                            Ok(actual) if actual.eq_ignore_ascii_case(expected) => {}
                            Ok(actual) => problems.push(format!(
                                "{} has SHA-256 {actual}, the Hub lists {expected}",
                                file.path
                            )),
                            Err(e) => problems.push(format!("{}: {e}", file.path)),
                        }
                    }
                }
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(PMetalError::Hub(format!(
            "the download is not what the Hub listed: {}",
            problems.into_iter().take(10).collect::<Vec<_>>().join("; ")
        )))
    }
}

/// SHA-256 of a file, as lowercase hex.
pub(crate) fn sha256_of(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// A GGUF-only repository has no `config.json`; derive one from the GGUF's
/// metadata so the directory loads like any other model.
fn ensure_gguf_config(snapshot: &Path, plan: &ModelPlan) {
    if snapshot.join("config.json").exists() {
        return;
    }
    let Some(first) = plan.weight_files().next() else {
        return;
    };
    match pmetal_gguf::GgufModel::open(snapshot.join(&first.path)) {
        Ok(model) => {
            if let Some(path) =
                pmetal_gguf::config::write_config_from_gguf(model.metadata_content(), snapshot)
            {
                tracing::info!(
                    "Generated config.json from GGUF metadata: {}",
                    path.display()
                );
            }
        }
        Err(e) => tracing::warn!("Could not read GGUF metadata for config generation: {e}"),
    }
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
    download_file_with(model_id, filename, &DownloadOptions::new(revision, token)).await
}

/// [`download_file`] with explicit [`DownloadOptions`] (only the connection
/// settings and the revision apply). Offline, the cached copy or an error.
pub async fn download_file_with(
    model_id: &str,
    filename: &str,
    options: &DownloadOptions,
) -> Result<PathBuf> {
    let client = options.hub.client()?;
    let repo = RepoHandle::model(&client, model_id)?;
    repo.download_file(
        options.revision.as_deref(),
        filename,
        options.hub.is_offline(),
    )
    .await
    .map_err(|e| repo.error(&e))
}

/// Download all safetensors files for a model.
///
/// Exactly the shards the weight index names (or the single `model.safetensors`),
/// from one commit. The paths are in shard-name order.
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
    let options = DownloadOptions {
        weights: WeightFormat::Safetensors,
        ..DownloadOptions::new(revision, token)
    };
    let dir = download_model_with(model_id, &options).await?;
    let report = verify_model_dir(&dir);
    if report.weight_files.is_empty() {
        return Err(PMetalError::Hub(format!(
            "No safetensors weights found for {model_id}"
        )));
    }
    Ok(report.weight_files)
}

/// Shared by the dataset downloads.
pub(crate) fn progress_handle(callback: &Option<ProgressCallback>) -> Option<Progress> {
    callback
        .clone()
        .map(|cb| Progress::new(CallbackHandler::new(cb)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with(files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, bytes) in files {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        dir
    }

    fn listed(path: &str, size: u64, sha256: Option<&str>) -> RepoFile {
        RepoFile {
            path: path.into(),
            size,
            sha256: sha256.map(str::to_owned),
        }
    }

    fn sha256_of_bytes(bytes: &[u8]) -> String {
        let dir = snapshot_with(&[("x", bytes)]);
        sha256_of(&dir.path().join("x")).unwrap()
    }

    #[test]
    fn files_of_the_listed_size_pass() {
        let dir = snapshot_with(&[("a.bin", b"12345"), ("sub/b.json", b"{}")]);
        let planned = [listed("a.bin", 5, None), listed("sub/b.json", 2, None)];
        check_downloaded(dir.path(), &planned, Verification::Size).unwrap();
    }

    #[test]
    fn a_missing_or_short_file_fails_naming_it() {
        let dir = snapshot_with(&[("a.bin", b"123")]);
        let planned = [listed("a.bin", 5, None), listed("gone.json", 2, None)];
        let err = check_downloaded(dir.path(), &planned, Verification::Size)
            .unwrap_err()
            .to_string();
        assert!(err.contains("a.bin is 3 bytes, the Hub lists 5"), "{err}");
        assert!(err.contains("gone.json is missing"), "{err}");
    }

    #[test]
    fn no_verification_trusts_the_transfer() {
        let dir = snapshot_with(&[]);
        check_downloaded(dir.path(), &[listed("a.bin", 5, None)], Verification::None).unwrap();
    }

    #[test]
    fn hashing_catches_a_flipped_byte_that_size_cannot() {
        let content = b"the weights";
        let digest = sha256_of_bytes(content);
        let good = snapshot_with(&[("w.bin", content)]);
        let planned = [listed("w.bin", content.len() as u64, Some(&digest))];
        check_downloaded(good.path(), &planned, Verification::Sha256).unwrap();

        let corrupt = snapshot_with(&[("w.bin", b"the weighTs")]);
        check_downloaded(corrupt.path(), &planned, Verification::Size).unwrap();
        let err = check_downloaded(corrupt.path(), &planned, Verification::Sha256)
            .unwrap_err()
            .to_string();
        assert!(err.contains("SHA-256"), "{err}");
    }

    #[test]
    fn sha256_matches_the_known_digest() {
        assert_eq!(
            sha256_of_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn the_cache_answers_only_for_requests_it_can_satisfy() {
        let plain = DownloadOptions::default();
        assert!(plain.cache_may_answer());
        let pinned = DownloadOptions {
            revision: Some("a".repeat(40)),
            ..Default::default()
        };
        assert!(pinned.cache_may_answer(), "a commit never changes");
        let branch = DownloadOptions {
            revision: Some("main".into()),
            ..Default::default()
        };
        assert!(!branch.cache_may_answer(), "a branch moves; ask the Hub");
        let quant = DownloadOptions {
            gguf_quant: Some("Q8_0".into()),
            ..Default::default()
        };
        assert!(!quant.cache_may_answer());
        let format = DownloadOptions {
            weights: WeightFormat::Gguf,
            ..Default::default()
        };
        assert!(!format.cache_may_answer());
    }

    #[test]
    fn options_debug_does_not_show_the_token() {
        let options =
            DownloadOptions::new(None, Some(&SecretString::new("hf_secret_value_0123456789")));
        assert!(!format!("{options:?}").contains("hf_secret_value"));
    }
}
