//! Model uploading to HuggingFace Hub.
//!
//! Uses the HuggingFace Hub HTTP API directly since `hf-hub` 0.4 does not
//! expose upload functionality.  The upload flow is:
//!
//! 1. Create the repository if it does not exist (`POST /api/repos/create`).
//! 2. Collect all uploadable files from the model directory.
//! 3. For each large binary file (`.safetensors`, `.gguf`, …) run the Git LFS
//!    pre-upload handshake so HuggingFace stores the object in S3/Xet.
//! 4. Build a single commit via `POST /api/models/{repo_id}/commit/main`
//!    using NDJSON:
//!    - JSON-config / tokenizer files → `{"key":"file","value":{"content":<b64>,"path":…,"encoding":"base64"}}`
//!    - Pre-uploaded LFS blobs       → `{"key":"lfsFile","value":{"path":…,"algo":"sha256","oid":…,"size":…}}`

use base64::Engine as _;
use pmetal_core::{PMetalError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs;
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HF_API_BASE: &str = "https://huggingface.co";

/// Files larger than this threshold are uploaded as LFS objects.  HuggingFace
/// itself uses 10 MiB; we mirror that default.
const LFS_THRESHOLD_BYTES: u64 = 10 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Internal helpers / newtypes
// ---------------------------------------------------------------------------

/// A file collected from the model directory that is ready to be uploaded.
#[derive(Debug)]
struct UploadCandidate {
    /// Absolute path on the local filesystem.
    local_path: PathBuf,
    /// Relative path that will appear inside the HF repo.
    repo_path: String,
    /// File size in bytes.
    size: u64,
}

// ---------------------------------------------------------------------------
// LFS types (subset of the Git LFS Batch API)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LfsBatchRequest {
    operation: &'static str,
    transfers: Vec<&'static str>,
    hash_algo: &'static str,
    objects: Vec<LfsObject>,
}

#[derive(Serialize)]
struct LfsObject {
    oid: String,
    size: u64,
}

#[derive(Deserialize, Debug)]
struct LfsBatchResponse {
    objects: Vec<LfsBatchObject>,
}

#[derive(Deserialize, Debug)]
struct LfsBatchObject {
    oid: String,
    size: u64,
    #[serde(default)]
    actions: Option<LfsBatchActions>,
}

#[derive(Deserialize, Debug)]
struct LfsBatchActions {
    upload: Option<LfsUploadAction>,
    verify: Option<LfsVerifyAction>,
}

#[derive(Deserialize, Debug)]
struct LfsUploadAction {
    href: String,
    #[serde(default)]
    header: std::collections::HashMap<String, String>,
}

#[derive(Deserialize, Debug)]
struct LfsVerifyAction {
    href: String,
    #[serde(default)]
    header: std::collections::HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Upload a model directory to a HuggingFace Hub repository.
///
/// # Arguments
/// * `model_path`  – Local directory containing the model files.
/// * `repo_id`     – Hub repository in the form `"owner/repo-name"`.
/// * `token`       – HuggingFace write token (Bearer auth).
pub async fn upload_model<P: AsRef<Path>>(model_path: P, repo_id: &str, token: &str) -> Result<()> {
    let model_path = model_path.as_ref();

    if !model_path.is_dir() {
        return Err(PMetalError::InvalidArgument(format!(
            "model_path '{}' is not a directory",
            model_path.display()
        )));
    }

    let client = Client::builder()
        .user_agent("pmetal/0.1.0")
        .build()
        .map_err(|e| PMetalError::Hub(format!("failed to build HTTP client: {e}")))?;

    // 1. Create the repository (idempotent – existing repos are fine).
    ensure_repo_exists(&client, repo_id, token).await?;

    // 2. Collect files to upload.
    let candidates = collect_files(model_path)?;

    if candidates.is_empty() {
        tracing::warn!("No uploadable files found in '{}'", model_path.display());
        return Ok(());
    }

    tracing::info!(
        repo_id,
        file_count = candidates.len(),
        "Beginning upload to HuggingFace Hub"
    );

    // 3. Partition into inline (small/text) vs LFS (large binary) files.
    let (inline_files, lfs_files): (Vec<_>, Vec<_>) = candidates
        .into_iter()
        .partition(|c| c.size < LFS_THRESHOLD_BYTES);

    // 4. Pre-upload LFS objects and gather their metadata.
    let lfs_entries = if lfs_files.is_empty() {
        vec![]
    } else {
        preupload_lfs_objects(&client, repo_id, token, &lfs_files).await?
    };

    // 5. Build the NDJSON commit body and POST it.
    commit_files(&client, repo_id, token, &inline_files, &lfs_entries).await?;

    tracing::info!(repo_id, "Upload complete");
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 1 – Ensure the repository exists
// ---------------------------------------------------------------------------

async fn ensure_repo_exists(client: &Client, repo_id: &str, token: &str) -> Result<()> {
    // repo_id is "owner/name" – split into org/name if needed.
    let (org, name) = split_repo_id(repo_id)?;

    let body = json!({
        "type": "model",
        "name": name,
        "organization": org,
        "private": false,
    });

    let url = format!("{HF_API_BASE}/api/repos/create");
    let resp = client
        .post(&url)
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .map_err(|e| PMetalError::Hub(format!("create repo request failed: {e}")))?;

    match resp.status().as_u16() {
        200 | 201 => {
            tracing::info!(repo_id, "Repository created");
        }
        409 => {
            // Conflict – repo already exists, that is fine.
            tracing::debug!(repo_id, "Repository already exists");
        }
        status => {
            let body = resp.text().await.unwrap_or_default();
            return Err(PMetalError::Hub(format!(
                "create repo returned HTTP {status}: {body}"
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Step 2 – Collect uploadable files
// ---------------------------------------------------------------------------

fn collect_files(model_path: &Path) -> Result<Vec<UploadCandidate>> {
    let mut candidates = Vec::new();

    for entry in WalkDir::new(model_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let local_path = entry.path().to_path_buf();

        if !is_uploadable(&local_path) {
            continue;
        }

        let metadata = entry.metadata().map_err(|e| {
            PMetalError::Io(std::io::Error::other(format!(
                "metadata error for {}: {e}",
                local_path.display()
            )))
        })?;

        let size = metadata.len();

        // Build the repo-relative path (always use forward slashes).
        let rel = local_path
            .strip_prefix(model_path)
            .map_err(|e| PMetalError::InvalidArgument(format!("path strip_prefix failed: {e}")))?;
        let repo_path = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");

        tracing::debug!(
            file = %repo_path,
            size,
            "Queued for upload"
        );

        candidates.push(UploadCandidate {
            local_path,
            repo_path,
            size,
        });
    }

    Ok(candidates)
}

/// Returns `true` for files that should be uploaded to the Hub.
fn is_uploadable(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        // Files without extensions: check by name.
        return matches!(name, "tokenizer" | "vocab");
    };

    matches!(
        ext,
        "safetensors"
            | "gguf"
            | "ggml"
            | "bin"
            | "pt"
            | "pth"
            | "json"
            | "yaml"
            | "yml"
            | "txt"
            | "md"
            | "model"
    )
}

// ---------------------------------------------------------------------------
// Step 3 – LFS pre-upload
// ---------------------------------------------------------------------------

/// Represents a file that has been pre-uploaded to LFS and is ready to be
/// referenced in the commit.
struct LfsEntry {
    repo_path: String,
    oid: String,
    size: u64,
}

async fn preupload_lfs_objects(
    client: &Client,
    repo_id: &str,
    token: &str,
    files: &[UploadCandidate],
) -> Result<Vec<LfsEntry>> {
    tracing::info!(
        count = files.len(),
        "Computing SHA-256 hashes for LFS objects"
    );

    // Hash all files first so we can batch the LFS request.
    let mut hashed: Vec<(&UploadCandidate, String)> = Vec::with_capacity(files.len());
    for candidate in files {
        let oid = sha256_file(&candidate.local_path).await?;
        tracing::debug!(
            file = %candidate.repo_path,
            oid = %oid,
            size = candidate.size,
            "Computed SHA-256"
        );
        hashed.push((candidate, oid));
    }

    // Batch request to the LFS endpoint.
    let lfs_url = format!("{HF_API_BASE}/{repo_id}.git/info/lfs/objects/batch");
    let lfs_body = LfsBatchRequest {
        operation: "upload",
        transfers: vec!["basic", "multipart"],
        hash_algo: "sha_256",
        objects: hashed
            .iter()
            .map(|(c, oid)| LfsObject {
                oid: oid.clone(),
                size: c.size,
            })
            .collect(),
    };

    let lfs_resp = client
        .post(&lfs_url)
        .bearer_auth(token)
        .header("Accept", "application/vnd.git-lfs+json")
        .header("Content-Type", "application/vnd.git-lfs+json")
        .json(&lfs_body)
        .send()
        .await
        .map_err(|e| PMetalError::Hub(format!("LFS batch request failed: {e}")))?;

    let status = lfs_resp.status();
    if !status.is_success() {
        let body = lfs_resp.text().await.unwrap_or_default();
        return Err(PMetalError::Hub(format!(
            "LFS batch returned HTTP {status}: {body}"
        )));
    }

    let batch: LfsBatchResponse = lfs_resp
        .json()
        .await
        .map_err(|e| PMetalError::Hub(format!("Failed to parse LFS batch response: {e}")))?;

    // Upload each object that the server indicated needs to be uploaded.
    let oid_to_candidate: std::collections::HashMap<&str, &UploadCandidate> =
        hashed.iter().map(|(c, oid)| (oid.as_str(), *c)).collect();

    let mut entries: Vec<LfsEntry> = Vec::with_capacity(hashed.len());

    for obj in &batch.objects {
        let candidate = oid_to_candidate
            .get(obj.oid.as_str())
            .ok_or_else(|| PMetalError::Hub(format!("Unknown LFS oid: {}", obj.oid)))?;

        if let Some(ref actions) = obj.actions {
            if let Some(ref upload_action) = actions.upload {
                tracing::info!(
                    file = %candidate.repo_path,
                    size = obj.size,
                    "Uploading LFS object"
                );
                upload_lfs_object(client, token, candidate, &obj.oid, upload_action).await?;

                // Run verify step if the server requested it.
                if let Some(ref verify_action) = actions.verify {
                    verify_lfs_object(client, token, &obj.oid, obj.size, verify_action).await?;
                }
            } else {
                tracing::debug!(
                    oid = %obj.oid,
                    "LFS object already present on server – skipping upload"
                );
            }
        }

        entries.push(LfsEntry {
            repo_path: candidate.repo_path.clone(),
            oid: obj.oid.clone(),
            size: obj.size,
        });
    }

    Ok(entries)
}

async fn upload_lfs_object(
    client: &Client,
    _token: &str,
    candidate: &UploadCandidate,
    _oid: &str,
    action: &LfsUploadAction,
) -> Result<()> {
    let data = fs::read(&candidate.local_path).await?;

    let mut req = client.put(&action.href).body(data);
    for (k, v) in &action.header {
        req = req.header(k.as_str(), v.as_str());
    }

    let resp = req.send().await.map_err(|e| {
        PMetalError::Hub(format!("LFS PUT failed for '{}': {e}", candidate.repo_path))
    })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(PMetalError::Hub(format!(
            "LFS PUT returned HTTP {status} for '{}': {body}",
            candidate.repo_path
        )));
    }

    tracing::debug!(
        file = %candidate.repo_path,
        "LFS object uploaded"
    );
    Ok(())
}

async fn verify_lfs_object(
    client: &Client,
    _token: &str,
    oid: &str,
    size: u64,
    action: &LfsVerifyAction,
) -> Result<()> {
    let body = json!({ "oid": oid, "size": size });

    let mut req = client
        .post(&action.href)
        .header("Content-Type", "application/vnd.git-lfs+json")
        .json(&body);
    for (k, v) in &action.header {
        req = req.header(k.as_str(), v.as_str());
    }

    let resp = req
        .send()
        .await
        .map_err(|e| PMetalError::Hub(format!("LFS verify request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(PMetalError::Hub(format!(
            "LFS verify returned HTTP {status}: {body}"
        )));
    }

    tracing::debug!(oid, "LFS object verified");
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 4 – Commit
// ---------------------------------------------------------------------------

async fn commit_files(
    client: &Client,
    repo_id: &str,
    token: &str,
    inline_files: &[UploadCandidate],
    lfs_entries: &[LfsEntry],
) -> Result<()> {
    // Build the NDJSON body line by line.
    let mut ndjson_lines: Vec<String> = Vec::new();

    // Header line.
    let header: Value = json!({
        "key": "header",
        "value": {
            "summary": "Upload model via pmetal",
            "description": "Automated upload using the pmetal CLI"
        }
    });
    ndjson_lines.push(serde_json::to_string(&header).map_err(|e| {
        PMetalError::Serialization(format!("Failed to serialize commit header: {e}"))
    })?);

    // Inline (small) files – base64-encode the content.
    for candidate in inline_files {
        tracing::info!(
            file = %candidate.repo_path,
            size = candidate.size,
            "Encoding inline file for commit"
        );
        let content = fs::read(&candidate.local_path).await?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&content);

        let line: Value = json!({
            "key": "file",
            "value": {
                "content": b64,
                "path": candidate.repo_path,
                "encoding": "base64"
            }
        });
        ndjson_lines.push(serde_json::to_string(&line).map_err(|e| {
            PMetalError::Serialization(format!("Failed to serialize file entry: {e}"))
        })?);
    }

    // LFS files – reference by OID.
    for entry in lfs_entries {
        tracing::info!(
            file = %entry.repo_path,
            oid = %entry.oid,
            size = entry.size,
            "Adding LFS file reference to commit"
        );
        let line: Value = json!({
            "key": "lfsFile",
            "value": {
                "path": entry.repo_path,
                "algo": "sha256",
                "oid": entry.oid,
                "size": entry.size
            }
        });
        ndjson_lines.push(serde_json::to_string(&line).map_err(|e| {
            PMetalError::Serialization(format!("Failed to serialize lfsFile entry: {e}"))
        })?);
    }

    let body = ndjson_lines.join("\n");
    let commit_url = format!("{HF_API_BASE}/api/models/{repo_id}/commit/main");

    tracing::info!(
        repo_id,
        inline_count = inline_files.len(),
        lfs_count = lfs_entries.len(),
        "Committing files to Hub"
    );

    let resp = client
        .post(&commit_url)
        .bearer_auth(token)
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .map_err(|e| PMetalError::Hub(format!("Commit request failed: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(PMetalError::Hub(format!(
            "Commit returned HTTP {status}: {body}"
        )));
    }

    let resp_json: Value = resp
        .json()
        .await
        .map_err(|e| PMetalError::Hub(format!("Failed to parse commit response: {e}")))?;

    if let Some(url) = resp_json.get("commitUrl").and_then(Value::as_str) {
        tracing::info!(commit_url = %url, "Commit successful");
    } else {
        tracing::info!("Commit successful");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Compute the hex-encoded SHA-256 digest of a file.
async fn sha256_file(path: &Path) -> Result<String> {
    let data = fs::read(path).await?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let digest = hasher.finalize();
    Ok(format!("{digest:x}"))
}

/// Split `"owner/repo"` into `("owner", "repo")`.
fn split_repo_id(repo_id: &str) -> Result<(&str, &str)> {
    let mut parts = repo_id.splitn(2, '/');
    let org = parts.next().ok_or_else(|| {
        PMetalError::InvalidArgument(format!("repo_id '{repo_id}' has no owner component"))
    })?;
    let name = parts.next().ok_or_else(|| {
        PMetalError::InvalidArgument(format!(
            "repo_id '{repo_id}' must be in the form 'owner/repo-name'"
        ))
    })?;
    Ok((org, name))
}
