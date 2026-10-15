//! Publishing a local model directory to the HuggingFace Hub.
//!
//! The Hub protocol lives in `hf-hub`: it asks the preupload endpoint which
//! files the repository stores as regular git blobs and which go to large-file
//! storage, streams the large ones to Xet storage, and writes everything in
//! one commit. That makes an upload atomic (either every file lands in the
//! commit or none does) and resumable (content already stored is not sent
//! again when the upload is retried). This module decides what to send and in
//! which repository.

use std::path::{Path, PathBuf};

use hf_hub::RepoTypeModel;
use pmetal_core::{PMetalError, Result, SecretString};

use crate::auth::resolve_token;
use crate::client::{HubOptions, hub_error_in};
use crate::download::progress_handle;
use crate::progress::ProgressCallback;

/// Files never worth publishing: VCS metadata, Finder and AppleDouble
/// droppings (exFAT and network volumes grow a `._name` beside every file),
/// the Hub's own download cache, and the credential files a working directory
/// tends to hold.
const ALWAYS_IGNORED: &[&str] = &[
    ".git/**",
    "**/.DS_Store",
    "**/._*",
    ".cache/**",
    "**/.cache/**",
    "**/.env",
    "**/.env.*",
    "**/.netrc",
];

/// How to publish a directory.
#[derive(Clone, Default)]
pub struct UploadOptions {
    /// Create the repository as private. Ignored when it already exists.
    pub private: bool,
    /// Branch to commit to; the default branch when `None`.
    pub revision: Option<String>,
    /// Commit summary; `"Upload <dir> with pmetal"` when `None`.
    pub commit_message: Option<String>,
    /// Open a pull request instead of committing to the branch.
    pub create_pr: bool,
    /// Extra glob patterns (relative to the directory) to leave out.
    pub ignore: Vec<String>,
    /// Connection settings (endpoint, retries, timeouts). The token passed to
    /// [`upload_model`] takes precedence over `hub.token`.
    pub hub: HubOptions,
    /// Called with progress snapshots (bytes, rate, ETA) while files transfer.
    pub progress: Option<ProgressCallback>,
}

impl std::fmt::Debug for UploadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadOptions")
            .field("private", &self.private)
            .field("revision", &self.revision)
            .field("commit_message", &self.commit_message)
            .field("create_pr", &self.create_pr)
            .field("ignore", &self.ignore)
            .field("hub", &self.hub)
            .field("progress", &self.progress.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

/// Where an upload landed.
#[derive(Debug, Clone)]
pub struct UploadReport {
    /// URL of the repository.
    pub repo_url: String,
    /// URL of the commit (or pull request), when the Hub returns one.
    pub commit_url: Option<String>,
}

/// The ignore globs for an upload: the always-ignored set plus the caller's.
fn ignore_patterns(extra: &[String]) -> Vec<String> {
    ALWAYS_IGNORED
        .iter()
        .map(|p| (*p).to_owned())
        .chain(extra.iter().cloned())
        .collect()
}

/// Upload the contents of `model_dir` to the model repository `repo_id`
/// (`owner/name`), creating the repository if it does not exist.
///
/// `token` is a write token; `None` falls back to `options.hub.token`, then
/// `HF_TOKEN` or the token a prior `hf auth login` stored. Without any, the
/// call fails before contacting the Hub.
pub async fn upload_model(
    model_dir: impl AsRef<Path>,
    repo_id: &str,
    token: Option<&SecretString>,
    options: &UploadOptions,
) -> Result<UploadReport> {
    let model_dir = model_dir.as_ref();
    if !model_dir.is_dir() {
        return Err(PMetalError::InvalidArgument(format!(
            "'{}' is not a directory",
            model_dir.display()
        )));
    }
    let form_error = || {
        PMetalError::InvalidArgument(format!(
            "repository '{repo_id}' must be in the form 'owner/name'"
        ))
    };
    let (owner, name) = repo_id.split_once('/').ok_or_else(form_error)?;
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        return Err(form_error());
    }
    crate::repo::validate_repo_id(repo_id)?;

    let mut hub = options.hub.clone();
    if token.is_some() {
        hub.token = token.cloned();
    }
    if hub.is_offline() {
        return Err(PMetalError::Hub(
            "cannot upload in offline mode: publishing needs the network".into(),
        ));
    }
    if resolve_token(hub.token.as_ref()).is_none() {
        return Err(PMetalError::Hub(
            "uploading needs an access token with write permission: set HF_TOKEN, run \
             `hf auth login`, or pass a token"
                .into(),
        ));
    }
    let client = hub.client()?;

    let repo_url = client
        .create_repository()
        .repo_id(repo_id)
        .repo_type(RepoTypeModel)
        .private(options.private)
        .exist_ok(true)
        .send()
        .await
        .map_err(|e| hub_error_in(repo_id, &e))?
        .url;

    let commit_message = options.commit_message.clone().unwrap_or_else(|| {
        let dir_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_owned());
        format!("Upload {dir_name} with pmetal")
    });

    let commit = client
        .model(owner, name)
        .upload_folder()
        .folder_path(PathBuf::from(model_dir))
        .commit_message(commit_message)
        .maybe_revision(options.revision.clone())
        .create_pr(options.create_pr)
        .ignore_patterns(ignore_patterns(&options.ignore))
        .maybe_progress(progress_handle(&options.progress))
        .send()
        .await
        .map_err(|e| hub_error_in(repo_id, &e))?;

    Ok(UploadReport {
        repo_url,
        commit_url: commit.commit_url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_ignored_come_first_then_the_callers() {
        let patterns = ignore_patterns(&["checkpoints/**".to_owned()]);
        assert_eq!(patterns.len(), ALWAYS_IGNORED.len() + 1);
        assert_eq!(patterns.last().map(String::as_str), Some("checkpoints/**"));
        assert!(patterns.iter().any(|p| p == "**/._*"));
    }

    #[test]
    fn credential_files_are_never_published() {
        for secret in ["**/.env", "**/.env.*", "**/.netrc"] {
            assert!(ignore_patterns(&[]).iter().any(|p| p == secret), "{secret}");
        }
    }
}
