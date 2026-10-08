//! Publishing a local model directory to the HuggingFace Hub.
//!
//! The Hub protocol lives in `hf-hub`: it asks the preupload endpoint which
//! files the repository stores as regular git blobs and which go to large-file
//! storage, streams the large ones to Xet storage, and writes everything in
//! one commit. This module decides what to send and in which repository.

use std::path::{Path, PathBuf};

use hf_hub::RepoTypeModel;
use pmetal_core::{PMetalError, Result, SecretString};

use crate::download::{build_client, hub_error};

/// Files never worth publishing: VCS metadata, Finder and AppleDouble
/// droppings (exFAT and network volumes grow a `._name` beside every file),
/// and the Hub's own download cache.
const ALWAYS_IGNORED: &[&str] = &[
    ".git/**",
    "**/.DS_Store",
    "**/._*",
    ".cache/**",
    "**/.cache/**",
];

/// How to publish a directory.
#[derive(Debug, Clone, Default)]
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
/// `token` is a write token; `None` falls back to `HF_TOKEN` or the token a
/// prior `hf auth login` stored.
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
    let Some((owner, name)) = repo_id.split_once('/') else {
        return Err(PMetalError::InvalidArgument(format!(
            "repository '{repo_id}' must be in the form 'owner/name'"
        )));
    };
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        return Err(PMetalError::InvalidArgument(format!(
            "repository '{repo_id}' must be in the form 'owner/name'"
        )));
    }

    let client = build_client(token)?;

    let repo_url = client
        .create_repository()
        .repo_id(repo_id)
        .repo_type(RepoTypeModel)
        .private(options.private)
        .exist_ok(true)
        .send()
        .await
        .map_err(hub_error)?
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
        .send()
        .await
        .map_err(hub_error)?;

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
}
