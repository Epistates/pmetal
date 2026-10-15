//! One handle over a model or a dataset repository: the operations a
//! download needs, with Hub failures turned into user-facing errors.

use std::path::PathBuf;

use futures::StreamExt;
use hf_hub::progress::Progress;
use hf_hub::repository::RepoTreeEntry;
use hf_hub::{HFClient, HFError, HFRepository, RepoTypeDataset, RepoTypeModel, split_id};
use pmetal_core::{PMetalError, Result};

use crate::cache::{RepoKind, repo_folder};
use crate::client::hub_error_in;
use crate::select::RepoFile;

enum Inner {
    Model(HFRepository<RepoTypeModel>),
    Dataset(HFRepository<RepoTypeDataset>),
}

/// A repository on the Hub, addressed by `owner/name`.
pub(crate) struct RepoHandle {
    inner: Inner,
    pub(crate) id: String,
    pub(crate) kind: RepoKind,
}

/// Run `$body` with `$repo` bound to the concrete repository.
macro_rules! on_repo {
    ($self:expr, $repo:ident => $body:expr) => {
        match &$self.inner {
            Inner::Model($repo) => $body,
            Inner::Dataset($repo) => $body,
        }
    };
}

/// Whether `id` is a repository id the Hub could have: `name` or
/// `owner/name`, with the characters the Hub allows. Rejects everything that
/// could mean something else in a path or URL.
pub(crate) fn validate_repo_id(id: &str) -> Result<()> {
    let ok_part = |p: &str| {
        !p.is_empty()
            && p.chars().any(|c| c != '.')
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    let parts: Vec<&str> = id.split('/').collect();
    if matches!(parts.len(), 1 | 2) && parts.iter().all(|p| ok_part(p)) {
        Ok(())
    } else {
        Err(PMetalError::InvalidArgument(format!(
            "'{id}' is not a Hugging Face repository id (expected 'owner/name')"
        )))
    }
}

impl RepoHandle {
    pub(crate) fn model(client: &HFClient, id: &str) -> Result<Self> {
        validate_repo_id(id)?;
        let (owner, name) = split_id(id);
        Ok(Self {
            inner: Inner::Model(client.model(owner, name)),
            id: id.to_owned(),
            kind: RepoKind::Model,
        })
    }

    pub(crate) fn dataset(client: &HFClient, id: &str) -> Result<Self> {
        validate_repo_id(id)?;
        let (owner, name) = split_id(id);
        Ok(Self {
            inner: Inner::Dataset(client.dataset(owner, name)),
            id: id.to_owned(),
            kind: RepoKind::Dataset,
        })
    }

    /// The cache folder name of this repository.
    pub(crate) fn folder(&self) -> String {
        repo_folder(self.kind, &self.id)
    }

    fn err(&self, e: &HFError) -> PMetalError {
        hub_error_in(&self.id, e)
    }

    /// The commit `revision` (a branch, tag or commit id; the default branch
    /// when `None`) points at right now.
    pub(crate) async fn resolve_commit(
        &self,
        revision: Option<&str>,
    ) -> std::result::Result<String, HFError> {
        let sha = match &self.inner {
            Inner::Model(r) => {
                r.info()
                    .maybe_revision(revision.map(str::to_owned))
                    .send()
                    .await?
                    .sha
            }
            Inner::Dataset(r) => {
                r.info()
                    .maybe_revision(revision.map(str::to_owned))
                    .send()
                    .await?
                    .sha
            }
        };
        sha.ok_or_else(|| HFError::malformed_response("the repository info has no commit id"))
    }

    /// Every file of the repository at `commit`, with sizes and (for
    /// large-file-storage files) content hashes.
    pub(crate) async fn list_files(
        &self,
        commit: &str,
    ) -> std::result::Result<Vec<RepoFile>, HFError> {
        macro_rules! collect {
            ($repo:expr) => {{
                let stream = $repo
                    .list_tree()
                    .revision(commit.to_owned())
                    .recursive(true)
                    .send()?;
                futures::pin_mut!(stream);
                let mut files = Vec::new();
                while let Some(entry) = stream.next().await {
                    if let RepoTreeEntry::File {
                        path, size, lfs, ..
                    } = entry?
                    {
                        files.push(RepoFile {
                            path,
                            size,
                            sha256: lfs.and_then(|l| l.sha256),
                        });
                    }
                }
                Ok(files)
            }};
        }
        match &self.inner {
            Inner::Model(r) => collect!(r),
            Inner::Dataset(r) => collect!(r),
        }
    }

    /// The bytes of one file at `commit`, read into memory (indexes, configs,
    /// the README) without storing it in the cache. A file larger than
    /// `max_bytes` is an error, not an allocation.
    pub(crate) async fn read_file(
        &self,
        commit: &str,
        path: &str,
        max_bytes: usize,
    ) -> Result<Vec<u8>> {
        on_repo!(self, r => {
            let (_, mut stream) = r
                .download_file_stream()
                .filename(path)
                .revision(commit)
                .send()
                .await
                .map_err(|e| self.err(&e))?;
            let mut body = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| self.err(&e))?;
                if body.len() + chunk.as_ref().len() > max_bytes {
                    return Err(PMetalError::Hub(format!(
                        "{path} in {} is larger than {max_bytes} bytes; refusing to read it",
                        self.id
                    )));
                }
                body.extend_from_slice(chunk.as_ref());
            }
            Ok(body)
        })
    }

    /// Download one file of `commit` into the cache; its snapshot path.
    pub(crate) async fn download_file(
        &self,
        revision: Option<&str>,
        path: &str,
        local_only: bool,
    ) -> std::result::Result<PathBuf, HFError> {
        on_repo!(self, r => r
            .download_file()
            .filename(path)
            .maybe_revision(revision.map(str::to_owned))
            .local_files_only(local_only)
            .send()
            .await)
    }

    /// Download exactly `paths` of `commit` into the cache, concurrently; the
    /// snapshot directory.
    pub(crate) async fn download_snapshot(
        &self,
        commit: &str,
        paths: &[String],
        max_workers: Option<usize>,
        progress: Option<Progress>,
    ) -> Result<PathBuf> {
        let patterns: Vec<String> = paths.iter().map(|p| exact_glob(p)).collect();
        on_repo!(self, r => r
            .snapshot_download()
            .revision(commit.to_owned())
            .allow_patterns(patterns)
            .maybe_max_workers(max_workers)
            .maybe_progress(progress)
            .send()
            .await
            .map_err(|e| self.err(&e)))
    }

    /// Map a Hub failure on this repository to a user-facing error.
    pub(crate) fn error(&self, e: &HFError) -> PMetalError {
        self.err(e)
    }
}

/// A glob that matches exactly `path`: every glob metacharacter escaped.
pub(crate) fn exact_glob(path: &str) -> String {
    let mut glob = String::with_capacity(path.len());
    for c in path.chars() {
        if matches!(c, '*' | '?' | '[' | ']' | '{' | '}' | '\\' | '!') {
            glob.push('\\');
        }
        glob.push(c);
    }
    glob
}

/// Whether a failure means the network (or the Hub) is unavailable, as opposed
/// to a definite answer from the Hub such as "no such repository": the cases
/// where serving from the cache is better than failing.
pub(crate) fn is_network_unavailable(e: &HFError) -> bool {
    match e {
        HFError::Request { source, .. } => source.is_timeout() || source.is_connect(),
        HFError::Http { context } => matches!(context.status.as_u16(), 500 | 502 | 503 | 504),
        HFError::RateLimited { .. } => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_ids_are_validated() {
        for ok in [
            "gpt2",
            "Qwen/Qwen3-0.6B",
            "meta-llama/Llama-3.2-1B",
            "a_b/c.d-e",
        ] {
            assert!(validate_repo_id(ok).is_ok(), "{ok}");
        }
        for bad in [
            "", "/", "a/", "/a", "a/b/c", "..", "a/..", "../etc", "a b/c", "a/b?x=1", "a/b#c",
            "a\\b", "ａ/b",
        ] {
            assert!(validate_repo_id(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn globs_match_the_file_name_literally() {
        assert_eq!(
            exact_glob("model-00001-of-00002.safetensors"),
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(exact_glob("a[1]*?.bin"), r"a\[1\]\*\?.bin");
        assert_eq!(exact_glob("dir/{x}.gguf"), r"dir/\{x\}.gguf");
    }
}
