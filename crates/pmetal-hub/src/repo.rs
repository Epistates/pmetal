//! One handle over a model or a dataset repository: the operations a
//! download needs, with Hub failures turned into user-facing errors.

use std::path::PathBuf;

use futures::StreamExt;
use hf_hub::progress::Progress;
use hf_hub::{HFError, HFRepository, RepoTypeDataset, RepoTypeModel, split_id};
use pmetal_core::{PMetalError, Result};
use serde::Deserialize;

use crate::api::{ApiError, get_page};
use crate::cache::{InProgress, RepoKind, repo_folder, snapshot_dir};
use crate::client::{HubOptions, hub_error_in};
use crate::select::RepoFile;

/// The most pages of a file listing read before giving up (the Hub's pages
/// hold on the order of a thousand entries).
const MAX_TREE_PAGES: usize = 1000;
/// The most bytes read from one page of a file listing.
const MAX_TREE_PAGE_BYTES: usize = 32 * 1024 * 1024;

enum Inner {
    Model(HFRepository<RepoTypeModel>),
    Dataset(HFRepository<RepoTypeDataset>),
}

/// A repository on the Hub, addressed by `owner/name`.
pub(crate) struct RepoHandle {
    inner: Inner,
    options: HubOptions,
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
    pub(crate) fn model(options: &HubOptions, id: &str) -> Result<Self> {
        validate_repo_id(id)?;
        let (owner, name) = split_id(id);
        Ok(Self {
            inner: Inner::Model(options.client()?.model(owner, name)),
            options: options.clone(),
            id: id.to_owned(),
            kind: RepoKind::Model,
        })
    }

    pub(crate) fn dataset(options: &HubOptions, id: &str) -> Result<Self> {
        validate_repo_id(id)?;
        let (owner, name) = split_id(id);
        Ok(Self {
            inner: Inner::Dataset(options.client()?.dataset(owner, name)),
            options: options.clone(),
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
    ///
    /// The listing is read from the Hub's tree endpoint directly, page by
    /// page: the download client's typed listing drops the digest (the Hub
    /// names it `lfs.oid`), and a listing without digests would let a
    /// corrupted download pass `Verification::Sha256`.
    pub(crate) async fn list_files(
        &self,
        commit: &str,
    ) -> std::result::Result<Vec<RepoFile>, ApiError> {
        let endpoint = self.options.resolved_endpoint();
        let kind = match self.kind {
            RepoKind::Model => "models",
            RepoKind::Dataset => "datasets",
        };
        let first = format!("{endpoint}/api/{kind}/{}/tree/{commit}", self.id);
        let origin = Origin::of(&first).map_err(ApiError::definite)?;
        let mut query = vec![("recursive".to_owned(), "true".to_owned())];
        let mut url = first;
        let mut seen = std::collections::HashSet::new();
        let mut files = Vec::new();
        for _ in 0..MAX_TREE_PAGES {
            let page = get_page(
                &self.options,
                &url,
                &query,
                MAX_TREE_PAGE_BYTES,
                Some(&self.id),
            )
            .await?;
            for file in parse_tree_page(&page.body).map_err(|e| {
                ApiError::definite(PMetalError::Hub(format!(
                    "the Hub's file listing of {} is unreadable: {e}",
                    self.id
                )))
            })? {
                // A listing that names a path out of the snapshot directory
                // is hostile or broken; its files are never fetched, checked
                // or removed.
                if !crate::select::is_safe_relative_path(&file.path) {
                    tracing::warn!("{}: ignoring the unsafe path {:?}", self.id, file.path);
                    continue;
                }
                files.push(file);
            }
            let Some(next) = page.next else {
                return Ok(files);
            };
            // The next page carries its own query (a cursor); the token goes
            // only to the Hub it was meant for.
            let next = origin.follow(&url, &next).map_err(ApiError::definite)?;
            if !seen.insert(next.clone()) {
                return Err(ApiError::definite(PMetalError::Hub(format!(
                    "the Hub's file listing of {} links to a page it already gave",
                    self.id
                ))));
            }
            url = next;
            query.clear();
        }
        Err(ApiError::definite(PMetalError::Hub(format!(
            "the Hub's file listing of {} has more than {MAX_TREE_PAGES} pages; refusing to read it",
            self.id
        ))))
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
    /// snapshot directory, and the mark that it is still being filled. The
    /// caller checks what arrived and then calls [`InProgress::finish`]; until
    /// then cache lookups skip the snapshot.
    pub(crate) async fn download_snapshot(
        &self,
        commit: &str,
        paths: &[String],
        max_workers: Option<usize>,
        progress: Option<Progress>,
    ) -> Result<(PathBuf, InProgress)> {
        let patterns: Vec<String> = paths.iter().map(|p| exact_glob(p)).collect();
        let cache_root = on_repo!(self, r => r.client().cache_dir().to_path_buf());
        let snapshot = snapshot_dir(&cache_root, &self.folder(), commit);
        let mark = InProgress::begin(&snapshot)?;
        let downloaded = on_repo!(self, r => r
            .snapshot_download()
            .revision(commit.to_owned())
            .allow_patterns(patterns)
            .maybe_max_workers(max_workers)
            .maybe_progress(progress)
            .send()
            .await
            .map_err(|e| self.err(&e)))?;
        Ok((downloaded, mark))
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

/// The scheme, host and port a listing may be read from: the Hub's own.
struct Origin(reqwest::Url);

impl Origin {
    fn of(url: &str) -> Result<Self> {
        reqwest::Url::parse(url)
            .map(Self)
            .map_err(|e| PMetalError::Hub(format!("the Hub address {url:?} is not a URL: {e}")))
    }

    /// `next` (absolute, or relative to `current`) if it stays on this
    /// origin; a listing must not send the credential to another host.
    fn follow(&self, current: &str, next: &str) -> Result<String> {
        let base = reqwest::Url::parse(current)
            .map_err(|e| PMetalError::Hub(format!("the Hub address {current:?}: {e}")))?;
        let target = base.join(next).map_err(|e| {
            PMetalError::Hub(format!("the Hub's next page {next:?} is not a URL: {e}"))
        })?;
        if target.origin() == self.0.origin() {
            Ok(target.into())
        } else {
            Err(PMetalError::Hub(format!(
                "the Hub's file listing points at another host ({}); refusing to follow it",
                target.host_str().unwrap_or("?")
            )))
        }
    }
}

/// One entry of the Hub's tree endpoint. Only what the downloader needs.
#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: Option<u64>,
    lfs: Option<TreeLfs>,
}

/// The large-file-storage record of a tree entry. Xet-backed files carry it
/// too (next to their `xetHash`); `oid` is the SHA-256 of the file's content.
#[derive(Deserialize)]
struct TreeLfs {
    oid: Option<String>,
}

/// The files of one page of the tree endpoint's answer.
fn parse_tree_page(body: &[u8]) -> std::result::Result<Vec<RepoFile>, String> {
    let entries: Vec<TreeEntry> = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    let mut files = Vec::new();
    for entry in entries.into_iter().filter(|e| e.kind == "file") {
        let size = entry
            .size
            .ok_or_else(|| format!("{} has no size", entry.path))?;
        // A digest that is not 64 hex digits can never match: refuse it here
        // instead of letting verification quietly skip the file.
        let sha256 = match entry.lfs.and_then(|l| l.oid) {
            None => None,
            Some(oid) if oid.len() == 64 && oid.bytes().all(|b| b.is_ascii_hexdigit()) => {
                Some(oid.to_ascii_lowercase())
            }
            Some(oid) => return Err(format!("{} has the malformed digest {oid:?}", entry.path)),
        };
        files.push(RepoFile {
            path: entry.path,
            size,
            sha256,
        });
    }
    Ok(files)
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

    const DIGEST: &str = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b";

    #[test]
    fn the_digest_of_a_large_file_is_the_lfs_oid() {
        // The shape the Hub answers with, an Xet-backed file included: the
        // top-level `oid` is the git blob id, `lfs.oid` the content's SHA-256.
        let body = format!(
            r#"[
              {{"type":"directory","oid":"aa","path":"sub"}},
              {{"type":"file","oid":"{}","size":11343,"path":"LICENSE"}},
              {{"type":"file","oid":"{}","size":1503300328,
                "lfs":{{"oid":"{}","size":1503300328,"pointerSize":135}},
                "xetHash":"6f87d3bf","path":"model.safetensors"}}
            ]"#,
            "6".repeat(40),
            "a".repeat(40),
            DIGEST.to_ascii_uppercase(),
        );
        let files = parse_tree_page(body.as_bytes()).unwrap();
        assert_eq!(files.len(), 2, "directories are not files");
        assert_eq!(files[0].sha256, None);
        assert_eq!(files[1].size, 1_503_300_328);
        assert_eq!(files[1].sha256.as_deref(), Some(DIGEST));
    }

    #[test]
    fn a_digest_that_cannot_match_is_refused_not_skipped() {
        let body = r#"[{"type":"file","oid":"x","size":5,"path":"w.bin",
                         "lfs":{"oid":"not-a-digest","size":5,"pointerSize":1}}]"#;
        let err = parse_tree_page(body.as_bytes()).unwrap_err();
        assert!(err.contains("w.bin") && err.contains("malformed"), "{err}");
        assert!(parse_tree_page(b"{}").is_err());
        assert!(parse_tree_page(br#"[{"type":"file","oid":"x","path":"a"}]"#).is_err());
    }

    #[test]
    fn a_listing_stays_on_the_hubs_own_origin() {
        let origin = Origin::of("http://hub.example:8080/api/models/o/m/tree/abc").unwrap();
        let here = "http://hub.example:8080/api/models/o/m/tree/abc";
        assert_eq!(
            origin
                .follow(
                    here,
                    "http://hub.example:8080/api/models/o/m/tree/abc?cursor=1"
                )
                .unwrap(),
            "http://hub.example:8080/api/models/o/m/tree/abc?cursor=1"
        );
        // Relative links resolve against the current page.
        assert_eq!(
            origin.follow(here, "/api/x?cursor=2").unwrap(),
            "http://hub.example:8080/api/x?cursor=2"
        );
        for elsewhere in [
            "http://evil.example:8080/api/x",
            "http://hub.example:9090/api/x",
            "https://hub.example:8080/api/x",
            "http://hub.example.evil.example:8080/api/x",
        ] {
            assert!(origin.follow(here, elsewhere).is_err(), "{elsewhere}");
        }
    }

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
