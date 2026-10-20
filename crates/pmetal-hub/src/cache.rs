//! Local cache management, in the standard Hub cache layout
//! (`models--owner--name/{blobs,refs,snapshots}`), shared with the Hub's own
//! tools: a model downloaded by one is found by the others.

use std::path::{Path, PathBuf};

use pmetal_core::Result;

use crate::verify::verify_model_dir;

/// Get the HuggingFace home directory, resolved the way the Python
/// `huggingface_hub` library does it.
///
/// Resolution order:
/// 1. `HF_HOME`
/// 2. `$XDG_CACHE_HOME/huggingface`
/// 3. `~/.cache/huggingface` — default on all platforms
pub(crate) fn hf_home() -> PathBuf {
    if let Some(home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(home);
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("huggingface");
    }
    dirs::home_dir()
        .unwrap_or_default()
        .join(".cache")
        .join("huggingface")
}

/// Get the HuggingFace hub cache directory.
///
/// Every download goes through a client pointed at this directory, so the
/// cache lookups here and the downloads agree by construction.
///
/// Resolution order:
/// 1. `HF_HUB_CACHE` — direct override for the hub cache directory
/// 2. `HUGGINGFACE_HUB_CACHE` — its legacy name
/// 3. `<HF home>/hub`
pub fn cache_dir() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("hub"))
}

/// Get the HuggingFace datasets cache directory.
///
/// Resolution order:
/// 1. `HF_DATASETS_CACHE` — direct override
/// 2. `<HF home>/datasets`
pub fn datasets_cache_dir() -> PathBuf {
    std::env::var_os("HF_DATASETS_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("datasets"))
}

/// Get the pmetal-specific cache directory (for non-HF local state).
pub fn pmetal_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .map(|p| p.join("pmetal"))
        .unwrap_or_else(|| PathBuf::from(".cache/pmetal"))
}

/// The kind of repository a cache folder holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RepoKind {
    Model,
    Dataset,
}

impl RepoKind {
    fn prefix(self) -> &'static str {
        match self {
            RepoKind::Model => "models",
            RepoKind::Dataset => "datasets",
        }
    }
}

/// `models--owner--name`: the cache folder of a repository.
pub(crate) fn repo_folder(kind: RepoKind, repo_id: &str) -> String {
    format!("{}--{}", kind.prefix(), repo_id.replace('/', "--"))
}

/// Whether `revision` is a full commit id (40 hex digits), not a name.
pub(crate) fn is_commit_sha(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The commit a cached revision name points at (`refs/<name>`).
pub(crate) fn read_ref(cache_root: &Path, folder: &str, revision: &str) -> Option<String> {
    if !crate::select::is_safe_relative_path(revision) {
        return None;
    }
    let sha = std::fs::read_to_string(cache_root.join(folder).join("refs").join(revision)).ok()?;
    let sha = sha.trim();
    is_commit_sha(sha).then(|| sha.to_owned())
}

/// Record that `revision` currently points at `sha` (`refs/<revision>`), as
/// the Hub's own tools do, so later offline lookups of the name find the
/// snapshot. Written to a temporary file and renamed into place.
pub(crate) fn write_ref(
    cache_root: &Path,
    folder: &str,
    revision: &str,
    sha: &str,
) -> std::io::Result<()> {
    // A revision name is never a path: refuse anything that would climb out
    // of `refs/`.
    if !crate::select::is_safe_relative_path(revision) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("'{revision}' is not a usable revision name"),
        ));
    }
    let path = cache_root.join(folder).join("refs").join(revision);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Beside the ref, named after it: `v1.0` and `v1.1` must not share one.
    let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(".pmetal-tmp");
    let tmp = path.with_file_name(tmp_name);
    std::fs::write(&tmp, sha)?;
    std::fs::rename(&tmp, &path)
}

/// The snapshot directory of a commit.
pub(crate) fn snapshot_dir(cache_root: &Path, folder: &str, sha: &str) -> PathBuf {
    cache_root.join(folder).join("snapshots").join(sha)
}

/// The name of the file that marks a snapshot a download is still working on.
const IN_PROGRESS_MARKER: &str = ".pmetal-incomplete";

/// A snapshot a download has begun and not yet checked. The marker is written
/// before the first byte moves and removed by [`finish`](Self::finish) once
/// every file has been verified, so a snapshot left half-filled by an error,
/// a crash or a killed process is never taken for a finished one: lookups skip
/// it, and the next download fills it in.
pub(crate) struct InProgress(PathBuf);

impl InProgress {
    /// Mark `snapshot` as being downloaded.
    pub(crate) fn begin(snapshot: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(snapshot)?;
        let marker = snapshot.join(IN_PROGRESS_MARKER);
        std::fs::write(&marker, b"")?;
        Ok(Self(marker))
    }

    /// The download is complete and verified: clear the mark.
    pub(crate) fn finish(self) {
        let _ = std::fs::remove_file(self.0);
    }
}

/// Pick a cached snapshot of a repository that `accept`s. A snapshot a
/// download has not finished is never accepted.
///
/// With a `revision`: that commit, or the commit the revision name points at.
/// Without: what `refs/main` points at first, then every snapshot, newest
/// first. `None` when nothing acceptable is cached.
pub(crate) fn find_snapshot(
    cache_root: &Path,
    kind: RepoKind,
    repo_id: &str,
    revision: Option<&str>,
    accept: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let folder = repo_folder(kind, repo_id);
    let mut candidates: Vec<PathBuf> = Vec::new();
    match revision {
        Some(rev) if is_commit_sha(rev) => candidates.push(snapshot_dir(cache_root, &folder, rev)),
        Some(rev) => {
            if let Some(sha) = read_ref(cache_root, &folder, rev) {
                candidates.push(snapshot_dir(cache_root, &folder, &sha));
            }
        }
        None => {
            if let Some(sha) = read_ref(cache_root, &folder, "main") {
                candidates.push(snapshot_dir(cache_root, &folder, &sha));
            }
            let mut others: Vec<_> = std::fs::read_dir(cache_root.join(&folder).join("snapshots"))
                .ok()?
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| {
                    let modified = e
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    (modified, e.path())
                })
                .collect();
            others.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
            candidates.extend(others.into_iter().map(|(_, p)| p));
        }
    }
    candidates
        .into_iter()
        .find(|p| p.is_dir() && !p.join(IN_PROGRESS_MARKER).exists() && accept(p))
}

/// Evict a single model from the local HuggingFace hub cache.
///
/// Removes the cache directory for `repo_id` (e.g., `"Qwen/Qwen3-0.6B"`).
/// The directory is resolved as `<cache_dir>/models--<org>--<name>`, matching
/// the layout used by `huggingface_hub` and the `hf-hub` crate.
///
/// Returns `Ok(())` if the directory did not exist (idempotent).
pub fn evict_model(repo_id: &str) -> Result<()> {
    let dir = cache_dir().join(repo_folder(RepoKind::Model, repo_id));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(())
}

/// Check if a model is already cached locally and return its snapshot path.
///
/// A snapshot counts only if [`verify_model_dir`] finds it complete: a
/// readable `config.json` and every weight file present and the length its
/// header declares. That keeps a download interrupted before the last shard
/// from being mistaken for a model. Prefers the snapshot `refs/main` points at.
/// This is a fast local-only check: no network calls, and only file headers are
/// read.
pub fn find_cached_model(repo_id: &str) -> Option<PathBuf> {
    find_cached_model_in(&cache_dir(), repo_id, None)
}

/// [`find_cached_model`] in a given cache directory, for one revision (a
/// commit id or a branch or tag name the cache has seen) or, with `None`, the
/// best snapshot.
pub fn find_cached_model_in(
    cache_root: &Path,
    repo_id: &str,
    revision: Option<&str>,
) -> Option<PathBuf> {
    find_snapshot(cache_root, RepoKind::Model, repo_id, revision, |snapshot| {
        verify_model_dir(snapshot).is_loadable_language_model()
    })
}

/// A model repository in the local cache, and the snapshot to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedModel {
    /// The repository id (`owner/name`).
    pub repo_id: String,
    /// The snapshot directory: what `refs/main` points at, else the newest.
    pub snapshot: PathBuf,
}

/// A dataset repository in the local cache, and the snapshot to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedDataset {
    /// The repository id (`owner/name`).
    pub repo_id: String,
    /// The snapshot directory: what `refs/main` points at, else the newest.
    pub snapshot: PathBuf,
}

/// Every model repository in the local cache that has a finished snapshot,
/// ordered by repository id.
///
/// Local-only and cheap: it reads directory listings and refs, not weights, so
/// a repository whose download left a snapshot the loader will reject is still
/// listed (use [`find_cached_model`] to ask whether one can be loaded). A
/// snapshot a download has not finished is skipped.
pub fn list_cached_models() -> Vec<CachedModel> {
    list_cached_models_in(&cache_dir())
}

/// [`list_cached_models`] in a given cache directory.
pub fn list_cached_models_in(cache_root: &Path) -> Vec<CachedModel> {
    list_repos(cache_root, RepoKind::Model)
        .into_iter()
        .map(|(repo_id, snapshot)| CachedModel { repo_id, snapshot })
        .collect()
}

/// Every dataset repository in the local cache that has a finished snapshot,
/// ordered by repository id. See [`list_cached_models`].
pub fn list_cached_datasets() -> Vec<CachedDataset> {
    list_cached_datasets_in(&cache_dir())
}

/// [`list_cached_datasets`] in a given cache directory.
pub fn list_cached_datasets_in(cache_root: &Path) -> Vec<CachedDataset> {
    list_repos(cache_root, RepoKind::Dataset)
        .into_iter()
        .map(|(repo_id, snapshot)| CachedDataset { repo_id, snapshot })
        .collect()
}

/// The `(repository id, snapshot)` of every finished repository of `kind`.
fn list_repos(cache_root: &Path, kind: RepoKind) -> Vec<(String, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(cache_root) else {
        return Vec::new();
    };
    let prefix = format!("{}--", kind.prefix());
    let mut repos: Vec<(String, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let folder = entry.file_name().into_string().ok()?;
            let encoded = folder.strip_prefix(&prefix)?;
            // The folder name is the repository id with `/` written `--`.
            let repo_id = encoded.replace("--", "/");
            let snapshot = find_snapshot(cache_root, kind, &repo_id, None, |s| {
                std::fs::read_dir(s).is_ok_and(|mut files| files.next().is_some())
            })?;
            Some((repo_id, snapshot))
        })
        .collect();
    repos.sort_by(|a, b| a.0.cmp(&b.0));
    repos
}

/// Check if a dataset is already cached locally and return its path.
///
/// Same as [`find_cached_model`] but for datasets, whose snapshots count as
/// soon as they hold a file.
pub fn find_cached_dataset(repo_id: &str) -> Option<PathBuf> {
    find_snapshot(&cache_dir(), RepoKind::Dataset, repo_id, None, |snapshot| {
        std::fs::read_dir(snapshot).is_ok_and(|mut entries| entries.next().is_some())
    })
}

/// Where a path sits in a Hub cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLocation {
    /// The cache directory (the one holding `models--owner--name`, the shared
    /// `blobs/` store and `.locks`).
    pub cache_root: PathBuf,
    /// The repository folder (`models--owner--name`) the path is inside.
    pub repo_dir: PathBuf,
}

impl CacheLocation {
    /// Whether `file` is stored by this cache: inside the repository folder,
    /// or in the cache-wide blob store that `huggingface_hub` shares between
    /// repositories so a file is kept once however many repositories hold it.
    ///
    /// Both paths should be canonical: a snapshot file is a symlink into
    /// `blobs/`, and it is what the link points at that has to be checked.
    pub fn stores(&self, file: &Path) -> bool {
        file.starts_with(&self.repo_dir) || file.starts_with(self.cache_root.join("blobs"))
    }
}

/// The Hub cache a path lies in, if it lies in one: the nearest ancestor named
/// like a cache repository folder (`models--…`, `datasets--…`, `spaces--…`)
/// and the cache directory above it.
///
/// Purely lexical: the path is not touched, so canonicalise it first to see
/// through symlinks.
pub fn locate_in_cache(path: &Path) -> Option<CacheLocation> {
    path.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?;
        let is_repo = ["models--", "datasets--", "spaces--"]
            .iter()
            .any(|prefix| name.starts_with(prefix));
        is_repo.then(|| CacheLocation {
            cache_root: dir.parent().map(Path::to_path_buf).unwrap_or_default(),
            repo_dir: dir.to_path_buf(),
        })
    })
}

/// The top-level entries of a Hub cache directory that belong to the cache:
/// repository folders and the lock directory.
fn is_cache_entry(name: &str) -> bool {
    name.starts_with("models--")
        || name.starts_with("datasets--")
        || name.starts_with("spaces--")
        || name == ".locks"
}

/// Clear the hub cache: every cached repository, model and dataset alike.
///
/// Removes only the cache's own entries (`models--*`, `datasets--*`,
/// `spaces--*`, `.locks`), never the directory itself or anything else that
/// happens to live in it, so a mis-set `HF_HUB_CACHE` cannot turn this into
/// "delete that directory".
pub fn clear_cache() -> Result<()> {
    clear_cache_in(&cache_dir())
}

fn clear_cache_in(cache: &Path) -> Result<()> {
    if !cache.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(cache)?.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !is_cache_entry(&name) {
            continue;
        }
        // A symlinked entry is unlinked, never followed.
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(entry.path())?;
        } else {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// Get cache size in bytes.
pub fn cache_size() -> Result<u64> {
    let cache = cache_dir();
    if !cache.exists() {
        return Ok(0);
    }

    let mut size = 0u64;
    for entry in walkdir::WalkDir::new(&cache).into_iter().flatten() {
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                size += metadata.len();
            }
        }
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::write_test_safetensors;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    /// A cache with `models--o--m` holding a complete snapshot at `sha`.
    fn put_model(root: &Path, sha: &str, with_weights: bool) -> PathBuf {
        let snap = snapshot_dir(root, "models--o--m", sha);
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), "{}").unwrap();
        if with_weights {
            write_test_safetensors(&snap.join("model.safetensors"), &[("w", "F16", &[4])]);
        }
        snap
    }

    #[test]
    fn a_snapshot_without_weights_is_not_cached() {
        // What an interrupted download leaves: the config, no weights.
        let root = tempfile::tempdir().unwrap();
        put_model(root.path(), SHA_A, false);
        assert!(find_cached_model_in(root.path(), "o/m", None).is_none());
    }

    #[test]
    fn a_snapshot_a_download_has_not_finished_is_not_cached() {
        let root = tempfile::tempdir().unwrap();
        let snap = put_model(root.path(), SHA_A, true);
        let mark = InProgress::begin(&snap).unwrap();
        // Weights and config are all there, but nobody has checked the rest.
        assert!(find_cached_model_in(root.path(), "o/m", None).is_none());
        assert!(find_cached_model_in(root.path(), "o/m", Some(SHA_A)).is_none());
        mark.finish();
        assert_eq!(find_cached_model_in(root.path(), "o/m", None), Some(snap));
    }

    #[test]
    fn a_snapshot_missing_a_shard_is_not_cached() {
        let root = tempfile::tempdir().unwrap();
        let snap = put_model(root.path(), SHA_A, false);
        write_test_safetensors(&snap.join("s1.safetensors"), &[("w", "F16", &[4])]);
        std::fs::write(
            snap.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"s1.safetensors","b":"s2.safetensors"}}"#,
        )
        .unwrap();
        assert!(find_cached_model_in(root.path(), "o/m", None).is_none());
    }

    #[test]
    fn the_cached_models_are_listed_by_repository_with_the_snapshot_main_points_at() {
        let root = tempfile::tempdir().unwrap();
        // `o/m`: two snapshots, `main` on the older.
        let wanted = put_model(root.path(), SHA_A, true);
        std::thread::sleep(std::time::Duration::from_millis(20));
        put_model(root.path(), SHA_B, true);
        write_ref(root.path(), "models--o--m", "main", SHA_A).unwrap();
        // A repository whose download is still going, one with nothing in it,
        // a dataset, and a file that is not a repository.
        let busy = snapshot_dir(root.path(), "models--a--busy", SHA_A);
        std::fs::create_dir_all(&busy).unwrap();
        std::fs::write(busy.join("config.json"), "{}").unwrap();
        let _mark = InProgress::begin(&busy).unwrap();
        std::fs::create_dir_all(snapshot_dir(root.path(), "models--a--empty", SHA_A)).unwrap();
        std::fs::create_dir_all(root.path().join("datasets--d--x/snapshots")).unwrap();
        std::fs::write(root.path().join("models--stray"), "x").unwrap();
        // A repository without an owner, and a model with a late-sorting id.
        let plain = snapshot_dir(root.path(), "models--gpt2", SHA_A);
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("config.json"), "{}").unwrap();

        assert_eq!(
            list_cached_models_in(root.path()),
            [
                CachedModel {
                    repo_id: "gpt2".into(),
                    snapshot: plain
                },
                CachedModel {
                    repo_id: "o/m".into(),
                    snapshot: wanted
                },
            ]
        );
        assert!(list_cached_models_in(&root.path().join("missing")).is_empty());
    }

    #[test]
    fn the_cached_datasets_are_listed_apart_from_the_models() {
        let root = tempfile::tempdir().unwrap();
        put_model(root.path(), SHA_A, true);
        let data = snapshot_dir(root.path(), "datasets--org--corpus", SHA_A);
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("train.jsonl"), "{}\n").unwrap();
        assert_eq!(
            list_cached_datasets_in(root.path()),
            [CachedDataset {
                repo_id: "org/corpus".into(),
                snapshot: data
            }]
        );
        assert_eq!(list_cached_models_in(root.path()).len(), 1);
    }

    #[test]
    fn a_complete_snapshot_is_found() {
        let root = tempfile::tempdir().unwrap();
        let snap = put_model(root.path(), SHA_A, true);
        assert_eq!(find_cached_model_in(root.path(), "o/m", None), Some(snap));
    }

    #[test]
    fn main_wins_over_a_newer_snapshot_of_another_commit() {
        let root = tempfile::tempdir().unwrap();
        let wanted = put_model(root.path(), SHA_A, true);
        std::thread::sleep(std::time::Duration::from_millis(20));
        put_model(root.path(), SHA_B, true); // newer on disk
        write_ref(root.path(), "models--o--m", "main", SHA_A).unwrap();
        assert_eq!(find_cached_model_in(root.path(), "o/m", None), Some(wanted));
    }

    #[test]
    fn a_revision_resolves_through_refs_or_as_a_commit() {
        let root = tempfile::tempdir().unwrap();
        let a = put_model(root.path(), SHA_A, true);
        let b = put_model(root.path(), SHA_B, true);
        write_ref(root.path(), "models--o--m", "v1", SHA_A).unwrap();
        assert_eq!(
            find_cached_model_in(root.path(), "o/m", Some("v1")),
            Some(a)
        );
        assert_eq!(
            find_cached_model_in(root.path(), "o/m", Some(SHA_B)),
            Some(b)
        );
        assert!(find_cached_model_in(root.path(), "o/m", Some("never-seen")).is_none());
    }

    #[test]
    fn a_ref_cannot_climb_out_of_the_cache() {
        let root = tempfile::tempdir().unwrap();
        assert!(write_ref(root.path(), "models--o--m", "../../escape", SHA_A).is_err());
        assert!(write_ref(root.path(), "models--o--m", "a//b", SHA_A).is_err());
        write_ref(root.path(), "models--o--m", "refs/pr/1", SHA_A).unwrap();
        assert_eq!(
            read_ref(root.path(), "models--o--m", "refs/pr/1").as_deref(),
            Some(SHA_A)
        );
    }

    #[test]
    fn a_ref_name_cannot_read_outside_the_cache_either() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("models--o--m/refs")).unwrap();
        // A file elsewhere that happens to hold a commit id.
        std::fs::write(root.path().join("elsewhere"), SHA_A).unwrap();
        assert_eq!(
            read_ref(root.path(), "models--o--m", "../../elsewhere"),
            None
        );
        assert_eq!(read_ref(root.path(), "models--o--m", "/etc/hosts"), None);
        // And dotted release names write distinct files.
        write_ref(root.path(), "models--o--m", "v1.0", SHA_A).unwrap();
        write_ref(root.path(), "models--o--m", "v1.1", SHA_B).unwrap();
        assert_eq!(
            read_ref(root.path(), "models--o--m", "v1.0").as_deref(),
            Some(SHA_A)
        );
        assert_eq!(
            read_ref(root.path(), "models--o--m", "v1.1").as_deref(),
            Some(SHA_B)
        );
    }

    #[test]
    fn commit_ids_are_forty_hex_digits() {
        assert!(is_commit_sha(SHA_A));
        assert!(!is_commit_sha("main"));
        assert!(!is_commit_sha(&SHA_A[..39]));
        assert!(!is_commit_sha(&"g".repeat(40)));
    }

    #[test]
    fn repo_folders_follow_the_hub_layout() {
        assert_eq!(
            repo_folder(RepoKind::Model, "Qwen/Qwen3-0.6B"),
            "models--Qwen--Qwen3-0.6B"
        );
        assert_eq!(
            repo_folder(RepoKind::Dataset, "tatsu-lab/alpaca"),
            "datasets--tatsu-lab--alpaca"
        );
    }

    #[test]
    fn clearing_removes_repositories_and_leaves_everything_else() {
        // A mis-set HF_HUB_CACHE pointing at a directory with real contents.
        let root = tempfile::tempdir().unwrap();
        put_model(root.path(), SHA_A, true);
        std::fs::create_dir_all(root.path().join("datasets--o--d/snapshots")).unwrap();
        std::fs::create_dir_all(root.path().join(".locks/models--o--m")).unwrap();
        std::fs::create_dir_all(root.path().join("Documents")).unwrap();
        std::fs::write(root.path().join("Documents/thesis.txt"), "keep").unwrap();
        std::fs::write(root.path().join("notes.txt"), "keep").unwrap();

        clear_cache_in(root.path()).unwrap();

        assert!(!root.path().join("models--o--m").exists());
        assert!(!root.path().join("datasets--o--d").exists());
        assert!(!root.path().join(".locks").exists());
        assert!(root.path().join("Documents/thesis.txt").exists());
        assert!(root.path().join("notes.txt").exists());
        assert!(root.path().is_dir());
    }

    #[test]
    fn clearing_a_missing_cache_is_fine() {
        clear_cache_in(Path::new("/definitely/not/here")).unwrap();
    }

    #[test]
    fn a_snapshot_path_locates_its_repository_and_cache() {
        let snap = Path::new("/data/hub/models--o--m/snapshots/abc");
        let location = locate_in_cache(snap).unwrap();
        assert_eq!(location.cache_root, Path::new("/data/hub"));
        assert_eq!(location.repo_dir, Path::new("/data/hub/models--o--m"));
        assert!(locate_in_cache(Path::new("/data/plain/model")).is_none());
    }

    #[test]
    fn a_location_stores_its_own_blobs_and_the_shared_store_but_nothing_else() {
        let location = locate_in_cache(Path::new("/data/hub/models--o--m/snapshots/abc")).unwrap();
        assert!(location.stores(Path::new("/data/hub/models--o--m/blobs/1234")));
        assert!(location.stores(Path::new("/data/hub/blobs/ab/1234")));
        // Another repository's own blobs are not this one's.
        assert!(!location.stores(Path::new("/data/hub/models--x--y/blobs/1234")));
        assert!(!location.stores(Path::new("/etc/passwd")));
        // A sibling that merely shares a name prefix is not the repository.
        assert!(!location.stores(Path::new("/data/hub/models--o--m2/blobs/1234")));
    }
}
