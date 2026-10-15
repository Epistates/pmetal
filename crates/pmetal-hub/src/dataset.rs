//! Dataset downloads from the HuggingFace Hub.

use std::path::{Path, PathBuf};

use pmetal_core::{PMetalError, Result, SecretString};

use crate::cache::{is_commit_sha, write_ref};
use crate::download::{DownloadOptions, check_downloaded, progress_handle};
use crate::repo::{RepoHandle, is_network_unavailable};
use crate::select::RepoFile;

/// Dataset files worth downloading for local consumption.
const DATASET_EXTENSIONS: &[&str] = &[
    ".parquet", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".arrow",
];

/// Where the Hub keeps its automatic Parquet conversion of a dataset.
const PARQUET_CONVERSION_REVISION: &str = "refs/convert/parquet";

fn is_data_file(path: &str) -> bool {
    !path.starts_with('.')
        && !path.split('/').any(|c| c.starts_with('.'))
        && DATASET_EXTENSIONS.iter().any(|ext| path.ends_with(ext))
}

/// Whether `path` belongs to the split called `split`: some path component,
/// extension removed, is `split` or `split-…` (`train.parquet`, `train/0000.parquet`,
/// `data/train-00000-of-00004.parquet`). `train_sft` is another split, not `train`.
pub(crate) fn matches_split(path: &str, split: &str) -> bool {
    let prefix = format!("{split}-");
    path.split('/').any(|component| {
        let stem = component.split('.').next().unwrap_or(component);
        stem == split || stem.starts_with(&prefix)
    })
}

/// The files of a dataset to fetch: README and data files, narrowed to
/// `split` when some files belong to it (a dataset that does not name its
/// splits in its paths keeps all its data files).
pub(crate) fn select_dataset_files(files: &[RepoFile], split: Option<&str>) -> Vec<RepoFile> {
    let data: Vec<&RepoFile> = files.iter().filter(|f| is_data_file(&f.path)).collect();
    let narrowed: Vec<&RepoFile> = match split {
        Some(s) => {
            let named: Vec<&RepoFile> = data
                .iter()
                .copied()
                .filter(|f| matches_split(&f.path, s))
                .collect();
            if named.is_empty() { data } else { named }
        }
        None => data,
    };
    let mut out: Vec<RepoFile> = narrowed.into_iter().cloned().collect();
    out.extend(files.iter().filter(|f| f.path == "README.md").cloned());
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

/// The Parquet files of `split`, in shard order.
pub(crate) fn select_parquet_files(files: &[RepoFile], split: &str) -> Vec<RepoFile> {
    let mut out: Vec<RepoFile> = files
        .iter()
        .filter(|f| f.path.ends_with(".parquet") && matches_split(&f.path, split))
        .cloned()
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// The split names a dataset's Parquet files suggest, for an error message.
fn parquet_splits(files: &[RepoFile]) -> Vec<String> {
    let mut splits: Vec<String> = files
        .iter()
        .filter(|f| f.path.ends_with(".parquet"))
        .filter_map(|f| {
            let name = f.path.rsplit('/').next()?;
            let stem = name.split('.').next()?;
            // `train-00000-of-00004` is split `train`; `0000.parquet` inside a
            // `train/` directory is split `train`.
            let guess = if stem.chars().all(|c| c.is_ascii_digit()) {
                f.path.rsplit('/').nth(1)?
            } else {
                stem.split('-').next()?
            };
            Some(guess.to_owned())
        })
        .collect();
    splits.sort();
    splits.dedup();
    splits
}

/// Download a dataset from HuggingFace Hub.
///
/// Returns the path to the dataset directory containing downloaded files:
/// the README and every data file (`.parquet`, `.json`, `.jsonl`, `.csv`,
/// `.tsv`, `.txt`, `.arrow`), all from one commit.
///
/// # Arguments
/// * `dataset_id` - Dataset identifier (e.g., "tatsu-lab/alpaca")
/// * `split` - Optional dataset split name; when set, only the files that
///   belong to that split are fetched, if the dataset names its splits in
///   its file paths.
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
pub async fn download_dataset(
    dataset_id: &str,
    split: Option<&str>,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<PathBuf> {
    download_dataset_with(dataset_id, split, &DownloadOptions::new(revision, token)).await
}

/// [`download_dataset`] with explicit [`DownloadOptions`] (the connection
/// settings, revision, worker count and progress callback apply).
pub async fn download_dataset_with(
    dataset_id: &str,
    split: Option<&str>,
    options: &DownloadOptions,
) -> Result<PathBuf> {
    crate::repo::validate_repo_id(dataset_id)?;
    let cache_root = options.hub.resolved_cache_dir();
    if options.hub.is_offline() {
        return find_cached_dataset_in(&cache_root, dataset_id, options.revision.as_deref(), split)
            .ok_or_else(|| offline_error(dataset_id, &cache_root));
    }
    let client = options.hub.client()?;
    let repo = RepoHandle::dataset(&client, dataset_id)?;
    let commit = match repo.resolve_commit(options.revision.as_deref()).await {
        Ok(commit) => commit,
        Err(e) if is_network_unavailable(&e) => {
            return find_cached_dataset_in(
                &cache_root,
                dataset_id,
                options.revision.as_deref(),
                split,
            )
            .ok_or_else(|| repo.error(&e));
        }
        Err(e) => return Err(repo.error(&e)),
    };
    let files = repo.list_files(&commit).await.map_err(|e| repo.error(&e))?;
    let wanted = select_dataset_files(&files, split);
    if !wanted.iter().any(|f| is_data_file(&f.path)) {
        return Err(PMetalError::Hub(format!(
            "No dataset files found in {dataset_id} (looked for {})",
            DATASET_EXTENSIONS.join(", ")
        )));
    }
    let paths: Vec<String> = wanted.iter().map(|f| f.path.clone()).collect();
    let (snapshot, in_progress) = repo
        .download_snapshot(
            &commit,
            &paths,
            options.max_workers,
            progress_handle(&options.progress),
        )
        .await?;
    check_downloaded(&snapshot, &wanted, options.verify)?;
    record_ref(&cache_root, &repo, options.revision.as_deref(), &commit);
    in_progress.finish();
    Ok(snapshot)
}

fn offline_error(dataset_id: &str, cache_root: &Path) -> PMetalError {
    PMetalError::Hub(format!(
        "dataset {dataset_id} is not in the local cache ({}) and offline mode is on: download \
         it once with network access",
        cache_root.display()
    ))
}

/// A cached snapshot of the dataset that holds the requested split.
fn find_cached_dataset_in(
    cache_root: &Path,
    dataset_id: &str,
    revision: Option<&str>,
    split: Option<&str>,
) -> Option<PathBuf> {
    crate::cache::find_snapshot(
        cache_root,
        crate::cache::RepoKind::Dataset,
        dataset_id,
        revision,
        |snap| {
            let files = walk_files(snap);
            match split {
                Some(s) => files.iter().any(|f| is_data_file(f) && matches_split(f, s)),
                None => files.iter().any(|f| is_data_file(f)),
            }
        },
    )
}

/// Repo-relative paths of the files under `dir`, links followed.
fn walk_files(dir: &Path) -> Vec<String> {
    walkdir::WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .flatten()
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            e.path()
                .strip_prefix(dir)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
        })
        .collect()
}

fn record_ref(cache_root: &Path, repo: &RepoHandle, revision: Option<&str>, commit: &str) {
    let name = match revision {
        Some(r) if is_commit_sha(r) => return,
        Some(r) => r,
        None => "main",
    };
    if let Err(e) = write_ref(cache_root, &repo.folder(), name, commit) {
        tracing::warn!("Could not record {name} for {}: {e}", repo.id);
    }
}

/// Download dataset Parquet files from HuggingFace Hub.
///
/// Lists the dataset once and fetches every Parquet file that belongs to
/// `split` (`data/train-00000-of-00004.parquet`, `train/0000.parquet`,
/// `train.parquet`, ...), in shard order. A dataset without Parquet files of
/// its own falls back to the Hub's automatic conversion (the
/// `refs/convert/parquet` branch).
///
/// # Arguments
/// * `dataset_id` - Dataset identifier (e.g., "tatsu-lab/alpaca")
/// * `split` - Dataset split (e.g., "train", "test")
/// * `revision` - Optional revision/branch
/// * `token` - Optional authentication token (as SecretString for security)
///
/// # Returns
/// Paths of the downloaded Parquet files, in shard order.
pub async fn download_dataset_parquet(
    dataset_id: &str,
    split: &str,
    revision: Option<&str>,
    token: Option<&SecretString>,
) -> Result<Vec<PathBuf>> {
    download_dataset_parquet_with(dataset_id, split, &DownloadOptions::new(revision, token)).await
}

/// [`download_dataset_parquet`] with explicit [`DownloadOptions`].
pub async fn download_dataset_parquet_with(
    dataset_id: &str,
    split: &str,
    options: &DownloadOptions,
) -> Result<Vec<PathBuf>> {
    crate::repo::validate_repo_id(dataset_id)?;
    let cache_root = options.hub.resolved_cache_dir();
    if options.hub.is_offline() {
        let dir = find_cached_dataset_in(
            &cache_root,
            dataset_id,
            options.revision.as_deref(),
            Some(split),
        )
        .ok_or_else(|| offline_error(dataset_id, &cache_root))?;
        return parquet_in(&dir, split).ok_or_else(|| offline_error(dataset_id, &cache_root));
    }
    let client = options.hub.client()?;
    let repo = RepoHandle::dataset(&client, dataset_id)?;

    let primary = options.revision.as_deref();
    let mut revisions = vec![primary];
    if primary != Some(PARQUET_CONVERSION_REVISION) {
        revisions.push(Some(PARQUET_CONVERSION_REVISION));
    }
    for (attempt, revision) in revisions.into_iter().enumerate() {
        let is_conversion = attempt > 0;
        let commit = match repo.resolve_commit(revision).await {
            Ok(commit) => commit,
            // Not every dataset has a conversion branch.
            Err(hf_hub::HFError::RevisionNotFound { .. }) if is_conversion => continue,
            Err(e) => return Err(repo.error(&e)),
        };
        let files = repo.list_files(&commit).await.map_err(|e| repo.error(&e))?;
        let parquet = select_parquet_files(&files, split);
        if parquet.is_empty() {
            if is_conversion {
                return Err(no_parquet_error(dataset_id, split, &files));
            }
            continue;
        }
        let paths: Vec<String> = parquet.iter().map(|f| f.path.clone()).collect();
        let (snapshot, in_progress) = repo
            .download_snapshot(
                &commit,
                &paths,
                options.max_workers,
                progress_handle(&options.progress),
            )
            .await?;
        check_downloaded(&snapshot, &parquet, options.verify)?;
        if !is_conversion {
            record_ref(&cache_root, &repo, revision, &commit);
        }
        in_progress.finish();
        return Ok(paths.iter().map(|p| snapshot.join(p)).collect());
    }
    Err(PMetalError::Hub(format!(
        "No parquet files found for split '{split}' in dataset '{dataset_id}'"
    )))
}

fn no_parquet_error(dataset_id: &str, split: &str, files: &[RepoFile]) -> PMetalError {
    let splits = parquet_splits(files);
    let hint = if splits.is_empty() {
        String::new()
    } else {
        format!("; Parquet splits found: {}", splits.join(", "))
    };
    PMetalError::Hub(format!(
        "No parquet files found for split '{split}' in dataset '{dataset_id}'{hint}"
    ))
}

fn parquet_in(dir: &Path, split: &str) -> Option<Vec<PathBuf>> {
    let mut paths: Vec<String> = walk_files(dir)
        .into_iter()
        .filter(|f| f.ends_with(".parquet") && matches_split(f, split))
        .collect();
    paths.sort();
    (!paths.is_empty()).then(|| paths.into_iter().map(|p| dir.join(p)).collect())
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
    let options = DownloadOptions::new(revision, token);
    let client = options.hub.client()?;
    let repo = RepoHandle::dataset(&client, dataset_id)?;
    repo.download_file(revision, filename, options.hub.is_offline())
        .await
        .map_err(|e| repo.error(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(paths: &[&str]) -> Vec<RepoFile> {
        paths.iter().map(|p| RepoFile::new(*p, 10)).collect()
    }

    fn paths(files: &[RepoFile]) -> Vec<&str> {
        files.iter().map(|f| f.path.as_str()).collect()
    }

    #[test]
    fn splits_match_the_layouts_datasets_use() {
        for (path, split, yes) in [
            ("train.parquet", "train", true),
            ("data/train-00000-of-00004.parquet", "train", true),
            ("data/train-00000-of-00001-0123abcd.parquet", "train", true),
            ("train/0000.parquet", "train", true),
            ("default/train/0000.parquet", "train", true),
            ("en/train-0001.parquet", "train", true),
            ("data/test-00000-of-00001.parquet", "train", false),
            ("data/train_sft-00000-of-00001.parquet", "train", false),
            ("data/train_sft-00000-of-00001.parquet", "train_sft", true),
            ("data/trainer.parquet", "train", false),
            ("data/validation-00000.parquet", "val", false),
        ] {
            assert_eq!(matches_split(path, split), yes, "{path} / {split}");
        }
    }

    #[test]
    fn dataset_files_are_data_and_the_readme() {
        let listing = files(&[
            "README.md",
            ".gitattributes",
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
            "data/test-00000-of-00001.parquet",
            "images/cat.png",
            "script.py",
        ]);
        let all = select_dataset_files(&listing, None);
        assert_eq!(
            paths(&all),
            [
                "README.md",
                "data/test-00000-of-00001.parquet",
                "data/train-00000-of-00002.parquet",
                "data/train-00001-of-00002.parquet",
            ]
        );
        let train = select_dataset_files(&listing, Some("train"));
        assert_eq!(
            paths(&train),
            [
                "README.md",
                "data/train-00000-of-00002.parquet",
                "data/train-00001-of-00002.parquet",
            ]
        );
    }

    #[test]
    fn a_split_the_paths_do_not_name_keeps_every_data_file() {
        let listing = files(&["data.jsonl", "README.md"]);
        let selected = select_dataset_files(&listing, Some("train"));
        assert_eq!(paths(&selected), ["README.md", "data.jsonl"]);
    }

    #[test]
    fn parquet_files_come_in_shard_order() {
        let listing = files(&[
            "data/train-00002-of-00003.parquet",
            "data/train-00000-of-00003.parquet",
            "data/train-00001-of-00003.parquet",
            "data/test-00000-of-00001.parquet",
            "data/train.json",
        ]);
        assert_eq!(
            paths(&select_parquet_files(&listing, "train")),
            [
                "data/train-00000-of-00003.parquet",
                "data/train-00001-of-00003.parquet",
                "data/train-00002-of-00003.parquet",
            ]
        );
    }

    #[test]
    fn the_error_for_a_missing_split_names_the_ones_that_exist() {
        let listing = files(&[
            "data/train-00000-of-00001.parquet",
            "validation/0000.parquet",
        ]);
        let err = no_parquet_error("o/d", "test", &listing).to_string();
        assert!(err.contains("train") && err.contains("validation"), "{err}");
    }
}
