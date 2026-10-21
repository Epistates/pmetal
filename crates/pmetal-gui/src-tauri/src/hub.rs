//! Talking to Hugging Face: searching, downloading, and what is cached.
//!
//! Model search, the cache and downloads are `pmetal-hub`'s, the same code the
//! command line runs. Dataset search is the one request made here directly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use pmetal::core::SecretString;
use pmetal::hub::{DeviceSpec, FitLevel, HfSearchResult, ProgressCallback, TransferProgress};
use serde::Serialize;

use crate::error::{AppError, Result};
use crate::models::format_size;

/// How long a Hub request may take.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// A model search result.
#[derive(Debug, Clone, Serialize)]
pub struct HubSearchResult {
    pub id: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub downloads_formatted: String,
    pub likes: u64,
    pub pipeline_tag: Option<String>,
    pub is_gated: bool,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    /// Parameters, in billions, when the Hub or the name says.
    pub params_b: Option<f64>,
    /// Weight size in bytes, when the Hub lists it.
    pub size_bytes: Option<u64>,
    /// How it would run on this Mac: `fits`, `tight` or `too_large`.
    pub fit: Option<String>,
}

pub fn fit_name(level: FitLevel) -> &'static str {
    match level {
        FitLevel::Fits => "fits",
        FitLevel::Tight => "tight",
        FitLevel::TooLarge => "too_large",
    }
}

impl HubSearchResult {
    pub fn of(result: &HfSearchResult, device: Option<&DeviceSpec>) -> Self {
        Self {
            author: result.model_id.split('/').next().map(str::to_string),
            id: result.model_id.clone(),
            downloads: result.downloads,
            downloads_formatted: pmetal::hub::format_downloads(result.downloads),
            likes: result.likes,
            pipeline_tag: result.pipeline_tag.clone(),
            is_gated: result.gated,
            library_name: result.library_name.clone(),
            tags: result.tags.clone(),
            params_b: result.estimated_params_b,
            size_bytes: result.safetensors_bytes,
            fit: device
                .and_then(|d| result.fit(d, pmetal::hub::DEFAULT_FIT_CONTEXT))
                .map(|f| fit_name(f.fit_level).to_string()),
        }
    }
}

/// Search the Hub for models (most downloaded first), or list the trending
/// ones when `query` is empty.
pub async fn search_models(
    query: &str,
    limit: usize,
    token: Option<&SecretString>,
    device: Option<&DeviceSpec>,
) -> Result<Vec<HubSearchResult>> {
    let mut options = pmetal::hub::SearchOptions::new(query.trim());
    options.limit = limit.clamp(1, 100);
    options.hub = pmetal::hub::HubOptions {
        token: token.cloned(),
        ..Default::default()
    };
    if query.trim().is_empty() {
        options.sort = pmetal::hub::SearchSort::Trending;
    }
    let results = pmetal::hub::search_models_with(&options)
        .await
        .map_err(|e| hub_error("search for models", &e.to_string()))?;
    Ok(results
        .iter()
        .map(|r| HubSearchResult::of(r, device))
        .collect())
}

/// A dataset search result.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetSearchResult {
    pub id: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub downloads_formatted: String,
    pub likes: u64,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

/// Search the Hub for datasets, or list the trending ones when `query` is empty.
pub async fn search_datasets(
    query: &str,
    limit: usize,
    token: Option<&SecretString>,
) -> Result<Vec<DatasetSearchResult>> {
    let endpoint = pmetal::hub::HubOptions::default().resolved_endpoint();
    let query = query.trim();
    let mut params: Vec<(&str, String)> = vec![
        ("limit", limit.clamp(1, 100).to_string()),
        (
            "sort",
            if query.is_empty() {
                "trendingScore"
            } else {
                "downloads"
            }
            .to_string(),
        ),
        ("direction", "-1".to_string()),
    ];
    if !query.is_empty() {
        params.push(("search", query.to_string()));
    }
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .user_agent(concat!("pmetal-gui/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| hub_error("search for datasets", &e.to_string()))?;
    let mut request = client
        .get(format!("{endpoint}/api/datasets"))
        .query(&params);
    if let Some(token) = token {
        request = request.bearer_auth(token.expose_secret());
    }
    let response = request
        .send()
        .await
        .map_err(|e| hub_error("search for datasets", &e.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(hub_error("search for datasets", &format!("HTTP {status}")));
    }
    let items: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| hub_error("search for datasets", &e.to_string()))?;
    Ok(parse_datasets(&items))
}

fn parse_datasets(items: &[serde_json::Value]) -> Vec<DatasetSearchResult> {
    items
        .iter()
        .filter_map(|v| {
            let id = v["id"].as_str()?.to_string();
            let downloads = v["downloads"].as_u64().unwrap_or(0);
            Some(DatasetSearchResult {
                author: id
                    .split('/')
                    .next()
                    .filter(|_| id.contains('/'))
                    .map(str::to_string),
                downloads_formatted: pmetal::hub::format_downloads(downloads),
                downloads,
                likes: v["likes"].as_u64().unwrap_or(0),
                tags: v["tags"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|t| t.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
                description: v["description"].as_str().map(str::to_string),
                id,
            })
        })
        .collect()
}

/// What failed against the Hub, and what to try.
fn hub_error(action: &str, detail: &str) -> AppError {
    let lower = detail.to_lowercase();
    let hint = if lower.contains("401") || lower.contains("403") || lower.contains("token") {
        " If the repository is private or gated, add an access token in Settings."
    } else if lower.contains("dns")
        || lower.contains("connect")
        || lower.contains("timed out")
        || lower.contains("error sending request")
    {
        " Check the internet connection."
    } else {
        ""
    };
    AppError::new(format!("Could not {action}: {detail}.{hint}"))
}

// ---------------------------------------------------------------------------
// Datasets in the cache
// ---------------------------------------------------------------------------

/// A dataset in the Hub cache.
#[derive(Debug, Clone, Serialize)]
pub struct CachedDatasetInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub size_formatted: String,
}

pub fn cached_datasets(cache: &Path) -> Vec<CachedDatasetInfo> {
    pmetal::hub::list_cached_datasets_in(cache)
        .into_iter()
        .map(|d| {
            let size = crate::models::dir_size(&d.snapshot);
            CachedDatasetInfo {
                name: d.repo_id,
                path: d.snapshot.display().to_string(),
                size_bytes: size,
                size_formatted: format_size(size),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Queued,
    Downloading,
    Completed,
    Failed,
    Cancelled,
}

impl DownloadState {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Downloading)
    }
}

/// A model download, as the webview draws it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DownloadView {
    pub id: String,
    /// `model` or `dataset`.
    pub kind: &'static str,
    /// The repository: a model id or a dataset id.
    pub model_id: String,
    pub state: DownloadState,
    pub files_done: usize,
    pub total_files: usize,
    pub bytes_done: u64,
    pub total_bytes: u64,
    pub bytes_per_sec: Option<f64>,
    pub eta_seconds: Option<u64>,
    pub current_file: Option<String>,
    pub path: Option<String>,
    pub error: Option<String>,
    pub started_at: DateTime<Utc>,
}

/// Where a download's changes go.
pub type DownloadSink = Arc<dyn Fn(&DownloadView) + Send + Sync>;

struct Entry {
    view: DownloadView,
    task: Option<tokio::task::AbortHandle>,
}

/// The downloads in progress and recently finished. Limits how many transfer
/// at once, so several do not each crawl.
#[derive(Clone)]
pub struct Downloads {
    inner: Arc<Mutex<HashMap<String, Entry>>>,
    permits: Arc<tokio::sync::Semaphore>,
    sink: DownloadSink,
}

fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Downloads {
    pub fn new(at_once: usize, sink: DownloadSink) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            permits: Arc::new(tokio::sync::Semaphore::new(at_once)),
            sink,
        }
    }

    pub fn list(&self) -> Vec<DownloadView> {
        let mut all: Vec<_> = locked(&self.inner)
            .values()
            .map(|e| e.view.clone())
            .collect();
        all.sort_by_key(|v| std::cmp::Reverse(v.started_at));
        all
    }

    fn update(&self, id: &str, change: impl FnOnce(&mut DownloadView)) {
        let view = {
            let mut inner = locked(&self.inner);
            let Some(entry) = inner.get_mut(id) else {
                return;
            };
            // A download that was cancelled stays cancelled, whatever the
            // aborted transfer reports last.
            if !entry.view.state.is_active() {
                return;
            }
            change(&mut entry.view);
            entry.view.clone()
        };
        (self.sink)(&view);
    }

    /// Start downloading `model_id` with `run`, which is given a callback for
    /// progress and resolves to the downloaded folder.
    pub fn start<F, Fut>(&self, kind: &'static str, model_id: &str, run: F) -> Result<DownloadView>
    where
        F: FnOnce(ProgressCallback) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = std::result::Result<PathBuf, String>> + Send + 'static,
    {
        if locked(&self.inner)
            .values()
            .any(|e| e.view.kind == kind && e.view.model_id == model_id && e.view.state.is_active())
        {
            return Err(AppError::new(format!(
                "{model_id} is already downloading. Its progress is on the Models page."
            )));
        }
        let id = uuid::Uuid::new_v4().to_string();
        let view = DownloadView {
            id: id.clone(),
            kind,
            model_id: model_id.to_string(),
            state: DownloadState::Queued,
            files_done: 0,
            total_files: 0,
            bytes_done: 0,
            total_bytes: 0,
            bytes_per_sec: None,
            eta_seconds: None,
            current_file: None,
            path: None,
            error: None,
            started_at: Utc::now(),
        };
        locked(&self.inner).insert(
            id.clone(),
            Entry {
                view: view.clone(),
                task: None,
            },
        );
        (self.sink)(&view);

        let this = self.clone();
        let task_id = id.clone();
        let handle = tokio::spawn(async move {
            // Wait for a turn; a queued download shows as queued.
            let Ok(_permit) = this.permits.clone().acquire_owned().await else {
                return;
            };
            this.update(&task_id, |v| v.state = DownloadState::Downloading);
            let progress = {
                let this = this.clone();
                let id = task_id.clone();
                let callback: ProgressCallback = Arc::new(move |p: &TransferProgress| {
                    let p = p.clone();
                    this.update(&id, |v| {
                        v.files_done = p.files_done;
                        v.total_files = p.total_files;
                        v.bytes_done = p.bytes_done;
                        v.total_bytes = p.total_bytes;
                        v.bytes_per_sec = p.bytes_per_sec;
                        v.eta_seconds = p.eta.map(|d| d.as_secs());
                        v.current_file = p.current_file;
                    });
                });
                callback
            };
            let outcome = run(progress).await;
            this.update(&task_id, |v| match outcome {
                Ok(path) => {
                    v.state = DownloadState::Completed;
                    v.path = Some(path.display().to_string());
                    v.bytes_done = v.bytes_done.max(v.total_bytes);
                    v.eta_seconds = None;
                }
                Err(e) => {
                    v.state = DownloadState::Failed;
                    v.error = Some(e);
                }
            });
        });
        if let Some(entry) = locked(&self.inner).get_mut(&id) {
            entry.task = Some(handle.abort_handle());
        }
        Ok(view)
    }

    /// Stop a download. What was fetched stays in the cache, and a later
    /// download of the same model resumes from it.
    pub fn cancel(&self, id: &str) -> Result<()> {
        let view = {
            let mut inner = locked(&self.inner);
            let entry = inner
                .get_mut(id)
                .ok_or_else(|| AppError::new("That download is no longer in the list."))?;
            if !entry.view.state.is_active() {
                return Ok(());
            }
            if let Some(task) = entry.task.take() {
                task.abort();
            }
            entry.view.state = DownloadState::Cancelled;
            entry.view.eta_seconds = None;
            entry.view.clone()
        };
        (self.sink)(&view);
        Ok(())
    }

    /// Cancel every download. For when the app quits.
    pub fn cancel_all(&self) {
        let ids: Vec<String> = locked(&self.inner)
            .iter()
            .filter(|(_, e)| e.view.state.is_active())
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            let _ = self.cancel(&id);
        }
    }

    /// Drop finished downloads from the list.
    pub fn clear_finished(&self) {
        locked(&self.inner).retain(|_, e| e.view.state.is_active());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn downloads(at_once: usize) -> (Downloads, Arc<Mutex<Vec<DownloadView>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink_seen = seen.clone();
        let sink: DownloadSink = Arc::new(move |v| sink_seen.lock().unwrap().push(v.clone()));
        (Downloads::new(at_once, sink), seen)
    }

    async fn until(mut done: impl FnMut() -> bool) {
        for _ in 0..200 {
            if done() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out");
    }

    fn progress(done: u64, total: u64) -> TransferProgress {
        TransferProgress {
            files_done: 0,
            total_files: 2,
            bytes_done: done,
            total_bytes: total,
            bytes_per_sec: Some(10.0),
            eta: Some(Duration::from_secs(7)),
            current_file: Some("model.safetensors".into()),
            finished: false,
        }
    }

    #[tokio::test]
    async fn a_download_reports_progress_and_completes_with_its_folder() {
        let (downloads, seen) = downloads(2);
        downloads
            .start("model", "acme/m", |report| async move {
                report(&progress(50, 100));
                Ok(PathBuf::from("/cache/acme-m"))
            })
            .unwrap();
        until(|| downloads.list()[0].state == DownloadState::Completed).await;

        let done = downloads.list().remove(0);
        assert_eq!(done.path.as_deref(), Some("/cache/acme-m"));
        assert_eq!(done.bytes_done, 100);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.first().unwrap().state, DownloadState::Queued);
        let mid = seen.iter().find(|v| v.bytes_done == 50).unwrap();
        assert_eq!((mid.total_bytes, mid.eta_seconds), (100, Some(7)));
        assert_eq!(mid.current_file.as_deref(), Some("model.safetensors"));
    }

    #[tokio::test]
    async fn a_failed_download_keeps_the_reason() {
        let (downloads, _) = downloads(2);
        downloads
            .start("model", "acme/m", |_| async {
                Err("repository not found".to_string())
            })
            .unwrap();
        until(|| downloads.list()[0].state == DownloadState::Failed).await;
        assert_eq!(
            downloads.list()[0].error.as_deref(),
            Some("repository not found")
        );
    }

    #[tokio::test]
    async fn the_same_model_is_not_downloaded_twice_at_once() {
        let (downloads, _) = downloads(2);
        downloads
            .start("model", "acme/m", |_| async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok(PathBuf::new())
            })
            .unwrap();
        let err = downloads
            .start("model", "acme/m", |_| async { Ok(PathBuf::new()) })
            .unwrap_err();
        assert!(err.message().contains("already downloading"));
        downloads.cancel_all();
    }

    #[tokio::test]
    async fn only_so_many_transfer_at_once_and_the_rest_wait_their_turn() {
        let (downloads, _) = downloads(1);
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        for name in ["a/1", "a/2", "a/3"] {
            let (running, peak) = (running.clone(), peak.clone());
            downloads
                .start("model", name, move |_| async move {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    running.fetch_sub(1, Ordering::SeqCst);
                    Ok(PathBuf::new())
                })
                .unwrap();
        }
        assert!(
            downloads
                .list()
                .iter()
                .any(|v| v.state == DownloadState::Queued)
        );
        until(|| {
            downloads
                .list()
                .iter()
                .all(|v| v.state == DownloadState::Completed)
        })
        .await;
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_cancelled_download_stops_and_stays_cancelled() {
        let (downloads, _) = downloads(2);
        let finished = Arc::new(AtomicUsize::new(0));
        let flag = finished.clone();
        let view = downloads
            .start("model", "acme/m", move |_| async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                flag.fetch_add(1, Ordering::SeqCst);
                Ok(PathBuf::from("/late"))
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        downloads.cancel(&view.id).unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(downloads.list()[0].state, DownloadState::Cancelled);
        assert!(downloads.list()[0].path.is_none());
        assert_eq!(
            finished.load(Ordering::SeqCst),
            0,
            "the transfer was not aborted"
        );
        // And may be started again.
        downloads
            .start("model", "acme/m", |_| async { Ok(PathBuf::new()) })
            .unwrap();
    }

    #[test]
    fn dataset_results_are_read_leniently() {
        let items: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"id":"acme/data","downloads":1500,"likes":3,"tags":["a","b"],"description":"d"},
                {"downloads":1},
                {"id":"plain"}]"#,
        )
        .unwrap();
        let results = parse_datasets(&items);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].author.as_deref(), Some("acme"));
        assert_eq!(results[0].downloads_formatted, "1.5K");
        assert_eq!(results[1].author, None);
    }

    #[test]
    fn hub_errors_say_what_to_try() {
        assert!(
            hub_error("search", "HTTP 401")
                .message()
                .contains("Settings")
        );
        assert!(
            hub_error("search", "error sending request")
                .message()
                .contains("connection")
        );
        assert!(!hub_error("search", "weird").message().contains("Settings"));
    }
}
