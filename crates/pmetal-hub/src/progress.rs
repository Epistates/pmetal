//! Download and upload progress with throughput and ETA.
//!
//! The Hub client reports raw per-file events. [`TransferProgress`] is what a
//! progress bar wants instead: bytes and files done against the total, a
//! smoothed transfer rate, and the time left.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hf_hub::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler, UploadEvent};

/// A snapshot of a running download or upload.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferProgress {
    /// Files finished, including files that were already in the cache.
    pub files_done: usize,
    /// Files the download covers.
    pub total_files: usize,
    /// Bytes received so far.
    pub bytes_done: u64,
    /// Bytes to receive in total (files already cached are not counted).
    pub total_bytes: u64,
    /// Transfer rate over the last few seconds; `None` until there is enough
    /// to measure.
    pub bytes_per_sec: Option<f64>,
    /// Estimated time left, from `bytes_per_sec`; `None` when the rate is
    /// unknown or zero.
    pub eta: Option<Duration>,
    /// The file most recently reported on.
    pub current_file: Option<String>,
    /// The download is complete.
    pub finished: bool,
}

/// Receives [`TransferProgress`] snapshots. Must be quick and must not block:
/// it runs on the transfer path.
pub type ProgressCallback = Arc<dyn Fn(&TransferProgress) + Send + Sync>;

/// How far back the transfer rate looks.
const RATE_WINDOW: Duration = Duration::from_secs(5);
/// The shortest span a rate is measured over.
const MIN_RATE_SPAN: Duration = Duration::from_millis(400);

#[derive(Default)]
struct FileState {
    bytes: u64,
    total: u64,
    done: bool,
}

/// What an upload reports; the Hub client measures the rate itself.
#[derive(Default)]
struct UploadState {
    bytes: u64,
    rate: Option<f64>,
    files: usize,
}

/// Folds Hub events into progress snapshots. The clock is a parameter so the
/// arithmetic is testable.
pub(crate) struct Tracker {
    files: HashMap<String, FileState>,
    upload: UploadState,
    total_files: usize,
    total_bytes: u64,
    samples: VecDeque<(Instant, u64)>,
    started: Option<Instant>,
    current: Option<String>,
    finished: bool,
}

impl Tracker {
    pub(crate) fn new() -> Self {
        Self {
            files: HashMap::new(),
            upload: UploadState::default(),
            total_files: 0,
            total_bytes: 0,
            samples: VecDeque::new(),
            started: None,
            current: None,
            finished: false,
        }
    }

    fn bytes_done(&self) -> u64 {
        self.files.values().map(|f| f.bytes).sum()
    }

    /// Apply an event; the snapshot to report, if the event changes anything
    /// a user sees.
    pub(crate) fn on_event(
        &mut self,
        now: Instant,
        event: &DownloadEvent,
    ) -> Option<TransferProgress> {
        match event {
            DownloadEvent::Start {
                total_files,
                total_bytes,
            } => {
                self.total_files = *total_files;
                self.total_bytes = *total_bytes;
                self.started.get_or_insert(now);
            }
            DownloadEvent::Progress { files } => {
                self.started.get_or_insert(now);
                for update in files {
                    let state = self.files.entry(update.filename.clone()).or_default();
                    // Updates are absolute for a file, never negative.
                    state.bytes = state.bytes.max(update.bytes_completed);
                    state.total = state.total.max(update.total_bytes);
                    state.done |= update.status == FileStatus::Complete;
                    self.current = Some(update.filename.clone());
                }
            }
            // Per-file events already carry these bytes; counting the batch
            // total as well would double them.
            DownloadEvent::AggregateProgress { .. } => return None,
            DownloadEvent::Complete => self.finished = true,
        }
        let done = self.bytes_done();
        self.samples.push_back((now, done));
        while self
            .samples
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > RATE_WINDOW)
        {
            self.samples.pop_front();
        }
        Some(self.snapshot(now, done))
    }

    /// Apply an upload event. The Hub client measures the rate itself while
    /// bytes move; the ETA follows from it. Committing (the last, silent step)
    /// is reported as the current file so a bar does not sit at 100% unexplained.
    pub(crate) fn on_upload_event(&mut self, event: &UploadEvent) -> Option<TransferProgress> {
        match event {
            UploadEvent::Start {
                total_files,
                total_bytes,
            } => {
                self.total_files = *total_files;
                self.total_bytes = *total_bytes;
            }
            UploadEvent::Progress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
                files,
                ..
            } => {
                self.total_bytes = *total_bytes;
                self.upload.bytes = self.upload.bytes.max(*bytes_completed);
                self.upload.rate = *bytes_per_sec;
                let complete = files
                    .iter()
                    .filter(|f| f.status == FileStatus::Complete)
                    .count();
                self.upload.files = self.upload.files.max(complete);
                self.current = files
                    .iter()
                    .find(|f| f.status == FileStatus::InProgress)
                    .map(|f| f.filename.clone())
                    .or_else(|| self.current.clone());
            }
            UploadEvent::Committing => self.current = Some("committing".to_owned()),
            UploadEvent::Complete => {
                self.finished = true;
                self.upload.bytes = self.total_bytes;
                self.upload.files = self.total_files;
            }
        }
        let eta = self.upload.rate.filter(|r| *r > 0.0).map(|r| {
            Duration::from_secs_f64(self.total_bytes.saturating_sub(self.upload.bytes) as f64 / r)
        });
        Some(TransferProgress {
            files_done: self.upload.files,
            total_files: self.total_files,
            bytes_done: self.upload.bytes,
            total_bytes: self.total_bytes,
            bytes_per_sec: self.upload.rate,
            eta: if self.finished {
                Some(Duration::ZERO)
            } else {
                eta
            },
            current_file: self.current.clone(),
            finished: self.finished,
        })
    }

    fn snapshot(&self, now: Instant, done: u64) -> TransferProgress {
        let total_bytes = self
            .total_bytes
            .max(self.files.values().map(|f| f.total).sum());
        let rate = self.rate(now, done);
        let eta = rate
            .filter(|r| *r > 0.0)
            .map(|r| Duration::from_secs_f64(total_bytes.saturating_sub(done) as f64 / r));
        TransferProgress {
            files_done: self.files.values().filter(|f| f.done).count(),
            total_files: self.total_files.max(self.files.len()),
            bytes_done: done,
            total_bytes,
            bytes_per_sec: rate,
            eta: if self.finished {
                Some(Duration::ZERO)
            } else {
                eta
            },
            current_file: self.current.clone(),
            finished: self.finished,
        }
    }

    /// Bytes per second over the recent window, falling back to the average
    /// since the start while the window is still too short.
    fn rate(&self, now: Instant, done: u64) -> Option<f64> {
        let measure = |from: Instant, bytes: u64| {
            let span = now.duration_since(from);
            (span >= MIN_RATE_SPAN).then(|| done.saturating_sub(bytes) as f64 / span.as_secs_f64())
        };
        let (oldest_time, oldest_bytes) = *self.samples.front()?;
        measure(oldest_time, oldest_bytes).or_else(|| measure(self.started?, 0))
    }
}

/// Adapts a [`ProgressCallback`] to the Hub client's event stream.
pub(crate) struct CallbackHandler {
    tracker: Mutex<Tracker>,
    callback: ProgressCallback,
}

impl CallbackHandler {
    pub(crate) fn new(callback: ProgressCallback) -> Self {
        Self {
            tracker: Mutex::new(Tracker::new()),
            callback,
        }
    }
}

impl ProgressHandler for CallbackHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let snapshot = match (event, self.tracker.lock()) {
            (ProgressEvent::Download(event), Ok(mut tracker)) => {
                tracker.on_event(Instant::now(), event)
            }
            (ProgressEvent::Upload(event), Ok(mut tracker)) => tracker.on_upload_event(event),
            _ => None,
        };
        if let Some(snapshot) = snapshot {
            (self.callback)(&snapshot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::progress::FileProgress;

    fn file(name: &str, done: u64, total: u64, status: FileStatus) -> FileProgress {
        FileProgress {
            filename: name.into(),
            bytes_completed: done,
            total_bytes: total,
            status,
        }
    }

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn bytes_files_rate_and_eta_follow_the_events() {
        let t0 = Instant::now();
        let mut tracker = Tracker::new();
        tracker.on_event(
            t0,
            &DownloadEvent::Start {
                total_files: 2,
                total_bytes: 1000,
            },
        );
        let p = tracker
            .on_event(
                at(t0, 1000),
                &DownloadEvent::Progress {
                    files: vec![
                        file("a", 100, 400, FileStatus::InProgress),
                        file("b", 100, 600, FileStatus::InProgress),
                    ],
                },
            )
            .unwrap();
        assert_eq!(p.bytes_done, 200);
        assert_eq!(p.total_bytes, 1000);
        assert_eq!(p.files_done, 0);
        assert_eq!(p.total_files, 2);
        // 200 bytes in one second; 800 left.
        let rate = p.bytes_per_sec.unwrap();
        assert!((rate - 200.0).abs() < 1e-6, "{rate}");
        assert_eq!(p.eta.unwrap().as_secs(), 4);

        let p = tracker
            .on_event(
                at(t0, 2000),
                &DownloadEvent::Progress {
                    files: vec![file("a", 400, 400, FileStatus::Complete)],
                },
            )
            .unwrap();
        assert_eq!(p.bytes_done, 500);
        assert_eq!(p.files_done, 1);
        assert_eq!(p.current_file.as_deref(), Some("a"));
    }

    #[test]
    fn a_file_reported_complete_without_bytes_does_not_inflate_the_total() {
        // Cache hits come as Complete with no sizes.
        let t0 = Instant::now();
        let mut tracker = Tracker::new();
        tracker.on_event(
            t0,
            &DownloadEvent::Start {
                total_files: 2,
                total_bytes: 500,
            },
        );
        let p = tracker
            .on_event(
                at(t0, 10),
                &DownloadEvent::Progress {
                    files: vec![file("cached", 0, 0, FileStatus::Complete)],
                },
            )
            .unwrap();
        assert_eq!(p.files_done, 1);
        assert_eq!(p.total_bytes, 500);
        assert_eq!(p.bytes_done, 0);
    }

    #[test]
    fn the_batch_aggregate_is_not_counted_twice() {
        let t0 = Instant::now();
        let mut tracker = Tracker::new();
        tracker.on_event(
            t0,
            &DownloadEvent::Start {
                total_files: 1,
                total_bytes: 1000,
            },
        );
        tracker.on_event(
            at(t0, 500),
            &DownloadEvent::Progress {
                files: vec![file("a", 300, 1000, FileStatus::InProgress)],
            },
        );
        assert!(
            tracker
                .on_event(
                    at(t0, 600),
                    &DownloadEvent::AggregateProgress {
                        bytes_completed: 300,
                        total_bytes: 1000,
                        bytes_per_sec: Some(1.0),
                    },
                )
                .is_none()
        );
    }

    #[test]
    fn the_rate_tracks_the_recent_window_not_the_whole_history() {
        let t0 = Instant::now();
        let mut tracker = Tracker::new();
        tracker.on_event(
            t0,
            &DownloadEvent::Start {
                total_files: 1,
                total_bytes: 1_000_000,
            },
        );
        let mut update = |ms: u64, bytes: u64| {
            tracker
                .on_event(
                    at(t0, ms),
                    &DownloadEvent::Progress {
                        files: vec![file("a", bytes, 1_000_000, FileStatus::InProgress)],
                    },
                )
                .unwrap()
        };
        // 100 kB in the first second, then a stall, then 1 MB/s.
        update(1_000, 100_000);
        update(10_000, 100_000);
        let p = update(12_000, 300_000);
        // Over the last five seconds: (300k - 100k) / 2s at best, never the
        // 25 kB/s average since the start.
        assert!(p.bytes_per_sec.unwrap() > 50_000.0, "{:?}", p.bytes_per_sec);
    }

    #[test]
    fn complete_means_nothing_left() {
        let t0 = Instant::now();
        let mut tracker = Tracker::new();
        tracker.on_event(
            t0,
            &DownloadEvent::Start {
                total_files: 1,
                total_bytes: 10,
            },
        );
        tracker.on_event(
            at(t0, 1000),
            &DownloadEvent::Progress {
                files: vec![file("a", 10, 10, FileStatus::Complete)],
            },
        );
        let p = tracker
            .on_event(at(t0, 1100), &DownloadEvent::Complete)
            .unwrap();
        assert!(p.finished);
        assert_eq!(p.eta, Some(Duration::ZERO));
        assert_eq!(p.bytes_done, 10);
    }

    #[test]
    fn no_rate_before_enough_time_has_passed() {
        let t0 = Instant::now();
        let mut tracker = Tracker::new();
        tracker.on_event(
            t0,
            &DownloadEvent::Start {
                total_files: 1,
                total_bytes: 1000,
            },
        );
        let p = tracker
            .on_event(
                at(t0, 50),
                &DownloadEvent::Progress {
                    files: vec![file("a", 10, 1000, FileStatus::InProgress)],
                },
            )
            .unwrap();
        assert!(p.bytes_per_sec.is_none());
        assert!(p.eta.is_none());
    }
}
