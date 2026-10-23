//! What a training run was made from, written beside its results.
//!
//! A run's output folder holds an adapter (or weights) and, written when the
//! run starts, a `training_info.json` saying which model it started from and on
//! what. The terminal UI and the desktop app read it to list an adapter with
//! its base model and to find the base model to chat with; the trainer writes
//! it, so a run started from the command line, the terminal UI, the app or
//! Python leaves the same record.
//!
//! Folders written by earlier releases hold files of two shapes (one from the
//! terminal UI, one from the app); both read as a [`TrainingInfo`] with the
//! fields they had.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The file's name.
pub const TRAINING_INFO_FILE: &str = "training_info.json";

/// What a run started from, and when.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingInfo {
    /// The model the run started from, as it was named (a Hub id or a path).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    /// Where that model was on disk when the run started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model_path: Option<String>,
    /// The dataset it trained on, as it was named.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// The kind of run (`lora`, `qlora`, `distill`, `grpo`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// When the run started, RFC 3339.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    /// The `pmetal` that ran it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pmetal_version: Option<String>,
}

impl TrainingInfo {
    /// The record of a run that starts now, from `base_model`. `created` and
    /// `pmetal_version` are filled in.
    pub fn starting(base_model: impl Into<String>) -> Self {
        Self {
            base_model: Some(base_model.into()),
            created: Some(chrono::Utc::now().to_rfc3339()),
            pmetal_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            ..Self::default()
        }
    }

    /// Where the record is in `dir`.
    pub fn path_in(dir: &Path) -> PathBuf {
        dir.join(TRAINING_INFO_FILE)
    }

    /// The record in `dir`, if there is a readable one. A file from an earlier
    /// release reads with the fields it had; one that is not a record at all
    /// reads as none.
    pub fn read(dir: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(Self::path_in(dir)).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Write the record to `dir` (made if it is missing), replacing any there.
    ///
    /// # Errors
    ///
    /// The folder cannot be made or the file written.
    pub fn write(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let text = serde_json::to_string_pretty(self).map_err(io::Error::from)?;
        std::fs::write(Self::path_in(dir), text)
    }

    /// [`write`](Self::write), where a failure is a warning: the run does not
    /// depend on its record.
    pub fn write_or_warn(&self, dir: &Path) {
        if let Err(e) = self.write(dir) {
            tracing::warn!(
                "could not record what {} was trained from: {e}",
                dir.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_is_written_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("run");
        let info = TrainingInfo {
            dataset: Some("acme/data".into()),
            method: Some("lora".into()),
            base_model_path: Some("/cache/base".into()),
            ..TrainingInfo::starting("org/base")
        };
        info.write(&out).unwrap();
        assert_eq!(TrainingInfo::read(&out), Some(info.clone()));
        assert_eq!(
            info.pmetal_version.as_deref(),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert!(
            info.created
                .as_deref()
                .is_some_and(|t| chrono::DateTime::parse_from_rfc3339(t).is_ok())
        );
    }

    #[test]
    fn the_files_earlier_releases_wrote_still_read() {
        let dir = tempfile::tempdir().unwrap();
        // The terminal UI's.
        std::fs::write(
            TrainingInfo::path_in(dir.path()),
            r#"{"base_model":"org/base","base_model_path":"/cache/base","created":"2026-03-18T21:45:07+00:00"}"#,
        )
        .unwrap();
        let tui = TrainingInfo::read(dir.path()).unwrap();
        assert_eq!(tui.base_model.as_deref(), Some("org/base"));
        assert_eq!(tui.base_model_path.as_deref(), Some("/cache/base"));
        assert_eq!(tui.dataset, None);
        // The app's, with a null where it knew nothing.
        std::fs::write(
            TrainingInfo::path_in(dir.path()),
            r#"{"base_model":"org/base","dataset":null,"created":"2026-03-18T21:45:07+01:00"}"#,
        )
        .unwrap();
        let app = TrainingInfo::read(dir.path()).unwrap();
        assert_eq!(app.base_model.as_deref(), Some("org/base"));
        assert_eq!(app.dataset, None);
        // Only the dataset, as a hand-written record might be.
        std::fs::write(
            TrainingInfo::path_in(dir.path()),
            r#"{"dataset":"acme/chat-data"}"#,
        )
        .unwrap();
        assert_eq!(
            TrainingInfo::read(dir.path()).unwrap().dataset.as_deref(),
            Some("acme/chat-data")
        );
    }

    #[test]
    fn what_is_not_a_record_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(TrainingInfo::read(dir.path()), None);
        std::fs::write(TrainingInfo::path_in(dir.path()), "not json").unwrap();
        assert_eq!(TrainingInfo::read(dir.path()), None);
    }
}
