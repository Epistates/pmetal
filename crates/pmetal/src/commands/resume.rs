//! `--resume` for the MTP, assistant and DFlash draft trainers: which
//! checkpoint to continue from, and how far it got.
//!
//! A run writes a checkpoint (`model.safetensors`, `config.json`, the
//! optimizer state and `metadata.json`) every `--checkpoint-every` steps into
//! `<output>/checkpoints/step_<N>/`, and the final one into `<output>/`. Resume
//! takes any of them, or the output directory itself, in which case the one
//! that got furthest.

use std::path::{Path, PathBuf};

use anyhow::Context;

/// A checkpoint to continue from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResumePoint {
    /// The checkpoint directory: its weights are the model to continue.
    pub(crate) dir: PathBuf,
    /// Optimizer steps already taken. The run starts here, and the data
    /// supplies the batches from this step on.
    pub(crate) step: usize,
}

/// The step a checkpoint directory's `metadata.json` records, if it is one.
fn checkpoint_step(dir: &Path) -> anyhow::Result<Option<usize>> {
    let path = dir.join("metadata.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("could not read '{}'", path.display()))?;
    let meta: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("'{}' is not JSON", path.display()))?;
    let step = meta
        .get("step")
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("'{}' records no step", path.display()))?;
    Ok(Some(step as usize))
}

/// The checkpoint `path` names: itself, or the one furthest along among the
/// checkpoints an output directory holds.
pub(crate) fn find_resume_point(path: &Path) -> anyhow::Result<ResumePoint> {
    anyhow::ensure!(
        path.is_dir(),
        "--resume: '{}' is not a directory",
        path.display()
    );
    let mut candidates: Vec<ResumePoint> = Vec::new();
    let mut consider = |dir: PathBuf| -> anyhow::Result<()> {
        if let Some(step) = checkpoint_step(&dir)? {
            candidates.push(ResumePoint { dir, step });
        }
        Ok(())
    };
    consider(path.to_path_buf())?;
    if let Ok(entries) = std::fs::read_dir(path.join("checkpoints")) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                consider(entry.path())?;
            }
        }
    }
    candidates
        .into_iter()
        .max_by_key(|point| point.step)
        .with_context(|| {
            format!(
                "--resume: '{}' holds no checkpoint (a directory with metadata.json, or an \
                 output directory with checkpoints/step_N)",
                path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(dir: &Path, step: u64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("metadata.json"),
            format!(r#"{{"step": {step}, "loss": 1.0, "learning_rate": 0.0001, "checkpoint_type": "dflash_draft"}}"#),
        )
        .unwrap();
    }

    #[test]
    fn an_output_directory_resumes_from_its_furthest_checkpoint() {
        let out = tempfile::tempdir().unwrap();
        checkpoint(&out.path().join("checkpoints").join("step_100"), 100);
        checkpoint(&out.path().join("checkpoints").join("step_200"), 200);
        let point = find_resume_point(out.path()).unwrap();
        assert_eq!(point.step, 200);
        assert!(point.dir.ends_with("step_200"));

        // The final checkpoint at the root counts, and wins when it is later.
        checkpoint(out.path(), 300);
        assert_eq!(find_resume_point(out.path()).unwrap().dir, out.path());
        // A checkpoint directory names itself.
        let one = out.path().join("checkpoints").join("step_100");
        assert_eq!(find_resume_point(&one).unwrap().step, 100);
    }

    #[test]
    fn a_directory_without_a_checkpoint_says_what_it_needs() {
        let empty = tempfile::tempdir().unwrap();
        let err = find_resume_point(empty.path()).unwrap_err().to_string();
        assert!(
            err.contains("holds no checkpoint") && err.contains("metadata.json"),
            "{err}"
        );
        let err = find_resume_point(&empty.path().join("missing"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a directory"), "{err}");

        std::fs::write(empty.path().join("metadata.json"), r#"{"loss": 1.0}"#).unwrap();
        let err = format!("{:#}", find_resume_point(empty.path()).unwrap_err());
        assert!(err.contains("records no step"), "{err}");
    }
}
