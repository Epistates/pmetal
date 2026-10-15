//! Loads **real released checkpoints** through `DynamicModel::load`, which
//! refuses a checkpoint that does not account for the whole model.
//!
//! The synthetic tests in `loader_strictness` prove the rule on checkpoints
//! built from the model's own parameter names, where the two cannot disagree.
//! What they cannot show is whether a *published* layout fits: a tensor the
//! architecture was never taught to read, or a parameter a real checkpoint does
//! not carry. Every defect this check has found so far was invisible until a
//! real file was loaded.
//!
//! The checkpoints are gigabytes and carry their upstream licences, so this is
//! `#[ignore]`d and gated on `PMETAL_REAL_CHECKPOINTS`: a `:`-separated list of
//! directories, each either one checkpoint (it holds `config.json`) or a folder
//! of them, in plain `<org>__<name>/` form or the Hub cache's
//! `models--<org>--<name>/snapshots/<rev>/` layout.
//!
//! ```bash
//! PMETAL_REAL_CHECKPOINTS=<dir>[:<dir>...] \
//!     cargo test -p pmetal-models --test real_checkpoint_layouts -- --ignored --nocapture
//! ```
//!
//! Checkpoints whose architecture pmetal declines (an encoder family, an
//! unreleased one) are reported as skipped, not failures: the question here is
//! whether what pmetal *does* load loads completely.

use std::path::{Path, PathBuf};

use pmetal_models::{DynamicModel, ModelArchitecture};

/// Every checkpoint directory under `root`.
fn checkpoints_under(root: &Path) -> Vec<PathBuf> {
    if root.join("config.json").exists() {
        return vec![root.to_path_buf()];
    }
    if root
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("models--"))
    {
        return checkpoints_under_parent(root.parent().unwrap_or(root), Some(root));
    }
    checkpoints_under_parent(root, None)
}

fn checkpoints_under_parent(root: &Path, only: Option<&Path>) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return found;
    };
    let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for entry in entries {
        if only.is_some_and(|only| only != entry) {
            continue;
        }
        let name = entry.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('_') || name.starts_with('.') || !entry.is_dir() {
            continue;
        }
        if name.starts_with("models--") {
            // Newest snapshot of a Hub cache repository.
            let Ok(snapshots) = std::fs::read_dir(entry.join("snapshots")) else {
                continue;
            };
            let mut snapshots: Vec<_> = snapshots.flatten().map(|e| e.path()).collect();
            snapshots.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
            if let Some(latest) = snapshots.pop() {
                if latest.join("config.json").exists() {
                    found.push(latest);
                }
            }
        } else if entry.join("config.json").exists() {
            found.push(entry);
        }
    }
    found
}

fn has_weights(dir: &Path) -> bool {
    dir.join("model.safetensors").exists()
        || dir.join("model.safetensors.index.json").exists()
        || dir.join("pytorch_model.bin").exists()
        || dir.join("pytorch_model.bin.index.json").exists()
}

/// `org/name` for a checkpoint directory, for the report.
fn label(dir: &Path) -> String {
    let text = dir.to_string_lossy();
    if let Some(repo) = text
        .split('/')
        .find(|part| part.starts_with("models--"))
        .map(|part| part.trim_start_matches("models--").replace("--", "/"))
    {
        return repo;
    }
    dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .replace("__", "/")
}

#[test]
#[ignore = "needs PMETAL_REAL_CHECKPOINTS"]
fn real_checkpoints_load_with_every_tensor_accounted_for() {
    let Ok(roots) = std::env::var("PMETAL_REAL_CHECKPOINTS") else {
        eprintln!("PMETAL_REAL_CHECKPOINTS is not set; nothing to load");
        return;
    };
    let dirs: Vec<PathBuf> = roots
        .split(':')
        .filter(|root| !root.is_empty())
        .flat_map(|root| checkpoints_under(Path::new(root)))
        .filter(|dir| has_weights(dir))
        .collect();
    assert!(!dirs.is_empty(), "no checkpoints found under {roots}");

    let mut failures = Vec::new();
    for dir in &dirs {
        let name = label(dir);
        let arch = match ModelArchitecture::detect(dir) {
            Ok(arch) => arch,
            Err(e) => {
                println!("SKIP  {name}: {e}");
                continue;
            }
        };
        let started = std::time::Instant::now();
        match DynamicModel::load(dir) {
            Ok(model) => {
                drop(model);
                println!(
                    "OK    {name} ({arch}) in {:.1}s",
                    started.elapsed().as_secs_f32()
                );
            }
            Err(e) => {
                println!("FAIL  {name} ({arch}): {e}");
                failures.push(name.clone());
            }
        }
        if let Err(e) = pmetal_runtime::check_unobserved_error() {
            println!("FAIL  {name}: an op threw during the load: {e}");
            failures.push(format!("{name} (op error)"));
        }
    }
    assert!(
        failures.is_empty(),
        "checkpoints that did not load: {failures:?}"
    );
}
