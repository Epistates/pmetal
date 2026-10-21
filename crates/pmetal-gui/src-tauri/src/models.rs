//! The models on this Mac: the Hub cache, what the app trained, and the
//! folders a person added.
//!
//! The Hub cache is read through `pmetal-hub`, which resolves its location the
//! way every other tool does (`HF_HUB_CACHE`, `HF_HOME`, ...), so a model
//! downloaded by any of them is listed here. This module adds the two places
//! the Hub knows nothing about.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Where a model was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    /// The Hugging Face hub cache.
    HfCache,
    /// A fine-tuned output.
    Trained,
    /// A folder the person added, or another app's model folder.
    Custom,
}

/// A model on disk.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CachedModel {
    /// What the model is called: its Hub id, or its folder's name.
    pub id: String,
    /// Where its files are: the snapshot, for a Hub model.
    pub path: String,
    /// What to give a job to use it: the Hub id for a Hub model (so the job
    /// resolves it as everything else does), the full path for the rest.
    pub reference: String,
    pub size: u64,
    pub size_formatted: String,
    pub downloaded_at: Option<DateTime<Utc>>,
    /// `text-generation`, `embedding`, `audio` or `image`.
    pub model_type: String,
    pub source: ModelSource,
}

/// Where a scan looks.
#[derive(Debug, Clone)]
pub struct ScanPlan {
    /// The Hub cache.
    pub hub_cache: PathBuf,
    /// Where trained outputs are written.
    pub output_root: PathBuf,
    /// Folders the person added.
    pub custom_dirs: Vec<PathBuf>,
    /// Other apps' model folders, scanned when they exist.
    pub well_known: Vec<PathBuf>,
}

impl ScanPlan {
    /// The folders other apps keep models in.
    pub fn well_known_dirs() -> Vec<PathBuf> {
        let home = dirs::home_dir().unwrap_or_default();
        vec![
            home.join(".lmstudio").join("models"),
            home.join(".cache").join("lm-studio").join("models"),
        ]
    }

    /// Every folder a model can be under, other than the Hub cache.
    pub fn roots(&self) -> Vec<PathBuf> {
        std::iter::once(self.output_root.clone())
            .chain(self.custom_dirs.iter().cloned())
            .chain(self.well_known.iter().cloned())
            .collect()
    }
}

/// Find every model the plan reaches. Blocking: it reads directories.
///
/// Trained outputs come first, then by size, largest first.
pub fn scan(plan: &ScanPlan) -> Vec<CachedModel> {
    let mut models = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for cached in pmetal_hub_list(&plan.hub_cache) {
        if let Some(model) = hub_model(&cached.0, &cached.1) {
            seen.insert(cached.1.canonicalize().unwrap_or(cached.1));
            models.push(model);
        }
    }
    scan_tree(
        &plan.output_root,
        2,
        ModelSource::Trained,
        &mut models,
        &mut seen,
    );
    for dir in plan.custom_dirs.iter().chain(&plan.well_known) {
        scan_tree(dir, 2, ModelSource::Custom, &mut models, &mut seen);
    }

    models.sort_by(|a, b| {
        let trained = |m: &CachedModel| m.source == ModelSource::Trained;
        trained(b)
            .cmp(&trained(a))
            .then(b.size.cmp(&a.size))
            .then(a.id.cmp(&b.id))
    });
    models
}

fn pmetal_hub_list(cache: &Path) -> Vec<(String, PathBuf)> {
    pmetal::hub::list_cached_models_in(cache)
        .into_iter()
        .map(|m| (m.repo_id, m.snapshot))
        .collect()
}

fn hub_model(repo_id: &str, snapshot: &Path) -> Option<CachedModel> {
    let size = dir_size(snapshot);
    Some(CachedModel {
        id: repo_id.to_string(),
        path: snapshot.display().to_string(),
        reference: repo_id.to_string(),
        size,
        size_formatted: format_size(size),
        downloaded_at: modified(snapshot),
        model_type: infer_model_type(repo_id),
        source: ModelSource::HfCache,
    })
}

/// Look for models in `dir` and up to `depth - 1` levels below it. A folder
/// that is a model is not searched further.
fn scan_tree(
    dir: &Path,
    depth: usize,
    default_source: ModelSource,
    models: &mut Vec<CachedModel>,
    seen: &mut HashSet<PathBuf>,
) {
    if depth == 0 || !dir.is_dir() {
        return;
    }
    if is_model_dir(dir) {
        add_dir(dir, default_source, models, seen);
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    children.sort();
    for child in children {
        scan_tree(&child, depth - 1, default_source, models, seen);
    }
}

fn add_dir(
    dir: &Path,
    default_source: ModelSource,
    models: &mut Vec<CachedModel>,
    seen: &mut HashSet<PathBuf>,
) {
    let real = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    if !seen.insert(real.clone()) {
        return;
    }
    let source = if is_trained_dir(dir) {
        ModelSource::Trained
    } else {
        default_source
    };
    let Some(name) = dir.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return;
    };
    // The model's own name for itself, when it gives one that is not a path.
    let id = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|cfg| cfg["_name_or_path"].as_str().map(str::to_string))
        .filter(|s| !s.is_empty() && !s.starts_with('/'))
        .unwrap_or(name);
    let size = dir_size(dir);
    models.push(CachedModel {
        model_type: infer_model_type(&id),
        id,
        path: real.display().to_string(),
        reference: real.display().to_string(),
        size,
        size_formatted: format_size(size),
        downloaded_at: modified(dir),
        source,
    });
}

/// Whether `dir` holds a model or an adapter.
fn is_model_dir(dir: &Path) -> bool {
    if dir.join("config.json").is_file() || dir.join("adapter_config.json").is_file() {
        return true;
    }
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors") || name.ends_with(".gguf")
        })
    })
}

/// Whether `dir` is something the app or `pmetal train` wrote.
fn is_trained_dir(dir: &Path) -> bool {
    [
        "adapter_config.json",
        "adapter_model.safetensors",
        "lora_weights.safetensors",
        "training_state.json",
    ]
    .iter()
    .any(|f| dir.join(f).exists())
}

fn modified(path: &Path) -> Option<DateTime<Utc>> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(DateTime::<Utc>::from)
}

/// The size of everything under `path`, following symlinks, since a Hub
/// snapshot is links into its blob store.
pub fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![path.to_path_buf()];
    let mut visited: HashSet<PathBuf> = HashSet::new();
    while let Some(dir) = stack.pop() {
        // A link back up the tree must not loop.
        if let Ok(real) = dir.canonicalize()
            && !visited.insert(real)
        {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match std::fs::metadata(&path) {
                Ok(m) if m.is_file() => total += m.len(),
                Ok(m) if m.is_dir() => stack.push(path),
                _ => {}
            }
        }
    }
    total
}

pub fn infer_model_type(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    if lower.contains("embed") {
        "embedding"
    } else if lower.contains("whisper") || lower.contains("wav2vec") || lower.contains("parakeet") {
        "audio"
    } else if lower.contains("flux") || lower.contains("stable-diffusion") || lower.contains("sdxl")
    {
        "image"
    } else {
        "text-generation"
    }
    .to_string()
}

/// A byte count in 1024-based units.
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1_024;
    const MB: u64 = KB * 1_024;
    const GB: u64 = MB * 1_024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// The `config.json` of a model folder.
pub fn read_config(model_path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// A Hub cache holding one model, laid out as the Hub lays it out.
    fn hub_cache(root: &Path, repo: &str) -> PathBuf {
        let folder = format!("models--{}", repo.replace('/', "--"));
        let snapshot = root.join(&folder).join("snapshots").join("abc123");
        let blob = root.join(&folder).join("blobs").join("deadbeef");
        write(&blob, &"x".repeat(2048));
        write(
            &snapshot.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":8,"vocab_size":16}"#,
        );
        std::os::unix::fs::symlink(&blob, snapshot.join("model.safetensors")).unwrap();
        write(&root.join(&folder).join("refs").join("main"), "abc123");
        snapshot
    }

    fn plan(dir: &Path) -> ScanPlan {
        ScanPlan {
            hub_cache: dir.join("hub"),
            output_root: dir.join("out"),
            custom_dirs: vec![dir.join("custom")],
            well_known: vec![],
        }
    }

    #[test]
    fn the_hub_cache_is_read_where_the_plan_says_and_a_model_is_given_by_its_id() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = hub_cache(&dir.path().join("hub"), "acme/tiny-embed");
        let models = scan(&plan(dir.path()));
        assert_eq!(models.len(), 1, "{models:?}");
        let model = &models[0];
        assert_eq!(model.id, "acme/tiny-embed");
        assert_eq!(model.reference, "acme/tiny-embed");
        assert_eq!(model.source, ModelSource::HfCache);
        assert_eq!(model.path, snapshot.display().to_string());
        // The size follows the links into the blob store.
        assert!(model.size >= 2048, "{}", model.size);
        assert_eq!(model.model_type, "embedding");
    }

    #[test]
    fn trained_and_custom_models_are_given_by_their_path() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("out/run-1/adapter_config.json"), "{}");
        write(&dir.path().join("out/run-1/adapter_model.safetensors"), "w");
        write(&dir.path().join("custom/some-model/config.json"), "{}");
        write(&dir.path().join("custom/some-model/model.safetensors"), "w");

        let models = scan(&plan(dir.path()));

        assert_eq!(models.len(), 2, "{models:?}");
        // Trained first.
        assert_eq!(models[0].source, ModelSource::Trained);
        assert_eq!(models[0].id, "run-1");
        assert!(models[0].reference.ends_with("out/run-1"));
        assert!(std::path::Path::new(&models[0].reference).is_absolute());
        assert_eq!(models[1].source, ModelSource::Custom);
        assert_eq!(models[1].id, "some-model");
    }

    #[test]
    fn a_folder_that_is_a_model_is_not_searched_for_more() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("custom/m/config.json"), "{}");
        write(&dir.path().join("custom/m/checkpoint-1/config.json"), "{}");
        let models = scan(&plan(dir.path()));
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn a_model_reached_two_ways_is_listed_once() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("custom/m/config.json"), "{}");
        let mut p = plan(dir.path());
        p.custom_dirs = vec![dir.path().join("custom"), dir.path().join("custom/m")];
        assert_eq!(scan(&p).len(), 1);
    }

    #[test]
    fn scanning_does_not_write_into_the_folders_it_reads() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("custom/gguf-only/model.gguf"), "GGUF");
        scan(&plan(dir.path()));
        let files: Vec<_> = std::fs::read_dir(dir.path().join("custom/gguf-only"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(files, ["model.gguf"]);
    }

    #[test]
    fn a_link_back_up_the_tree_does_not_loop() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("custom/m/config.json"), "{}");
        std::os::unix::fs::symlink(dir.path().join("custom"), dir.path().join("custom/m/loop"))
            .unwrap();
        let _ = scan(&plan(dir.path()));
        assert!(dir_size(&dir.path().join("custom/m")) < 100);
    }

    #[test]
    fn sizes_read_as_people_write_them() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(2048), "2 KB");
        assert_eq!(format_size(3 * 1024 * 1024 * 1024), "3.00 GB");
    }
}
