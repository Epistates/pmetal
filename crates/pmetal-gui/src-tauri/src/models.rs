//! The models on this Mac, as the app lists them: the Hub cache, what the app
//! trained, and the folders a person added.
//!
//! The scan is `pmetal_hub::local`'s, the one every surface shares: it reads
//! the Hub cache the way every other tool does (`HF_HUB_CACHE`, `HF_HOME`, ...)
//! and the other places a model can be. This module presents what it found.

use chrono::{DateTime, Utc};
use pmetal::hub::local::LocalModel;
pub use pmetal::hub::local::{ModelSource, ScanPlan, read_config};
use serde::Serialize;

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

impl CachedModel {
    fn of(model: &LocalModel) -> Self {
        Self {
            id: model.id.clone(),
            path: model.path.display().to_string(),
            reference: model.reference.clone(),
            size: model.size_bytes,
            size_formatted: pmetal::core::format::human_size(model.size_bytes),
            downloaded_at: model.modified.map(DateTime::<Utc>::from),
            model_type: infer_model_type(&model.id),
            source: model.source,
        }
    }
}

/// Find every model the plan reaches. Blocking: it reads directories.
///
/// Trained outputs come first, then by size, largest first.
pub fn scan(plan: &ScanPlan) -> Vec<CachedModel> {
    let mut models: Vec<CachedModel> = pmetal::hub::local::scan(plan)
        .iter()
        .map(CachedModel::of)
        .collect();
    models.sort_by(|a, b| {
        let trained = |m: &CachedModel| m.source == ModelSource::Trained;
        trained(b)
            .cmp(&trained(a))
            .then(b.size.cmp(&a.size))
            .then(a.id.cmp(&b.id))
    });
    models
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use pmetal::hub::local::dir_size;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// A Hub cache holding one model, laid out as the Hub lays it out.
    fn hub_cache(root: &Path, repo: &str) -> PathBuf {
        let folder = pmetal::core::paths::hub_repo_folder("models", repo);
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
    fn a_size_is_shown_the_way_the_hub_shows_it() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("custom/m/config.json"), "{}");
        write(
            &dir.path().join("custom/m/model.safetensors"),
            &"x".repeat(2_000_000),
        );
        let models = scan(&plan(dir.path()));
        assert_eq!(models[0].size_formatted, "2 MB");
    }
}
