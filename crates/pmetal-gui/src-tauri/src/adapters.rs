//! LoRA adapters trained on this Mac.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// A LoRA adapter folder.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TrainedAdapter {
    /// The adapter folder.
    pub path: String,
    /// "Base model + dataset", or the folder's name.
    pub name: String,
    pub base_model: Option<String>,
    pub rank: Option<u32>,
    pub alpha: Option<f32>,
    /// The weights file's size.
    pub size_bytes: u64,
}

/// Every adapter in the folders under `root`, up to two levels down, largest
/// first. Blocking.
pub fn scan(root: &Path) -> Vec<TrainedAdapter> {
    let mut adapters = Vec::new();
    let mut seen = HashSet::new();
    let folders = |dir: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect()
    };
    for first in folders(root) {
        add(&first, &mut adapters, &mut seen);
        for second in folders(&first) {
            add(&second, &mut adapters, &mut seen);
        }
    }
    adapters.sort_by_key(|a| std::cmp::Reverse(a.size_bytes));
    adapters
}

fn add(dir: &Path, adapters: &mut Vec<TrainedAdapter>, seen: &mut HashSet<PathBuf>) {
    if !seen.insert(dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf())) {
        return;
    }
    let has_config = dir.join("adapter_config.json").exists();
    let weights = pmetal::lora::peft::find_adapter_weights(dir);
    if !has_config && weights.is_none() {
        return;
    }
    let read_json = |name: &str| {
        std::fs::read_to_string(dir.join(name))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    };
    let config = if has_config {
        read_json("adapter_config.json")
    } else {
        None
    };
    let info = pmetal::core::TrainingInfo::read(dir);
    let base_model = config
        .as_ref()
        .and_then(pmetal::lora::peft::adapter_base_model)
        .or_else(|| info.as_ref().and_then(|i| i.base_model.clone()));
    let dataset = info.and_then(|i| i.dataset);
    let folder = dir.file_name().map_or_else(
        || dir.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let short = |id: &str| id.rsplit('/').next().unwrap_or(id).to_string();
    let name = match (&base_model, &dataset) {
        (Some(base), Some(data)) => format!("{} + {}", short(base), short(data)),
        (Some(base), None) => format!("{} — {folder}", short(base)),
        _ => folder,
    };
    adapters.push(TrainedAdapter {
        path: dir.display().to_string(),
        name,
        base_model,
        rank: config
            .as_ref()
            .and_then(|c| c["r"].as_u64())
            .map(|v| v as u32),
        alpha: config
            .as_ref()
            .and_then(|c| c["alpha"].as_f64().or_else(|| c["lora_alpha"].as_f64()))
            .map(|v| v as f32),
        size_bytes: weights
            .and_then(|w| w.metadata().ok())
            .map_or(0, |m| m.len()),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn adapters_are_found_with_their_rank_base_model_and_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("run-a");
        write(
            &a.join("adapter_config.json"),
            r#"{"r":16,"lora_alpha":32,"base_model_name_or_path":"Qwen/Qwen3-0.6B"}"#,
        );
        write(&a.join("adapter_model.safetensors"), "weights");
        write(
            &a.join("training_info.json"),
            r#"{"dataset":"acme/chat-data"}"#,
        );
        // One level further down, and not an adapter.
        write(&dir.path().join("group/run-b/adapter_config.json"), "{}");
        write(&dir.path().join("group/not-one/readme.md"), "x");

        let found = scan(dir.path());

        assert_eq!(found.len(), 2, "{found:?}");
        let a = found.iter().find(|f| f.path.ends_with("run-a")).unwrap();
        assert_eq!(a.name, "Qwen3-0.6B + chat-data");
        assert_eq!((a.rank, a.alpha), (Some(16), Some(32.0)));
        assert_eq!(a.base_model.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(a.size_bytes, 7);
        assert!(found.iter().any(|f| f.name == "run-b"));
    }

    #[test]
    fn a_missing_folder_has_no_adapters() {
        assert!(scan(Path::new("/no/such/folder")).is_empty());
    }
}
