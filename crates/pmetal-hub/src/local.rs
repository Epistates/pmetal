//! The models on this machine: the Hub cache, what was trained here, and the
//! folders a person added.
//!
//! One scan for every surface. The terminal UI, the desktop app and the MCP
//! server each present the result their own way (a table, a list, JSON); what
//! counts as a model folder, how a Hub snapshot is told from a trained output,
//! how big a model is and what a trained adapter was trained from are decided
//! here, once. The Hub cache is read the way every other tool reads it
//! ([`list_cached_models_in`]), so a model downloaded by any of them is listed;
//! the folders other apps keep models in are scanned as the interop paths they
//! are ([`ScanPlan::well_known_dirs`]).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use pmetal_core::TrainingInfo;
use serde::{Deserialize, Serialize};

use crate::list_cached_models_in;

/// How many levels a scan looks at, a root being the first: trained outputs
/// are found up to three folders down (`output/grpo/run--lora`, ...), the
/// folders a person added and other apps' directly in them.
const TRAINED_DEPTH: usize = 4;
const CUSTOM_DEPTH: usize = 2;

/// Where a model was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    /// The Hugging Face Hub cache.
    HfCache,
    /// A fine-tuned output: adapters, or weights a training run wrote.
    Trained,
    /// A folder the person added, or another app's model folder.
    Custom,
}

/// A model on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalModel {
    /// What the model is called: its Hub id, or what its folder says it is
    /// (`_name_or_path` when that is not a path), else the folder's name.
    pub id: String,
    /// The folder's name.
    pub folder: String,
    /// Where its files are: the snapshot, for a Hub model.
    pub path: PathBuf,
    /// What to give a job to use it: the Hub id for a Hub model (so the job
    /// resolves it as everything else does), the full path for the rest.
    pub reference: String,
    /// Bytes on disk, links followed (a Hub snapshot is links into its blobs).
    pub size_bytes: u64,
    /// When the folder was last changed.
    pub modified: Option<SystemTime>,
    /// Where it was found.
    pub source: ModelSource,
    /// Whether it is a LoRA adapter, not a whole model.
    pub is_adapter: bool,
    /// The model a trained folder was trained from: what its
    /// `training_info.json` records, else, for an adapter, the folder's name
    /// before `--` (how `pmetal` names an output `<base>--<method>`).
    pub base_model: Option<String>,
    /// The architecture its `config.json` names.
    pub architecture: Option<String>,
    /// The format its weights are in.
    pub weights: Option<&'static str>,
    /// Whether it carries a quantization config.
    pub quantized: bool,
}

/// Where a scan looks.
#[derive(Debug, Clone, Default)]
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
    /// The folders other desktop apps keep downloaded models in, which a scan
    /// reads when they exist.
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

/// Find every model the plan reaches: the Hub cache, then trained outputs,
/// then the other folders. A folder reached twice is listed once. The order is
/// the scan's; a surface sorts for its own display. Blocking: it reads
/// directories.
pub fn scan(plan: &ScanPlan) -> Vec<LocalModel> {
    let mut models = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for cached in list_cached_models_in(&plan.hub_cache) {
        seen.insert(
            cached
                .snapshot
                .canonicalize()
                .unwrap_or_else(|_| cached.snapshot.clone()),
        );
        models.push(describe(
            cached.repo_id.clone(),
            cached.repo_id,
            &cached.snapshot,
            ModelSource::HfCache,
        ));
    }
    scan_tree(
        &plan.output_root,
        TRAINED_DEPTH,
        ModelSource::Trained,
        &mut models,
        &mut seen,
    );
    for dir in plan.custom_dirs.iter().chain(&plan.well_known) {
        scan_tree(
            dir,
            CUSTOM_DEPTH,
            ModelSource::Custom,
            &mut models,
            &mut seen,
        );
    }
    models
}

/// Look for models in `dir` and `depth - 1` levels below it. A folder
/// that is a model is not searched further; hidden folders and checkpoint
/// folders are not looked in.
fn scan_tree(
    dir: &Path,
    depth: usize,
    default_source: ModelSource,
    models: &mut Vec<LocalModel>,
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
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| !n.starts_with('.') && n != "checkpoints")
        })
        .collect();
    children.sort();
    for child in children {
        scan_tree(&child, depth - 1, default_source, models, seen);
    }
}

fn add_dir(
    dir: &Path,
    default_source: ModelSource,
    models: &mut Vec<LocalModel>,
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
    let Some(folder) = dir.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return;
    };
    // The model's own name for itself, when it gives one that is not a path.
    let id = read_config(dir)
        .and_then(|cfg| cfg["_name_or_path"].as_str().map(str::to_string))
        .filter(|s| !s.is_empty() && !s.starts_with('/'))
        .unwrap_or_else(|| folder.clone());
    models.push(describe(id, real.display().to_string(), &real, source));
}

fn describe(id: String, reference: String, dir: &Path, source: ModelSource) -> LocalModel {
    let is_adapter = is_lora_adapter(dir);
    let folder = dir
        .file_name()
        .map_or_else(|| id.clone(), |name| name.to_string_lossy().into_owned());
    // A trained folder says what it was trained from; failing that, an adapter
    // is named `<base>--<method>`.
    let base_model = TrainingInfo::read(dir)
        .and_then(|info| info.base_model)
        .or_else(|| {
            is_adapter
                .then(|| folder.split_once("--").map(|(base, _)| base.to_string()))
                .flatten()
        });
    LocalModel {
        folder,
        id,
        path: dir.to_path_buf(),
        reference,
        size_bytes: dir_size(dir),
        modified: std::fs::metadata(dir).and_then(|m| m.modified()).ok(),
        source,
        is_adapter,
        base_model,
        architecture: read_config(dir).and_then(|cfg| {
            cfg["model_type"]
                .as_str()
                .or_else(|| cfg["architectures"].as_array()?.first()?.as_str())
                .map(String::from)
        }),
        weights: weight_format(dir),
        quantized: dir.join("quantization_config.json").exists(),
    }
}

/// The commits of a repository the cache holds a folder for, by name (each is
/// a directory under `snapshots/`), sorted.
pub fn cached_snapshots(hub_cache: &Path, repo_id: &str) -> Vec<String> {
    let folder = pmetal_core::paths::hub_repo_folder("models", repo_id);
    let mut names: Vec<String> = std::fs::read_dir(hub_cache.join(folder).join("snapshots"))
        .map(|dir| {
            dir.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// The `config.json` of a model folder.
pub fn read_config(dir: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// Whether `dir` holds a model or an adapter.
pub fn is_model_dir(dir: &Path) -> bool {
    if dir.join("config.json").is_file() || dir.join("adapter_config.json").is_file() {
        return true;
    }
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors")
                || name.ends_with(".safetensors.index.json")
                || name.ends_with(".gguf")
        })
    })
}

/// Whether `dir` is a LoRA adapter (it has adapter weights or an adapter
/// config).
pub fn is_lora_adapter(dir: &Path) -> bool {
    [
        "adapter_config.json",
        "adapter_model.safetensors",
        "lora_weights.safetensors",
    ]
    .iter()
    .any(|f| dir.join(f).exists())
}

/// Whether `dir` is something a training run wrote.
fn is_trained_dir(dir: &Path) -> bool {
    is_lora_adapter(dir) || dir.join("training_state.json").exists()
}

/// The format `dir`'s weights are in.
fn weight_format(dir: &Path) -> Option<&'static str> {
    if dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists() {
        return Some("SafeTensors");
    }
    if dir.join("pytorch_model.bin").exists() {
        return Some("PyTorch");
    }
    let (mut shards, mut gguf) = (false, false);
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        shards |= name.ends_with(".safetensors");
        gguf |= name.ends_with(".gguf");
    }
    if shards {
        Some("SafeTensors (sharded)")
    } else if gguf {
        Some("GGUF")
    } else {
        None
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    const CONFIG: &str = r#"{"model_type":"llama","hidden_size":64}"#;

    fn plan(root: &Path) -> ScanPlan {
        ScanPlan {
            hub_cache: root.join("hub"),
            output_root: root.join("output"),
            custom_dirs: vec![root.join("mine")],
            well_known: Vec::new(),
        }
    }

    /// A cache with one finished snapshot of `org/base`, `main` naming it.
    fn cache_base(root: &Path) -> PathBuf {
        let sha = "a".repeat(40);
        let snapshot = root.join(format!("hub/models--org--base/snapshots/{sha}"));
        write(&snapshot.join("config.json"), CONFIG);
        write(&snapshot.join("model.safetensors"), "weights");
        write(&root.join("hub/models--org--base/refs/main"), &sha);
        snapshot
    }

    #[test]
    fn the_scan_finds_cached_trained_and_custom_models() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        let snapshot = cache_base(r);
        write(&r.join("output/grpo/run--lora/adapter_config.json"), "{}");
        write(
            &r.join("output/grpo/run--lora/training_info.json"),
            r#"{"base_model":"org/base","dataset":"d"}"#,
        );
        write(&r.join("mine/tiny/config.json"), CONFIG);
        write(&r.join("mine/tiny/model.safetensors"), "12345");

        let models = scan(&plan(r));
        let by_id = |id: &str| {
            models
                .iter()
                .find(|m| m.id == id)
                .unwrap_or_else(|| panic!("{id} is not in {models:#?}"))
        };

        let cached = by_id("org/base");
        assert_eq!(cached.source, ModelSource::HfCache);
        assert_eq!(cached.reference, "org/base");
        assert_eq!(cached.path, snapshot);
        assert_eq!(cached.size_bytes, 7 + CONFIG.len() as u64);
        assert_eq!(cached.architecture.as_deref(), Some("llama"));
        assert_eq!(cached.weights, Some("SafeTensors"));

        let adapter = by_id("run--lora");
        assert_eq!(adapter.source, ModelSource::Trained);
        assert!(adapter.is_adapter);
        assert_eq!(adapter.base_model.as_deref(), Some("org/base"));
        assert_eq!(adapter.folder, "run--lora");

        let custom = by_id("tiny");
        assert_eq!(custom.source, ModelSource::Custom);
        assert_eq!(
            custom.reference,
            custom.path.canonicalize().unwrap().display().to_string()
        );
        assert_eq!(custom.size_bytes, 5 + CONFIG.len() as u64);
        assert_eq!(models.len(), 3);
    }

    #[test]
    fn an_adapter_without_a_record_is_named_for_its_base_and_checkpoints_are_skipped() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        write(
            &r.join("output/Qwen3-0.6B--lora/adapter_model.safetensors"),
            "w",
        );
        write(
            &r.join("output/Qwen3-0.6B--lora/checkpoints/step-1/adapter_model.safetensors"),
            "w",
        );
        write(&r.join("output/.hidden/config.json"), "{}");
        let models = scan(&plan(r));
        assert_eq!(models.len(), 1, "{models:?}");
        assert_eq!(models[0].base_model.as_deref(), Some("Qwen3-0.6B"));
    }

    #[test]
    fn a_folder_reached_twice_is_listed_once_and_a_models_own_name_is_used() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        write(
            &r.join("mine/m/config.json"),
            r#"{"_name_or_path":"org/named"}"#,
        );
        write(
            &r.join("mine/p/config.json"),
            r#"{"_name_or_path":"/tmp/p"}"#,
        );
        let mut plan = plan(r);
        plan.custom_dirs.push(r.join("mine"));
        plan.well_known.push(r.join("mine/m"));
        let models = scan(&plan);
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["org/named", "p"]);
    }

    #[test]
    fn an_empty_or_missing_root_has_no_models() {
        let root = tempfile::tempdir().unwrap();
        assert!(scan(&plan(root.path())).is_empty());
    }

    #[test]
    fn a_link_back_up_the_tree_does_not_loop() {
        let root = tempfile::tempdir().unwrap();
        write(&root.path().join("a/file"), "123");
        std::os::unix::fs::symlink(root.path(), root.path().join("a/up")).unwrap();
        assert_eq!(dir_size(&root.path().join("a")), 3);
    }
}
