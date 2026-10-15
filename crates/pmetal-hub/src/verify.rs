//! Checking that a model directory is complete.
//!
//! A download interrupted after the config but before the last weight shard
//! leaves a directory that looks like a model and fails at load time, or, for
//! a file cut mid-write, mid-inference. [`verify_model_dir`] reads only
//! headers (a few kilobytes per file, however large the weights) and says what
//! is missing.

use std::path::{Path, PathBuf};

use crate::select::{WeightFormat, shards_in_index};
use crate::tensors::{SafetensorsSummary, summarize_safetensors};

/// What [`verify_model_dir`] found.
#[derive(Debug, Clone, Default)]
pub struct ModelDirReport {
    /// The weight format present (`None` when there are no weights).
    pub format: Option<WeightFormat>,
    /// The weight files, in path order (for a split GGUF, all shards).
    pub weight_files: Vec<PathBuf>,
    /// Total size of the weight files in bytes.
    pub weight_bytes: u64,
    /// Tensor totals, for safetensors weights.
    pub safetensors: Option<SafetensorsSummary>,
    /// Why the directory cannot be loaded; empty when it is complete.
    pub problems: Vec<String>,
    /// What is absent but not necessarily needed: a missing `config.json`
    /// (a language model cannot load without one; other models have none).
    pub warnings: Vec<String>,
}

impl ModelDirReport {
    /// Whether nothing is missing or damaged.
    pub fn is_complete(&self) -> bool {
        self.problems.is_empty()
    }

    /// Whether this is a complete model a language-model loader can open: no
    /// problems, and a `config.json` unless the weights are GGUF (which carry
    /// their own configuration).
    pub fn is_loadable_language_model(&self) -> bool {
        self.is_complete() && self.warnings.is_empty()
    }
}

/// Check a model directory: weights of one format that are all present, each
/// safetensors file exactly as long as its header says, each GGUF set complete
/// with every tensor inside its file, and a readable `config.json` if there is
/// one (its absence is only a warning, except that a GGUF needs none).
///
/// A directory with no weights at all is incomplete. Weights are looked for
/// directly in `dir`. Symlinked files (the Hub cache layout) are followed; a
/// dangling link counts as missing.
pub fn verify_model_dir(dir: &Path) -> ModelDirReport {
    let mut report = ModelDirReport::default();
    if !dir.is_dir() {
        report
            .problems
            .push(format!("{} is not a directory", dir.display()));
        return report;
    }
    let names = match top_level_files(dir) {
        Ok(names) => names,
        Err(e) => {
            report
                .problems
                .push(format!("cannot read {}: {e}", dir.display()));
            return report;
        }
    };
    let has = |pred: &dyn Fn(&str) -> bool| names.iter().any(|n| pred(n));
    let has_safetensors = has(&|n| n.ends_with(".safetensors"));
    let has_gguf = has(&|n| n.ends_with(".gguf") && !n.starts_with("mmproj"));
    let has_pytorch = has(&|n| n.ends_with(".bin") && n.starts_with("pytorch_model"));

    if has_safetensors {
        check_safetensors(dir, &names, &mut report);
    } else if has_gguf {
        check_gguf(dir, &names, &mut report);
    } else if has_pytorch {
        check_pytorch(dir, &names, &mut report);
    } else {
        // An index without any shard, or nothing at all.
        if names.iter().any(|n| n.ends_with(".index.json")) {
            report
                .problems
                .push("a weight index is present but none of its shards are".into());
        } else {
            report
                .problems
                .push("no model weights (.safetensors, .gguf or pytorch_model*.bin)".into());
        }
    }

    if report.format != Some(WeightFormat::Gguf) {
        match std::fs::read(dir.join("config.json")) {
            Ok(bytes) => {
                if serde_json::from_slice::<serde_json::Value>(&bytes).is_err() {
                    report.problems.push("config.json is not valid JSON".into());
                }
            }
            // Not every repository has one (a diffusion checkpoint's weights
            // are the whole model), but a language model cannot load without.
            Err(_) => report.warnings.push("config.json is missing".into()),
        }
    }
    report
}

/// File names directly in `dir`, files and symlinks alike, sorted. Hidden
/// files are left out: exFAT and network volumes grow an AppleDouble stub
/// (`._model.safetensors`) beside every file, and Finder leaves `.DS_Store`.
fn top_level_files(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_ok_and(|t| t.is_dir()))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| !name.starts_with('.'))
        .collect();
    names.sort();
    Ok(names)
}

/// The size of `path`, following links; `None` when it is missing or dangling.
fn size_of(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .filter(|m| m.is_file())
        .map(|m| m.len())
}

fn check_safetensors(dir: &Path, names: &[String], report: &mut ModelDirReport) {
    report.format = Some(WeightFormat::Safetensors);
    let index_name = "model.safetensors.index.json";
    let mut files: Vec<String> = names
        .iter()
        .filter(|n| n.ends_with(".safetensors"))
        .cloned()
        .collect();
    if names.iter().any(|n| n == index_name) {
        match std::fs::read(dir.join(index_name))
            .map_err(|e| e.to_string())
            .and_then(|b| shards_in_index(&b).map_err(|e| e.to_string()))
        {
            Ok(shards) => {
                for shard in &shards {
                    if size_of(&dir.join(shard)).is_none() {
                        report.problems.push(format!("shard {shard} is missing"));
                    }
                }
                // Only the shards the index names are the model; another
                // layout beside them is not checked.
                files = shards;
            }
            Err(e) => report.problems.push(format!("{index_name}: {e}")),
        }
    }
    let mut total = SafetensorsSummary::default();
    for name in &files {
        let path = dir.join(name);
        let Some(size) = size_of(&path) else {
            continue; // already reported as missing
        };
        report.weight_files.push(path.clone());
        report.weight_bytes += size;
        match summarize_safetensors(&path) {
            Ok(summary) => total.merge(&summary),
            Err(e) => report.problems.push(e.to_string()),
        }
    }
    report.safetensors = Some(total);
}

fn check_gguf(dir: &Path, names: &[String], report: &mut ModelDirReport) {
    report.format = Some(WeightFormat::Gguf);
    // Open each model once: a split set opens from any of its shards.
    let mut opened_sets: Vec<PathBuf> = Vec::new();
    for name in names
        .iter()
        .filter(|n| n.ends_with(".gguf") && !n.starts_with("mmproj"))
    {
        let path = dir.join(name);
        let entry = pmetal_gguf::first_shard_path(&path).unwrap_or(path.clone());
        if opened_sets.contains(&entry) {
            continue;
        }
        opened_sets.push(entry.clone());
        match pmetal_gguf::GgufModel::open(&entry) {
            Ok(model) => {
                for shard in model.shard_paths() {
                    report.weight_bytes += size_of(shard).unwrap_or(0);
                    report.weight_files.push(shard.to_path_buf());
                }
            }
            Err(e) => report.problems.push(format!("{name}: {e}")),
        }
    }
}

fn check_pytorch(dir: &Path, names: &[String], report: &mut ModelDirReport) {
    report.format = Some(WeightFormat::PyTorch);
    let index_name = "pytorch_model.bin.index.json";
    let mut files: Vec<String> = names
        .iter()
        .filter(|n| n.ends_with(".bin") && n.starts_with("pytorch_model"))
        .cloned()
        .collect();
    if names.iter().any(|n| n == index_name) {
        match std::fs::read(dir.join(index_name))
            .map_err(|e| e.to_string())
            .and_then(|b| shards_in_index(&b).map_err(|e| e.to_string()))
        {
            Ok(shards) => {
                for shard in &shards {
                    if size_of(&dir.join(shard)).is_none() {
                        report.problems.push(format!("shard {shard} is missing"));
                    }
                }
                files = shards;
            }
            Err(e) => report.problems.push(format!("{index_name}: {e}")),
        }
    }
    for name in files {
        let path = dir.join(&name);
        if let Some(size) = size_of(&path) {
            report.weight_bytes += size;
            report.weight_files.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::write_test_safetensors;

    fn model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        dir
    }

    fn write_index(dir: &Path, shards: &[&str]) {
        let map: serde_json::Map<_, _> = shards
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("t{i}"), serde_json::json!(s)))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({ "weight_map": map }).to_string(),
        )
        .unwrap();
    }

    #[test]
    fn a_complete_single_file_model_passes() {
        let dir = model_dir();
        write_test_safetensors(
            &dir.path().join("model.safetensors"),
            &[("w", "BF16", &[8, 8])],
        );
        let report = verify_model_dir(dir.path());
        assert!(report.is_complete(), "{:?}", report.problems);
        assert_eq!(report.format, Some(WeightFormat::Safetensors));
        assert_eq!(report.safetensors.unwrap().data_bytes(), 128);
    }

    #[test]
    fn a_config_with_no_weights_is_not_a_model() {
        let dir = model_dir();
        let report = verify_model_dir(dir.path());
        assert!(!report.is_complete());
        assert!(
            report.problems[0].contains("no model weights"),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn a_missing_shard_is_named() {
        let dir = model_dir();
        write_index(dir.path(), &["s1.safetensors", "s2.safetensors"]);
        write_test_safetensors(&dir.path().join("s1.safetensors"), &[("w", "F16", &[4])]);
        let report = verify_model_dir(dir.path());
        assert!(
            report.problems.iter().any(|p| p.contains("s2.safetensors")),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn a_dangling_cache_link_counts_as_missing() {
        let dir = model_dir();
        write_index(dir.path(), &["s1.safetensors"]);
        std::os::unix::fs::symlink(
            dir.path().join("../no-such-blob"),
            dir.path().join("s1.safetensors"),
        )
        .unwrap();
        let report = verify_model_dir(dir.path());
        assert!(!report.is_complete());
    }

    #[test]
    fn a_shard_cut_short_is_caught_by_its_header() {
        let dir = model_dir();
        write_index(dir.path(), &["s1.safetensors"]);
        let shard = dir.path().join("s1.safetensors");
        write_test_safetensors(&shard, &[("w", "F32", &[256])]);
        let bytes = std::fs::read(&shard).unwrap();
        std::fs::write(&shard, &bytes[..bytes.len() - 100]).unwrap();
        let report = verify_model_dir(dir.path());
        assert!(
            report.problems.iter().any(|p| p.contains("cut short")),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn appledouble_and_finder_files_are_not_weights() {
        // exFAT and network volumes grow a `._name` stub beside every file.
        let dir = model_dir();
        write_test_safetensors(&dir.path().join("model.safetensors"), &[("w", "F16", &[4])]);
        std::fs::write(dir.path().join("._model.safetensors"), [0u8; 4096]).unwrap();
        std::fs::write(dir.path().join("._config.json"), [0u8; 4096]).unwrap();
        std::fs::write(dir.path().join(".DS_Store"), b"x").unwrap();
        let report = verify_model_dir(dir.path());
        assert!(report.is_complete(), "{:?}", report.problems);
        assert_eq!(report.weight_files.len(), 1);
    }

    #[test]
    fn a_second_layout_beside_an_index_is_not_the_models_problem() {
        let dir = model_dir();
        write_index(dir.path(), &["s1.safetensors"]);
        write_test_safetensors(&dir.path().join("s1.safetensors"), &[("w", "F16", &[4])]);
        std::fs::write(dir.path().join("consolidated.safetensors"), b"junk").unwrap();
        assert!(verify_model_dir(dir.path()).is_complete());
    }

    #[test]
    fn a_missing_config_is_a_warning_and_gguf_needs_none() {
        let dir = tempfile::tempdir().unwrap();
        write_test_safetensors(&dir.path().join("model.safetensors"), &[("w", "F16", &[4])]);
        let report = verify_model_dir(dir.path());
        // Complete weights, but not something a language-model loader can open.
        assert!(report.is_complete());
        assert!(report.warnings.iter().any(|p| p.contains("config.json")));
        assert!(!report.is_loadable_language_model());

        let gguf = tempfile::tempdir().unwrap();
        write_gguf(gguf.path(), "m.gguf", None, &["a"]);
        let report = verify_model_dir(gguf.path());
        assert!(report.warnings.is_empty());
        assert!(report.is_loadable_language_model());
    }

    #[test]
    fn a_config_that_is_not_json_is_a_problem() {
        let dir = model_dir();
        write_test_safetensors(&dir.path().join("model.safetensors"), &[("w", "F16", &[4])]);
        std::fs::write(dir.path().join("config.json"), "not json").unwrap();
        let report = verify_model_dir(dir.path());
        assert!(report.problems.iter().any(|p| p.contains("config.json")));
    }

    #[test]
    fn a_missing_directory_is_reported() {
        let report = verify_model_dir(Path::new("/definitely/not/here"));
        assert!(!report.is_complete());
    }

    fn write_gguf(dir: &Path, name: &str, split: Option<(u16, u16, i32)>, tensors: &[&str]) {
        use pmetal_gguf::{GgufBuilder, MetadataValue};
        let mut builder = GgufBuilder::with_model("llama", "m");
        if let Some((no, count, total)) = split {
            builder.add_metadata("split.no", MetadataValue::Uint16(no));
            builder.add_metadata("split.count", MetadataValue::Uint16(count));
            builder.add_metadata("split.tensors.count", MetadataValue::Int32(total));
        }
        for name in tensors {
            builder.add_f32_tensor(*name, vec![4], vec![1.0; 4]);
        }
        std::fs::write(dir.join(name), builder.build_to_bytes().unwrap()).unwrap();
    }

    #[test]
    fn a_single_gguf_needs_no_config() {
        let dir = tempfile::tempdir().unwrap();
        write_gguf(dir.path(), "m-Q4_K_M.gguf", None, &["a"]);
        let report = verify_model_dir(dir.path());
        assert!(report.is_complete(), "{:?}", report.problems);
        assert_eq!(report.format, Some(WeightFormat::Gguf));
        assert_eq!(report.weight_files.len(), 1);
        assert!(report.weight_bytes > 0);
    }

    #[test]
    fn a_split_gguf_is_checked_as_one_model() {
        let dir = tempfile::tempdir().unwrap();
        write_gguf(dir.path(), "m-00001-of-00002.gguf", Some((0, 2, 2)), &["a"]);
        write_gguf(dir.path(), "m-00002-of-00002.gguf", Some((1, 2, 2)), &["b"]);
        let report = verify_model_dir(dir.path());
        assert!(report.is_complete(), "{:?}", report.problems);
        assert_eq!(report.weight_files.len(), 2);
    }

    #[test]
    fn a_split_gguf_missing_a_shard_is_incomplete() {
        let dir = tempfile::tempdir().unwrap();
        write_gguf(dir.path(), "m-00001-of-00002.gguf", Some((0, 2, 2)), &["a"]);
        let report = verify_model_dir(dir.path());
        assert!(!report.is_complete());
        assert!(
            report.problems.iter().any(|p| p.contains("missing")),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn a_gguf_cut_short_is_incomplete() {
        let dir = tempfile::tempdir().unwrap();
        write_gguf(dir.path(), "m.gguf", None, &["a", "b", "c"]);
        let path = dir.path().join("m.gguf");
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 20]).unwrap();
        assert!(!verify_model_dir(dir.path()).is_complete());
    }

    #[test]
    fn pytorch_shards_are_checked_against_their_index() {
        let dir = model_dir();
        std::fs::write(dir.path().join("pytorch_model-00001-of-00002.bin"), b"x").unwrap();
        std::fs::write(
            dir.path().join("pytorch_model.bin.index.json"),
            r#"{"weight_map":{"a":"pytorch_model-00001-of-00002.bin","b":"pytorch_model-00002-of-00002.bin"}}"#,
        )
        .unwrap();
        let report = verify_model_dir(dir.path());
        assert_eq!(report.format, Some(WeightFormat::PyTorch));
        assert!(
            report.problems.iter().any(|p| p.contains("00002-of-00002")),
            "{:?}",
            report.problems
        );
    }
}
