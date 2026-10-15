//! Deciding which files of a repository a download needs.
//!
//! A model repository lists far more than one needs to run it: weights in
//! several formats, several quantizations of the same weights, a second copy
//! of the weights in another layout, ONNX exports, sample images. This module
//! turns the file listing (names and sizes, nothing else) into the exact set
//! to fetch. It does no I/O, so every rule has a unit test.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use pmetal_core::{PMetalError, Result};

/// A file of a repository at one commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    /// Path relative to the repository root, `/`-separated.
    pub path: String,
    /// Size in bytes.
    pub size: u64,
    /// SHA-256 of the content, for files stored in large-file storage (the
    /// weights); `None` for small files, which the Hub identifies by git id.
    pub sha256: Option<String>,
}

impl RepoFile {
    /// A file with a known size and no content hash.
    pub fn new(path: impl Into<String>, size: u64) -> Self {
        Self {
            path: path.into(),
            size,
            sha256: None,
        }
    }

    fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }

    fn dir(&self) -> &str {
        self.path.rsplit_once('/').map_or("", |(d, _)| d)
    }
}

/// Which weight format a download should fetch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WeightFormat {
    /// Safetensors when the repository has them, else GGUF. A repository that
    /// publishes only PyTorch pickles is an error under `Auto`: nothing here
    /// loads them, and saying so beats a "successful" download of no weights.
    #[default]
    Auto,
    /// Safetensors only.
    Safetensors,
    /// One GGUF quantization (see [`DownloadOptions::gguf_quant`]).
    ///
    /// [`DownloadOptions::gguf_quant`]: crate::DownloadOptions::gguf_quant
    Gguf,
    /// PyTorch `pytorch_model*.bin` pickles.
    PyTorch,
}

/// What kind of weights a plan selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedWeights {
    /// Safetensors files.
    Safetensors,
    /// One GGUF quantization, possibly split over several files.
    Gguf {
        /// The quantization label, e.g. `Q4_K_M`.
        quant: String,
    },
    /// PyTorch pickles.
    PyTorch,
}

/// The files a model download fetches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPlan {
    /// The weight format selected.
    pub weights: SelectedWeights,
    /// Every file to fetch, in path order: weights, indexes, configs,
    /// tokenizers.
    pub files: Vec<RepoFile>,
}

impl ModelPlan {
    /// Total bytes to fetch.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// The weight files of the plan (not indexes or configs), in path order.
    pub fn weight_files(&self) -> impl Iterator<Item = &RepoFile> {
        self.files.iter().filter(|f| is_weight_file(&f.path))
    }
}

/// A GGUF quantization available in a repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufVariant {
    /// The quantization label, e.g. `Q4_K_M`, `IQ4_XS`, `BF16`.
    pub quant: String,
    /// The file or, for a split model, all its shards in order.
    pub files: Vec<RepoFile>,
}

impl GgufVariant {
    /// Total size in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// Files never worth fetching for running a model: other frameworks' weights,
/// archives, media, notebooks.
const SKIP_EXTENSIONS: &[&str] = &[
    ".onnx",
    ".onnx_data",
    ".ot",
    ".msgpack",
    ".h5",
    ".keras",
    ".pb",
    ".tflite",
    ".mlmodel",
    ".ckpt",
    ".npz",
    ".npy",
    ".pkl",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".mp4",
    ".mov",
    ".wav",
    ".mp3",
    ".flac",
    ".pdf",
    ".ipynb",
];

/// Directories that hold alternative exports or checkpoints of the same model.
const SKIP_DIRS: &[&str] = &["onnx", "original", "openvino", "coreml", "assets"];

/// Anything else bigger than this is not a config or a tokenizer.
const MAX_AUX_BYTES: u64 = 512 * 1024 * 1024;

/// Whether `path` is a weight file in some format.
pub(crate) fn is_weight_file(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.ends_with(".safetensors")
        || name.ends_with(".gguf")
        || is_pytorch_weight(name)
        || name.ends_with(".pth")
        || name.ends_with(".pt")
}

/// `pytorch_model.bin`, `pytorch_model-00001-of-00003.bin`, `adapter_model.bin`,
/// `model.bin`: the pickled state dicts, not training state like `optimizer.bin`.
fn is_pytorch_weight(name: &str) -> bool {
    name.ends_with(".bin")
        && (name.starts_with("pytorch_model")
            || name.starts_with("adapter_model")
            || name == "model.bin")
}

fn is_index(name: &str, weights_suffix: &str) -> bool {
    name.ends_with(&format!("{weights_suffix}.index.json"))
}

/// Whether a file that is not weights should come along (configs, tokenizers,
/// the README).
fn wanted_aux(file: &RepoFile) -> bool {
    let name = file.name();
    if name.starts_with('.') || name == ".gitattributes" {
        return false;
    }
    if is_weight_file(&file.path) || name.ends_with(".bin") {
        // Weights are chosen by the format rules; stray `.bin` files are
        // training state.
        return false;
    }
    if file.dir().split('/').any(|d| SKIP_DIRS.contains(&d)) {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    if SKIP_EXTENSIONS.iter().any(|ext| lower.ends_with(ext)) {
        return false;
    }
    file.size <= MAX_AUX_BYTES
}

/// The paths of the index files that decide which shards of `files` to fetch:
/// the ones the plan must read before it can be completed.
pub fn index_files_to_read(files: &[RepoFile], format: WeightFormat) -> Vec<String> {
    let wanted = |name: &str| match format {
        WeightFormat::Safetensors | WeightFormat::Auto => is_index(name, ".safetensors"),
        WeightFormat::PyTorch => is_index(name, ".bin"),
        WeightFormat::Gguf => false,
    };
    files
        .iter()
        .filter(|f| wanted(f.name()))
        .map(|f| f.path.clone())
        .collect()
}

/// The shard file names a safetensors/PyTorch `*.index.json` refers to, each
/// once, in name order. `Err` if the JSON is not an index.
pub fn shards_in_index(json: &[u8]) -> Result<Vec<String>> {
    let value: serde_json::Value = serde_json::from_slice(json)
        .map_err(|e| PMetalError::Hub(format!("weight index is not valid JSON: {e}")))?;
    let map = value
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| PMetalError::Hub("weight index has no `weight_map`".into()))?;
    let shards: BTreeSet<String> = map
        .values()
        .filter_map(|v| v.as_str())
        .map(str::to_owned)
        .collect();
    if shards.is_empty() {
        return Err(PMetalError::Hub("weight index names no shards".into()));
    }
    Ok(shards.into_iter().collect())
}

/// Choose the files of a model download.
///
/// `files` is the repository listing; `indexes` maps the path of each
/// `*.index.json` returned by [`index_files_to_read`] to the shard names it
/// lists (from [`shards_in_index`]). `gguf_quant` picks a GGUF quantization by
/// label (case-insensitive); `None` picks the best general-purpose one.
pub fn plan_model(
    files: &[RepoFile],
    indexes: &HashMap<String, Vec<String>>,
    format: WeightFormat,
    gguf_quant: Option<&str>,
) -> Result<ModelPlan> {
    let has = |pred: &dyn Fn(&RepoFile) -> bool| files.iter().any(pred);
    let has_safetensors = has(&|f| f.name().ends_with(".safetensors"));
    let has_gguf = !gguf_variants(files).is_empty();
    let has_pytorch = has(&|f| is_pytorch_weight(f.name()));

    let chosen = match format {
        WeightFormat::Safetensors => WeightFormat::Safetensors,
        WeightFormat::Gguf => WeightFormat::Gguf,
        WeightFormat::PyTorch => WeightFormat::PyTorch,
        WeightFormat::Auto if has_safetensors => WeightFormat::Safetensors,
        WeightFormat::Auto if has_gguf => WeightFormat::Gguf,
        WeightFormat::Auto if has_pytorch => {
            return Err(PMetalError::Hub(
                "this repository publishes only PyTorch pickle weights (pytorch_model*.bin), \
                 which pmetal does not load directly; ask for the PyTorch format explicitly, or \
                 look for a safetensors or GGUF conversion of the model"
                    .into(),
            ));
        }
        WeightFormat::Auto => {
            return Err(PMetalError::Hub(
                "this repository has no model weights (no .safetensors, .gguf or pytorch_model*.bin)"
                    .into(),
            ));
        }
    };

    let mut selected: BTreeMap<String, RepoFile> = BTreeMap::new();
    let by_path: HashMap<&str, &RepoFile> = files.iter().map(|f| (f.path.as_str(), f)).collect();

    let weights = match chosen {
        WeightFormat::Safetensors => {
            if !has_safetensors {
                return Err(PMetalError::Hub(
                    "this repository has no .safetensors files".into(),
                ));
            }
            select_sharded(
                files,
                &by_path,
                indexes,
                ".safetensors",
                |n| n.ends_with(".safetensors"),
                &mut selected,
            )?;
            SelectedWeights::Safetensors
        }
        WeightFormat::PyTorch => {
            if !has_pytorch {
                return Err(PMetalError::Hub(
                    "this repository has no pytorch_model*.bin files".into(),
                ));
            }
            select_sharded(
                files,
                &by_path,
                indexes,
                ".bin",
                is_pytorch_weight,
                &mut selected,
            )?;
            SelectedWeights::PyTorch
        }
        WeightFormat::Gguf => {
            let variants = gguf_variants(files);
            let variant = pick_gguf(&variants, gguf_quant)?;
            for f in &variant.files {
                selected.insert(f.path.clone(), f.clone());
            }
            SelectedWeights::Gguf {
                quant: variant.quant.clone(),
            }
        }
        WeightFormat::Auto => unreachable!("resolved above"),
    };

    for file in files {
        if wanted_aux(file) {
            selected
                .entry(file.path.clone())
                .or_insert_with(|| file.clone());
        }
    }
    Ok(ModelPlan {
        weights,
        files: selected.into_values().collect(),
    })
}

/// Add the weight files of one sharded format to `selected`: with an index in
/// a directory, exactly the shards it names (a repository can hold a second
/// copy of the weights in another layout next to them); without one, every
/// weight file of the directory.
fn select_sharded(
    files: &[RepoFile],
    by_path: &HashMap<&str, &RepoFile>,
    indexes: &HashMap<String, Vec<String>>,
    suffix: &str,
    is_shard: impl Fn(&str) -> bool,
    selected: &mut BTreeMap<String, RepoFile>,
) -> Result<()> {
    let mut indexed_dirs = BTreeSet::new();
    for file in files.iter().filter(|f| is_index(f.name(), suffix)) {
        let Some(shards) = indexes.get(&file.path) else {
            return Err(PMetalError::Hub(format!(
                "{} must be read to choose its shards, but was not provided",
                file.path
            )));
        };
        indexed_dirs.insert(file.dir().to_owned());
        selected.insert(file.path.clone(), file.clone());
        for shard in shards {
            let path = if file.dir().is_empty() {
                shard.clone()
            } else {
                format!("{}/{shard}", file.dir())
            };
            let found = by_path.get(path.as_str()).ok_or_else(|| {
                PMetalError::Hub(format!(
                    "{} lists the shard {shard}, which the repository does not contain",
                    file.path
                ))
            })?;
            selected.insert(path, (*found).clone());
        }
    }
    for file in files {
        if is_shard(file.name()) && !indexed_dirs.contains(file.dir()) {
            selected.insert(file.path.clone(), file.clone());
        }
    }
    Ok(())
}

/// Split `stem` of `name-00001-of-00004` into `(base, number, count)`.
fn shard_parts(stem: &str) -> Option<(&str, u32, u32)> {
    let (head, count) = stem.rsplit_once("-of-")?;
    let (base, no) = head.rsplit_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (no.len() == count.len() && digits(no) && digits(count))
        .then(|| Some((base, no.parse().ok()?, count.parse().ok()?)))
        .flatten()
}

/// The quantization label in a GGUF file or directory name, upper-cased:
/// `Qwen3-0.6B-Q4_K_M` is `Q4_K_M`, `UD-Q4_K_XL` is `Q4_K_XL`, `…-BF16` is `BF16`.
pub fn gguf_quant_label(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let upper = stem.to_ascii_uppercase();
    upper
        .split(['-', '.', ' '])
        .rev()
        .find(|tok| is_quant_token(tok))
        .map(str::to_owned)
}

fn is_quant_token(tok: &str) -> bool {
    if matches!(tok, "F32" | "F16" | "BF16" | "MXFP4" | "NVFP4") {
        return true;
    }
    let rest = ["IQ", "TQ", "Q"]
        .iter()
        .find_map(|p| tok.strip_prefix(p))
        .unwrap_or("");
    let mut chars = rest.chars();
    chars.next().is_some_and(|c| c.is_ascii_digit())
        && rest
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Whether a GGUF file is a multimodal projector or an importance matrix, not
/// the language model.
fn is_gguf_auxiliary(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("mmproj") || lower.contains("imatrix")
}

/// Every complete GGUF quantization in `files`, smallest first. A split model
/// counts once, with all its shards; a set with a shard missing from the
/// listing is left out (it could not be loaded).
pub fn gguf_variants(files: &[RepoFile]) -> Vec<GgufVariant> {
    /// `(shard number, shard count, file)`; a single file is shard 1 of 1.
    type Shard<'a> = (u32, u32, &'a RepoFile);
    // (directory, base name) -> shards.
    let mut groups: BTreeMap<(String, String), Vec<Shard<'_>>> = BTreeMap::new();
    for file in files {
        let name = file.name();
        let Some(stem) = name.strip_suffix(".gguf") else {
            continue;
        };
        if is_gguf_auxiliary(name) {
            continue;
        }
        let (base, no, count) = shard_parts(stem).unwrap_or((stem, 1, 1));
        groups
            .entry((file.dir().to_owned(), base.to_owned()))
            .or_default()
            .push((no, count, file));
    }
    let mut variants = Vec::new();
    for ((dir, base), mut shards) in groups {
        shards.sort_by_key(|(no, _, _)| *no);
        let count = shards[0].1;
        let complete = shards.len() as u32 == count
            && shards
                .iter()
                .enumerate()
                .all(|(i, (no, c, _))| *no == i as u32 + 1 && *c == count);
        if !complete {
            continue;
        }
        // The label is in the file name, or, when a repository sorts its
        // quantizations into directories, in the directory name.
        let label = gguf_quant_label(&base).or_else(|| {
            Path::new(&dir)
                .components()
                .rev()
                .find_map(|c| gguf_quant_label(&c.as_os_str().to_string_lossy()))
        });
        variants.push(GgufVariant {
            quant: label.unwrap_or_else(|| base.to_ascii_uppercase()),
            files: shards.into_iter().map(|(_, _, f)| f.clone()).collect(),
        });
    }
    variants.sort_by(|a, b| {
        a.total_bytes()
            .cmp(&b.total_bytes())
            .then_with(|| a.quant.cmp(&b.quant))
    });
    variants
}

/// Quantizations in order of preference when the caller names none: the usual
/// quality/size sweet spots first.
const GGUF_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q6_K", "Q4_K_XL", "Q5_K_XL", "IQ4_XS", "IQ4_NL",
    "Q3_K_M", "Q4_0", "Q8_0", "F16", "BF16",
];

fn pick_gguf<'a>(variants: &'a [GgufVariant], wanted: Option<&str>) -> Result<&'a GgufVariant> {
    if variants.is_empty() {
        return Err(PMetalError::Hub(
            "this repository has no complete .gguf model".into(),
        ));
    }
    let available = || {
        variants
            .iter()
            .map(|v| v.quant.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(wanted) = wanted {
        let wanted = wanted.trim().to_ascii_uppercase();
        let mut matches = variants.iter().filter(|v| v.quant == wanted);
        return match (matches.next(), matches.next()) {
            (Some(only), None) => Ok(only),
            // The same quantization in several places: the smallest listing
            // order (variants are sorted by size) is the first.
            (Some(first), Some(_)) => Ok(first),
            (None, _) => Err(PMetalError::Hub(format!(
                "no {wanted} quantization in this repository; available: {}",
                available()
            ))),
        };
    }
    GGUF_PREFERENCE
        .iter()
        .find_map(|pref| variants.iter().find(|v| v.quant == *pref))
        // Nothing recognised: the smallest model is the one most likely to fit.
        .or_else(|| variants.first())
        .ok_or_else(|| PMetalError::Hub("no GGUF model to choose".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(path: &str, size: u64) -> RepoFile {
        RepoFile::new(path, size)
    }

    fn paths(plan: &ModelPlan) -> Vec<&str> {
        plan.files.iter().map(|f| f.path.as_str()).collect()
    }

    const AUX: [&str; 3] = ["config.json", "tokenizer.json", "README.md"];

    fn aux() -> Vec<RepoFile> {
        AUX.iter().map(|p| f(p, 100)).collect()
    }

    #[test]
    fn single_file_safetensors_takes_the_weights_and_the_configs() {
        let mut files = aux();
        files.push(f("model.safetensors", 1000));
        files.push(f(".gitattributes", 5));
        files.push(f("pytorch_model.bin", 1000));
        files.push(f("onnx/model.onnx", 1000));
        files.push(f("assets/logo.png", 10));
        let plan = plan_model(&files, &HashMap::new(), WeightFormat::Auto, None).unwrap();
        assert_eq!(plan.weights, SelectedWeights::Safetensors);
        assert_eq!(
            paths(&plan),
            [
                "README.md",
                "config.json",
                "model.safetensors",
                "tokenizer.json"
            ]
        );
    }

    #[test]
    fn an_index_names_the_shards_and_a_second_layout_is_left_behind() {
        let mut files = aux();
        files.extend([
            f("model.safetensors.index.json", 50),
            f("model-00001-of-00002.safetensors", 1000),
            f("model-00002-of-00002.safetensors", 1000),
            f("consolidated.safetensors", 2000),
        ]);
        let indexes = HashMap::from([(
            "model.safetensors.index.json".to_owned(),
            vec![
                "model-00001-of-00002.safetensors".to_owned(),
                "model-00002-of-00002.safetensors".to_owned(),
            ],
        )]);
        let plan = plan_model(&files, &indexes, WeightFormat::Auto, None).unwrap();
        assert!(!paths(&plan).contains(&"consolidated.safetensors"));
        assert!(paths(&plan).contains(&"model-00002-of-00002.safetensors"));
        assert!(paths(&plan).contains(&"model.safetensors.index.json"));
        assert_eq!(plan.weight_files().count(), 2);
    }

    #[test]
    fn an_index_naming_a_missing_shard_is_an_error() {
        let files = vec![
            f("model.safetensors.index.json", 50),
            f("model-00001-of-00002.safetensors", 1000),
        ];
        let indexes = HashMap::from([(
            "model.safetensors.index.json".to_owned(),
            vec![
                "model-00001-of-00002.safetensors".to_owned(),
                "model-00002-of-00002.safetensors".to_owned(),
            ],
        )]);
        let err = plan_model(&files, &indexes, WeightFormat::Auto, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model-00002-of-00002.safetensors"), "{err}");
    }

    #[test]
    fn indexes_apply_per_directory() {
        let files = vec![
            f("text/model.safetensors.index.json", 50),
            f("text/model-00001-of-00001.safetensors", 1000),
            f("text/stray.safetensors", 1000),
            f("vision/model.safetensors", 500),
        ];
        let indexes = HashMap::from([(
            "text/model.safetensors.index.json".to_owned(),
            vec!["model-00001-of-00001.safetensors".to_owned()],
        )]);
        let plan = plan_model(&files, &indexes, WeightFormat::Auto, None).unwrap();
        assert_eq!(
            paths(&plan),
            [
                "text/model-00001-of-00001.safetensors",
                "text/model.safetensors.index.json",
                "vision/model.safetensors",
            ]
        );
    }

    #[test]
    fn a_gguf_only_repository_gets_one_quantization_by_preference() {
        let mut files = aux();
        files.extend([
            f("m-Q8_0.gguf", 800),
            f("m-Q4_K_M.gguf", 400),
            f("m-Q2_K.gguf", 200),
            f("mmproj-F16.gguf", 50),
        ]);
        let plan = plan_model(&files, &HashMap::new(), WeightFormat::Auto, None).unwrap();
        assert_eq!(
            plan.weights,
            SelectedWeights::Gguf {
                quant: "Q4_K_M".into()
            }
        );
        assert!(paths(&plan).contains(&"m-Q4_K_M.gguf"));
        assert!(!paths(&plan).contains(&"m-Q8_0.gguf"));
        assert!(!paths(&plan).contains(&"mmproj-F16.gguf"));
    }

    #[test]
    fn a_named_quantization_is_matched_case_insensitively() {
        let files = vec![f("m-Q8_0.gguf", 800), f("m-Q4_K_M.gguf", 400)];
        let plan = plan_model(&files, &HashMap::new(), WeightFormat::Gguf, Some("q8_0")).unwrap();
        assert_eq!(paths(&plan), ["m-Q8_0.gguf"]);
    }

    #[test]
    fn an_unavailable_quantization_lists_what_exists() {
        let files = vec![f("m-Q8_0.gguf", 800), f("m-Q4_K_M.gguf", 400)];
        let err = plan_model(&files, &HashMap::new(), WeightFormat::Gguf, Some("Q2_K"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Q2_K") && err.contains("Q4_K_M") && err.contains("Q8_0"),
            "{err}"
        );
    }

    #[test]
    fn a_split_gguf_comes_whole_and_counts_once() {
        let files = vec![
            f("m-Q4_K_M-00001-of-00003.gguf", 300),
            f("m-Q4_K_M-00002-of-00003.gguf", 300),
            f("m-Q4_K_M-00003-of-00003.gguf", 100),
            f("m-Q8_0.gguf", 900),
        ];
        let variants = gguf_variants(&files);
        assert_eq!(variants.len(), 2);
        let q4 = variants.iter().find(|v| v.quant == "Q4_K_M").unwrap();
        assert_eq!(q4.files.len(), 3);
        assert_eq!(q4.total_bytes(), 700);
        let plan = plan_model(&files, &HashMap::new(), WeightFormat::Auto, None).unwrap();
        assert_eq!(plan.weight_files().count(), 3);
    }

    #[test]
    fn a_split_gguf_with_a_shard_missing_is_not_offered() {
        let files = vec![
            f("m-Q4_K_M-00001-of-00003.gguf", 300),
            f("m-Q4_K_M-00003-of-00003.gguf", 100),
            f("m-Q8_0.gguf", 900),
        ];
        let variants = gguf_variants(&files);
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].quant, "Q8_0");
    }

    #[test]
    fn quantizations_sorted_into_directories_are_labelled_by_directory() {
        let files = vec![
            f("Q4_K_M/model-00001-of-00002.gguf", 100),
            f("Q4_K_M/model-00002-of-00002.gguf", 100),
            f("BF16/model.gguf", 900),
        ];
        let labels: Vec<_> = gguf_variants(&files).into_iter().map(|v| v.quant).collect();
        assert_eq!(labels, ["Q4_K_M", "BF16"]);
    }

    #[test]
    fn quant_labels_are_read_from_real_file_names() {
        for (name, label) in [
            ("Qwen3-0.6B-Q4_K_M.gguf", Some("Q4_K_M")),
            ("Qwen3-0.6B-UD-Q4_K_XL.gguf", Some("Q4_K_XL")),
            ("gemma-3-1b-it-IQ4_XS.gguf", Some("IQ4_XS")),
            ("Qwen3-0.6B-BF16.gguf", Some("BF16")),
            ("llama-2-7b.Q5_K_S.gguf", Some("Q5_K_S")),
            ("model-f16.gguf", Some("F16")),
            ("Meta-Llama-3-8B.gguf", None),
            // "Q" in a model's own name is not a quantization.
            ("Qwen2.5-7B.gguf", None),
        ] {
            assert_eq!(gguf_quant_label(name).as_deref(), label, "{name}");
        }
    }

    #[test]
    fn pytorch_only_is_an_explicit_error_unless_asked_for() {
        let mut files = aux();
        files.push(f("pytorch_model.bin", 1000));
        files.push(f("training_args.bin", 10));
        let err = plan_model(&files, &HashMap::new(), WeightFormat::Auto, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("PyTorch"), "{err}");
        let plan = plan_model(&files, &HashMap::new(), WeightFormat::PyTorch, None).unwrap();
        assert_eq!(plan.weights, SelectedWeights::PyTorch);
        assert!(paths(&plan).contains(&"pytorch_model.bin"));
        assert!(!paths(&plan).contains(&"training_args.bin"));
    }

    #[test]
    fn a_repository_without_weights_is_an_error() {
        let err = plan_model(&aux(), &HashMap::new(), WeightFormat::Auto, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no model weights"), "{err}");
    }

    #[test]
    fn oversized_unknown_files_are_not_configs() {
        let files = vec![f("model.safetensors", 10), f("blob.dat", 600 * 1024 * 1024)];
        let plan = plan_model(&files, &HashMap::new(), WeightFormat::Auto, None).unwrap();
        assert_eq!(paths(&plan), ["model.safetensors"]);
    }

    #[test]
    fn indexes_are_found_and_parsed() {
        let files = vec![
            f("model.safetensors.index.json", 1),
            f("pytorch_model.bin.index.json", 1),
            f("model-00001-of-00001.safetensors", 1),
        ];
        assert_eq!(
            index_files_to_read(&files, WeightFormat::Auto),
            ["model.safetensors.index.json"]
        );
        assert_eq!(
            index_files_to_read(&files, WeightFormat::PyTorch),
            ["pytorch_model.bin.index.json"]
        );
        let shards = shards_in_index(br#"{"weight_map":{"a":"s2","b":"s1","c":"s2"}}"#).unwrap();
        assert_eq!(shards, ["s1", "s2"]);
        assert!(shards_in_index(b"{}").is_err());
        assert!(shards_in_index(b"not json").is_err());
    }
}
