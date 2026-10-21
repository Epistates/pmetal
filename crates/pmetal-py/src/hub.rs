//! The Hugging Face Hub: download, verify, search, size for this Mac, publish.
//!
//! All of it is `pmetal-hub`, the code `pmetal download`, `verify`, `search`
//! and `upload` run. Every call here waits with the GIL released, and Ctrl-C
//! abandons a transfer (what was fetched stays in the cache for the next
//! call).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pmetal_core::SecretString;
use pmetal_hub::search::{SearchOptions, SearchSort, search_models_with};
use pmetal_hub::{
    DeviceSpec, DownloadOptions, HubOptions, ProgressCallback, TransferProgress, UploadOptions,
    Verification, WeightFormat,
};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::{config_err, pmetal_to_pyerr};
use crate::json::to_py;
use crate::runtime::{block_on_interruptible, block_on_interruptible_with};

/// The connection settings every Hub call takes.
#[derive(Default, Clone)]
pub struct HubArgs {
    pub revision: Option<String>,
    pub token: Option<String>,
    pub offline: Option<bool>,
    pub endpoint: Option<String>,
    pub cache_dir: Option<PathBuf>,
}

impl HubArgs {
    fn hub_options(&self) -> HubOptions {
        HubOptions {
            token: self.token.as_deref().map(SecretString::new),
            endpoint: self.endpoint.clone(),
            cache_dir: self.cache_dir.clone(),
            offline: self.offline,
            ..HubOptions::default()
        }
    }

    fn download_options(&self) -> DownloadOptions {
        DownloadOptions {
            hub: self.hub_options(),
            revision: self.revision.clone(),
            ..DownloadOptions::default()
        }
    }
}

/// A model reference as a local directory: a directory path is itself, a Hub
/// id is downloaded (or served from the cache). Ctrl-C interrupts the wait.
pub fn resolve_model(py: Python<'_>, reference: &str, hub: &HubArgs) -> PyResult<PathBuf> {
    let id = reference.to_string();
    let options = hub.download_options();
    let dir = block_on_interruptible(py, async move {
        pmetal_hub::resolve_model_path_with(&id, &options).await
    })?
    .map_err(pmetal_to_pyerr)?;
    // A reference that is neither a Hub id nor a directory resolves to itself.
    if !dir.is_dir() {
        return Err(PyFileNotFoundError::new_err(format!(
            "{reference}: not a model directory, and not a Hub id (org/name)"
        )));
    }
    Ok(dir)
}

fn weight_format(name: &str) -> PyResult<WeightFormat> {
    match name {
        "auto" => Ok(WeightFormat::Auto),
        "safetensors" => Ok(WeightFormat::Safetensors),
        "gguf" => Ok(WeightFormat::Gguf),
        "pytorch" => Ok(WeightFormat::PyTorch),
        other => Err(PyValueError::new_err(format!(
            "weights must be 'auto', 'safetensors', 'gguf' or 'pytorch', not {other:?}"
        ))),
    }
}

fn verification(name: &str) -> PyResult<Verification> {
    match name {
        "none" => Ok(Verification::None),
        "size" => Ok(Verification::Size),
        "sha256" => Ok(Verification::Sha256),
        other => Err(PyValueError::new_err(format!(
            "verify must be 'none', 'size' or 'sha256', not {other:?}"
        ))),
    }
}

/// A progress callback that calls `callback` with a dict, and keeps what it
/// raised so the transfer can be abandoned and the exception raised.
struct ProgressBridge {
    callback: ProgressCallback,
    failure: Arc<Mutex<Option<PyErr>>>,
}

impl ProgressBridge {
    fn new(callable: Py<PyAny>) -> Self {
        let failure: Arc<Mutex<Option<PyErr>>> = Arc::default();
        let slot = failure.clone();
        let callback: ProgressCallback = Arc::new(move |p: &TransferProgress| {
            if slot.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
                return;
            }
            Python::attach(|py| {
                let result = progress_dict(py, p).and_then(|d| callable.bind(py).call1((d,)));
                if let Err(e) = result {
                    *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                }
            });
        });
        Self { callback, failure }
    }

    fn poll(&self) -> impl FnMut() -> Option<PyErr> + use<> {
        let failure = self.failure.clone();
        move || failure.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Raise what the callback raised, if it did.
    fn finish(&self) -> PyResult<()> {
        match self
            .failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn progress_dict<'py>(py: Python<'py>, p: &TransferProgress) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("files_done", p.files_done)?;
    d.set_item("total_files", p.total_files)?;
    d.set_item("bytes_done", p.bytes_done)?;
    d.set_item("total_bytes", p.total_bytes)?;
    d.set_item("bytes_per_sec", p.bytes_per_sec)?;
    d.set_item("eta_seconds", p.eta.map(|eta| eta.as_secs_f64()))?;
    d.set_item("current_file", p.current_file.as_deref())?;
    d.set_item("finished", p.finished)?;
    Ok(d)
}

/// Download a model from the Hugging Face Hub and return its directory.
///
/// Fetches the configs, the tokenizer and one set of weights, all from one
/// commit, concurrently, into the standard Hub cache, then checks them. With
/// no `revision` a complete cached copy is returned without contacting the
/// Hub; name `"main"` to check for a newer commit.
///
/// Args:
///     model_id: `org/name`, or `org/name:Q4_K_M` for one GGUF quantization.
///     revision: Branch, tag or commit.
///     token: Access token for private and gated repositories; `HF_TOKEN` or `hf auth login` when None.
///     weights: `"auto"` (safetensors, else GGUF), `"safetensors"`, `"gguf"` or `"pytorch"`.
///     gguf_quant: The GGUF quantization to fetch (`"Q4_K_M"`, `"Q8_0"`); implies `weights="gguf"`.
///     verify: `"none"`, `"size"` (every file exactly the size the Hub lists) or `"sha256"` (large files hashed too).
///     offline: Never touch the network: the cache answers or the call fails. None follows `HF_HUB_OFFLINE`.
///     endpoint: A mirror or private deployment; `HF_ENDPOINT` when None.
///     cache_dir: The Hub cache directory; the standard one when None.
///     max_workers: Files fetched at once (eight when None).
///     progress: Called with a dict (`bytes_done`, `total_bytes`, `files_done`, `total_files`, `bytes_per_sec`, `eta_seconds`, `current_file`, `finished`). If it raises, the download stops and the exception propagates.
///
/// Raises:
///     HubError: The Hub could not be reached, the repository does not exist, or a file failed its check.
#[pyfunction]
#[pyo3(signature = (model_id, *, revision=None, token=None, weights="auto", gguf_quant=None,
    verify="size", offline=None, endpoint=None, cache_dir=None, max_workers=None, progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn download_model(
    py: Python<'_>,
    model_id: &str,
    revision: Option<String>,
    token: Option<String>,
    weights: &str,
    gguf_quant: Option<String>,
    verify: &str,
    offline: Option<bool>,
    endpoint: Option<String>,
    cache_dir: Option<PathBuf>,
    max_workers: Option<usize>,
    progress: Option<Py<PyAny>>,
) -> PyResult<String> {
    let hub = HubArgs {
        revision,
        token,
        offline,
        endpoint,
        cache_dir,
    };
    let (id, suffix_quant) = pmetal_hub::split_quant_selector(model_id);
    let gguf_quant = gguf_quant.or_else(|| suffix_quant.map(str::to_owned));
    let mut weights = weight_format(weights)?;
    if gguf_quant.is_some() && weights == WeightFormat::Auto {
        weights = WeightFormat::Gguf;
    }
    let bridge = progress.map(ProgressBridge::new);
    let options = DownloadOptions {
        weights,
        gguf_quant,
        verify: verification(verify)?,
        max_workers,
        progress: bridge.as_ref().map(|b| b.callback.clone()),
        ..hub.download_options()
    };
    let id = id.to_string();
    let future = async move { pmetal_hub::download_model_with(&id, &options).await };
    let path = match &bridge {
        Some(bridge) => block_on_interruptible_with(py, future, bridge.poll())?,
        None => block_on_interruptible(py, future)?,
    };
    if let Some(bridge) = &bridge {
        bridge.finish()?;
    }
    path.map(|p| p.to_string_lossy().into_owned())
        .map_err(pmetal_to_pyerr)
}

/// Download one file of a model repository and return its path.
///
/// Args:
///     model_id: `org/name`.
///     filename: The file's path in the repository (`"config.json"`).
///     revision: Branch, tag or commit.
///     token: Access token; `HF_TOKEN` or `hf auth login` when None.
///     offline: Serve the cached copy or fail. None follows `HF_HUB_OFFLINE`.
///     endpoint: A mirror or private deployment.
///     cache_dir: The Hub cache directory.
#[pyfunction]
#[pyo3(signature = (model_id, filename, *, revision=None, token=None, offline=None,
    endpoint=None, cache_dir=None))]
#[allow(clippy::too_many_arguments)]
pub fn download_file(
    py: Python<'_>,
    model_id: &str,
    filename: &str,
    revision: Option<String>,
    token: Option<String>,
    offline: Option<bool>,
    endpoint: Option<String>,
    cache_dir: Option<PathBuf>,
) -> PyResult<String> {
    let options = HubArgs {
        revision,
        token,
        offline,
        endpoint,
        cache_dir,
    }
    .download_options();
    let (id, name) = (model_id.to_string(), filename.to_string());
    block_on_interruptible(py, async move {
        pmetal_hub::download_file_with(&id, &name, &options).await
    })?
    .map(|p| p.to_string_lossy().into_owned())
    .map_err(pmetal_to_pyerr)
}

/// What a model download would fetch.
#[pyclass(name = "DownloadPlan", module = "pmetal", frozen, skip_from_py_object)]
pub struct PyDownloadPlan {
    /// The repository.
    #[pyo3(get)]
    repo_id: String,
    /// The commit the revision resolved to.
    #[pyo3(get)]
    commit: String,
    /// `(path, size_in_bytes)` of every file, in path order.
    #[pyo3(get)]
    files: Vec<(String, u64)>,
    /// The size of all of them.
    #[pyo3(get)]
    total_bytes: u64,
}

#[pymethods]
impl PyDownloadPlan {
    fn __repr__(&self) -> String {
        format!(
            "DownloadPlan(repo_id={:?}, commit={:?}, files={}, total_bytes={})",
            self.repo_id,
            self.commit,
            self.files.len(),
            self.total_bytes
        )
    }
}

/// The commit, the files and the size a `download_model` call would fetch,
/// without fetching them. Takes the same selection arguments.
#[pyfunction]
#[pyo3(signature = (model_id, *, revision=None, token=None, weights="auto", gguf_quant=None,
    offline=None, endpoint=None, cache_dir=None))]
#[allow(clippy::too_many_arguments)]
pub fn plan_download(
    py: Python<'_>,
    model_id: &str,
    revision: Option<String>,
    token: Option<String>,
    weights: &str,
    gguf_quant: Option<String>,
    offline: Option<bool>,
    endpoint: Option<String>,
    cache_dir: Option<PathBuf>,
) -> PyResult<PyDownloadPlan> {
    let hub = HubArgs {
        revision,
        token,
        offline,
        endpoint,
        cache_dir,
    };
    let (id, suffix_quant) = pmetal_hub::split_quant_selector(model_id);
    let gguf_quant = gguf_quant.or_else(|| suffix_quant.map(str::to_owned));
    let mut weights = weight_format(weights)?;
    if gguf_quant.is_some() && weights == WeightFormat::Auto {
        weights = WeightFormat::Gguf;
    }
    let options = DownloadOptions {
        weights,
        gguf_quant,
        ..hub.download_options()
    };
    let id = id.to_string();
    let plan = block_on_interruptible(py, async move {
        pmetal_hub::plan_model_download(&id, &options).await
    })?
    .map_err(pmetal_to_pyerr)?;
    Ok(PyDownloadPlan {
        total_bytes: plan.model.total_bytes(),
        files: plan
            .model
            .files
            .iter()
            .map(|f| (f.path.clone(), f.size))
            .collect(),
        repo_id: plan.repo_id,
        commit: plan.commit,
    })
}

/// What `verify_model` found in a model directory.
#[pyclass(name = "ModelReport", module = "pmetal", frozen, skip_from_py_object)]
pub struct PyModelReport {
    /// The weight format present: `"safetensors"`, `"gguf"`, `"pytorch"`, or None without weights.
    #[pyo3(get)]
    format: Option<String>,
    /// The weight files, in path order.
    #[pyo3(get)]
    weight_files: Vec<String>,
    /// Their total size in bytes.
    #[pyo3(get)]
    weight_bytes: u64,
    /// Why the directory cannot be loaded; empty when it is complete.
    #[pyo3(get)]
    problems: Vec<String>,
    /// What is absent but not necessarily needed (a missing `config.json`).
    #[pyo3(get)]
    warnings: Vec<String>,
}

#[pymethods]
impl PyModelReport {
    /// Whether nothing is missing or damaged.
    #[getter]
    fn is_complete(&self) -> bool {
        self.problems.is_empty()
    }

    /// Whether a language-model loader can open the directory.
    #[getter]
    fn is_loadable_language_model(&self) -> bool {
        self.problems.is_empty() && self.warnings.is_empty()
    }

    fn __bool__(&self) -> bool {
        self.problems.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelReport(format={:?}, weight_files={}, weight_bytes={}, problems={:?}, warnings={:?})",
            self.format,
            self.weight_files.len(),
            self.weight_bytes,
            self.problems,
            self.warnings
        )
    }
}

/// Check that a model directory is complete and undamaged, reading headers
/// only: every weight file there and as long as its header says, a GGUF set
/// complete, a readable `config.json`.
///
/// Args:
///     model: A model directory, or a Hub id with a copy in the Hub cache.
///     cache_dir: The Hub cache directory, for a Hub id.
#[pyfunction]
#[pyo3(signature = (model, *, cache_dir=None))]
pub fn verify_model(
    py: Python<'_>,
    model: PathBuf,
    cache_dir: Option<PathBuf>,
) -> PyResult<PyModelReport> {
    let model = model.to_string_lossy().into_owned();
    let model = model.as_str();
    let dir = if std::path::Path::new(model).is_dir() || !pmetal_hub::is_hf_id(model) {
        PathBuf::from(model)
    } else {
        // A Hub id is checked where the cache has it; nothing is fetched.
        let hub = HubArgs {
            offline: Some(true),
            cache_dir,
            ..HubArgs::default()
        };
        resolve_model(py, model, &hub)?
    };
    let report = py.detach(|| pmetal_hub::verify_model_dir(&dir));
    Ok(PyModelReport {
        format: report.format.map(|f| {
            match f {
                WeightFormat::Safetensors => "safetensors",
                WeightFormat::Gguf => "gguf",
                WeightFormat::PyTorch => "pytorch",
                WeightFormat::Auto => "auto",
            }
            .to_string()
        }),
        weight_files: report
            .weight_files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
        weight_bytes: report.weight_bytes,
        problems: report.problems,
        warnings: report.warnings,
    })
}

/// The Mac's GPU memory budget and bandwidth, asked of the Metal device.
fn device_spec() -> Option<DeviceSpec> {
    let ctx = pmetal_metal::context::MetalContext::global().ok()?;
    let props = ctx.properties();
    Some(DeviceSpec::from_device(
        props.recommended_working_set_size,
        props.memory_bandwidth_gbps,
        props.has_unified_memory,
    ))
}

fn require_device() -> PyResult<DeviceSpec> {
    device_spec().ok_or_else(|| {
        crate::error::backend_err("no Metal device: model fit can only be estimated on a Mac")
    })
}

fn search_sort(name: &str) -> PyResult<SearchSort> {
    match name {
        "downloads" => Ok(SearchSort::Downloads),
        "likes" => Ok(SearchSort::Likes),
        "trending" => Ok(SearchSort::Trending),
        "updated" => Ok(SearchSort::Updated),
        other => Err(PyValueError::new_err(format!(
            "sort must be 'downloads', 'likes', 'trending' or 'updated', not {other:?}"
        ))),
    }
}

/// Search the Hub for models, with how each would run on this Mac.
///
/// Each result is a dict of the Hub's facts (`model_id`, `downloads`, `likes`,
/// `tags`, `safetensors_bytes`, `estimated_params_b`, `license`, `gated`, ...)
/// plus `fit`: the `estimate_fit` dict for this device at `context_length`, or
/// None when the Hub gave no size to estimate from (or there is no Metal
/// device).
///
/// Args:
///     query: Text matched against model names.
///     limit: At most this many results (100 at most).
///     tags: Tags every result must have (`"gguf"`, `"mlx"`, `"moe"`, `"license:apache-2.0"`).
///     author: Only models from this user or organization.
///     pipeline_tag: `"text-generation"` by default; `"any"` searches every kind of model.
///     sort: `"downloads"`, `"likes"`, `"trending"` or `"updated"`.
///     context_length: The context the fit is estimated for.
///     token, endpoint: Connection settings, as for `download_model`.
#[pyfunction]
#[pyo3(signature = (query, *, limit=15, tags=None, author=None,
    pipeline_tag="text-generation", sort="downloads", context_length=4096,
    token=None, endpoint=None))]
#[allow(clippy::too_many_arguments)]
pub fn search_models<'py>(
    py: Python<'py>,
    query: &str,
    limit: usize,
    tags: Option<Vec<String>>,
    author: Option<String>,
    pipeline_tag: &str,
    sort: &str,
    context_length: u32,
    token: Option<String>,
    endpoint: Option<String>,
) -> PyResult<Bound<'py, PyList>> {
    let mut options = SearchOptions::new(query);
    options.limit = limit;
    options.tags = tags.unwrap_or_default();
    options.author = author;
    options.pipeline_tag = (pipeline_tag != "any").then(|| pipeline_tag.to_string());
    options.sort = search_sort(sort)?;
    options.hub = HubArgs {
        token,
        endpoint,
        ..HubArgs::default()
    }
    .hub_options();
    let results = block_on_interruptible(py, async move { search_models_with(&options).await })?
        .map_err(pmetal_to_pyerr)?;

    let device = device_spec();
    let list = PyList::empty(py);
    for result in &results {
        let mut value = serde_json::to_value(result).map_err(config_err)?;
        let fit = device
            .as_ref()
            .and_then(|d| result.fit(d, context_length))
            .map(|fit| serde_json::to_value(fit).map_err(config_err))
            .transpose()?
            .unwrap_or(serde_json::Value::Null);
        if let Some(object) = value.as_object_mut() {
            object.insert("fit".to_string(), fit);
        }
        list.append(to_py(py, &value)?)?;
    }
    Ok(list)
}

/// Estimate how a model runs on this Mac: weight and KV-cache memory, fit,
/// decode speed (from memory bandwidth) and LoRA training memory.
///
/// Args:
///     model: A model directory, or a Hub id (its `config.json` and tensor sizes are read from the Hub).
///     context_length: The context the KV cache and training estimate are for.
///     kv_cache_bits: KV-cache quantization (8 or 4); fp16 when None.
///     token, endpoint: Connection settings for a Hub id.
///
/// Returns:
///     A dict: `fit_level` (`"Fits"`, `"Tight"` or `"TooLarge"`), `weights_gb`,
///     `kv_cache_gb`, `total_required_gb`, `available_gb`, `utilization_pct`,
///     `estimated_tps`, `training_memory_gb`, `training_fit`,
///     `recommended_batch_size`, `context_length`, `notes`, ...
///
/// Raises:
///     BackendError: There is no Metal device.
///     ConfigError: The directory has no readable `config.json`.
#[pyfunction]
#[pyo3(signature = (model, *, context_length=4096, kv_cache_bits=None, token=None, endpoint=None))]
pub fn estimate_fit<'py>(
    py: Python<'py>,
    model: PathBuf,
    context_length: u32,
    kv_cache_bits: Option<u8>,
    token: Option<String>,
    endpoint: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let model = model.to_string_lossy().into_owned();
    let model = model.as_str();
    let device = require_device()?;
    let spec = if std::path::Path::new(model).is_dir() {
        let dir = PathBuf::from(model);
        py.detach(|| pmetal_hub::fit::model_spec_from_dir(&dir))
            .map(|s| s.with_context(context_length))
            .ok_or_else(|| {
                config_err(format!(
                    "{model} has no readable config.json to estimate from"
                ))
            })?
    } else {
        let id = model.to_string();
        let options = HubArgs {
            token,
            endpoint,
            ..HubArgs::default()
        }
        .hub_options();
        let meta = block_on_interruptible(py, async move {
            pmetal_hub::search::fetch_model_meta_with(&id, &options).await
        })?
        .map_err(pmetal_to_pyerr)?;
        meta.model_spec(context_length)
    };
    let estimate = pmetal_hub::estimate_fit(&spec.with_kv_cache_bits(kv_cache_bits), &device);
    let value = serde_json::to_value(estimate).map_err(config_err)?;
    to_py(py, &value)
}

/// Where an upload landed.
#[pyclass(name = "UploadReport", module = "pmetal", frozen, skip_from_py_object)]
pub struct PyUploadReport {
    /// The repository's URL.
    #[pyo3(get)]
    repo_url: String,
    /// The commit's (or pull request's) URL, when the Hub returns one.
    #[pyo3(get)]
    commit_url: Option<String>,
}

#[pymethods]
impl PyUploadReport {
    fn __repr__(&self) -> String {
        format!(
            "UploadReport(repo_url={:?}, commit_url={:?})",
            self.repo_url, self.commit_url
        )
    }
}

/// Publish a model directory to the Hub, creating the repository if needed.
///
/// The commit is atomic. Credential files, `.git` and Finder droppings are
/// never sent.
///
/// Args:
///     path: The directory (a fused model, an adapter, a quantized export).
///     repo: The target repository, `owner/name`.
///     token: Access token; `HF_TOKEN` or `hf auth login` when None.
///     private: Create the repository private (no effect on an existing one).
///     revision: Branch to commit to.
///     message: The commit message.
///     create_pr: Open a pull request instead of committing to the branch.
///     exclude: Globs, relative to `path`, to leave out.
///     endpoint: A mirror or private deployment.
///     progress: As for `download_model`.
#[pyfunction]
#[pyo3(signature = (path, repo, *, token=None, private=false, revision=None, message=None,
    create_pr=false, exclude=None, endpoint=None, progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn upload_model(
    py: Python<'_>,
    path: PathBuf,
    repo: &str,
    token: Option<String>,
    private: bool,
    revision: Option<String>,
    message: Option<String>,
    create_pr: bool,
    exclude: Option<Vec<String>>,
    endpoint: Option<String>,
    progress: Option<Py<PyAny>>,
) -> PyResult<PyUploadReport> {
    let secret = token.as_deref().map(SecretString::new);
    let bridge = progress.map(ProgressBridge::new);
    let options = UploadOptions {
        private,
        revision,
        commit_message: message,
        create_pr,
        ignore: exclude.unwrap_or_default(),
        hub: HubArgs {
            endpoint,
            ..HubArgs::default()
        }
        .hub_options(),
        progress: bridge.as_ref().map(|b| b.callback.clone()),
    };
    let repo = repo.to_string();
    let future =
        async move { pmetal_hub::upload_model(&path, &repo, secret.as_ref(), &options).await };
    let report = match &bridge {
        Some(bridge) => block_on_interruptible_with(py, future, bridge.poll())?,
        None => block_on_interruptible(py, future)?,
    };
    if let Some(bridge) = &bridge {
        bridge.finish()?;
    }
    let report = report.map_err(pmetal_to_pyerr)?;
    Ok(PyUploadReport {
        repo_url: report.repo_url,
        commit_url: report.commit_url,
    })
}
