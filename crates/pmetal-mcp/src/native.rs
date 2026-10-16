//! Tools the server implements itself rather than by running a subcommand:
//! the Hub, the local model cache, job tracking and control, and talking to a
//! running `pmetal serve`.
//!
//! Each tool has an arguments struct. The struct is the tool's input schema
//! (through `schemars`) and what the call is parsed into (through `serde`, with
//! unknown fields refused), so what the schema promises and what the handler
//! accepts are one declaration.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use pmetal_core::LrControlCommand;
use pmetal_core::jobs::DecideSpec;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use turbomcp::prelude::*;

use crate::jobs::{JobManager, JobSummary, LogPage};
use crate::policy::PathPolicy;
use crate::results::{self, schema_of, structured};
use crate::serve_client;
use crate::tooldef::Hints;
use crate::util;

/// What native tools share.
pub struct Core {
    pub jobs: JobManager,
    pub policy: PathPolicy,
    /// One permit per command that may run to completion at once.
    blocking: tokio::sync::Semaphore,
}

impl Core {
    pub fn new(jobs: JobManager, policy: PathPolicy) -> Self {
        let permits = jobs.limits().max_blocking.max(1);
        Self {
            jobs,
            policy,
            blocking: tokio::sync::Semaphore::new(permits),
        }
    }

    /// Run `pmetal <argv>` to completion. Each command may load a model, so only
    /// a few run at once; the rest wait their turn, and a cancelled request stops
    /// waiting.
    pub async fn run(&self, argv: &[String], ctx: &RequestContext) -> McpResult<String> {
        // One acquire future for the whole wait: the semaphore queues in order,
        // and a fresh future on every check would go to the back each time.
        let acquire = self.blocking.acquire();
        tokio::pin!(acquire);
        let mut check = tokio::time::interval(Duration::from_millis(250));
        let _permit = loop {
            tokio::select! {
                permit = &mut acquire => {
                    break permit.map_err(|_| McpError::internal("the server is shutting down"))?;
                }
                _ = check.tick() => {
                    if ctx.is_cancelled() {
                        return Err(McpError::cancelled("the request was cancelled"));
                    }
                }
            }
        };
        util::run_pmetal(self.jobs.binary(), argv, ctx).await
    }
}

/// A tool whose arguments are the struct `A`.
fn tool_for<A: JsonSchema>(name: &str, description: &str, hints: Hints) -> Tool {
    let mut schema = schema_of::<A>();
    // The arguments are an object that takes no other keys.
    if let Some(object) = schema.as_object_mut() {
        object.insert("type".into(), json!("object"));
        object.insert("additionalProperties".into(), json!(false));
        object.remove("description");
    }
    Tool::new(name, description)
        .with_schema(turbomcp::ToolInputSchema::from_value(schema))
        .with_annotations(hints.annotations())
}

fn parse<A: for<'de> Deserialize<'de>>(args: Value) -> McpResult<A> {
    let args = if args.is_null() { json!({}) } else { args };
    serde_json::from_value(args)
        .map_err(|e| McpError::invalid_params(format!("invalid arguments: {e}")))
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

/// A job to act on.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct JobArgs {
    /// Job ID returned when the job was started.
    job_id: String,
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

async fn device_info(_core: &Core, args: Value) -> McpResult<ToolResult> {
    parse::<NoArgs>(args)?;
    structured(&util::build_device_info_json()?)
}

// ---------------------------------------------------------------------------
// Hub
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Sort {
    Downloads,
    Likes,
    Trending,
    Updated,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    /// Search query, e.g. 'qwen3 0.6B' or 'llama 8b instruct'.
    query: String,
    /// Most results to return (default 10).
    #[schemars(range(min = 1, max = 100))]
    limit: Option<u32>,
    /// Only models from this user or organization.
    author: Option<String>,
    /// Tags every result must have: gguf, mlx, moe, license:apache-2.0, a language.
    tags: Option<Vec<String>>,
    /// Pipeline tag (default text-generation).
    pipeline_tag: Option<String>,
    /// Result order (default downloads).
    sort: Option<Sort>,
}

async fn search_models(_core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: SearchArgs = parse(args)?;
    let mut options = pmetal_hub::SearchOptions::new(&args.query);
    options.limit = args.limit.unwrap_or(10) as usize;
    options.author = args.author;
    options.tags = args.tags.unwrap_or_default();
    if let Some(tag) = args.pipeline_tag {
        options.pipeline_tag = Some(tag);
    }
    options.sort = match args.sort.unwrap_or(Sort::Downloads) {
        Sort::Downloads => pmetal_hub::search::SearchSort::Downloads,
        Sort::Likes => pmetal_hub::search::SearchSort::Likes,
        Sort::Trending => pmetal_hub::search::SearchSort::Trending,
        Sort::Updated => pmetal_hub::search::SearchSort::Updated,
    };
    let device = util::build_device_spec().ok();
    let found = pmetal_hub::search_models_with(&options)
        .await
        .map_err(hub_error)?;
    let results: Vec<Value> = found
        .iter()
        .map(|r| {
            let fit = device
                .as_ref()
                .and_then(|d| r.fit(d, pmetal_hub::fit::DEFAULT_FIT_CONTEXT))
                .map(|fit| {
                    json!({
                        "level": fit.fit_level.label(),
                        "total_gb": fit.total_required_gb,
                        "weights_gb": fit.weights_gb,
                        "weights_measured": fit.weights_measured,
                        "context_length": fit.context_length,
                        "training_gb": fit.training_memory_gb,
                        "training_fit": fit.training_fit.label(),
                        "estimated_tps": fit.estimated_tps,
                    })
                });
            json!({
                "model_id": r.model_id,
                "downloads": r.downloads,
                "likes": r.likes,
                "pipeline_tag": r.pipeline_tag,
                "library_name": r.library_name,
                "license": r.license,
                "gated": r.gated,
                "last_modified": r.last_modified,
                "tags": r.tags,
                "estimated_params_b": r.estimated_params_b,
                "safetensors_bytes": r.safetensors_bytes,
                "fit": fit,
            })
        })
        .collect();
    structured(&json!({ "results": results }))
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Weights {
    Auto,
    Safetensors,
    Gguf,
    Pytorch,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Verify {
    None,
    Size,
    Sha256,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DownloadArgs {
    /// Hugging Face model ID, e.g. 'Qwen/Qwen3-0.6B'.
    model_id: String,
    /// Branch, tag or commit. Name a commit to pin the download; leave out to use
    /// a cached copy when there is one.
    revision: Option<String>,
    /// Which weights to fetch (default auto: safetensors, else GGUF, else PyTorch).
    weights: Option<Weights>,
    /// For GGUF weights, the quantization to fetch, e.g. 'Q4_K_M'.
    gguf_quant: Option<String>,
    /// How to check the files after the transfer (default size).
    verify: Option<Verify>,
    /// Report what would be fetched (commit, files, bytes) without fetching it.
    dry_run: Option<bool>,
}

fn hub_error(e: pmetal_core::PMetalError) -> McpError {
    McpError::internal(util::redact_secrets(&e.to_string()).into_owned())
}

async fn download_model(_core: &Core, args: Value, ctx: &RequestContext) -> McpResult<ToolResult> {
    let args: DownloadArgs = parse(args)?;
    let progress: Arc<std::sync::Mutex<Option<pmetal_hub::TransferProgress>>> =
        Arc::new(std::sync::Mutex::new(None));
    let sink = progress.clone();
    let options = pmetal_hub::DownloadOptions {
        revision: args.revision.clone(),
        weights: match args.weights.unwrap_or(Weights::Auto) {
            Weights::Auto => pmetal_hub::WeightFormat::Auto,
            Weights::Safetensors => pmetal_hub::WeightFormat::Safetensors,
            Weights::Gguf => pmetal_hub::WeightFormat::Gguf,
            Weights::Pytorch => pmetal_hub::WeightFormat::PyTorch,
        },
        gguf_quant: args.gguf_quant.clone(),
        verify: match args.verify.unwrap_or(Verify::Size) {
            Verify::None => pmetal_hub::Verification::None,
            Verify::Size => pmetal_hub::Verification::Size,
            Verify::Sha256 => pmetal_hub::Verification::Sha256,
        },
        progress: Some(Arc::new(move |p: &pmetal_hub::TransferProgress| {
            *sink
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(p.clone());
        })),
        ..pmetal_hub::DownloadOptions::default()
    };

    if args.dry_run.unwrap_or(false) {
        let plan = pmetal_hub::plan_model_download(&args.model_id, &options)
            .await
            .map_err(hub_error)?;
        return structured(&json!({
            "model_id": plan.repo_id,
            "commit": plan.commit,
            "weights": format!("{:?}", plan.model.weights),
            "total_bytes": plan.model.total_bytes(),
            "files": plan.model.files.iter().map(|f| json!({"path": f.path, "size": f.size})).collect::<Vec<_>>(),
        }));
    }

    let download = pmetal_hub::download_model_with(&args.model_id, &options);
    tokio::pin!(download);
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    let path = loop {
        tokio::select! {
            result = &mut download => break result.map_err(hub_error)?,
            _ = tick.tick() => {
                if ctx.is_cancelled() {
                    // Dropping the future stops the transfer; what arrived stays
                    // in the cache's in-progress area.
                    return Err(McpError::cancelled("the download was cancelled"));
                }
                if ctx.wants_progress() {
                    let snapshot = progress
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone();
                    if let Some(p) = snapshot {
                        let message = format!(
                            "{} of {} files, {} of {}{}",
                            p.files_done,
                            p.total_files,
                            pmetal_hub::fit::format_bytes(p.bytes_done),
                            pmetal_hub::fit::format_bytes(p.total_bytes),
                            p.current_file.as_deref().map(|f| format!(", {f}")).unwrap_or_default(),
                        );
                        let _ = ctx
                            .report_progress(p.bytes_done as f64, Some(p.total_bytes as f64), Some(&message))
                            .await;
                    }
                }
            }
        }
    };
    let commit = path.file_name().map(|n| n.to_string_lossy().into_owned());
    structured(&json!({
        "model_id": args.model_id,
        "path": path.display().to_string(),
        "commit": commit,
    }))
}

async fn list_local_models(_core: &Core, args: Value) -> McpResult<ToolResult> {
    parse::<NoArgs>(args)?;
    let models = tokio::task::spawn_blocking(|| {
        let cache = pmetal_hub::cache_dir();
        let mut models = Vec::new();
        let Ok(entries) = std::fs::read_dir(&cache) else {
            return models;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(folder) = name.strip_prefix("models--") else {
                continue;
            };
            let model_id = folder.replace("--", "/");
            let snapshots: Vec<String> = std::fs::read_dir(entry.path().join("snapshots"))
                .map(|d| {
                    d.flatten()
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .filter(|n| !n.starts_with('.'))
                        .collect()
                })
                .unwrap_or_default();
            if snapshots.is_empty() {
                continue;
            }
            let loadable = pmetal_hub::find_cached_model(&model_id);
            models.push(json!({
                "model_id": model_id,
                "path": loadable.as_ref().map(|p| p.display().to_string()),
                "complete": loadable.is_some(),
                "snapshots": snapshots,
                "size_gb": dir_size(&entry.path()) as f64 / (1024.0 * 1024.0 * 1024.0),
            }));
        }
        models.sort_by(|a, b| a["model_id"].as_str().cmp(&b["model_id"].as_str()));
        models
    })
    .await
    .map_err(|e| McpError::internal(format!("listing the cache failed: {e}")))?;
    structured(
        &json!({ "cache_dir": pmetal_hub::cache_dir().display().to_string(), "models": models }),
    )
}

/// Total size of the regular files under `path`. Links are not followed.
fn dir_size(path: &Path) -> u64 {
    let mut size = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(m) = entry.metadata() {
                if m.is_dir() {
                    size += dir_size(&entry.path());
                } else {
                    size += m.len();
                }
            }
        }
    }
    size
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ModelArgs {
    /// Model ID of a cached model, or a local model directory.
    model: String,
}

/// The directory of a model that is on this machine: a local path, or a
/// complete copy in the cache.
fn local_model_dir(policy: &PathPolicy, model: &str) -> McpResult<PathBuf> {
    let reference = policy
        .check_reference("model", model)
        .map_err(McpError::invalid_params)?;
    let path = Path::new(&reference);
    if path.is_absolute() && path.is_dir() {
        return Ok(path.to_path_buf());
    }
    pmetal_hub::find_cached_model(&reference).ok_or_else(|| {
        McpError::invalid_params(format!(
            "{model} is not on this machine. Download it first with download_model"
        ))
    })
}

async fn model_info(policy: &PathPolicy, args: Value) -> McpResult<ToolResult> {
    let args: ModelArgs = parse(args)?;
    let dir = local_model_dir(policy, &args.model)?;
    let config_path = dir.join("config.json");
    let bytes = tokio::fs::read(&config_path).await.map_err(|e| {
        McpError::invalid_params(format!("cannot read {}: {e}", config_path.display()))
    })?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err(McpError::invalid_params("config.json is larger than 4 MiB"));
    }
    let config: Value = serde_json::from_slice(&bytes)
        .map_err(|e| McpError::invalid_params(format!("invalid config.json: {e}")))?;
    structured(&json!({
        "model": args.model,
        "path": dir.display().to_string(),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "rope_theta": config.get("rope_theta"),
        "torch_dtype": config.get("torch_dtype"),
        "num_experts": config.get("num_local_experts").or(config.get("num_experts")),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "full_config": config,
    }))
}

async fn verify_model(policy: &PathPolicy, args: Value) -> McpResult<ToolResult> {
    let args: ModelArgs = parse(args)?;
    let dir = local_model_dir(policy, &args.model)?;
    let report = tokio::task::spawn_blocking({
        let dir = dir.clone();
        move || pmetal_hub::verify_model_dir(&dir)
    })
    .await
    .map_err(|e| McpError::internal(format!("verification failed: {e}")))?;
    structured(&json!({
        "model": args.model,
        "path": dir.display().to_string(),
        "complete": report.is_complete(),
        "loadable": report.is_loadable_language_model(),
        "format": report.format.map(|f| format!("{f:?}")),
        "weight_files": report.weight_files.len(),
        "weight_bytes": report.weight_bytes,
        "problems": report.problems,
        "warnings": report.warnings,
    }))
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct EvictArgs {
    /// Model ID to remove from the cache, e.g. 'Qwen/Qwen3-0.6B'.
    model_id: String,
}

async fn delete_local_model(policy: &PathPolicy, args: Value) -> McpResult<ToolResult> {
    let args: EvictArgs = parse(args)?;
    // Only a Hub ID names something in the cache; a path is not a model to evict.
    let id = policy
        .check_reference("model_id", &args.model_id)
        .map_err(McpError::invalid_params)?;
    if Path::new(&id).is_absolute() || id.matches('/').count() != 1 || id.contains(':') {
        return Err(McpError::invalid_params(
            "model_id must be a Hub ID of the form owner/name",
        ));
    }
    let existed = pmetal_hub::find_cached_model(&id).is_some();
    pmetal_hub::evict_model(&id).map_err(hub_error)?;
    structured(&json!({ "model_id": id, "removed": existed }))
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct FitArgs {
    /// Model ID or local path.
    model: String,
    /// Context length in tokens (default 4096).
    #[schemars(range(min = 1, max = 4194304))]
    context_length: Option<u32>,
    /// Estimate the model at this quantization (fp16, q4_k_m, fp8, ...) instead
    /// of the one it is published in.
    quantization: Option<String>,
}

async fn model_fit(policy: &PathPolicy, args: Value) -> McpResult<ToolResult> {
    let args: FitArgs = parse(args)?;
    let context = args
        .context_length
        .unwrap_or(pmetal_hub::fit::DEFAULT_FIT_CONTEXT);
    let reference = policy
        .check_reference("model", &args.model)
        .map_err(McpError::invalid_params)?;
    let (mut spec, source) = match local_dir_of(&reference) {
        Some(dir) => (
            pmetal_hub::fit::model_spec_from_dir(&dir).ok_or_else(|| {
                McpError::invalid_params(format!("{} has no readable config.json", dir.display()))
            })?,
            "local",
        ),
        None => {
            let meta = pmetal_hub::search::fetch_model_meta(&reference, None)
                .await
                .map_err(hub_error)?;
            (meta.model_spec(context), "hub")
        }
    };
    spec = spec.with_context(context);
    if let Some(q) = &args.quantization {
        // The weights as published say nothing about another format's size.
        spec.quantization = q.clone();
        spec.weight_bytes = None;
    }
    let device = util::build_device_spec()?;
    let fit = pmetal_hub::estimate_fit(&spec, &device);
    structured(&json!({
        "model": args.model,
        "source": source,
        "params_b": spec.params_b,
        "quantization": spec.quantization,
        "context_length": context,
        "inference": {
            "fit_level": fit.fit_level.label(),
            "total_required_gb": fit.total_required_gb,
            "weights_gb": fit.weights_gb,
            "weights_measured": fit.weights_measured,
            "kv_cache_gb": fit.kv_cache_gb,
            "overhead_gb": fit.overhead_gb,
            "available_gb": fit.available_gb,
            "utilization_pct": fit.utilization_pct,
            "estimated_tps": fit.estimated_tps,
        },
        "training": {
            "fit_level": fit.training_fit.label(),
            "memory_gb": fit.training_memory_gb,
            "recommended_batch_size": fit.recommended_batch_size,
        },
        "notes": fit.notes,
    }))
}

/// The directory a reference names or a cached model sits in, if it is here.
fn local_dir_of(reference: &str) -> Option<PathBuf> {
    let path = Path::new(reference);
    if path.is_absolute() && path.is_dir() {
        return Some(path.to_path_buf());
    }
    pmetal_hub::find_cached_model(reference)
}

// ---------------------------------------------------------------------------
// Output directories
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct InspectArgs {
    /// A job's output directory, or a model directory.
    path: String,
}

async fn inspect_output(policy: &PathPolicy, args: Value) -> McpResult<ToolResult> {
    let args: InspectArgs = parse(args)?;
    let root = policy
        .check_path("path", &args.path)
        .map_err(McpError::invalid_params)?;
    let report = tokio::task::spawn_blocking(move || inspect_dir(&root))
        .await
        .map_err(|e| McpError::internal(format!("inspection failed: {e}")))??;
    structured(&report)
}

fn inspect_dir(root: &Path) -> McpResult<Value> {
    const MAX_ENTRIES: usize = 200;
    const MAX_DEPTH: usize = 3;
    if !root.is_dir() {
        return Err(McpError::invalid_params(format!(
            "{} is not a directory",
            root.display()
        )));
    }
    let mut files = Vec::new();
    let mut truncated = false;
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let Ok(meta) = entry.metadata() else { continue };
            let path = entry.path();
            if meta.is_dir() {
                if depth + 1 < MAX_DEPTH {
                    stack.push((path, depth + 1));
                }
            } else if files.len() < MAX_ENTRIES {
                let relative = path.strip_prefix(root).unwrap_or(&path);
                files.push(json!({"path": relative.display().to_string(), "bytes": meta.len()}));
            } else {
                truncated = true;
            }
        }
    }
    let adapter_config = read_json_small(&root.join("adapter_config.json"));
    let last_metrics = last_jsonl_record(&root.join("metrics.jsonl"));
    let has = |name: &str| root.join(name).exists();
    Ok(json!({
        "path": root.display().to_string(),
        "files": files,
        "truncated": truncated,
        "has_adapter": has("adapter_config.json") || has("adapter_model.safetensors") || has("lora_weights.safetensors"),
        "has_model_config": has("config.json"),
        "adapter_config": adapter_config,
        "last_metrics": last_metrics,
    }))
}

fn read_json_small(path: &Path) -> Option<Value> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > 1024 * 1024 {
        return None;
    }
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// The last step record of a JSONL metrics file, reading only its end.
fn last_jsonl_record(path: &Path) -> Option<Value> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    let window = len.min(64 * 1024);
    file.seek(SeekFrom::Start(len - window)).ok()?;
    let mut buf = Vec::new();
    file.take(window).read_to_end(&mut buf).ok()?;
    String::from_utf8_lossy(&buf)
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|v| v.get("step").is_some())
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, JsonSchema)]
struct JobList {
    jobs: Vec<JobSummary>,
}

async fn list_jobs(core: &Core, args: Value) -> McpResult<ToolResult> {
    parse::<NoArgs>(args)?;
    structured(&JobList {
        jobs: core.jobs.list_summaries(),
    })
}

async fn job_status(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: JobArgs = parse(args)?;
    structured(&core.jobs.get_summary(&args.job_id)?)
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct LogsArgs {
    /// Job ID returned when the job was started.
    job_id: String,
    /// Most recent lines to return (default 50).
    #[schemars(range(min = 1, max = 10000))]
    tail: Option<u32>,
    /// Only lines after this position: pass the `next_since` of an earlier call
    /// to read what a job has printed since.
    since: Option<u64>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct Logs {
    job_id: String,
    #[serde(flatten)]
    page: LogPage,
}

async fn job_logs(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: LogsArgs = parse(args)?;
    let page = core
        .jobs
        .get_logs(&args.job_id, args.tail.unwrap_or(50) as usize, args.since)?;
    structured(&Logs {
        job_id: args.job_id,
        page,
    })
}

async fn stop_job(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: JobArgs = parse(args)?;
    core.jobs.stop(&args.job_id).await?;
    structured(&json!({
        "job_id": args.job_id,
        "status": "stopping",
        "message": "SIGTERM sent to the job's process group; SIGKILL follows if it has not exited after the grace period.",
    }))
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetLrArgs {
    /// Job ID of the running training job.
    job_id: String,
    /// New learning rate, e.g. 1e-5.
    #[schemars(range(min = 1e-12, max = 1.0))]
    lr: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReduceLrArgs {
    /// Job ID of the running training job.
    job_id: String,
    /// Factor to multiply the learning rate by, e.g. 0.5 to halve it.
    #[schemars(range(min = 0.01, max = 1.0))]
    factor: f64,
}

/// What a training job reads from its control file.
fn send_control(core: &Core, job_id: &str, command: &LrControlCommand) -> McpResult<ToolResult> {
    let (subcommand, dir) = core.jobs.running_output(job_id)?;
    if !CONTROLLABLE.contains(&subcommand.as_str()) {
        return Err(McpError::invalid_request(format!(
            "`{subcommand}` jobs do not read learning-rate control; \
             it works on {}",
            CONTROLLABLE.join(", ")
        )));
    }
    pmetal_core::write_lr_control(&dir, command).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            McpError::unavailable(
                "the job has not read the previous control command yet; try again in a few seconds",
            )
        } else {
            McpError::internal(format!(
                "failed to write the control file in {}: {e}",
                dir.display()
            ))
        }
    })?;
    structured(&json!({
        "job_id": job_id,
        "requested": command,
        "message": "Takes effect within about 10 training steps.",
    }))
}

/// The subcommands whose training loop polls the control file.
const CONTROLLABLE: &[&str] = &["train", "distill", "grpo", "rlkd"];

async fn job_set_lr(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: SetLrArgs = parse(args)?;
    send_control(
        core,
        &args.job_id,
        &LrControlCommand::SetLr { value: args.lr },
    )
}

async fn job_reduce_lr(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: ReduceLrArgs = parse(args)?;
    send_control(
        core,
        &args.job_id,
        &LrControlCommand::ReduceLr {
            factor: args.factor,
        },
    )
}

async fn job_reset_lr(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: JobArgs = parse(args)?;
    send_control(core, &args.job_id, &LrControlCommand::ResetLr)
}

async fn job_save_checkpoint(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: JobArgs = parse(args)?;
    send_control(core, &args.job_id, &LrControlCommand::SaveCheckpoint)
}

async fn job_graceful_stop(core: &Core, args: Value) -> McpResult<ToolResult> {
    let args: JobArgs = parse(args)?;
    send_control(core, &args.job_id, &LrControlCommand::GracefulStop)
}

// ---------------------------------------------------------------------------
// Talking to a running server
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ChatArgs {
    /// User message.
    message: String,
    /// Port of the `pmetal serve` instance on this machine (default 8080).
    #[schemars(range(min = 1024, max = 65535))]
    port: Option<u32>,
    /// System prompt.
    system: Option<String>,
    /// Sampling temperature (the server's default when left out).
    #[schemars(range(min = 0.0, max = 2.0))]
    temperature: Option<f64>,
    /// Most tokens to generate (the server's default when left out).
    #[schemars(range(min = 1))]
    max_tokens: Option<u32>,
}

async fn chat(_core: &Core, args: Value, ctx: &RequestContext) -> McpResult<ToolResult> {
    let args: ChatArgs = parse(args)?;
    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": args.message}));
    let mut body = json!({ "messages": messages });
    if let Some(t) = args.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(m) = args.max_tokens {
        body["max_tokens"] = json!(m);
    }
    let reply = serve_client::post_json(
        args.port.unwrap_or(8080),
        "/v1/chat/completions",
        &body,
        ctx,
    )
    .await?;
    let result: Value = serde_json::from_str(&reply)
        .map_err(|e| McpError::internal(format!("invalid response from serve: {e}")))?;
    let choice = result.pointer("/choices/0");
    let content = choice
        .and_then(|c| c.pointer("/message/content"))
        .and_then(Value::as_str);
    let Some(content) = content else {
        return structured(&json!({ "response": result }));
    };
    let details = json!({
        "content": content,
        "finish_reason": choice.and_then(|c| c.get("finish_reason")),
        "model": result.get("model"),
        "usage": result.get("usage"),
    });
    Ok(results::text(content).with_structured(&details))
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DecideArgs {
    /// Decision model ID or path (e.g. Cloudflare/clef-flash), loaded for this
    /// call. Leave out when port is given.
    model: Option<String>,
    /// Request body as a JSON string: {"model", "state", "questions": {id:
    /// {"type": "noul"|"choice"|"score", "instructions", "criteria"}}}.
    request: String,
    /// Longest prompt in tokens (default 16384). For a server, set it when
    /// starting the server.
    #[schemars(range(min = 1, max = 1048576))]
    max_length: Option<u32>,
    /// Port of a running decision-model server (start_serve) to ask instead of
    /// loading the model. Its images must be base64, not file paths.
    #[schemars(range(min = 1024, max = 65535))]
    port: Option<u32>,
}

async fn decide(
    core: &Core,
    policy: &PathPolicy,
    args: Value,
    ctx: &RequestContext,
) -> McpResult<ToolResult> {
    let args: DecideArgs = parse(args)?;
    let body = serde_json::from_str::<Value>(&args.request)
        .map_err(|e| McpError::invalid_params(format!("request is not JSON: {e}")))?;
    match (args.model, args.port) {
        (Some(_), Some(_)) => Err(McpError::invalid_params(
            "give model or port, not both: a server answers with the model it serves",
        )),
        (None, None) => Err(McpError::invalid_params(
            "give model to load it for this call, or port to ask a running server",
        )),
        (None, Some(port)) => {
            if args.max_length.is_some() {
                return Err(McpError::invalid_params(
                    "max_length is the server's: set it when starting the server",
                ));
            }
            let reply = serve_client::post_json(port, "/v1/systemone", &body, ctx).await?;
            Ok(results::text(reply))
        }
        (Some(model), None) => {
            let model = policy
                .check_reference("model", &model)
                .map_err(McpError::invalid_params)?;
            let staged = crate::staging::stage_private_file("decide", args.request.as_bytes())
                .map_err(|e| McpError::internal(format!("cannot stage the request: {e}")))?;
            let mut spec = DecideSpec {
                model,
                request: staged.path().display().to_string(),
                max_length: args
                    .max_length
                    .map_or(DecideSpec::default().max_length, |n| n as usize),
                compact: true,
            };
            let argv = {
                use pmetal_core::JobFields as _;
                spec.normalize().map_err(|errors| {
                    McpError::invalid_params(format!(
                        "validation failed: {}",
                        errors
                            .iter()
                            .map(|e| format!("{}: {}", e.field, e.message))
                            .collect::<Vec<_>>()
                            .join("; ")
                    ))
                })?;
                let mut argv = vec![DecideSpec::subcommand().to_string()];
                argv.extend(spec.to_argv());
                argv
            };
            let output = core.run(&argv, ctx).await;
            drop(staged);
            output.map(results::text)
        }
    }
}

// ---------------------------------------------------------------------------
// The registry
// ---------------------------------------------------------------------------

/// Every native tool's definition.
pub fn definitions() -> Vec<Tool> {
    let mut tools = vec![
        tool_for::<NoArgs>(
            "device_info",
            "Get Apple Silicon device information: GPU architecture, ANE cores, memory bandwidth, NAX support and unified memory capacity.",
            Hints::READ,
        ),
        tool_for::<SearchArgs>(
            "search_models",
            "Search Hugging Face Hub for models. Returns IDs, sizes, downloads and whether each fits this device's memory for inference and training.",
            Hints::READ.open_world(),
        ),
        tool_for::<DownloadArgs>(
            "download_model",
            "Download a model from Hugging Face Hub to the local cache and return where it is. Verifies the files; pin a commit with revision. With dry_run, reports what would be fetched. May take several minutes for large models.",
            Hints::CREATE.idempotent(true).open_world(),
        ),
        tool_for::<NoArgs>(
            "list_local_models",
            "List models in the local Hugging Face cache, with whether each is complete and how large it is.",
            Hints::READ,
        ),
        tool_for::<ModelArgs>(
            "model_info",
            "Read a local or cached model's config.json: architecture, hidden size, layers, heads, vocabulary, experts.",
            Hints::READ,
        ),
        tool_for::<ModelArgs>(
            "verify_model",
            "Check that a local or cached model is complete: every weight file present and as long as its header says.",
            Hints::READ,
        ),
        tool_for::<EvictArgs>(
            "delete_local_model",
            "Remove a model from the local Hugging Face cache.",
            Hints::READ.destructive(),
        ),
        tool_for::<FitArgs>(
            "model_fit",
            "Estimate the memory a model needs on this device and how fast it would run: inference and training memory, KV cache, tokens per second, fit level. Optionally at another quantization.",
            Hints::READ.open_world(),
        ),
        tool_for::<InspectArgs>(
            "inspect_output",
            "Inspect a training job's output directory or a model directory: files and sizes, adapter configuration, the last recorded training step.",
            Hints::READ,
        ),
        tool_for::<NoArgs>(
            "list_jobs",
            "List background jobs (training, distillation, GRPO, serving, ...) with their status and metrics, newest first.",
            Hints::READ,
        ),
        tool_for::<JobArgs>(
            "job_status",
            "Get a background job's status, exit code, timing and training metrics (step, loss, learning rate, tokens per second).",
            Hints::READ,
        ),
        tool_for::<LogsArgs>(
            "job_logs",
            "Get a background job's recent output, stdout and stderr in the order printed. Pass since to read only what is new.",
            Hints::READ,
        ),
        tool_for::<JobArgs>(
            "stop_job",
            "Stop a running background job: SIGTERM to its process group, SIGKILL if it does not exit.",
            Hints::READ.destructive().idempotent(true),
        ),
        tool_for::<SetLrArgs>(
            "job_set_lr",
            "Set the learning rate of a running training job (train, distill, grpo, rlkd). Takes effect within about 10 steps.",
            Hints::CREATE.idempotent(true),
        ),
        tool_for::<ReduceLrArgs>(
            "job_reduce_lr",
            "Multiply the learning rate of a running training job by a factor, e.g. 0.5 to halve it.",
            Hints::CREATE,
        ),
        tool_for::<JobArgs>(
            "job_reset_lr",
            "Return a running training job's learning rate to its schedule, clearing adaptive and manual adjustments.",
            Hints::CREATE.idempotent(true),
        ),
        tool_for::<JobArgs>(
            "job_save_checkpoint",
            "Save a checkpoint of a running training job now. Training continues.",
            Hints::CREATE,
        ),
        tool_for::<JobArgs>(
            "job_graceful_stop",
            "Stop a running training job cleanly: restore the best weights, save a final checkpoint, then exit.",
            Hints::CREATE.idempotent(true),
        ),
        tool_for::<ChatArgs>(
            "chat",
            "Send one chat message to a `pmetal serve` instance on this machine (start one with start_serve) and return the reply.",
            Hints::READ.idempotent(false),
        ),
        tool_for::<DecideArgs>(
            "decide",
            "Answer typed questions about a state with a decision model (Clef, Clef-flash): takes a /v1/systemone request body and returns one probability per allowed option of every question. With model, loads the model for this call; with port, asks a decision-model server already running (start_serve), which answers many requests without reloading.",
            Hints::READ.open_world(),
        ),
    ];
    // What the job tools return is declared, so a client can rely on its shape.
    for tool in &mut tools {
        let output = match tool.name.as_str() {
            "job_status" => schema_of::<JobSummary>(),
            "list_jobs" => schema_of::<JobList>(),
            "job_logs" => schema_of::<Logs>(),
            _ => continue,
        };
        tool.output_schema = Some(turbomcp::ToolOutputSchema::from_value(output));
    }
    tools
}

/// Whether the native tool `name` takes a path or a model reference, which the
/// server checks against the roots the client lists as well as its own.
pub fn takes_paths(name: &str) -> bool {
    matches!(
        name,
        "model_info"
            | "verify_model"
            | "delete_local_model"
            | "model_fit"
            | "inspect_output"
            | "decide"
    )
}

/// Run the native tool `name`, or `None` if there is none.
pub async fn call(
    core: &Core,
    policy: &PathPolicy,
    name: &str,
    args: Value,
    ctx: &RequestContext,
) -> Option<McpResult<ToolResult>> {
    Some(match name {
        "device_info" => device_info(core, args).await,
        "search_models" => search_models(core, args).await,
        "download_model" => download_model(core, args, ctx).await,
        "list_local_models" => list_local_models(core, args).await,
        "model_info" => model_info(policy, args).await,
        "verify_model" => verify_model(policy, args).await,
        "delete_local_model" => delete_local_model(policy, args).await,
        "model_fit" => model_fit(policy, args).await,
        "inspect_output" => inspect_output(policy, args).await,
        "list_jobs" => list_jobs(core, args).await,
        "job_status" => job_status(core, args).await,
        "job_logs" => job_logs(core, args).await,
        "stop_job" => stop_job(core, args).await,
        "job_set_lr" => job_set_lr(core, args).await,
        "job_reduce_lr" => job_reduce_lr(core, args).await,
        "job_reset_lr" => job_reset_lr(core, args).await,
        "job_save_checkpoint" => job_save_checkpoint(core, args).await,
        "job_graceful_stop" => job_graceful_stop(core, args).await,
        "chat" => chat(core, args, ctx).await,
        "decide" => decide(core, policy, args, ctx).await,
        _ => return None,
    })
}
