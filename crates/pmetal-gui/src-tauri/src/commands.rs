//! The commands the webview can call.
//!
//! Each is a thin wrapper over [`AppState`] or a module; the rules live there.
//! Every command is listed in `build.rs`, which is what lets the capability
//! file name them one by one: a command not listed there cannot be called at
//! all.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_dialog::DialogExt;

use crate::adapters::{self, TrainedAdapter};
use crate::datasets::{self, DatasetPeek};
use crate::error::{AppError, Result};
use crate::hub::{self, CachedDatasetInfo, DatasetSearchResult, DownloadView, HubSearchResult};
use crate::inference::{self, ChatTurn};
use crate::jobs::form::FormInfo;
use crate::jobs::kinds::{self, FieldIssue};
use crate::jobs::{EventSink, JobView, OutputLine, OutputPage, StartError, StepPage};
use crate::models::{self, CachedModel};
use crate::state::{AppState, ConfigView, SettingsPatch};
use crate::system::{self, ModelFitInfo, SystemInfo};

// ---------------------------------------------------------------------------
// System and settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_system_info() -> Result<SystemInfo> {
    tokio::task::spawn_blocking(system::system_info)
        .await
        .map_err(|e| AppError::new(e.to_string()))
}

#[tauri::command]
pub async fn get_config(state: State<'_, AppState>) -> Result<ConfigView> {
    Ok(state.config().await)
}

#[tauri::command]
pub async fn set_config(state: State<'_, AppState>, patch: SettingsPatch) -> Result<ConfigView> {
    state.update_settings(patch).await
}

#[tauri::command]
pub async fn set_hf_token(state: State<'_, AppState>, token: String) -> Result<ConfigView> {
    state.set_token(token).await
}

#[tauri::command]
pub async fn clear_hf_token(state: State<'_, AppState>) -> Result<ConfigView> {
    state.clear_token().await
}

/// Let the person choose a folder. The webview gets the path and no
/// filesystem access of its own.
#[tauri::command]
pub async fn pick_directory(app: AppHandle, title: Option<String>) -> Result<Option<String>> {
    let mut dialog = app.dialog().file();
    if let Some(title) = title {
        dialog = dialog.set_title(title);
    }
    let picked = tokio::task::spawn_blocking(move || dialog.blocking_pick_folder())
        .await
        .map_err(|e| AppError::new(e.to_string()))?;
    Ok(picked
        .and_then(|p| p.into_path().ok())
        .map(|p| p.display().to_string()))
}

/// Let the person choose a file.
#[tauri::command]
pub async fn pick_file(app: AppHandle, title: Option<String>) -> Result<Option<String>> {
    let mut dialog = app.dialog().file();
    if let Some(title) = title {
        dialog = dialog.set_title(title);
    }
    let picked = tokio::task::spawn_blocking(move || dialog.blocking_pick_file())
        .await
        .map_err(|e| AppError::new(e.to_string()))?;
    Ok(picked
        .and_then(|p| p.into_path().ok())
        .map(|p| p.display().to_string()))
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> Result<Vec<CachedModel>> {
    let models = state.models().await;
    if models.is_empty() {
        // Nothing yet: the first scan has not finished, or there are no models.
        return Ok(state.refresh_models().await);
    }
    Ok(models)
}

#[tauri::command]
pub async fn refresh_models(state: State<'_, AppState>) -> Result<Vec<CachedModel>> {
    Ok(state.refresh_models().await)
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub path: String,
    pub size: u64,
    pub size_formatted: String,
    pub model_type: Option<String>,
    pub hidden_size: Option<u64>,
    pub num_layers: Option<u64>,
    pub vocab_size: Option<u64>,
    pub context_length: Option<u64>,
}

#[tauri::command]
pub async fn get_model_info(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<Option<ModelInfo>> {
    let Some(model) = state.find_model(&model_id).await else {
        return Ok(None);
    };
    let path = std::path::PathBuf::from(&model.path);
    let config = tokio::task::spawn_blocking(move || models::read_config(&path))
        .await
        .map_err(|e| AppError::new(e.to_string()))?;
    let get = |keys: &[&str]| {
        config
            .as_ref()
            .and_then(|c| keys.iter().find_map(|k| c[*k].as_u64()))
    };
    Ok(Some(ModelInfo {
        model_type: config
            .as_ref()
            .and_then(|c| c["model_type"].as_str().map(str::to_string))
            .or(Some(model.model_type.clone())),
        hidden_size: get(&["hidden_size"]),
        num_layers: get(&["num_hidden_layers", "n_layer"]),
        vocab_size: get(&["vocab_size"]),
        context_length: get(&["max_position_embeddings", "max_seq_len", "n_ctx"]),
        id: model.id,
        path: model.path,
        size: model.size,
        size_formatted: model.size_formatted,
    }))
}

/// What a model recommends: sampling from its `generation_config.json`, and
/// its shape from `config.json`.
#[derive(Debug, Default, Serialize)]
pub struct ModelDefaults {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub max_new_tokens: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub max_position_embeddings: Option<u32>,
    pub hidden_size: Option<u32>,
    pub num_hidden_layers: Option<u32>,
    pub vocab_size: Option<u32>,
}

#[tauri::command]
pub async fn get_model_defaults(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<ModelDefaults> {
    let Some(model) = state.find_model(&model_id).await else {
        return Ok(ModelDefaults::default());
    };
    let path = std::path::PathBuf::from(&model.path);
    tokio::task::spawn_blocking(move || model_defaults(&path, &model.id))
        .await
        .map_err(|e| AppError::new(e.to_string()))
}

fn model_defaults(path: &std::path::Path, id: &str) -> ModelDefaults {
    use pmetal::data::chat_templates::{ChatTemplateKwargs, detect_chat_template};
    use pmetal::data::inference_config::{context_window, default_max_tokens};

    let mut defaults = ModelDefaults::default();
    if let Some(generation) = std::fs::read_to_string(path.join("generation_config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    {
        defaults.temperature = generation["temperature"].as_f64().map(|v| v as f32);
        defaults.top_k = generation["top_k"].as_u64().map(|v| v as u32);
        defaults.top_p = generation["top_p"].as_f64().map(|v| v as f32);
        defaults.repetition_penalty = generation["repetition_penalty"].as_f64().map(|v| v as f32);
    }
    // The output budget `pmetal infer` would pick.
    let thinking = detect_chat_template(path, id).thinks_with(&ChatTemplateKwargs::new());
    let budget = default_max_tokens(path, thinking).for_prompt(0, context_window(path));
    defaults.max_new_tokens = Some(u32::try_from(budget).unwrap_or(u32::MAX));
    if let Some(config) = models::read_config(path) {
        let get = |k: &str| config[k].as_u64().map(|v| v as u32);
        defaults.max_position_embeddings =
            get("max_position_embeddings").or_else(|| get("max_seq_len"));
        defaults.hidden_size = get("hidden_size");
        defaults.num_hidden_layers = get("num_hidden_layers");
        defaults.vocab_size = get("vocab_size");
    }
    defaults
}

#[tauri::command]
pub async fn delete_model(state: State<'_, AppState>, model_id: String) -> Result<()> {
    state.delete_model(&model_id).await
}

#[tauri::command]
pub async fn add_model_directory(
    state: State<'_, AppState>,
    path: String,
) -> Result<Vec<CachedModel>> {
    state.add_model_directory(&path).await
}

#[tauri::command]
pub async fn remove_model_directory(
    state: State<'_, AppState>,
    path: String,
) -> Result<Vec<CachedModel>> {
    state.remove_model_directory(&path).await
}

#[tauri::command]
pub async fn search_hub_models(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<HubSearchResult>> {
    let token = state.token_for_hub().await;
    let device = tokio::task::spawn_blocking(system::device_spec)
        .await
        .ok()
        .flatten();
    hub::search_models(
        &query,
        limit.unwrap_or(20) as usize,
        token.as_ref(),
        device.as_ref(),
    )
    .await
}

#[tauri::command]
pub async fn get_trending_models(
    state: State<'_, AppState>,
    limit: Option<u32>,
) -> Result<Vec<HubSearchResult>> {
    search_hub_models(state, String::new(), limit).await
}

/// How a model would run on this Mac. A model on disk is measured; one that is
/// only on the Hub is estimated from what the Hub says about it.
#[tauri::command]
pub async fn get_model_fit(state: State<'_, AppState>, model_id: String) -> Result<ModelFitInfo> {
    let device = tokio::task::spawn_blocking(system::device_spec)
        .await
        .ok()
        .flatten()
        .ok_or_else(|| {
            AppError::new("This Mac's GPU could not be queried, so a fit cannot be estimated.")
        })?;
    if let Some(model) = state.find_model(&model_id).await {
        let path = std::path::PathBuf::from(&model.path);
        let spec = tokio::task::spawn_blocking(move || pmetal::hub::model_spec_from_dir(&path))
            .await
            .map_err(|e| AppError::new(e.to_string()))?;
        if let Some(spec) = spec {
            return Ok(system::fit(&spec, &device));
        }
    }
    let token = state.token_for_hub().await;
    let meta = pmetal::hub::fetch_model_meta(&model_id, token.as_ref())
        .await
        .map_err(|e| AppError::new(format!("Could not look up {model_id} on the Hub: {e}")))?;
    Ok(system::fit(
        &meta.model_spec(pmetal::hub::DEFAULT_FIT_CONTEXT),
        &device,
    ))
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn download_model(
    state: State<'_, AppState>,
    model_id: String,
    revision: Option<String>,
) -> Result<DownloadView> {
    let model_id = model_id.trim().to_string();
    if !pmetal::hub::is_hf_id(&model_id) {
        return Err(AppError::new(format!(
            "'{model_id}' is not a Hugging Face model id. Use the form owner/name, \
             for example Qwen/Qwen3-0.6B."
        )));
    }
    let token = state.token_for_hub().await;
    let id = model_id.clone();
    state
        .downloads
        .start("model", &model_id, move |progress| async move {
            let mut options =
                pmetal::hub::DownloadOptions::new(revision.as_deref(), token.as_ref());
            options.progress = Some(progress);
            pmetal::hub::download_model_with(&id, &options)
                .await
                .map_err(|e| download_error(&id, &e.to_string()))
        })
}

#[tauri::command]
pub async fn download_dataset(
    state: State<'_, AppState>,
    dataset_id: String,
) -> Result<DownloadView> {
    let dataset_id = dataset_id.trim().to_string();
    if !pmetal::hub::is_hf_id(&dataset_id) {
        return Err(AppError::new(format!(
            "'{dataset_id}' is not a Hugging Face dataset id. Use the form owner/name."
        )));
    }
    let token = state.token_for_hub().await;
    let id = dataset_id.clone();
    state
        .downloads
        .start("dataset", &dataset_id, move |progress| async move {
            let mut options = pmetal::hub::DownloadOptions::new(None, token.as_ref());
            options.progress = Some(progress);
            pmetal::hub::download_dataset_with(&id, None, &options)
                .await
                .map_err(|e| download_error(&id, &e.to_string()))
        })
}

fn download_error(id: &str, detail: &str) -> String {
    let lower = detail.to_lowercase();
    let hint = if lower.contains("401")
        || lower.contains("403")
        || lower.contains("token")
        || lower.contains("gated")
    {
        " If it is gated or private, accept its terms on huggingface.co and add an access token in Settings."
    } else if lower.contains("404") || lower.contains("not found") {
        " Check the spelling of the name."
    } else {
        ""
    };
    format!("Could not download {id}: {detail}.{hint}")
}

#[tauri::command]
pub async fn list_downloads(state: State<'_, AppState>) -> Result<Vec<DownloadView>> {
    Ok(state.downloads.list())
}

#[tauri::command]
pub async fn cancel_download(state: State<'_, AppState>, download_id: String) -> Result<()> {
    state.downloads.cancel(&download_id)
}

#[tauri::command]
pub async fn clear_finished_downloads(state: State<'_, AppState>) -> Result<()> {
    state.downloads.clear_finished();
    Ok(())
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn search_hub_datasets(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<DatasetSearchResult>> {
    let token = state.token_for_hub().await;
    hub::search_datasets(&query, limit.unwrap_or(20) as usize, token.as_ref()).await
}

#[tauri::command]
pub async fn get_trending_datasets(
    state: State<'_, AppState>,
    limit: Option<u32>,
) -> Result<Vec<DatasetSearchResult>> {
    search_hub_datasets(state, String::new(), limit).await
}

#[tauri::command]
pub async fn list_cached_datasets(state: State<'_, AppState>) -> Result<Vec<CachedDatasetInfo>> {
    let cache = state.paths.hub_cache.clone();
    tokio::task::spawn_blocking(move || hub::cached_datasets(&cache))
        .await
        .map_err(|e| AppError::new(e.to_string()))
}

#[tauri::command]
pub async fn peek_dataset_columns(path: String, limit: Option<usize>) -> Result<DatasetPeek> {
    tokio::task::spawn_blocking(move || datasets::peek(&path, limit))
        .await
        .map_err(|e| AppError::new(e.to_string()))?
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// Why a job did not start.
#[derive(Debug, Serialize)]
pub struct StartFailure {
    pub message: String,
    /// The fields of the form that are wrong, each with what is wrong.
    pub issues: Vec<FieldIssue>,
}

impl From<StartError> for StartFailure {
    fn from(e: StartError) -> Self {
        match e {
            StartError::Message(m) => Self {
                message: m.message().to_string(),
                issues: Vec::new(),
            },
            StartError::Invalid(issues) => Self {
                message: issues
                    .iter()
                    .take(3)
                    .map(|i| {
                        if i.field.is_empty() {
                            i.message.clone()
                        } else {
                            format!("{}: {}", i.field.replace('_', " "), i.message)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
                issues,
            },
        }
    }
}

/// The form of every job the app can start.
#[tauri::command]
pub async fn list_job_forms() -> Result<Vec<FormInfo>> {
    Ok(kinds::all().iter().map(|k| k.form()).collect())
}

#[tauri::command]
pub async fn start_job(
    state: State<'_, AppState>,
    kind: String,
    values: BTreeMap<String, String>,
) -> std::result::Result<JobView, StartFailure> {
    let token = state.token_for_hub().await;
    state
        .jobs
        .start(&kind, &values, token.as_ref())
        .map_err(StartFailure::from)
}

#[tauri::command]
pub async fn stop_job(state: State<'_, AppState>, job_id: String) -> Result<()> {
    state.jobs.stop(&job_id)
}

#[tauri::command]
pub async fn forget_job(state: State<'_, AppState>, job_id: String) -> Result<()> {
    state.jobs.forget(&job_id)
}

#[tauri::command]
pub async fn list_jobs(state: State<'_, AppState>) -> Result<Vec<JobView>> {
    Ok(state.jobs.list())
}

#[tauri::command]
pub async fn job_output(
    state: State<'_, AppState>,
    job_id: String,
    tail: Option<usize>,
    since: Option<u64>,
) -> Result<OutputPage> {
    state
        .jobs
        .output(&job_id, tail.unwrap_or(500).min(5000), since)
}

#[tauri::command]
pub async fn job_steps(
    state: State<'_, AppState>,
    job_id: String,
    since: Option<u64>,
) -> Result<StepPage> {
    state.jobs.steps(&job_id, since.unwrap_or(0))
}

#[tauri::command]
pub async fn list_trained_adapters(state: State<'_, AppState>) -> Result<Vec<TrainedAdapter>> {
    let root = state.paths.output_root.clone();
    tokio::task::spawn_blocking(move || adapters::scan(&root))
        .await
        .map_err(|e| AppError::new(e.to_string()))
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

struct ChatEvents(AppHandle);

impl inference::Sink for ChatEvents {
    fn token(&self, session_id: &str, piece: inference::Piece<'_>) {
        let _ = self.0.emit(
            "chat-token",
            serde_json::json!({
                "session_id": session_id,
                "text": piece.text,
                "thinking": piece.thinking,
            }),
        );
    }

    fn done(&self, metrics: &inference::Metrics) {
        let _ = self.0.emit("chat-done", metrics);
    }

    fn failed(&self, session_id: &str, message: &str) {
        let _ = self.0.emit(
            "chat-error",
            serde_json::json!({ "session_id": session_id, "error": message }),
        );
    }
}

/// Start a reply. Its events carry `session_id`, which the page chose.
#[tauri::command]
pub async fn start_chat(
    state: State<'_, AppState>,
    app: AppHandle,
    spec: pmetal::core::jobs::InferSpec,
    messages: Option<Vec<ChatTurn>>,
    session_id: String,
) -> Result<()> {
    let token = state.token_for_hub().await;
    state
        .chat
        .start(
            spec,
            messages.unwrap_or_default(),
            token,
            Arc::new(ChatEvents(app)),
            session_id,
        )
        .await
}

#[tauri::command]
pub async fn stop_chat(state: State<'_, AppState>, session_id: Option<String>) -> Result<()> {
    state.chat.cancel(session_id.as_deref());
    Ok(())
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Sends job updates to the webview as Tauri events.
pub struct TauriJobEvents(pub AppHandle);

impl EventSink for TauriJobEvents {
    fn update(&self, job: &JobView) {
        let _ = self.0.emit("job-update", job);
    }

    fn output(&self, id: &str, lines: &[OutputLine]) {
        let _ = self.0.emit(
            "job-output",
            serde_json::json!({ "job_id": id, "lines": lines }),
        );
    }

    fn resync(&self) {
        let _ = self.0.emit("jobs-resync", ());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_start_names_the_fields_and_keeps_them_for_the_form() {
        let failure = StartFailure::from(StartError::Invalid(vec![
            FieldIssue::new("learning_rate", "must be at most 1"),
            FieldIssue::new("", "the job could not be built"),
        ]));
        assert_eq!(
            failure.message,
            "learning rate: must be at most 1; the job could not be built"
        );
        assert_eq!(failure.issues.len(), 2);
        let json = serde_json::to_value(&failure).unwrap();
        assert_eq!(json["issues"][0]["field"], "learning_rate");
    }

    #[test]
    fn a_refused_start_without_fields_is_just_a_message() {
        let failure = StartFailure::from(StartError::Message(AppError::new(
            "3 jobs are already running.",
        )));
        assert_eq!(failure.message, "3 jobs are already running.");
        assert!(failure.issues.is_empty());
    }

    #[test]
    fn a_failed_download_says_what_to_try() {
        assert!(download_error("a/b", "HTTP 401").contains("Settings"));
        assert!(download_error("a/b", "repository not found (404)").contains("spelling"));
        let plain = download_error("a/b", "disk full");
        assert!(plain.starts_with("Could not download a/b: disk full."));
    }

    #[test]
    fn a_model_recommends_what_its_own_files_say() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.05}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"max_position_embeddings":8192,"hidden_size":64,"num_hidden_layers":2,"vocab_size":100}"#,
        )
        .unwrap();
        let d = model_defaults(dir.path(), "acme/tiny");
        assert_eq!(d.temperature, Some(0.6));
        assert_eq!((d.top_k, d.top_p), (Some(20), Some(0.95)));
        assert_eq!(d.repetition_penalty, Some(1.05));
        assert_eq!(d.max_position_embeddings, Some(8192));
        assert_eq!(
            (d.hidden_size, d.num_hidden_layers, d.vocab_size),
            (Some(64), Some(2), Some(100))
        );
        assert!(d.max_new_tokens.is_some_and(|n| n > 0));
    }

    #[test]
    fn a_model_with_no_files_recommends_nothing_of_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let d = model_defaults(dir.path(), "acme/tiny");
        assert_eq!((d.temperature, d.top_k, d.hidden_size), (None, None, None));
    }
}
