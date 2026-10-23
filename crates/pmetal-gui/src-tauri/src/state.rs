//! What the app holds while it runs, and the operations on it.
//!
//! The Tauri commands are thin wrappers over these methods, so everything
//! with a rule in it (what may be deleted, what a path must be, where the
//! token goes) can be tested without a window.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pmetal::core::SecretString;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::{AppError, Result};
use crate::hub::Downloads;
use crate::inference::Sessions;
use crate::jobs::JobHub;
use crate::models::{CachedModel, ModelSource, ScanPlan};
use crate::secrets::HfToken;
use crate::settings::{self, Settings, Theme};

/// Where the app keeps and finds things.
#[derive(Debug, Clone)]
pub struct AppPaths {
    /// The settings file.
    pub settings: PathBuf,
    /// Where generated job outputs go.
    pub output_root: PathBuf,
    /// The Hub cache, as `pmetal-hub` resolves it.
    pub hub_cache: PathBuf,
}

impl AppPaths {
    pub fn detect() -> Self {
        Self {
            settings: settings::default_path(),
            output_root: crate::paths::output_root(),
            hub_cache: pmetal::hub::cache_dir(),
        }
    }
}

/// The settings, as the webview sees them. The token is not in it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ConfigView {
    pub default_model: Option<String>,
    pub theme: Theme,
    pub custom_model_dirs: Vec<String>,
    /// The Hub cache in use. Set by `HF_HOME` or `HF_HUB_CACHE`, not here.
    pub hf_cache_dir: String,
    /// Where jobs write what they produce.
    pub output_dir: String,
    /// Whether a Hugging Face token is saved in the keychain.
    pub hf_token_set: bool,
    /// Things worth telling the person about the app's own state.
    pub notes: Vec<String>,
    /// The `pmetal` command-line tool jobs run, or why there is none.
    pub pmetal_binary: Option<String>,
    pub pmetal_binary_error: Option<String>,
}

/// A change to the settings; what is left out stays as it was.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct SettingsPatch {
    /// An empty string clears it.
    pub default_model: Option<String>,
    pub theme: Option<Theme>,
}

pub struct AppState {
    pub paths: AppPaths,
    settings: RwLock<Settings>,
    pub token: HfToken,
    pub jobs: JobHub,
    models: RwLock<Vec<CachedModel>>,
    pub downloads: Downloads,
    pub sessions: Arc<Sessions>,
    notes: Mutex<Vec<String>>,
    binary: std::result::Result<PathBuf, String>,
}

impl AppState {
    pub fn new(
        paths: AppPaths,
        token: HfToken,
        binary: std::result::Result<PathBuf, String>,
        downloads: Downloads,
    ) -> Self {
        let (settings, report) = settings::load(&paths.settings, &token);
        let mut notes = Vec::new();
        if let Some(error) = report.token_migration_error {
            notes.push(format!(
                "Your Hugging Face token could not be moved to the keychain: {error} \
                 It is still in the old settings file; open Settings and save it again."
            ));
        }
        if let Some(aside) = report.set_aside {
            notes.push(format!(
                "The settings file could not be read, so the defaults are in use. \
                 The old file was kept as {}.",
                aside.display()
            ));
        }
        Self {
            jobs: JobHub::new(binary.clone(), paths.output_root.clone()),
            paths,
            settings: RwLock::new(settings),
            token,
            models: RwLock::new(Vec::new()),
            downloads,
            sessions: Arc::new(Sessions::default()),
            notes: Mutex::new(notes),
            binary,
        }
    }

    /// End everything the app started, ready for it to exit: downloads are
    /// cancelled (what was fetched stays cached and resumes next time), a chat
    /// stops at its next token, and every job is stopped, which gives a
    /// training run time to write its adapter.
    pub async fn shutdown(&self) {
        self.downloads.cancel_all();
        self.sessions.cancel(None);
        self.jobs.shutdown().await;
    }

    // ----- settings -----------------------------------------------------

    pub async fn config(&self) -> ConfigView {
        let settings = self.settings.read().await.clone();
        let token = self.token.clone();
        let hf_token_set = tokio::task::spawn_blocking(move || token.is_set())
            .await
            .unwrap_or(false);
        ConfigView {
            default_model: settings.default_model,
            theme: settings.theme,
            custom_model_dirs: settings.custom_model_dirs,
            hf_cache_dir: self.paths.hub_cache.display().to_string(),
            output_dir: self.paths.output_root.display().to_string(),
            hf_token_set,
            notes: self.notes.lock().map(|n| n.clone()).unwrap_or_default(),
            pmetal_binary: self.binary.as_ref().ok().map(|p| p.display().to_string()),
            pmetal_binary_error: self.binary.as_ref().err().cloned(),
        }
    }

    pub async fn update_settings(&self, patch: SettingsPatch) -> Result<ConfigView> {
        {
            let mut settings = self.settings.write().await;
            let mut next = settings.clone();
            if let Some(model) = patch.default_model {
                let model = model.trim().to_string();
                next.default_model = (!model.is_empty()).then_some(model);
            }
            if let Some(theme) = patch.theme {
                next.theme = theme;
            }
            self.persist(&next)?;
            *settings = next;
        }
        Ok(self.config().await)
    }

    fn persist(&self, settings: &Settings) -> Result<()> {
        settings::save(&self.paths.settings, settings).map_err(|e| {
            AppError::new(format!(
                "Could not save settings to {}: {e}. Check that the folder is writable.",
                self.paths.settings.display()
            ))
        })
    }

    /// Save the Hugging Face token in the keychain.
    pub async fn set_token(&self, token: String) -> Result<ConfigView> {
        let store = self.token.clone();
        tokio::task::spawn_blocking(move || store.set(&token))
            .await
            .map_err(|e| AppError::new(e.to_string()))??;
        // A token saved here supersedes one that failed to migrate.
        if let Ok(mut notes) = self.notes.lock() {
            notes.retain(|n| !n.contains("Hugging Face token"));
        }
        Ok(self.config().await)
    }

    pub async fn clear_token(&self) -> Result<ConfigView> {
        let store = self.token.clone();
        tokio::task::spawn_blocking(move || store.clear())
            .await
            .map_err(|e| AppError::new(e.to_string()))??;
        Ok(self.config().await)
    }

    /// The saved token, if there is one. A keychain that cannot be read gives
    /// none, with a warning in the log: the job then runs as an anonymous
    /// user and says what it needs if the repository is gated.
    pub async fn token_for_hub(&self) -> Option<SecretString> {
        let store = self.token.clone();
        match tokio::task::spawn_blocking(move || store.get()).await {
            Ok(Ok(token)) => token,
            Ok(Err(e)) => {
                tracing::warn!("keychain read failed: {e}");
                None
            }
            Err(_) => None,
        }
    }

    // ----- models -------------------------------------------------------

    async fn scan_plan(&self) -> ScanPlan {
        let custom_dirs = self
            .settings
            .read()
            .await
            .custom_model_dirs
            .iter()
            .map(PathBuf::from)
            .collect();
        ScanPlan {
            hub_cache: self.paths.hub_cache.clone(),
            output_root: self.paths.output_root.clone(),
            custom_dirs,
            well_known: ScanPlan::well_known_dirs(),
        }
    }

    pub async fn refresh_models(&self) -> Vec<CachedModel> {
        let plan = self.scan_plan().await;
        let found = tokio::task::spawn_blocking(move || crate::models::scan(&plan))
            .await
            .unwrap_or_default();
        *self.models.write().await = found.clone();
        found
    }

    pub async fn models(&self) -> Vec<CachedModel> {
        self.models.read().await.clone()
    }

    pub async fn find_model(&self, id_or_path: &str) -> Option<CachedModel> {
        self.models
            .read()
            .await
            .iter()
            .find(|m| m.id == id_or_path || m.path == id_or_path || m.reference == id_or_path)
            .cloned()
    }

    pub async fn add_model_directory(&self, raw: &str) -> Result<Vec<CachedModel>> {
        let dir = crate::paths::scan_dir(raw).map_err(AppError::new)?;
        {
            let mut settings = self.settings.write().await;
            let text = dir.display().to_string();
            if !settings.custom_model_dirs.contains(&text) {
                let mut next = settings.clone();
                next.custom_model_dirs.push(text);
                self.persist(&next)?;
                *settings = next;
            }
        }
        Ok(self.refresh_models().await)
    }

    pub async fn remove_model_directory(&self, raw: &str) -> Result<Vec<CachedModel>> {
        {
            let mut settings = self.settings.write().await;
            let mut next = settings.clone();
            next.custom_model_dirs.retain(|d| d != raw);
            self.persist(&next)?;
            *settings = next;
        }
        Ok(self.refresh_models().await)
    }

    /// Delete a model from disk. A Hub model is evicted from the Hub cache; any
    /// other must be inside a folder the app scans, and never one of those
    /// folders itself.
    pub async fn delete_model(&self, id: &str) -> Result<()> {
        let model = self.find_model(id).await.ok_or_else(|| {
            AppError::new(format!(
                "'{id}' is not in the model list. Refresh the list and try again."
            ))
        })?;
        if let Some(label) = self.jobs.model_in_use(|title| {
            title == model.reference || title == model.id || title == model.path
        }) {
            return Err(AppError::new(format!(
                "Cannot delete {}: a {label} job is using it. Stop the job first.",
                model.id
            )));
        }
        match model.source {
            ModelSource::HfCache => {
                // The repository's folder in the cache, named the way the Hub
                // names it. The id came from a folder name, but a path is
                // never built from text without checking where it lands.
                let folder = self
                    .paths
                    .hub_cache
                    .join(pmetal::core::paths::hub_repo_folder("models", &model.id));
                if folder.parent() != Some(self.paths.hub_cache.as_path()) || !folder.is_dir() {
                    return Err(AppError::new(format!(
                        "Could not find {} in the Hugging Face cache. Refresh the model list.",
                        model.id
                    )));
                }
                tokio::fs::remove_dir_all(&folder)
                    .await
                    .map_err(|e| AppError::new(format!("Could not delete {}: {e}", model.id)))?;
            }
            ModelSource::Trained | ModelSource::Custom => {
                let plan = self.scan_plan().await;
                let path = PathBuf::from(&model.path);
                let real = path
                    .canonicalize()
                    .map_err(|e| AppError::new(format!("Could not find {}: {e}", model.path)))?;
                let allowed = plan.roots().iter().any(|root| {
                    root.canonicalize()
                        .is_ok_and(|root| real != root && real.starts_with(root))
                });
                if !allowed {
                    return Err(AppError::new(format!(
                        "{} is not inside a folder PMetal manages, so it will not delete it. \
                         Delete it in Finder if you want it gone.",
                        model.path
                    )));
                }
                tokio::fs::remove_dir_all(&real)
                    .await
                    .map_err(|e| AppError::new(format!("Could not delete {}: {e}", model.path)))?;
            }
        }
        self.models.write().await.retain(|m| m.path != model.path);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
