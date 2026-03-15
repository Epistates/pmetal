mod commands;
mod state;

use commands::*;
use state::AppState;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("pmetal_gui=info")),
        )
        .try_init();

    let app_state = AppState::new();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(app_state)
        .setup(|app| {
            let state = app.state::<AppState>();

            // Start the broadcast -> Tauri event forwarder
            start_event_forwarder(app.handle().clone(), &state);

            // Load persisted config and refresh model cache on startup.
            // We clone the inner Arcs because AppState itself is not Clone
            // (tokio::process::Child is not Clone).
            let init_state = AppStateInit {
                config: state.config.clone(),
                cached_models: state.cached_models.clone(),
            };

            tauri::async_runtime::spawn(async move {
                init_state.load_config().await;
                init_state.refresh_cached_models().await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // System
            get_config,
            set_config,
            get_system_info,
            get_device_info,
            get_dashboard_stats,
            // Models
            list_models,
            get_model_info,
            download_model,
            delete_model,
            search_hub_models,
            get_trending_models,
            get_model_fit,
            // Datasets
            search_hub_datasets,
            get_trending_datasets,
            list_cached_datasets,
            // Training
            start_training,
            get_training_status,
            list_training_runs,
            stop_training,
            // Distillation
            start_distillation,
            get_distillation_status,
            list_distillation_runs,
            stop_distillation,
            // GRPO
            start_grpo,
            get_grpo_status,
            list_grpo_runs,
            stop_grpo,
            // Inference
            start_inference,
            stop_inference,
            // Merge / Fuse / Quantize
            merge_models,
            get_merge_strategies,
            fuse_lora,
            quantize_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running pmetal-gui");
}

// ---------------------------------------------------------------------------
// Lightweight init handle
//
// AppState owns a `HashMap<String, tokio::process::Child>` which is not Clone.
// Rather than making AppState Clone (which would require Arc<Mutex<Child>>
// everywhere and complicate kill()), we pass just the Arcs we need for
// startup init tasks.
// ---------------------------------------------------------------------------

struct AppStateInit {
    config: std::sync::Arc<tokio::sync::RwLock<state::AppConfig>>,
    cached_models: std::sync::Arc<tokio::sync::RwLock<Vec<state::CachedModel>>>,
}

impl AppStateInit {
    async fn load_config(&self) {
        let path = AppState::config_path_pub();
        if let Ok(data) = tokio::fs::read_to_string(&path).await {
            if let Ok(cfg) = serde_json::from_str::<state::AppConfig>(&data) {
                *self.config.write().await = cfg;
                tracing::info!("Loaded config from {}", path.display());
            }
        }
    }

    async fn refresh_cached_models(&self) {
        let cache_root = {
            let cfg = self.config.read().await;
            std::path::PathBuf::from(&cfg.cache_dir)
        };

        let hub_models_dir = cache_root.join("hub");
        let models = state::scan_hub_cache_pub(&hub_models_dir).await;
        *self.cached_models.write().await = models;
        tracing::info!(
            "Refreshed model cache: {} models found",
            self.cached_models.read().await.len()
        );
    }
}
