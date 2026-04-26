mod commands;
mod state;

use commands::*;
use state::AppState;
use tauri::{Manager, WindowEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Set up file + stderr logging, plus a panic hook that writes to the log.
    let log_path = init_logging("gui");
    install_panic_hook(log_path);

    let app_state = AppState::new();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(app_state)
        .setup(|app| {
            #[cfg(desktop)]
            {
                // Auto-updater disabled — requires signing key in CI.
                // app.handle()
                //     .plugin(tauri_plugin_updater::Builder::new().build())?;
                app.handle().plugin(tauri_plugin_process::init())?;
            }

            let state = app.state::<AppState>();

            // Log startup diagnostics for troubleshooting
            log_startup_diagnostics();

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
            get_model_defaults,
            download_model,
            delete_model,
            search_hub_models,
            get_trending_models,
            get_model_fit,
            add_model_directory,
            remove_model_directory,
            list_model_directories,
            // Datasets
            search_hub_datasets,
            get_trending_datasets,
            list_cached_datasets,
            download_dataset,
            peek_dataset_columns,
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
            // Serve
            start_serve,
            stop_serve,
            list_serve_instances,
            // Bench
            start_bench,
            stop_bench,
            list_bench_runs,
            // Eval
            start_eval,
            stop_eval,
            list_eval_runs,
            // Pretrain
            start_pretrain,
            stop_pretrain,
            list_pretrain_runs,
            // Inference
            start_inference,
            stop_inference,
            // Adapters
            list_trained_adapters,
            // Merge / Fuse / Quantize
            merge_models,
            get_merge_strategies,
            fuse_lora,
            quantize_model,
        ])
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                // Cancel all active training/distillation/GRPO runs so Metal
                // command buffers can drain before the process exits.
                if let Some(state) = window.try_state::<AppState>() {
                    if let Ok(flags) = state.cancel_flags.try_read() {
                        for flag in flags.values() {
                            flag.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                    }

                    // Drain the broadcast channel so in-flight events emitted
                    // just before cancellation have a chance to be forwarded to
                    // the frontend before the process exits.
                    //
                    // Strategy: send a sentinel ModelRemoved("") event to flush
                    // the forwarder's receive loop, then wait up to 1 s for the
                    // broadcast channel to become idle (no more receivers). If
                    // the drain window elapses we proceed anyway — the 1 s bound
                    // prevents hanging when a task is blocked on a long Metal op.
                    //
                    // Note: We cannot join the forwarder task here because Tauri's
                    // `on_window_event` handler is synchronous and the task handle
                    // is not exposed. The sleep gives async tasks one tick to flush.
                    let _ = state
                        .event_tx
                        .send(crate::state::AppEvent::ModelRemoved {
                            model_id: String::new(),
                        });
                }

                // Give in-flight GPU work and the event forwarder up to 1 s to
                // flush, then force-exit to skip C++ destructor crashes from
                // MLX's Metal device cleanup.
                std::thread::sleep(std::time::Duration::from_millis(1_000));
                std::process::exit(0);
            }
        })
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

/// Log startup diagnostics so crash reports have context.
fn log_startup_diagnostics() {
    tracing::info!(
        version = pmetal::version::VERSION,
        arch = std::env::consts::ARCH,
        os = std::env::consts::OS,
        "PMetal GUI starting"
    );

    match pmetal::metal::MetalContext::global() {
        Ok(ctx) => {
            let props = ctx.properties();
            tracing::info!(
                gpu = %props.name,
                gpu_cores = props.gpu_core_count,
                ane_cores = props.ane_core_count,
                bandwidth_gbps = format!("{:.1}", props.memory_bandwidth_gbps),
                "Metal device initialized"
            );
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "Metal device initialization failed — GPU features will be unavailable"
            );
        }
    }

    let device = pmetal::version::device_info();
    tracing::info!(
        total_memory_gb = format!("{:.1}", device.memory_total_gb),
        available_memory_gb = format!("{:.1}", device.memory_available_gb),
        "System memory"
    );
}

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
        // Build a temporary AppState-like struct to reuse the full scanning logic
        let app_state = AppState::new();
        *app_state.config.write().await = self.config.read().await.clone();
        app_state.refresh_cached_models().await;

        let models = app_state.cached_models.read().await.clone();
        let count = models.len();
        *self.cached_models.write().await = models;
        tracing::info!("Refreshed model cache: {count} models found");
    }
}

// ---------------------------------------------------------------------------
// Persistent logging
// ---------------------------------------------------------------------------

/// Log directory: ~/.cache/pmetal/logs/
fn log_dir() -> std::path::PathBuf {
    let dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("pmetal")
        .join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Initialize logging with both stderr and file output.
///
/// Log file: `~/.cache/pmetal/logs/{component}.log`
/// Rotates on startup (previous log renamed to `{component}.log.1`).
/// Returns the log file path for use in the panic hook.
fn init_logging(component: &str) -> Option<std::path::PathBuf> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(
            "pmetal_gui=info,pmetal_trainer=info,pmetal_mlx=info,pmetal_lora=info,pmetal_models=info,pmetal=info",
        )
    });

    let dir = log_dir();
    let log_path = dir.join(format!("{component}.log"));

    // Rotate: keep one previous log
    let prev = dir.join(format!("{component}.log.1"));
    if log_path.exists() {
        let _ = std::fs::rename(&log_path, &prev);
    }

    let file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(_) => {
            // Fall back to stderr-only logging
            let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
            return None;
        }
    };

    // Tee writer: stderr + file
    let file = std::sync::Arc::new(std::sync::Mutex::new(file));
    let file_clone = file.clone();

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(move || TeeWriter {
            stderr: std::io::stderr(),
            file: Some(file_clone.clone()),
        })
        .try_init();

    Some(log_path)
}

/// Writer that tees output to both stderr and a log file.
struct TeeWriter {
    stderr: std::io::Stderr,
    file: Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>>,
}

impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.stderr.write(buf)?;
        if let Some(ref file) = self.file {
            if let Ok(mut f) = file.lock() {
                // Strip ANSI escape codes for the file
                let _ = f.write_all(&buf[..n]);
            }
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stderr.flush()?;
        if let Some(ref file) = self.file {
            if let Ok(mut f) = file.lock() {
                let _ = f.flush();
            }
        }
        Ok(())
    }
}

/// Install a panic hook that writes crash details to the log file.
///
/// On panic, appends the panic info + backtrace to the log file so that
/// even if the GUI exits immediately, crash context is preserved on disk.
fn install_panic_hook(log_path: Option<std::path::PathBuf>) {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Write to the log file directly (tracing may not work in a panic)
        if let Some(ref path) = log_path {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path) {
                let _ = writeln!(f, "\n=== PANIC ===");
                let _ = writeln!(f, "Time: {:?}", std::time::SystemTime::now());
                let _ = writeln!(f, "{info}");
                let bt = std::backtrace::Backtrace::force_capture();
                let _ = writeln!(f, "{bt}");
                let _ = writeln!(f, "=== END PANIC ===\n");
            }
        }
        // Also log via tracing (may or may not work depending on panic context)
        tracing::error!("PANIC: {info}");
        original(info);
    }));
}
