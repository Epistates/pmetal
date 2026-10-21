mod adapters;
mod binary;
mod commands;
mod datasets;
mod error;
mod hub;
mod inference;
mod jobs;
mod logging;
mod models;
mod paths;
mod secrets;
mod settings;
mod state;
mod system;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use commands::*;
use state::{AppPaths, AppState};
use tauri::{AppHandle, Emitter, Manager, RunEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

/// Downloads that may transfer at once.
const DOWNLOADS_AT_ONCE: usize = 3;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let log_path = logging::init("gui");

    // The state exists before the window does; what it sends to the window it
    // sends through this handle, set once the app is built.
    let handle: Arc<OnceLock<AppHandle>> = Arc::new(OnceLock::new());
    let download_handle = handle.clone();
    let downloads = hub::Downloads::new(
        DOWNLOADS_AT_ONCE,
        Arc::new(move |view| {
            if let Some(app) = download_handle.get() {
                let _ = app.emit("download-update", view);
            }
        }),
    );
    let binary = binary::locate();
    match &binary {
        Ok(path) => tracing::info!(path = %path.display(), "using the pmetal command-line tool"),
        Err(reason) => tracing::error!("{reason}"),
    }
    let state = AppState::new(
        AppPaths::detect(),
        secrets::HfToken::system(),
        binary,
        downloads,
    );
    logging::install_panic_hook(log_path, state.jobs.supervisor().clone());

    let quitting = Arc::new(AtomicBool::new(false));
    let quitting_for_events = quitting.clone();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state)
        .setup(move |app| {
            let _ = handle.set(app.handle().clone());

            // Must run before anything touches MLX, which loads its metallib
            // lazily on the first array operation.
            ensure_metallib(app.handle());
            // Load it now, so a library MLX can't load is reported once, here,
            // rather than failing (or aborting) the first GPU job.
            if let Err(e) = pmetal_runtime::inline_array::init_device() {
                tracing::error!("MLX could not load its Metal library: {e}");
                app.dialog()
                    .message(format!(
                        "PMetal could not load its Metal library, so chat won't run and \
                         benchmarks may fail.\n\n{e}\n\nReinstall PMetal, or set \
                         PMETAL_METALLIB_PATH to a working mlx.metallib."
                    ))
                    .title("GPU unavailable")
                    .kind(MessageDialogKind::Error)
                    .show(|_| {});
            }
            log_startup_diagnostics();

            let state = app.state::<AppState>();
            let events = Arc::new(TauriJobEvents(app.handle().clone()));
            let jobs = state.jobs.clone();
            tauri::async_runtime::spawn(async move {
                jobs.pump(events);
            });

            // Find the models in the background; the window is already up.
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = app_handle.state::<AppState>();
                let found = state.refresh_models().await;
                tracing::info!("found {} models", found.len());
                let _ = app_handle.emit("models-updated", ());
            });

            // A job outlives its parent unless the parent says otherwise.
            // `kill`, a logout or a shutdown arrive as signals, not as a window
            // closing, so they are given the same orderly end.
            let signal_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                wait_for_termination_signal().await;
                tracing::info!("terminated by a signal; stopping jobs");
                stop_everything_and_exit(&signal_handle).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            get_config,
            set_config,
            set_hf_token,
            clear_hf_token,
            pick_directory,
            pick_file,
            list_models,
            refresh_models,
            get_model_info,
            get_model_defaults,
            delete_model,
            add_model_directory,
            remove_model_directory,
            search_hub_models,
            get_trending_models,
            get_model_fit,
            download_model,
            download_dataset,
            list_downloads,
            cancel_download,
            clear_finished_downloads,
            search_hub_datasets,
            get_trending_datasets,
            list_cached_datasets,
            peek_dataset_columns,
            list_job_forms,
            start_job,
            stop_job,
            forget_job,
            list_jobs,
            job_output,
            job_steps,
            list_trained_adapters,
            start_chat,
            stop_chat,
        ])
        .build(tauri::generate_context!())
        .expect("error while building pmetal-gui");

    app.run(move |app, event| {
        // Closing the last window, Cmd+Q and the Dock's Quit all end up here.
        if let RunEvent::ExitRequested { api, code, .. } = event
            && code.is_none()
        {
            api.prevent_exit();
            if quitting_for_events.swap(true, Ordering::SeqCst) {
                return;
            }
            let app = app.clone();
            let quitting = quitting.clone();
            tauri::async_runtime::spawn(async move {
                if !confirm_quit(&app).await {
                    quitting.store(false, Ordering::SeqCst);
                    return;
                }
                stop_everything_and_exit(&app).await;
            });
        }
    });
}

/// Ask before quitting with jobs running. Quitting stops them: a training run
/// saves its adapter first, but nothing continues without the app.
async fn confirm_quit(app: &AppHandle) -> bool {
    let running = app.state::<AppState>().jobs.running();
    if running == 0 {
        return true;
    }
    let message = if running == 1 {
        "A job is still running. Quitting stops it; a training run saves its adapter first."
            .to_string()
    } else {
        format!(
            "{running} jobs are still running. Quitting stops them; training runs save their \
             adapters first."
        )
    };
    let app = app.clone();
    tokio::task::spawn_blocking(move || {
        app.dialog()
            .message(message)
            .title("Quit PMetal?")
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Stop jobs and quit".to_string(),
                "Keep running".to_string(),
            ))
            .blocking_show()
    })
    .await
    .unwrap_or(false)
}

/// Stop the jobs and leave. Training gets time to write its adapter (SIGTERM,
/// then SIGKILL after a grace period); nothing the app started outlives it.
async fn stop_everything_and_exit(app: &AppHandle) {
    let _ = app.emit("app-quitting", ());
    app.state::<AppState>().shutdown().await;
    // MLX's Metal teardown can crash in a C++ destructor on a normal exit;
    // everything that mattered has been stopped and saved by now.
    std::process::exit(0);
}

async fn wait_for_termination_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut term), Ok(mut int), Ok(mut hup)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
        signal(SignalKind::hangup()),
    ) else {
        std::future::pending::<()>().await;
        return;
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
        _ = hup.recv() => {}
    }
}

/// Point the MLX backend at the `mlx.metallib` built with the MLX this app
/// links.
///
/// MLX cannot load a single Metal kernel without it. The bundled app carries
/// one as a Tauri resource; a build from a source checkout uses the one its
/// build produced. The search is `pmetal_runtime::metallib::locate`'s, as the
/// CLI's: `PMETAL_METALLIB_PATH`, the per-version metallib cache, the build's
/// own file, then the bundle resource and a copy next to the executable. The
/// `pmetal` processes this app starts run their own search.
fn ensure_metallib(app: &AppHandle) {
    use pmetal_runtime::metallib;
    use tauri::path::BaseDirectory;

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(path) = app.path().resolve("mlx.metallib", BaseDirectory::Resource) {
        candidates.push(path);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("mlx.metallib"));
    }

    // A file that is there but fails the header check would make MLX fail
    // its first allocation, so it is skipped with a warning.
    let found = metallib::locate(metallib::BUILT_SHA256, &candidates, |path, e| {
        tracing::warn!(path = %path.display(), "skipping unusable mlx.metallib: {e}");
    });
    match found {
        Some(located) => {
            tracing::info!(path = %located.path.display(), "Found mlx.metallib");
            pmetal_runtime::inline_array::set_metallib_path(&located.path.to_string_lossy());
        }
        None => tracing::error!(
            "mlx.metallib not found — MLX kernels will fail to load. Reinstall PMetal, or set \
             PMETAL_METALLIB_PATH to the mlx.metallib built with MLX {}.",
            metallib::MLX_REVISION
        ),
    }
}

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
        Err(e) => tracing::error!(
            error = %e,
            "Metal device initialization failed — GPU features will be unavailable"
        ),
    }
    let device = pmetal::version::device_info();
    tracing::info!(
        total_memory_gb = format!("{:.1}", device.memory_total_gb),
        available_memory_gb = format!("{:.1}", device.memory_available_gb),
        "System memory"
    );
}

#[cfg(test)]
mod tests;
