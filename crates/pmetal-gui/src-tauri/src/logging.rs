//! The app's log file, and what a crash leaves behind.

use std::io::Write;
use std::path::PathBuf;

use pmetal::core::logging::{LogOptions, LogOutput};

/// Log to standard error and to `<component>.log`, keeping the previous run's
/// log as `<component>.log.1`. Returns the log file's path for the panic hook.
pub fn init(component: &str) -> Option<PathBuf> {
    pmetal::core::logging::init(
        LogOptions::new(component)
            .output(LogOutput::StderrAndFile)
            .default_filter(
                "pmetal_gui=info,pmetal_gui_lib=info,pmetal_supervisor=info,pmetal_trainer=info,\
                 pmetal_runtime=info,pmetal_hub=info,pmetal=info",
            ),
    )
}

/// On a panic: write what happened to the log file, and end the jobs. The jobs
/// are processes of their own, so an app that dies without stopping them
/// leaves them training with nobody to watch.
pub fn install_panic_hook(log_path: Option<PathBuf>, supervisor: pmetal_supervisor::Supervisor) {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(path) = &log_path
            && let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path)
        {
            let _ = writeln!(f, "\n=== PANIC ===\n{info}");
            let _ = writeln!(f, "{}", std::backtrace::Backtrace::force_capture());
            let _ = writeln!(f, "=== END PANIC ===\n");
        }
        tracing::error!("PANIC: {info}");
        // A panic in a worker thread is survivable; the jobs are stopped only
        // when the main thread goes down.
        if std::thread::current().name() == Some("main") {
            supervisor.kill_all_now();
        }
        original(info);
    }));
}
