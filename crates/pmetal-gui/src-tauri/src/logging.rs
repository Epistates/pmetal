//! The app's log file, and what a crash leaves behind.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// `~/.cache/pmetal/logs`
fn log_dir() -> PathBuf {
    let dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("pmetal")
        .join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Log to standard error and to `<component>.log`, keeping the previous run's
/// log as `<component>.log.1`. Returns the log file's path for the panic hook.
pub fn init(component: &str) -> Option<PathBuf> {
    use std::os::unix::fs::OpenOptionsExt;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(
            "pmetal_gui=info,pmetal_gui_lib=info,pmetal_supervisor=info,pmetal_trainer=info,\
             pmetal_runtime=info,pmetal_hub=info,pmetal=info",
        )
    });
    let dir = log_dir();
    let path = dir.join(format!("{component}.log"));
    if path.exists() {
        let _ = std::fs::rename(&path, dir.join(format!("{component}.log.1")));
    }
    // Readable by the owner only: paths and model names are in it.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&path);
    let Ok(file) = file else {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
        return None;
    };
    let file = Arc::new(Mutex::new(file));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_writer(move || Tee {
            stderr: std::io::stderr(),
            file: file.clone(),
        })
        .try_init();
    Some(path)
}

struct Tee {
    stderr: std::io::Stderr,
    file: Arc<Mutex<std::fs::File>>,
}

impl Write for Tee {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.stderr.write(buf)?;
        if let Ok(mut f) = self.file.lock() {
            let _ = f.write_all(&buf[..n]);
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stderr.flush()?;
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
        Ok(())
    }
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
