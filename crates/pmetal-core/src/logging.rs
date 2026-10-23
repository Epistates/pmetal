//! Where PMetal's programs log, and what they say.
//!
//! The command line, the terminal UI, the desktop app and the MCP server all
//! set up `tracing` the same way: a filter from `RUST_LOG` (or the program's
//! own default), the Hub download client's per-request chatter held to
//! warnings, standard error, and a log file under [`log_dir`] that only its
//! owner can read (it names paths and models), the previous run's kept as
//! `<component>.log.1`.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Overrides the log file's path (and turns the rotation off).
pub const LOG_FILE_ENV: &str = "PMETAL_LOG_FILE";

/// Where a program's log lines go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogOutput {
    /// Standard error only.
    Stderr,
    /// Standard error and the log file.
    StderrAndFile,
    /// The log file only: a terminal interface owns standard error.
    File,
}

/// How a program logs.
#[derive(Debug, Clone, Copy)]
pub struct LogOptions<'a> {
    /// Names the log file, `<component>.log`.
    pub component: &'a str,
    /// Where the lines go.
    pub output: LogOutput,
    /// The filter when `RUST_LOG` is not set, in `RUST_LOG`'s syntax.
    pub default_filter: &'a str,
}

impl<'a> LogOptions<'a> {
    /// Options for `component`: both sinks, `info` unless `RUST_LOG` says
    /// otherwise.
    pub const fn new(component: &'a str) -> Self {
        Self {
            component,
            output: LogOutput::StderrAndFile,
            default_filter: "info",
        }
    }

    /// Send the lines to `output`.
    #[must_use]
    pub const fn output(mut self, output: LogOutput) -> Self {
        self.output = output;
        self
    }

    /// Use `filter` when `RUST_LOG` is not set.
    #[must_use]
    pub const fn default_filter(mut self, filter: &'a str) -> Self {
        self.default_filter = filter;
        self
    }
}

/// `~/.cache/pmetal/logs`, created if it is missing.
pub fn log_dir() -> PathBuf {
    let dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("pmetal")
        .join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// The filter a program logs with: `RUST_LOG`, or `default_filter`, with the
/// Hub download client held to warnings. It logs every request at info, several
/// lines per downloaded file. A more specific `RUST_LOG=xet_client=info` still
/// wins.
pub fn filter(default_filter: &str) -> EnvFilter {
    let base = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    base.add_directive("xet=warn".parse().expect("a static directive"))
        .add_directive("hf_xet=warn".parse().expect("a static directive"))
}

/// Open `path` for writing as a new file readable by its owner only.
fn open_private(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}

/// Where `component`'s log goes, rotating the last run's out of the way.
fn log_path(component: &str) -> PathBuf {
    if let Some(custom) = std::env::var_os(LOG_FILE_ENV) {
        return PathBuf::from(custom);
    }
    let dir = log_dir();
    let path = dir.join(format!("{component}.log"));
    if path.exists() {
        let _ = std::fs::rename(&path, dir.join(format!("{component}.log.1")));
    }
    path
}

/// Start logging. Returns the log file's path, for a panic hook that wants to
/// record there, or `None` when no file is being written (by choice, or
/// because it could not be opened; standard error still works then).
///
/// A second call does nothing: the first subscriber stays.
pub fn init(options: LogOptions<'_>) -> Option<PathBuf> {
    let filter = filter(options.default_filter);
    let want_stderr = options.output != LogOutput::File;
    let (path, file) = if options.output == LogOutput::Stderr {
        (None, None)
    } else {
        let path = log_path(options.component);
        let file = open_private(&path).ok();
        (file.is_some().then_some(path), file)
    };

    let colour = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let stderr = want_stderr.then(|| {
        tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(colour)
    });
    let file = file.map(|file| {
        tracing_subscriber::fmt::layer()
            .with_writer(Mutex::new(file))
            .with_ansi(false)
    });
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(stderr)
        .with(file)
        .try_init();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_file_is_readable_by_its_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.log");
        drop(open_private(&path).unwrap());
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn the_hub_download_client_is_held_to_warnings() {
        let rendered = filter("info").to_string();
        assert!(rendered.contains("xet=warn"), "{rendered}");
        assert!(rendered.contains("hf_xet=warn"), "{rendered}");
    }
}
