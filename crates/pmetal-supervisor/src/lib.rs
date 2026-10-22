//! Runs `pmetal` subcommands as child processes and owns them until they end.
//!
//! The terminal UI, the MCP server, the Python bindings and the desktop app
//! all start long jobs (training, quantization, a server) as
//! `pmetal <subcommand> <argv…>` children. This crate is the one place that
//! does it:
//!
//! - **Process groups.** A job leads its own group; stopping it signals the
//!   group (SIGTERM, then SIGKILL after a grace period), so the helpers it
//!   spawned go with it, and dropping the last [`Supervisor`] handle kills
//!   what is left. No child outlives the program that started it.
//! - **Bounded, ordered output.** Both streams share one ring bounded by
//!   lines, bytes and line length. Lines are redacted (tokens) and stripped of
//!   terminal escape sequences, so they are safe to draw in a terminal and to
//!   show to a model. Output that is not UTF-8 is repaired, not dropped.
//! - **Progress from the job itself.** The files a trainer writes
//!   (`--log-events`, `--log-metrics`) and `JobEvent` lines on standard output
//!   are followed while the job runs and folded into [`JobMetrics`].
//! - **Events on a bounded channel.** [`Supervisor::subscribe`] publishes what
//!   happens to every job ([`Event`]). A subscriber that falls behind is told
//!   so and catches up from the supervisor's state; a job never waits for a
//!   reader.
//!
//! ```no_run
//! # async fn demo() -> Result<(), pmetal_supervisor::Error> {
//! use pmetal_supervisor::{JobRequest, Limits, Supervisor, resolve_pmetal_binary};
//!
//! let supervisor = Supervisor::new(resolve_pmetal_binary(), Limits::default());
//! let mut events = supervisor.subscribe();
//! let id = supervisor.spawn(JobRequest {
//!     subcommand: "info".into(),
//!     ..JobRequest::default()
//! })?;
//! while let Ok(event) = events.recv().await {
//!     if event.job == id {
//!         println!("{:?}", event.kind);
//!     }
//! }
//! supervisor.shutdown().await;
//! # Ok(())
//! # }
//! ```

mod group;
mod log;
mod metrics;
mod supervisor;
mod tail;
mod text;

pub use group::{kill_group, resolve_pmetal_binary};
pub use log::{LogLine, LogPage, MAX_LINE_BYTES, MAX_LOG_BYTES, MAX_LOG_LINES, Stream};
pub use metrics::{JobMetrics, parse_record};
pub use supervisor::{
    Error, Event, EventKind, JobId, JobRequest, JobSummary, Limits, ProcessStatus, StepPage,
    Supervisor,
};
pub use tail::Tail;
pub use text::{redact_argv, redact_secrets, sanitize_line};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
