#[cfg(feature = "trainer")]
pub(crate) mod bench;
#[cfg(feature = "distributed")]
pub(crate) mod cluster;
pub(crate) mod completions;
pub(crate) mod dataset;
pub(crate) mod decide;
pub(crate) mod dflash;
#[cfg(feature = "trainer")]
pub(crate) mod distill;
pub(crate) mod download;
#[cfg(feature = "trainer")]
pub(crate) mod embed_train;
#[cfg(all(feature = "lora", feature = "trainer"))]
pub(crate) mod eval;
#[cfg(feature = "trainer")]
pub(crate) mod events;
#[cfg(feature = "lora")]
pub(crate) mod fuse;
pub(crate) mod gguf_export;
#[cfg(feature = "trainer")]
pub(crate) mod grpo;
pub(crate) mod imatrix;
pub(crate) mod infer;
pub(crate) mod info;
#[cfg(feature = "trainer")]
pub(crate) mod init;
#[cfg(feature = "trainer")]
pub(crate) mod machine_load;
#[cfg(feature = "merge")]
pub(crate) mod merge;
pub(crate) mod ollama;
#[cfg(feature = "trainer")]
pub(crate) mod preference;
pub(crate) mod quantize;
pub(crate) mod quantize_source;
#[cfg(feature = "trainer")]
pub(crate) mod resume;
#[cfg(feature = "trainer")]
pub(crate) mod rlkd;
pub(crate) mod search;
#[cfg(feature = "serve")]
pub(crate) mod serve;
// `pmetal infer` stops on a request too; the trainers use the rest of it.
#[cfg_attr(not(feature = "trainer"), allow(dead_code))]
pub(crate) mod stop;
pub(crate) mod tokenize;
#[cfg(feature = "trainer")]
pub(crate) mod train;
#[cfg(feature = "trainer")]
pub(crate) mod train_diffusion;
#[cfg(feature = "trainer")]
pub(crate) mod train_draft;
#[cfg(feature = "trainer")]
pub(crate) mod train_mtp;
pub(crate) mod verify;
#[cfg(feature = "vocoder")]
pub(crate) mod vocode;

/// The error for a spec that failed its own validation: every problem, with
/// the flag each belongs to, so the message can be acted on without reading
/// the source.
pub(crate) fn invalid_spec<S: pmetal_core::JobFields>(
    errors: &[pmetal_core::FieldError],
) -> anyhow::Error {
    let lines: Vec<String> = errors
        .iter()
        .map(|error| {
            let flag = S::field_descriptors()
                .iter()
                .find(|d| d.name == error.field)
                .and_then(|d| d.argv)
                .map_or_else(|| error.field.clone(), str::to_string);
            if error.message == "required" {
                format!("  {flag} is required")
            } else {
                format!("  {flag} {}", error.message)
            }
        })
        .collect();
    anyhow::anyhow!(
        "invalid arguments for `pmetal {}`:\n{}",
        S::subcommand(),
        lines.join("\n")
    )
}

/// What a surface adds to a trainer command's run: whether the run prints its
/// banner and summary (the command line does, the TUI and GUI draw their own),
/// and the callbacks that carry its progress out.
#[cfg(feature = "trainer")]
pub(crate) struct RunContext {
    pub(crate) emit_console_output: bool,
    pub(crate) callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>>,
}

#[cfg(feature = "trainer")]
impl RunContext {
    /// The command line's: a banner and a summary, no extra callbacks.
    pub(crate) fn console() -> Self {
        Self {
            emit_console_output: true,
            callbacks: Vec::new(),
        }
    }
}

/// Leave beside a run's results what it started from (`training_info.json`),
/// so the terminal UI and the desktop app can list the adapter with its base
/// model. A record that cannot be written is a warning: the run does not
/// depend on it.
#[cfg(feature = "trainer")]
pub(crate) fn record_training(
    output_dir: &std::path::Path,
    base_model: &str,
    base_model_path: &std::path::Path,
    dataset: &str,
    method: &str,
) {
    pmetal_core::TrainingInfo {
        base_model_path: Some(base_model_path.display().to_string()),
        dataset: Some(dataset.to_string()),
        method: Some(method.to_string()),
        ..pmetal_core::TrainingInfo::starting(base_model)
    }
    .write_or_warn(output_dir);
}

/// Run a trainer command the way the command line does: `--log-events`
/// applied (the run gets the event log's callback, and a failed log write is
/// reported when it ends), and a request to stop honoured (Ctrl-C or SIGTERM
/// ends the run after the step in flight, which saves what it has; the command
/// then exits with the interrupted status).
#[cfg(feature = "trainer")]
pub(crate) async fn with_run_context<F, Fut>(
    events: Option<&std::path::Path>,
    job_id: &str,
    run: F,
) -> anyhow::Result<()>
where
    F: FnOnce(RunContext) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    let log = events.map(events::EventLog::create).transpose()?;
    let stop = stop::StopFlag::watch();
    let mut ctx = RunContext::console();
    ctx.callbacks
        .extend(log.iter().map(|log| log.callback(job_id)));
    ctx.callbacks.push(Box::new(stop.clone()));
    let result = run(ctx).await;
    let result = match log {
        Some(log) => log.finish(result),
        None => result,
    };
    stop.finish(result)
}
