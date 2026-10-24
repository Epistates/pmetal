//! PMetal CLI - LLM fine-tuning for Apple Silicon.

pub mod cli;
mod commands;
#[cfg(any(feature = "models", feature = "native-only"))]
pub mod native_inference;
mod pack_experts;
#[cfg(feature = "dashboard")]
mod tui;

use std::path::{Path, PathBuf};

use cli::{Cli, Commands};

/// The gzip-compressed `mlx.metallib` built with the MLX this binary links,
/// embedded by build.rs. Empty when the build had none.
#[cfg(target_os = "macos")]
static MLX_METALLIB_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mlx.metallib.gz"));

/// SHA-256 of the embedded metallib, uncompressed.
const EMBEDDED_METALLIB_SHA256: Option<&str> = option_env!("PMETAL_EMBEDDED_METALLIB_SHA256");

/// Write the embedded metallib into the metallib cache and return its path:
/// the user's (`~/.cache/pmetal/lib`), or one under the temporary directory
/// when that can't be written.
#[cfg(target_os = "macos")]
fn extract_embedded_metallib(sha256: &str) -> Option<PathBuf> {
    use flate2::read::GzDecoder;
    use pmetal_runtime::metallib::Cache;
    use std::io::Read;

    if MLX_METALLIB_GZ.is_empty() {
        return None;
    }
    let mut bytes = Vec::new();
    if let Err(e) = GzDecoder::new(MLX_METALLIB_GZ).read_to_end(&mut bytes) {
        tracing::warn!("the embedded mlx.metallib does not decompress: {e}");
        return None;
    }
    let caches = Cache::user()
        .into_iter()
        .chain([Cache::at(std::env::temp_dir().join("pmetal-metallib"))]);
    for cache in caches {
        match cache.install(&bytes, sha256) {
            Ok(path) => {
                tracing::info!("Extracted the embedded mlx.metallib to {}", path.display());
                return Some(path);
            }
            Err(e) => tracing::warn!(
                dir = %cache.dir().display(),
                "could not write the embedded mlx.metallib: {e}"
            ),
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn extract_embedded_metallib(_sha256: &str) -> Option<PathBuf> {
    None
}

/// Point MLX at the `mlx.metallib` built with the MLX this binary links,
/// before anything builds its Metal device.
///
/// In order (`pmetal_runtime::metallib::locate`): `PMETAL_METALLIB_PATH`; the
/// metallib cache's entry for the embedded copy; the file the build produced,
/// while it exists; an `mlx.metallib` next to the executable, as the release
/// tarball ships it. Failing those, the embedded copy is written into the
/// cache. Each binary's copy is keyed by MLX revision and content hash, so
/// installed pmetal versions don't overwrite each other's.
fn ensure_metallib() {
    use pmetal_runtime::metallib;

    let colocated: Vec<PathBuf> = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("mlx.metallib")))
        .into_iter()
        .collect();
    let sha256 = EMBEDDED_METALLIB_SHA256.or(metallib::BUILT_SHA256);
    let found = metallib::locate(sha256, &colocated, |path, e| {
        tracing::warn!(path = %path.display(), "skipping unusable mlx.metallib: {e}");
    })
    .map(|located| located.path)
    .or_else(|| EMBEDDED_METALLIB_SHA256.and_then(extract_embedded_metallib));

    match found {
        Some(path) => {
            tracing::debug!(path = %path.display(), "Using mlx.metallib");
            pmetal_runtime::inline_array::set_metallib_path(&path.to_string_lossy());
        }
        None => eprintln!(
            "\x1b[1;33mwarning:\x1b[0m no mlx.metallib for this build's MLX ({revision}) was \
             found, and none is embedded in this binary.\n\
             Set PMETAL_METALLIB_PATH to the mlx.metallib built with it, or reinstall pmetal.",
            revision = metallib::MLX_REVISION,
        ),
    }
}

/// Synchronous entry point.
///
/// `ensure_metallib` runs here rather than in `tokio_main` because its
/// download fallback blocks on a runtime of its own, which can't start inside
/// `#[tokio::main]`. Logging is set up first so its messages are recorded, and
/// argument parsing before that, so `--help` and usage errors exit without
/// searching for (or downloading) a metallib. MLX's Metal device is then built
/// before any command that uses MLX, which measured no slower than not doing
/// it; the others (`Commands::uses_mlx`) skip both steps.
/// A run stopped by Ctrl-C after finishing what it was doing: the exit status
/// is 130 (128 + SIGINT), as for any process the interrupt ends.
#[derive(Debug)]
pub(crate) struct Interrupted;

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("interrupted")
    }
}

impl std::error::Error for Interrupted {}

/// The exit status of a failed run: 130 for an interrupt, 1 otherwise (2 is
/// the argument parser's, for a command line it cannot read).
fn exit_status(error: &anyhow::Error) -> u8 {
    if error.is::<Interrupted>() { 130 } else { 1 }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let status = exit_status(&error);
            if status != 130 {
                eprintln!("Error: {error:#}");
            }
            std::process::ExitCode::from(status)
        }
    }
}

fn run() -> anyhow::Result<()> {
    let (cli, given_flags) = Cli::parse_from_env();

    // A command that writes no event log must not take the flag: the caller
    // would wait on a file that never fills.
    #[cfg(feature = "trainer")]
    if cli.log_events.is_some() && !cli.command.writes_event_log() {
        anyhow::bail!(
            "--log-events is written by {} only; this command writes no event log",
            Commands::event_log_commands().join(", ")
        );
    }

    // A training run keeps less of the GPU's work pinned (and so fits more, and
    // gradient checkpointing saves what it should). Every command that writes
    // an event log is one. MLX reads the setting when it first touches the GPU,
    // so it comes before anything else here does.
    #[cfg(feature = "trainer")]
    if cli.command.writes_event_log() {
        pmetal_runtime::memory::bound_command_buffer_memory();
    }

    // The TUI owns the terminal, so it logs to the file only.
    #[cfg(feature = "dashboard")]
    let is_tui = matches!(
        cli.command,
        Commands::Tui { .. } | Commands::Dashboard { .. }
    );
    #[cfg(not(feature = "dashboard"))]
    let is_tui = false;
    init_logging(if is_tui { "tui" } else { "cli" }, is_tui);

    if cli.command.uses_mlx() {
        ensure_metallib();
        // Load the library now: if MLX can't, the command would otherwise fail
        // later, possibly by aborting from a runtime call with no error guard.
        pmetal_runtime::inline_array::init_device().map_err(|e| {
            anyhow::anyhow!(
                "MLX could not load its Metal library: {e}\n\
                 Remove that file, or set PMETAL_METALLIB_PATH to a working mlx.metallib."
            )
        })?;
        #[cfg(feature = "metal")]
        let _ = pmetal_metal::context::MetalContext::device_available();
    }
    tokio_main(cli, given_flags)
}

// ---------------------------------------------------------------------------
// Persistent logging
// ---------------------------------------------------------------------------

/// Initialize logging with file output to `~/.cache/pmetal/logs/{component}.log`.
///
/// - **CLI mode**: logs to both stderr and file
/// - **TUI mode**: logs to file only (stderr would corrupt the terminal), or
///   to `PMETAL_LOG_FILE` if set
///
/// The previous log is rotated to `{component}.log.1`. The setup is shared
/// with the desktop app and the MCP server (`pmetal_core::logging`).
fn init_logging(component: &str, suppress_stderr: bool) {
    use pmetal_core::logging::{LogOptions, LogOutput};
    pmetal_core::logging::init(LogOptions::new(component).output(if suppress_stderr {
        LogOutput::File
    } else {
        LogOutput::StderrAndFile
    }));
}

#[tokio::main]
async fn tokio_main(
    cli: Cli,
    #[cfg_attr(not(feature = "trainer"), allow(unused_variables))]
    given_flags: std::collections::BTreeSet<String>,
) -> anyhow::Result<()> {
    // Capture global flags before `cli` is moved into the `match` below.
    #[cfg(feature = "trainer")]
    let log_events_path = cli.log_events.clone();

    match cli.command {
        #[cfg(feature = "trainer")]
        Commands::Train(args) => {
            commands::train::run_train(args, log_events_path.as_deref(), given_flags).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainMtp(args) => {
            let job_id = format!("train-mtp:{}", args.model);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::train_mtp::run_train_mtp(args, ctx)
            })
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainDraft(args) => {
            let job_id = format!("train-draft:{}", args.target);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::train_draft::run_train_draft(args, ctx)
            })
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainDiffusion(args) => {
            use pmetal_core::jobs::TrainDiffusionSpec;
            let mut spec = TrainDiffusionSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<TrainDiffusionSpec>(&errors))?;
            let job_id = format!("train-diffusion:{}", spec.model);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::train_diffusion::run_train_diffusion(spec, ctx)
            })
            .await?;
        }

        #[cfg(feature = "mcp")]
        Commands::Mcp => {
            pmetal_mcp::run_stdio()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        #[cfg(feature = "serve")]
        Commands::Serve(args) => {
            use pmetal_core::jobs::ServeSpec;
            let (mut spec, api_keys) = args.into_spec();
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<ServeSpec>(&errors))?;
            commands::serve::run_serve(spec, api_keys).await?;
        }

        Commands::Decide(args) => {
            let crate::cli::decide::DecideArgs {
                model,
                request,
                max_length,
                compact,
            } = args;
            commands::decide::run_decide(&model, &request, max_length, compact).await?;
        }

        Commands::Infer(args) => {
            use pmetal_core::jobs::InferSpec;
            let mut spec = InferSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<InferSpec>(&errors))?;
            commands::infer::run_inference(spec).await?;
        }

        Commands::Download(args) => {
            tracing::info!(model = %args.model, "Downloading model");
            commands::download::run_download(args).await?;
        }

        Commands::Upload(args) => {
            commands::download::run_upload(args).await?;
        }

        Commands::Verify(args) => {
            commands::verify::run_verify(args)?;
        }

        Commands::Completions(args) => {
            commands::completions::run_completions(args.shell);
        }

        Commands::PackExperts(args) => {
            let crate::cli::pack_experts::PackExpertsArgs {
                model,
                output,
                bits,
            } = args;
            pack_experts::pack_experts(Path::new(&model), Path::new(&output), bits)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        Commands::Search(args) => {
            commands::search::run_search(args).await?;
        }

        Commands::Dflash(args) => {
            let crate::cli::dflash::DflashArgs {
                target,
                draft,
                prompt,
                max_new_tokens,
                temperature,
                speculative_tokens,
                draft_fp8,
                json,
                no_chat,
                tree_budget,
                compare_greedy,
            } = args;
            commands::dflash::run_dflash(
                &target,
                &draft,
                &prompt,
                max_new_tokens,
                temperature,
                speculative_tokens,
                draft_fp8,
                json,
                no_chat,
                tree_budget,
                compare_greedy,
            )
            .await?;
        }

        Commands::Memory { json } => {
            let stats = pmetal_runtime::memory::get_memory_stats();
            if json {
                let obj = serde_json::json!({
                    "total_gb": stats.total_gb(),
                    "used_gb": stats.used_gb(),
                    "available_gb": stats.available_gb(),
                    "peak_gb": stats.peak_gb(),
                });
                println!("{}", serde_json::to_string_pretty(&obj)?);
            } else {
                println!("Memory Statistics:");
                println!("  Total:     {:.2} GB", stats.total_gb());
                if stats.total_gb() > 0.0 {
                    let used_pct = (stats.used_gb() / stats.total_gb()) * 100.0;
                    let peak_pct = (stats.peak_gb() / stats.total_gb()) * 100.0;
                    println!("  Used:      {:.2} GB ({:.0}%)", stats.used_gb(), used_pct);
                    println!(
                        "  Available: {:.2} GB ({:.0}%)",
                        stats.available_gb(),
                        100.0 - used_pct
                    );
                    println!("  Peak:      {:.2} GB ({:.0}%)", stats.peak_gb(), peak_pct);
                } else {
                    println!("  Used:      {:.2} GB", stats.used_gb());
                    println!("  Available: {:.2} GB", stats.available_gb());
                    println!("  Peak:      {:.2} GB", stats.peak_gb());
                }
            }
        }

        #[cfg(feature = "trainer")]
        Commands::Bench(args) => {
            let crate::cli::bench::BenchArgs {
                model,
                batch_size,
                seq_len,
            } = args;
            commands::bench::run_benchmark(&model, batch_size, seq_len).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchCorpus {
            quick,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "benchmark corpus output"))
                .transpose()?;
            commands::bench::run_kernel_benchmark_corpus(quick, validated_output.as_deref(), json)?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchWorkload {
            preset,
            model,
            dataset,
            experts_dir,
            prompt_samples,
            max_prompt_tokens,
            inference_context,
            decode_steps,
            inference_warmup_passes,
            inference_session_repeats,
            inference_repeats,
            train_samples,
            train_steps,
            batch_size,
            max_seq_len,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "workload benchmark output"))
                .transpose()?;
            if let Some(preset) = preset {
                commands::bench::run_workload_benchmark_preset(
                    preset,
                    inference_warmup_passes,
                    inference_session_repeats,
                    experts_dir.as_deref(),
                    validated_output.as_deref(),
                    json,
                )
                .await?;
            } else {
                commands::bench::run_workload_benchmark(
                    &model,
                    &dataset,
                    experts_dir.as_deref(),
                    prompt_samples,
                    max_prompt_tokens,
                    inference_context,
                    decode_steps,
                    inference_warmup_passes,
                    inference_session_repeats,
                    inference_repeats,
                    train_samples,
                    train_steps,
                    batch_size,
                    max_seq_len,
                    validated_output.as_deref(),
                    json,
                )
                .await?;
            }
        }

        #[cfg(feature = "trainer")]
        Commands::BenchGdn {
            model,
            stage,
            layer,
            batch_size,
            seq_len,
            warmup_iterations,
            benchmark_iterations,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "GDN benchmark output"))
                .transpose()?;
            commands::bench::run_gdn_decode_benchmark(
                &model,
                stage,
                layer,
                batch_size,
                seq_len,
                warmup_iterations,
                benchmark_iterations,
                validated_output.as_deref(),
                json,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Init { output } => {
            // Validate output path to prevent path traversal attacks
            let validated_output = validate_output_path(&output, "config output")?;
            commands::init::generate_sample_config(&validated_output.to_string_lossy())?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchGen { model } => {
            commands::bench::run_gen_benchmark(&model).await?;
        }

        Commands::Ollama { action } => {
            commands::ollama::run_ollama_command(action).await?;
        }

        #[cfg(feature = "vocoder")]
        Commands::Vocode(args) => {
            use pmetal_core::jobs::VocodeSpec;
            let mut spec = VocodeSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<VocodeSpec>(&errors))?;
            commands::vocode::run_vocode(spec).await?;
        }

        #[cfg(feature = "lora")]
        Commands::Fuse(args) => {
            use pmetal_core::jobs::FuseSpec;
            let mut spec = FuseSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<FuseSpec>(&errors))?;
            commands::fuse::run_fuse(spec).await?;
        }

        Commands::Quantize(args) => {
            use pmetal_core::jobs::QuantizeSpec;
            let mut spec = QuantizeSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<QuantizeSpec>(&errors))?;
            commands::quantize::run_quantize(spec).await?;
        }

        Commands::Imatrix(args) => {
            use pmetal_core::jobs::ImatrixSpec;
            let mut spec = ImatrixSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<ImatrixSpec>(&errors))?;
            commands::imatrix::run_imatrix(spec).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Distill(args) => {
            use pmetal_core::jobs::DistillSpec;
            let mut spec = DistillSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<DistillSpec>(&errors))?;
            let job_id = format!("distill:{}", spec.student);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::distill::run_distillation(spec, ctx)
            })
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Grpo(args) => {
            use pmetal_core::jobs::GrpoSpec;
            let mut spec = GrpoSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<GrpoSpec>(&errors))?;
            let job_id = format!("grpo:{}", spec.model);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::grpo::run_grpo(spec, ctx)
            })
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Rlkd(args) => {
            use pmetal_core::jobs::RlkdSpec;
            let mut spec = RlkdSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<RlkdSpec>(&errors))?;
            let job_id = format!("rlkd:{}", spec.model);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::rlkd::run_rlkd(spec, ctx)
            })
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Preference(args) => {
            use pmetal_core::jobs::PreferenceSpec;
            let mut spec = PreferenceSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<PreferenceSpec>(&errors))?;
            let job_id = format!("{}:{}", spec.loss, spec.model);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::preference::run_preference(spec, ctx)
            })
            .await?;
        }

        Commands::Dataset { action } => {
            commands::dataset::run_dataset_command(action).await?;
        }

        Commands::Tokenize(args) => {
            let crate::cli::tokenize::TokenizeArgs {
                input,
                output,
                tokenizer,
                text_column,
                docs_per_shard,
            } = args;
            commands::tokenize::run_tokenize(
                &input,
                &output,
                &tokenizer,
                &text_column,
                docs_per_shard,
            )
            .await?;
        }

        #[cfg(feature = "dashboard")]
        Commands::Dashboard { metrics_file } => {
            let path = metrics_file.map(std::path::PathBuf::from);
            tui::run_dashboard(path).await?;
        }

        #[cfg(not(feature = "dashboard"))]
        Commands::Dashboard { .. } => {
            anyhow::bail!(
                "The dashboard needs the 'dashboard' feature: cargo build --features dashboard"
            );
        }

        #[cfg(feature = "dashboard")]
        Commands::Tui { metrics_file } => {
            let path = metrics_file.map(std::path::PathBuf::from);
            tui::run(path).await?;
        }

        Commands::Info { json } => {
            commands::info::run_info(json).await?;
        }

        #[cfg(feature = "merge")]
        Commands::Merge(args) => {
            let crate::cli::merge::MergeArgs {
                model_a,
                model_b,
                output,
                method,
                base,
                t,
                weight_a,
                weight_b,
                density,
                dtype,
            } = args;
            commands::merge::run_merge_command(
                &model_a,
                &model_b,
                &output,
                &method,
                base.as_deref(),
                t,
                weight_a,
                weight_b,
                density,
                &dtype,
            )
            .await?;
        }

        #[cfg(all(feature = "lora", feature = "trainer"))]
        Commands::Eval(args) => {
            use pmetal_core::jobs::EvalSpec;
            let mut spec = EvalSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<EvalSpec>(&errors))?;
            commands::eval::run_eval(spec).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::EmbedTrain(args) => {
            let crate::cli::embed_train::EmbedTrainArgs {
                model,
                dataset,
                output,
                loss,
                pooling,
                temperature,
                margin,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                weight_decay,
                no_normalize,
                lora,
                lora_r,
                lora_alpha,
                log_every,
                seed,
            } = args;
            let job_id = format!("embed-train:{model}");
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::embed_train::run_embed_train(
                    &model,
                    &dataset,
                    &output,
                    &loss,
                    &pooling,
                    temperature,
                    margin,
                    learning_rate,
                    batch_size,
                    epochs,
                    max_seq_len,
                    weight_decay,
                    !no_normalize,
                    log_every,
                    seed,
                    lora.then_some((lora_r, lora_alpha)),
                    ctx,
                )
            })
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Pretrain(args) => {
            let job_id = format!("pretrain:{}", args.arch);
            commands::with_run_context(log_events_path.as_deref(), &job_id, |ctx| {
                commands::pretrain::run_pretrain(args, ctx)
            })
            .await?;
        }

        #[cfg(feature = "distributed")]
        Commands::Cluster(args) => {
            commands::cluster::run(args.command).await?;
        }
    }

    Ok(())
}

/// The absolute path an output option names, its directory created
/// ([`pmetal_core::paths::output_path`]).
fn validate_output_path(path: &str, context: &str) -> anyhow::Result<PathBuf> {
    Ok(pmetal_core::paths::output_path(path, context)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_interrupt_exits_130_and_any_other_failure_1() {
        assert_eq!(exit_status(&Interrupted.into()), 130);
        assert_eq!(exit_status(&anyhow::anyhow!("boom")), 1);
        // Wrapped in context, an interrupt is still one.
        let wrapped = anyhow::Error::from(Interrupted).context("generating");
        assert_eq!(exit_status(&wrapped), 130);
    }
}
