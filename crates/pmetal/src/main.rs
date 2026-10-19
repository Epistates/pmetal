//! PMetal CLI - LLM fine-tuning for Apple Silicon.

pub mod cli;
mod commands;
mod dashboard;
#[cfg(any(feature = "models", feature = "native-only"))]
pub mod native_inference;
mod pack_experts;
#[cfg(feature = "dashboard")]
mod tui;

use std::path::{Path, PathBuf};

use clap::Parser;
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
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // The TUI owns the terminal, so it logs to the file only.
    #[cfg(feature = "dashboard")]
    let is_tui = matches!(cli.command, Commands::Tui { .. });
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
    tokio_main(cli)
}

// ---------------------------------------------------------------------------
// Persistent logging
// ---------------------------------------------------------------------------

/// Initialize logging with file output to `~/.cache/pmetal/logs/{component}.log`.
///
/// - **CLI mode**: logs to both stderr and file
/// - **TUI mode**: logs to file only (stderr would corrupt the terminal), or
///   to `PMETAL_LOG_FILE` if set for backwards compatibility
///
/// The previous log is rotated to `{component}.log.1`.
fn init_logging(component: &str, suppress_stderr: bool) {
    // Determine log file path
    let log_path = if let Ok(custom) = std::env::var("PMETAL_LOG_FILE") {
        std::path::PathBuf::from(custom)
    } else {
        let dir = pmetal_log_dir();
        let path = dir.join(format!("{component}.log"));

        // Rotate: keep one previous log
        let prev = dir.join(format!("{component}.log.1"));
        if path.exists() {
            let _ = std::fs::rename(&path, &prev);
        }
        path
    };

    // hf-hub's Xet transfer client logs every request at INFO, several lines
    // per downloaded file. Hold it (`xet_*` and `hf_xet`) to warnings; a
    // more specific `RUST_LOG=xet_client=info` still wins.
    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into())
        .add_directive("xet=warn".parse().expect("static directive"))
        .add_directive("hf_xet=warn".parse().expect("static directive"));

    if suppress_stderr {
        // TUI: file only (no stderr to avoid corrupting terminal)
        if let Ok(file) = std::fs::File::create(&log_path) {
            tracing_subscriber::fmt()
                .with_writer(std::sync::Mutex::new(file))
                .with_env_filter(filter)
                .with_ansi(false)
                .init();
        }
    } else {
        // CLI: stderr + tee to file
        let file = std::fs::File::create(&log_path).ok();
        let file = file.map(std::sync::Mutex::new).map(std::sync::Arc::new);
        let file_clone = file.clone();

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(move || TeeWriter {
                stderr: std::io::stderr(),
                file: file_clone.clone(),
            })
            .init();
    }
}

fn pmetal_log_dir() -> std::path::PathBuf {
    let dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("pmetal")
        .join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
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

#[tokio::main]
async fn tokio_main(cli: Cli) -> anyhow::Result<()> {
    // Capture global flags before `cli` is moved into the `match` below.
    #[cfg(feature = "trainer")]
    let log_events_path = cli.log_events.clone();

    match cli.command {
        #[cfg(feature = "trainer")]
        Commands::Train(args) => {
            commands::train::run_train(args, log_events_path.as_deref()).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainMtp(args) => {
            commands::train_mtp::run_train_mtp(args).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainDraft(args) => {
            commands::train_draft::run_train_draft(args).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainDiffusion(args) => {
            use pmetal_core::jobs::TrainDiffusionSpec;
            let mut spec = TrainDiffusionSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<TrainDiffusionSpec>(&errors))?;
            commands::train_diffusion::run_train_diffusion(spec).await?;
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
            commands::bench::generate_sample_config(&validated_output.to_string_lossy())?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchFfi => {
            commands::bench::run_ffi_benchmark()?;
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
            let crate::cli::quantize::QuantizeArgs {
                model,
                output,
                imatrix,
                method,
                lora,
                kl_calibrate,
                target_bpw,
                kl_threshold,
                format,
                bits,
                group_size,
            } = args;
            if format == "mlx" {
                // MLX safetensors path — no LoRA fusion support in this path yet.
                if lora.is_some() {
                    anyhow::bail!(
                        "--lora is not yet supported with --format mlx; \
                         fuse the adapter first with `pmetal fuse` then quantize"
                    );
                }
                commands::quantize::run_quantization_mlx(
                    &model, &output, bits, group_size, target_bpw,
                )
                .await?;
            } else {
                #[cfg(feature = "lora")]
                if let Some(lora_path) = &lora {
                    // Fuse LoRA first, then quantize the fused model
                    let fused_dir = format!("{output}.fused_tmp");
                    commands::fuse::run_fuse(pmetal_core::jobs::FuseSpec {
                        model: model.clone(),
                        lora: lora_path.clone(),
                        output: fused_dir.clone(),
                        ..Default::default()
                    })
                    .await?;
                    commands::quantize::run_quantization(
                        &fused_dir,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                    // Clean up temp dir
                    let _ = std::fs::remove_dir_all(&fused_dir);
                } else {
                    commands::quantize::run_quantization(
                        &model,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                }

                #[cfg(not(feature = "lora"))]
                {
                    if lora.is_some() {
                        anyhow::bail!(
                            "LoRA fusion during quantization requires the `lora` feature"
                        );
                    }
                    commands::quantize::run_quantization(
                        &model,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                }
            }
        }

        #[cfg(feature = "trainer")]
        Commands::Distill(args) => {
            use pmetal_core::jobs::DistillSpec;
            let mut spec = DistillSpec::from(args);
            spec.normalize()
                .map_err(|errors| commands::invalid_spec::<DistillSpec>(&errors))?;
            let job_id = format!("distill:{}", spec.student);
            commands::with_event_log(log_events_path.as_deref(), &job_id, |ctx| {
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
            commands::with_event_log(log_events_path.as_deref(), &job_id, |ctx| {
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
            commands::with_event_log(log_events_path.as_deref(), &job_id, |ctx| {
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
            commands::with_event_log(log_events_path.as_deref(), &job_id, |ctx| {
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

        Commands::Dashboard { metrics_file } => {
            let path = metrics_file.map(std::path::PathBuf::from);
            dashboard::run_dashboard(path)?;
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
            let crate::cli::eval::EvalArgs {
                model,
                dataset,
                lora,
                max_seq_len,
                num_samples,
                json,
            } = args;
            commands::eval::run_eval(
                &model,
                &dataset,
                lora.as_deref(),
                max_seq_len,
                num_samples,
                json,
            )
            .await?;
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
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Pretrain(args) => {
            let crate::cli::pretrain::PretrainArgs {
                arch,
                shards,
                eval_shards,
                seq_len,
                batch_size,
                steps,
                learning_rate,
                min_lr,
                warmup_steps,
                lr_schedule,
                weight_decay,
                max_grad_norm,
                eos_token_id,
                output,
                checkpoint_every,
                resume,
                model_config: _model_config,
                z_loss,
                document_mask,
                gradient_accumulation_steps,
                log_every,
                eval_every,
                eval_batches,
                seed,
            } = args;
            use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
            use pmetal_runtime::compat::random;
            use pmetal_trainer::pretrain::{self, PretrainConfig};

            random::seed(seed);
            let _output_dir = validate_output_path(&output, "pretrain output")?;

            // Collect shard paths (each CLI value is a literal path)
            let shard_paths: Vec<std::path::PathBuf> =
                shards.iter().map(std::path::PathBuf::from).collect();
            anyhow::ensure!(!shard_paths.is_empty(), "no shard files specified");
            println!(
                "Pretraining on {} shards, {} steps",
                shard_paths.len(),
                steps
            );

            // Parse LR schedule
            let lr_sched = lr_schedule
                .parse::<pmetal_core::LrSchedulerType>()
                .map_err(|e| anyhow::anyhow!("--lr-schedule: {e}"))?;

            // Build model via factory (supports llama, qwen2, qwen3, gemma, gemma4, mistral, phi, gpt-oss)
            use pmetal_runtime::compat::module::ModuleParameters;
            let config_path = _model_config.as_deref().map(std::path::Path::new);
            let mut model = pretrain::create_model(&arch, config_path)?;
            let n_layers = pretrain::n_layers(&model);
            println!(
                "Architecture: {arch}, layers: {n_layers}, params: {}",
                model.num_parameters()
            );

            let config = PretrainConfig {
                num_steps: steps,
                learning_rate,
                min_lr,
                warmup_steps,
                lr_schedule: lr_sched,
                weight_decay,
                betas: (0.9, 0.95),
                eps: 1e-8,
                max_grad_norm: if max_grad_norm > 0.0 {
                    Some(max_grad_norm)
                } else {
                    None
                },
                ignore_index: None,
                z_loss_coef: if z_loss > 0.0 { Some(z_loss) } else { None },
                n_layers,
                apply_init: resume.is_none(),
                checkpoint_every: if checkpoint_every > 0 {
                    Some(checkpoint_every)
                } else {
                    None
                },
                checkpoint_dir: Some(std::path::PathBuf::from(&output)),
                gradient_accumulation_steps,
                log_every,
                eval_every,
                eval_batches,
            };

            let mut optimizer =
                pmetal_runtime::compat::optimizers::AdamWBuilder::new(learning_rate)
                    .weight_decay(weight_decay)
                    .betas((0.9, 0.95))
                    .eps(1e-8)
                    .build()?;
            let mut start_step = 0usize;
            let mut resume_position = None;

            // Resume from checkpoint if requested
            if let Some(ref ckpt_dir) = resume {
                let meta = pretrain::load_checkpoint(
                    std::path::Path::new(ckpt_dir),
                    &mut model,
                    &mut optimizer,
                )?;
                start_step = meta.step as usize;
                resume_position = meta.stream_position;
                println!("Resumed from step {}, loss {:.4}", meta.step, meta.loss);
            }

            // Streaming data pipeline
            let stream_config = StreamConfig {
                shard_paths,
                seq_len,
                batch_size,
                eos_token_id,
                resume_from: resume_position,
            };
            let mut reader = StreamingShardReader::new(stream_config)?;
            if resume.is_some() && resume_position.is_none() && start_step > 0 {
                let batches_to_skip = start_step.saturating_mul(gradient_accumulation_steps.max(1));
                for _ in 0..batches_to_skip {
                    reader.try_next_batch()?.ok_or_else(|| {
                        anyhow::anyhow!("stream exhausted while seeking resume step")
                    })?;
                }
            }

            // Convert streaming batches to MLX arrays. A shard that cannot be
            // read is an error that stops the run, not the end of the data.
            let batch_iter = std::iter::from_fn(move || {
                let (rows, pos) = match reader.try_next_rows() {
                    Ok(Some(batch)) => batch,
                    Ok(None) => return None,
                    Err(e) => return Some(Err(pretrain::PretrainError::Data(e.to_string()))),
                };
                let flat: Vec<i32> = rows
                    .iter()
                    .flat_map(|row| row.tokens.iter().map(|&t| t as i32))
                    .collect();
                let tokens = pmetal_runtime::InlineArray::from_i32_slice_shaped(
                    &flat,
                    &[rows.len() as i32, seq_len as i32],
                );
                Some(Ok(pretrain::PretrainBatch {
                    tokens,
                    position: Some(pos),
                    doc_lengths: document_mask
                        .then(|| rows.into_iter().map(|row| row.doc_lengths).collect()),
                }))
            });

            let eval_iter = if !eval_shards.is_empty() && eval_every > 0 {
                let eval_paths: Vec<std::path::PathBuf> =
                    eval_shards.iter().map(std::path::PathBuf::from).collect();
                let eval_reader = StreamingShardReader::new(StreamConfig {
                    shard_paths: eval_paths,
                    seq_len,
                    batch_size,
                    eos_token_id,
                    resume_from: None,
                })?;
                Some(eval_reader.map(move |(batch, _pos)| {
                    let flat: Vec<i32> = batch
                        .iter()
                        .flat_map(|seq| seq.iter().map(|&t| t as i32))
                        .collect();
                    pmetal_runtime::InlineArray::from_i32_slice_shaped(
                        &flat,
                        &[batch.len() as i32, seq_len as i32],
                    )
                }))
            } else {
                None
            };

            // Run
            let losses = pretrain::run_pretrain_with_state(
                &mut model, &config, batch_iter, optimizer, start_step, eval_iter,
            )?;

            // Print summary
            if !losses.is_empty() {
                let last = losses.last().copied().unwrap_or(0.0);
                println!(
                    "Pretraining complete: {} steps, final loss {:.4}",
                    losses.len(),
                    last,
                );
            }
        }

        #[cfg(feature = "distributed")]
        Commands::Cluster(args) => {
            commands::cluster::run(args.command).await?;
        }
    }

    Ok(())
}

fn validate_output_path(path: &str, context: &str) -> anyhow::Result<PathBuf> {
    use std::path::{Component, PathBuf};

    let path = PathBuf::from(path);

    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            anyhow::bail!(
                "Path traversal detected in {}: '{}' contains '..' component. Please use a safe path.",
                context,
                path.display()
            );
        }
    }

    let cwd = std::env::current_dir()?;
    let resolved = if path.is_absolute() {
        path.clone()
    } else {
        cwd.join(&path)
    };

    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let canonical = if resolved.exists() {
        resolved.canonicalize()?
    } else if let Some(parent) = resolved.parent() {
        let canonical_parent = parent.canonicalize()?;
        if let Some(filename) = resolved.file_name() {
            canonical_parent.join(filename)
        } else {
            canonical_parent
        }
    } else {
        resolved
    };

    let cwd_canonical = cwd.canonicalize()?;
    let home_dir = dirs::home_dir();
    let temp_dir = std::env::temp_dir().canonicalize().ok();

    let is_safe = canonical.starts_with(&cwd_canonical)
        || home_dir
            .as_ref()
            .map(|h| canonical.starts_with(h))
            .unwrap_or(false)
        || temp_dir
            .as_ref()
            .map(|t| canonical.starts_with(t))
            .unwrap_or(false)
        || canonical.starts_with("/tmp")
        || canonical.starts_with("/private/tmp");

    if !is_safe {
        anyhow::bail!(
            "Unsafe output path for {}: '{}' resolves to '{}'",
            context,
            path.display(),
            canonical.display()
        );
    }

    Ok(canonical)
}
