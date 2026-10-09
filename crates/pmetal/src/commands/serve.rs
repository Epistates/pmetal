/// Start the OpenAI-compatible inference server.
#[cfg(feature = "serve")]
#[expect(clippy::too_many_arguments, reason = "one argument per CLI flag")]
pub(crate) async fn run_serve(
    model_id: String,
    port: u16,
    host: String,
    max_seq_len: Option<usize>,
    experts_dir: Option<String>,
    fp8: bool,
    kv_quant: Option<u8>,
    no_kv_quant: bool,
    kv_group_size: usize,
    kv_turboquant: bool,
    kv_turboquant_preset: Option<String>,
    ane_enabled: bool,
    ane_max_seq_len: usize,
    draft_model: Option<String>,
    continuous_batch: bool,
    cb_max_slots: usize,
    cb_max_queue_depth: usize,
    cb_block_size: usize,
    cb_max_blocks: usize,
) -> anyhow::Result<()> {
    use pmetal::inference_runner::{
        CacheModeRequest, TurboQuantPreset, explicit_cache_mode_override,
    };
    use pmetal_models::dispatcher::DynamicModel;
    use pmetal_serve::{BatcherConfig, InferenceEngine, ServeConfig};

    // Resolve model path
    tracing::info!("Resolving model: {}", model_id);
    let model_path = pmetal_hub::resolve_model_path(&model_id, None, None).await?;
    if pmetal_models::decision::is_decision_model(&model_path) {
        return run_decision_serve(model_id, &model_path, port, host).await;
    }
    let max_seq_len = match max_seq_len {
        Some(tokens) => tokens,
        None => {
            // One full-length cache per continuous-batching slot.
            let sequences = if continuous_batch {
                cb_max_slots.max(1)
            } else {
                1
            };
            let default =
                pmetal::inference_runner::default_serve_context_len(&model_path, sequences);
            tracing::info!(
                tokens = default.tokens,
                context_window = ?default.context_window,
                memory_cap = ?default.memory_cap,
                "Context length (--max-seq-len) from the model"
            );
            default.tokens
        }
    };
    let draft_path = match draft_model {
        Some(draft) => {
            let path = pmetal_hub::resolve_model_path(&draft, None, None).await?;
            if !ane_enabled || !pmetal_models::dflash_drafter::is_dflash_draft(&path) {
                anyhow::bail!(
                    "--draft-model takes a DFlash draft model, which drafts for the ANE \
                     engine with --ane; {draft} isn't one, or --ane is off"
                );
            }
            Some(path)
        }
        None => None,
    };

    // Load tokenizer — use pmetal_data::Tokenizer for config-aware special token
    // resolution (needed by collect_all_stop_tokens inside InferenceEngine::new).
    tracing::info!("Loading tokenizer...");
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&model_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let kv_turboquant_preset = match kv_turboquant_preset.as_deref() {
        Some("q2_5") => Some(TurboQuantPreset::Q2_5),
        Some("q3_5") => Some(TurboQuantPreset::Q3_5),
        Some(other) => {
            anyhow::bail!("unsupported TurboQuant preset `{other}`");
        }
        None => None,
    };
    let cache_mode_request = CacheModeRequest {
        kv_quant,
        kv_k_bits: None,
        kv_v_bits: None,
        kv_group_size,
        kv_turboquant,
        kv_turboquant_preset,
        no_kv_quant,
        fp8,
    };

    // The model loads on the engine's own thread, where every request will
    // run: arrays it leaves unevaluated can't be evaluated anywhere else.
    tracing::info!("Loading model from {:?}...", model_path);
    let (cache_mode_tx, cache_mode_rx) = std::sync::mpsc::channel();
    let load_path = model_path.clone();
    let load = move || -> anyhow::Result<DynamicModel> {
        let mut model = DynamicModel::load_with_options(
            &load_path,
            pmetal_models::dispatcher::DynamicModelLoadOptions {
                prefer_expert_offload: experts_dir.is_some(),
            },
        )?;

        // Quantize to FP8 if requested
        if fp8 {
            tracing::info!("Quantizing model weights to FP8 E4M3...");
            model.quantize_fp8()?;
        }

        // Enable expert offloading if a packed experts directory is provided
        if let Some(ref experts_dir) = experts_dir {
            model.enable_expert_offloading(std::path::Path::new(experts_dir))?;
        } else if model.requires_expert_offloading() {
            anyhow::bail!(
                "this model requires expert offloading; repack routed experts with `pmetal pack-experts` and pass --experts-dir <packed_dir>"
            );
        }

        // Resolve the KV cache mode override from the model's own cache
        // configuration, as CLI/GUI inference does, so dense and MoE models
        // share one TurboQuant/KV selection policy.
        let base_cache = model.create_cache(max_seq_len);
        let _ = cache_mode_tx.send(explicit_cache_mode_override(
            base_cache.config(),
            cache_mode_request,
        ));
        Ok(model)
    };

    let engine = InferenceEngine::new_with_backend(
        load,
        tokenizer,
        model_id.clone(),
        &model_path,
        max_seq_len,
        ane_enabled,
        ane_max_seq_len,
    )?;
    tracing::info!("Model loaded successfully");
    let engine = match cache_mode_rx.recv().ok().flatten() {
        Some(mode) => {
            tracing::info!(mode = %mode.describe(), "KV cache override");
            engine.with_cache_mode_override(mode)
        }
        None => engine,
    };
    let engine = match draft_path {
        Some(path) => engine.with_ane_drafter(path),
        None => engine,
    };

    // Start server
    let continuous_batching = if continuous_batch {
        Some(BatcherConfig {
            max_slots: cb_max_slots.max(1),
            max_queue_depth: cb_max_queue_depth.max(1),
            block_size: cb_block_size.max(1),
            max_blocks: cb_max_blocks,
        })
    } else {
        None
    };

    let config = ServeConfig {
        port,
        host,
        continuous_batching,
        ..Default::default()
    };

    pmetal_serve::server::run_server(engine, config).await?;

    Ok(())
}

/// Serve a decision model (a Clef release) on `/v1/systemone`.
///
/// A decision model answers in one forward pass and never generates, so none of
/// the generation flags (KV cache, ANE, drafting, batching) apply; prompts use
/// the release's own 16,384-token budget.
async fn run_decision_serve(
    model_id: String,
    model_path: &std::path::Path,
    port: u16,
    host: String,
) -> anyhow::Result<()> {
    use pmetal_models::decision::DEFAULT_MAX_LENGTH;
    use pmetal_serve::{DecisionEngine, ServeConfig};

    tracing::info!("Loading decision model from {:?}...", model_path);
    let started = std::time::Instant::now();
    let engine = DecisionEngine::load(model_path.to_path_buf(), model_id, DEFAULT_MAX_LENGTH)?;
    tracing::info!(
        "Decision model loaded in {:.1}s; generation endpoints are off, POST /v1/systemone answers",
        started.elapsed().as_secs_f64()
    );
    let config = ServeConfig {
        port,
        host,
        ..Default::default()
    };
    pmetal_serve::decision::run_decision_server(engine, config).await
}
