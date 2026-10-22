use pmetal_core::jobs::ServeSpec;

/// The server configuration a spec's access options make: its host, port,
/// CORS origins and request limit, with `api_keys` (`--api-key`) and the keys of
/// `PMETAL_API_KEY` (comma-separated) as the keys clients must send.
#[cfg(feature = "serve")]
fn access_config(spec: &ServeSpec, mut api_keys: Vec<String>) -> pmetal_serve::ServeConfig {
    if let Ok(from_env) = std::env::var("PMETAL_API_KEY") {
        api_keys.extend(
            from_env
                .split(',')
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .map(str::to_owned),
        );
    }
    pmetal_serve::ServeConfig {
        port: spec.port,
        host: spec.host.clone(),
        api_keys,
        cors_allowed_origins: spec.cors_origin.clone(),
        request_timeout: (spec.request_timeout > 0)
            .then(|| std::time::Duration::from_secs(spec.request_timeout)),
        ..Default::default()
    }
}

/// Start the OpenAI-compatible inference server.
#[cfg(feature = "serve")]
pub(crate) async fn run_serve(spec: ServeSpec, api_keys: Vec<String>) -> anyhow::Result<()> {
    let ServeSpec {
        model: ref model_id,
        max_seq_len,
        ref experts_dir,
        fp8,
        kv_quant,
        no_kv_quant,
        kv_group_size,
        kv_turboquant,
        ref kv_turboquant_preset,
        ane: ane_enabled,
        ane_max_seq_len,
        ref draft_model,
        mtp,
        ref mtp_model,
        draft_tokens,
        continuous_batch,
        cb_max_slots,
        cb_max_queue_depth,
        cb_block_size,
        cb_max_blocks,
        cb_exact,
        ..
    } = spec;
    let model_id = model_id.clone();
    let experts_dir = experts_dir.clone();
    let kv_turboquant_preset = kv_turboquant_preset.clone();
    let draft_model = draft_model.clone();
    let mtp_model = mtp_model.clone();
    #[cfg(not(feature = "ane"))]
    if ane_enabled {
        anyhow::bail!("--ane needs the 'ane' feature: cargo build --features ane");
    }
    use pmetal::inference_runner::{CacheModeRequest, TurboQuantPreset};
    use pmetal_models::dispatcher::DynamicModel;
    use pmetal_serve::{BatcherConfig, InferenceEngine, ServeConfig, SpeculatorSpec};

    // Resolve model path
    tracing::info!("Resolving model: {}", model_id);
    let model_path = pmetal_hub::resolve_model_path(&model_id, None, None).await?;
    if pmetal_models::decision::is_decision_model(&model_path) {
        return run_decision_serve(model_id, &model_path, access_config(&spec, api_keys)).await;
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
    // A draft model drafts for the ANE engine with --ane (a DFlash draft
    // model, run on the GPU), and for the GPU engine without it.
    let draft_path = match draft_model {
        Some(draft) => {
            let path = pmetal_hub::resolve_model_path(&draft, None, None).await?;
            if ane_enabled && !pmetal_models::dflash_drafter::is_dflash_draft(&path) {
                anyhow::bail!(
                    "with --ane, --draft-model takes a DFlash draft model, which drafts for the \
                     ANE engine; {draft} isn't one"
                );
            }
            Some(path)
        }
        None => None,
    };
    let speculation = if mtp {
        let checkpoint = match &mtp_model {
            Some(head) => pmetal_hub::resolve_model_path(head, None, None).await?,
            None => model_path.clone(),
        };
        Some(SpeculatorSpec::QwenMtp {
            checkpoint,
            draft_tokens: draft_tokens.unwrap_or(3),
        })
    } else {
        draft_path
            .clone()
            .filter(|_| !ane_enabled)
            .map(|path| SpeculatorSpec::Draft {
                path,
                tokens: draft_tokens,
            })
    };

    // Load tokenizer — use pmetal_data::Tokenizer for config-aware special token
    // resolution (needed by collect_all_stop_tokens inside InferenceEngine::new).
    tracing::info!("Loading tokenizer...");
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&model_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let kv_turboquant_preset = kv_turboquant_preset
        .as_deref()
        .map(|name| name.parse::<TurboQuantPreset>())
        .transpose()
        .map_err(|e| anyhow::anyhow!("--kv-turboquant-preset: {e}"))?;
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
    // Each request's cache is chosen on the model thread by the policy
    // `pmetal infer` uses, from the model's own cache layout.
    let engine = engine.with_cache_request(cache_mode_request);
    let engine = match draft_path.filter(|_| ane_enabled) {
        Some(path) => engine.with_ane_drafter(path),
        None => engine,
    };
    if let Some(spec) = speculation {
        // Loaded and checked against the model on its own thread: a drafter
        // that does not fit it is an error here, not on the first request.
        let info = engine.enable_speculation(spec)?;
        tracing::info!(
            drafter = info.kind,
            draft_tokens = info.draft_tokens,
            "Speculative decoding on; a request's tokens are the model's own, sooner"
        );
    }

    // Start server
    let continuous_batching = if continuous_batch {
        Some(BatcherConfig {
            max_slots: cb_max_slots.max(1),
            max_queue_depth: cb_max_queue_depth.max(1),
            block_size: cb_block_size.max(1),
            max_blocks: cb_max_blocks,
            per_slot_decode: cb_exact,
            ..BatcherConfig::default()
        })
    } else {
        None
    };

    let config = ServeConfig {
        continuous_batching,
        ..access_config(&spec, api_keys)
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
    config: pmetal_serve::ServeConfig,
) -> anyhow::Result<()> {
    use pmetal_models::decision::DEFAULT_MAX_LENGTH;
    use pmetal_serve::DecisionEngine;

    tracing::info!("Loading decision model from {:?}...", model_path);
    let started = std::time::Instant::now();
    let engine = DecisionEngine::load(model_path.to_path_buf(), model_id, DEFAULT_MAX_LENGTH)?;
    tracing::info!(
        "Decision model loaded in {:.1}s; generation endpoints are off, POST /v1/systemone answers",
        started.elapsed().as_secs_f64()
    );
    pmetal_serve::decision::run_decision_server(engine, config).await
}
