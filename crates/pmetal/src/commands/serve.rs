/// Start the OpenAI-compatible inference server.
#[cfg(feature = "serve")]
pub(crate) async fn run_serve(
    model_id: String,
    port: u16,
    host: String,
    max_seq_len: usize,
    experts_dir: Option<String>,
    fp8: bool,
    kv_quant: Option<u8>,
    no_kv_quant: bool,
    kv_group_size: usize,
    kv_turboquant: bool,
    kv_turboquant_preset: Option<String>,
    ane_enabled: bool,
    ane_max_seq_len: usize,
    ane_real_time: bool,
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

    // Load model
    tracing::info!("Loading model from {:?}...", model_path);
    let mut model = DynamicModel::load_with_options(
        &model_path,
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

    tracing::info!("Model loaded successfully");

    // Resolve KV cache mode override using the same model-derived base-cache
    // configuration path as CLI/GUI inference. This keeps dense and MoE models
    // on one canonical TurboQuant/KV selection policy.
    let kv_turboquant_preset = match kv_turboquant_preset.as_deref() {
        Some("q2_5") => Some(TurboQuantPreset::Q2_5),
        Some("q3_5") => Some(TurboQuantPreset::Q3_5),
        Some(other) => {
            anyhow::bail!("unsupported TurboQuant preset `{other}`");
        }
        None => None,
    };
    let base_cache = model.create_cache(max_seq_len);
    let cache_mode_override = explicit_cache_mode_override(
        base_cache.config(),
        CacheModeRequest {
            kv_quant,
            kv_k_bits: None,
            kv_v_bits: None,
            kv_group_size,
            kv_turboquant,
            kv_turboquant_preset,
            no_kv_quant,
            fp8,
        },
    );
    if let Some(ref mode) = cache_mode_override {
        tracing::info!(mode = %mode.describe(), "KV cache override");
    }

    // Create inference engine
    let engine = InferenceEngine::new_with_options(
        model,
        tokenizer,
        model_id.clone(),
        &model_path,
        max_seq_len,
        ane_enabled,
        ane_max_seq_len,
        ane_real_time,
        cache_mode_override,
    )?;
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
