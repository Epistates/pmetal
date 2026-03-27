use std::path::PathBuf;

/// Start the OpenAI-compatible inference server.
#[cfg(feature = "serve")]
pub(crate) async fn run_serve(
    model_id: String,
    lora_path: Option<String>,
    port: u16,
    host: String,
    max_seq_len: usize,
    experts_dir: Option<String>,
    kv_turboquant: bool,
    kv_turboquant_preset: Option<String>,
    ane_enabled: bool,
    ane_max_seq_len: usize,
    ane_real_time: bool,
) -> anyhow::Result<()> {
    use pmetal_mlx::CacheMode;
    use pmetal_mlx::kv_cache::turboquant::TurboQuantConfig;
    use pmetal_models::dispatcher::DynamicModel;
    use pmetal_serve::{InferenceEngine, ServeConfig};

    // Resolve model path
    let model_path = if model_id.contains('/') && !PathBuf::from(&model_id).exists() {
        tracing::info!("Downloading model from HuggingFace: {}", model_id);
        pmetal_hub::download_model(&model_id, None, None).await?
    } else {
        PathBuf::from(&model_id)
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

    // Enable expert offloading if a packed experts directory is provided
    if let Some(ref experts_dir) = experts_dir {
        model.enable_expert_offloading(std::path::Path::new(experts_dir))?;
    } else if model.requires_expert_offloading() {
        anyhow::bail!(
            "this model requires expert offloading; repack routed experts with `pmetal pack-experts` and pass --experts-dir <packed_dir>"
        );
    }

    // Apply LoRA adapter if specified
    if let Some(ref _lora) = lora_path {
        // TODO: DynamicModel does not yet support runtime LoRA application.
        // For now, merge the adapter into the base model first:
        //   pmetal merge --base <model> --lora <adapter> --output <merged>
        anyhow::bail!(
            "Serving with a LoRA adapter requires pre-merging. \
             Use `pmetal merge --base <model> --lora <adapter> --output <merged>` \
             then serve the merged model."
        );
    }

    tracing::info!("Model loaded successfully");

    // Resolve KV cache mode override
    let cache_mode_override = if kv_turboquant || kv_turboquant_preset.is_some() {
        let config = match kv_turboquant_preset.as_deref() {
            Some("q2_5") => {
                tracing::info!("TurboQuant KV cache: q2.5 preset (2.5 bits, ~6.4x compression)");
                TurboQuantConfig::preset_q2_5(128) // head_dim refined by dispatcher sanitization
            }
            Some("q3_5") => {
                tracing::info!(
                    "TurboQuant KV cache: q3.5 preset (3.5 bits, ~4.6x compression, near-lossless)"
                );
                TurboQuantConfig::preset_q3_5(128)
            }
            _ => {
                tracing::info!("TurboQuant KV cache: uniform 4-bit keys, 4-bit values");
                TurboQuantConfig::uniform(4, 4)
            }
        };
        Some(CacheMode::TurboQuant { config })
    } else {
        None
    };

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

    // Start server
    let config = ServeConfig {
        port,
        host,
        ..Default::default()
    };

    pmetal_serve::server::run_server(engine, config).await?;

    Ok(())
}
