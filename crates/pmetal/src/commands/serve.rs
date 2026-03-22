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
) -> anyhow::Result<()> {
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
    let mut model = DynamicModel::load(&model_path)?;

    // Enable expert offloading if a packed experts directory is provided
    if let Some(ref experts_dir) = experts_dir {
        model.enable_expert_offloading(std::path::Path::new(experts_dir))?;
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

    // Create inference engine
    let engine =
        InferenceEngine::new(model, tokenizer, model_id.clone(), &model_path, max_seq_len)?;

    // Start server
    let config = ServeConfig {
        port,
        host,
        ..Default::default()
    };

    pmetal_serve::server::run_server(engine, config).await?;

    Ok(())
}
