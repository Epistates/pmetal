use std::path::{Path, PathBuf};

fn parse_loss(loss: &str) -> anyhow::Result<pmetal_trainer::embedding_trainer::EmbeddingLossType> {
    use pmetal_trainer::embedding_trainer::EmbeddingLossType;
    Ok(match loss {
        "info_nce" | "infonce" => EmbeddingLossType::InfoNce,
        "mnrl" | "multiple_negatives" => EmbeddingLossType::Mnrl,
        "triplet" => EmbeddingLossType::Triplet,
        "cosent" => EmbeddingLossType::CoSent,
        "cosine" | "cosine_similarity" => EmbeddingLossType::CosineSimilarity,
        other => anyhow::bail!(
            "Unknown loss type '{other}'. Choose one of: info_nce, mnrl, triplet, cosent, cosine_similarity"
        ),
    })
}

/// A pooling mode by name; `auto` is the model's own (or mean without one).
fn parse_pooling(
    pooling: &str,
    model: Option<pmetal_models::pooling::PoolingMode>,
) -> anyhow::Result<pmetal_models::pooling::PoolingMode> {
    use pmetal_models::pooling::PoolingMode;
    Ok(match pooling {
        "auto" => model.unwrap_or(PoolingMode::Mean),
        "mean" => PoolingMode::Mean,
        "cls" => PoolingMode::Cls,
        "max" => PoolingMode::Max,
        "last_token" | "last" => PoolingMode::LastToken,
        "weighted_mean" | "weighted" => PoolingMode::WeightedMean,
        other => anyhow::bail!(
            "Unknown pooling mode '{other}'. Choose one of: auto, mean, cls, max, last_token, weighted_mean"
        ),
    })
}

/// Copy `from` into `to`, a file or a directory with everything under it.
fn copy_recursive(from: &Path, to: &Path) -> std::io::Result<()> {
    if from.is_dir() {
        std::fs::create_dir_all(to)?;
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            copy_recursive(&entry.path(), &to.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(from, to).map(|_| ())
    }
}

/// Run embedding / sentence-transformer training: the whole encoder, or LoRA
/// adapters on it with `lora = Some((rank, alpha))`.
///
/// A sentence-transformers directory sets the defaults: the encoder is where
/// its `modules.json` says, the pooling is the one it was trained with
/// (`--pooling auto`) and the sequence length is capped at its
/// `max_seq_length`; a full fine-tune writes the whole pipeline back out.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_embed_train(
    model_id: &str,
    dataset_path: &str,
    output_dir: &str,
    loss_str: &str,
    pooling_str: &str,
    temperature: f32,
    margin: f32,
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
    max_seq_len: usize,
    weight_decay: f64,
    normalize: bool,
    log_every: usize,
    seed: u64,
    lora: Option<(usize, f32)>,
) -> anyhow::Result<()> {
    use pmetal_data::EmbeddingDataset;
    use pmetal_models::architectures::encoder::EncoderModel;
    use pmetal_models::pooling::SentenceTransformerConfig;
    use pmetal_trainer::embedding_trainer::{EmbeddingTrainer, EmbeddingTrainerConfig};

    // `seed` also orders the batches; this covers MLX's own key (the LoRA
    // initialisation).
    pmetal_runtime::compat::random::seed(seed);

    // A Hub id is downloaded, with the sentence-transformers files that say
    // how the encoder is pooled (`modules.json`, `1_Pooling/config.json`); a
    // path is used as it is.
    let resolved = pmetal_hub::resolve_model_path(model_id, None, None).await?;
    let model_path = resolved
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Non-UTF-8 model path: {}", resolved.display()))?;

    let loss_type = parse_loss(loss_str)?;
    let model_dir = Path::new(model_path);
    let st = SentenceTransformerConfig::from_dir(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to read the sentence-transformers config: {e}"))?;
    let pooling_mode = parse_pooling(pooling_str, st.as_ref().map(|c| c.pooling))?;
    let encoder_dir: PathBuf = match &st {
        Some(c) if !c.transformer_path.is_empty() => model_dir.join(&c.transformer_path),
        _ => model_dir.to_path_buf(),
    };
    let max_seq_len = match st.as_ref().and_then(|c| c.max_seq_length) {
        Some(model_max) if model_max < max_seq_len => {
            tracing::info!(
                "Capping --max-seq-len {max_seq_len} at the model's max_seq_length {model_max}"
            );
            model_max
        }
        _ => max_seq_len,
    };

    tracing::info!("Loading tokenizer from '{}'", model_path);
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(model_path)
        .or_else(|_| pmetal_data::Tokenizer::from_model_dir(&encoder_dir))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    tracing::info!("Loading dataset from '{}'", dataset_path);
    let dataset = EmbeddingDataset::from_jsonl(dataset_path)
        .map_err(|e| anyhow::anyhow!("Failed to load dataset: {}", e))?;
    tracing::info!(
        "Loaded {} examples ({})",
        dataset.len(),
        match &dataset {
            EmbeddingDataset::Pairs(_) => "pairs",
            EmbeddingDataset::Triplets(_) => "triplets",
        }
    );

    let config = EmbeddingTrainerConfig {
        training: pmetal_core::TrainingConfig {
            learning_rate,
            weight_decay,
            batch_size,
            num_epochs: epochs,
            max_seq_len,
            output_dir: output_dir.to_string(),
            ..Default::default()
        },
        loss_type,
        temperature,
        margin,
        pooling_mode,
        normalize,
        max_seq_len,
        log_every,
        seed,
        shuffle: true,
    };
    let mut trainer = EmbeddingTrainer::new(config)?;
    // A request to stop (Ctrl-C, SIGTERM) ends the run after the step in flight;
    // what is trained so far is saved below, and the command exits as interrupted.
    let stop = super::stop::StopFlag::watch();
    trainer.add_callback(Box::new(stop.clone()));
    tracing::info!(
        "Starting embedding training: loss={loss_str}, pooling={pooling_mode:?}, T={temperature}, \
         batch={batch_size}, epochs={epochs}, {}",
        if lora.is_some() {
            "LoRA"
        } else {
            "full fine-tune"
        }
    );

    let output = Path::new(output_dir);
    std::fs::create_dir_all(output)
        .map_err(|e| anyhow::anyhow!("Failed to create output dir: {}", e))?;
    super::record_training(
        output,
        model_id,
        &resolved,
        dataset_path,
        if lora.is_some() { "lora" } else { "full" },
    );

    if let Some((rank, alpha)) = lora {
        let lora_config = pmetal_core::LoraConfig {
            r: rank,
            alpha,
            ..Default::default()
        };
        let base = pmetal_models::DynamicModel::load(&encoder_dir)
            .map_err(|e| anyhow::anyhow!("Failed to load the encoder: {}", e))?;
        let mut model = pmetal_lora::AdaptedModel::attach(base, lora_config.clone())?;
        trainer.run(&mut model, &tokenizer, &dataset)?;
        tracing::info!("Saving the trained adapters to '{}'", output_dir);
        pmetal_trainer::orchestrator::save_adapter(&model, output, &lora_config, Some(model_id))?;
        return stop.finish(Ok(()));
    }

    tracing::info!("Loading the encoder from '{}'", encoder_dir.display());
    let config_str = std::fs::read_to_string(encoder_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let mut model = EncoderModel::from_config_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to build the encoder: {}", e))?;
    let weights = pmetal_models::loader::load_weights(&encoder_dir)
        .map_err(|e| anyhow::anyhow!("Failed to read weights: {}", e))?;
    pmetal_models::loader::load_encoder_weights(&mut model, weights)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;

    trainer.run(&mut model, &tokenizer, &dataset)?;

    // The encoder under the checkpoint's own tensor names, so
    // sentence-transformers and `pmetal` load what this writes.
    tracing::info!("Saving trained model to '{}'", output_dir);
    let encoder_out = match &st {
        Some(c) if !c.transformer_path.is_empty() => output.join(&c.transformer_path),
        _ => output.to_path_buf(),
    };
    std::fs::create_dir_all(&encoder_out)
        .map_err(|e| anyhow::anyhow!("Failed to create output dir: {}", e))?;
    let weights_path = encoder_out.join("model.safetensors");
    let weights_str = weights_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Non-UTF-8 output path: {}", weights_path.display()))?;
    let tensors = pmetal_models::loader::encoder_checkpoint_tensors(&model)?;
    let entries: Vec<(&str, &pmetal_runtime::compat::Array)> = tensors
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    pmetal_runtime::compat::Array::save_safetensors(weights_str, &entries)?;

    // Everything else a loader needs to read the model back: the encoder's
    // configuration and tokenizer, and, for a sentence-transformers model,
    // the pipeline that pools it (`modules.json`, the pooling and normalize
    // modules, `sentence_bert_config.json`).
    for file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        for (from_dir, to_dir) in [
            (&encoder_dir, &encoder_out),
            (&model_dir.to_path_buf(), &output.to_path_buf()),
        ] {
            let src = from_dir.join(file);
            if src.exists() {
                copy_recursive(&src, &to_dir.join(file))?;
            }
        }
    }
    if st.is_some() {
        for name in [
            "modules.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
        ] {
            let src = model_dir.join(name);
            if src.exists() {
                copy_recursive(&src, &output.join(name))?;
            }
        }
        let modules: serde_json::Value = std::fs::read_to_string(model_dir.join("modules.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        for module in modules.as_array().into_iter().flatten() {
            let path = module["path"].as_str().unwrap_or_default();
            let transformer = st
                .as_ref()
                .map(|c| c.transformer_path.as_str())
                .unwrap_or_default();
            if !path.is_empty() && path != transformer && model_dir.join(path).exists() {
                copy_recursive(&model_dir.join(path), &output.join(path))?;
            }
        }
    }

    tracing::info!(
        "Embedding training complete. Model saved to '{}'",
        output_dir
    );
    stop.finish(Ok(()))
}
