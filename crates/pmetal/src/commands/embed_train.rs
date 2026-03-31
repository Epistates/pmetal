/// Run embedding / sentence-transformer training.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_embed_train(
    model_path: &str,
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
) -> anyhow::Result<()> {
    use pmetal_bridge::compat::optimizers::{AdamWBuilder, Optimizer};
    use pmetal_data::EmbeddingDataset;
    use pmetal_models::architectures::bert::BertForEmbedding;
    use pmetal_models::pooling::PoolingMode;
    use pmetal_trainer::embedding_trainer::{
        EmbeddingLossType, EmbeddingTrainer, EmbeddingTrainerConfig,
    };

    // Parse loss type
    let loss_type = match loss_str {
        "info_nce" | "infonce" => EmbeddingLossType::InfoNce,
        "mnrl" | "multiple_negatives" => EmbeddingLossType::Mnrl,
        "triplet" => EmbeddingLossType::Triplet,
        "cosent" => EmbeddingLossType::CoSent,
        "cosine" | "cosine_similarity" => EmbeddingLossType::CosineSimilarity,
        other => {
            anyhow::bail!(
                "Unknown loss type '{}'. Choose one of: info_nce, mnrl, triplet, cosent, cosine_similarity",
                other
            );
        }
    };

    // Parse pooling mode
    let pooling_mode = match pooling_str {
        "mean" => PoolingMode::Mean,
        "cls" => PoolingMode::Cls,
        "max" => PoolingMode::Max,
        "last_token" | "last" => PoolingMode::LastToken,
        "weighted_mean" | "weighted" => PoolingMode::WeightedMean,
        other => {
            anyhow::bail!(
                "Unknown pooling mode '{}'. Choose one of: mean, cls, max, last_token, weighted_mean",
                other
            );
        }
    };

    tracing::info!("Loading tokenizer from '{}'", model_path);
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    tracing::info!("Loading BERT config from '{}'", model_path);
    let config_path = std::path::Path::new(model_path).join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let mut model = BertForEmbedding::from_config_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to build BERT model: {}", e))?;

    // Load pretrained weights
    tracing::info!("Loading weights from '{}'", model_path);
    pmetal_models::loader::load_generic_weights(&mut model, model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {:?}", e))?;

    tracing::info!("Loading dataset from '{}'", dataset_path);
    let dataset = EmbeddingDataset::from_jsonl(dataset_path)
        .map_err(|e| anyhow::anyhow!("Failed to load dataset: {}", e))?;
    tracing::info!(
        "Loaded {} examples ({:?})",
        dataset.len(),
        match &dataset {
            EmbeddingDataset::Pairs(_) => "pairs",
            EmbeddingDataset::Triplets(_) => "triplets",
        }
    );

    // Build optimizer
    let optimizer = AdamWBuilder::new(learning_rate as f32)
        .weight_decay(weight_decay as f32)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build optimizer: {}", e))?;
    let mut optimizer = optimizer;

    // Build trainer config
    let training_cfg = pmetal_core::TrainingConfig {
        batch_size,
        num_epochs: epochs,
        ..Default::default()
    };

    let config = EmbeddingTrainerConfig {
        training: training_cfg,
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

    let trainer = EmbeddingTrainer::new(config);
    tracing::info!(
        "Starting embedding training: loss={}, pooling={}, T={}, batch={}, epochs={}",
        loss_str,
        pooling_str,
        temperature,
        batch_size,
        epochs
    );

    // For BERT we can use a DynamicLoraModel-like wrapper, but since BertForEmbedding
    // is not yet in the LoRA dispatch, we call into the trainer directly via the
    // generic TrainableModel bound. We use a thin adapter.
    //
    // NOTE: BertForEmbedding implements ModuleParameters via #[derive(ModuleParameters)],
    // but not TrainableModel (which also requires LoRA bookkeeping methods). For now,
    // we run the training loop manually, which mirrors how EmbeddingTrainer works
    // internally. The full TrainableModel impl for BERT (with LoRA) is a follow-on task.

    tracing::info!("Running training epochs...");
    let n_epochs = trainer.config.training.num_epochs;
    let log_every = trainer.config.log_every;

    // Manual training loop using the encode_and_loss helpers exposed by the trainer
    // until BertForEmbedding implements TrainableModel.
    use pmetal_bridge::compat::{eval_params, module::ModuleParameters, nn};
    use pmetal_models::pooling::{normalize_embeddings, pool};
    use pmetal_trainer::contrastive_loss;

    let actual_batch_size = trainer.config.training.batch_size;
    let max_len = trainer.config.max_seq_len;
    let loss_type_copy = trainer.config.loss_type;
    let temperature_copy = trainer.config.temperature;
    let margin_copy = trainer.config.margin;
    let pooling_copy = trainer.config.pooling_mode;
    let normalize_copy = trainer.config.normalize;

    let mut step = 0usize;

    match dataset {
        EmbeddingDataset::Pairs(ref pairs) => {
            let mut indices: Vec<usize> = (0..pairs.len()).collect();
            for epoch in 0..n_epochs {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(epoch as u64));
                indices.shuffle(&mut rng);

                let n_batches = pairs.len().div_ceil(actual_batch_size);
                for batch_idx in 0..n_batches {
                    let start = batch_idx * actual_batch_size;
                    let end = (start + actual_batch_size).min(pairs.len());
                    let batch: Vec<&pmetal_data::EmbeddingPair> =
                        indices[start..end].iter().map(|&i| &pairs[i]).collect();

                    // Tokenize
                    let pad_id = tokenizer.pad_token_id().unwrap_or(0) as i32;
                    let mut ids_a_raw: Vec<Vec<i32>> = Vec::new();
                    let mut ids_b_raw: Vec<Vec<i32>> = Vec::new();
                    let mut actual_max_a = 0usize;
                    let mut actual_max_b = 0usize;
                    for p in &batch {
                        let a: Vec<i32> = tokenizer
                            .encode(&p.text_a)
                            .unwrap_or_default()
                            .iter()
                            .take(max_len)
                            .map(|&x| x as i32)
                            .collect();
                        let b: Vec<i32> = tokenizer
                            .encode(&p.text_b)
                            .unwrap_or_default()
                            .iter()
                            .take(max_len)
                            .map(|&x| x as i32)
                            .collect();
                        actual_max_a = actual_max_a.max(a.len());
                        actual_max_b = actual_max_b.max(b.len());
                        ids_a_raw.push(a);
                        ids_b_raw.push(b);
                    }
                    let bs = batch.len();
                    let mut flat_a = vec![pad_id; bs * actual_max_a];
                    let mut mask_a = vec![0i32; bs * actual_max_a];
                    let mut flat_b = vec![pad_id; bs * actual_max_b];
                    let mut mask_b = vec![0i32; bs * actual_max_b];
                    for (i, (a, b)) in ids_a_raw.iter().zip(ids_b_raw.iter()).enumerate() {
                        for (j, &id) in a.iter().enumerate() {
                            flat_a[i * actual_max_a + j] = id;
                            mask_a[i * actual_max_a + j] = 1;
                        }
                        for (j, &id) in b.iter().enumerate() {
                            flat_b[i * actual_max_b + j] = id;
                            mask_b[i * actual_max_b + j] = 1;
                        }
                    }
                    let ids_a = pmetal_bridge::compat::Array::from_slice(
                        &flat_a,
                        &[bs as i32, actual_max_a as i32],
                    );
                    let m_a = pmetal_bridge::compat::Array::from_slice(
                        &mask_a,
                        &[bs as i32, actual_max_a as i32],
                    );
                    let ids_b = pmetal_bridge::compat::Array::from_slice(
                        &flat_b,
                        &[bs as i32, actual_max_b as i32],
                    );
                    let m_b = pmetal_bridge::compat::Array::from_slice(
                        &mask_b,
                        &[bs as i32, actual_max_b as i32],
                    );
                    let labels_data: Vec<f32> =
                        batch.iter().map(|p| p.label.unwrap_or(1.0)).collect();
                    let labels =
                        pmetal_bridge::compat::Array::from_slice(&labels_data, &[bs as i32]);

                    let loss_fn = |m: &mut BertForEmbedding,
                                   (ia, ma, ib, mb, lbl): (
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                    )|
                     -> Result<
                        pmetal_bridge::compat::Array,
                        pmetal_bridge::compat::Exception,
                    > {
                        let ha = m.forward(ia, Some(ma))?;
                        let hb = m.forward(ib, Some(mb))?;
                        let ea = pool(&ha, ma, pooling_copy)?;
                        let eb = pool(&hb, mb, pooling_copy)?;
                        let ea = if normalize_copy {
                            normalize_embeddings(&ea)?
                        } else {
                            ea
                        };
                        let eb = if normalize_copy {
                            normalize_embeddings(&eb)?
                        } else {
                            eb
                        };
                        match loss_type_copy {
                            EmbeddingLossType::InfoNce | EmbeddingLossType::Mnrl => {
                                contrastive_loss::info_nce_loss(&ea, &eb, temperature_copy)
                            }
                            EmbeddingLossType::CoSent => {
                                contrastive_loss::cosent_loss(&ea, &eb, lbl, temperature_copy)
                            }
                            EmbeddingLossType::CosineSimilarity => {
                                contrastive_loss::cosine_similarity_loss(&ea, &eb, lbl)
                            }
                            EmbeddingLossType::Triplet => {
                                contrastive_loss::info_nce_loss(&ea, &eb, temperature_copy)
                            }
                        }
                    };

                    let mut lag = nn::value_and_grad(loss_fn);
                    let (loss, grads) = lag(&mut model, (&ids_a, &m_a, &ids_b, &m_b, &labels))
                        .map_err(|e| anyhow::anyhow!("Forward/backward error: {}", e))?;
                    optimizer
                        .update(&mut model, grads)
                        .map_err(|e| anyhow::anyhow!("Optimizer error: {}", e))?;
                    eval_params(model.trainable_parameters())
                        .map_err(|e| anyhow::anyhow!("Eval error: {}", e))?;

                    step += 1;
                    if step % log_every == 0 {
                        let lv: f32 = loss.item();
                        tracing::info!(
                            "step={} loss={:.4} epoch={}/{}",
                            step,
                            lv,
                            epoch + 1,
                            n_epochs
                        );
                    }
                }
                tracing::info!("Epoch {}/{} complete", epoch + 1, n_epochs);
            }
        }
        EmbeddingDataset::Triplets(ref triplets) => {
            let mut indices: Vec<usize> = (0..triplets.len()).collect();
            for epoch in 0..n_epochs {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(epoch as u64));
                indices.shuffle(&mut rng);

                let n_batches = triplets.len().div_ceil(actual_batch_size);
                for batch_idx in 0..n_batches {
                    let start = batch_idx * actual_batch_size;
                    let end = (start + actual_batch_size).min(triplets.len());
                    let batch: Vec<&pmetal_data::EmbeddingTriplet> =
                        indices[start..end].iter().map(|&i| &triplets[i]).collect();

                    let pad_id = tokenizer.pad_token_id().unwrap_or(0) as i32;
                    macro_rules! tok_batch {
                        ($texts:expr) => {{
                            let mut raw: Vec<Vec<i32>> = Vec::new();
                            let mut mlen = 0usize;
                            for t in $texts.iter() {
                                let ids: Vec<i32> = tokenizer
                                    .encode(t)
                                    .unwrap_or_default()
                                    .iter()
                                    .take(max_len)
                                    .map(|&x| x as i32)
                                    .collect();
                                mlen = mlen.max(ids.len());
                                raw.push(ids);
                            }
                            let bs2 = $texts.len();
                            let mut flat = vec![pad_id; bs2 * mlen];
                            let mut msk = vec![0i32; bs2 * mlen];
                            for (i, ids) in raw.iter().enumerate() {
                                for (j, &id) in ids.iter().enumerate() {
                                    flat[i * mlen + j] = id;
                                    msk[i * mlen + j] = 1;
                                }
                            }
                            let ids_arr = pmetal_bridge::compat::Array::from_slice(
                                &flat,
                                &[bs2 as i32, mlen as i32],
                            );
                            let msk_arr = pmetal_bridge::compat::Array::from_slice(
                                &msk,
                                &[bs2 as i32, mlen as i32],
                            );
                            (ids_arr, msk_arr)
                        }};
                    }

                    let anchors: Vec<&str> = batch.iter().map(|t| t.anchor.as_str()).collect();
                    let positives: Vec<&str> = batch.iter().map(|t| t.positive.as_str()).collect();
                    let negatives: Vec<&str> = batch.iter().map(|t| t.negative.as_str()).collect();
                    let (ids_a, m_a) = tok_batch!(anchors);
                    let (ids_p, m_p) = tok_batch!(positives);
                    let (ids_n, m_n) = tok_batch!(negatives);

                    let loss_fn = |m: &mut BertForEmbedding,
                                   (ia, ma, ip, mp, i_n, mn): (
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                        &pmetal_bridge::compat::Array,
                    )|
                     -> Result<
                        pmetal_bridge::compat::Array,
                        pmetal_bridge::compat::Exception,
                    > {
                        let ha = m.forward(ia, Some(ma))?;
                        let hp = m.forward(ip, Some(mp))?;
                        let hn = m.forward(i_n, Some(mn))?;
                        let ea = pool(&ha, ma, pooling_copy)?;
                        let ep = pool(&hp, mp, pooling_copy)?;
                        let en = pool(&hn, mn, pooling_copy)?;
                        let ea = if normalize_copy {
                            normalize_embeddings(&ea)?
                        } else {
                            ea
                        };
                        let ep = if normalize_copy {
                            normalize_embeddings(&ep)?
                        } else {
                            ep
                        };
                        let en = if normalize_copy {
                            normalize_embeddings(&en)?
                        } else {
                            en
                        };
                        contrastive_loss::triplet_loss(&ea, &ep, &en, margin_copy)
                    };

                    let mut lag = nn::value_and_grad(loss_fn);
                    let (loss, grads) = lag(&mut model, (&ids_a, &m_a, &ids_p, &m_p, &ids_n, &m_n))
                        .map_err(|e| anyhow::anyhow!("Forward/backward error: {}", e))?;
                    optimizer
                        .update(&mut model, grads)
                        .map_err(|e| anyhow::anyhow!("Optimizer error: {}", e))?;
                    eval_params(model.trainable_parameters())
                        .map_err(|e| anyhow::anyhow!("Eval error: {}", e))?;

                    step += 1;
                    if step % log_every == 0 {
                        let lv: f32 = loss.item();
                        tracing::info!(
                            "step={} loss={:.4} epoch={}/{}",
                            step,
                            lv,
                            epoch + 1,
                            n_epochs
                        );
                    }
                }
                tracing::info!("Epoch {}/{} complete", epoch + 1, n_epochs);
            }
        }
    }

    // Save trained weights
    tracing::info!("Saving trained model to '{}'", output_dir);
    std::fs::create_dir_all(output_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create output dir: {}", e))?;
    let output_path = std::path::Path::new(output_dir).join("model.safetensors");
    use pmetal_bridge::compat::module::ModuleParametersExt;
    let output_path_str = output_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Non-UTF-8 output path: {}", output_path.display()))?;
    let flattened = model.flatten_params();
    let entries: Vec<(&str, &pmetal_bridge::compat::Array)> = flattened
        .iter()
        .map(|(name, value)| (name.as_ref(), value))
        .collect();
    pmetal_bridge::compat::Array::save_safetensors(output_path_str, &entries);

    // Copy config.json and tokenizer files so the output is a complete model directory
    for file in &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let src = std::path::Path::new(model_path).join(file);
        if src.exists() {
            std::fs::copy(&src, std::path::Path::new(output_dir).join(file)).ok();
        }
    }

    tracing::info!(
        "Embedding training complete. Model saved to '{}'",
        output_dir
    );
    Ok(())
}
