use pmetal_data::{DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};

// Phase 4 follow-up: the `parse_method` / `parse_loss_type` /
// `parse_compression_method` / `resolve_offline_config` helpers below duplicate
// the enum options that are now declared (and descriptor-validated) in
// `pmetal_core::jobs::DistillSpec`. The Phase 5 CLI adoption sweep should
// replace `run_distillation_cli`'s flat parameter list with a single
// `DistillSpec` argument, call `spec.normalize()` at the call site, and remove
// these matchers in favour of the already-validated typed enums.

#[derive(Debug, Clone, Default)]
pub(crate) struct OfflineCliOptions {
    pub enabled: bool,
    pub cache_path: Option<String>,
    pub generate: bool,
    pub compression: String,
    pub top_k: usize,
}

fn parse_method(method_str: &str) -> anyhow::Result<pmetal_distill::DistillMethod> {
    use pmetal_distill::DistillMethod;

    match method_str.to_lowercase().as_str() {
        "online" => Ok(DistillMethod::Online),
        "offline" => Ok(DistillMethod::Offline),
        "progressive" => Ok(DistillMethod::Progressive),
        other => anyhow::bail!(
            "Unknown distillation method '{}'. Valid options: online, offline, progressive",
            other
        ),
    }
}

fn resolve_method(
    method_str: &str,
    offline: &OfflineCliOptions,
) -> anyhow::Result<pmetal_distill::DistillMethod> {
    let requested = parse_method(method_str)?;
    if offline.enabled {
        if requested != pmetal_distill::DistillMethod::Online
            && requested != pmetal_distill::DistillMethod::Offline
        {
            anyhow::bail!("--offline conflicts with --method {method_str}");
        }
        Ok(pmetal_distill::DistillMethod::Offline)
    } else {
        Ok(requested)
    }
}

fn parse_loss_type(loss_type_str: &str) -> anyhow::Result<pmetal_distill::LossType> {
    use pmetal_distill::LossType;

    match loss_type_str.to_lowercase().as_str() {
        "kl_divergence" => Ok(LossType::KlDivergence),
        "jensen_shannon" => Ok(LossType::JensenShannon),
        "soft_cross_entropy" => Ok(LossType::SoftCrossEntropy),
        "mse_loss" => Ok(LossType::MseLoss),
        other => anyhow::bail!(
            "Unknown loss type '{}'. Valid options: kl_divergence, jensen_shannon, soft_cross_entropy, mse_loss",
            other
        ),
    }
}

fn parse_compression_method(
    compression: &str,
) -> anyhow::Result<pmetal_distill::CompressionMethod> {
    use pmetal_distill::CompressionMethod;

    match compression.to_lowercase().as_str() {
        "none" => Ok(CompressionMethod::None),
        "top_k" | "top-k" => Ok(CompressionMethod::TopK),
        "int8" => Ok(CompressionMethod::Int8),
        "int4" => Ok(CompressionMethod::Int4),
        other => anyhow::bail!(
            "Unknown offline compression '{}'. Valid options: none, top_k, int8, int4",
            other
        ),
    }
}

fn resolve_offline_config(
    method: pmetal_distill::DistillMethod,
    offline: &OfflineCliOptions,
    output_dir: &std::path::Path,
) -> anyhow::Result<Option<pmetal_distill::OfflineConfig>> {
    if method != pmetal_distill::DistillMethod::Offline {
        if offline.cache_path.is_some() || offline.generate {
            anyhow::bail!("offline-specific flags require --offline or --method offline");
        }
        return Ok(None);
    }

    let compression = parse_compression_method(&offline.compression)?;
    let logits_path = offline
        .cache_path
        .as_deref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| output_dir.join("teacher_logits"));

    Ok(Some(pmetal_distill::OfflineConfig {
        logits_path,
        compression,
        top_k: offline.top_k,
        generate: offline.generate,
    }))
}

fn validate_offline_cache(
    cache: &pmetal_distill::LogitCache,
    teacher_id: &str,
    max_seq_len: usize,
    dataset_len: usize,
) -> anyhow::Result<()> {
    let metadata = cache.metadata();
    if metadata.model != teacher_id {
        anyhow::bail!(
            "offline cache {:?} was generated for teacher '{}', but this run uses '{}'",
            cache.cache_dir(),
            metadata.model,
            teacher_id
        );
    }
    if metadata.max_seq_len != max_seq_len {
        anyhow::bail!(
            "offline cache {:?} was generated with max_seq_len={}, but this run uses {}",
            cache.cache_dir(),
            metadata.max_seq_len,
            max_seq_len
        );
    }
    if metadata.num_sequences < dataset_len {
        anyhow::bail!(
            "offline cache {:?} is incomplete: contains {} sequences, expected at least {}",
            cache.cache_dir(),
            metadata.num_sequences,
            dataset_len
        );
    }
    if metadata.vocab_size == 0 {
        anyhow::bail!(
            "offline cache {:?} is missing vocab metadata",
            cache.cache_dir()
        );
    }
    Ok(())
}

fn validate_cache_for_generation(
    cache: &pmetal_distill::LogitCache,
    teacher_id: &str,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    let metadata = cache.metadata();
    if metadata.num_sequences == 0 {
        return Ok(());
    }
    if metadata.model != teacher_id {
        anyhow::bail!(
            "offline cache {:?} already contains logits for teacher '{}'; choose a different --offline-cache path for '{}'",
            cache.cache_dir(),
            metadata.model,
            teacher_id
        );
    }
    if metadata.max_seq_len != 0 && metadata.max_seq_len != max_seq_len {
        anyhow::bail!(
            "offline cache {:?} was created with max_seq_len={}, but this run uses {}",
            cache.cache_dir(),
            metadata.max_seq_len,
            max_seq_len
        );
    }
    Ok(())
}
/// Run knowledge distillation.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_distillation_cli(
    teacher_id: &str,
    student_id: &str,
    dataset_path: &str,
    output_dir: &str,
    method_str: &str,
    loss_type_str: &str,
    temperature: f32,
    alpha: f32,
    rationale: bool,
    rationale_weight: f32,
    lora_r: usize,
    lora_alpha: f32,
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
    max_seq_len: usize,
    seed: u64,
    log_metrics: Option<String>,
    emit_console_output: bool,
    extra_callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>>,
    text_column: Option<String>,
    text_columns: Option<Vec<String>>,
    column_separator: String,
    prompt_column: Option<String>,
    response_column: Option<String>,
    offline: OfflineCliOptions,
) -> anyhow::Result<()> {
    use pmetal_core::LoraConfig;
    use pmetal_distill::{DistillConfig, Distiller, LogitCache, LossConfig};
    use pmetal_lora::DynamicLoraModel;
    use pmetal_trainer::{DistillationTrainer, TrainingLoopConfig, generate_teacher_logit_cache};
    use std::path::{Path, PathBuf};

    let column_cfg = crate::commands::build_column_config(
        text_column,
        text_columns,
        column_separator,
        prompt_column,
        response_column,
    );

    let validated_distill_output = crate::validate_output_path(output_dir, "distillation output")?;
    let method = resolve_method(method_str, &offline)?;
    let loss_type = parse_loss_type(loss_type_str)?;
    let offline_config =
        resolve_offline_config(method.clone(), &offline, &validated_distill_output)?;

    if emit_console_output {
        println!("========================================");
        println!("  PMetal Knowledge Distillation");
        println!("========================================");
        println!("Teacher:       {}", teacher_id);
        println!("Student:       {}", student_id);
        println!("Dataset:       {}", dataset_path);
        println!("Output:        {}", output_dir);
        println!("Method:        {:?}", method);
        println!("Loss Type:     {}", loss_type_str);
        println!("Temperature:   {}", temperature);
        println!("Alpha:         {}", alpha);
        if rationale {
            println!("Rationale:     enabled (weight: {})", rationale_weight);
        }
        if let Some(ref offline_cfg) = offline_config {
            println!("Offline Cache: {}", offline_cfg.logits_path.display());
            println!("Compression:   {:?}", offline_cfg.compression);
        }
        println!("LoRA Rank:     {}", lora_r);
        println!("LR:            {:.2e}", learning_rate);
        println!("Batch Size:    {}", batch_size);
        println!("Epochs:        {}", epochs);
        println!("Max Seq Len:   {}", max_seq_len);
        println!("========================================\n");
    }

    tracing::info!("Resolving teacher model: {}", teacher_id);
    let teacher_path = pmetal_hub::resolve_model_path(teacher_id, None, None).await?;

    tracing::info!("Resolving student model: {}", student_id);
    let student_path = pmetal_hub::resolve_model_path(student_id, None, None).await?;

    tracing::info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&student_path)?;
    let chat_template =
        pmetal_data::chat_templates::detect_chat_template(&student_path, student_id);

    tracing::info!("Loading dataset: {}", dataset_path);
    let is_parquet = Path::new(dataset_path)
        .extension()
        .is_some_and(|ext| ext == "parquet");
    let train_dataset = if is_parquet {
        tracing::info!("Detected Parquet format");
        let result = TrainingDataset::from_parquet_tokenized(
            dataset_path,
            &tokenizer,
            "text",
            max_seq_len,
            None,
        );
        match result {
            Ok(ds) => ds,
            Err(_) => TrainingDataset::from_parquet_tokenized(
                dataset_path,
                &tokenizer,
                "content",
                max_seq_len,
                None,
            )?,
        }
    } else {
        TrainingDataset::from_jsonl_tokenized(
            dataset_path,
            &tokenizer,
            DatasetFormat::Auto,
            max_seq_len,
            Some(&chat_template),
            column_cfg.as_ref(),
        )?
    };
    tracing::info!(
        "Distillation dataset loaded: {} samples",
        train_dataset.len()
    );
    {
        let stats = train_dataset.compute_statistics(max_seq_len);
        tracing::info!(
            "Dataset: lengths mean={:.0}, p95={}, p99={}, truncated={:.1}%",
            stats.mean_length,
            stats.p95_length,
            stats.p99_length,
            stats.truncated_pct,
        );
        for w in &train_dataset.validate_seq_len(max_seq_len) {
            tracing::warn!("{}", w);
        }
    }

    tracing::info!("Loading student model...");
    let student_lora_config = LoraConfig {
        r: lora_r,
        alpha: lora_alpha,
        ..Default::default()
    };
    let mut student_model =
        DynamicLoraModel::from_pretrained(&student_path, student_lora_config.clone())?;

    let distill_config = DistillConfig {
        teacher: teacher_id.to_string(),
        student: student_id.to_string(),
        method: method.clone(),
        loss: LossConfig {
            loss_type,
            temperature,
            alpha,
            rationale,
            rationale_weight,
            ..Default::default()
        },
        offline: offline_config.clone(),
        output_path: Some(validated_distill_output.clone()),
        training: pmetal_distill::TrainingConfig {
            batch_size,
            learning_rate,
            epochs,
            max_seq_len,
            ..Default::default()
        },
    };

    let distiller = Distiller::new(distill_config)?;

    #[allow(clippy::needless_update)]
    let training_loop_config = TrainingLoopConfig {
        training: pmetal_core::TrainingConfig {
            learning_rate: learning_rate as f64,
            batch_size,
            num_epochs: epochs,
            max_seq_len,
            output_dir: validated_distill_output.display().to_string(),
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size,
            max_seq_len,
            shuffle: true,
            seed,
            pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
            drop_last: false,
            ..Default::default()
        },
        use_metal_flash_attention: true,
        log_every: 1,
        checkpoint_every: 100,
        eval_every: 0,
        use_jit_compilation: false,
        use_sequence_packing: true,
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 4,
        embedding_lr: None,
        eager_evaluation: false,
        use_metal_fused_optimizer: true,
        loraplus_lr_ratio: None,
        neftune_noise_alpha: None,
        use_cut_cross_entropy: false,
        ..Default::default()
    };

    let mut trainer = DistillationTrainer::new(distiller, training_loop_config);

    {
        let adaptive_config = pmetal_trainer::AdaptiveLrConfig::for_distillation();
        let control_file = validated_distill_output.join(".lr_control.json");
        trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);
        tracing::info!("Adaptive LR enabled (spike/plateau/divergence detection)");
    }

    if let Some(ref metrics_path) = log_metrics {
        use pmetal_trainer::callbacks::MetricsJsonCallback;
        let path = if metrics_path.contains('/') || metrics_path.contains('\\') {
            PathBuf::from(metrics_path)
        } else {
            validated_distill_output.join(metrics_path)
        };
        let callback =
            MetricsJsonCallback::new(&path)?.with_run_name(format!("distill-{}", student_id));
        trainer.add_callback(Box::new(callback));
    }
    for callback in extra_callbacks {
        trainer.add_callback(callback);
    }

    match method {
        pmetal_distill::DistillMethod::Online | pmetal_distill::DistillMethod::Progressive => {
            tracing::info!("Loading teacher model...");
            let teacher_lora_config = LoraConfig {
                r: 0,
                ..Default::default()
            };
            let mut teacher_model =
                DynamicLoraModel::from_pretrained(&teacher_path, teacher_lora_config)?;
            trainer
                .run(
                    &mut student_model,
                    &mut teacher_model,
                    train_dataset,
                    None,
                    None,
                )
                .map_err(|e| anyhow::anyhow!("Distillation failed: {}", e))?;
        }
        pmetal_distill::DistillMethod::Offline => {
            let offline_cfg = offline_config.expect("offline config required for offline mode");
            let metadata_path = offline_cfg.logits_path.join("metadata.json");
            let mut cache = if metadata_path.exists() {
                LogitCache::load(&offline_cfg.logits_path)?
            } else {
                LogitCache::new(
                    &offline_cfg.logits_path,
                    offline_cfg.compression.clone(),
                    offline_cfg.top_k,
                )?
            };

            let cache_complete = metadata_path.exists()
                && validate_offline_cache(&cache, teacher_id, max_seq_len, train_dataset.len())
                    .is_ok();
            if !cache_complete {
                validate_cache_for_generation(&cache, teacher_id, max_seq_len)?;
                tracing::info!(
                    "Generating teacher logits for offline distillation into {:?}",
                    offline_cfg.logits_path
                );
                let teacher_lora_config = LoraConfig {
                    r: 0,
                    ..Default::default()
                };
                let mut teacher_model =
                    DynamicLoraModel::from_pretrained(&teacher_path, teacher_lora_config)?;
                generate_teacher_logit_cache(
                    &mut teacher_model,
                    &train_dataset,
                    &mut cache,
                    teacher_id,
                    max_seq_len,
                )
                .map_err(|e| anyhow::anyhow!("Failed to generate teacher logits: {}", e))?;
            }

            validate_offline_cache(&cache, teacher_id, max_seq_len, train_dataset.len())?;
            trainer
                .run_offline(&mut student_model, &cache, train_dataset, None, None)
                .map_err(|e| anyhow::anyhow!("Offline distillation failed: {}", e))?;
        }
    }

    let lora_output = validated_distill_output.join("lora_weights.safetensors");
    tracing::info!("Saving distilled student adapters to {:?}", lora_output);
    std::fs::create_dir_all(&validated_distill_output)?;
    use pmetal_lora::TrainableModel;
    student_model.save_lora_weights(&lora_output)?;
    crate::save_adapter_config(
        &lora_output,
        student_lora_config.r,
        student_lora_config.alpha,
        &student_lora_config.target_modules,
        student_lora_config.use_rslora,
        Some(student_id),
    )?;

    if emit_console_output {
        println!("\n========================================");
        println!("  Distillation Complete");
        println!("========================================");
        println!("  Adapters:  {}", lora_output.display());
        println!("  Student:   {}", student_id);
        println!("  Template:  {:?}", chat_template.template_type);
        println!("========================================");
        println!("\nNext steps:");
        println!(
            "  Inference:  pmetal infer --model {} --lora {} --prompt \"Your prompt\"",
            student_id,
            lora_output.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_method_rejects_conflicts() {
        let err = resolve_method(
            "progressive",
            &OfflineCliOptions {
                enabled: true,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("--offline conflicts"));
    }

    #[test]
    fn resolve_offline_config_defaults_to_output_cache_dir() {
        let cfg = resolve_offline_config(
            pmetal_distill::DistillMethod::Offline,
            &OfflineCliOptions {
                enabled: true,
                compression: "top_k".to_string(),
                top_k: 128,
                ..Default::default()
            },
            std::path::Path::new("./output/distilled"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            cfg.logits_path,
            std::path::PathBuf::from("./output/distilled/teacher_logits")
        );
    }
}
