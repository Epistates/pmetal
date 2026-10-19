use pmetal_data::{DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};

use pmetal_core::jobs::DistillSpec;

use super::RunContext;

fn parse_method(method: &str) -> anyhow::Result<pmetal_distill::DistillMethod> {
    use pmetal_distill::DistillMethod;

    match method.to_lowercase().as_str() {
        "online" => Ok(DistillMethod::Online),
        "offline" => Ok(DistillMethod::Offline),
        "progressive" => Ok(DistillMethod::Progressive),
        other => anyhow::bail!(
            "unknown distillation method '{other}'; valid: online, offline, progressive"
        ),
    }
}

/// The loss a spec asks for, and whether its KL runs in reverse.
fn loss_from_spec(spec: &DistillSpec) -> anyhow::Result<(pmetal_distill::LossType, bool)> {
    use pmetal_distill::LossType;

    let skew = |default: f32| spec.skew_alpha.unwrap_or(default);
    let uses_skew = matches!(
        spec.loss_type.as_str(),
        "jsd_skewed" | "skew_kl" | "skew_reverse_kl"
    );
    if spec.skew_alpha.is_some() && !uses_skew {
        anyhow::bail!(
            "--skew-alpha applies to jsd_skewed, skew_kl and skew_reverse_kl, not {}",
            spec.loss_type
        );
    }
    if spec.universal_logit_top_k.is_some() && spec.loss_type != "universal_logit" {
        anyhow::bail!(
            "--universal-logit-top-k applies to universal_logit, not {}",
            spec.loss_type
        );
    }
    Ok(match spec.loss_type.to_lowercase().as_str() {
        "kl_divergence" => (LossType::KlDivergence, false),
        "reverse_kl" => (LossType::KlDivergence, true),
        "jensen_shannon" => (LossType::JensenShannon, false),
        "soft_cross_entropy" => (LossType::SoftCrossEntropy, false),
        "mse_loss" => (LossType::MseLoss, false),
        "jsd_skewed" => (LossType::JsdSkewed { alpha: skew(0.5) }, false),
        "skew_kl" => (
            LossType::SkewKl {
                alpha: skew(0.1),
                reverse: false,
            },
            false,
        ),
        "skew_reverse_kl" => (
            LossType::SkewKl {
                alpha: skew(0.1),
                reverse: true,
            },
            false,
        ),
        "universal_logit" => (
            LossType::UniversalLogit {
                top_k: spec.universal_logit_top_k,
            },
            false,
        ),
        other => anyhow::bail!(
            "unknown loss type '{other}'; valid: kl_divergence, reverse_kl, jensen_shannon, \
             soft_cross_entropy, mse_loss, jsd_skewed, skew_kl, skew_reverse_kl, universal_logit"
        ),
    })
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
        "sampled" => Ok(CompressionMethod::Sampled),
        other => anyhow::bail!(
            "unknown offline compression '{other}'; valid: none, top_k, int8, int4, sampled"
        ),
    }
}

fn resolve_offline_config(
    spec: &DistillSpec,
    method: &pmetal_distill::DistillMethod,
    output_dir: &std::path::Path,
) -> anyhow::Result<Option<pmetal_distill::OfflineConfig>> {
    if *method != pmetal_distill::DistillMethod::Offline {
        return Ok(None);
    }

    let compression = parse_compression_method(&spec.offline_compression)?;
    let logits_path = spec
        .offline_cache
        .as_deref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| output_dir.join("teacher_logits"));

    Ok(Some(pmetal_distill::OfflineConfig {
        logits_path,
        compression,
        top_k: spec.offline_top_k,
        generate: spec.offline_generate,
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
pub(crate) async fn run_distillation(spec: DistillSpec, ctx: RunContext) -> anyhow::Result<()> {
    let DistillSpec {
        teacher: ref teacher_id,
        student: ref student_id,
        dataset: ref dataset_path,
        eval_dataset: ref eval_dataset_path,
        output_dir: ref output_dir,
        temperature,
        alpha,
        rationale,
        rationale_weight,
        on_policy,
        on_policy_max_tokens,
        lora_r,
        lora_alpha,
        learning_rate,
        batch_size,
        epochs,
        max_seq_len,
        seed,
        checkpoint_every,
        resume,
        ref log_metrics,
        ..
    } = spec;
    let optimizer: pmetal_core::OptimizerType =
        spec.optimizer.parse().map_err(anyhow::Error::msg)?;
    let RunContext {
        emit_console_output,
        callbacks: extra_callbacks,
    } = ctx;
    let teacher_id = teacher_id.as_str();
    let student_id = student_id.as_str();
    let dataset_path = dataset_path.as_str();
    let output_dir = output_dir.as_str();
    use pmetal_core::LoraConfig;
    use pmetal_distill::{DistillConfig, Distiller, LogitCache, LossConfig};
    use pmetal_lora::DynamicLoraModel;
    use pmetal_trainer::{DistillationTrainer, TrainingLoopConfig, generate_teacher_logit_cache};
    use std::path::PathBuf;

    // The data loader takes `seed` too; this one covers MLX's own key, which
    // initialises the student's LoRA weights.
    pmetal_runtime::compat::random::seed(seed);

    let column_cfg = pmetal_data::DatasetColumnConfig::from_selectors(
        spec.text_column.clone(),
        spec.text_columns.as_deref(),
        spec.column_separator.clone(),
        spec.prompt_column.clone(),
        spec.response_column.clone(),
    );

    let validated_distill_output = crate::validate_output_path(output_dir, "distillation output")?;
    let method = parse_method(spec.effective_method())?;
    let (loss_type, reverse_kl) = loss_from_spec(&spec)?;
    let offline_config = resolve_offline_config(&spec, &method, &validated_distill_output)?;

    if emit_console_output {
        println!("========================================");
        println!("  PMetal Knowledge Distillation");
        println!("========================================");
        println!("Teacher:       {}", teacher_id);
        println!("Student:       {}", student_id);
        println!("Dataset:       {}", dataset_path);
        println!("Output:        {}", output_dir);
        println!("Method:        {:?}", method);
        println!("Loss Type:     {}", spec.loss_type);
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

    // A Hub dataset ID is downloaded; a path is used as it is.
    let dataset_file = pmetal_trainer::resolve_dataset_path(dataset_path).await?;
    tracing::info!("Loading dataset: {}", dataset_file.display());
    let train_dataset = TrainingDataset::from_jsonl_tokenized(
        &dataset_file,
        &tokenizer,
        DatasetFormat::Auto,
        max_seq_len,
        Some(&chat_template),
        column_cfg.as_ref(),
    )?;
    tracing::info!(
        "Distillation dataset loaded: {} samples",
        train_dataset.len()
    );
    let eval_dataset = if let Some(eval_path) = eval_dataset_path {
        let eval_file = pmetal_trainer::resolve_dataset_path(eval_path).await?;
        tracing::info!("Loading distillation eval dataset: {}", eval_file.display());
        let ds = TrainingDataset::from_jsonl_tokenized(
            &eval_file,
            &tokenizer,
            DatasetFormat::Auto,
            max_seq_len,
            Some(&chat_template),
            column_cfg.as_ref(),
        )?;
        tracing::info!("Distillation eval dataset loaded: {} samples", ds.len());
        Some(ds)
    } else {
        None
    };
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
            reverse_kl,
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
            optimizer,
            batch_size,
            num_epochs: epochs,
            max_seq_len,
            // One optimizer step per batch.
            gradient_accumulation_steps: 1,
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
        log_every: 1,
        checkpoint_every,
        eval_every: if eval_dataset.is_some() {
            checkpoint_every.max(1)
        } else {
            0
        },
        use_sequence_packing: true,
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 4,
        embedding_lr: None,
        loraplus_lr_ratio: None,
        use_cut_cross_entropy: false,
        ..Default::default()
    };

    let mut trainer = DistillationTrainer::new(distiller, training_loop_config)?;
    if on_policy > 0.0 {
        trainer = trainer.with_on_policy(pmetal_trainer::OnPolicyConfig {
            probability: on_policy,
            max_new_tokens: on_policy_max_tokens,
            temperature: 1.0,
            stop_tokens: pmetal_data::inference_config::collect_all_stop_tokens(
                &student_path,
                &tokenizer,
                Some(chat_template.template_type),
            ),
        })?;
        tracing::info!(
            "On-policy distillation: {:.0}% of batches are the student's own samples (up to {} tokens)",
            on_policy * 100.0,
            on_policy_max_tokens
        );
    }

    {
        let adaptive_config = pmetal_trainer::AdaptiveLrConfig::for_distillation();
        let control_file = pmetal_core::lr_control_path(&validated_distill_output);
        trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);
        tracing::info!("Adaptive LR enabled (spike/plateau/divergence detection)");
    }

    if let Some(metrics_path) = log_metrics {
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

    let checkpoint_dir = validated_distill_output.join("checkpoints");
    let checkpoint_manager =
        pmetal_trainer::CheckpointManager::new(&checkpoint_dir)?.with_max_checkpoints(3);
    if resume {
        if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
            tracing::info!(
                "Resuming distillation from checkpoint at step {}",
                metadata.step
            );
            use pmetal_lora::TrainableModel;
            student_model.set_lora_parameters(&lora_params);
            trainer.set_progress(metadata.step, metadata.epoch);
        } else {
            tracing::info!(
                "No distillation checkpoint found in {:?}, starting fresh",
                checkpoint_dir
            );
        }
    }

    match method {
        pmetal_distill::DistillMethod::Online | pmetal_distill::DistillMethod::Progressive => {
            tracing::info!("Loading teacher model (frozen)...");
            let mut teacher_model = pmetal_models::DynamicModel::load(&teacher_path)
                .map_err(|e| anyhow::anyhow!("Failed to load teacher model: {}", e))?;
            trainer
                .run(
                    &mut student_model,
                    &mut teacher_model,
                    train_dataset,
                    eval_dataset.clone(),
                    Some(&checkpoint_manager),
                )
                .map_err(|e| anyhow::anyhow!("Distillation failed: {}", e))?;
        }
        pmetal_distill::DistillMethod::Offline => {
            let offline_cfg = offline_config.expect("offline config required for offline mode");
            let metadata_path = offline_cfg.logits_path.join("metadata.json");
            let mut cache = if metadata_path.exists() {
                LogitCache::load(&offline_cfg.logits_path)?
            } else {
                // A sampled cache's draws are taken at the loss's temperature,
                // which the loss then has to use.
                LogitCache::new(
                    &offline_cfg.logits_path,
                    offline_cfg.compression.clone(),
                    offline_cfg.top_k,
                )?
                .with_sampling_temperature(temperature)
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
                let mut teacher_model = pmetal_models::DynamicModel::load(&teacher_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load teacher model: {}", e))?;
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
                .run_offline(
                    &mut student_model,
                    &cache,
                    train_dataset,
                    eval_dataset.clone(),
                    Some(&checkpoint_manager),
                )
                .map_err(|e| anyhow::anyhow!("Offline distillation failed: {}", e))?;
        }
    }

    tracing::info!(
        "Saving distilled student adapters to {:?}",
        validated_distill_output
    );
    let lora_output = pmetal_trainer::orchestrator::save_adapter(
        &student_model,
        &validated_distill_output,
        &student_lora_config,
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

    fn spec(loss_type: &str) -> DistillSpec {
        DistillSpec {
            loss_type: loss_type.into(),
            ..Default::default()
        }
    }

    #[test]
    fn resolve_offline_config_defaults_to_output_cache_dir() {
        let cfg = resolve_offline_config(
            &DistillSpec::default(),
            &pmetal_distill::DistillMethod::Offline,
            std::path::Path::new("./output/distilled"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            cfg.logits_path,
            std::path::PathBuf::from("./output/distilled/teacher_logits")
        );
    }

    /// Every loss the distiller can build is one the command line can ask
    /// for, with the parameters it names reaching it.
    #[test]
    fn every_loss_type_reaches_the_distiller() {
        use pmetal_distill::LossType;
        let skewed = |alpha: Option<f32>, loss: &str| {
            let (kind, reverse_kl) = loss_from_spec(&DistillSpec {
                skew_alpha: alpha,
                ..spec(loss)
            })
            .unwrap();
            assert!(!reverse_kl);
            kind
        };
        assert!(
            matches!(skewed(None, "jsd_skewed"), LossType::JsdSkewed { alpha } if alpha == 0.5)
        );
        assert!(
            matches!(skewed(Some(0.1), "jsd_skewed"), LossType::JsdSkewed { alpha } if alpha == 0.1)
        );
        assert!(matches!(
            skewed(None, "skew_kl"),
            LossType::SkewKl { alpha, reverse: false } if alpha == 0.1
        ));
        assert!(matches!(
            skewed(Some(0.3), "skew_reverse_kl"),
            LossType::SkewKl { alpha, reverse: true } if alpha == 0.3
        ));
        let (kind, reverse_kl) = loss_from_spec(&spec("reverse_kl")).unwrap();
        assert!(matches!(kind, LossType::KlDivergence) && reverse_kl);
        let (kind, _) = loss_from_spec(&DistillSpec {
            universal_logit_top_k: Some(64),
            ..spec("universal_logit")
        })
        .unwrap();
        assert!(matches!(kind, LossType::UniversalLogit { top_k: Some(64) }));
        for plain in [
            "kl_divergence",
            "jensen_shannon",
            "soft_cross_entropy",
            "mse_loss",
        ] {
            loss_from_spec(&spec(plain)).unwrap();
        }
    }

    #[test]
    fn a_loss_parameter_the_loss_does_not_read_is_refused() {
        let skew = loss_from_spec(&DistillSpec {
            skew_alpha: Some(0.2),
            ..spec("kl_divergence")
        })
        .unwrap_err()
        .to_string();
        assert!(skew.contains("--skew-alpha"), "{skew}");
        let top_k = loss_from_spec(&DistillSpec {
            universal_logit_top_k: Some(8),
            ..spec("mse_loss")
        })
        .unwrap_err()
        .to_string();
        assert!(top_k.contains("--universal-logit-top-k"), "{top_k}");
        assert!(loss_from_spec(&spec("kl")).is_err());
    }
}
