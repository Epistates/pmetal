use pmetal_data::{DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};

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
) -> anyhow::Result<()> {
    use pmetal_core::LoraConfig;
    use pmetal_distill::{DistillConfig, DistillMethod, Distiller, LossConfig, LossType};
    use pmetal_lora::DynamicLoraModel;
    use pmetal_trainer::{DistillationTrainer, TrainingLoopConfig};
    use std::path::{Path, PathBuf};

    let column_cfg = crate::commands::build_column_config(
        text_column,
        text_columns,
        column_separator,
        prompt_column,
        response_column,
    );

    if emit_console_output {
        println!("========================================");
        println!("  PMetal Knowledge Distillation");
        println!("========================================");
        println!("Teacher:       {}", teacher_id);
        println!("Student:       {}", student_id);
        println!("Dataset:       {}", dataset_path);
        println!("Output:        {}", output_dir);
        println!("Method:        {}", method_str);
        println!("Loss Type:     {}", loss_type_str);
        println!("Temperature:   {}", temperature);
        println!("Alpha:         {}", alpha);
        if rationale {
            println!("Rationale:     enabled (weight: {})", rationale_weight);
        }
        println!("LoRA Rank:     {}", lora_r);
        println!("LR:            {:.2e}", learning_rate);
        println!("Batch Size:    {}", batch_size);
        println!("Epochs:        {}", epochs);
        println!("Max Seq Len:   {}", max_seq_len);
        println!("========================================\n");
    }

    // 1. Resolve and Download Models
    tracing::info!("Resolving teacher model: {}", teacher_id);
    let teacher_path = if teacher_id.contains('/') && !Path::new(teacher_id).exists() {
        pmetal_hub::download_model(teacher_id, None, None).await?
    } else {
        PathBuf::from(teacher_id)
    };

    tracing::info!("Resolving student model: {}", student_id);
    let student_path = if student_id.contains('/') && !Path::new(student_id).exists() {
        pmetal_hub::download_model(student_id, None, None).await?
    } else {
        PathBuf::from(student_id)
    };

    // 2. Load Tokenizer (from student, with config-aware special token resolution)
    tracing::info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&student_path)?;

    // 2b. Detect chat template from student model
    let chat_template =
        pmetal_data::chat_templates::detect_chat_template(&student_path, student_id);

    // 3. Load Dataset (supports both JSONL and Parquet)
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

    // 4. Load Teacher Model (Frozen)
    tracing::info!("Loading teacher model...");
    let teacher_lora_config = LoraConfig {
        r: 0,
        ..Default::default()
    };
    let mut teacher_model = DynamicLoraModel::from_pretrained(&teacher_path, teacher_lora_config)?;

    // 5. Load Student Model (Trainable LoRA)
    tracing::info!("Loading student model...");
    let student_lora_config = LoraConfig {
        r: lora_r,
        alpha: lora_alpha as f32,
        ..Default::default()
    };
    let mut student_model =
        DynamicLoraModel::from_pretrained(&student_path, student_lora_config.clone())?;

    // 6. Setup Distillation Engine
    let method = match method_str.to_lowercase().as_str() {
        "online" => DistillMethod::Online,
        "offline" => DistillMethod::Offline,
        "progressive" => DistillMethod::Progressive,
        other => anyhow::bail!(
            "Unknown distillation method '{}'. Valid options: online, offline, progressive",
            other
        ),
    };

    let loss_type = match loss_type_str.to_lowercase().as_str() {
        "kl_divergence" => LossType::KlDivergence,
        "jensen_shannon" => LossType::JensenShannon,
        "soft_cross_entropy" => LossType::SoftCrossEntropy,
        "mse_loss" => LossType::MseLoss,
        other => anyhow::bail!(
            "Unknown loss type '{}'. Valid options: kl_divergence, jensen_shannon, soft_cross_entropy, mse_loss",
            other
        ),
    };

    let validated_distill_output = crate::validate_output_path(output_dir, "distillation output")?;
    let distill_config = DistillConfig {
        teacher: teacher_id.to_string(),
        student: student_id.to_string(),
        method,
        loss: LossConfig {
            loss_type,
            temperature,
            alpha,
            rationale,
            rationale_weight,
            ..Default::default()
        },
        offline: None,
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

    // 7. Setup Trainer
    #[allow(clippy::needless_update)] // ..Default covers cfg-gated fields (distributed)
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
        use_jit_compilation: true,
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

    // 7b. Enable adaptive LR for distillation (more conservative config)
    {
        let adaptive_config = pmetal_trainer::AdaptiveLrConfig::for_distillation();
        let control_file = validated_distill_output.join(".lr_control.json");
        trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);
        tracing::info!("Adaptive LR enabled (spike/plateau/divergence detection)");
    }

    // 7c. Add metrics callback if requested (TUI dashboard)
    if let Some(ref metrics_path) = log_metrics {
        use pmetal_trainer::callbacks::MetricsJsonCallback;
        let path = if metrics_path.contains('/') || metrics_path.contains('\\') {
            std::path::PathBuf::from(metrics_path)
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

    // 8. Run Distillation
    trainer
        .run(
            &mut student_model,
            &mut teacher_model,
            train_dataset,
            None,
            None,
        )
        .map_err(|e| anyhow::anyhow!("Distillation failed: {}", e))?;

    // 9. Save Student Adapters
    let lora_output = validated_distill_output.join("lora_weights.safetensors");
    tracing::info!("Saving distilled student adapters to {:?}", lora_output);
    std::fs::create_dir_all(&validated_distill_output)?;
    use pmetal_lora::TrainableModel;
    student_model.save_lora_weights(&lora_output)?;
    // Save adapter config so inference knows r/alpha/target_modules without guessing
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
            "  Inference:  pmetal run {} --lora {}",
            student_id,
            lora_output.display()
        );
    }
    Ok(())
}
