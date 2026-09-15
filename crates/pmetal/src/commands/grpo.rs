use pmetal_core::TrainingConfig;
use pmetal_data::{DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};
use pmetal_trainer::{GrpoConfig, GrpoTrainer, TrainingLoopConfig};
use std::path::PathBuf;

/// Run GRPO (Group Relative Policy Optimization) for reasoning models.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_grpo_cli(
    model_id: &str,
    dataset_path: &str,
    output_dir: &str,
    num_generations: usize,
    beta: f64,
    learning_rate: f64,
    epochs: usize,
    lora_r: usize,
    lora_alpha: f32,
    max_seq_len: usize,
    max_completion_length: usize,
    seed: u64,
    checkpoint_every: usize,
    resume: bool,
    dapo: bool,
    reasoning_rewards: bool,
    use_metal_flash_attention: bool,
    vlm_mode: bool,
    max_image_size: usize,
    reward_model_path: Option<String>,
    reward_model_max_length: usize,
    reward_model_weight: f64,
    reward_model_template: Option<String>,
    async_rewards: bool,
    use_speculative: bool,
    speculative_draft_tokens: usize,
    kv_cache_bits: Option<u8>,
    _log_metrics: Option<String>,
    emit_console_output: bool,
    extra_callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>>,
    text_column: Option<String>,
    text_columns: Option<Vec<String>>,
    column_separator: String,
    prompt_column: Option<String>,
    response_column: Option<String>,
) -> anyhow::Result<()> {
    use pmetal_core::LoraConfig;
    use pmetal_lora::{DynamicLoraModel, TrainableModel};
    use pmetal_models::DynamicModel;

    pmetal_bridge::compat::random::seed(seed);

    let column_cfg = crate::commands::build_column_config(
        text_column,
        text_columns,
        column_separator,
        prompt_column,
        response_column,
    );

    if emit_console_output {
        println!("========================================");
        println!("  PMetal GRPO Reasoning Training");
        println!("========================================");
        println!("Model:         {}", model_id);
        println!("Dataset:       {}", dataset_path);
        println!("Output:        {}", output_dir);
        println!("Generations:   {}", num_generations);
        println!("Beta:          {}", beta);
        println!("LR:            {:.2e}", learning_rate);
        println!("DAPO:          {}", dapo);
        println!("Reasoning Rew: {}", reasoning_rewards);
        if vlm_mode {
            println!("VLM Mode:      enabled (max_image_size={})", max_image_size);
        }
        println!("========================================\n");
    }

    // 1. Resolve and Download Model
    tracing::info!("Resolving model: {}", model_id);
    let model_path = pmetal_hub::resolve_model_path(model_id, None, None).await?;

    // 2. Load Tokenizer (with config-aware special token resolution)
    tracing::info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;

    // 2b. Detect chat template
    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);

    // 3. Load Dataset (Prompts — supports both JSONL and Parquet)
    tracing::info!("Loading prompt dataset: {}", dataset_path);
    let is_parquet = std::path::Path::new(dataset_path)
        .extension()
        .is_some_and(|ext| ext == "parquet");
    let dataset = if is_parquet {
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
    tracing::info!("GRPO dataset loaded: {} samples", dataset.len());
    {
        let stats = dataset.compute_statistics(max_seq_len);
        tracing::info!(
            "Dataset: lengths mean={:.0}, p95={}, p99={}, truncated={:.1}%",
            stats.mean_length,
            stats.p95_length,
            stats.p99_length,
            stats.truncated_pct,
        );
        for w in &dataset.validate_seq_len(max_seq_len) {
            tracing::warn!("{}", w);
        }
    }

    // 4. Resolve the reward model's path before loading anything, so a bad ID
    // fails now rather than after minutes of weight loading. It also has to
    // happen here: a loaded model is not `Send`, and holding one across this
    // `await` would make the whole command's future unspawnable.
    let resolved_reward_model_path = match reward_model_path {
        Some(ref rm_path_str) => Some(
            pmetal_hub::resolve_model_path(rm_path_str, None, None)
                .await
                .map(|path| (rm_path_str.clone(), path))?,
        ),
        None => None,
    };

    // 5. Load Model (Trainable LoRA)
    tracing::info!("Loading model with LoRA...");
    let lora_config = LoraConfig {
        r: lora_r,
        alpha: lora_alpha as f32,
        ..Default::default()
    };
    let mut model = DynamicLoraModel::from_pretrained(&model_path, lora_config.clone())?;

    // 5. Setup GRPO Config
    let mut grpo_config = GrpoConfig::new(num_generations).with_beta(beta);
    grpo_config.max_completion_length = max_completion_length;
    grpo_config.max_prompt_length = max_seq_len;
    grpo_config.vlm_mode = vlm_mode;
    grpo_config.max_image_size = max_image_size;
    grpo_config.reward_model_path = reward_model_path.clone();
    grpo_config.reward_model_max_length = reward_model_max_length;
    grpo_config.reward_model_weight = reward_model_weight;
    grpo_config.reward_model_chat_template = reward_model_template.clone();
    grpo_config.async_rewards = async_rewards;
    grpo_config.use_speculative = use_speculative;
    grpo_config.speculative_draft_tokens = speculative_draft_tokens;
    grpo_config.kv_cache_bits = kv_cache_bits;
    grpo_config.seed = Some(seed);

    if use_speculative && emit_console_output {
        println!(
            "Speculative decoding: enabled (draft_tokens={})",
            speculative_draft_tokens
        );
    }

    if dapo {
        grpo_config = grpo_config.for_dapo();
    }

    // 6. Setup Reward Functions
    let mut rewards = pmetal_trainer::CombinedReward::new();

    if reasoning_rewards {
        // Simple reasoning format reward: favors completions with <thinking> tags
        struct FormatReward;
        impl pmetal_trainer::RewardFunction for FormatReward {
            fn compute(
                &self,
                _prompts: &[String],
                completions: &[String],
                _images: Option<&[Vec<pmetal_bridge::compat::Array>]>,
            ) -> pmetal_trainer::GrpoResult<Vec<f64>> {
                Ok(completions
                    .iter()
                    .map(|c| {
                        if c.contains("<thinking>") && c.contains("</thinking>") {
                            1.0
                        } else if c.contains("<thinking>") {
                            0.5
                        } else {
                            0.0
                        }
                    })
                    .collect())
            }
            fn name(&self) -> &str {
                "format_reward"
            }
        }

        // Length reward: small penalty for being too long or too short
        struct LengthReward(usize);
        impl pmetal_trainer::RewardFunction for LengthReward {
            fn compute(
                &self,
                _prompts: &[String],
                completions: &[String],
                _images: Option<&[Vec<pmetal_bridge::compat::Array>]>,
            ) -> pmetal_trainer::GrpoResult<Vec<f64>> {
                Ok(completions
                    .iter()
                    .map(|c| {
                        let len = c.len();
                        if len > self.0 {
                            -0.1
                        } else if len < 10 {
                            -0.5
                        } else {
                            0.1
                        }
                    })
                    .collect())
            }
            fn name(&self) -> &str {
                "length_reward"
            }
        }

        rewards = rewards
            .add(Box::new(FormatReward), 1.0)
            .add(Box::new(LengthReward(max_seq_len * 2)), 0.2);
    } else {
        // Default dummy reward if none specified
        struct DummyReward;
        impl pmetal_trainer::RewardFunction for DummyReward {
            fn compute(
                &self,
                _p: &[String],
                completions: &[String],
                _i: Option<&[Vec<pmetal_bridge::compat::Array>]>,
            ) -> pmetal_trainer::GrpoResult<Vec<f64>> {
                Ok(vec![0.1; completions.len()])
            }
            fn name(&self) -> &str {
                "dummy"
            }
        }
        rewards = rewards.add(Box::new(DummyReward), 1.0);
    }

    // 6b. Load ML reward model if configured. Its path was resolved in step 4.
    if let Some((ref rm_path_str, ref rm_path)) = resolved_reward_model_path {
        if emit_console_output {
            println!("Loading ML reward model: {}", rm_path_str);
        }
        tracing::info!("Loading ML reward model from: {}", rm_path_str);

        let rm_tokenizer = pmetal_data::Tokenizer::from_model_dir(rm_path)
            .map_err(|e| anyhow::anyhow!("Failed to load reward model tokenizer: {}", e))?;

        let rm_config = pmetal_trainer::reward_model::RewardModelConfig {
            model_path: rm_path_str.clone(),
            max_length: reward_model_max_length,
            chat_template: reward_model_template.clone(),
            weight: reward_model_weight,
            ..Default::default()
        };

        let ml_reward = pmetal_trainer::reward_model::MLRewardModel::from_pretrained(
            &rm_path,
            rm_tokenizer,
            rm_config,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load ML reward model: {}", e))?;

        rewards = rewards.add(Box::new(ml_reward), reward_model_weight);

        if emit_console_output {
            println!("ML reward model loaded (weight={:.2})", reward_model_weight);
        }
    }

    // 7. Setup Trainer
    let training_config = TrainingConfig {
        learning_rate,
        batch_size: 1, // GRPO generates num_generations per prompt, so batch_size 1 is typical
        num_epochs: epochs,
        max_seq_len,
        output_dir: output_dir.to_string(),
        save_steps: if checkpoint_every > 0 {
            Some(checkpoint_every)
        } else {
            None
        },
        ..Default::default()
    };

    #[allow(clippy::needless_update)] // ..Default covers cfg-gated fields (distributed)
    let _training_loop_config = TrainingLoopConfig {
        training: training_config.clone(),
        dataloader: DataLoaderConfig {
            batch_size: 1,
            max_seq_len,
            shuffle: true,
            seed,
            pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
            drop_last: false,
            ..Default::default()
        },
        use_metal_flash_attention,
        log_every: 1,
        checkpoint_every: 50,
        eval_every: 0,
        use_jit_compilation: true,
        use_sequence_packing: false, // GRPO usually doesn't pack generations
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 4,
        embedding_lr: None,
        eager_evaluation: true, // GRPO generates first, then trains - eager helps memory
        use_metal_fused_optimizer: true,
        loraplus_lr_ratio: None,
        neftune_noise_alpha: None,
        use_cut_cross_entropy: false,
        ..Default::default()
    };

    let mut trainer = GrpoTrainer::new(grpo_config, training_config)?;

    // Wire metrics callback if --log-metrics was provided
    if let Some(ref metrics_path) = _log_metrics {
        use pmetal_trainer::callbacks::MetricsJsonCallback;
        let path = if metrics_path.contains('/') || metrics_path.contains('\\') {
            PathBuf::from(metrics_path)
        } else {
            PathBuf::from(output_dir).join(metrics_path)
        };
        let callback = MetricsJsonCallback::new(&path)?.with_run_name(format!(
            "grpo-{}",
            model_id.split('/').next_back().unwrap_or(model_id)
        ));
        trainer.add_callback(Box::new(callback));
    }
    for callback in extra_callbacks {
        trainer.add_callback(callback);
    }

    // Enable adaptive LR with control file for TUI communication
    {
        let adaptive_config = pmetal_trainer::AdaptiveLrConfig::default();
        let control_file = PathBuf::from(output_dir).join(".lr_control.json");
        trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);
    }

    let checkpoint_dir = PathBuf::from(output_dir).join("checkpoints");
    let checkpoint_manager =
        pmetal_trainer::CheckpointManager::new(&checkpoint_dir)?.with_max_checkpoints(3);
    if resume {
        if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
            tracing::info!("Resuming GRPO from checkpoint at step {}", metadata.step);
            model.set_lora_parameters(&lora_params);
            trainer.set_step(metadata.step);
        } else {
            tracing::info!(
                "No GRPO checkpoint found in {:?}, starting fresh",
                checkpoint_dir
            );
        }
    }
    trainer.set_checkpoint_manager(checkpoint_manager);

    // 7b. Setup Optimizer
    let mut optimizer = pmetal_bridge::compat::optimizers::AdamWBuilder::new(learning_rate as f32)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build optimizer: {}", e))?;

    // 8. Run Training
    if emit_console_output {
        println!("Starting GRPO training loop...");
    }

    // Load reference model (frozen)
    if emit_console_output {
        println!("Loading reference model...");
    }
    let mut ref_model = DynamicModel::load(&model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load reference model: {}", e))?;

    if async_rewards {
        // Pipelined path: reward scoring runs in a background thread so the GPU
        // training step can overlap with the next sample's reward computation.
        if emit_console_output {
            println!("Async rewards enabled — reward scoring will pipeline with GPU training.");
        }
        trainer
            .run_async(
                &mut model,
                Some(&mut ref_model),
                &tokenizer,
                &dataset,
                Box::new(rewards),
                &mut optimizer,
                |opt, lr| {
                    opt.lr = pmetal_bridge::array!(lr);
                },
            )
            .map_err(|e| anyhow::anyhow!("GRPO training error: {}", e))?;
    } else {
        trainer
            .run(
                &mut model,
                Some(&mut ref_model),
                &tokenizer,
                &dataset,
                &rewards,
                &mut optimizer,
                |opt, lr| {
                    opt.lr = pmetal_bridge::array!(lr);
                },
            )
            .map_err(|e| anyhow::anyhow!("GRPO training error: {}", e))?;
    }

    let output_dir_path = PathBuf::from(output_dir);
    std::fs::create_dir_all(&output_dir_path)?;
    let final_path = output_dir_path.join("lora_weights.safetensors");
    model.save_lora_weights(&final_path)?;
    crate::save_adapter_config(
        &final_path,
        lora_config.r,
        lora_config.alpha,
        &lora_config.target_modules,
        lora_config.use_rslora,
        Some(model_id),
    )?;

    if emit_console_output {
        println!(
            "\nGRPO training complete! Model weights saved to: {}",
            output_dir
        );
    }
    Ok(())
}
