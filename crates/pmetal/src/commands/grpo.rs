use pmetal_core::TrainingConfig;
use pmetal_data::Tokenizer;
use pmetal_trainer::{GrpoConfig, GrpoTrainer};
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
    optimizer: pmetal_core::OptimizerType,
    epochs: usize,
    lora_r: usize,
    lora_alpha: f32,
    max_seq_len: usize,
    max_completion_length: usize,
    seed: u64,
    checkpoint_every: usize,
    resume: bool,
    dapo: bool,
    loss_type: &str,
    num_iterations: usize,
    reasoning_rewards: bool,
    vlm_mode: bool,
    max_image_size: usize,
    reward_model_path: Option<String>,
    reward_model_max_length: usize,
    reward_model_weight: f64,
    reward_model_template: Option<String>,
    async_rewards: bool,
    kv_cache_bits: Option<u8>,
    _log_metrics: Option<String>,
    emit_console_output: bool,
    extra_callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>>,
    prompt_column: Option<String>,
    response_column: Option<String>,
) -> anyhow::Result<()> {
    use pmetal_core::LoraConfig;
    use pmetal_lora::{DynamicLoraModel, TrainableModel};
    use pmetal_models::DynamicModel;

    pmetal_runtime::compat::random::seed(seed);

    let prompt_columns = pmetal_trainer::grpo::PromptColumns {
        prompt: prompt_column,
        reference: response_column,
    };

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
        println!("Loss type:     {}", loss_type);
        println!("Updates/batch: {}", num_iterations);
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

    // 3. Load the prompts: each row's conversation up to where the model
    // writes (generation prompt included), with its answer kept aside as the
    // reference for the accuracy reward.
    tracing::info!("Loading prompt dataset: {}", dataset_path);
    let prompts = pmetal_trainer::grpo::load_prompts_with(
        dataset_path,
        &tokenizer,
        &chat_template,
        max_seq_len,
        &prompt_columns,
    )?;

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
        alpha: lora_alpha,
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
    grpo_config.kv_cache_bits = kv_cache_bits;
    grpo_config.seed = Some(seed);
    // A completion ends where the model would stop: every end-of-turn and
    // end-of-text token it declares, not only the tokenizer's EOS.
    grpo_config.stop_tokens = pmetal_data::inference_config::collect_all_stop_tokens(
        std::path::Path::new(&model_path),
        &tokenizer,
        Some(chat_template.template_type),
    );

    if dapo {
        grpo_config = grpo_config.for_dapo();
    }
    grpo_config = grpo_config
        .with_loss_preset(loss_type)
        .map_err(|e| anyhow::anyhow!(e))?;
    grpo_config.num_iterations = num_iterations.max(1);
    if grpo_config.importance_sampling == pmetal_trainer::ImportanceSampling::Sequence
        && grpo_config.num_iterations == 1
    {
        tracing::warn!(
            "GSPO's sequence-level ratio is 1 on the first update of a batch, so with \
             --num-iterations 1 it trains exactly like GRPO; pass --num-iterations 2 or more"
        );
    }

    // 6. Reward functions: the reasoning rewards, which add the accuracy of
    // the final answer when every prompt has a reference answer; or the
    // accuracy alone; and the ML reward model, if configured.
    let with_references = prompts.iter().all(|p| p.reference.is_some());
    if !with_references && prompts.iter().any(|p| p.reference.is_some()) {
        tracing::warn!(
            "only some prompts have a reference answer; the accuracy reward is off. Give every \
             row an `answer` to use it"
        );
    }
    let mut rewards = if reasoning_rewards {
        pmetal_trainer::CombinedReward::reasoning(max_seq_len * 2, with_references)
    } else if with_references {
        pmetal_trainer::CombinedReward::new()
            .add(Box::new(pmetal_trainer::AccuracyReward::new()), 1.0)
    } else {
        pmetal_trainer::CombinedReward::new()
    };

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
        };

        let ml_reward = pmetal_trainer::reward_model::MLRewardModel::from_pretrained(
            rm_path,
            rm_tokenizer,
            rm_config,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load ML reward model: {}", e))?;

        rewards = rewards.add(Box::new(ml_reward), reward_model_weight);

        if emit_console_output {
            println!("ML reward model loaded (weight={:.2})", reward_model_weight);
        }
    }

    // A group of completions that all score alike has no advantage to learn
    // from, so a run with nothing to score by would train on nothing.
    if rewards.is_empty() {
        anyhow::bail!(
            "GRPO needs a reward: pass --reasoning-rewards, give the dataset an `answer` column \
             (every row) to reward correct answers, or configure --reward-model"
        );
    }

    // 7. Setup Trainer
    let training_config = TrainingConfig {
        learning_rate,
        optimizer,
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
        let control_file = pmetal_core::lr_control_path(std::path::Path::new(output_dir));
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
    let mut optimizer = pmetal_trainer::TrainOptimizer::from_config(&trainer.training_config);

    // 8. Run Training
    if emit_console_output {
        println!("Starting GRPO training loop...");
    }

    // The reference model (frozen) is only the KL penalty's, so it is loaded
    // only when the penalty is on.
    let mut ref_model = if beta > 0.0 {
        if emit_console_output {
            println!("Loading reference model...");
        }
        Some(
            DynamicModel::load(&model_path)
                .map_err(|e| anyhow::anyhow!("Failed to load reference model: {}", e))?,
        )
    } else {
        None
    };

    if async_rewards {
        // Pipelined path: reward scoring runs in a background thread so the GPU
        // training step can overlap with the next sample's reward computation.
        if emit_console_output {
            println!("Async rewards enabled — reward scoring will pipeline with GPU training.");
        }
        trainer
            .run_async(
                &mut model,
                ref_model.as_mut(),
                &tokenizer,
                &prompts,
                Box::new(rewards),
                &mut optimizer,
                |opt, lr| opt.set_lr(lr),
            )
            .map_err(|e| anyhow::anyhow!("GRPO training error: {}", e))?;
    } else {
        trainer
            .run(
                &mut model,
                ref_model.as_mut(),
                &tokenizer,
                &prompts,
                &rewards,
                &mut optimizer,
                |opt, lr| opt.set_lr(lr),
            )
            .map_err(|e| anyhow::anyhow!("GRPO training error: {}", e))?;
    }

    pmetal_trainer::orchestrator::save_adapter(
        &model,
        &PathBuf::from(output_dir),
        &lora_config,
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
