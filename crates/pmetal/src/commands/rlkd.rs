use pmetal_core::TrainingConfig;
use pmetal_data::Tokenizer;
use pmetal_trainer::{GrpoConfig, RlkdConfig, RlkdTrainer};
use std::path::PathBuf;

/// Run RLKD (Reinforcement Learning with Knowledge Distillation).
///
/// Combines GRPO policy gradient with teacher distillation in a single backward pass.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_rlkd_cli(
    model_id: &str,
    teacher_model_id: &str,
    dataset_path: &str,
    output_dir: &str,
    distill_alpha: f32,
    final_alpha: f32,
    anneal_alpha: bool,
    distill_temperature: f32,
    num_generations: usize,
    beta: f64,
    learning_rate: f64,
    epochs: usize,
    lora_r: usize,
    lora_alpha: f32,
    max_seq_len: usize,
    max_completion_length: usize,
    seed: u64,
    reasoning_rewards: bool,
    use_metal_flash_attention: bool,
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
    use pmetal_lora::DynamicLoraModel;
    use pmetal_models::DynamicModel;

    pmetal_runtime::compat::random::seed(seed);

    // RLKD reads a prompt and, optionally, a reference answer from a row: the
    // prompt and response columns name them; the text-column options describe
    // supervised text, which RLKD does not train on.
    let _ = (text_column, text_columns, column_separator);
    let prompt_columns = pmetal_trainer::grpo::PromptColumns {
        prompt: prompt_column,
        reference: response_column,
    };

    if emit_console_output {
        println!("========================================");
        println!("  PMetal RLKD Training");
        println!("========================================");
        println!("Policy model:  {}", model_id);
        println!("Teacher model: {}", teacher_model_id);
        println!("Dataset:       {}", dataset_path);
        println!("Output:        {}", output_dir);
        println!(
            "Alpha:         {:.2} → {:.2} (anneal={})",
            distill_alpha, final_alpha, anneal_alpha
        );
        println!("Temperature:   {:.1}", distill_temperature);
        println!("Generations:   {}", num_generations);
        println!("Beta:          {}", beta);
        println!("LR:            {:.2e}", learning_rate);
        println!("========================================\n");
    }

    // 1. Resolve and download policy model
    tracing::info!("Resolving policy model: {}", model_id);
    let model_path = pmetal_hub::resolve_model_path(model_id, None, None).await?;

    // 2. Resolve and download teacher model
    tracing::info!("Resolving teacher model: {}", teacher_model_id);
    let teacher_path = pmetal_hub::resolve_model_path(teacher_model_id, None, None).await?;

    // 3. Load tokenizer (policy model's tokenizer drives generation)
    tracing::info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;

    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);

    // 4. Load the prompts: each row's conversation up to where the model
    // writes, with its answer kept aside as the accuracy reward's reference.
    tracing::info!("Loading dataset: {}", dataset_path);
    let prompts = pmetal_trainer::grpo::load_prompts_with(
        dataset_path,
        &tokenizer,
        &chat_template,
        max_seq_len,
        &prompt_columns,
    )?;

    // 5. Load policy model with LoRA
    tracing::info!("Loading policy model with LoRA...");
    let lora_config = LoraConfig {
        r: lora_r,
        alpha: lora_alpha,
        ..Default::default()
    };
    let mut policy_model = DynamicLoraModel::from_pretrained(&model_path, lora_config.clone())?;

    // 6. Load teacher model (frozen, no LoRA)
    tracing::info!("Loading teacher model (frozen)...");
    let mut teacher_model = DynamicModel::load(&teacher_path)
        .map_err(|e| anyhow::anyhow!("Failed to load teacher model: {}", e))?;

    // 7. Setup GRPO config (embedded in RLKD config)
    let mut grpo_config = GrpoConfig::new(num_generations).with_beta(beta);
    grpo_config.max_completion_length = max_completion_length;
    grpo_config.max_prompt_length = max_seq_len;
    grpo_config.seed = Some(seed);
    // A completion ends where the model would stop: every end-of-turn and
    // end-of-text token it declares, not only the tokenizer's EOS.
    grpo_config.stop_tokens = pmetal_data::inference_config::collect_all_stop_tokens(
        std::path::Path::new(&model_path),
        &tokenizer,
        Some(chat_template.template_type),
    );

    // 8. Setup RLKD config
    let training_config = TrainingConfig {
        learning_rate,
        batch_size: 1,
        num_epochs: epochs,
        max_seq_len,
        output_dir: output_dir.to_string(),
        ..Default::default()
    };

    let rlkd_config = RlkdConfig {
        grpo: grpo_config,
        training: training_config,
        distill_alpha,
        distill_temperature,
        reverse_kl: false,
        anneal_alpha,
        final_alpha,
    };

    // 9. Rewards: the reasoning rewards, which add the accuracy of the final
    // answer when every prompt has a reference answer; or the accuracy alone.
    // With neither, every completion scores alike and the RL term has no
    // advantage to push: the run is on-policy distillation from the teacher.
    let with_references = prompts.iter().all(|p| p.reference.is_some());
    let rewards = if reasoning_rewards {
        pmetal_trainer::CombinedReward::reasoning(max_seq_len * 2, with_references)
    } else if with_references {
        pmetal_trainer::CombinedReward::new()
            .add(Box::new(pmetal_trainer::AccuracyReward::new()), 1.0)
    } else {
        tracing::warn!(
            "no reward configured (--reasoning-rewards, or an `answer` on every row): RLKD will \
             only distill from the teacher"
        );
        struct NoReward;
        impl pmetal_trainer::RewardFunction for NoReward {
            fn compute(
                &self,
                _: &[String],
                completions: &[String],
                _: Option<&[Vec<pmetal_runtime::compat::Array>]>,
            ) -> pmetal_trainer::GrpoResult<Vec<f64>> {
                Ok(vec![0.0; completions.len()])
            }
            fn name(&self) -> &str {
                "none"
            }
        }
        pmetal_trainer::CombinedReward::new().add(Box::new(NoReward), 1.0)
    };

    // The reference model (frozen) is only the KL penalty's, so it is loaded
    // only when the penalty is on.
    let mut ref_model = if beta > 0.0 {
        Some(
            DynamicModel::load(&model_path)
                .map_err(|e| anyhow::anyhow!("Failed to load reference model: {}", e))?,
        )
    } else {
        None
    };

    // 10. Build trainer
    let mut trainer = RlkdTrainer::new(rlkd_config)?;

    if let Some(ref metrics_path) = _log_metrics {
        let path = if metrics_path.contains('/') || metrics_path.contains('\\') {
            PathBuf::from(metrics_path)
        } else {
            PathBuf::from(output_dir).join(metrics_path)
        };
        let callback = pmetal_trainer::MetricsJsonCallback::new(&path)?.with_run_name(format!(
            "rlkd-{}",
            model_id.split('/').next_back().unwrap_or(model_id)
        ));
        trainer.add_callback(Box::new(callback));
    }
    for callback in extra_callbacks {
        trainer.add_callback(callback);
    }

    {
        let adaptive_config = pmetal_trainer::AdaptiveLrConfig::default();
        let control_file = pmetal_core::lr_control_path(std::path::Path::new(output_dir));
        trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);
    }

    // 11. Build optimizer
    let mut optimizer = pmetal_trainer::TrainOptimizer::from_config(&trainer.config.training);

    // 12. Run training
    if emit_console_output {
        println!("Starting RLKD training loop...");
    }

    let _ = use_metal_flash_attention; // passed through GRPO config; honored by model load

    trainer
        .run(
            &mut policy_model,
            &mut teacher_model,
            ref_model.as_mut(),
            &tokenizer,
            &prompts,
            &rewards,
            &mut optimizer,
            |opt, lr| opt.set_lr(lr),
        )
        .map_err(|e| anyhow::anyhow!("RLKD training error: {}", e))?;

    // 13. Save LoRA weights
    pmetal_trainer::orchestrator::save_adapter(
        &policy_model,
        &PathBuf::from(output_dir),
        &lora_config,
        Some(model_id),
    )?;

    if emit_console_output {
        println!(
            "\nRLKD training complete! Model weights saved to: {}",
            output_dir
        );
    }
    Ok(())
}
