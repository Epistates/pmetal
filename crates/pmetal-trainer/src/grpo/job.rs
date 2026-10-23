//! `pmetal grpo` as a library call: the job a [`GrpoSpec`] describes.
//!
//! The command line, the terminal UI, the desktop app and the Python bindings
//! all run this one function; what a surface adds is in [`GrpoJobOptions`]
//! (callbacks that carry progress out, and rewards of its own, like the
//! bindings' Python callables).

use std::path::PathBuf;
use std::time::Duration;

use pmetal_core::jobs::GrpoSpec;
use pmetal_core::{LoraConfig, TrainingCallback, TrainingConfig};
use pmetal_data::Tokenizer;
use pmetal_lora::{DynamicLoraModel, TrainableModel};
use pmetal_models::DynamicModel;

use super::{
    AccuracyReward, CombinedReward, ExternalCommandReward, ExternalRewardConfig, GrpoConfig,
    GrpoError, GrpoResult, GrpoTrainer, ImportanceSampling, JsonReward, PromptColumns, RegexReward,
    RewardFunction,
};

/// What a surface adds to a GRPO run.
#[derive(Default)]
pub struct GrpoJobOptions {
    /// Whether the run prints its banner and summary (the command line does;
    /// the other surfaces draw their own).
    pub emit_console_output: bool,
    /// Callbacks that carry the run's progress out and can stop it.
    pub callbacks: Vec<Box<dyn TrainingCallback>>,
    /// Rewards of the caller's own, with their weights, summed with the ones
    /// the spec asks for.
    pub extra_rewards: Vec<(Box<dyn RewardFunction>, f64)>,
}

/// How a GRPO run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrpoJobResult {
    /// Optimizer steps taken.
    pub steps: usize,
    /// A callback asked the run to stop before it finished. The adapter holds
    /// what the steps so far trained and has been saved like a finished run's.
    pub stopped: bool,
}

/// The rewards a spec asks for, besides the ML reward model (which needs a
/// model loaded): the reasoning rewards or the accuracy of the final answer,
/// the spec's reward commands and its verifiers.
///
/// `with_references` is whether every prompt has a reference answer, which is
/// what the accuracy reward compares against.
///
/// # Errors
///
/// [`GrpoError::Config`] for a reward command that cannot start, a pattern
/// that does not compile or a schema that cannot be read.
pub fn rewards_from_spec(spec: &GrpoSpec, with_references: bool) -> GrpoResult<CombinedReward> {
    let mut rewards = if spec.reasoning_rewards {
        CombinedReward::reasoning(spec.max_seq_len * 2, with_references)
    } else if with_references {
        CombinedReward::new().add(Box::new(AccuracyReward::new()), 1.0)
    } else {
        CombinedReward::new()
    };

    let weights = spec.reward_cmd_weights().map_err(GrpoError::Config)?;
    for (command, weight) in spec.reward_cmd.iter().zip(weights) {
        let config = ExternalRewardConfig::from_command_line(command)?
            .with_timeout(Duration::from_secs(spec.reward_timeout));
        rewards = rewards.add(Box::new(ExternalCommandReward::spawn(config)?), weight);
    }
    if let Some(pattern) = &spec.reward_regex {
        rewards = rewards.add(
            Box::new(RegexReward::new(pattern)?),
            spec.reward_regex_weight,
        );
    }
    if let Some(path) = &spec.reward_json_schema {
        let text = std::fs::read_to_string(path).map_err(|e| {
            GrpoError::Config(format!("cannot read the reward JSON schema {path}: {e}"))
        })?;
        rewards = rewards.add(
            Box::new(JsonReward::with_schema_text(&text)?),
            spec.reward_json_weight,
        );
    } else if spec.reward_json {
        rewards = rewards.add(Box::new(JsonReward::any()), spec.reward_json_weight);
    }
    Ok(rewards)
}

/// Run the GRPO job `spec` describes, in this process.
///
/// The spec must have been normalized. A run a callback stopped is saved like
/// a finished one and reported in [`GrpoJobResult::stopped`], not as an error.
///
/// # Errors
///
/// Anything that stops the run: a model or dataset that cannot be loaded, a
/// reward that fails (a command that exits, a callable that raises), an
/// out-of-memory.
pub async fn run_grpo_job(
    spec: GrpoSpec,
    options: GrpoJobOptions,
) -> anyhow::Result<GrpoJobResult> {
    let GrpoSpec {
        model: ref model_id,
        dataset: ref dataset_path,
        ref output_dir,
        num_generations,
        beta,
        learning_rate,
        epochs,
        lora_r,
        lora_alpha,
        max_seq_len,
        max_completion_length,
        seed,
        checkpoint_every,
        resume,
        dapo,
        ref loss_type,
        num_iterations,
        reasoning_rewards,
        vlm: vlm_mode,
        max_image_size,
        reward_model: ref reward_model_path,
        reward_model_max_length,
        reward_model_weight,
        ref reward_model_template,
        async_rewards,
        grpo_kv_bits: kv_cache_bits,
        ref prompt_column,
        ref response_column,
        ref log_metrics,
        ..
    } = spec;
    let optimizer: pmetal_core::OptimizerType =
        spec.optimizer.parse().map_err(anyhow::Error::msg)?;
    let GrpoJobOptions {
        emit_console_output,
        callbacks: extra_callbacks,
        extra_rewards,
    } = options;
    let (model_id, dataset_path, output_dir) = (
        model_id.as_str(),
        dataset_path.as_str(),
        output_dir.as_str(),
    );
    let loss_type = loss_type.as_str();

    pmetal_runtime::compat::random::seed(seed);

    let prompt_columns = PromptColumns {
        prompt: prompt_column.clone(),
        reference: response_column.clone(),
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
    let model_path = crate::orchestrator::resolve_model_path(model_id).await?;
    // What the run started from, left beside its results for the adapter list.
    pmetal_core::TrainingInfo {
        base_model_path: Some(model_path.display().to_string()),
        dataset: Some(dataset_path.to_string()),
        method: Some("grpo".to_string()),
        ..pmetal_core::TrainingInfo::starting(model_id)
    }
    .write_or_warn(std::path::Path::new(output_dir));

    // 2. Load Tokenizer (with config-aware special token resolution)
    tracing::info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;

    // 2b. Detect chat template
    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);

    // 3. Load the prompts: each row's conversation up to where the model
    // writes (generation prompt included), with its answer kept aside as the
    // reference for the accuracy reward.
    let dataset_file = crate::resolve_dataset_path(dataset_path).await?;
    tracing::info!("Loading prompt dataset: {}", dataset_file.display());
    let prompts = super::load_prompts_with(
        &dataset_file,
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
        Some(rm_path_str) => Some((
            rm_path_str.clone(),
            crate::orchestrator::resolve_model_path(rm_path_str).await?,
        )),
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
    if grpo_config.importance_sampling == ImportanceSampling::Sequence
        && grpo_config.num_iterations == 1
    {
        tracing::warn!(
            "GSPO's sequence-level ratio is 1 on the first update of a batch, so with \
             --num-iterations 1 it trains exactly like GRPO; pass --num-iterations 2 or more"
        );
    }

    // 6. Reward functions: the reasoning rewards, which add the accuracy of
    // the final answer when every prompt has a reference answer; or the
    // accuracy alone; the spec's reward commands and verifiers; the caller's
    // own; and the ML reward model, if configured.
    let with_references = prompts.iter().all(|p| p.reference.is_some());
    if !with_references && prompts.iter().any(|p| p.reference.is_some()) {
        tracing::warn!(
            "only some prompts have a reference answer; the accuracy reward is off. Give every \
             row an `answer` to use it"
        );
    }
    let mut rewards = rewards_from_spec(&spec, with_references)?;
    for (function, weight) in extra_rewards {
        rewards = rewards.add(function, weight);
    }

    // 6b. Load ML reward model if configured. Its path was resolved in step 4.
    if let Some((ref rm_path_str, ref rm_path)) = resolved_reward_model_path {
        if emit_console_output {
            println!("Loading ML reward model: {}", rm_path_str);
        }
        tracing::info!("Loading ML reward model from: {}", rm_path_str);

        let rm_tokenizer = Tokenizer::from_model_dir(rm_path)
            .map_err(|e| anyhow::anyhow!("Failed to load reward model tokenizer: {}", e))?;

        let rm_config = crate::reward_model::RewardModelConfig {
            model_path: rm_path_str.clone(),
            max_length: reward_model_max_length,
            chat_template: reward_model_template.clone(),
        };

        let ml_reward =
            crate::reward_model::MLRewardModel::from_pretrained(rm_path, rm_tokenizer, rm_config)
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
             (every row) to reward correct answers, or configure --reward-cmd, --reward-regex, \
             --reward-json or --reward-model"
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
    if let Some(metrics_path) = log_metrics {
        use crate::callbacks::MetricsJsonCallback;
        let path = pmetal_core::paths::resolve_in(std::path::Path::new(output_dir), metrics_path);
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
        let adaptive_config = crate::AdaptiveLrConfig::default();
        let control_file = pmetal_core::lr_control_path(std::path::Path::new(output_dir));
        trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);
    }

    let checkpoint_dir = PathBuf::from(output_dir).join("checkpoints");
    let checkpoint_manager =
        crate::CheckpointManager::new(&checkpoint_dir)?.with_max_checkpoints(3);
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
    let mut optimizer = crate::TrainOptimizer::from_config(&trainer.training_config);

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

    // A run a stop request ended has trained part of the way; the policy is
    // saved below like a finished one.
    let outcome = if async_rewards {
        // Pipelined path: reward scoring runs in a background thread so the GPU
        // training step can overlap with the next sample's reward computation.
        if emit_console_output {
            println!("Async rewards enabled — reward scoring will pipeline with GPU training.");
        }
        trainer.run_async(
            &mut model,
            ref_model.as_mut(),
            &tokenizer,
            &prompts,
            Box::new(rewards),
            &mut optimizer,
            |opt, lr| opt.set_lr(lr),
        )
    } else {
        trainer.run(
            &mut model,
            ref_model.as_mut(),
            &tokenizer,
            &prompts,
            &rewards,
            &mut optimizer,
            |opt, lr| opt.set_lr(lr),
        )
    };
    let stopped = match outcome {
        Ok(()) => false,
        Err(GrpoError::Cancelled) => true,
        Err(e) => return Err(anyhow::anyhow!("GRPO training error: {e}")),
    };

    crate::orchestrator::save_adapter(
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
    Ok(GrpoJobResult {
        steps: trainer.step,
        stopped,
    })
}
