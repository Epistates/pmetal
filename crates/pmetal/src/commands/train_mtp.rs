use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use pmetal_core::LrSchedulerType;
use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
use pmetal_models::architectures::{
    Gemma4AssistantForCausalLM, Qwen3NextMtpForCausalLM, load_gemma4_assistant_from_dir,
    load_qwen3_next_mtp_from_dir,
};
use pmetal_models::dispatcher::{DynamicModel, ModelArchitecture};
use pmetal_trainer::{
    MtpTrainingConfig, batch_from_tokens, initialize_qwen3_next_mtp_from_target,
    make_gemma4_assistant_config, make_qwen3_next_mtp_config, run_gemma4_assistant_mtp_training,
    run_qwen3_next_mtp_training,
};

use crate::cli::train_mtp::{TrainMtpArgs, TrainMtpFamily};

pub(crate) async fn run_train_mtp(args: TrainMtpArgs) -> Result<()> {
    let config = build_config(&args);
    let batches = shard_batches(
        &args.shards,
        args.seq_len,
        args.batch_size,
        args.eos_token_id,
    )?;
    let target_path = pmetal_hub::resolve_model_path(&args.model, None, None)
        .await
        .map_err(|e| anyhow::anyhow!("resolve target model {}: {e}", args.model))?;
    let arch = ModelArchitecture::detect(&target_path)
        .map_err(|e| anyhow::anyhow!("detect target architecture: {e}"))?;
    let family = resolve_family(args.family, arch)?;

    match family {
        TrainMtpFamily::Qwen3Next => {
            let mut target = match DynamicModel::load(&target_path)
                .map_err(|e| anyhow::anyhow!("load target model: {e}"))?
            {
                DynamicModel::Qwen3Next(model) => model,
                other => anyhow::bail!(
                    "train-mtp qwen3-next requires Qwen3Next target, got {:?}",
                    other.architecture()
                ),
            };
            pmetal_bridge::compat::random::seed(args.seed);
            let mut mtp = if let Some(init) = args.init_mtp.as_deref() {
                let init_path = pmetal_hub::resolve_model_path(init, None, None)
                    .await
                    .map_err(|e| anyhow::anyhow!("resolve init MTP {init}: {e}"))?;
                load_qwen3_next_mtp_from_dir(&init_path, &target.config)
                    .map_err(|e| anyhow::anyhow!("load init Qwen MTP: {e}"))?
            } else {
                let mtp_config = make_qwen3_next_mtp_config(&target.config, args.mtp_layers);
                let mut mtp = Qwen3NextMtpForCausalLM::new(mtp_config)
                    .map_err(|e| anyhow::anyhow!("create Qwen MTP model: {e}"))?;
                initialize_qwen3_next_mtp_from_target(&target, &mut mtp);
                mtp
            };
            let result = run_qwen3_next_mtp_training(&mut target, &mut mtp, &config, batches)
                .map_err(|e| anyhow::anyhow!("train Qwen MTP: {e}"))?;
            println!(
                "Qwen MTP checkpoint saved to {} (final_loss={:.4})",
                result.output_dir.display(),
                result.final_loss.unwrap_or(f32::NAN)
            );
        }
        TrainMtpFamily::Gemma4 => {
            let mut target = match DynamicModel::load(&target_path)
                .map_err(|e| anyhow::anyhow!("load target model: {e}"))?
            {
                DynamicModel::Gemma4(model) => model,
                other => anyhow::bail!(
                    "train-mtp gemma4 requires Gemma4 target, got {:?}",
                    other.architecture()
                ),
            };
            pmetal_bridge::compat::random::seed(args.seed);
            let mut assistant = if let Some(init) = args.init_mtp.as_deref() {
                let init_path = pmetal_hub::resolve_model_path(init, None, None)
                    .await
                    .map_err(|e| anyhow::anyhow!("resolve init assistant {init}: {e}"))?;
                let (assistant, _) = load_gemma4_assistant_from_dir(&init_path)
                    .map_err(|e| anyhow::anyhow!("load init Gemma assistant: {e}"))?;
                assistant
            } else {
                let assistant_config =
                    make_gemma4_assistant_config(&target.config, args.assistant_layers)
                        .map_err(|e| anyhow::anyhow!("create Gemma assistant config: {e}"))?;
                Gemma4AssistantForCausalLM::new(assistant_config)
                    .map_err(|e| anyhow::anyhow!("create Gemma assistant: {e}"))?
            };
            let result =
                run_gemma4_assistant_mtp_training(&mut target, &mut assistant, &config, batches)
                    .map_err(|e| anyhow::anyhow!("train Gemma assistant MTP: {e}"))?;
            println!(
                "Gemma assistant checkpoint saved to {} (final_loss={:.4})",
                result.output_dir.display(),
                result.final_loss.unwrap_or(f32::NAN)
            );
        }
        TrainMtpFamily::Auto => unreachable!("auto is resolved before dispatch"),
    }
    Ok(())
}

fn resolve_family(requested: TrainMtpFamily, arch: ModelArchitecture) -> Result<TrainMtpFamily> {
    match requested {
        TrainMtpFamily::Auto => match arch {
            ModelArchitecture::Qwen3Next => Ok(TrainMtpFamily::Qwen3Next),
            ModelArchitecture::Gemma4 => Ok(TrainMtpFamily::Gemma4),
            other => {
                anyhow::bail!("train-mtp auto supports Qwen3Next/Qwen3.6 and Gemma4; got {other:?}")
            }
        },
        TrainMtpFamily::Qwen3Next => Ok(TrainMtpFamily::Qwen3Next),
        TrainMtpFamily::Gemma4 => Ok(TrainMtpFamily::Gemma4),
    }
}

fn build_config(args: &TrainMtpArgs) -> MtpTrainingConfig {
    MtpTrainingConfig {
        num_steps: args.steps,
        learning_rate: args.learning_rate,
        min_lr: args.min_lr,
        warmup_steps: args.warmup_steps,
        lr_schedule: parse_lr_schedule(&args.lr_schedule),
        weight_decay: args.weight_decay,
        max_grad_norm: if args.max_grad_norm > 0.0 {
            Some(args.max_grad_norm)
        } else {
            None
        },
        checkpoint_every: if args.checkpoint_every > 0 {
            Some(args.checkpoint_every)
        } else {
            None
        },
        checkpoint_dir: PathBuf::from(&args.output),
        log_every: args.log_every,
        num_assistant_tokens: args.num_assistant_tokens.max(1),
        ..Default::default()
    }
}

fn parse_lr_schedule(value: &str) -> LrSchedulerType {
    match value.to_ascii_lowercase().as_str() {
        "constant" => LrSchedulerType::Constant,
        "linear" => LrSchedulerType::Linear,
        _ => LrSchedulerType::Cosine,
    }
}

fn shard_batches(
    shards: &[String],
    seq_len: usize,
    batch_size: usize,
    eos_token_id: u32,
) -> Result<impl Iterator<Item = pmetal_bridge::compat::Array>> {
    anyhow::ensure!(!shards.is_empty(), "--shards must not be empty");
    anyhow::ensure!(seq_len >= 2, "--seq-len must be >= 2");
    let reader = StreamingShardReader::new(StreamConfig {
        shard_paths: shards
            .iter()
            .map(Path::new)
            .map(Path::to_path_buf)
            .collect(),
        seq_len,
        batch_size: batch_size.max(1),
        eos_token_id,
        resume_from: None,
    })
    .context("open tokenized shards")?;
    Ok(reader.map(|(tokens, _)| {
        batch_from_tokens(&tokens).expect("streaming shard reader must yield rectangular batches")
    }))
}
