use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use pmetal_core::LrSchedulerType;
use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::DFlashDraftModel;
use pmetal_models::architectures::dflash_draft::DFlashDraftConfig;
use pmetal_models::dflash_decoder::load_dflash_draft_from_dir;
use pmetal_trainer::{MtpTrainingConfig, batch_from_tokens, run_dflash_draft_training};

use crate::cli::train_draft::TrainDraftArgs;

pub(crate) async fn run_train_draft(args: TrainDraftArgs) -> Result<()> {
    anyhow::ensure!(
        args.draft.is_some() || args.draft_config.is_some(),
        "train-draft requires --draft or --draft-config"
    );
    let config = build_config(&args);
    let batches = shard_batches(
        &args.shards,
        args.seq_len,
        args.batch_size,
        args.eos_token_id,
    )?;
    let target_path = pmetal_hub::resolve_model_path(&args.target, None, None)
        .await
        .map_err(|e| anyhow::anyhow!("resolve target model {}: {e}", args.target))?;
    let mut target = match DynamicModel::load(&target_path)
        .map_err(|e| anyhow::anyhow!("load target model: {e}"))?
    {
        DynamicModel::Qwen3(model) => model,
        other => anyhow::bail!(
            "train-draft requires a Qwen3 target for the current DFlash objective, got {:?}",
            other.architecture()
        ),
    };

    let mut draft = if let Some(draft) = args.draft.as_deref() {
        let draft_path = pmetal_hub::resolve_model_path(draft, None, None)
            .await
            .map_err(|e| anyhow::anyhow!("resolve draft checkpoint {draft}: {e}"))?;
        let (draft, report) = load_dflash_draft_from_dir(&draft_path)
            .map_err(|e| anyhow::anyhow!("load DFlash draft: {e}"))?;
        if !report.skipped.is_empty() {
            eprintln!(
                "DFlash draft loaded with {} unused tensor key(s)",
                report.skipped.len()
            );
        }
        draft
    } else {
        let config_path = args
            .draft_config
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("train-draft requires --draft or --draft-config"))?;
        let config_bytes = std::fs::read(config_path)
            .with_context(|| format!("read DFlash config {config_path}"))?;
        let config: DFlashDraftConfig = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("parse DFlash config {config_path}"))?;
        pmetal_bridge::compat::random::seed(args.seed);
        DFlashDraftModel::new(config).map_err(|e| anyhow::anyhow!("create DFlash draft: {e}"))?
    };

    let result = run_dflash_draft_training(&mut target, &mut draft, &config, batches)
        .map_err(|e| anyhow::anyhow!("train DFlash draft: {e}"))?;
    println!(
        "DFlash draft checkpoint saved to {} (final_loss={:.4})",
        result.output_dir.display(),
        result.final_loss.unwrap_or(f32::NAN)
    );
    Ok(())
}

fn build_config(args: &TrainDraftArgs) -> MtpTrainingConfig {
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
