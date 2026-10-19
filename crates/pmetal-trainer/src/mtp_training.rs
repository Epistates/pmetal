//! MTP / draft-model training and checkpoint export.
//!
//! This module is intentionally small and architecture-specific. It trains only
//! the predictor/draft model while the target model is frozen, then exports a
//! runtime-loadable checkpoint layout:
//! - Gemma 4 assistant: `config.json`, `generation_config.json`, `model.safetensors`
//! - Qwen3Next/Qwen3.6 MTP: `config.json`, `generation_config.json`, `model.safetensors`
//! - DFlash draft: `config.json`, `model.safetensors`

use std::path::{Path, PathBuf};

use pmetal_cache::{
    kv_cache::{KVCache, KVCacheConfig},
    speculative::SpecCapture,
};
use pmetal_core::{LearningRateScheduler, LrSchedulerType};
use pmetal_models::architectures::{
    DFlashDraftModel, Gemma4AssistantConfig, Gemma4AssistantForCausalLM,
    Gemma4AssistantGenerationConfig, Gemma4Config, Gemma4ForCausalLM, Qwen3ForCausalLM,
    Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextMtpForCausalLM,
};
use pmetal_models::{Gemma4MtpKvSources, validate_gemma4_mtp_pair, validate_qwen3_next_mtp_pair};
use pmetal_runtime::compat::{
    Array, Exception, ModuleParameters, ModuleParametersExt, Param, nn, ops,
    optimizers::{AdamW, AdamWBuilder, Optimizer, StatefulOptimizer},
};
use serde::{Deserialize, Serialize};

use crate::checkpoint::TrainState;
use crate::dflash_training::{DFlashTrainStepConfig, dflash_train_step};

#[derive(Debug, Clone)]
pub struct MtpTrainingConfig {
    pub num_steps: usize,
    pub learning_rate: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub lr_schedule: LrSchedulerType,
    pub weight_decay: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub max_grad_norm: Option<f32>,
    pub checkpoint_every: Option<usize>,
    pub checkpoint_dir: PathBuf,
    pub log_every: usize,
    pub num_assistant_tokens: usize,
    /// DFlash drafts: block anchors trained per sequence per step, sharing one
    /// target pass.
    pub dflash_blocks_per_step: usize,
    /// DFlash drafts: seed of the anchors' random positions.
    pub seed: u64,
    /// A checkpoint directory of an earlier run to continue: its optimizer
    /// moments, f32 master copies and step are restored and the run picks up
    /// at that step of `num_steps`. The model must already hold that
    /// checkpoint's weights, and `batches` must supply the batches from that
    /// step on; the data position is not recorded.
    pub resume_from: Option<PathBuf>,
}

impl MtpTrainingConfig {
    /// Check the hyperparameters, naming every one that cannot work. Every
    /// run entry point checks them before it touches the output directory.
    pub fn validate(&self) -> Result<(), Exception> {
        let mut problems = Vec::new();
        if self.num_steps == 0 {
            problems.push("num_steps must be at least 1".to_string());
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            problems.push(format!(
                "learning_rate must be positive, got {}",
                self.learning_rate
            ));
        }
        if !(self.min_lr.is_finite() && self.min_lr >= 0.0 && self.min_lr <= self.learning_rate) {
            problems.push(format!(
                "min_lr must be in [0, learning_rate], got {}",
                self.min_lr
            ));
        }
        for (name, beta) in [("beta1", self.betas.0), ("beta2", self.betas.1)] {
            if !(0.0..1.0).contains(&beta) {
                problems.push(format!("{name} must be in [0, 1), got {beta}"));
            }
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            problems.push(format!("eps must be positive, got {}", self.eps));
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            problems.push(format!(
                "weight_decay must not be negative, got {}",
                self.weight_decay
            ));
        }
        if let Some(norm) = self.max_grad_norm {
            if !(norm.is_finite() && norm > 0.0) {
                problems.push(format!("max_grad_norm must be positive, got {norm}"));
            }
        }
        if self.num_assistant_tokens == 0 {
            problems.push("num_assistant_tokens must be at least 1".to_string());
        }
        if self.dflash_blocks_per_step == 0 {
            problems.push("dflash_blocks_per_step must be at least 1".to_string());
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Exception::custom(format!(
                "invalid MTP training configuration: {}",
                problems.join("; ")
            )))
        }
    }
}

impl Default for MtpTrainingConfig {
    fn default() -> Self {
        Self {
            num_steps: 1000,
            learning_rate: 2e-4,
            min_lr: 1e-5,
            warmup_steps: 100,
            lr_schedule: LrSchedulerType::Cosine,
            weight_decay: 0.01,
            betas: (0.9, 0.95),
            eps: 1e-8,
            max_grad_norm: Some(1.0),
            checkpoint_every: Some(500),
            checkpoint_dir: PathBuf::from("./mtp-output"),
            log_every: 10,
            num_assistant_tokens: 6,
            dflash_blocks_per_step: 4,
            seed: 42,
            resume_from: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpCheckpointMeta {
    pub step: u64,
    pub loss: f32,
    pub learning_rate: f32,
    pub checkpoint_type: String,
}

#[derive(Debug, Clone)]
pub struct MtpTrainingResult {
    pub losses: Vec<f32>,
    pub final_loss: Option<f32>,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub enum MtpCheckpointKind {
    Qwen3Next,
    Gemma4Assistant,
    DFlashDraft,
}

impl MtpCheckpointKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3Next => "qwen3_next_mtp",
            Self::Gemma4Assistant => "gemma4_assistant_mtp",
            Self::DFlashDraft => "dflash_draft",
        }
    }
}

pub fn make_qwen3_next_mtp_config(target: &Qwen3NextConfig, mtp_layers: usize) -> Qwen3NextConfig {
    let mut config = target.clone();
    let layers = mtp_layers.max(1) as i32;
    config.mtp_num_hidden_layers = Some(layers);
    config.num_nextn_predict_layers = Some(layers);
    config
}

pub fn make_gemma4_assistant_config(
    target: &Gemma4Config,
    assistant_layers: usize,
) -> Result<Gemma4AssistantConfig, Exception> {
    let first_shared = target.first_kv_shared_layer_idx();
    let mut available = Vec::new();
    for layer_idx in 0..first_shared {
        let kind = if target.is_full_attention(layer_idx) {
            "full_attention"
        } else {
            "sliding_attention"
        };
        if !available.iter().any(|item| item == kind) {
            available.push(kind.to_string());
        }
    }
    if available.is_empty() {
        return Err(Exception::custom(
            "Gemma 4 assistant training requires at least one non-shared target KV source layer",
        ));
    }

    let layer_count = assistant_layers.max(1);
    let mut text_config = target.clone();
    text_config.model_type = "gemma4_assistant_text".to_string();
    text_config.num_hidden_layers = layer_count as i32;
    text_config.num_kv_shared_layers = Some(layer_count as i32);
    text_config.layer_types = (0..layer_count)
        .map(|idx| available[idx % available.len()].clone())
        .collect();

    Ok(Gemma4AssistantConfig {
        model_type: "gemma4_assistant".to_string(),
        text_config,
        backbone_hidden_size: target.hidden_size,
        use_ordered_embeddings: false,
        num_centroids: 2048,
        centroid_intermediate_top_k: 32,
        tie_word_embeddings: false,
    })
}

pub fn initialize_qwen3_next_mtp_from_target(
    target: &Qwen3NextForCausalLM,
    mtp: &mut Qwen3NextMtpForCausalLM,
) {
    mtp.model.embed_tokens.weight = Param::new(target.model.embed_tokens.weight.value.clone());
    if let (Some(target_head), Some(mtp_head)) = (target.lm_head.as_ref(), mtp.lm_head.as_mut()) {
        mtp_head.weight = Param::new(target_head.weight.value.clone());
    }
}

/// The predictor's loss on `input_ids`: at each position `i` it reads the
/// target's hidden state `h[i]` beside the embedding of the next token
/// `x[i + 1]`, as it does when drafting, and learns the target's own choice
/// for `x[i + 2]`.
pub fn qwen3_next_mtp_loss(
    target: &mut Qwen3NextForCausalLM,
    mtp: &mut Qwen3NextMtpForCausalLM,
    input_ids: &Array,
) -> Result<Array, Exception> {
    if input_ids.ndim() != 2 || input_ids.dim(1) < 3 {
        return Err(Exception::custom(format!(
            "qwen3_next_mtp_loss expects input_ids [B, T>=3], got {:?}",
            input_ids.shape()
        )));
    }
    let batch = input_ids.dim(0);
    let seq = input_ids.dim(1);
    let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
    let (target_hidden, target_logits) =
        target.forward_hidden_with_capture(input_ids, None, None, None, &mut capture)?;
    target_hidden.eval();

    let input_tokens = input_ids.slice(&[0, 1], &[batch, seq - 1]);
    let hidden = target_hidden.slice(&[0, 0, 0], &[batch, seq - 2, target_hidden.dim(2)]);
    let teacher_logits = target_logits.slice(&[0, 1, 0], &[batch, seq - 1, target_logits.dim(2)]);
    let labels = ops::argmax_axis(&teacher_logits, -1);

    let layers = mtp.config.mtp_num_hidden_layers().max(1);
    let mut total: Option<Array> = None;
    for layer_idx in 0..layers {
        let (_, logits) = mtp.forward_logits(&input_tokens, &hidden, None, None, layer_idx)?;
        let loss = pmetal_runtime::training::cross_entropy_loss(&logits, &labels, -100);
        total = Some(match total {
            Some(acc) => acc.add(&loss),
            None => loss,
        });
    }
    Ok(total
        .ok_or_else(|| Exception::custom("qwen3_next_mtp_loss produced no layer losses"))?
        .divide(&Array::from_f32(layers as f32)))
}

/// The assistant's loss on `input_ids`: at each position `i` it reads the
/// target's hidden state `h[i]` beside the scaled embedding of the next token
/// `x[i + 1]`, attends to the target's cache up to `i` and runs at position
/// `i + 1`, as it does when drafting, and learns `x[i + 2]`.
pub fn gemma4_assistant_mtp_loss(
    target: &mut Gemma4ForCausalLM,
    assistant: &mut Gemma4AssistantForCausalLM,
    input_ids: &Array,
) -> Result<Array, Exception> {
    validate_gemma4_mtp_pair(target, assistant)?;
    let kv_sources = Gemma4MtpKvSources::new(&target.config, &assistant.config.text_config)?;
    if input_ids.ndim() != 2 || input_ids.dim(1) < 3 {
        return Err(Exception::custom(format!(
            "gemma4_assistant_mtp_loss expects input_ids [B, T>=3], got {:?}",
            input_ids.shape()
        )));
    }
    let batch = input_ids.dim(0);
    let seq = input_ids.dim(1);
    let mut total: Option<Array> = None;

    for pos in 0..(seq - 2) {
        let prefix = input_ids.slice(&[0, 0], &[batch, pos + 1]);
        let mut cache = gemma4_target_cache(target, seq as usize);
        let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
        let (hidden, _) =
            target.forward_hidden_with_capture(&prefix, None, Some(&mut cache), &mut capture)?;
        hidden.eval();
        let last_hidden = hidden.slice(&[0, pos, 0], &[batch, pos + 1, hidden.dim(2)]);
        let token = input_ids.slice(&[0, pos + 1], &[batch, pos + 2]);
        let token_embed = target.model.embed_scaled(&token);
        let inputs_embeds = ops::concatenate_axis(&[&token_embed, &last_hidden], -1);
        let shared = kv_sources.shared_states(&cache)?;
        let (_, logits) = assistant.forward_logits(&inputs_embeds, &shared, pos + 1)?;
        let labels = input_ids.slice(&[0, pos + 2], &[batch, pos + 3]);
        let vocab = assistant.config.text_config.vocab_size;
        let logits = logits.reshape(&[batch, 1, vocab]);
        let loss = pmetal_runtime::training::cross_entropy_loss(&logits, &labels, -100);
        total = Some(match total {
            Some(acc) => acc.add(&loss),
            None => loss,
        });
    }

    Ok(total
        .ok_or_else(|| Exception::custom("gemma4_assistant_mtp_loss produced no position losses"))?
        .divide(&Array::from_f32((seq - 2) as f32)))
}

pub fn run_qwen3_next_mtp_training<I>(
    target: &mut Qwen3NextForCausalLM,
    mtp: &mut Qwen3NextMtpForCausalLM,
    config: &MtpTrainingConfig,
    batches: I,
) -> Result<MtpTrainingResult, Exception>
where
    I: Iterator<Item = Array>,
{
    validate_qwen3_next_mtp_pair(&target.config, mtp)?;
    let mtp_config = mtp.config.clone();
    run_training_loop(
        mtp,
        config,
        batches,
        MtpCheckpointKind::Qwen3Next,
        |model, batch| qwen3_next_mtp_loss(target, model, batch),
        |dir, model, optimizer, meta| {
            save_qwen3_next_mtp_checkpoint(dir, model, &mtp_config, optimizer, meta)
        },
    )
}

pub fn run_gemma4_assistant_mtp_training<I>(
    target: &mut Gemma4ForCausalLM,
    assistant: &mut Gemma4AssistantForCausalLM,
    config: &MtpTrainingConfig,
    batches: I,
) -> Result<MtpTrainingResult, Exception>
where
    I: Iterator<Item = Array>,
{
    validate_gemma4_mtp_pair(target, assistant)?;
    run_training_loop(
        assistant,
        config,
        batches,
        MtpCheckpointKind::Gemma4Assistant,
        |model, batch| gemma4_assistant_mtp_loss(target, model, batch),
        |dir, model, optimizer, meta| {
            save_gemma4_assistant_checkpoint(
                dir,
                model,
                &Gemma4AssistantGenerationConfig {
                    num_assistant_tokens: config.num_assistant_tokens.max(1),
                },
                optimizer,
                meta,
            )
        },
    )
}

pub fn run_dflash_draft_training<I>(
    target: &mut Qwen3ForCausalLM,
    draft: &mut DFlashDraftModel,
    config: &MtpTrainingConfig,
    batches: I,
) -> Result<MtpTrainingResult, Exception>
where
    I: Iterator<Item = Array>,
{
    run_training_loop(
        draft,
        config,
        batches.enumerate(),
        MtpCheckpointKind::DFlashDraft,
        |model, (step, batch)| {
            let seq = batch.dim(1) as usize;
            let block_size = model.block_size();
            // An anchor needs a token of context before it and a whole block after.
            if block_size < 2 || block_size + 1 > seq {
                return Err(Exception::custom(format!(
                    "DFlash block_size {block_size} needs sequences of at least {} tokens, got {seq}",
                    block_size + 1
                )));
            }
            let anchors = config.dflash_blocks_per_step.max(1);
            let block_starts = (0..anchors)
                .map(|i| {
                    let draw = pmetal_runtime::decode::draw_bits(
                        config.seed,
                        (*step * anchors + i) as u64,
                    );
                    1 + (draw % (seq - block_size) as u64) as usize
                })
                .collect();
            let out = dflash_train_step(
                target,
                model,
                batch,
                &DFlashTrainStepConfig {
                    block_size,
                    block_starts,
                    position_decay: None,
                },
            )?;
            Ok(out.loss)
        },
        save_dflash_draft_checkpoint,
    )
}

fn run_training_loop<M, I, B, F, S>(
    model: &mut M,
    config: &MtpTrainingConfig,
    mut batches: I,
    kind: MtpCheckpointKind,
    mut loss_fn: F,
    mut save_fn: S,
) -> Result<MtpTrainingResult, Exception>
where
    M: ModuleParameters,
    I: Iterator<Item = B>,
    F: FnMut(&mut M, &B) -> Result<Array, Exception>,
    S: FnMut(&Path, &M, &AdamW, &MtpCheckpointMeta) -> Result<(), Exception>,
{
    config.validate()?;
    std::fs::create_dir_all(&config.checkpoint_dir)
        .map_err(|e| Exception::custom(format!("create output dir: {e}")))?;
    let mut optimizer = AdamWBuilder::new(config.learning_rate)
        .weight_decay(config.weight_decay)
        .betas(config.betas)
        .eps(config.eps)
        .build()?;
    let mut first_step = 0;
    if let Some(dir) = &config.resume_from {
        load_optimizer_state(dir, &mut optimizer)?;
        first_step = optimizer.step_count() as usize;
        if first_step >= config.num_steps {
            return Err(Exception::custom(format!(
                "{} is at step {first_step}, which leaves nothing of the {} steps asked for",
                dir.display(),
                config.num_steps
            )));
        }
        eprintln!("resumed from {} at step {first_step}", dir.display());
    }
    let scheduler = LearningRateScheduler::new(
        config.learning_rate as f64,
        config.num_steps,
        config.warmup_steps,
        config.lr_schedule,
    )
    .with_min_lr(config.min_lr as f64);

    let mut losses = Vec::with_capacity(config.num_steps);
    let start = std::time::Instant::now();
    for step in first_step..config.num_steps {
        let lr = scheduler.get_lr(step) as f32;
        optimizer.set_lr(lr);
        let batch = batches.next().ok_or_else(|| {
            Exception::custom(format!("MTP batch iterator exhausted at step {step}"))
        })?;

        let mut vag = nn::value_and_grad(|model: &mut M, batch: &B| loss_fn(model, batch));
        let (loss, mut grads) = vag(model, &batch)?;
        if let Some(max_norm) = config.max_grad_norm {
            pmetal_runtime::training::clip_grad_norm_map(&mut grads, max_norm);
        }
        optimizer.update(model, grads)?;
        pmetal_runtime::compat::eval_params(model.trainable_parameters())?;

        loss.eval();
        let loss_value = crate::step_check::check_step(step + 1, loss.item::<f32>())?;
        losses.push(loss_value);

        if config.log_every > 0 && (step + 1) % config.log_every == 0 {
            let elapsed = start.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "step {:>6} | {} loss {:.4} | lr {:.2e} | {:.2} step/s",
                step + 1,
                kind.as_str(),
                loss_value,
                lr,
                (step + 1) as f64 / elapsed
            );
        }

        if let Some(every) = config.checkpoint_every
            && every > 0
            && (step + 1) % every == 0
        {
            let dir = config
                .checkpoint_dir
                .join("checkpoints")
                .join(format!("step_{}", step + 1));
            save_fn(
                &dir,
                model,
                &optimizer,
                &MtpCheckpointMeta {
                    step: optimizer.step_count(),
                    loss: loss_value,
                    learning_rate: lr,
                    checkpoint_type: kind.as_str().to_string(),
                },
            )?;
            eprintln!("checkpoint saved: {}", dir.display());
        }
    }

    let final_loss = losses.last().copied();
    if let Some(loss) = final_loss {
        let lr = scheduler.get_lr(config.num_steps.saturating_sub(1)) as f32;
        save_fn(
            &config.checkpoint_dir,
            model,
            &optimizer,
            &MtpCheckpointMeta {
                step: optimizer.step_count(),
                loss,
                learning_rate: lr,
                checkpoint_type: kind.as_str().to_string(),
            },
        )?;
        eprintln!(
            "final checkpoint saved: {}",
            config.checkpoint_dir.display()
        );
    }

    Ok(MtpTrainingResult {
        losses,
        final_loss,
        output_dir: config.checkpoint_dir.clone(),
    })
}

pub fn save_qwen3_next_mtp_checkpoint(
    dir: &Path,
    model: &Qwen3NextMtpForCausalLM,
    config: &Qwen3NextConfig,
    optimizer: &AdamW,
    meta: &MtpCheckpointMeta,
) -> Result<(), Exception> {
    std::fs::create_dir_all(dir)
        .map_err(|e| Exception::custom(format!("create qwen MTP checkpoint dir: {e}")))?;
    // The export carries the predictor's embedding table as
    // `mtp.embed_tokens.weight`, so its config says so; the loader refuses a
    // checkpoint whose table and flag disagree.
    let mut config = config.clone();
    config.mtp_use_dedicated_embeddings = Some(true);
    write_json(dir.join("config.json"), &config)?;
    write_json(
        dir.join("generation_config.json"),
        &serde_json::json!({ "mtp_draft_tokens": 3 }),
    )?;
    save_model_safetensors_prefixed(dir.join("model.safetensors"), model, qwen_mtp_export_key)?;
    save_optimizer_state(dir, optimizer)?;
    write_json(dir.join("metadata.json"), meta)?;
    Ok(())
}

pub fn save_gemma4_assistant_checkpoint(
    dir: &Path,
    model: &Gemma4AssistantForCausalLM,
    generation_config: &Gemma4AssistantGenerationConfig,
    optimizer: &AdamW,
    meta: &MtpCheckpointMeta,
) -> Result<(), Exception> {
    std::fs::create_dir_all(dir)
        .map_err(|e| Exception::custom(format!("create Gemma assistant checkpoint dir: {e}")))?;
    write_json(dir.join("config.json"), &model.config)?;
    write_json(
        dir.join("generation_config.json"),
        &serde_json::json!({ "num_assistant_tokens": generation_config.num_assistant_tokens.max(1) }),
    )?;
    save_model_safetensors_prefixed(dir.join("model.safetensors"), model, |key| key.to_string())?;
    save_optimizer_state(dir, optimizer)?;
    write_json(dir.join("metadata.json"), meta)?;
    Ok(())
}

pub fn save_dflash_draft_checkpoint(
    dir: &Path,
    model: &DFlashDraftModel,
    optimizer: &AdamW,
    meta: &MtpCheckpointMeta,
) -> Result<(), Exception> {
    std::fs::create_dir_all(dir)
        .map_err(|e| Exception::custom(format!("create DFlash checkpoint dir: {e}")))?;
    write_json(dir.join("config.json"), &model.config)?;
    save_model_safetensors_prefixed(dir.join("model.safetensors"), model, dflash_export_key)?;
    save_optimizer_state(dir, optimizer)?;
    write_json(dir.join("metadata.json"), meta)?;
    Ok(())
}

fn gemma4_target_cache(target: &Gemma4ForCausalLM, max_seq_len: usize) -> KVCache {
    KVCache::new(KVCacheConfig::new(
        target.config.num_hidden_layers.max(0) as usize,
        max_seq_len,
        target.config.num_key_value_heads as usize,
        target.config.head_dim as usize,
    ))
}

fn save_model_safetensors_prefixed<M, F>(
    path: impl AsRef<Path>,
    model: &M,
    map_key: F,
) -> Result<(), Exception>
where
    M: ModuleParameters,
    F: Fn(&str) -> String,
{
    let flat = model.flatten_params();
    for arr in flat.values() {
        arr.eval();
    }
    let entries_owned: Vec<(String, Array)> = flat
        .iter()
        .map(|(key, value)| (map_key(key.as_ref()), value.clone()))
        .collect();
    let entries: Vec<(&str, &Array)> = entries_owned
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    Array::save_safetensors(path_string(path.as_ref())?.as_str(), &entries)?;
    Ok(())
}

/// Write the optimizer's moments, counters and f32 master copies into `dir`,
/// in the layout every trainer shares ([`TrainState`]). The model file holds
/// the weights rounded to bf16, so a run that resumed from only those would
/// drop everything the rounding discarded.
fn save_optimizer_state(dir: &Path, optimizer: &AdamW) -> Result<(), Exception> {
    let state = optimizer.export_state();
    for array in state.tensors.values() {
        array.eval();
    }
    TrainState {
        optimizer: Some(state),
        loader: None,
    }
    .write(dir)
    .map_err(|e| Exception::custom(e.to_string()))
}

/// Restore `optimizer` from the checkpoint at `dir`. The model must already
/// hold that checkpoint's weights, as the f32 master copies are kept against
/// them.
fn load_optimizer_state(dir: &Path, optimizer: &mut AdamW) -> Result<(), Exception> {
    let state = TrainState::load(dir)
        .map_err(|e| Exception::custom(e.to_string()))?
        .and_then(|state| state.optimizer)
        .ok_or_else(|| {
            Exception::custom(format!(
                "{} holds no optimizer state to resume from",
                dir.display()
            ))
        })?;
    optimizer.import_state(&state)
}

fn qwen_mtp_export_key(key: &str) -> String {
    if key == "lm_head.weight" {
        key.to_string()
    } else if let Some(rest) = key.strip_prefix("model.") {
        format!("mtp.{rest}")
    } else {
        key.to_string()
    }
}

fn dflash_export_key(key: &str) -> String {
    key.strip_prefix("model.")
        .map(str::to_string)
        .unwrap_or_else(|| key.to_string())
}

fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<(), Exception> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| Exception::custom(format!("serialize json: {e}")))?;
    std::fs::write(path.as_ref(), json)
        .map_err(|e| Exception::custom(format!("write {}: {e}", path.as_ref().display())))
}

fn path_string(path: &Path) -> Result<String, Exception> {
    path.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| Exception::custom(format!("non-UTF-8 path: {}", path.display())))
}

pub fn batch_from_tokens(tokens: &[Vec<u32>]) -> Result<Array, Exception> {
    let batch = tokens.len();
    let seq = tokens
        .first()
        .map(|row| row.len())
        .ok_or_else(|| Exception::custom("empty token batch"))?;
    if seq < 2 {
        return Err(Exception::custom(
            "MTP training batches require seq_len >= 2",
        ));
    }
    if tokens.iter().any(|row| row.len() != seq) {
        return Err(Exception::custom("ragged token batch"));
    }
    let mut flat = Vec::with_capacity(batch * seq);
    for row in tokens {
        flat.extend_from_slice(row);
    }
    Ok(Array::from_slice(&flat, &[batch as i32, seq as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::{DFlashDraftConfig, DFlashExtras};

    fn tiny_qwen_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            model_type: "qwen3_6_moe_text".to_string(),
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(8),
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 1,
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            mlp_only_layers: Vec::new(),
            norm_topk_prob: false,
            partial_rotary_factor: 1.0,
            attention_bias: false,
            rope_scaling: None,
            rope_parameters: None,
            layer_types: Some(vec![
                "full_attention".to_string(),
                "full_attention".to_string(),
            ]),
            mtp_num_hidden_layers: Some(1),
            num_nextn_predict_layers: Some(1),
            ..Default::default()
        }
    }

    fn tiny_dflash_config() -> DFlashDraftConfig {
        DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 16,
            num_hidden_layers: 1,
            intermediate_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 64,
            max_position_embeddings: 64,
            rope_theta: Some(10_000.0),
            head_dim: 8,
            block_size: 4,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![0],
                mask_token_id: 7,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn qwen_mtp_export_uses_loadable_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_qwen_config();
        let model = Qwen3NextMtpForCausalLM::new(config.clone()).unwrap();
        let optimizer = AdamWBuilder::new(1e-4).build().unwrap();
        let meta = MtpCheckpointMeta {
            step: 0,
            loss: 0.0,
            learning_rate: 1e-4,
            checkpoint_type: "qwen3_next_mtp".to_string(),
        };
        save_qwen3_next_mtp_checkpoint(dir.path(), &model, &config, &optimizer, &meta).unwrap();

        let loaded =
            pmetal_models::architectures::load_qwen3_next_mtp_from_dir(dir.path(), &config)
                .unwrap();
        assert_eq!(loaded.config.mtp_num_hidden_layers(), 1);
        assert!(dir.path().join("model.safetensors").exists());
        assert!(dir.path().join("optimizer.safetensors").exists());
    }

    #[test]
    fn dflash_export_round_trips_through_loader() {
        let dir = tempfile::tempdir().unwrap();
        let model = DFlashDraftModel::new(tiny_dflash_config()).unwrap();
        let optimizer = AdamWBuilder::new(1e-4).build().unwrap();
        let meta = MtpCheckpointMeta {
            step: 0,
            loss: 0.0,
            learning_rate: 1e-4,
            checkpoint_type: "dflash_draft".to_string(),
        };
        save_dflash_draft_checkpoint(dir.path(), &model, &optimizer, &meta).unwrap();

        let (loaded, report) =
            pmetal_models::dflash_drafts::load_dflash_draft_from_dir(dir.path()).unwrap();
        assert_eq!(loaded.block_size(), 4);
        assert!(report.loaded > 0);
    }

    fn loss_value(loss: &Array) -> f32 {
        loss.eval();
        loss.item_f32()
    }

    #[test]
    fn the_training_configuration_is_validated() {
        assert!(MtpTrainingConfig::default().validate().is_ok());
        let bad = MtpTrainingConfig {
            num_steps: 0,
            learning_rate: f32::NAN,
            max_grad_norm: Some(-1.0),
            dflash_blocks_per_step: 0,
            ..Default::default()
        };
        let message = bad.validate().unwrap_err().to_string();
        for needle in [
            "num_steps",
            "learning_rate",
            "max_grad_norm",
            "dflash_blocks",
        ] {
            assert!(message.contains(needle), "{needle} missing: {message}");
        }
    }

    /// A half-precision parameter's f32 master copy is part of the optimizer
    /// checkpoint; an f32 parameter has none.
    #[test]
    fn the_optimizer_checkpoint_carries_the_f32_master_copies() {
        let mut optimizer = AdamWBuilder::new(1e-2).build().unwrap();
        optimizer.advance_step();
        let key: std::rc::Rc<str> = "w".into();
        let mut half = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])
            .as_dtype(pmetal_runtime::compat::Dtype::Bfloat16.as_i32());
        let mut full = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let grad = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.0], &[4]);
        optimizer.update_single(&key, &grad, &mut half).unwrap();
        optimizer
            .update_single(&"f".into(), &grad, &mut full)
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        save_optimizer_state(dir.path(), &optimizer).unwrap();
        let saved = pmetal_runtime::inline_array::load_safetensors_shard(
            dir.path().join("optimizer.safetensors").to_str().unwrap(),
        )
        .unwrap();
        let names: Vec<&str> = saved.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"master/w"), "{names:?}");
        assert!(!names.contains(&"master/f"), "{names:?}");
        let master = &saved.iter().find(|(n, _)| n == "master/w").unwrap().1;
        assert_eq!(
            master.dtype_raw(),
            pmetal_runtime::compat::Dtype::Float32.as_i32()
        );

        // ... and a fresh optimizer restored from it holds the same copy and
        // the same step.
        let mut restored = AdamWBuilder::new(1e-2).build().unwrap();
        load_optimizer_state(dir.path(), &mut restored).unwrap();
        assert_eq!(restored.step_count(), optimizer.step_count());
        let kept = restored.master_weights().count();
        assert_eq!(kept, 1);
    }

    /// A run stopped halfway and resumed from its checkpoint goes on as the
    /// uninterrupted run does: the moments, the f32 masters and the step come
    /// back. Without them the second half restarts Adam's bias correction and
    /// its losses part from the straight run's.
    #[test]
    fn a_resumed_mtp_run_continues_the_uninterrupted_one() {
        let config = tiny_qwen_config();
        let tokens: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32 + 1).collect();
        let batch = batch_from_tokens(&[tokens]).unwrap();
        let settings = |dir: &Path, steps: usize, resume: Option<PathBuf>| MtpTrainingConfig {
            num_steps: steps,
            learning_rate: 5e-3,
            warmup_steps: 0,
            lr_schedule: LrSchedulerType::Constant,
            checkpoint_every: None,
            checkpoint_dir: dir.to_path_buf(),
            log_every: 0,
            resume_from: resume,
            ..Default::default()
        };
        let fresh = || {
            pmetal_runtime::compat::random::seed(9);
            let target = Qwen3NextForCausalLM::new(config.clone()).unwrap();
            let mut mtp = Qwen3NextMtpForCausalLM::new(config.clone()).unwrap();
            initialize_qwen3_next_mtp_from_target(&target, &mut mtp);
            (target, mtp)
        };

        let straight_dir = tempfile::tempdir().unwrap();
        let (mut target, mut mtp) = fresh();
        let straight = run_qwen3_next_mtp_training(
            &mut target,
            &mut mtp,
            &settings(straight_dir.path(), 16, None),
            std::iter::repeat_n(batch.clone(), 16),
        )
        .unwrap();

        let half_dir = tempfile::tempdir().unwrap();
        let (mut target, mut mtp) = fresh();
        run_qwen3_next_mtp_training(
            &mut target,
            &mut mtp,
            &settings(half_dir.path(), 8, None),
            std::iter::repeat_n(batch.clone(), 8),
        )
        .unwrap();
        let resumed = run_qwen3_next_mtp_training(
            &mut target,
            &mut mtp,
            &settings(half_dir.path(), 16, Some(half_dir.path().to_path_buf())),
            std::iter::repeat_n(batch.clone(), 8),
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();

        assert_eq!(resumed.losses.len(), 8);
        for (i, (a, b)) in straight.losses[8..].iter().zip(&resumed.losses).enumerate() {
            assert!(
                (a - b).abs() <= 2e-3 * a.abs().max(1.0),
                "step {}: straight {a}, resumed {b}",
                8 + i
            );
        }
    }

    /// The predictor trains: against the target's own next-next choices the
    /// loss falls over a run, the predictor's parameters move, the frozen
    /// target's do not, and the exported checkpoint loads back.
    #[test]
    fn the_qwen_mtp_trains_against_the_frozen_target() {
        pmetal_runtime::compat::random::seed(9);
        let config = tiny_qwen_config();
        let mut target = Qwen3NextForCausalLM::new(config.clone()).unwrap();
        let mut mtp = Qwen3NextMtpForCausalLM::new(config.clone()).unwrap();
        initialize_qwen3_next_mtp_from_target(&target, &mut mtp);
        let before = mtp.model.embed_tokens.weight.value.clone();
        let target_before = target.model.embed_tokens.weight.value.clone();
        let tokens: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32 + 1).collect();
        let batch = batch_from_tokens(&[tokens]).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let result = run_qwen3_next_mtp_training(
            &mut target,
            &mut mtp,
            &MtpTrainingConfig {
                num_steps: 50,
                learning_rate: 5e-3,
                warmup_steps: 0,
                lr_schedule: LrSchedulerType::Constant,
                checkpoint_every: None,
                checkpoint_dir: dir.path().to_path_buf(),
                log_every: 0,
                ..Default::default()
            },
            std::iter::repeat_n(batch, 50),
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        let (first, last) = (result.losses[0], *result.losses.last().unwrap());
        assert!(last < 0.6 * first, "loss {first} -> {last}");
        let moved = |a: &Array, b: &Array| {
            let d = a.subtract(b).abs().max(None);
            d.eval();
            d.item_f32()
        };
        assert!(moved(&mtp.model.embed_tokens.weight.value, &before) > 1e-6);
        assert_eq!(
            moved(&target.model.embed_tokens.weight.value, &target_before),
            0.0
        );
        pmetal_models::architectures::load_qwen3_next_mtp_from_dir(dir.path(), &config).unwrap();
    }

    /// The predictor learns on the pairing it drafts with: the target's state
    /// at `i` beside the next token `x[i + 1]`, against the target's own
    /// choice at `i + 1`, which is its choice for `x[i + 2]`.
    #[test]
    fn the_qwen_mtp_loss_pairs_each_state_with_the_next_token() {
        pmetal_runtime::compat::random::seed(3);
        let config = tiny_qwen_config();
        let mut target = Qwen3NextForCausalLM::new(config.clone()).unwrap();
        let mut mtp = Qwen3NextMtpForCausalLM::new(config).unwrap();
        let ids = pmetal_runtime::decode::prompt_tokens_to_input(&[5, 9, 1, 33, 7, 12]);
        let loss = loss_value(&qwen3_next_mtp_loss(&mut target, &mut mtp, &ids).unwrap());

        let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
        let (states, logits) = target
            .forward_hidden_with_capture(&ids, None, None, None, &mut capture)
            .unwrap();
        let next = ops::slice_axis(&ids, 1, 1, 5);
        let labels = ops::argmax_axis(&ops::slice_axis(&logits, 1, 1, 5), -1);
        let (_, mtp_logits) = mtp
            .forward_logits(&next, &ops::slice_axis(&states, 1, 0, 4), None, None, 0)
            .unwrap();
        let want = loss_value(&pmetal_runtime::training::cross_entropy_loss(
            &mtp_logits,
            &labels,
            -100,
        ));
        assert!((loss - want).abs() < 1e-5, "loss {loss}, by hand {want}");
    }

    fn tiny_gemma_config(layer_types: &[&str], hidden: i32, shared: i32) -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4_text".to_string(),
            vocab_size: 64,
            hidden_size: hidden,
            intermediate_size: 2 * hidden,
            num_hidden_layers: layer_types.len() as i32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            global_head_dim: Some(8),
            num_global_key_value_heads: Some(1),
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            attention_k_eq_v: false,
            tie_word_embeddings: true,
            sliding_window: 16,
            final_logit_softcapping: None,
            layer_types: layer_types.iter().map(|t| t.to_string()).collect(),
            rope_parameters: None,
            _raw_rope_parameters: None,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            hidden_activation: None,
            num_kv_shared_layers: Some(shared),
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(false),
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
        }
    }

    /// The Gemma 4 assistant trains too: the loss falls over a run and its
    /// parameters move while the frozen target's stay put.
    #[test]
    fn the_gemma4_assistant_trains_against_the_frozen_target() {
        pmetal_runtime::compat::random::seed(12);
        let mut target = Gemma4ForCausalLM::new(tiny_gemma_config(
            &["sliding_attention", "full_attention"],
            32,
            0,
        ))
        .unwrap();
        let mut assistant = Gemma4AssistantForCausalLM::new(Gemma4AssistantConfig {
            model_type: "gemma4_assistant".to_string(),
            text_config: tiny_gemma_config(&["sliding_attention", "full_attention"], 16, 2),
            backbone_hidden_size: 32,
            use_ordered_embeddings: false,
            num_centroids: 2048,
            centroid_intermediate_top_k: 32,
            tie_word_embeddings: true,
        })
        .unwrap();
        let before: Vec<(String, Array)> = assistant
            .flatten_params()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let target_before = target.model.embed_tokens.weight.value.clone();
        let batch = batch_from_tokens(&[vec![5u32, 9, 1, 33, 7, 12]]).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let result = run_gemma4_assistant_mtp_training(
            &mut target,
            &mut assistant,
            &MtpTrainingConfig {
                num_steps: 40,
                learning_rate: 5e-3,
                warmup_steps: 0,
                lr_schedule: LrSchedulerType::Constant,
                checkpoint_every: None,
                checkpoint_dir: dir.path().to_path_buf(),
                log_every: 0,
                ..Default::default()
            },
            std::iter::repeat_n(batch, 40),
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        let (first, last) = (result.losses[0], *result.losses.last().unwrap());
        assert!(last < 0.7 * first, "loss {first} -> {last}");
        let after = assistant.flatten_params();
        let moved = before.iter().any(|(k, b)| {
            let d = after[k.as_str()].subtract(b).abs().max(None);
            d.eval();
            d.item_f32() > 1e-6
        });
        assert!(moved, "no assistant parameter moved");
        let d = target
            .model
            .embed_tokens
            .weight
            .value
            .subtract(&target_before)
            .abs()
            .max(None);
        d.eval();
        assert_eq!(d.item_f32(), 0.0);
    }

    /// The assistant learns on the inputs it drafts from: the target's state
    /// at `i` beside its scaled embedding of the next token `x[i + 1]`, the
    /// target's cache up to `i`, position `i + 1`, against `x[i + 2]`.
    #[test]
    fn the_gemma4_assistant_loss_reads_the_scaled_next_token_at_its_position() {
        pmetal_runtime::compat::random::seed(4);
        let mut target = Gemma4ForCausalLM::new(tiny_gemma_config(
            &["sliding_attention", "full_attention"],
            32,
            0,
        ))
        .unwrap();
        let mut assistant = Gemma4AssistantForCausalLM::new(Gemma4AssistantConfig {
            model_type: "gemma4_assistant".to_string(),
            text_config: tiny_gemma_config(&["sliding_attention", "full_attention"], 16, 2),
            backbone_hidden_size: 32,
            use_ordered_embeddings: false,
            num_centroids: 2048,
            centroid_intermediate_top_k: 32,
            tie_word_embeddings: true,
        })
        .unwrap();
        let tokens = [5u32, 9, 1, 33, 7];
        let ids = pmetal_runtime::decode::prompt_tokens_to_input(&tokens);
        let loss =
            loss_value(&gemma4_assistant_mtp_loss(&mut target, &mut assistant, &ids).unwrap());

        let mut total = 0.0;
        for i in 0..tokens.len() - 2 {
            let mut cache = gemma4_target_cache(&target, tokens.len());
            let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
            let prefix = pmetal_runtime::decode::prompt_tokens_to_input(&tokens[..=i]);
            let (states, _) = target
                .forward_hidden_with_capture(&prefix, None, Some(&mut cache), &mut capture)
                .unwrap();
            let state = ops::slice_axis(&states, 1, i as i32, i as i32 + 1);
            let next = pmetal_runtime::decode::prompt_tokens_to_input(&tokens[i + 1..i + 2]);
            let embedding = target.model.embed_scaled(&next);
            let shared = Gemma4MtpKvSources::new(&target.config, &assistant.config.text_config)
                .unwrap()
                .shared_states(&cache)
                .unwrap();
            let (_, logits) = assistant
                .forward_logits(
                    &ops::concatenate_axis(&[&embedding, &state], -1),
                    &shared,
                    i as i32 + 1,
                )
                .unwrap();
            let label = pmetal_runtime::decode::prompt_tokens_to_input(&tokens[i + 2..i + 3]);
            total += loss_value(&pmetal_runtime::training::cross_entropy_loss(
                &logits.reshape(&[1, 1, 64]),
                &label,
                -100,
            ));
        }
        let want = total / (tokens.len() - 2) as f32;
        assert!((loss - want).abs() < 1e-5, "loss {loss}, by hand {want}");
    }
}
