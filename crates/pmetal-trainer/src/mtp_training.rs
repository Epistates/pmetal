//! MTP / draft-model training and checkpoint export.
//!
//! This module is intentionally small and architecture-specific. It trains only
//! the predictor/draft model while the target model is frozen, then exports a
//! runtime-loadable checkpoint layout:
//! - Gemma 4 assistant: `config.json`, `generation_config.json`, `model.safetensors`
//! - Qwen3Next/Qwen3.6 MTP: `config.json`, `generation_config.json`, `model.safetensors`
//! - DFlash draft: `config.json`, `model.safetensors`

use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{
    Array, Exception, FlattenedModuleParam, Module, ModuleParameters, ModuleParametersExt, Param,
    nn, ops,
    optimizers::{AdamW, AdamWBuilder, Optimizer},
};
use pmetal_core::{LearningRateScheduler, LrSchedulerType};
use pmetal_mlx::{
    kv_cache::{KVCache, KVCacheConfig},
    speculative::SpecCapture,
};
use pmetal_models::architectures::{
    DFlashDraftConfig, DFlashDraftModel, Gemma4AssistantConfig, Gemma4AssistantForCausalLM,
    Gemma4AssistantGenerationConfig, Gemma4AssistantSharedKvStates, Gemma4Config,
    Gemma4ForCausalLM, Qwen3ForCausalLM, Qwen3NextConfig, Qwen3NextForCausalLM,
    Qwen3NextMtpForCausalLM,
};
use pmetal_models::{validate_gemma4_mtp_pair, validate_qwen3_next_mtp_pair};
use serde::{Deserialize, Serialize};

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

pub fn qwen3_next_mtp_loss(
    target: &mut Qwen3NextForCausalLM,
    mtp: &mut Qwen3NextMtpForCausalLM,
    input_ids: &Array,
) -> Result<Array, Exception> {
    if input_ids.ndim() != 2 || input_ids.dim(1) < 2 {
        return Err(Exception::custom(format!(
            "qwen3_next_mtp_loss expects input_ids [B, T>=2], got {:?}",
            input_ids.shape()
        )));
    }
    let batch = input_ids.dim(0);
    let seq = input_ids.dim(1);
    let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
    let (target_hidden, target_logits) =
        target.forward_hidden_with_capture(input_ids, None, None, None, &mut capture)?;
    target_hidden.eval();

    let input_tokens = input_ids.slice(&[0, 0], &[batch, seq - 1]);
    let hidden = target_hidden.slice(&[0, 0, 0], &[batch, seq - 1, target_hidden.dim(2)]);
    let teacher_logits = target_logits.slice(&[0, 0, 0], &[batch, seq - 1, target_logits.dim(2)]);
    let labels = ops::argmax_axis(&teacher_logits, -1);

    let layers = mtp.config.mtp_num_hidden_layers().max(1);
    let mut total: Option<Array> = None;
    for layer_idx in 0..layers {
        let (_, logits) = mtp.forward_logits(&input_tokens, &hidden, None, None, layer_idx)?;
        let loss = pmetal_bridge::training::cross_entropy_loss(&logits, &labels, -100);
        total = Some(match total {
            Some(acc) => acc.add(&loss),
            None => loss,
        });
    }
    Ok(total
        .ok_or_else(|| Exception::custom("qwen3_next_mtp_loss produced no layer losses"))?
        .divide(&Array::from_f32(layers as f32)))
}

pub fn gemma4_assistant_mtp_loss(
    target: &mut Gemma4ForCausalLM,
    assistant: &mut Gemma4AssistantForCausalLM,
    input_ids: &Array,
) -> Result<Array, Exception> {
    validate_gemma4_mtp_pair(target, assistant)?;
    if input_ids.ndim() != 2 || input_ids.dim(1) < 2 {
        return Err(Exception::custom(format!(
            "gemma4_assistant_mtp_loss expects input_ids [B, T>=2], got {:?}",
            input_ids.shape()
        )));
    }
    let batch = input_ids.dim(0);
    let seq = input_ids.dim(1);
    let mut total: Option<Array> = None;

    for pos in 0..(seq - 1) {
        let prefix = input_ids.slice(&[0, 0], &[batch, pos + 1]);
        let mut cache = gemma4_target_cache(target, seq as usize);
        let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
        let (hidden, _) =
            target.forward_hidden_with_capture(&prefix, None, Some(&mut cache), &mut capture)?;
        hidden.eval();
        let last_hidden = hidden.slice(&[0, pos, 0], &[batch, pos + 1, hidden.dim(2)]);
        let token = input_ids.slice(&[0, pos], &[batch, pos + 1]);
        let token_embed = Module::forward(&mut target.model.embed_tokens, &token)?;
        let inputs_embeds = ops::concatenate_axis(&[&token_embed, &last_hidden], -1);
        let shared = gemma4_shared_states_for_assistant(target, assistant, &cache)?;
        let (_, logits) = assistant.forward_logits(&inputs_embeds, &shared, pos)?;
        let labels = input_ids.slice(&[0, pos + 1], &[batch, pos + 2]);
        let vocab = assistant.config.text_config.vocab_size;
        let logits = logits.reshape(&[batch, 1, vocab]);
        let loss = pmetal_bridge::training::cross_entropy_loss(&logits, &labels, -100);
        total = Some(match total {
            Some(acc) => acc.add(&loss),
            None => loss,
        });
    }

    Ok(total
        .ok_or_else(|| Exception::custom("gemma4_assistant_mtp_loss produced no position losses"))?
        .divide(&Array::from_f32((seq - 1) as f32)))
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
    validate_qwen3_next_mtp_pair(target, mtp)?;
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
            if block_size < 2 || block_size > seq {
                return Err(Exception::custom(format!(
                    "DFlash block_size {block_size} must be in 2..={seq}"
                )));
            }
            let max_start = seq - block_size;
            let block_start = if max_start == 0 {
                0
            } else {
                step % (max_start + 1)
            };
            let out = dflash_train_step(
                target,
                model,
                batch,
                &DFlashTrainStepConfig {
                    block_size,
                    block_start,
                },
            )?;
            Ok(out.loss)
        },
        |dir, model, optimizer, meta| save_dflash_draft_checkpoint(dir, model, optimizer, meta),
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
    std::fs::create_dir_all(&config.checkpoint_dir)
        .map_err(|e| Exception::custom(format!("create output dir: {e}")))?;
    let mut optimizer = AdamWBuilder::new(config.learning_rate)
        .weight_decay(config.weight_decay)
        .betas(config.betas)
        .eps(config.eps)
        .build()?;
    let scheduler = LearningRateScheduler::new(
        config.learning_rate as f64,
        config.num_steps,
        config.warmup_steps,
        config.lr_schedule,
    )
    .with_min_lr(config.min_lr as f64);

    let mut losses = Vec::with_capacity(config.num_steps);
    let start = std::time::Instant::now();
    for step in 0..config.num_steps {
        let lr = scheduler.get_lr(step) as f32;
        optimizer.set_lr(lr);
        let batch = batches.next().ok_or_else(|| {
            Exception::custom(format!("MTP batch iterator exhausted at step {step}"))
        })?;

        let mut vag = nn::value_and_grad(|model: &mut M, batch: &B| loss_fn(model, batch));
        let (loss, mut grads) = vag(model, &batch)?;
        if let Some(max_norm) = config.max_grad_norm {
            pmetal_bridge::training::clip_grad_norm_map(&mut grads, max_norm);
        }
        optimizer.update(model, grads)?;
        pmetal_bridge::compat::eval_params(model.trainable_parameters())?;

        loss.eval();
        let loss_value = loss.item::<f32>();
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
    write_json(dir.join("config.json"), config)?;
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

fn gemma4_shared_states_for_assistant(
    target: &Gemma4ForCausalLM,
    assistant: &Gemma4AssistantForCausalLM,
    cache: &KVCache,
) -> Result<Gemma4AssistantSharedKvStates, Exception> {
    let target_config = &target.config;
    let assistant_config = &assistant.config.text_config;
    let total = target_config.num_hidden_layers.max(0) as usize;
    let first_shared = target_config.first_kv_shared_layer_idx().min(total);
    let mut full_attention = None;
    let mut sliding_attention = None;
    for layer_idx in 0..first_shared {
        if target_config.is_full_attention(layer_idx) {
            full_attention = Some(layer_idx);
        } else {
            sliding_attention = Some(layer_idx);
        }
    }
    for layer_idx in 0..assistant_config.num_hidden_layers.max(0) as usize {
        let source = if assistant_config.is_full_attention(layer_idx) {
            full_attention
        } else {
            sliding_attention
        };
        if source.is_none() {
            return Err(Exception::custom(format!(
                "Gemma assistant layer {layer_idx} has no matching target KV source"
            )));
        }
    }
    Ok(Gemma4AssistantSharedKvStates {
        full_attention: full_attention
            .map(|idx| {
                cache.get(idx).ok_or_else(|| {
                    Exception::custom(format!(
                        "Gemma target cache missing full KV source layer {idx}"
                    ))
                })
            })
            .transpose()?,
        sliding_attention: sliding_attention
            .map(|idx| {
                cache.get(idx).ok_or_else(|| {
                    Exception::custom(format!(
                        "Gemma target cache missing sliding KV source layer {idx}"
                    ))
                })
            })
            .transpose()?,
    })
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
    Array::save_safetensors(path_string(path.as_ref())?.as_str(), &entries);
    Ok(())
}

fn save_optimizer_state(dir: &Path, optimizer: &AdamW) -> Result<(), Exception> {
    let mut entries_owned = Vec::with_capacity(optimizer.state().len() * 2);
    for (key, (m, v)) in optimizer.state() {
        m.eval();
        v.eval();
        entries_owned.push((format!("{key}.__m"), m.clone()));
        entries_owned.push((format!("{key}.__v"), v.clone()));
    }
    let entries: Vec<(&str, &Array)> = entries_owned
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    Array::save_safetensors(
        path_string(&dir.join("optimizer.safetensors"))?.as_str(),
        &entries,
    );
    Ok(())
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
    use pmetal_models::architectures::DFlashExtras;

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
            rope_theta: 10_000.0,
            head_dim: 8,
            tie_word_embeddings: false,
            attention_bias: false,
            rope_scaling: None,
            block_size: 4,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![0],
                mask_token_id: 7,
            },
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
            pmetal_models::dflash_decoder::load_dflash_draft_from_dir(dir.path()).unwrap();
        assert_eq!(loaded.block_size(), 4);
        assert!(report.loaded > 0);
    }
}
