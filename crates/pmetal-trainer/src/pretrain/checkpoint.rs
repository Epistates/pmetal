//! Full-state pretraining checkpoint: model weights + optimizer state + metadata.
//!
//! Saves three files per checkpoint:
//! - `model.safetensors` — all model parameters
//! - `optimizer.safetensors` — the optimizer's tensors, named `<slot>/<param>`:
//!   AdamW's moments (`m`, `v`) and the f32 master copy (`master`) of every
//!   half-precision parameter, without which a resumed bf16 run lost every
//!   update the master had accumulated below bf16's resolution
//! - `metadata.json` — step count, loss, learning rate, the stream position
//!   and the optimizer's counters

use std::path::Path;

use pmetal_data::streaming::StreamPosition;
use pmetal_runtime::InlineArray;
use pmetal_runtime::compat::{
    ModuleParametersExt,
    module::ModuleParameters,
    optimizers::{AdamW, OptimizerState, StatefulOptimizer},
};

use super::PretrainError;

/// Metadata written alongside the checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub step: u64,
    pub loss: f32,
    pub learning_rate: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_position: Option<StreamPosition>,
}

/// The optimizer's counters, beside [`CheckpointMeta`].
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct OptimizerMeta {
    kind: String,
    scalars: std::collections::BTreeMap<String, f64>,
}

#[derive(serde::Serialize)]
struct MetadataOut<'a> {
    #[serde(flatten)]
    meta: &'a CheckpointMeta,
    optimizer: OptimizerMeta,
}

#[derive(serde::Deserialize)]
struct MetadataIn {
    #[serde(flatten)]
    meta: CheckpointMeta,
    #[serde(default)]
    optimizer: Option<OptimizerMeta>,
}

/// Save a full pretraining checkpoint to `dir/`.
pub fn save_checkpoint<M: ModuleParameters>(
    dir: &Path,
    model: &M,
    optimizer: &AdamW,
    meta: &CheckpointMeta,
) -> Result<(), PretrainError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| PretrainError::Checkpoint(format!("mkdir {}: {e}", dir.display())))?;

    // Model weights
    let flat = model.flatten_params();
    let entries: Vec<(&str, &InlineArray)> = flat.iter().map(|(k, v)| (k.as_ref(), v)).collect();
    for (_, arr) in &entries {
        arr.eval();
    }
    InlineArray::save_safetensors(&dir.join("model.safetensors").to_string_lossy(), &entries)
        .map_err(|e| PretrainError::Checkpoint(format!("save model weights: {e}")))?;

    // Optimizer state: moments, master copies and counters.
    let state = optimizer.export_state();
    let opt_entries: Vec<(&str, &InlineArray)> = state
        .tensors
        .iter()
        .map(|(name, array)| {
            array.eval();
            (name.as_str(), array)
        })
        .collect();
    InlineArray::save_safetensors(
        &dir.join("optimizer.safetensors").to_string_lossy(),
        &opt_entries,
    )
    .map_err(|e| PretrainError::Checkpoint(format!("save optimizer state: {e}")))?;

    // Metadata JSON
    let json = serde_json::to_string_pretty(&MetadataOut {
        meta,
        optimizer: OptimizerMeta {
            kind: state.kind.clone(),
            scalars: state.scalars.clone(),
        },
    })
    .map_err(|e| PretrainError::Checkpoint(format!("serialize meta: {e}")))?;
    std::fs::write(dir.join("metadata.json"), json)
        .map_err(|e| PretrainError::Checkpoint(format!("write meta: {e}")))?;

    Ok(())
}

/// Load a full pretraining checkpoint from `dir/`.
///
/// Populates `model` with loaded weights and `optimizer` with restored
/// moments, master copies and step counter. Returns the metadata so the
/// caller can resume from the saved step.
pub fn load_checkpoint<M: ModuleParameters>(
    dir: &Path,
    model: &mut M,
    optimizer: &mut AdamW,
) -> Result<CheckpointMeta, PretrainError> {
    // Metadata
    let meta_bytes = std::fs::read(dir.join("metadata.json"))
        .map_err(|e| PretrainError::Checkpoint(format!("read meta: {e}")))?;
    let MetadataIn {
        meta,
        optimizer: optimizer_meta,
    } = serde_json::from_slice(&meta_bytes)
        .map_err(|e| PretrainError::Checkpoint(format!("parse meta: {e}")))?;

    // Model weights
    let model_path = dir.join("model.safetensors");
    let loaded =
        pmetal_runtime::inline_array::load_safetensors_shard(&model_path.to_string_lossy())
            .ok_or_else(|| {
                PretrainError::Checkpoint(format!("load model: {}", model_path.display()))
            })?;

    {
        let mut flat = model.flatten_params_mut();
        for (key, arr) in &loaded {
            if let Some(dst) = flat.get_mut(key.as_str()) {
                **dst = arr.clone();
            }
        }
    }

    // Optimizer state
    let opt_path = dir.join("optimizer.safetensors");
    let opt_loaded =
        pmetal_runtime::inline_array::load_safetensors_shard(&opt_path.to_string_lossy())
            .ok_or_else(|| {
                PretrainError::Checkpoint(format!("load optimizer: {}", opt_path.display()))
            })?;

    let mut state = OptimizerState::new("adamw");
    state.tensors = opt_loaded.into_iter().collect();
    match optimizer_meta {
        Some(m) => {
            state.kind = m.kind;
            state.scalars = m.scalars;
        }
        None => {
            // A checkpoint from before the counters were recorded: its step
            // is the optimizer's, and its moments are named `param.__m` and
            // `param.__v`.
            state.set_scalar("step", meta.step as f64);
            let renamed: Vec<(String, InlineArray)> = state
                .tensors
                .iter()
                .filter_map(|(name, array)| {
                    if let Some(param) = name.strip_suffix(".__m") {
                        Some((format!("m/{param}"), array.clone()))
                    } else {
                        name.strip_suffix(".__v")
                            .map(|param| (format!("v/{param}"), array.clone()))
                    }
                })
                .collect();
            state.tensors = renamed.into_iter().collect();
        }
    }
    optimizer
        .import_state(&state)
        .map_err(|e| PretrainError::Checkpoint(format!("restore optimizer: {e}")))?;

    Ok(meta)
}
