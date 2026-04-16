//! Full-state pretraining checkpoint: model weights + optimizer m/v + metadata.
//!
//! Saves three files per checkpoint:
//! - `model.safetensors` — all model parameters
//! - `optimizer.safetensors` — AdamW momentum + velocity (keyed `param.__m`, `param.__v`)
//! - `metadata.json` — step count, loss, learning rate

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::InlineArray;
use pmetal_bridge::compat::{
    Array, ModuleParametersExt,
    module::ModuleParameters,
    optimizers::{AdamW, Optimizer, State},
};

use super::PretrainError;

/// Metadata written alongside the checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub step: u64,
    pub loss: f32,
    pub learning_rate: f32,
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
    InlineArray::save_safetensors(&dir.join("model.safetensors").to_string_lossy(), &entries);

    // Optimizer state: m and v for each parameter
    let state = optimizer.state();
    let mut opt_entries: Vec<(String, &InlineArray)> = Vec::with_capacity(state.len() * 2);
    for (key, (m, v)) in state {
        m.eval();
        v.eval();
        opt_entries.push((format!("{key}.__m"), m));
        opt_entries.push((format!("{key}.__v"), v));
    }
    let opt_ref: Vec<(&str, &InlineArray)> =
        opt_entries.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    InlineArray::save_safetensors(
        &dir.join("optimizer.safetensors").to_string_lossy(),
        &opt_ref,
    );

    // Metadata JSON
    let json = serde_json::to_string_pretty(meta)
        .map_err(|e| PretrainError::Checkpoint(format!("serialize meta: {e}")))?;
    std::fs::write(dir.join("metadata.json"), json)
        .map_err(|e| PretrainError::Checkpoint(format!("write meta: {e}")))?;

    Ok(())
}

/// Load a full pretraining checkpoint from `dir/`.
///
/// Populates `model` with loaded weights and `optimizer` with restored
/// momentum/velocity + step counter. Returns the metadata so the caller
/// can resume from the saved step.
pub fn load_checkpoint<M: ModuleParameters>(
    dir: &Path,
    model: &mut M,
    optimizer: &mut AdamW,
) -> Result<CheckpointMeta, PretrainError> {
    // Metadata
    let meta_bytes = std::fs::read(dir.join("metadata.json"))
        .map_err(|e| PretrainError::Checkpoint(format!("read meta: {e}")))?;
    let meta: CheckpointMeta = serde_json::from_slice(&meta_bytes)
        .map_err(|e| PretrainError::Checkpoint(format!("parse meta: {e}")))?;

    // Model weights
    let model_path = dir.join("model.safetensors");
    let loaded = pmetal_bridge::inline_array::load_safetensors_shard(&model_path.to_string_lossy())
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
        pmetal_bridge::inline_array::load_safetensors_shard(&opt_path.to_string_lossy())
            .ok_or_else(|| {
                PretrainError::Checkpoint(format!("load optimizer: {}", opt_path.display()))
            })?;

    let mut m_map: HashMap<String, Array> = HashMap::new();
    let mut v_map: HashMap<String, Array> = HashMap::new();
    for (key, arr) in opt_loaded {
        if let Some(param_name) = key.strip_suffix(".__m") {
            m_map.insert(param_name.to_string(), arr);
        } else if let Some(param_name) = key.strip_suffix(".__v") {
            v_map.insert(param_name.to_string(), arr);
        }
    }

    let mut state: State<(Array, Array)> = HashMap::new();
    for (key, m) in m_map {
        if let Some(v) = v_map.remove(&key) {
            state.insert(Rc::from(key.as_str()), (m, v));
        }
    }

    optimizer.restore_state(meta.step, state);

    Ok(meta)
}
