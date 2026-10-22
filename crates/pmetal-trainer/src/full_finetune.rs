//! Full-parameter fine-tuning of an existing checkpoint on the GPU.
//!
//! `pmetal train --full` trains every weight of the model through the same
//! loop, optimizers, gradient checkpointing and packing LoRA runs through: the
//! model is [`DynamicLoraModel::from_pretrained_full`], the inference path's own
//! model with every parameter trainable, so there is no second forward pass to
//! drift from the one `pmetal infer` and `serve` run. What is specific to a
//! full fine-tune lives here:
//!
//! - [`estimate_memory`] and [`MemoryEstimate::check`]: a full fine-tune holds
//!   the weights, a gradient and optimizer state per weight, so it needs several
//!   times the model's size where LoRA needs a little more than it. The run is
//!   refused, before anything loads, when that does not fit the device's
//!   recommended working set, with the options that would make it.
//! - [`SavePlan`] and [`save_full_model`]: the result is a whole Hugging
//!   Face-layout model (sharded safetensors under the checkpoint's own tensor
//!   names, and its config, tokenizer and chat template), not an adapter.
//!   Every parameter has to land under a name the source checkpoint has, which
//!   is checked once the model is built so a layout the saver cannot write is
//!   refused before the run, not after it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use pmetal_core::OptimizerType;
use pmetal_merge::{SafetensorsLoader, TensorLoader, TensorWriter};
use pmetal_runtime::compat::Dtype;
use serde_json::Value;

/// What a full fine-tune is asked for beyond a LoRA run's settings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FullFineTuneConfig {
    /// Keep the input token embedding as loaded; see
    /// [`pmetal_lora::FullFineTune::freeze_embeddings`].
    pub freeze_embeddings: bool,
    /// Run without checking the memory estimate against the device. The
    /// estimate is conservative; this is for a run known to fit.
    pub skip_memory_check: bool,
}

/// Why a full fine-tune cannot run or cannot be saved.
#[derive(Debug, thiserror::Error)]
pub enum FullFineTuneError {
    /// The run would not fit the device.
    #[error("{0}")]
    WontFit(String),
    /// The checkpoint or the model is not one a full fine-tune supports.
    #[error("{0}")]
    Unsupported(String),
    /// Writing the model failed.
    #[error("cannot save the fine-tuned model: {0}")]
    Save(String),
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// Bytes a trainable parameter costs in the gradient: an f32 accumulator, and
/// the clipped copy the optimizer step reads while it is alive.
const GRADIENT_BYTES: f64 = 8.0;
/// The runtime's own overhead (MLX, the Metal context, scratch buffers).
const OVERHEAD_BYTES: f64 = 2.0 * 1e9;

/// What a full fine-tune's memory estimate is made from.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRequest {
    /// Bytes the weights take as loaded.
    pub weight_bytes: u64,
    /// Parameters in the model.
    pub params: u64,
    /// Parameters that train (all but the frozen embedding).
    pub trainable: u64,
    /// Whether the weights are half precision (the optimizer then keeps an f32
    /// master copy of each).
    pub half_precision: bool,
    /// Elements in the token embedding.
    pub embedding_params: u64,
    /// The optimizer.
    pub optimizer: OptimizerType,
    /// Sequences per micro-batch.
    pub batch_size: u64,
    /// Tokens per sequence.
    pub seq_len: u64,
    /// Width of the model.
    pub hidden: u64,
    /// Decoder layers.
    pub layers: u64,
    /// Vocabulary size.
    pub vocab: u64,
    /// Whether activations are recomputed in the backward pass.
    pub gradient_checkpointing: bool,
    /// Whether the loss is cut cross-entropy (the logits are never formed).
    pub cut_cross_entropy: bool,
}

/// The memory a full fine-tune needs, by part, in bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryEstimate {
    /// The weights.
    pub weights: u64,
    /// Gradients.
    pub gradients: u64,
    /// The optimizer's moments, and master copies of half-precision weights.
    pub optimizer: u64,
    /// Activations and the loss's logits, for one micro-batch.
    pub activations: u64,
    /// The runtime's own overhead.
    pub overhead: u64,
}

/// Bytes of optimizer state per trainable parameter: moments in f32 (8-bit
/// AdamW's are a byte each), plus an f32 master copy when the weights are half
/// precision. Muon keeps one momentum for its matrices; the embedding trains
/// with AdamW, so its elements are counted at AdamW's rate on top.
fn optimizer_bytes(request: &MemoryRequest) -> f64 {
    let master = if request.half_precision { 4.0 } else { 0.0 };
    let trainable = request.trainable as f64;
    let embedding = request.embedding_params.min(request.trainable) as f64;
    let moments = match request.optimizer {
        OptimizerType::AdamW => 8.0 * trainable,
        OptimizerType::AdamW8bit => 2.1 * trainable,
        OptimizerType::AdamWScheduleFree => 8.0 * trainable,
        OptimizerType::Muon => 4.0 * trainable + 4.0 * embedding,
        OptimizerType::Lion | OptimizerType::Sgd => 4.0 * trainable,
        OptimizerType::Adafactor => 1.0 * trainable,
    };
    moments + master * trainable
}

/// Activations of one micro-batch with the layer boundaries kept and one layer
/// recomputed at a time (`gradient_checkpointing`), or every layer's kept; and
/// the logits with their gradient in f32 unless the loss never forms them.
fn activation_bytes(request: &MemoryRequest) -> f64 {
    let tokens = (request.batch_size * request.seq_len) as f64;
    let hidden = request.hidden as f64;
    let layers = request.layers as f64;
    let per_layer = 34.0; // one layer's activations, in hidden-sized 2-byte rows
    let kept = if request.gradient_checkpointing {
        tokens * hidden * (2.0 * layers + per_layer) * 2.0
    } else {
        tokens * hidden * per_layer * layers * 2.0
    };
    let logits = if request.cut_cross_entropy {
        0.0
    } else {
        tokens * request.vocab as f64 * 8.0
    };
    kept + logits
}

/// The memory a full fine-tune of `request` needs.
pub fn estimate_memory(request: &MemoryRequest) -> MemoryEstimate {
    MemoryEstimate {
        weights: request.weight_bytes,
        gradients: (request.trainable as f64 * GRADIENT_BYTES) as u64,
        optimizer: optimizer_bytes(request) as u64,
        activations: activation_bytes(request) as u64,
        overhead: OVERHEAD_BYTES as u64,
    }
}

impl MemoryEstimate {
    /// The whole estimate, in bytes.
    pub fn total(&self) -> u64 {
        self.weights + self.gradients + self.optimizer + self.activations + self.overhead
    }

    /// Refuse the run if it needs more than `budget` bytes, naming the parts
    /// and what would shrink them.
    ///
    /// # Errors
    ///
    /// [`FullFineTuneError::WontFit`] with the breakdown.
    pub fn check(&self, budget: u64, request: &MemoryRequest) -> Result<(), FullFineTuneError> {
        if self.total() <= budget {
            return Ok(());
        }
        let gb = |bytes: u64| bytes as f64 / 1e9;
        let mut advice = Vec::new();
        if !request.cut_cross_entropy {
            advice.push("--cut-cross-entropy (the logits are never formed)");
        }
        if request.optimizer == OptimizerType::AdamW {
            advice.push("--optimizer adamw_8bit (one byte per moment)");
        }
        if request.embedding_params > 0 {
            advice.push("--freeze-embeddings");
        }
        advice.push("a smaller --batch-size or --max-seq-len");
        advice.push("LoRA instead of --full");
        Err(FullFineTuneError::WontFit(format!(
            "a full fine-tune of this model needs about {:.1} GB (weights {:.1}, gradients \
             {:.1}, optimizer state {:.1}, activations {:.1}, runtime {:.1}) and the device's \
             recommended working set is {:.1} GB. Try {}. \
             (--no-memory-check runs it anyway.)",
            gb(self.total()),
            gb(self.weights),
            gb(self.gradients),
            gb(self.optimizer),
            gb(self.activations),
            gb(self.overhead),
            gb(budget),
            advice.join(", "),
        )))
    }
}

/// The model sizes in a `config.json`, which the multimodal wrappers nest under
/// `text_config`.
fn config_number(config: &Value, key: &str) -> Option<u64> {
    config
        .get(key)
        .or_else(|| config.get("text_config")?.get(key))?
        .as_u64()
}

/// The memory request for fully fine-tuning the checkpoint in `model_dir`,
/// read from its tensor headers and `config.json`: no weight is loaded.
///
/// # Errors
///
/// [`FullFineTuneError::Unsupported`] for a checkpoint that is not safetensors,
/// or has no `config.json`.
pub fn memory_request(
    model_dir: &Path,
    optimizer: OptimizerType,
    batch_size: usize,
    seq_len: usize,
    freeze_embeddings: bool,
    gradient_checkpointing: bool,
    cut_cross_entropy: bool,
) -> Result<MemoryRequest, FullFineTuneError> {
    let source = SafetensorsLoader::new(model_dir).map_err(|e| {
        FullFineTuneError::Unsupported(format!(
            "full fine-tuning reads a safetensors checkpoint: {e}"
        ))
    })?;
    let config: Value = std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .ok_or_else(|| {
            FullFineTuneError::Unsupported(format!(
                "{} has no readable config.json",
                model_dir.display()
            ))
        })?;
    let hidden = config_number(&config, "hidden_size").unwrap_or(0);
    let vocab = config_number(&config, "vocab_size").unwrap_or(0);
    let layers = config_number(&config, "num_hidden_layers").unwrap_or(0);
    let (mut params, mut weight_bytes, mut half) = (0u64, 0u64, false);
    for name in source.tensor_names() {
        let (Ok(shape), Ok(dtype)) = (source.tensor_shape(&name), source.tensor_dtype(&name))
        else {
            continue;
        };
        let elements: u64 = shape.iter().map(|&d| d as u64).product();
        let width = match dtype {
            safetensors::Dtype::F32 => 4,
            safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {
                half = true;
                2
            }
            _ => continue,
        };
        params += elements;
        weight_bytes += elements * width;
    }
    let embedding_params = hidden * vocab;
    Ok(MemoryRequest {
        weight_bytes,
        params,
        trainable: params.saturating_sub(if freeze_embeddings {
            embedding_params
        } else {
            0
        }),
        half_precision: half,
        embedding_params,
        optimizer,
        batch_size: batch_size as u64,
        seq_len: seq_len as u64,
        hidden,
        layers,
        vocab,
        gradient_checkpointing,
        cut_cross_entropy,
    })
}

/// The memory the device allows a run: its recommended working set, as the
/// device reports it.
pub fn device_budget() -> u64 {
    pmetal_runtime::inline_array::get_max_recommended_size() as u64
}

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

/// What a source checkpoint's tensor is, for planning a save.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensor {
    /// Its shape.
    pub shape: Vec<usize>,
    /// Whether it is a floating-point tensor (a packed one is not).
    pub float: bool,
}

/// Where each trained parameter goes in the saved checkpoint, and which source
/// tensors the model does not load and are carried over as they are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavePlan {
    /// The model's parameter path, and the checkpoint tensor name it is saved
    /// under, sorted by parameter path.
    pub renames: Vec<(String, String)>,
    /// Checkpoint tensors the model never loaded (an MTP head, a vision tower),
    /// copied through so the saved model is as complete as the source.
    pub passthrough: Vec<String>,
}

/// The names a parameter at `path` may have in a checkpoint: as the model names
/// it, without the trunk's `model.` (the encoder-only and text-tower
/// checkpoints that omit it), and with it.
fn candidate_names(path: &str) -> Vec<String> {
    let mut names = vec![path.to_string()];
    match path.strip_prefix("model.") {
        Some(rest) => names.push(rest.to_string()),
        None => names.push(format!("model.{path}")),
    }
    names
}

/// Buffers a checkpoint may hold that are recomputed on load, not parameters.
fn is_derived_buffer(name: &str) -> bool {
    name.ends_with("rotary_emb.inv_freq")
}

/// Plan where each of the model's `params` (path to shape) is saved among the
/// tensors of a `source` checkpoint.
///
/// # Errors
///
/// [`FullFineTuneError::Unsupported`] naming the parameters that have no tensor
/// of that shape in the source (a fused or renamed layout, or a quantized
/// checkpoint), and the source tensors that are not floating point and that no
/// parameter claims.
pub fn plan_save(
    source: &BTreeMap<String, SourceTensor>,
    params: &BTreeMap<String, Vec<usize>>,
) -> Result<SavePlan, FullFineTuneError> {
    let mut renames = Vec::with_capacity(params.len());
    let mut claimed: BTreeSet<&str> = BTreeSet::new();
    let mut unmapped: Vec<String> = Vec::new();
    let mut packed: Vec<String> = Vec::new();
    for (path, shape) in params {
        let found = candidate_names(path)
            .into_iter()
            .find_map(|name| source.get_key_value(&name));
        let Some((name, tensor)) = found else {
            unmapped.push(path.clone());
            continue;
        };
        if !tensor.float {
            packed.push(name.clone());
            continue;
        }
        if &tensor.shape != shape {
            unmapped.push(format!(
                "{path} (the model's is {shape:?}, the checkpoint's {name} is {:?})",
                tensor.shape
            ));
            continue;
        }
        if !claimed.insert(name.as_str()) {
            unmapped.push(format!("{path} (also claims {name})"));
            continue;
        }
        renames.push((path.clone(), name.clone()));
    }
    let list = |names: &[String]| {
        let shown: Vec<&str> = names.iter().take(4).map(String::as_str).collect();
        format!(
            "{}{}",
            shown.join(", "),
            if names.len() > shown.len() {
                ", ..."
            } else {
                ""
            }
        )
    };
    if !packed.is_empty() {
        return Err(FullFineTuneError::Unsupported(format!(
            "the checkpoint stores {} of the model's weights packed or quantized ({}); a full \
             fine-tune needs the dense checkpoint",
            packed.len(),
            list(&packed)
        )));
    }
    if !unmapped.is_empty() {
        return Err(FullFineTuneError::Unsupported(format!(
            "{} of the model's parameters have no tensor of the same name and shape in its \
             checkpoint ({}), so a fine-tuned model could not be written in the checkpoint's \
             layout; full fine-tuning does not support this architecture's layout yet",
            unmapped.len(),
            list(&unmapped)
        )));
    }
    let mut passthrough = Vec::new();
    let mut carried_non_float = Vec::new();
    for (name, tensor) in source {
        if claimed.contains(name.as_str()) || is_derived_buffer(name) {
            continue;
        }
        if tensor.float {
            passthrough.push(name.clone());
        } else {
            carried_non_float.push(name.clone());
        }
    }
    if !carried_non_float.is_empty() {
        return Err(FullFineTuneError::Unsupported(format!(
            "the checkpoint holds {} tensor(s) the model does not load and that are not \
             floating point ({}), which a saved model could not carry over",
            carried_non_float.len(),
            list(&carried_non_float)
        )));
    }
    Ok(SavePlan {
        renames,
        passthrough,
    })
}

/// The source checkpoint's tensors, read from its headers.
pub fn source_tensors(
    source: &SafetensorsLoader,
) -> Result<BTreeMap<String, SourceTensor>, FullFineTuneError> {
    let mut out = BTreeMap::new();
    for name in source.tensor_names() {
        let (Ok(shape), Ok(dtype)) = (source.tensor_shape(&name), source.tensor_dtype(&name))
        else {
            return Err(FullFineTuneError::Unsupported(format!(
                "cannot read the header of checkpoint tensor {name}"
            )));
        };
        let float = matches!(
            dtype,
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
        );
        out.insert(name, SourceTensor { shape, float });
    }
    Ok(out)
}

/// The safetensors dtype a model of `dtype` parameters is saved in.
fn on_disk(dtype: Dtype) -> safetensors::Dtype {
    match dtype {
        Dtype::Bfloat16 => safetensors::Dtype::BF16,
        Dtype::Float16 => safetensors::Dtype::F16,
        _ => safetensors::Dtype::F32,
    }
}

/// The dtype most of the model's elements are stored in, which the whole
/// checkpoint is saved in.
fn dominant_dtype(dtypes: impl Iterator<Item = (Dtype, usize)>) -> Dtype {
    let mut weight: BTreeMap<i32, usize> = BTreeMap::new();
    for (dtype, elements) in dtypes {
        *weight.entry(dtype.as_i32()).or_default() += elements;
    }
    weight
        .into_iter()
        .max_by_key(|(_, elements)| *elements)
        .map_or(Dtype::Float32, |(raw, _)| Dtype::from_raw(raw))
}

/// A fine-tuned model's parameters as a model reports them (path to value): what
/// [`save_full_model`] writes.
pub type Parameters = std::collections::HashMap<std::rc::Rc<str>, pmetal_runtime::compat::Array>;

/// The parameter paths of a model and their shapes, for [`plan_save`].
pub fn parameter_shapes(parameters: &Parameters) -> BTreeMap<String, Vec<usize>> {
    parameters
        .iter()
        .map(|(path, value)| {
            (
                path.to_string(),
                value.shape().iter().map(|&d| d as usize).collect(),
            )
        })
        .collect()
}

/// Write the fine-tuned `parameters` as a Hugging Face-layout model in
/// `output_dir`: the source checkpoint's config, tokenizer, chat template and
/// other side files, and the weights as sharded safetensors (one file when they
/// fit in one) with an index, under the source's tensor names and the model's
/// own dtype. Nothing under a final name exists until the weights are complete,
/// so an interrupted save leaves no torn model.
///
/// Returns the weights file (`model.safetensors`), or the index of a sharded
/// model.
///
/// # Errors
///
/// [`FullFineTuneError::Save`] if `output_dir` is the source directory (the
/// save would replace the weights it reads), or a file cannot be read or
/// written.
pub fn save_full_model(
    parameters: &Parameters,
    plan: &SavePlan,
    source_dir: &Path,
    output_dir: &Path,
) -> Result<PathBuf, FullFineTuneError> {
    let save = |e: &dyn std::fmt::Display| FullFineTuneError::Save(e.to_string());
    std::fs::create_dir_all(output_dir).map_err(|e| save(&e))?;
    let same = match (source_dir.canonicalize(), output_dir.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if same {
        return Err(FullFineTuneError::Save(format!(
            "the output directory {} is the model's own; choose another, or the save would \
             replace the weights it was trained from",
            output_dir.display()
        )));
    }

    let dtype = dominant_dtype(parameters.values().map(|value| {
        (
            value.dtype(),
            value.shape().iter().map(|&d| d as usize).product(),
        )
    }));
    pmetal_merge::copy_side_files(source_dir, output_dir).map_err(|e| save(&e))?;

    let mut writer = TensorWriter::with_dtype(output_dir, on_disk(dtype)).map_err(|e| save(&e))?;
    for (path, name) in &plan.renames {
        let value = parameters
            .get(path.as_str())
            .ok_or_else(|| FullFineTuneError::Save(format!("the model has no parameter {path}")))?;
        writer.write_tensor(name, value).map_err(|e| save(&e))?;
    }
    if !plan.passthrough.is_empty() {
        let source = SafetensorsLoader::new(source_dir).map_err(|e| save(&e))?;
        for name in &plan.passthrough {
            let value = source.load_tensor(name).map_err(|e| save(&e))?;
            writer.write_tensor(name, &value).map_err(|e| save(&e))?;
        }
    }
    writer.finalize().map_err(|e| save(&e))?;
    pmetal_merge::patch_config_dtype(output_dir, on_disk(dtype)).map_err(|e| save(&e))?;

    let single = output_dir.join("model.safetensors");
    Ok(if single.exists() {
        single
    } else {
        output_dir.join("model.safetensors.index.json")
    })
}

/// Everything a full fine-tune saves from `model`: plan against the source
/// checkpoint's layout, then write. See [`plan_save`] and [`save_full_model`].
pub fn plan_for_model(
    source_dir: &Path,
    parameters: &Parameters,
) -> Result<SavePlan, FullFineTuneError> {
    let source = SafetensorsLoader::new(source_dir).map_err(|e| {
        FullFineTuneError::Unsupported(format!(
            "full fine-tuning reads a safetensors checkpoint: {e}"
        ))
    })?;
    plan_save(&source_tensors(&source)?, &parameter_shapes(parameters))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(entries: &[(&str, &[usize], bool)]) -> BTreeMap<String, SourceTensor> {
        entries
            .iter()
            .map(|(name, shape, float)| {
                (
                    name.to_string(),
                    SourceTensor {
                        shape: shape.to_vec(),
                        float: *float,
                    },
                )
            })
            .collect()
    }

    fn params(entries: &[(&str, &[usize])]) -> BTreeMap<String, Vec<usize>> {
        entries
            .iter()
            .map(|(name, shape)| (name.to_string(), shape.to_vec()))
            .collect()
    }

    fn request() -> MemoryRequest {
        MemoryRequest {
            weight_bytes: 1_200_000_000,
            params: 600_000_000,
            trainable: 600_000_000,
            half_precision: true,
            embedding_params: 155_000_000,
            optimizer: OptimizerType::AdamW,
            batch_size: 1,
            seq_len: 512,
            hidden: 1024,
            layers: 28,
            vocab: 151_936,
            gradient_checkpointing: true,
            cut_cross_entropy: false,
        }
    }

    #[test]
    fn the_estimate_counts_weights_gradients_moments_masters_and_activations() {
        let r = request();
        let e = estimate_memory(&r);
        assert_eq!(e.weights, 1_200_000_000);
        assert_eq!(e.gradients, 4_800_000_000);
        // Two f32 moments and an f32 master per parameter.
        assert_eq!(e.optimizer, 600_000_000 * 12);
        // The logits dominate a small model's activations: 512 * 151936 * 8.
        assert!(e.activations > 600_000_000, "{}", e.activations);
        assert_eq!(
            e.total(),
            e.weights + e.gradients + e.optimizer + e.activations + e.overhead
        );
    }

    #[test]
    fn what_shrinks_a_full_fine_tune_shrinks_its_estimate() {
        let base = estimate_memory(&request());
        let eight_bit = estimate_memory(&MemoryRequest {
            optimizer: OptimizerType::AdamW8bit,
            ..request()
        });
        // The f32 master copy of the half-precision weights stays.
        assert!(eight_bit.optimizer < base.optimizer * 6 / 10);
        let frozen = estimate_memory(&MemoryRequest {
            trainable: 600_000_000 - 155_000_000,
            ..request()
        });
        assert!(frozen.gradients < base.gradients && frozen.optimizer < base.optimizer);
        let cut = estimate_memory(&MemoryRequest {
            cut_cross_entropy: true,
            ..request()
        });
        assert!(cut.activations < base.activations / 4);
        let uncheckpointed = estimate_memory(&MemoryRequest {
            gradient_checkpointing: false,
            ..request()
        });
        assert!(uncheckpointed.activations > base.activations);
        // Full precision weights need no master copy.
        let f32_weights = estimate_memory(&MemoryRequest {
            half_precision: false,
            ..request()
        });
        assert!(f32_weights.optimizer < base.optimizer);
    }

    #[test]
    fn a_run_that_does_not_fit_is_refused_with_the_breakdown_and_the_way_out() {
        let r = request();
        let e = estimate_memory(&r);
        assert!(e.check(e.total(), &r).is_ok());
        let FullFineTuneError::WontFit(message) = e.check(e.total() - 1, &r).unwrap_err() else {
            panic!("expected WontFit");
        };
        for needle in [
            "needs about",
            "gradients",
            "optimizer state",
            "recommended working set",
            "--optimizer adamw_8bit",
            "--freeze-embeddings",
            "--cut-cross-entropy",
            "LoRA",
            "--no-memory-check",
        ] {
            assert!(message.contains(needle), "{needle}: {message}");
        }
    }

    #[test]
    fn parameters_are_saved_under_the_checkpoints_own_names() {
        let src = source(&[
            ("model.embed_tokens.weight", &[8, 4], true),
            ("model.layers.0.mlp.up_proj.weight", &[6, 4], true),
            ("model.norm.weight", &[4], true),
            ("lm_head.weight", &[8, 4], true),
            ("mtp.fc.weight", &[4, 4], true),
            ("model.rotary_emb.inv_freq", &[2], true),
        ]);
        let ps = params(&[
            ("model.embed_tokens.weight", &[8, 4]),
            ("model.layers.0.mlp.up_proj.weight", &[6, 4]),
            ("model.norm.weight", &[4]),
            ("lm_head.weight", &[8, 4]),
        ]);
        let plan = plan_save(&src, &ps).unwrap();
        assert_eq!(plan.renames.len(), 4);
        assert!(plan.renames.iter().all(|(path, name)| path == name));
        // The MTP head the model never loaded is carried; the derived buffer is not.
        assert_eq!(plan.passthrough, ["mtp.fc.weight"]);
    }

    #[test]
    fn a_trunk_prefix_the_checkpoint_omits_is_found() {
        let src = source(&[
            ("embed_tokens.weight", &[8, 4], true),
            ("norm.weight", &[4], true),
        ]);
        let ps = params(&[
            ("model.embed_tokens.weight", &[8, 4]),
            ("model.norm.weight", &[4]),
        ]);
        let plan = plan_save(&src, &ps).unwrap();
        assert_eq!(
            plan.renames,
            [
                (
                    "model.embed_tokens.weight".to_string(),
                    "embed_tokens.weight".to_string()
                ),
                ("model.norm.weight".to_string(), "norm.weight".to_string()),
            ]
        );
    }

    #[test]
    fn a_layout_the_saver_cannot_write_is_refused_before_the_run() {
        let src = source(&[("model.layers.0.mlp.experts.0.up_proj.weight", &[6, 4], true)]);
        // A stacked expert tensor has no counterpart in the checkpoint.
        let stacked = params(&[("model.layers.0.mlp.experts.up_proj.weight", &[2, 6, 4])]);
        let FullFineTuneError::Unsupported(message) = plan_save(&src, &stacked).unwrap_err() else {
            panic!("expected Unsupported");
        };
        assert!(message.contains("experts.up_proj"), "{message}");
        assert!(message.contains("layout"), "{message}");

        // The right name with another shape is a different tensor.
        let src = source(&[("w", &[6, 4], true)]);
        let wrong_shape = plan_save(&src, &params(&[("w", &[4, 6])])).unwrap_err();
        assert!(wrong_shape.to_string().contains("[4, 6]"), "{wrong_shape}");

        // Two parameters cannot share a tensor.
        let src = source(&[("model.w", &[2], true)]);
        let twice = plan_save(&src, &params(&[("model.w", &[2]), ("w", &[2])])).unwrap_err();
        assert!(twice.to_string().contains("also claims"), "{twice}");
    }

    #[test]
    fn a_quantized_checkpoint_is_refused_as_one() {
        let src = source(&[
            ("model.w.weight", &[6, 1], false),
            ("model.w.scales", &[6, 1], true),
        ]);
        let err = plan_save(&src, &params(&[("model.w.weight", &[6, 4])])).unwrap_err();
        assert!(err.to_string().contains("packed or quantized"), "{err}");
        assert!(err.to_string().contains("dense checkpoint"), "{err}");
    }

    #[test]
    fn an_integer_tensor_the_model_does_not_load_cannot_be_carried() {
        let src = source(&[("model.w", &[2], true), ("extra.index", &[2], false)]);
        let err = plan_save(&src, &params(&[("model.w", &[2])])).unwrap_err();
        assert!(err.to_string().contains("extra.index"), "{err}");
    }
}
