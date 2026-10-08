//! The language model a decision head reads.
//!
//! The head needs two things from it: the final normalized hidden state of
//! every prompt token, and the LM-head matrix. Qwen3 and Qwen3.5 backbones (both
//! Clef releases) run on the fused native engine, the same forward `pmetal
//! infer` and `pmetal serve` use for those models, which prefills about twice as
//! fast as the general path. Any other architecture falls back to
//! [`DynamicModel::forward_hidden`].

use std::path::Path;

use pmetal_bridge::compat::{Array, Dtype, Exception};
use pmetal_bridge::qwen3_native::{self, LayerWeight, NativeCache, NativeWeights};

use super::DecisionError;
use crate::dispatcher::{DynamicModel, ModelArchitecture};

pub(crate) enum Backbone {
    Native(Box<NativeWeights>),
    Dynamic(Box<DynamicModel>),
}

impl Backbone {
    pub(crate) fn load(dir: &Path) -> Result<Self, DecisionError> {
        let load_error = |e: String| DecisionError::Load(format!("backbone: {e}"));
        if runs_natively(dir) {
            let config = qwen3_native::load_config(dir).map_err(load_error)?;
            let weights = qwen3_native::load_model(dir, &config).map_err(load_error)?;
            return Ok(Self::Native(Box::new(weights)));
        }
        let model = DynamicModel::load(dir).map_err(|e| load_error(e.to_string()))?;
        Ok(Self::Dynamic(Box::new(model)))
    }

    /// The LM-head matrix, `[vocab, hidden]`, if it is dense.
    pub(crate) fn output_embedding(&self) -> Option<Array> {
        match self {
            Self::Native(weights) => {
                if weights.tie_word_embeddings {
                    weights
                        .embed_scales
                        .is_none()
                        .then(|| weights.embed_w.clone())
                } else {
                    match weights.lm_head_w.as_ref()? {
                        // Stored pre-transposed, `[hidden, vocab]`.
                        LayerWeight::Dense(w) => Some(w.t()),
                        LayerWeight::Quantized { .. } => None,
                    }
                }
            }
            Self::Dynamic(model) => model.lm_head_weight(),
        }
    }

    /// Final normalized hidden states for one prompt, `[1, len, hidden]`.
    pub(crate) fn forward_hidden(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        match self {
            Self::Native(weights) => {
                // A fresh cache per prompt: nothing is carried between records.
                let mut cache = NativeCache::new_empty(weights);
                // The logits are never evaluated, so the LM-head matmul over
                // every position is never run.
                let (hidden, _logits) =
                    qwen3_native::forward_step_hidden(weights, input_ids, &mut cache);
                Ok(hidden)
            }
            Self::Dynamic(model) => model.forward_hidden(input_ids, None),
        }
    }
}

/// Whether `dir`'s `config.json` names a Qwen3 or Qwen3.5 model.
fn runs_natively(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    matches!(
        config["model_type"]
            .as_str()
            .and_then(ModelArchitecture::from_model_type),
        Some(ModelArchitecture::Qwen3 | ModelArchitecture::Qwen3Next)
    )
}

/// Whether a matrix is one the head can read rows of.
pub(crate) fn is_dense_float(array: &Array) -> bool {
    matches!(
        array.dtype(),
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    )
}
