//! Decision models: a language-model backbone read by a small joint schema
//! head, answering typed questions about a state in one forward pass.
//!
//! Supports the Clef releases (`Cloudflare/clef`, a Qwen3.8-27B backbone, and
//! `Cloudflare/clef-flash`, a Qwen3.5-9B-class backbone). A request names a
//! `state` (any string or JSON value) and a map of `questions`, each a
//! proposition (`noul`), a named choice (`choice`) or an ordered level
//! (`score`). The record is encoded into one prompt ([`encode`]), the backbone
//! runs once with no generation, and [`head::JointSchemaHead`] turns its final
//! hidden states into one logit per option of every question. [`systemone`]
//! answers a `POST /v1/systemone` request body with the matching response body.
//!
//! A release directory is a Qwen3.5 checkpoint plus `joint_head_config.json`
//! and `joint_head.safetensors`; [`is_decision_model`] checks for both.
//!
//! Text only for now: the Qwen3.5 vision encoder is not ported, so a request
//! with `images` or `videos` is refused rather than answered without them.

mod backbone;
pub mod encode;
pub mod head;
pub mod systemone;

use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, Dtype, Exception};

use backbone::Backbone;
pub use encode::{
    DEFAULT_MAX_LENGTH, EncodeOptions, EncodedQuestion, EncodedRecord, QuestionType, encode_record,
    python_json, render,
};
pub use head::{JointHeadConfig, JointSchemaHead};

/// The head's shape, next to the backbone's `config.json`.
pub const HEAD_CONFIG_FILE: &str = "joint_head_config.json";
/// The head's weights.
pub const HEAD_WEIGHTS_FILE: &str = "joint_head.safetensors";

/// Why a decision could not be made.
#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    /// The request is malformed; the message is the client's to read.
    #[error("{0}")]
    Request(String),
    /// The request asks for something this port does not do yet.
    #[error("{0}")]
    Unsupported(String),
    /// The release directory could not be loaded.
    #[error("failed to load decision model: {0}")]
    Load(String),
    /// Tokenizing the prompt failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// The forward pass failed.
    #[error("model error: {0}")]
    Model(#[from] Exception),
}

/// Whether `dir` holds a decision-model release.
pub fn is_decision_model(dir: &Path) -> bool {
    dir.join(HEAD_CONFIG_FILE).is_file() && dir.join(HEAD_WEIGHTS_FILE).is_file()
}

/// A loaded decision model: backbone, head, and the tokenizer that encodes its
/// records.
pub struct DecisionModel {
    backbone: Backbone,
    head: JointSchemaHead,
    /// The backbone's LM-head matrix `[vocab, hidden]`, read for option
    /// lexical vectors.
    output_embedding: Array,
    tokenizer: pmetal_data::Tokenizer,
    path: PathBuf,
}

impl std::fmt::Debug for DecisionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionModel")
            .field("path", &self.path)
            .field("head", &self.head.config)
            .finish_non_exhaustive()
    }
}

impl DecisionModel {
    /// Load a release directory: the backbone on the native Qwen engine (see
    /// `backbone`), the head in f32.
    ///
    /// The head runs in f32 whatever the backbone's dtype. It is about a
    /// percent of the backbone's compute and keeps its many small reductions
    /// (span means, the routing softmax, normalizations) out of bf16.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, DecisionError> {
        Self::load_with_head_dtype(dir, Dtype::Float32)
    }

    /// [`load`](Self::load) with the head held and computed in `head_dtype`.
    pub fn load_with_head_dtype(
        dir: impl AsRef<Path>,
        head_dtype: Dtype,
    ) -> Result<Self, DecisionError> {
        let dir = dir.as_ref();
        if !is_decision_model(dir) {
            return Err(DecisionError::Load(format!(
                "{} is not a decision model: it needs {HEAD_CONFIG_FILE} and {HEAD_WEIGHTS_FILE}",
                dir.display()
            )));
        }
        let head = JointSchemaHead::load(dir, head_dtype)?;
        let tokenizer = pmetal_data::Tokenizer::from_model_dir(dir)
            .map_err(|e| DecisionError::Load(format!("tokenizer: {e}")))?;
        let backbone = Backbone::load(dir)?;

        // A packed (quantized) LM head would be read as garbage rows.
        let hidden = head.config.hidden_size;
        let output_embedding = backbone
            .output_embedding()
            .filter(|w| backbone::is_dense_float(w) && w.ndim() == 2)
            .ok_or_else(|| {
                DecisionError::Load(
                    "the joint head reads option vectors from the backbone's LM head, which \
                     must be a dense matrix (quantized backbones are not supported)"
                        .into(),
                )
            })?;
        if output_embedding.dim(1) != hidden {
            return Err(DecisionError::Load(format!(
                "the joint head reads hidden size {hidden}, the backbone has {}",
                output_embedding.dim(1)
            )));
        }
        Ok(Self {
            backbone,
            head,
            output_embedding,
            tokenizer,
            path: dir.to_path_buf(),
        })
    }

    /// The release directory this model was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The joint head.
    pub fn head(&self) -> &JointSchemaHead {
        &self.head
    }

    /// The tokenizer records are encoded with.
    pub fn tokenizer(&self) -> &pmetal_data::Tokenizer {
        &self.tokenizer
    }

    /// Encode a record (`state`, `questions`, optional `id`) into a prompt.
    pub fn encode(
        &self,
        record: &serde_json::Value,
        options: EncodeOptions,
    ) -> Result<EncodedRecord, DecisionError> {
        encode_record(encode::tokenize_with(&self.tokenizer), record, options)
    }

    /// One logit per option of every question, in record order.
    pub fn logits(&mut self, record: &EncodedRecord) -> Result<Vec<Vec<f32>>, DecisionError> {
        let ids: Vec<i32> = record.input_ids.iter().map(|&id| id as i32).collect();
        let ids = Array::from_i32_slice_shaped(&ids, &[1, ids.len() as i32]);
        let started = std::time::Instant::now();
        let hidden = self.backbone.forward_hidden(&ids)?;
        hidden
            .try_eval()
            .map_err(|e| DecisionError::Model(Exception::custom(e.to_string())))?;
        let backbone_ms = started.elapsed().as_secs_f64() * 1e3;
        let logits = self.head.forward(&hidden, record, &self.output_embedding)?;
        let flat =
            pmetal_bridge::compat::ops::concatenate_axis(&logits.iter().collect::<Vec<_>>(), 0)
                .as_dtype(Dtype::Float32.as_i32());
        flat.try_eval()
            .map_err(|e| DecisionError::Model(Exception::custom(e.to_string())))?;
        pmetal_bridge::check_last_error()
            .map_err(|e| DecisionError::Model(Exception::custom(e.to_string())))?;
        tracing::debug!(
            tokens = record.input_ids.len(),
            backbone_ms,
            head_ms = started.elapsed().as_secs_f64() * 1e3 - backbone_ms,
            "decision forward"
        );
        let values = flat.as_slice::<f32>();
        let mut out = Vec::with_capacity(record.questions.len());
        let mut offset = 0;
        for question in &record.questions {
            let count = question.option_ids.len();
            out.push(values[offset..offset + count].to_vec());
            offset += count;
        }
        Ok(out)
    }

    /// Answer a `/v1/systemone` request body with its response body.
    pub fn systemone(
        &mut self,
        request: &serde_json::Value,
        max_length: usize,
    ) -> Result<serde_json::Value, DecisionError> {
        systemone::validate_request(request)?;
        let encoded = self.encode(
            request,
            EncodeOptions {
                max_length,
                max_state_tokens: None,
            },
        )?;
        let logits = self.logits(&encoded)?;
        systemone::response(request, &encoded, &logits)
    }
}
