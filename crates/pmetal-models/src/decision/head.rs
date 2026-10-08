//! The joint schema head: scores every option of every question from the
//! backbone's final hidden states.
//!
//! A port of the release's PyTorch `JointSchemaHead`, built from torch's own
//! layer definitions so the checkpoint loads as-is:
//!
//! * [`MultiheadAttention`] is `torch.nn.MultiheadAttention` (q/k/v packed in
//!   `in_proj_weight`, no mask, no causality).
//! * [`EvidenceRoutingLayer`] cross-attends the options to the prompt, pre-norm
//!   on both sides.
//! * [`SchemaDecoderLayer`] is `torch.nn.TransformerDecoderLayer` with
//!   `norm_first=True, activation="gelu"`: self-attention over the fields, then
//!   cross-attention to the prompt. The prompt memory is *not* normalized inside
//!   this layer, unlike in the routing layers.
//!
//! Every GELU is torch's default, the exact erf form; LayerNorm eps is torch's
//! 1e-5. The head runs in one dtype ([`JointSchemaHead::dtype`]); inputs are
//! cast to it.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use pmetal_bridge::compat::{Array, Dtype, Exception, ModuleParametersExt, Param, nn, ops};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use super::DecisionError;
use super::encode::EncodedRecord;

/// `joint_head_config.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JointHeadConfig {
    /// The backbone's hidden size.
    pub hidden_size: i32,
    /// The head's own width.
    pub width: i32,
    /// Evidence-routing layers.
    pub routing_layers: usize,
    /// Decoder layers over the fields.
    pub layers: usize,
    /// Attention heads, in every attention block.
    pub heads: i32,
    /// Feed-forward width, in every block.
    pub feedforward: i32,
}

/// `torch.nn.LayerNorm`'s default epsilon.
const LAYER_NORM_EPS: f32 = 1e-5;
/// `F.normalize`'s default epsilon.
const NORMALIZE_EPS: f32 = 1e-12;
/// `F.cosine_similarity`'s default epsilon.
const COSINE_EPS: f32 = 1e-8;
/// Both learned logit scales are clamped to `log(100)` before `exp`.
const MAX_LOG_LOGIT_SCALE: f32 = 4.605_170_2;

fn layer_norm(dims: i32) -> nn::LayerNorm {
    nn::LayerNorm::with_affine(dims, LAYER_NORM_EPS, true)
}

fn linear(in_dims: i32, out_dims: i32, bias: bool) -> Result<nn::Linear, Exception> {
    nn::Linear::new(in_dims, out_dims, bias)
}

/// `torch.nn.MultiheadAttention(width, heads, batch_first=True)`, unmasked.
#[derive(Debug)]
pub struct MultiheadAttention {
    /// `[3 * width, width]`: the q, k and v projections stacked in that order.
    pub in_proj_weight: Param<Array>,
    /// `[3 * width]`.
    pub in_proj_bias: Param<Array>,
    pub out_proj: nn::Linear,
    pub heads: i32,
    pub width: i32,
}
impl_module_params!(MultiheadAttention; in_proj_weight, in_proj_bias, out_proj);

impl MultiheadAttention {
    fn new(width: i32, heads: i32) -> Result<Self, Exception> {
        Ok(Self {
            in_proj_weight: Param::new(Array::zeros(&[3 * width, width], Dtype::Float32.as_i32())),
            in_proj_bias: Param::new(Array::zeros(&[3 * width], Dtype::Float32.as_i32())),
            out_proj: linear(width, width, true)?,
            heads,
            width,
        })
    }

    fn project(&self, x: &Array, part: i32) -> Array {
        let (start, end) = (part * self.width, (part + 1) * self.width);
        let weight = ops::slice_axis(&self.in_proj_weight.value, 0, start, end);
        let bias = ops::slice_axis(&self.in_proj_bias.value, 0, start, end);
        x.matmul(&weight.t()).add(&bias)
    }

    /// `[batch, len, width]` → `[batch, heads, len, head_dim]`.
    fn split_heads(&self, x: &Array) -> Array {
        let (batch, len) = (x.dim(0), x.dim(1));
        x.reshape(&[batch, len, self.heads, self.width / self.heads])
            .transpose_axes(&[0, 2, 1, 3])
    }

    /// Attend `query` `[batch, lq, width]` over `key`/`value` `[batch, lk, width]`.
    pub fn forward(&self, query: &Array, key: &Array, value: &Array) -> Array {
        let q = self.split_heads(&self.project(query, 0));
        let k = self.split_heads(&self.project(key, 1));
        let v = self.split_heads(&self.project(value, 2));
        let head_dim = (self.width / self.heads) as f32;
        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2]))
            .multiply(&Array::from_f32(head_dim.sqrt().recip()));
        let attended = ops::softmax_axis(&scores, -1).matmul(&v);
        let (batch, len) = (query.dim(0), query.dim(1));
        let merged = attended
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, len, self.width]);
        self.out_proj.forward(&merged)
    }
}

/// Two linear layers around an exact GELU: torch's
/// `Sequential(Linear, GELU, Dropout, Linear[, Dropout])` at inference.
fn gelu_mlp(input: &nn::Linear, output: &nn::Linear, x: &Array) -> Array {
    output.forward(&nn::gelu_erf(&input.forward(x)))
}

/// Routes evidence from the prompt into the option queries.
#[derive(Debug)]
pub struct EvidenceRoutingLayer {
    pub query_norm: nn::LayerNorm,
    pub memory_norm: nn::LayerNorm,
    pub attention: MultiheadAttention,
    pub feedforward_norm: nn::LayerNorm,
    /// Checkpoint `feedforward.0`.
    pub feedforward_in: nn::Linear,
    /// Checkpoint `feedforward.3`.
    pub feedforward_out: nn::Linear,
}
impl_module_params!(
    EvidenceRoutingLayer;
    query_norm, memory_norm, attention, feedforward_norm, feedforward_in, feedforward_out
);

impl EvidenceRoutingLayer {
    fn new(config: &JointHeadConfig) -> Result<Self, Exception> {
        Ok(Self {
            query_norm: layer_norm(config.width),
            memory_norm: layer_norm(config.width),
            attention: MultiheadAttention::new(config.width, config.heads)?,
            feedforward_norm: layer_norm(config.width),
            feedforward_in: linear(config.width, config.feedforward, true)?,
            feedforward_out: linear(config.feedforward, config.width, true)?,
        })
    }

    pub fn forward(&self, queries: &Array, memory: &Array) -> Array {
        let memory = self.memory_norm.forward(memory);
        let routed = self
            .attention
            .forward(&self.query_norm.forward(queries), &memory, &memory);
        let queries = queries.add(&routed);
        let update = gelu_mlp(
            &self.feedforward_in,
            &self.feedforward_out,
            &self.feedforward_norm.forward(&queries),
        );
        queries.add(&update)
    }
}

/// `torch.nn.TransformerDecoderLayer(norm_first=True, activation="gelu")`.
#[derive(Debug)]
pub struct SchemaDecoderLayer {
    pub self_attn: MultiheadAttention,
    pub multihead_attn: MultiheadAttention,
    pub linear1: nn::Linear,
    pub linear2: nn::Linear,
    pub norm1: nn::LayerNorm,
    pub norm2: nn::LayerNorm,
    pub norm3: nn::LayerNorm,
}
impl_module_params!(
    SchemaDecoderLayer;
    self_attn, multihead_attn, linear1, linear2, norm1, norm2, norm3
);

impl SchemaDecoderLayer {
    fn new(config: &JointHeadConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MultiheadAttention::new(config.width, config.heads)?,
            multihead_attn: MultiheadAttention::new(config.width, config.heads)?,
            linear1: linear(config.width, config.feedforward, true)?,
            linear2: linear(config.feedforward, config.width, true)?,
            norm1: layer_norm(config.width),
            norm2: layer_norm(config.width),
            norm3: layer_norm(config.width),
        })
    }

    pub fn forward(&self, fields: &Array, memory: &Array) -> Array {
        let normed = self.norm1.forward(fields);
        let fields = fields.add(&self.self_attn.forward(&normed, &normed, &normed));
        let fields = fields.add(&self.multihead_attn.forward(
            &self.norm2.forward(&fields),
            memory,
            memory,
        ));
        let update = gelu_mlp(&self.linear1, &self.linear2, &self.norm3.forward(&fields));
        fields.add(&update)
    }
}

/// The joint schema head.
#[derive(Debug)]
pub struct JointSchemaHead {
    pub config: JointHeadConfig,
    pub hidden_norm: nn::LayerNorm,
    pub memory_projection: nn::Linear,
    pub question_projection: nn::Linear,
    pub option_question_projection: nn::Linear,
    pub global_projection: nn::Linear,
    pub option_context_projection: nn::Linear,
    pub option_lexical_projection: nn::Linear,
    pub type_embedding: nn::Embedding,
    pub evidence_layers: Vec<EvidenceRoutingLayer>,
    pub option_summary_norm: nn::LayerNorm,
    pub layers: Vec<SchemaDecoderLayer>,
    pub field_norm: nn::LayerNorm,
    pub option_norm: nn::LayerNorm,
    /// Checkpoint `residual_scorer.0`.
    pub residual_scorer_in: nn::Linear,
    /// Checkpoint `residual_scorer.3`.
    pub residual_scorer_out: nn::Linear,
    pub prior_logit_scale: Param<Array>,
    pub joint_logit_scale: Param<Array>,
    pub residual_gate: Param<Array>,
}
impl_module_params!(
    JointSchemaHead;
    hidden_norm, memory_projection, question_projection, option_question_projection,
    global_projection, option_context_projection, option_lexical_projection, type_embedding,
    evidence_layers, option_summary_norm, layers, field_norm, option_norm,
    residual_scorer_in, residual_scorer_out, prior_logit_scale, joint_logit_scale, residual_gate
);

/// Checkpoint key → parameter path. torch's `Sequential` children are numbered,
/// which a Rust field cannot be.
fn parameter_path(key: &str) -> String {
    key.replace(".feedforward.0.", ".feedforward_in.")
        .replace(".feedforward.3.", ".feedforward_out.")
        .replace("residual_scorer.0.", "residual_scorer_in.")
        .replace("residual_scorer.3.", "residual_scorer_out.")
}

impl JointSchemaHead {
    /// An unloaded head of the given shape.
    pub fn new(config: JointHeadConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size;
        let width = config.width;
        let scalar = || Param::new(Array::zeros(&[], Dtype::Float32.as_i32()));
        Ok(Self {
            hidden_norm: layer_norm(hidden),
            memory_projection: linear(hidden, width, false)?,
            question_projection: linear(hidden, width, false)?,
            option_question_projection: linear(hidden, width, false)?,
            global_projection: linear(hidden, width, false)?,
            option_context_projection: linear(hidden, width, false)?,
            option_lexical_projection: linear(hidden, width, false)?,
            type_embedding: nn::Embedding::new(3, width)?,
            evidence_layers: (0..config.routing_layers)
                .map(|_| EvidenceRoutingLayer::new(&config))
                .collect::<Result<_, _>>()?,
            option_summary_norm: layer_norm(width),
            layers: (0..config.layers)
                .map(|_| SchemaDecoderLayer::new(&config))
                .collect::<Result<_, _>>()?,
            field_norm: layer_norm(width),
            option_norm: layer_norm(width),
            residual_scorer_in: linear(width * 4, width, true)?,
            residual_scorer_out: linear(width, 1, true)?,
            prior_logit_scale: scalar(),
            joint_logit_scale: scalar(),
            residual_gate: scalar(),
            config,
        })
    }

    /// Read `joint_head_config.json` and `joint_head.safetensors` from `dir`,
    /// holding the weights in `dtype`.
    pub fn load(dir: &Path, dtype: Dtype) -> Result<Self, DecisionError> {
        let config_path = dir.join(super::HEAD_CONFIG_FILE);
        let config: JointHeadConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| DecisionError::Load(format!("{}: {e}", config_path.display())))?,
        )
        .map_err(|e| DecisionError::Load(format!("{}: {e}", config_path.display())))?;
        let weights_path = dir.join(super::HEAD_WEIGHTS_FILE);
        let weights = crate::loader::load_safetensors_file(&weights_path)
            .map_err(|e| DecisionError::Load(e.to_string()))?;
        let mut head = Self::new(config).map_err(DecisionError::Model)?;
        head.load_weights(weights, dtype)?;
        Ok(head)
    }

    /// Assign a torch `state_dict` exactly as `load_state_dict(strict=True)`
    /// would: every parameter filled, nothing left over, shapes equal.
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, Array>,
        dtype: Dtype,
    ) -> Result<(), DecisionError> {
        let mut params = self.flatten_params_mut();
        let mut unexpected = Vec::new();
        let mut filled = HashSet::new();
        for (key, value) in weights {
            let path = parameter_path(&key);
            let Some(slot) = params.get_mut(path.as_str()) else {
                unexpected.push(key);
                continue;
            };
            if slot.shape() != value.shape() {
                return Err(DecisionError::Load(format!(
                    "joint head {key}: checkpoint shape {:?}, head expects {:?}",
                    value.shape(),
                    slot.shape()
                )));
            }
            **slot = value.as_dtype(dtype.as_i32());
            filled.insert(path);
        }
        let mut missing: Vec<&String> = params.keys().filter(|k| !filled.contains(*k)).collect();
        if !unexpected.is_empty() || !missing.is_empty() {
            unexpected.sort();
            missing.sort();
            return Err(DecisionError::Load(format!(
                "joint head checkpoint does not match the head: unexpected {unexpected:?}, \
                 missing {missing:?}"
            )));
        }
        drop(params);
        self.eval().map_err(DecisionError::Model)?;
        pmetal_bridge::check_last_error()
            .map_err(|e| DecisionError::Model(Exception::custom(e.to_string())))?;
        Ok(())
    }

    /// The dtype the head computes in.
    pub fn dtype(&self) -> Dtype {
        self.hidden_norm
            .weight
            .value
            .as_ref()
            .map_or(Dtype::Float32, Array::dtype)
    }

    /// One logit per option of every question in `record`.
    ///
    /// `hidden_states` is the backbone's final (normalized) hidden state for
    /// the record's prompt, `[len, hidden]` or `[1, len, hidden]`;
    /// `output_embedding` is the backbone's LM-head matrix `[vocab, hidden]`,
    /// whose rows give each option a lexical vector. The result is lazy, one
    /// `[options]` array per question in record order.
    pub fn forward(
        &self,
        hidden_states: &Array,
        record: &EncodedRecord,
        output_embedding: &Array,
    ) -> Result<Vec<Array>, DecisionError> {
        let dtype = self.dtype();
        let hidden = if hidden_states.ndim() == 3 {
            hidden_states.squeeze(0)
        } else {
            hidden_states.clone()
        };
        let length = record.input_ids.len();
        if hidden.ndim() != 2
            || hidden.dim(0) as usize != length
            || hidden.dim(1) != self.config.hidden_size
        {
            return Err(DecisionError::Request(format!(
                "hidden states {:?} do not match a {length}-token prompt of width {}",
                hidden.shape(),
                self.config.hidden_size
            )));
        }
        let questions = &record.questions;
        let question_count = questions.len();
        let option_counts: Vec<usize> = questions.iter().map(|q| q.option_spans.len()).collect();
        let option_total: usize = option_counts.iter().sum();
        if option_counts.contains(&0) {
            return Err(DecisionError::Request(
                "every question needs an option".into(),
            ));
        }

        let sequence = self.hidden_norm.forward(&hidden.as_dtype(dtype.as_i32()));
        let memory = self.memory_projection.forward(&sequence).expand_dims(0);
        let global_vector =
            ops::slice_axis(&sequence, 0, length as i32 - 1, length as i32).squeeze(0);

        // Every span mean is a row of one averaging matrix, so the question and
        // option context vectors each come out of a single matmul.
        let question_spans: Vec<(usize, usize)> =
            questions.iter().map(|q| q.question_span).collect();
        let option_spans: Vec<(usize, usize)> = questions
            .iter()
            .flat_map(|q| q.option_spans.iter().copied())
            .collect();
        let question_vectors = span_means(&question_spans, length, dtype)?.matmul(&sequence);
        let option_contexts = span_means(&option_spans, length, dtype)?.matmul(&sequence);

        // Lexical vectors: mean LM-head row over each option's tokens.
        let lexical_ids: Vec<i32> = option_spans
            .iter()
            .flat_map(|&(start, end)| record.input_ids[start..end].iter().map(|&id| id as i32))
            .collect();
        let mut lexical_spans = Vec::with_capacity(option_spans.len());
        let mut cursor = 0;
        for &(start, end) in &option_spans {
            lexical_spans.push((cursor, cursor + end - start));
            cursor += end - start;
        }
        let lexical_rows = output_embedding
            .take_axis(&Array::from_i32_slice(&lexical_ids), 0)
            .as_dtype(dtype.as_i32());
        let lexical = span_means(&lexical_spans, lexical_ids.len(), dtype)?.matmul(&lexical_rows);

        // Which question each option belongs to.
        let option_question: Vec<i32> = option_counts
            .iter()
            .enumerate()
            .flat_map(|(index, &count)| std::iter::repeat_n(index as i32, count))
            .collect();
        let option_question = Array::from_i32_slice(&option_question);

        let mut routed = self
            .option_context_projection
            .forward(&option_contexts)
            .add(&self.option_lexical_projection.forward(&lexical))
            .add(
                &self
                    .option_question_projection
                    .forward(&question_vectors)
                    .take_axis(&option_question, 0),
            )
            .expand_dims(0);
        for layer in &self.evidence_layers {
            routed = layer.forward(&routed, &memory);
        }
        let routed = routed.squeeze(0);

        let base_fields = self.question_projection.forward(&question_vectors);
        let scale = Array::from_f32((self.config.width as f32).sqrt().recip());
        let mut summaries = Vec::with_capacity(question_count);
        let mut offset = 0i32;
        for (index, &count) in option_counts.iter().enumerate() {
            let options = ops::slice_axis(&routed, 0, offset, offset + count as i32);
            offset += count as i32;
            let field = ops::slice_axis(&base_fields, 0, index as i32, index as i32 + 1);
            let weights = ops::softmax_axis(&options.matmul(&field.t()).multiply(&scale), 0);
            summaries.push(weights.multiply(&options).sum_axis(0, false));
        }
        let type_ids: Vec<i32> = questions.iter().map(|q| q.question_type.index()).collect();
        let mut fields = base_fields
            .add(
                &self
                    .option_summary_norm
                    .forward(&ops::stack_axis(&summaries, 0)),
            )
            .add(
                &self
                    .global_projection
                    .forward(&global_vector)
                    .expand_dims(0),
            )
            .add(
                &self
                    .type_embedding
                    .forward(&Array::from_i32_slice(&type_ids)),
            )
            .expand_dims(0);
        for layer in &self.layers {
            fields = layer.forward(&fields, &memory);
        }
        let fields = self.field_norm.forward(&fields.squeeze(0));

        // Scoring, for all options at once.
        let anchors = normalize(
            &question_vectors.add(&global_vector.expand_dims(0)),
            NORMALIZE_EPS,
        )
        .take_axis(&option_question, 0);
        let prior = clamped_scale(&self.prior_logit_scale.value).multiply(
            &normalize(&lexical, NORMALIZE_EPS)
                .multiply(&anchors)
                .sum_axis(-1, false),
        );
        let options = self.option_norm.forward(&routed);
        let field = fields.take_axis(&option_question, 0);
        let cosine = normalize(&field, COSINE_EPS)
            .multiply(&normalize(&options, COSINE_EPS))
            .sum_axis(-1, false);
        let features = ops::concatenate_axis(
            &[
                &field,
                &options,
                &field.multiply(&options),
                &field.subtract(&options).abs(),
            ],
            -1,
        );
        let residual = gelu_mlp(
            &self.residual_scorer_in,
            &self.residual_scorer_out,
            &features,
        )
        .squeeze(-1);
        let joint = clamped_scale(&self.joint_logit_scale.value)
            .multiply(&cosine)
            .add(&residual);
        let logits = prior.add(&self.residual_gate.value.sigmoid().multiply(&joint));

        let mut out = Vec::with_capacity(question_count);
        let mut offset = 0i32;
        for &count in &option_counts {
            out.push(ops::slice_axis(&logits, 0, offset, offset + count as i32));
            offset += count as i32;
        }
        debug_assert_eq!(offset as usize, option_total);
        pmetal_bridge::check_last_error()
            .map_err(|e| DecisionError::Model(Exception::custom(e.to_string())))?;
        Ok(out)
    }
}

/// `[spans, length]` matrix whose rows average the tokens of each span.
fn span_means(
    spans: &[(usize, usize)],
    length: usize,
    dtype: Dtype,
) -> Result<Array, DecisionError> {
    let mut rows = vec![0f32; spans.len() * length];
    for (row, &(start, end)) in spans.iter().enumerate() {
        if start >= end || end > length {
            return Err(DecisionError::Request(format!(
                "span {start}..{end} is empty or outside a {length}-token prompt"
            )));
        }
        let weight = 1.0 / (end - start) as f32;
        rows[row * length + start..row * length + end].fill(weight);
    }
    Ok(Array::from_f32_slice(&rows, &[spans.len() as i32, length as i32]).as_dtype(dtype.as_i32()))
}

/// `x / max(‖x‖₂, eps)` along the last axis: `F.normalize`, and the per-side
/// normalization inside `F.cosine_similarity`.
fn normalize(x: &Array, eps: f32) -> Array {
    let norm = x.square().sum_axis(-1, true).sqrt();
    x.divide(&ops::maximum(
        &norm,
        &Array::from_f32(eps).as_dtype(x.dtype().as_i32()),
    ))
}

/// `exp(min(scale, log 100))`.
fn clamped_scale(scale: &Array) -> Array {
    ops::minimum(
        scale,
        &Array::from_f32(MAX_LOG_LOGIT_SCALE).as_dtype(scale.dtype().as_i32()),
    )
    .exp()
}
