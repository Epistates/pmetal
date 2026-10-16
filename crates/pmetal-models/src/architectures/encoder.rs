//! The encoder-only models, behind one type.
//!
//! `ModelArchitecture::Bert` names every encoder this crate computes: BERT,
//! RoBERTa and DistilBERT (one [`BertModel`]) and ModernBERT (a different
//! block). [`EncoderModel`] is the model the dispatcher holds for it, so the
//! dispatcher, the server's embeddings endpoint and the embedding trainer need
//! not know which.

use pmetal_runtime::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, VisitLinears,
};
use serde_json::Value;

use super::bert::BertModel;
use super::modernbert::ModernBertModel;

/// A BERT-family or ModernBERT encoder.
#[derive(Debug)]
pub enum EncoderModel {
    Bert(BertModel),
    ModernBert(ModernBertModel),
}

impl EncoderModel {
    /// Build from a `config.json` string, by its `model_type`: ModernBERT for
    /// `"modernbert"`, the BERT family for everything else it names.
    pub fn from_config_str(config_str: &str) -> Result<Self, Exception> {
        let config: Value =
            json5::from_str(config_str).map_err(|e| Exception::custom(e.to_string()))?;
        let parse_error = |e: serde_json::Error| Exception::custom(e.to_string());
        if config.get("model_type").and_then(Value::as_str) == Some("modernbert") {
            let config = serde_json::from_value(config).map_err(parse_error)?;
            ModernBertModel::new(config).map(Self::ModernBert)
        } else {
            let config = serde_json::from_value(config).map_err(parse_error)?;
            BertModel::new(config).map(Self::Bert)
        }
    }

    /// Encode `input_ids` (`[batch, length]`); `attention_mask` is `[batch,
    /// length]`, non-zero for real tokens. Returns `[batch, length, hidden]`.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Bert(model) => model.forward(input_ids, attention_mask),
            Self::ModernBert(model) => model.forward(input_ids, attention_mask),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Bert(model) => model.config.vocab_size,
            Self::ModernBert(model) => model.config.vocab_size,
        }
    }

    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Bert(model) => model.config.hidden_size,
            Self::ModernBert(model) => model.config.hidden_size,
        }
    }
}

impl ModuleParameters for EncoderModel {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Bert(model) => model.num_parameters(),
            Self::ModernBert(model) => model.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Bert(model) => model.parameters(),
            Self::ModernBert(model) => model.parameters(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Bert(model) => model.trainable_parameters(),
            Self::ModernBert(model) => model.trainable_parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Bert(model) => model.parameters_mut(),
            Self::ModernBert(model) => model.parameters_mut(),
        }
    }
}

impl VisitLinears for EncoderModel {
    fn visit_linears_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut pmetal_runtime::compat::Linear),
    ) {
        match self {
            Self::Bert(model) => model.visit_linears_mut(prefix, f),
            Self::ModernBert(model) => model.visit_linears_mut(prefix, f),
        }
    }
}
