//! Loading and saving the BERT-family encoders (BERT, RoBERTa, DistilBERT).
//!
//! The model's parameter tree is its own (`layers.0.attention.query`,
//! `layers.0.attention_output.layer_norm`); the checkpoints are
//! `transformers`' (`encoder.layer.0.attention.self.query`,
//! `encoder.layer.0.attention.output.LayerNorm`; DistilBERT:
//! `transformer.layer.0.attention.q_lin`). One table per family names the
//! pairs, and both directions read it, so what [`bert_checkpoint_tensors`]
//! writes is what [`load_bert_weights`] reads and what `transformers` reads.

use pmetal_runtime::compat::{Array, ModuleParametersExt};
use std::collections::HashMap;

use super::assign::assign_renamed;
use super::{Exemption, LoadError, LoadPolicy};
use crate::architectures::bert::{BertModel, EncoderFamily};

/// `(model tail, checkpoint tail)` of each tensor group in an encoder layer,
/// after `layers.{i}.` and the family's layer prefix respectively.
fn layer_table(family: EncoderFamily) -> &'static [(&'static str, &'static str)] {
    match family {
        EncoderFamily::Bert | EncoderFamily::Roberta => &[
            ("attention.query", "attention.self.query"),
            ("attention.key", "attention.self.key"),
            ("attention.value", "attention.self.value"),
            ("attention_output.dense", "attention.output.dense"),
            ("attention_output.layer_norm", "attention.output.LayerNorm"),
            ("intermediate.dense", "intermediate.dense"),
            ("output.dense", "output.dense"),
            ("output.layer_norm", "output.LayerNorm"),
        ],
        EncoderFamily::DistilBert => &[
            ("attention.query", "attention.q_lin"),
            ("attention.key", "attention.k_lin"),
            ("attention.value", "attention.v_lin"),
            ("attention_output.dense", "attention.out_lin"),
            ("attention_output.layer_norm", "sa_layer_norm"),
            ("intermediate.dense", "ffn.lin1"),
            ("output.dense", "ffn.lin2"),
            ("output.layer_norm", "output_layer_norm"),
        ],
    }
}

/// Where the layers sit in a checkpoint.
fn layer_prefix(family: EncoderFamily) -> &'static str {
    match family {
        EncoderFamily::Bert | EncoderFamily::Roberta => "encoder.layer.",
        EncoderFamily::DistilBert => "transformer.layer.",
    }
}

/// The model parameter a checkpoint tensor names, or the key itself when it
/// names none (a task head, the pooler, a buffer), so that the load reports it.
///
/// A task model (`BertForMaskedLM`, `RobertaForSequenceClassification`) puts
/// the encoder under the family's prefix; a bare `*Model` does not. TensorFlow-era
/// checkpoints name LayerNorm's tensors `gamma` and `beta`.
pub fn bert_param_name(family: EncoderFamily, key: &str) -> String {
    let name = key.strip_prefix(family.prefix()).unwrap_or(key);
    let name = if name.contains("LayerNorm") || name.contains("layer_norm") {
        name.strip_suffix(".gamma")
            .map(|n| format!("{n}.weight"))
            .or_else(|| name.strip_suffix(".beta").map(|n| format!("{n}.bias")))
            .unwrap_or_else(|| name.to_string())
    } else {
        name.to_string()
    };

    if let Some(rest) = name.strip_prefix("embeddings.") {
        return match rest.strip_prefix("LayerNorm.") {
            Some(param) => format!("embeddings.layer_norm.{param}"),
            None if rest.ends_with("_embeddings.weight") => name,
            None => return key.to_string(),
        };
    }
    if let Some(rest) = name.strip_prefix(layer_prefix(family))
        && let Some((index, tail)) = rest.split_once('.')
        && index.bytes().all(|b| b.is_ascii_digit())
    {
        for (model, checkpoint) in layer_table(family) {
            if let Some(param) = tail
                .strip_prefix(checkpoint)
                .and_then(|p| p.strip_prefix('.'))
                .filter(|p| matches!(*p, "weight" | "bias"))
            {
                return format!("layers.{index}.{model}.{param}");
            }
        }
    }
    key.to_string()
}

/// The checkpoint name (the bare `*Model` layout, no family prefix) of a model
/// parameter: the inverse of [`bert_param_name`].
pub fn bert_checkpoint_name(family: EncoderFamily, parameter: &str) -> String {
    if let Some(param) = parameter.strip_prefix("embeddings.layer_norm.") {
        return format!("embeddings.LayerNorm.{param}");
    }
    if let Some(rest) = parameter.strip_prefix("layers.")
        && let Some((index, tail)) = rest.split_once('.')
    {
        for (model, checkpoint) in layer_table(family) {
            if let Some(param) = tail.strip_prefix(model).and_then(|p| p.strip_prefix('.')) {
                return format!("{}{index}.{checkpoint}.{param}", layer_prefix(family));
            }
        }
    }
    parameter.to_string()
}

/// Load a BERT, RoBERTa or DistilBERT encoder from a `transformers`
/// checkpoint: a bare `*Model` (as sentence-transformers saves it) or a task
/// model whose encoder sits under the family prefix.
///
/// Fails, naming the difference, unless every encoder tensor is consumed,
/// every parameter filled and every shape matches. The pooler and the task
/// heads are not part of this model and are named exemptions.
pub fn load_bert_weights(
    model: &mut BertModel,
    weights: HashMap<String, Array>,
) -> Result<(), LoadError> {
    let family = model
        .config
        .family()
        .map_err(|e| LoadError::SafeTensors(e.to_string()))?;
    let head = |pattern: &'static str| {
        Exemption::new(
            pattern,
            "a pooler or task head, which is not part of the encoder",
        )
    };
    let policy = LoadPolicy::strict().allowing_unused_tensors(&[
        head("*pooler.*"),
        head("cls.*"),
        head("lm_head.*"),
        head("classifier.*"),
        head("qa_outputs.*"),
        head("vocab_transform.*"),
        head("vocab_layer_norm.*"),
        head("vocab_projector.*"),
        head("pre_classifier.*"),
        head("*predictions.*"),
    ]);
    assign_renamed(
        model,
        weights,
        |key, value| (bert_param_name(family, key), value),
        &policy,
    )
}

/// The encoder's tensors under their `transformers` names (the bare `*Model`
/// layout, which `transformers` and sentence-transformers load directly).
///
/// What a fine-tuned encoder is saved as.
pub fn bert_checkpoint_tensors(model: &BertModel) -> Result<Vec<(String, Array)>, LoadError> {
    let family = model
        .config
        .family()
        .map_err(|e| LoadError::SafeTensors(e.to_string()))?;
    let mut tensors: Vec<(String, Array)> = model
        .flatten_params()
        .into_iter()
        .map(|(name, array)| (bert_checkpoint_name(family, &name), array))
        .collect();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::bert::BertConfig;

    fn config(model_type: &str) -> BertConfig {
        BertConfig {
            vocab_size: 20,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 16,
            model_type: Some(model_type.to_string()),
            ..Default::default()
        }
    }

    /// Every parameter of every family has a checkpoint name that maps back to
    /// it, under the bare layout and under the task-model prefix.
    #[test]
    fn checkpoint_names_and_parameter_names_are_inverses() {
        for model_type in ["bert", "roberta", "distilbert"] {
            let model = BertModel::new(config(model_type)).expect("builds");
            let family = model.config.family().unwrap();
            let names: Vec<String> = model
                .flatten_params()
                .keys()
                .map(|k| k.to_string())
                .collect();
            assert!(names.len() > 20, "{model_type}: {}", names.len());
            for name in names {
                let checkpoint = bert_checkpoint_name(family, &name);
                assert!(!checkpoint.starts_with("layers."), "{name} was not renamed");
                assert_eq!(bert_param_name(family, &checkpoint), name, "{model_type}");
                let prefixed = format!("{}{checkpoint}", family.prefix());
                assert_eq!(
                    bert_param_name(family, &prefixed),
                    name,
                    "{model_type} prefixed"
                );
            }
        }
    }

    #[test]
    fn tensorflow_era_layer_norm_names_and_unknown_keys() {
        let family = EncoderFamily::Bert;
        assert_eq!(
            bert_param_name(family, "bert.embeddings.LayerNorm.gamma"),
            "embeddings.layer_norm.weight"
        );
        assert_eq!(
            bert_param_name(family, "encoder.layer.1.output.LayerNorm.beta"),
            "layers.1.output.layer_norm.bias"
        );
        for unknown in [
            "cls.predictions.bias",
            "bert.pooler.dense.weight",
            "encoder.layer.0.attention.self.mystery.weight",
            "embeddings.position_ids",
        ] {
            assert_eq!(bert_param_name(family, unknown), unknown);
        }
    }
}
