//! Which optimizer group each trainable parameter trains in, and at what
//! learning rate and weight decay.
//!
//! Every training path reads the groups from one [`ParameterGroupConfig`]:
//! the standard, packed and compiled loops run an optimizer per group
//! ([`crate::ParamGroupOptimizer`]), whose AdamW updates each parameter in one
//! fused Metal dispatch at its group's rate and decay. A run therefore trains
//! the same way whichever loop it lands on.
//!
//! | Group | Parameters | Learning rate | Weight decay |
//! |-------|------------|---------------|--------------|
//! | [`ParamClass::NoDecay`] | biases and normalization weights ([`excluded_from_weight_decay`]) | base | 0 |
//! | [`ParamClass::Embedding`] | the token embedding and the output head, and adapters on them | base × embedding ratio | configured |
//! | [`ParamClass::LoraB`] | LoRA B matrices, with LoRA+ | base × LoRA+ ratio | configured |
//! | [`ParamClass::Regular`] | everything else | base | configured |
//!
//! The first matching row wins. The scheduler sets the base rate; the other
//! groups keep their ratio to it.
//!
//! The output head is the layer the model reports as its head
//! ([`Self::with_output_projection`](ParameterGroupConfig::with_output_projection),
//! found by identity, as PEFT finds it), whatever it is called; failing
//! that, an embedding is a parameter under a module named one of
//! [`EMBEDDING_MODULE_NAMES`]. Names match on whole path components, so
//! BERT's `attention.output.dense` is not an output head.

pub use pmetal_runtime::optimizer::{
    EMBEDDING_MODULE_NAMES, GroupRates, ParamClass, excluded_from_weight_decay,
};

/// The parameter groups of a training run.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterGroupConfig {
    /// Each group's learning rate (as a multiple of the base rate) and
    /// weight decay.
    pub rates: GroupRates,
    /// A parameter under a module with one of these names (a whole path
    /// component, ignoring case) is an embedding.
    pub embedding_modules: Vec<String>,
    /// The path of the model's output head, when the model has one of its
    /// own: its parameters, and its adapter's, are embeddings whatever the
    /// head is called.
    pub output_projection: Option<String>,
}

impl Default for ParameterGroupConfig {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl ParameterGroupConfig {
    /// Every group at the base rate, decayed by `weight_decay` except the
    /// no-decay group.
    pub fn new(weight_decay: f32) -> Self {
        Self {
            rates: GroupRates::new(weight_decay),
            embedding_modules: EMBEDDING_MODULE_NAMES
                .iter()
                .map(|name| name.to_string())
                .collect(),
            output_projection: None,
        }
    }

    /// Count the layer at `path` (as
    /// `DynamicModel::output_projection_path` reports it) as the output head:
    /// its parameters and its adapter's train at the embedding rate. `None`
    /// leaves the head to [`Self::embedding_modules`].
    pub fn with_output_projection(mut self, path: Option<String>) -> Self {
        self.output_projection = path;
        self
    }

    /// Train embeddings at `embedding_lr` while the base rate is `base_lr`,
    /// and at that multiple of the base rate as it moves.
    pub fn with_embedding_lr(mut self, embedding_lr: f32, base_lr: f32) -> Self {
        self.rates = self.rates.with_embedding_lr(embedding_lr, base_lr);
        self
    }

    /// Train LoRA B matrices at `ratio` times the base rate (LoRA+).
    pub fn with_loraplus_lr_ratio(mut self, ratio: f32) -> Self {
        self.rates = self.rates.with_loraplus_ratio(ratio);
        self
    }

    /// Count parameters under a module named `module` as embedding
    /// parameters.
    pub fn add_embedding_module(mut self, module: impl Into<String>) -> Self {
        self.embedding_modules.push(module.into());
        self
    }

    /// Whether the parameter called `name` trains without weight decay.
    pub fn is_no_decay(&self, name: &str) -> bool {
        excluded_from_weight_decay(name)
    }

    /// Whether the parameter called `name` belongs to the token embedding or
    /// the output head.
    pub fn is_embedding(&self, name: &str) -> bool {
        let in_head = self.output_projection.as_deref().is_some_and(|head| {
            name.strip_prefix(head)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
        });
        in_head || pmetal_runtime::optimizer::under_module(name, &self.embedding_modules)
    }

    /// The group the parameter called `name` trains in.
    ///
    /// A LoRA B matrix is [`ParamClass::LoraB`] only with LoRA+; without it,
    /// it trains with the A matrices.
    pub fn classify(&self, name: &str) -> ParamClass {
        let lower = name.to_lowercase();
        if self.is_no_decay(name) {
            ParamClass::NoDecay
        } else if self.is_embedding(name) {
            ParamClass::Embedding
        } else if self.rates.loraplus_ratio.is_some()
            && (lower.contains("lora_b") || lower.contains("lorab"))
        {
            ParamClass::LoraB
        } else {
            ParamClass::Regular
        }
    }

    /// The learning rate of `class` while the base rate is `base_lr`.
    pub fn learning_rate(&self, class: ParamClass, base_lr: f32) -> f32 {
        self.rates.learning_rate(class, base_lr)
    }

    /// The weight decay of `class`.
    pub fn weight_decay(&self, class: ParamClass) -> f32 {
        self.rates.weight_decay(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_parameter_lands_in_its_group() {
        let groups = ParameterGroupConfig::new(0.01).with_loraplus_lr_ratio(16.0);
        for (name, class) in [
            ("model.embed_tokens.weight", ParamClass::Embedding),
            ("lm_head.weight", ParamClass::Embedding),
            ("model.layers.0.input_layernorm.weight", ParamClass::NoDecay),
            ("model.layers.0.self_attn.q_proj.bias", ParamClass::NoDecay),
            ("model.norm.weight", ParamClass::NoDecay),
            (
                "model.layers.0.self_attn.q_proj.lora_a",
                ParamClass::Regular,
            ),
            ("model.layers.0.self_attn.q_proj.lora_b", ParamClass::LoraB),
            ("model.layers.0.mlp.down_proj.weight", ParamClass::Regular),
        ] {
            assert_eq!(groups.classify(name), class, "{name}");
        }
        // Without LoRA+ a B matrix trains with the A matrices.
        let plain = ParameterGroupConfig::new(0.01);
        assert_eq!(
            plain.classify("model.layers.0.self_attn.q_proj.lora_b"),
            ParamClass::Regular
        );
    }

    #[test]
    fn rates_are_multiples_of_the_base_rate() {
        let groups = ParameterGroupConfig::new(0.1)
            .with_embedding_lr(5e-5, 2e-4)
            .with_loraplus_lr_ratio(16.0);
        for base in [2e-4f32, 1e-4, 0.0] {
            assert_eq!(groups.learning_rate(ParamClass::Regular, base), base);
            assert_eq!(groups.learning_rate(ParamClass::NoDecay, base), base);
            assert!((groups.learning_rate(ParamClass::Embedding, base) - base / 4.0).abs() < 1e-12);
            assert_eq!(groups.learning_rate(ParamClass::LoraB, base), base * 16.0);
        }
        assert_eq!(groups.weight_decay(ParamClass::NoDecay), 0.0);
        for class in [
            ParamClass::Regular,
            ParamClass::Embedding,
            ParamClass::LoraB,
        ] {
            assert_eq!(groups.weight_decay(class), 0.1);
        }
    }

    #[test]
    fn custom_embedding_modules_count() {
        let groups = ParameterGroupConfig::new(0.0).add_embedding_module("my_custom_embedding");
        assert_eq!(
            groups.classify("my_custom_embedding.weight"),
            ParamClass::Embedding
        );
        assert_eq!(
            groups.classify("model.my_custom_embedding.lora_a"),
            ParamClass::Embedding
        );
        // A whole module name, not a part of one.
        assert_eq!(
            groups.classify("not_my_custom_embedding.weight"),
            ParamClass::Regular
        );
        assert_eq!(groups.classify("regular_layer.weight"), ParamClass::Regular);
    }

    /// The token embedding and the output head, and adapters on them, train
    /// at the embedding rate under the names BERT, Llama, GPT-2 and GPT-NeoX
    /// give them; no other projection does, whatever its name contains.
    #[test]
    fn embeddings_are_the_token_embedding_and_the_head_by_module_name() {
        let groups = ParameterGroupConfig::new(0.01).with_embedding_lr(5e-5, 2e-4);
        for name in [
            // Llama
            "model.embed_tokens.weight",
            "lm_head.weight",
            "lm_head.lora_a",
            "lm_head.lora_b",
            // GPT-2
            "transformer.wte.weight",
            // GPT-NeoX
            "gpt_neox.embed_in.weight",
            "embed_out.weight",
            "embed_out.lora_a",
            // BERT
            "bert.embeddings.word_embeddings.weight",
        ] {
            assert_eq!(groups.classify(name), ParamClass::Embedding, "{name}");
        }
        for name in [
            // BERT: the attention and MLP output projections.
            "bert.encoder.layer.0.attention.output.dense.lora_a",
            "bert.encoder.layer.0.attention.output.dense.weight",
            "bert.encoder.layer.0.output.dense.lora_b",
            "bert.encoder.layer.0.intermediate.dense.lora_a",
            // Llama
            "model.layers.0.self_attn.o_proj.lora_a",
            "model.layers.0.mlp.down_proj.weight",
            // GPT-2
            "transformer.wpe.weight",
            "transformer.h.0.attn.c_proj.lora_a",
            // GPT-NeoX
            "gpt_neox.layers.0.attention.dense.lora_a",
            "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
        ] {
            assert_eq!(groups.classify(name), ParamClass::Regular, "{name}");
        }
        // Normalization weights skip weight decay before anything else.
        assert_eq!(
            groups.classify("bert.encoder.layer.0.attention.output.LayerNorm.weight"),
            ParamClass::NoDecay
        );
    }

    /// The head the model reports is the head, whatever it is called, and
    /// only the parameters under it are: a head named `output`, as GGUF and
    /// InternLM2 name it, does not take BERT's `attention.output` with it.
    #[test]
    fn the_reported_output_head_is_an_embedding_whatever_its_name() {
        let groups =
            ParameterGroupConfig::new(0.01).with_output_projection(Some("output".to_string()));
        for name in ["output.weight", "output.lora_a", "output.lora_b"] {
            assert_eq!(groups.classify(name), ParamClass::Embedding, "{name}");
        }
        for name in [
            "output_proj.weight",
            "outputs.weight",
            "model.output.weight",
            "encoder.layer.0.attention.output.dense.lora_a",
        ] {
            assert_eq!(groups.classify(name), ParamClass::Regular, "{name}");
        }
        // Without the model's word, `output` is no embedding.
        let by_name = ParameterGroupConfig::new(0.01);
        assert_eq!(by_name.classify("output.lora_a"), ParamClass::Regular);
    }
}
