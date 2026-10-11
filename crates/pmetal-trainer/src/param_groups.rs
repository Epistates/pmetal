//! Which optimizer group each trainable parameter trains in, and at what
//! learning rate and weight decay.
//!
//! Every training path reads the groups from one [`ParameterGroupConfig`]:
//! the standard, packed and compiled loops run an optimizer per group
//! ([`crate::ParamGroupOptimizer`]), and the Metal fused AdamW
//! ([`crate::MlxMetalOptimizer`]) hands each group's rate and decay to its
//! kernel. A run therefore trains the same way whichever loop it lands on.
//!
//! | Group | Parameters | Learning rate | Weight decay |
//! |-------|------------|---------------|--------------|
//! | [`ParamClass::NoDecay`] | biases and normalization weights ([`excluded_from_weight_decay`]) | base | 0 |
//! | [`ParamClass::Embedding`] | names matching an embedding pattern | base × embedding ratio | configured |
//! | [`ParamClass::LoraB`] | LoRA B matrices, with LoRA+ | base × LoRA+ ratio | configured |
//! | [`ParamClass::Regular`] | everything else | base | configured |
//!
//! The first matching row wins. The scheduler sets the base rate; the other
//! groups keep their ratio to it.

pub use pmetal_bridge::optimizer::{GroupRates, ParamClass, excluded_from_weight_decay};

/// The parameter groups of a training run.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterGroupConfig {
    /// Each group's learning rate (as a multiple of the base rate) and
    /// weight decay.
    pub rates: GroupRates,
    /// A parameter whose lower-cased name contains one of these is an
    /// embedding.
    pub embedding_patterns: Vec<String>,
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
            embedding_patterns: [
                "embed_tokens",
                "lm_head",
                "wte",        // GPT-2 style
                "wpe",        // Position embeddings
                "token_embd", // GGUF style
                "output",     // Output projection
            ]
            .map(String::from)
            .to_vec(),
        }
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

    /// Count names containing `pattern` as embedding parameters.
    pub fn add_embedding_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.embedding_patterns.push(pattern.into());
        self
    }

    /// Whether the parameter called `name` trains without weight decay.
    pub fn is_no_decay(&self, name: &str) -> bool {
        excluded_from_weight_decay(name)
    }

    /// The group the parameter called `name` trains in.
    ///
    /// A LoRA B matrix is [`ParamClass::LoraB`] only with LoRA+; without it,
    /// it trains with the A matrices.
    pub fn classify(&self, name: &str) -> ParamClass {
        let lower = name.to_lowercase();
        if self.is_no_decay(name) {
            ParamClass::NoDecay
        } else if self
            .embedding_patterns
            .iter()
            .any(|pattern| lower.contains(&pattern.to_lowercase()))
        {
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
    fn custom_embedding_patterns_count() {
        let groups = ParameterGroupConfig::new(0.0).add_embedding_pattern("my_custom_embedding");
        assert_eq!(
            groups.classify("my_custom_embedding.weight"),
            ParamClass::Embedding
        );
        assert_eq!(groups.classify("regular_layer.weight"), ParamClass::Regular);
    }
}
