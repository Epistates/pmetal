//! Parameter grouping for per-layer learning rates.
//!
//! This module provides functionality to separate model parameters into groups
//! with different learning rates, weight decay, etc.
//!
//! Common use case: embeddings need a lower learning rate than LoRA parameters.

use std::collections::HashMap;

/// A group of parameters with shared optimizer settings.
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Parameter names in this group.
    pub param_names: Vec<String>,
    /// Learning rate for this group.
    pub learning_rate: f64,
    /// Weight decay for this group.
    pub weight_decay: f64,
    /// Description of this group (for logging).
    pub description: String,
}

impl ParameterGroup {
    /// Create a new parameter group.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            param_names: Vec::new(),
            learning_rate: 2e-4,
            weight_decay: 0.0,
            description: description.into(),
        }
    }

    /// Set the learning rate.
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the weight decay.
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Add a parameter name.
    pub fn add_param(&mut self, name: impl Into<String>) {
        self.param_names.push(name.into());
    }

    /// Check if this group contains a parameter.
    pub fn contains(&self, name: &str) -> bool {
        self.param_names.iter().any(|n| n == name)
    }

    /// Number of parameters in this group.
    pub fn len(&self) -> usize {
        self.param_names.len()
    }

    /// Check if group is empty.
    pub fn is_empty(&self) -> bool {
        self.param_names.is_empty()
    }
}

/// Parameter grouping configuration.
#[derive(Debug, Clone)]
pub struct ParameterGroupConfig {
    /// Base learning rate for non-embedding parameters.
    pub base_lr: f64,
    /// Learning rate for embedding parameters.
    pub embedding_lr: Option<f64>,
    /// Base weight decay.
    pub weight_decay: f64,
    /// Pattern to match embedding parameters.
    pub embedding_patterns: Vec<String>,
}

impl Default for ParameterGroupConfig {
    fn default() -> Self {
        Self {
            base_lr: 2e-4,
            embedding_lr: None,
            weight_decay: 0.0,
            embedding_patterns: vec![
                "embed_tokens".to_string(),
                "lm_head".to_string(),
                "wte".to_string(),      // GPT-2 style
                "wpe".to_string(),      // Position embeddings
                "token_embd".to_string(), // GGUF style
                "output".to_string(),   // Output projection
            ],
        }
    }
}

impl ParameterGroupConfig {
    /// Create a new config with the given base learning rate.
    pub fn new(base_lr: f64) -> Self {
        Self {
            base_lr,
            ..Default::default()
        }
    }

    /// Set the embedding learning rate.
    pub fn with_embedding_lr(mut self, lr: f64) -> Self {
        self.embedding_lr = Some(lr);
        self
    }

    /// Set the weight decay.
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Add an embedding pattern.
    pub fn add_embedding_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.embedding_patterns.push(pattern.into());
        self
    }
}

/// Builder for creating parameter groups from model parameters.
pub struct ParameterGroupBuilder {
    config: ParameterGroupConfig,
    embeddings: ParameterGroup,
    non_embeddings: ParameterGroup,
}

impl ParameterGroupBuilder {
    /// Create a new builder with the given config.
    pub fn new(config: ParameterGroupConfig) -> Self {
        let embedding_lr = config.embedding_lr.unwrap_or(config.base_lr);

        Self {
            embeddings: ParameterGroup::new("embeddings")
                .with_lr(embedding_lr)
                .with_weight_decay(config.weight_decay),
            non_embeddings: ParameterGroup::new("non_embeddings")
                .with_lr(config.base_lr)
                .with_weight_decay(config.weight_decay),
            config,
        }
    }

    /// Classify a parameter by name.
    pub fn add_parameter(&mut self, name: &str) {
        if self.is_embedding_param(name) {
            self.embeddings.add_param(name);
        } else {
            self.non_embeddings.add_param(name);
        }
    }

    /// Check if a parameter name matches embedding patterns.
    fn is_embedding_param(&self, name: &str) -> bool {
        let name_lower = name.to_lowercase();
        self.config
            .embedding_patterns
            .iter()
            .any(|pattern| name_lower.contains(&pattern.to_lowercase()))
    }

    /// Add all parameters from a name iterator.
    pub fn add_parameters<'a>(&mut self, names: impl Iterator<Item = &'a str>) {
        for name in names {
            self.add_parameter(name);
        }
    }

    /// Build the parameter groups.
    pub fn build(self) -> Vec<ParameterGroup> {
        let mut groups = Vec::new();

        if !self.non_embeddings.is_empty() {
            groups.push(self.non_embeddings);
        }
        if !self.embeddings.is_empty() {
            groups.push(self.embeddings);
        }

        groups
    }

    /// Get summary of parameter grouping.
    pub fn summary(&self) -> String {
        format!(
            "Parameter groups:\n  - {} non-embedding params (lr={:.2e})\n  - {} embedding params (lr={:.2e})",
            self.non_embeddings.len(),
            self.non_embeddings.learning_rate,
            self.embeddings.len(),
            self.embeddings.learning_rate,
        )
    }
}

/// Create parameter groups from a model's trainable parameters.
///
/// # Arguments
/// * `param_names` - Iterator of trainable parameter names
/// * `base_lr` - Base learning rate for non-embedding parameters
/// * `embedding_lr` - Optional separate learning rate for embeddings
/// * `weight_decay` - Weight decay for all parameters
///
/// # Returns
/// Vector of parameter groups with appropriate learning rates
pub fn create_parameter_groups<'a>(
    param_names: impl Iterator<Item = &'a str>,
    base_lr: f64,
    embedding_lr: Option<f64>,
    weight_decay: f64,
) -> Vec<ParameterGroup> {
    let config = ParameterGroupConfig {
        base_lr,
        embedding_lr,
        weight_decay,
        ..Default::default()
    };

    let mut builder = ParameterGroupBuilder::new(config);
    builder.add_parameters(param_names);
    builder.build()
}

/// Get the learning rate for a specific parameter.
pub fn get_param_lr(
    param_name: &str,
    groups: &[ParameterGroup],
    default_lr: f64,
) -> f64 {
    for group in groups {
        if group.contains(param_name) {
            return group.learning_rate;
        }
    }
    default_lr
}

/// Create a learning rate map from parameter groups.
pub fn lr_map_from_groups(groups: &[ParameterGroup]) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    for group in groups {
        for name in &group.param_names {
            map.insert(name.clone(), group.learning_rate);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_grouping() {
        let param_names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "model.layers.1.mlp.gate_proj.lora_A.weight",
            "lm_head.weight",
        ];

        let groups = create_parameter_groups(
            param_names.iter().map(|s| *s),
            2e-4,
            Some(5e-5),
            0.01,
        );

        assert_eq!(groups.len(), 2);

        // Find embedding group
        let embedding_group = groups.iter().find(|g| g.description == "embeddings").unwrap();
        assert_eq!(embedding_group.len(), 2); // embed_tokens and lm_head
        assert!((embedding_group.learning_rate - 5e-5).abs() < 1e-10);

        // Find non-embedding group
        let non_embedding_group = groups.iter().find(|g| g.description == "non_embeddings").unwrap();
        assert_eq!(non_embedding_group.len(), 3); // 3 LoRA params
        assert!((non_embedding_group.learning_rate - 2e-4).abs() < 1e-10);
    }

    #[test]
    fn test_no_embedding_lr() {
        let param_names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ];

        // Without separate embedding LR
        let groups = create_parameter_groups(param_names.iter().map(|s| *s), 2e-4, None, 0.0);

        // Both groups should have the same LR
        for group in &groups {
            assert!((group.learning_rate - 2e-4).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lr_map() {
        let groups = vec![
            {
                let mut g = ParameterGroup::new("lora").with_lr(2e-4);
                g.add_param("layers.0.q_proj.lora_A");
                g.add_param("layers.0.q_proj.lora_B");
                g
            },
            {
                let mut g = ParameterGroup::new("embeddings").with_lr(5e-5);
                g.add_param("embed_tokens");
                g
            },
        ];

        let lr_map = lr_map_from_groups(&groups);

        assert!((lr_map["layers.0.q_proj.lora_A"] - 2e-4).abs() < 1e-10);
        assert!((lr_map["embed_tokens"] - 5e-5).abs() < 1e-10);
    }

    #[test]
    fn test_custom_patterns() {
        let config = ParameterGroupConfig::new(1e-4)
            .with_embedding_lr(1e-5)
            .add_embedding_pattern("my_custom_embedding");

        let mut builder = ParameterGroupBuilder::new(config);
        builder.add_parameter("my_custom_embedding.weight");
        builder.add_parameter("regular_layer.weight");

        let groups = builder.build();

        let embedding_group = groups.iter().find(|g| g.description == "embeddings").unwrap();
        assert!(embedding_group.contains("my_custom_embedding.weight"));

        let non_embedding_group = groups.iter().find(|g| g.description == "non_embeddings").unwrap();
        assert!(non_embedding_group.contains("regular_layer.weight"));
    }
}
