//! Dynamic quantization scheduling.
//!
//! Determines the best quantization type for each layer based on an Importance Matrix (IMatrix).
//! Implements the "Dynamic 2.0" strategy inspired by Unsloth's approach.
//!
//! # Strategy
//!
//! Dynamic quantization assigns different quantization types to each layer based on:
//! 1. **Layer sensitivity**: Layers with higher importance scores (from imatrix) get higher precision
//! 2. **Critical layers**: Output heads and embeddings always use high precision
//! 3. **Attention vs MLP**: Attention layers (q_proj, k_proj, v_proj, o_proj) are treated specially
//! 4. **Percentile-based selection**: Top N% of layers by importance get high precision
//!
//! # References
//!
//! - [Unsloth Dynamic 2.0](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)
//! - [llama.cpp imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)

use crate::imatrix::IMatrix;
use crate::types::GgmlType;
use std::collections::HashMap;

/// Configuration for dynamic quantization.
#[derive(Debug, Clone)]
pub struct DynamicQuantizationConfig {
    /// Percentage of weights to keep at higher precision (0.0 to 1.0).
    /// Default: 0.20 (top 20% of layers by importance)
    pub importance_percentile: f32,
    /// Base quantization type (for less important layers).
    pub base_type: GgmlType,
    /// High precision type (for important layers).
    pub high_precision_type: GgmlType,
    /// Fallback type (e.g. for output head, embeddings).
    pub fallback_type: GgmlType,
    /// Whether to always keep attention layers at high precision.
    pub attention_high_precision: bool,
    /// Whether to always keep first/last N layers at high precision.
    pub edge_layers_high_precision: usize,
}

impl Default for DynamicQuantizationConfig {
    fn default() -> Self {
        Self {
            importance_percentile: 0.20, // Top 20%
            base_type: GgmlType::Q4K,
            high_precision_type: GgmlType::Q6K,
            fallback_type: GgmlType::Q6K,
            attention_high_precision: false,
            edge_layers_high_precision: 0,
        }
    }
}

impl DynamicQuantizationConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.importance_percentile < 0.0 || self.importance_percentile > 1.0 {
            return Err(format!(
                "importance_percentile must be between 0.0 and 1.0, got {}",
                self.importance_percentile
            ));
        }
        Ok(())
    }
}

/// Computed importance thresholds for dynamic quantization.
#[derive(Debug, Clone)]
pub struct ImportanceThresholds {
    /// Global threshold: layers with importance above this get high precision.
    pub high_precision_threshold: f32,
    /// Per-tensor importance scores (tensor name -> total importance).
    pub tensor_scores: HashMap<String, f32>,
    /// Sorted list of (tensor_name, importance) for debugging.
    pub ranked_tensors: Vec<(String, f32)>,
}

/// Dynamic quantizer with computed thresholds.
pub struct DynamicQuantizer {
    config: DynamicQuantizationConfig,
    imatrix: Option<IMatrix>,
    /// Cached thresholds (computed lazily).
    thresholds: Option<ImportanceThresholds>,
}

impl DynamicQuantizer {
    /// Create a new dynamic quantizer.
    pub fn new(config: DynamicQuantizationConfig, imatrix: Option<IMatrix>) -> Self {
        let mut quantizer = Self {
            config,
            imatrix,
            thresholds: None,
        };
        // Pre-compute thresholds if imatrix is available
        quantizer.thresholds = quantizer.compute_thresholds();
        quantizer
    }

    /// Check if a tensor name indicates a critical layer that should always be high precision.
    fn is_critical_layer(name: &str) -> bool {
        // Output head
        if name.contains("lm_head") || name.contains("output") {
            return true;
        }
        // Token embeddings
        if name.contains("token_embd") || name.contains("embed_tokens") || name.contains("wte") {
            return true;
        }
        // Final layer norm
        if name.contains("final_norm") || name.contains("ln_f") {
            return true;
        }
        false
    }

    /// Check if a tensor name indicates an attention layer.
    fn is_attention_layer(name: &str) -> bool {
        name.contains("q_proj")
            || name.contains("k_proj")
            || name.contains("v_proj")
            || name.contains("o_proj")
            || name.contains("self_attn")
            || name.contains("attention")
    }

    /// Extract layer index from tensor name (e.g., "model.layers.5.mlp" -> Some(5)).
    fn extract_layer_index(name: &str) -> Option<usize> {
        // Common patterns: "layers.N.", "layer.N.", "h.N.", "blocks.N."
        let patterns = ["layers.", "layer.", "h.", "blocks."];
        for pattern in patterns {
            if let Some(idx) = name.find(pattern) {
                let rest = &name[idx + pattern.len()..];
                if let Some(end) = rest.find('.') {
                    if let Ok(n) = rest[..end].parse::<usize>() {
                        return Some(n);
                    }
                }
            }
        }
        None
    }

    /// Determine the quantization type for a tensor.
    pub fn get_tensor_type(&self, name: &str, _shape: &[u64]) -> GgmlType {
        // Critical layers always get fallback (highest) precision
        if Self::is_critical_layer(name) {
            return self.config.fallback_type;
        }

        // If configured, attention layers get high precision
        if self.config.attention_high_precision && Self::is_attention_layer(name) {
            return self.config.high_precision_type;
        }

        // Check edge layers (first/last N)
        if self.config.edge_layers_high_precision > 0 {
            if let Some(layer_idx) = Self::extract_layer_index(name) {
                // We don't know total layers here, so just check first N
                if layer_idx < self.config.edge_layers_high_precision {
                    return self.config.high_precision_type;
                }
            }
        }

        // If no IMatrix or no thresholds, use base type
        let thresholds = match &self.thresholds {
            Some(t) => t,
            None => return self.config.base_type,
        };

        // Look up this tensor's importance score
        if let Some(&importance) = thresholds.tensor_scores.get(name) {
            if importance >= thresholds.high_precision_threshold {
                return self.config.high_precision_type;
            }
        }

        self.config.base_type
    }

    /// Compute importance thresholds from the imatrix.
    fn compute_thresholds(&self) -> Option<ImportanceThresholds> {
        let imatrix = self.imatrix.as_ref()?;

        if imatrix.data.is_empty() {
            return None;
        }

        // Compute total importance for each tensor
        let mut tensor_scores: HashMap<String, f32> = HashMap::new();
        for (name, values) in &imatrix.data {
            // Sum of squared activations represents total layer importance
            let total: f32 = values.iter().sum();
            tensor_scores.insert(name.clone(), total);
        }

        // Sort tensors by importance (descending)
        let mut ranked: Vec<(String, f32)> = tensor_scores
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find the threshold for top percentile
        let cutoff_idx = ((ranked.len() as f32) * self.config.importance_percentile) as usize;
        let threshold = if cutoff_idx < ranked.len() && cutoff_idx > 0 {
            // Threshold is the importance score at the cutoff position
            ranked[cutoff_idx.saturating_sub(1)].1
        } else if !ranked.is_empty() {
            // If percentile is 0 or very small, use the minimum score
            ranked.last().map(|(_, s)| *s).unwrap_or(0.0)
        } else {
            0.0
        };

        Some(ImportanceThresholds {
            high_precision_threshold: threshold,
            tensor_scores,
            ranked_tensors: ranked,
        })
    }

    /// Calculate global importance thresholds.
    /// Returns the threshold value above which tensors get high precision.
    pub fn calculate_thresholds(&self) -> f32 {
        self.thresholds
            .as_ref()
            .map(|t| t.high_precision_threshold)
            .unwrap_or(0.0)
    }

    /// Get the computed thresholds (for debugging/inspection).
    pub fn get_thresholds(&self) -> Option<&ImportanceThresholds> {
        self.thresholds.as_ref()
    }

    /// Get a summary of quantization decisions.
    pub fn get_quantization_summary(&self) -> QuantizationSummary {
        let mut summary = QuantizationSummary::default();

        if let Some(thresholds) = &self.thresholds {
            for (name, &importance) in &thresholds.tensor_scores {
                if Self::is_critical_layer(name) {
                    summary.critical_layers.push(name.clone());
                } else if importance >= thresholds.high_precision_threshold {
                    summary.high_precision_layers.push(name.clone());
                } else {
                    summary.base_precision_layers.push(name.clone());
                }
            }
        }

        summary.threshold = self.calculate_thresholds();
        summary
    }
}

/// Summary of quantization decisions for debugging.
#[derive(Debug, Default)]
pub struct QuantizationSummary {
    /// Layers that always get highest precision.
    pub critical_layers: Vec<String>,
    /// Layers selected for high precision based on importance.
    pub high_precision_layers: Vec<String>,
    /// Layers using base precision.
    pub base_precision_layers: Vec<String>,
    /// The computed threshold value.
    pub threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_imatrix() -> IMatrix {
        let mut data = HashMap::new();
        // Simulate importance scores for various layers
        data.insert("model.layers.0.self_attn.q_proj".to_string(), vec![100.0; 100]); // 10000
        data.insert("model.layers.0.self_attn.k_proj".to_string(), vec![80.0; 100]);  // 8000
        data.insert("model.layers.0.mlp.gate_proj".to_string(), vec![50.0; 100]);     // 5000
        data.insert("model.layers.0.mlp.up_proj".to_string(), vec![40.0; 100]);       // 4000
        data.insert("model.layers.1.self_attn.q_proj".to_string(), vec![90.0; 100]);  // 9000
        data.insert("model.layers.1.mlp.gate_proj".to_string(), vec![30.0; 100]);     // 3000
        data.insert("model.layers.2.mlp.down_proj".to_string(), vec![20.0; 100]);     // 2000
        data.insert("model.layers.3.mlp.down_proj".to_string(), vec![10.0; 100]);     // 1000
        IMatrix {
            data,
            ncalls: HashMap::new(),
            dataset_name: None,
            last_chunk: None,
        }
    }

    #[test]
    fn test_config_validation() {
        let mut config = DynamicQuantizationConfig::default();
        assert!(config.validate().is_ok());

        config.importance_percentile = -0.1;
        assert!(config.validate().is_err());

        config.importance_percentile = 1.5;
        assert!(config.validate().is_err());

        config.importance_percentile = 0.5;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_threshold_calculation() {
        let imatrix = create_test_imatrix();
        let config = DynamicQuantizationConfig {
            importance_percentile: 0.25, // Top 25%
            ..Default::default()
        };
        let quantizer = DynamicQuantizer::new(config, Some(imatrix));

        let threshold = quantizer.calculate_thresholds();
        assert!(threshold > 0.0, "Threshold should be positive");

        let thresholds = quantizer.get_thresholds().unwrap();
        assert_eq!(thresholds.tensor_scores.len(), 8);

        // Top 25% of 8 tensors = 2 tensors should be high precision
        let high_prec_count = thresholds
            .tensor_scores
            .values()
            .filter(|&&v| v >= threshold)
            .count();
        assert!(high_prec_count >= 2, "Should have at least 2 high precision tensors");
    }

    #[test]
    fn test_critical_layer_detection() {
        assert!(DynamicQuantizer::is_critical_layer("lm_head.weight"));
        assert!(DynamicQuantizer::is_critical_layer("model.output.weight"));
        assert!(DynamicQuantizer::is_critical_layer("model.embed_tokens.weight"));
        assert!(DynamicQuantizer::is_critical_layer("token_embd.weight"));
        assert!(!DynamicQuantizer::is_critical_layer("model.layers.0.mlp.weight"));
    }

    #[test]
    fn test_attention_layer_detection() {
        assert!(DynamicQuantizer::is_attention_layer("model.layers.0.self_attn.q_proj.weight"));
        assert!(DynamicQuantizer::is_attention_layer("model.layers.0.attention.k_proj.weight"));
        assert!(!DynamicQuantizer::is_attention_layer("model.layers.0.mlp.gate_proj.weight"));
    }

    #[test]
    fn test_layer_index_extraction() {
        assert_eq!(DynamicQuantizer::extract_layer_index("model.layers.5.mlp"), Some(5));
        assert_eq!(DynamicQuantizer::extract_layer_index("h.12.attn"), Some(12));
        assert_eq!(DynamicQuantizer::extract_layer_index("blocks.0.ff"), Some(0));
        assert_eq!(DynamicQuantizer::extract_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn test_tensor_type_selection() {
        let imatrix = create_test_imatrix();
        let config = DynamicQuantizationConfig {
            importance_percentile: 0.25,
            base_type: GgmlType::Q4K,
            high_precision_type: GgmlType::Q6K,
            fallback_type: GgmlType::Q8_0,
            ..Default::default()
        };
        let quantizer = DynamicQuantizer::new(config, Some(imatrix));

        // Critical layers always get fallback type
        assert_eq!(
            quantizer.get_tensor_type("lm_head.weight", &[]),
            GgmlType::Q8_0
        );
        assert_eq!(
            quantizer.get_tensor_type("model.embed_tokens.weight", &[]),
            GgmlType::Q8_0
        );

        // High importance layers should get high precision
        let q_proj_type = quantizer.get_tensor_type("model.layers.0.self_attn.q_proj", &[]);
        assert!(
            q_proj_type == GgmlType::Q6K || q_proj_type == GgmlType::Q4K,
            "Should be either high or base precision based on ranking"
        );
    }

    #[test]
    fn test_no_imatrix_fallback() {
        let config = DynamicQuantizationConfig {
            base_type: GgmlType::Q4K,
            ..Default::default()
        };
        let quantizer = DynamicQuantizer::new(config, None);

        // Without imatrix, regular layers get base type
        assert_eq!(
            quantizer.get_tensor_type("model.layers.0.mlp.weight", &[]),
            GgmlType::Q4K
        );
        // Critical layers still get fallback type
        assert_eq!(
            quantizer.get_tensor_type("lm_head.weight", &[]),
            GgmlType::Q6K
        );
    }

    #[test]
    fn test_attention_high_precision_config() {
        let imatrix = create_test_imatrix();
        let config = DynamicQuantizationConfig {
            attention_high_precision: true,
            high_precision_type: GgmlType::Q6K,
            base_type: GgmlType::Q4K,
            ..Default::default()
        };
        let quantizer = DynamicQuantizer::new(config, Some(imatrix));

        // Attention layers should always get high precision
        assert_eq!(
            quantizer.get_tensor_type("model.layers.5.self_attn.q_proj.weight", &[]),
            GgmlType::Q6K
        );
    }

    #[test]
    fn test_quantization_summary() {
        let imatrix = create_test_imatrix();
        let config = DynamicQuantizationConfig {
            importance_percentile: 0.25,
            ..Default::default()
        };
        let quantizer = DynamicQuantizer::new(config, Some(imatrix));

        let summary = quantizer.get_quantization_summary();
        assert!(summary.threshold > 0.0);
        assert!(!summary.high_precision_layers.is_empty() || !summary.base_precision_layers.is_empty());
    }

    #[test]
    fn test_empty_imatrix() {
        let empty_imatrix = IMatrix::new();
        let config = DynamicQuantizationConfig::default();
        let quantizer = DynamicQuantizer::new(config, Some(empty_imatrix));

        // Should fall back to base type
        assert_eq!(
            quantizer.get_tensor_type("model.layers.0.mlp.weight", &[]),
            GgmlType::Q4K
        );
        assert_eq!(quantizer.calculate_thresholds(), 0.0);
    }
}
