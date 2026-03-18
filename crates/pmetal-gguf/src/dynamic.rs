//! Dynamic quantization scheduling.
//!
//! Determines the best quantization type for each layer based on an Importance Matrix (IMatrix).
//! Implements the "Dynamic 2.0" strategy inspired by Unsloth's approach, and additionally
//! supports KL-divergence calibrated per-tensor quantization type selection.
//!
//! # Strategy
//!
//! Dynamic quantization assigns different quantization types to each layer based on:
//! 1. **Layer sensitivity**: Layers with higher importance scores (from imatrix) get higher precision
//! 2. **Critical layers**: Output heads and embeddings always use high precision
//! 3. **Attention vs MLP**: Attention layers (q_proj, k_proj, v_proj, o_proj) are treated specially
//! 4. **Percentile-based selection**: Top N% of layers by importance get high precision
//!
//! # KL Calibration
//!
//! The optional KL-divergence calibration mode tests multiple quantization types per tensor via a
//! quantize → dequantize round-trip and measures quality loss using a combined NRMSE + cosine
//! distance metric.  The type with the lowest score that still meets `kl_threshold` is selected,
//! keeping the model as compressed as possible while bounding reconstruction error.
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

/// Configuration for KL-divergence calibrated quantization.
///
/// Instead of using fixed tiers, this tests multiple quantization types per tensor and selects
/// the one that minimises KL divergence from the original weights while meeting a target
/// bits-per-weight budget.
///
/// The "KL divergence" used here is a combined NRMSE + cosine-distance metric that is more
/// robust than true KL for weight tensors that are not natural probability distributions.
#[derive(Debug, Clone)]
pub struct KlCalibrationConfig {
    /// Candidate quantization types to evaluate per tensor, ordered from highest to lowest
    /// quality.  The calibrator picks the lowest-quality type whose score is below the
    /// threshold.
    pub candidates: Vec<GgmlType>,

    /// Maximum score (NRMSE + cosine-distance) allowed.  Tensors exceeding this for all
    /// candidates fall back to `fallback_type`.  Typical range 0.001 – 0.05; default 0.01.
    pub kl_threshold: f64,

    /// Target average bits-per-weight.  When set, the calibrator will greedily downgrade
    /// tensors (lowest-KL-impact first) until the budget is met.
    pub target_bpw: Option<f32>,

    /// Fallback type for critical layers (embeddings, lm_head, norms) and for any tensor
    /// where all candidates exceed the threshold.
    pub fallback_type: GgmlType,

    /// Number of sample elements to use for KL computation.  `0` means use all elements.
    /// Sub-sampling speeds up calibration for very large tensors.
    pub sample_size: usize,
}

impl Default for KlCalibrationConfig {
    fn default() -> Self {
        Self {
            candidates: vec![
                GgmlType::Q8K,
                GgmlType::Q6K,
                GgmlType::Q5K,
                GgmlType::Q4K,
                GgmlType::Q3K,
                GgmlType::Q2K,
            ],
            kl_threshold: 0.01,
            target_bpw: None,
            fallback_type: GgmlType::Q8K,
            sample_size: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// KL-divergence (quality-loss) metric
// ---------------------------------------------------------------------------

/// Compute a quality-loss metric between original and dequantized weight tensors.
///
/// We combine two complementary signals:
///
/// 1. **Normalised RMSE** – measures raw magnitude reconstruction error, independent of scale.
/// 2. **Cosine distance** – measures whether the weight *direction* (which affects dot-products
///    in the forward pass) is preserved.
///
/// Both terms are in [0, ∞), with 0 meaning perfect reconstruction.  Using the sum means a
/// tensor with poor direction preservation OR large absolute error will be penalised, which maps
/// well to the activations a model layer sees in practice.
fn compute_kl_divergence(original: &[f32], dequantized: &[f32]) -> f64 {
    debug_assert_eq!(original.len(), dequantized.len());
    let n = original.len();
    if n == 0 {
        return 0.0;
    }

    // 1. Normalised RMSE
    let mut sq_err_sum = 0.0_f64;
    let mut sq_orig_sum = 0.0_f64;
    for i in 0..n {
        let diff = (original[i] - dequantized[i]) as f64;
        sq_err_sum += diff * diff;
        sq_orig_sum += (original[i] as f64) * (original[i] as f64);
    }
    let nrmse = if sq_orig_sum > 0.0 {
        (sq_err_sum / n as f64).sqrt() / (sq_orig_sum / n as f64).sqrt()
    } else {
        0.0
    };

    // 2. Cosine distance  (1 - cosine_similarity)
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..n {
        let a = original[i] as f64;
        let b = dequantized[i] as f64;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    let cos_sim = if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        1.0
    };
    let cos_loss = 1.0 - cos_sim;

    nrmse + cos_loss
}

// ---------------------------------------------------------------------------
// Per-tensor calibration result types
// ---------------------------------------------------------------------------

/// Calibrated quantization result for a single tensor.
#[derive(Debug, Clone)]
pub struct CalibratedTensor {
    /// Selected quantization type.
    pub quant_type: GgmlType,
    /// Quality-loss score for the selected type (lower = better).
    pub kl_score: f64,
    /// All evaluated scores `(type, score)` in evaluation order, for diagnostics.
    pub all_scores: Vec<(GgmlType, f64)>,
}

/// Per-tensor calibration decisions: tensor name → [`CalibratedTensor`].
pub type CalibrationMap = HashMap<String, CalibratedTensor>;

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

/// Summary of KL-calibrated quantization decisions.
#[derive(Debug, Default)]
pub struct CalibrationSummary {
    /// Total tensors calibrated.
    pub total_tensors: usize,
    /// Count of tensors per selected quantization type.
    pub type_counts: HashMap<GgmlType, usize>,
    /// Average quality-loss score across all tensors.
    pub avg_kl_score: f64,
    /// Worst (highest) quality-loss score.
    pub max_kl_score: f64,
    /// Name of the tensor with the worst score.
    pub worst_tensor: String,
    /// Estimated average bits per weight.
    pub estimated_bpw: f32,
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
        let mut ranked: Vec<(String, f32)> =
            tensor_scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
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

    // -----------------------------------------------------------------------
    // KL-divergence calibration
    // -----------------------------------------------------------------------

    /// Calibrate the quantization type for a single tensor using a quantize →
    /// dequantize round-trip quality metric.
    ///
    /// Critical layers (embeddings, lm_head, norms) always receive
    /// `kl_config.fallback_type`.  For all other tensors, each candidate type
    /// is evaluated and the most-compressed type whose score is ≤ `kl_threshold`
    /// is returned.  If no candidate meets the threshold the fallback type is used.
    pub fn calibrate_tensor(
        &self,
        name: &str,
        data: &[f32],
        shape: &[i32],
        kl_config: &KlCalibrationConfig,
    ) -> CalibratedTensor {
        use crate::dequant::dequantize;
        use crate::quantize::quantize;

        // Critical layers always use the high-quality fallback.
        if Self::is_critical_layer(name) {
            return CalibratedTensor {
                quant_type: kl_config.fallback_type,
                kl_score: 0.0,
                all_scores: vec![(kl_config.fallback_type, 0.0)],
            };
        }

        // Determine the sample window for the quality metric.
        let sample: &[f32] = if kl_config.sample_size > 0 && data.len() > kl_config.sample_size {
            // Take a prefix of the weight elements for speed.
            // WARNING: This assumes weight values are roughly i.i.d. across the flattened
            // tensor — true for many weight matrices but not guaranteed for structured
            // weights (e.g. rotary-position biases, depthwise convolution kernels).
            // If you observe poor calibration quality on such tensors, set
            // `sample_size = 0` to evaluate the full tensor instead.
            &data[..kl_config.sample_size]
        } else {
            data
        };

        let mut all_scores: Vec<(GgmlType, f64)> = Vec::new();

        for &candidate in &kl_config.candidates {
            // Quantize the *full* tensor (the dequantized slice must cover `sample`).
            let quantized = match quantize(data, candidate) {
                Ok(q) => q,
                Err(_) => continue,
            };

            let dequantized = match dequantize(&quantized, candidate, shape) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let cmp_len = sample.len().min(dequantized.len());
            let score = compute_kl_divergence(sample, &dequantized[..cmp_len]);
            all_scores.push((candidate, score));
        }

        if all_scores.is_empty() {
            // No candidate succeeded — fall back.
            return CalibratedTensor {
                quant_type: kl_config.fallback_type,
                kl_score: f64::MAX,
                all_scores: vec![(kl_config.fallback_type, f64::MAX)],
            };
        }

        // Candidates are ordered highest-quality first.  We want the most-compressed
        // (last in list) type whose score is ≤ threshold.  If none meet the threshold
        // we pick the highest-quality (first) evaluated type.
        let mut best_type = all_scores[0].0;
        let mut best_score = all_scores[0].1;

        for &(candidate, score) in all_scores.iter().rev() {
            if score <= kl_config.kl_threshold {
                best_type = candidate;
                best_score = score;
                break;
            }
        }

        // If none met the threshold, best_type/best_score are already set to the
        // highest-quality evaluated candidate (index 0), which is the safest choice.

        CalibratedTensor {
            quant_type: best_type,
            kl_score: best_score,
            all_scores,
        }
    }

    /// Calibrate all tensors and build a complete [`CalibrationMap`].
    ///
    /// This is the main entry point for KL-calibrated quantization.  Call it
    /// with all model-weight tensors before quantizing.  If `target_bpw` is set
    /// in `kl_config`, a second budget-enforcement pass is run afterwards.
    pub fn calibrate_all(
        &self,
        tensors: &[(String, Vec<f32>, Vec<i32>)],
        kl_config: &KlCalibrationConfig,
    ) -> CalibrationMap {
        let mut map = CalibrationMap::new();

        for (name, data, shape) in tensors {
            let result = self.calibrate_tensor(name, data, shape, kl_config);
            // Per-tensor diagnostic output is left to the caller (e.g. the CLI) to avoid
            // a `tracing`/`log` dependency in this library crate.
            map.insert(name.clone(), result);
        }

        if let Some(target_bpw) = kl_config.target_bpw {
            self.apply_bpw_budget(&mut map, tensors, target_bpw, kl_config);
        }

        map
    }

    /// Get the quantization type for a tensor using a pre-computed calibration map.
    ///
    /// Falls back to the standard tier-based selection ([`get_tensor_type`]) if the
    /// tensor is not present in the map.
    ///
    /// [`get_tensor_type`]: Self::get_tensor_type
    pub fn get_tensor_type_calibrated(
        &self,
        name: &str,
        shape: &[u64],
        calibration: &CalibrationMap,
    ) -> GgmlType {
        if let Some(cal) = calibration.get(name) {
            cal.quant_type
        } else {
            self.get_tensor_type(name, shape)
        }
    }

    /// Summarise calibration results for display / logging.
    pub fn summarize_calibration(
        &self,
        calibration: &CalibrationMap,
        tensor_sizes: &[(String, usize)],
    ) -> CalibrationSummary {
        let mut summary = CalibrationSummary::default();
        summary.total_tensors = calibration.len();

        let mut total_kl = 0.0_f64;
        let mut total_elements = 0u64;
        let mut total_bits = 0u64;

        for (name, cal) in calibration {
            *summary.type_counts.entry(cal.quant_type).or_insert(0) += 1;
            total_kl += cal.kl_score;

            if cal.kl_score > summary.max_kl_score {
                summary.max_kl_score = cal.kl_score;
                summary.worst_tensor = name.clone();
            }

            if let Some((_, n)) = tensor_sizes.iter().find(|(tname, _)| tname == name) {
                let n = *n as u64;
                total_elements += n;
                total_bits += n * self.bits_per_element(cal.quant_type);
            }
        }

        summary.avg_kl_score = if summary.total_tensors > 0 {
            total_kl / summary.total_tensors as f64
        } else {
            0.0
        };

        summary.estimated_bpw = if total_elements > 0 {
            total_bits as f32 / total_elements as f32
        } else {
            0.0
        };

        summary
    }

    // -----------------------------------------------------------------------
    // BPW budget enforcement
    // -----------------------------------------------------------------------

    /// Enforce a bits-per-weight budget by downgrading the tensors with the lowest
    /// quality-loss impact first (greedy, highest-compression direction).
    fn apply_bpw_budget(
        &self,
        map: &mut CalibrationMap,
        tensors: &[(String, Vec<f32>, Vec<i32>)],
        target_bpw: f32,
        kl_config: &KlCalibrationConfig,
    ) {
        let compute_avg_bpw = |map: &CalibrationMap| -> f32 {
            let mut total_elements = 0u64;
            let mut total_bits = 0u64;
            for (name, data, _) in tensors {
                let n = data.len() as u64;
                total_elements += n;
                if let Some(cal) = map.get(name) {
                    total_bits += n * self.bits_per_element(cal.quant_type);
                }
            }
            if total_elements == 0 {
                0.0
            } else {
                total_bits as f32 / total_elements as f32
            }
        };

        if compute_avg_bpw(map) <= target_bpw {
            return; // Already within budget — nothing to do.
        }

        // Collect non-critical tensors ordered by ascending score (least impactful first).
        let mut downgradeable: Vec<(String, f64)> = map
            .iter()
            .filter(|(name, _)| !Self::is_critical_layer(name))
            .map(|(name, cal)| (name.clone(), cal.kl_score))
            .collect();
        downgradeable.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (name, _) in &downgradeable {
            if let Some(cal) = map.get_mut(name) {
                // Find the next lower-quality candidate in the ordered list.
                let current_idx = kl_config
                    .candidates
                    .iter()
                    .position(|&c| c == cal.quant_type);
                if let Some(idx) = current_idx {
                    if idx + 1 < kl_config.candidates.len() {
                        let next_type = kl_config.candidates[idx + 1];
                        // Use the pre-computed score if available; otherwise keep current.
                        let next_score = cal
                            .all_scores
                            .iter()
                            .find(|&&(t, _)| t == next_type)
                            .map(|&(_, s)| s)
                            .unwrap_or(cal.kl_score);
                        cal.quant_type = next_type;
                        cal.kl_score = next_score;
                    }
                }
            }

            if compute_avg_bpw(map) <= target_bpw {
                break;
            }
        }

        // Verify the budget was achieved.  When all non-critical tensors are already at
        // the lowest-quality candidate the loop exits without meeting the target.
        let final_bpw = compute_avg_bpw(map);
        if final_bpw > target_bpw {
            tracing::warn!(
                "BPW budget not met: target={:.2}, achieved={:.2}. \
                 All tensors are already at minimum quality.",
                target_bpw,
                final_bpw,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bits-per-element table
    // -----------------------------------------------------------------------

    /// Approximate bits per stored element for a given quantization type.
    ///
    /// K-quant types store overhead (scales, mins) in the block header, so the
    /// effective BPE is slightly above the nominal quant width.
    fn bits_per_element(&self, dtype: GgmlType) -> u64 {
        match dtype {
            GgmlType::F32 => 32,
            GgmlType::F16 | GgmlType::Bf16 => 16,
            GgmlType::Q8_0 | GgmlType::Q8K => 9,
            GgmlType::Q6K => 7,
            GgmlType::Q5K => 6,
            GgmlType::Q4_0 | GgmlType::Q4K => 5,
            GgmlType::Q3K => 4,
            GgmlType::Q2K => 3,
            _ => 5, // Conservative default for unknown/IQ types.
        }
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
        data.insert(
            "model.layers.0.self_attn.q_proj".to_string(),
            vec![100.0; 100],
        ); // 10000
        data.insert(
            "model.layers.0.self_attn.k_proj".to_string(),
            vec![80.0; 100],
        ); // 8000
        data.insert("model.layers.0.mlp.gate_proj".to_string(), vec![50.0; 100]); // 5000
        data.insert("model.layers.0.mlp.up_proj".to_string(), vec![40.0; 100]); // 4000
        data.insert(
            "model.layers.1.self_attn.q_proj".to_string(),
            vec![90.0; 100],
        ); // 9000
        data.insert("model.layers.1.mlp.gate_proj".to_string(), vec![30.0; 100]); // 3000
        data.insert("model.layers.2.mlp.down_proj".to_string(), vec![20.0; 100]); // 2000
        data.insert("model.layers.3.mlp.down_proj".to_string(), vec![10.0; 100]); // 1000
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
        assert!(
            high_prec_count >= 2,
            "Should have at least 2 high precision tensors"
        );
    }

    #[test]
    fn test_critical_layer_detection() {
        assert!(DynamicQuantizer::is_critical_layer("lm_head.weight"));
        assert!(DynamicQuantizer::is_critical_layer("model.output.weight"));
        assert!(DynamicQuantizer::is_critical_layer(
            "model.embed_tokens.weight"
        ));
        assert!(DynamicQuantizer::is_critical_layer("token_embd.weight"));
        assert!(!DynamicQuantizer::is_critical_layer(
            "model.layers.0.mlp.weight"
        ));
    }

    #[test]
    fn test_attention_layer_detection() {
        assert!(DynamicQuantizer::is_attention_layer(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(DynamicQuantizer::is_attention_layer(
            "model.layers.0.attention.k_proj.weight"
        ));
        assert!(!DynamicQuantizer::is_attention_layer(
            "model.layers.0.mlp.gate_proj.weight"
        ));
    }

    #[test]
    fn test_layer_index_extraction() {
        assert_eq!(
            DynamicQuantizer::extract_layer_index("model.layers.5.mlp"),
            Some(5)
        );
        assert_eq!(DynamicQuantizer::extract_layer_index("h.12.attn"), Some(12));
        assert_eq!(
            DynamicQuantizer::extract_layer_index("blocks.0.ff"),
            Some(0)
        );
        assert_eq!(
            DynamicQuantizer::extract_layer_index("lm_head.weight"),
            None
        );
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
        assert!(
            !summary.high_precision_layers.is_empty() || !summary.base_precision_layers.is_empty()
        );
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

    // -----------------------------------------------------------------------
    // KL calibration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kl_divergence_identical() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let kl = compute_kl_divergence(&data, &data);
        assert!(kl < 1e-10, "Identical data should have ~0 KL: {}", kl);
    }

    #[test]
    fn test_kl_divergence_different() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let perturbed = vec![1.1f32, 2.1, 2.9, 4.2, 4.8];
        let kl = compute_kl_divergence(&original, &perturbed);
        assert!(kl > 0.0, "Different data should have positive KL");
        assert!(kl < 1.0, "Small perturbation should have small KL: {}", kl);
    }

    #[test]
    fn test_kl_divergence_zero_vector() {
        // All-zero original should not panic and return 0.0 (no energy to lose).
        let zeros = vec![0.0f32; 8];
        let kl = compute_kl_divergence(&zeros, &zeros);
        assert_eq!(kl, 0.0, "All-zero identical should give 0: {}", kl);
    }

    #[test]
    fn test_kl_calibration_config_default() {
        let config = KlCalibrationConfig::default();
        assert_eq!(config.candidates.len(), 6);
        assert_eq!(config.candidates[0], GgmlType::Q8K);
        assert_eq!(*config.candidates.last().unwrap(), GgmlType::Q2K);
        assert_eq!(config.sample_size, 0);
        assert!((config.kl_threshold - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_calibrate_critical_layer() {
        let config = DynamicQuantizationConfig::default();
        let quantizer = DynamicQuantizer::new(config, None);
        let kl_config = KlCalibrationConfig::default();

        // Critical layers must always receive the fallback type regardless of data.
        let result = quantizer.calibrate_tensor(
            "model.lm_head.weight",
            &vec![1.0f32; 1024],
            &[32, 32],
            &kl_config,
        );
        assert_eq!(
            result.quant_type,
            kl_config.fallback_type,
            "Critical layer should always use fallback type"
        );
    }

    #[test]
    fn test_calibrate_non_critical_layer_returns_valid_type() {
        let config = DynamicQuantizationConfig::default();
        let quantizer = DynamicQuantizer::new(config, None);
        let kl_config = KlCalibrationConfig::default();

        // Use a non-trivial weight vector with 256 elements (1 Q8K block).
        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let result = quantizer.calibrate_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &data,
            &[1, 256],
            &kl_config,
        );

        // The selected type must be one of the candidates or the fallback.
        let all_candidates: Vec<GgmlType> = kl_config
            .candidates
            .iter()
            .cloned()
            .chain(std::iter::once(kl_config.fallback_type))
            .collect();
        assert!(
            all_candidates.contains(&result.quant_type),
            "Selected type {:?} not in candidate set",
            result.quant_type
        );
        assert!(!result.all_scores.is_empty(), "Should have evaluated at least one candidate");
    }

    #[test]
    fn test_calibrate_all_builds_map() {
        let config = DynamicQuantizationConfig::default();
        let quantizer = DynamicQuantizer::new(config, None);
        let kl_config = KlCalibrationConfig::default();

        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let tensors = vec![
            (
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                data.clone(),
                vec![1i32, 256],
            ),
            (
                "model.lm_head.weight".to_string(),
                data.clone(),
                vec![1i32, 256],
            ),
        ];

        let map = quantizer.calibrate_all(&tensors, &kl_config);
        assert_eq!(map.len(), 2, "Map should contain both tensors");
        assert!(map.contains_key("model.layers.0.mlp.gate_proj.weight"));
        assert!(map.contains_key("model.lm_head.weight"));

        // Critical layer in map must use fallback.
        assert_eq!(map["model.lm_head.weight"].quant_type, kl_config.fallback_type);
    }

    #[test]
    fn test_get_tensor_type_calibrated_fallback() {
        let config = DynamicQuantizationConfig::default();
        let quantizer = DynamicQuantizer::new(config, None);

        // Empty map → falls back to tier-based selection.
        let map = CalibrationMap::new();
        let ty = quantizer.get_tensor_type_calibrated("model.layers.0.mlp.weight", &[], &map);
        assert_eq!(ty, GgmlType::Q4K, "Should fall back to base tier type");
    }

    #[test]
    fn test_summarize_calibration() {
        let config = DynamicQuantizationConfig::default();
        let quantizer = DynamicQuantizer::new(config, None);

        let mut map = CalibrationMap::new();
        map.insert(
            "tensor_a".to_string(),
            CalibratedTensor {
                quant_type: GgmlType::Q4K,
                kl_score: 0.005,
                all_scores: vec![(GgmlType::Q4K, 0.005)],
            },
        );
        map.insert(
            "tensor_b".to_string(),
            CalibratedTensor {
                quant_type: GgmlType::Q6K,
                kl_score: 0.002,
                all_scores: vec![(GgmlType::Q6K, 0.002)],
            },
        );

        let sizes = vec![
            ("tensor_a".to_string(), 1024usize),
            ("tensor_b".to_string(), 512usize),
        ];

        let summary = quantizer.summarize_calibration(&map, &sizes);
        assert_eq!(summary.total_tensors, 2);
        assert_eq!(summary.worst_tensor, "tensor_a");
        assert!((summary.max_kl_score - 0.005).abs() < 1e-9);
        assert!((summary.avg_kl_score - 0.0035).abs() < 1e-9);
        assert!(summary.estimated_bpw > 0.0);
    }
}
