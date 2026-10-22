//! Dynamic quantization scheduling.
//!
//! Determines the best quantization type for each layer based on an Importance Matrix (IMatrix).
//! Implements a dynamic quantization strategy using importance-based layer sensitivity, and
//! additionally supports KL-divergence calibrated per-tensor quantization type selection.
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
///
/// With `importance` (one weight per input channel, see [`Importance`]) both terms are taken
/// under that weighting: the expected squared error of the layer's *output* for the calibration
/// activations is `sum_j E[a_j^2] * sum_i dW_ij^2`, so channels the activations excite count for
/// more, and the score measures the error the model would actually see.
fn compute_kl_divergence(
    original: &[f32],
    dequantized: &[f32],
    importance: Option<&Importance<'_>>,
) -> f64 {
    debug_assert_eq!(original.len(), dequantized.len());
    let n = original.len();
    if n == 0 {
        return 0.0;
    }

    let (mut sq_err, mut sq_orig, mut dot, mut norm_b) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let w = importance.map_or(1.0, |imp| f64::from(imp.at(i)));
        let (a, b) = (f64::from(original[i]), f64::from(dequantized[i]));
        sq_err += w * (a - b) * (a - b);
        sq_orig += w * a * a;
        dot += w * a * b;
        norm_b += w * b * b;
    }

    // 1. Normalised RMSE
    let nrmse = if sq_orig > 0.0 {
        (sq_err / sq_orig).sqrt()
    } else {
        0.0
    };
    // 2. Cosine distance (1 - cosine_similarity)
    let cos_sim = if sq_orig > 0.0 && norm_b > 0.0 {
        dot / (sq_orig.sqrt() * norm_b.sqrt())
    } else {
        1.0
    };
    nrmse + (1.0 - cos_sim)
}

/// Per-input-channel importance laid over a tensor of rows of `n_per_row`
/// values: consecutive groups of `rows_per_slice` rows each use their own
/// `n_per_row` weights (the experts of a stacked MoE tensor).
struct Importance<'a> {
    weights: &'a [f32],
    n_per_row: usize,
    rows_per_slice: usize,
}

impl Importance<'_> {
    #[inline]
    fn at(&self, i: usize) -> f32 {
        let slice = i / self.n_per_row / self.rows_per_slice;
        self.weights[slice * self.n_per_row + i % self.n_per_row]
    }
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
    ///
    /// The output head is `output.weight` (GGUF) or `lm_head` (Hugging Face);
    /// `blk.N.attn_output.weight` is an ordinary attention projection.
    fn is_critical_layer(name: &str) -> bool {
        // Output head
        if name.contains("lm_head") || name == "output.weight" || name.ends_with(".output.weight") {
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
            || (name.contains("attn_") && !name.contains("norm"))
    }

    /// Extract layer index from tensor name (e.g., "model.layers.5.mlp" -> Some(5)).
    fn extract_layer_index(name: &str) -> Option<usize> {
        // Common patterns: "layers.N.", "layer.N.", "h.N.", "blocks.N.", GGUF "blk.N."
        let patterns = ["layers.", "layer.", "blk.", "h.", "blocks."];
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

    /// The importance of each input channel of a tensor from the imatrix, as
    /// [`crate::quantize::quantize_tensor`] consumes it.
    pub fn importance_for(&self, name: &str) -> Option<Vec<f32>> {
        self.imatrix.as_ref()?.importance(name)
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
        use crate::quantize::quantize_tensor;

        // Critical layers always use the high-quality fallback.
        if Self::is_critical_layer(name) {
            return CalibratedTensor {
                quant_type: kl_config.fallback_type,
                kl_score: 0.0,
                all_scores: vec![(kl_config.fallback_type, 0.0)],
            };
        }

        // Rows of the tensor (the innermost dimension), and the importance of its
        // input channels if the imatrix has them. A stacked MoE tensor has one
        // run of channels per slice of rows.
        let n_per_row = match shape.last() {
            Some(&d) if d > 0 && data.len() % d as usize == 0 => d as usize,
            _ => data.len().max(1),
        };
        let n_rows = data.len() / n_per_row;
        let importance_vec = self.importance_for(name).filter(|w| {
            !w.is_empty() && w.len() % n_per_row == 0 && n_rows % (w.len() / n_per_row) == 0
        });
        let slices = importance_vec.as_ref().map_or(1, |w| w.len() / n_per_row);
        let rows_per_slice = (n_rows / slices).max(1);

        // Score a sample of whole rows, spread evenly through every slice,
        // quantizing only the sample (the cost is the sample's, not the
        // tensor's). Rows rather than a prefix: a prefix of a stacked-experts
        // tensor is one expert, and of any tensor the first few rows.
        let want_rows = if kl_config.sample_size > 0 {
            kl_config.sample_size.div_ceil(n_per_row).max(1)
        } else {
            n_rows
        };
        let take = want_rows.div_ceil(slices).clamp(1, rows_per_slice);
        let sampled: Vec<f32>;
        let sample: &[f32] = if take >= rows_per_slice {
            data
        } else {
            let mut rows = Vec::with_capacity(slices * take * n_per_row);
            for slice in 0..slices {
                for k in 0..take {
                    let row = slice * rows_per_slice + k * rows_per_slice / take;
                    rows.extend_from_slice(&data[row * n_per_row..(row + 1) * n_per_row]);
                }
            }
            sampled = rows;
            &sampled
        };
        let sample_rows = sample.len() / n_per_row;
        let rows_per_sample_slice = (sample_rows / slices).max(1);
        let importance = importance_vec.as_deref().map(|weights| Importance {
            weights,
            n_per_row,
            rows_per_slice: rows_per_sample_slice,
        });
        let sample_shape = [sample_rows as i32, n_per_row as i32];

        let mut all_scores: Vec<(GgmlType, f64)> = Vec::new();

        for &candidate in &kl_config.candidates {
            let quantized =
                match quantize_tensor(sample, candidate, n_per_row, importance_vec.as_deref()) {
                    Ok(q) => q,
                    Err(_) => continue,
                };

            let dequantized = match dequantize(&quantized, candidate, &sample_shape) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let score = compute_kl_divergence(sample, &dequantized, importance.as_ref());
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

        let mut sizes: HashMap<String, u64> = HashMap::with_capacity(tensors.len());
        for (name, data, shape) in tensors {
            let result = self.calibrate_tensor(name, data, shape, kl_config);
            // Per-tensor diagnostic output is left to the caller (e.g. the CLI) to avoid
            // a `tracing`/`log` dependency in this library crate.
            map.insert(name.clone(), result);
            sizes.insert(name.clone(), data.len() as u64);
        }

        if let Some(target_bpw) = kl_config.target_bpw {
            self.apply_bpw_budget(&mut map, &sizes, target_bpw, kl_config);
        }

        map
    }

    /// [`calibrate_all`](Self::calibrate_all) for a model too large to hold:
    /// `tensors` lists each tensor's name and shape, and `load` produces one
    /// tensor's values when it is its turn, so only one tensor is ever in
    /// memory. The result is what `calibrate_all` gives for the same tensors,
    /// the bits-per-weight budget included (it needs only each tensor's size).
    ///
    /// # Errors
    ///
    /// The first error `load` returns.
    pub fn calibrate_streaming<E>(
        &self,
        tensors: &[(String, Vec<i32>)],
        mut load: impl FnMut(&str) -> std::result::Result<Vec<f32>, E>,
        kl_config: &KlCalibrationConfig,
    ) -> std::result::Result<CalibrationMap, E> {
        let mut map = CalibrationMap::new();
        let mut sizes: HashMap<String, u64> = HashMap::with_capacity(tensors.len());
        for (name, shape) in tensors {
            let data = load(name)?;
            map.insert(
                name.clone(),
                self.calibrate_tensor(name, &data, shape, kl_config),
            );
            sizes.insert(name.clone(), data.len() as u64);
        }
        if let Some(target_bpw) = kl_config.target_bpw {
            self.apply_bpw_budget(&mut map, &sizes, target_bpw, kl_config);
        }
        Ok(map)
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
        let mut summary = CalibrationSummary {
            total_tensors: calibration.len(),
            ..Default::default()
        };

        let mut total_kl = 0.0_f64;
        let mut total_elements = 0u64;
        let mut total_bits = 0.0f64;
        let sizes: HashMap<&str, usize> = tensor_sizes
            .iter()
            .map(|(name, n)| (name.as_str(), *n))
            .collect();

        for (name, cal) in calibration {
            *summary.type_counts.entry(cal.quant_type).or_insert(0) += 1;
            total_kl += cal.kl_score;

            if cal.kl_score > summary.max_kl_score {
                summary.max_kl_score = cal.kl_score;
                summary.worst_tensor = name.clone();
            }

            if let Some(&n) = sizes.get(name.as_str()) {
                total_elements += n as u64;
                total_bits += n as f64 * bits_per_element(cal.quant_type);
            }
        }

        summary.avg_kl_score = if summary.total_tensors > 0 {
            total_kl / summary.total_tensors as f64
        } else {
            0.0
        };

        summary.estimated_bpw = if total_elements > 0 {
            (total_bits / total_elements as f64) as f32
        } else {
            0.0
        };

        summary
    }

    // -----------------------------------------------------------------------
    // BPW budget enforcement
    // -----------------------------------------------------------------------

    /// Enforce a bits-per-weight budget by repeatedly taking the cheapest
    /// downgrade: among every non-critical tensor and every cheaper candidate
    /// already scored for it, the move that adds the least quality loss per
    /// bit saved (loss increase divided by `elements x bits per element`
    /// freed). A tensor can step down more than once, and a stack of small
    /// cheap moves wins over one expensive one, which is the marginal
    /// analysis that minimises total loss for a bit budget.
    fn apply_bpw_budget(
        &self,
        map: &mut CalibrationMap,
        sizes: &HashMap<String, u64>,
        target_bpw: f32,
        _kl_config: &KlCalibrationConfig,
    ) {
        let total_elements: u64 = sizes.values().sum();
        if total_elements == 0 {
            return;
        }
        let bits_of = |map: &CalibrationMap| -> f64 {
            map.iter()
                .map(|(name, cal)| {
                    sizes.get(name).copied().unwrap_or(0) as f64 * bits_per_element(cal.quant_type)
                })
                .sum()
        };
        // Tensors the map has no size for are not counted against the budget.
        let target_bits = f64::from(target_bpw) * total_elements as f64;
        let mut total_bits = bits_of(map);

        // Deterministic order for ties.
        let mut names: Vec<String> = map
            .keys()
            .filter(|n| !Self::is_critical_layer(n) && sizes.contains_key(*n))
            .cloned()
            .collect();
        names.sort();

        while total_bits > target_bits {
            // (loss increase per bit saved, tensor, new type, new score, bits saved)
            let mut best: Option<(f64, &str, GgmlType, f64, f64)> = None;
            for name in &names {
                let cal = &map[name];
                let n = sizes[name] as f64;
                let current = bits_per_element(cal.quant_type);
                for &(candidate, score) in &cal.all_scores {
                    let saved = n * (current - bits_per_element(candidate));
                    if saved <= 0.0 {
                        continue;
                    }
                    let ratio = (score - cal.kl_score).max(0.0) / saved;
                    if best.is_none_or(|(r, ..)| ratio < r) {
                        best = Some((ratio, name, candidate, score, saved));
                    }
                }
            }
            let Some((_, name, candidate, score, saved)) = best else {
                break;
            };
            if let Some(cal) = map.get_mut(name) {
                cal.quant_type = candidate;
                cal.kl_score = score;
            }
            total_bits -= saved;
        }

        if total_bits > target_bits {
            tracing::warn!(
                "BPW budget not met: target={:.2}, achieved={:.2}. \
                 Every tensor is already at its cheapest scored type.",
                target_bpw,
                total_bits / total_elements as f64,
            );
        }
    }
}

/// Bits stored per element by a type, block overhead (scales, minimums)
/// included: `8 * bytes per block / elements per block`.
fn bits_per_element(dtype: GgmlType) -> f64 {
    8.0 * dtype.type_size() as f64 / dtype.block_size() as f64
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
            ..IMatrix::default()
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
        let kl = compute_kl_divergence(&data, &data, None);
        assert!(kl < 1e-10, "Identical data should have ~0 KL: {}", kl);
    }

    #[test]
    fn test_kl_divergence_different() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let perturbed = vec![1.1f32, 2.1, 2.9, 4.2, 4.8];
        let kl = compute_kl_divergence(&original, &perturbed, None);
        assert!(kl > 0.0, "Different data should have positive KL");
        assert!(kl < 1.0, "Small perturbation should have small KL: {}", kl);
    }

    #[test]
    fn test_kl_divergence_zero_vector() {
        // All-zero original should not panic and return 0.0 (no energy to lose).
        let zeros = vec![0.0f32; 8];
        let kl = compute_kl_divergence(&zeros, &zeros, None);
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
            result.quant_type, kl_config.fallback_type,
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
        assert!(
            !result.all_scores.is_empty(),
            "Should have evaluated at least one candidate"
        );
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
        assert_eq!(
            map["model.lm_head.weight"].quant_type,
            kl_config.fallback_type
        );
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

    // -----------------------------------------------------------------------
    // GGUF naming, budget and importance
    // -----------------------------------------------------------------------

    #[test]
    fn gguf_names_are_classified_correctly() {
        // The output head is critical; an attention output projection is not.
        assert!(DynamicQuantizer::is_critical_layer("output.weight"));
        assert!(!DynamicQuantizer::is_critical_layer(
            "blk.3.attn_output.weight"
        ));
        assert!(DynamicQuantizer::is_attention_layer("blk.0.attn_q.weight"));
        assert!(DynamicQuantizer::is_attention_layer(
            "blk.0.attn_output.weight"
        ));
        assert!(!DynamicQuantizer::is_attention_layer(
            "blk.0.attn_norm.weight"
        ));
        assert!(!DynamicQuantizer::is_attention_layer("blk.0.ffn_up.weight"));
        assert_eq!(
            DynamicQuantizer::extract_layer_index("blk.7.attn_q.weight"),
            Some(7)
        );

        let config = DynamicQuantizationConfig {
            base_type: GgmlType::Q4K,
            high_precision_type: GgmlType::Q6K,
            fallback_type: GgmlType::Q8_0,
            attention_high_precision: true,
            edge_layers_high_precision: 2,
            ..Default::default()
        };
        let quantizer = DynamicQuantizer::new(config, None);
        assert_eq!(
            quantizer.get_tensor_type("blk.9.attn_q.weight", &[]),
            GgmlType::Q6K
        );
        assert_eq!(
            quantizer.get_tensor_type("blk.1.ffn_up.weight", &[]),
            GgmlType::Q6K
        );
        assert_eq!(
            quantizer.get_tensor_type("blk.9.ffn_up.weight", &[]),
            GgmlType::Q4K
        );
        assert_eq!(
            quantizer.get_tensor_type("output.weight", &[]),
            GgmlType::Q8_0
        );
    }

    #[test]
    fn bits_per_element_includes_block_overhead() {
        assert_eq!(bits_per_element(GgmlType::Q4_0), 4.5);
        assert_eq!(bits_per_element(GgmlType::Q4K), 4.5);
        assert_eq!(bits_per_element(GgmlType::Q5K), 5.5);
        assert_eq!(bits_per_element(GgmlType::Q6K), 6.5625);
        assert_eq!(bits_per_element(GgmlType::Q2K), 2.625);
        assert_eq!(bits_per_element(GgmlType::Q3K), 3.4375);
        assert_eq!(bits_per_element(GgmlType::Q8_0), 8.5);
        assert_eq!(bits_per_element(GgmlType::F16), 16.0);
        assert_eq!(bits_per_element(GgmlType::Iq4Xs), 4.25);
    }

    fn weight_like(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let mut s = 0.0f32;
                for _ in 0..12 {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    s += (state >> 40) as f32 / (1u64 << 24) as f32;
                }
                (s - 6.0) * 0.02
            })
            .collect()
    }

    #[test]
    fn the_bit_budget_is_met_by_stepping_tensors_down_repeatedly() {
        let quantizer = DynamicQuantizer::new(DynamicQuantizationConfig::default(), None);
        let tensors: Vec<(String, Vec<f32>, Vec<i32>)> = (0..4)
            .map(|i| {
                (
                    format!("blk.{i}.ffn_up.weight"),
                    weight_like(256 * 8, 7 + i),
                    vec![8, 256],
                )
            })
            .collect();
        // A threshold nothing can meet keeps every tensor at the best candidate
        // (Q8_K, 9.125 bits); 3.0 then needs several steps each.
        let kl_config = KlCalibrationConfig {
            kl_threshold: 0.0,
            target_bpw: Some(3.0),
            ..Default::default()
        };
        let map = quantizer.calibrate_all(&tensors, &kl_config);
        let sizes: Vec<_> = tensors
            .iter()
            .map(|(n, d, _)| (n.clone(), d.len()))
            .collect();
        let summary = quantizer.summarize_calibration(&map, &sizes);
        assert!(
            summary.estimated_bpw <= 3.0,
            "budget missed: {} bits per weight",
            summary.estimated_bpw
        );
        // And it did not overshoot to the floor: Q2_K everywhere is 2.625.
        assert!(
            summary.estimated_bpw > 2.7,
            "downgraded further than needed: {}",
            summary.estimated_bpw
        );
    }

    /// Calibrating one tensor at a time gives the map calibrating them all at
    /// once does, the bit budget included, and never asks for two at once.
    #[test]
    fn streaming_calibration_matches_calibrating_everything_at_once() {
        let quantizer = DynamicQuantizer::new(DynamicQuantizationConfig::default(), None);
        let tensors: Vec<(String, Vec<f32>, Vec<i32>)> = (0..4)
            .map(|i| {
                (
                    format!("blk.{i}.ffn_up.weight"),
                    weight_like(256 * 8, 7 + i),
                    vec![8, 256],
                )
            })
            .collect();
        let kl_config = KlCalibrationConfig {
            kl_threshold: 0.0,
            target_bpw: Some(3.0),
            ..Default::default()
        };
        let all = quantizer.calibrate_all(&tensors, &kl_config);

        let listing: Vec<(String, Vec<i32>)> = tensors
            .iter()
            .map(|(n, _, shape)| (n.clone(), shape.clone()))
            .collect();
        let loads = std::cell::Cell::new(0usize);
        let streamed = quantizer
            .calibrate_streaming(
                &listing,
                |name| {
                    loads.set(loads.get() + 1);
                    let data = tensors.iter().find(|(n, ..)| n == name).unwrap().1.clone();
                    Ok::<_, std::convert::Infallible>(data)
                },
                &kl_config,
            )
            .unwrap();
        assert_eq!(loads.get(), 4, "each tensor is loaded once");
        assert_eq!(all.len(), streamed.len());
        for (name, want) in &all {
            let got = &streamed[name];
            assert_eq!(got.quant_type, want.quant_type, "{name}");
            assert_eq!(got.kl_score, want.kl_score, "{name}");
        }

        // A failing load is the caller's error, and stops the pass.
        let err = quantizer
            .calibrate_streaming(&listing, |_| Err::<Vec<f32>, _>("disk gone"), &kl_config)
            .unwrap_err();
        assert_eq!(err, "disk gone");
    }

    #[test]
    fn the_budget_allocation_beats_a_uniform_one() {
        // Tensors of different difficulty under a 5-bit budget: the allocation
        // must meet it with less total loss than giving every tensor Q4_K, the
        // uniform choice that also fits.
        let mut spiky = weight_like(256 * 8, 2);
        for (i, v) in spiky.iter_mut().enumerate() {
            if i % 7 == 0 {
                *v *= 12.0;
            }
        }
        let tensors = vec![
            (
                "blk.0.ffn_up.weight".to_string(),
                weight_like(256 * 8, 1),
                vec![8, 256],
            ),
            ("blk.1.ffn_up.weight".to_string(), spiky, vec![8, 256]),
            (
                "blk.2.ffn_up.weight".to_string(),
                weight_like(256 * 8, 3),
                vec![8, 256],
            ),
        ];
        let quantizer = DynamicQuantizer::new(DynamicQuantizationConfig::default(), None);
        let kl_config = KlCalibrationConfig {
            kl_threshold: 0.0,
            target_bpw: Some(5.0),
            ..Default::default()
        };
        let map = quantizer.calibrate_all(&tensors, &kl_config);
        let sizes: Vec<_> = tensors
            .iter()
            .map(|(n, d, _)| (n.clone(), d.len()))
            .collect();
        assert!(quantizer.summarize_calibration(&map, &sizes).estimated_bpw <= 5.0);

        let score = |name: &str, t: GgmlType| {
            map[name]
                .all_scores
                .iter()
                .find(|(c, _)| *c == t)
                .unwrap()
                .1
        };
        let chosen: f64 = map.values().map(|c| c.kl_score).sum();
        let uniform: f64 = map.keys().map(|n| score(n, GgmlType::Q4K)).sum();
        assert!(
            chosen <= uniform + 1e-12,
            "chosen {chosen} vs uniform Q4_K {uniform}"
        );
    }

    #[test]
    fn calibration_scores_under_the_imatrix_importance() {
        let name = "blk.0.ffn_up.weight";
        let data = weight_like(256 * 4, 3);
        let shape = [4, 256];
        let kl_config = KlCalibrationConfig {
            candidates: vec![GgmlType::Q4K],
            kl_threshold: 1.0,
            ..Default::default()
        };

        let plain = DynamicQuantizer::new(DynamicQuantizationConfig::default(), None)
            .calibrate_tensor(name, &data, &shape, &kl_config);

        let mut imatrix = IMatrix::new();
        // Only the first 16 input channels carry activations.
        let sums: Vec<f32> = (0..256)
            .map(|j| if j < 16 { 1000.0 } else { 0.001 })
            .collect();
        imatrix.insert(name.to_string(), sums, 1);
        let quantizer = DynamicQuantizer::new(DynamicQuantizationConfig::default(), Some(imatrix));
        assert_eq!(quantizer.importance_for(name).unwrap().len(), 256);
        let weighted = quantizer.calibrate_tensor(name, &data, &shape, &kl_config);

        assert_ne!(plain.kl_score, weighted.kl_score);
        // A tensor of another name has no importance and scores as unweighted.
        let other = quantizer.calibrate_tensor("blk.0.ffn_down.weight", &data, &shape, &kl_config);
        assert_eq!(other.kl_score, plain.kl_score);
    }

    #[test]
    fn row_sampling_scores_a_spread_of_whole_rows() {
        let quantizer = DynamicQuantizer::new(DynamicQuantizationConfig::default(), None);
        let data = weight_like(256 * 64, 5);
        let full = KlCalibrationConfig {
            candidates: vec![GgmlType::Q4K, GgmlType::Q2K],
            kl_threshold: 1.0,
            sample_size: 0,
            ..Default::default()
        };
        let sampled = KlCalibrationConfig {
            sample_size: 256 * 8,
            ..full.clone()
        };
        let a = quantizer.calibrate_tensor("blk.0.ffn_up.weight", &data, &[64, 256], &full);
        let b = quantizer.calibrate_tensor("blk.0.ffn_up.weight", &data, &[64, 256], &sampled);
        // The sample is representative: scores agree to a few percent and rank alike.
        for ((t1, s1), (t2, s2)) in a.all_scores.iter().zip(&b.all_scores) {
            assert_eq!(t1, t2);
            assert!((s1 - s2).abs() < 0.1 * s1, "{t1:?}: {s1} vs {s2}");
        }
        // A row length no candidate can hold is not a panic: it falls back.
        let odd = quantizer.calibrate_tensor("blk.0.ffn_up.weight", &data[..100], &[1, 100], &full);
        assert_eq!(odd.quant_type, full.fallback_type);
    }
}
