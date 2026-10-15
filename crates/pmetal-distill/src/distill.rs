//! Main distillation orchestration.
//!
//! This module provides the high-level API for running knowledge distillation,
//! including online, offline, and progressive distillation modes.

use pmetal_runtime::compat::{Array, Dtype, ops};
use pmetal_runtime::training::CrossEntropy;

use crate::{
    DistillConfig, DistillError, DistillMethod, LossType, Result, SparseLogits, SparseTargets,
    losses::{
        AttentionTransferLoss, DistillLoss, HiddenStateLoss, JensenShannonLoss, KlDivergenceLoss,
        MseLoss, SoftCrossEntropyLoss, SparseKlLoss,
        hidden_state::{LayerDistillation, mean_over_layer_pairs},
        reduce_per_token, tempered_log_probs,
    },
};

// End-to-end orchestration (model loading, dataset iteration, optimizer steps,
// checkpointing) lives in `pmetal-trainer`; this crate is the loss math and the
// per-batch `Distiller::compute_loss`.

/// Builder for creating a Distiller instance.
pub struct DistillerBuilder {
    config: Option<DistillConfig>,
    loss: Option<Box<dyn DistillLoss>>,
    hidden_loss: Option<HiddenStateLoss>,
}

impl DistillerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: None,
            loss: None,
            hidden_loss: None,
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: DistillConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set a custom loss function.
    pub fn with_loss(mut self, loss: Box<dyn DistillLoss>) -> Self {
        self.loss = Some(loss);
        self
    }

    /// Set hidden state loss.
    pub fn with_hidden_loss(mut self, loss: HiddenStateLoss) -> Self {
        self.hidden_loss = Some(loss);
        self
    }

    /// Build the Distiller.
    pub fn build(self) -> Result<Distiller> {
        let config = self
            .config
            .ok_or_else(|| DistillError::InvalidConfig("Configuration is required".to_string()))?;
        config.validate()?;

        // Create loss function from config if not provided
        let loss: Box<dyn DistillLoss> = self.loss.unwrap_or_else(|| {
            if config.loss.rationale {
                if config.loss.outcome_supervised {
                    Box::new(crate::reasoning::OutcomeSupervisedRationaleLoss::new(
                        config.loss.rationale_weight,
                    ))
                } else {
                    // Explicit reasoning markers (`start_marker`/`end_marker`)
                    // are token strings; turning them into a per-token mask
                    // needs the tokenizer, so the training loop does it
                    // (`reasoning::generate_reasoning_mask`) and passes the
                    // mask as the per-token weights.
                    Box::new(crate::reasoning::RationaleLoss::new(
                        config.loss.rationale_weight,
                    ))
                }
            } else {
                match config.loss.loss_type.clone() {
                    LossType::KlDivergence => {
                        if config.loss.reverse_kl {
                            Box::new(KlDivergenceLoss::reverse())
                        } else {
                            Box::new(KlDivergenceLoss::new())
                        }
                    }
                    LossType::JensenShannon => Box::new(JensenShannonLoss::new()),
                    LossType::SoftCrossEntropy => Box::new(SoftCrossEntropyLoss::new()),
                    LossType::MseLoss => Box::new(MseLoss::new()),
                    LossType::JsdSkewed { alpha } => {
                        Box::new(crate::losses::JsdSkewedLoss::new(alpha))
                    }
                    LossType::UniversalLogit { top_k } => {
                        let mut loss = crate::losses::UniversalLogitLoss::new();
                        if let Some(k) = top_k {
                            loss = loss.with_top_k(k);
                        }
                        Box::new(loss)
                    }
                    LossType::SkewKl { alpha, reverse } => Box::new(if reverse {
                        crate::losses::SkewKlLoss::reverse(alpha)
                    } else {
                        crate::losses::SkewKlLoss::forward(alpha)
                    }),
                    LossType::Gkd {
                        lambda,
                        sampler_temperature,
                    } => Box::new(crate::losses::GkdLoss::new(lambda, sampler_temperature)),
                }
            }
        });

        // The attention-transfer term, from `loss.attention`.
        let attention = config.loss.attention.as_ref().map(|a| AttentionTerm {
            loss: AttentionTransferLoss::new()
                .with_loss_type(a.loss_type.clone())
                .with_head_reduction(a.head_reduction.clone()),
            layer_mapping: a.layer_mapping.clone(),
            weight: a.weight,
        });

        // The hidden-state term: the config names the layer pairs and the
        // weight; a loss handed to the builder replaces the config's loss type.
        let layer_distill = match (&config.loss.hidden_state, self.hidden_loss) {
            (Some(hidden), injected) => Some(LayerDistillation::new(
                hidden.layer_mapping.clone(),
                injected.unwrap_or_else(|| HiddenStateLoss::new(hidden.loss_type.clone())),
                hidden.weight,
            )),
            (None, Some(_)) => {
                return Err(DistillError::InvalidConfig(
                    "a hidden-state loss needs `loss.hidden_state` to name the layer pairs to align"
                        .to_string(),
                ));
            }
            (None, None) => None,
        };

        Ok(Distiller {
            config,
            loss,
            layer_distill,
            attention,
        })
    }
}

impl Default for DistillerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Main distillation engine.
pub struct Distiller {
    /// Configuration.
    config: DistillConfig,
    /// Primary loss function.
    loss: Box<dyn DistillLoss>,
    /// Hidden-state alignment across the configured layer pairs, if any.
    layer_distill: Option<LayerDistillation>,
    /// Attention-map transfer across the configured layer pairs, if any.
    attention: Option<AttentionTerm>,
}

/// The attention-transfer term of a [`Distiller`], from `loss.attention`.
struct AttentionTerm {
    loss: AttentionTransferLoss,
    layer_mapping: Vec<(usize, usize)>,
    weight: f32,
}

impl Distiller {
    /// Create a new distiller with default settings.
    pub fn new(config: DistillConfig) -> Result<Self> {
        DistillerBuilder::new().with_config(config).build()
    }

    /// Compute distillation loss for a batch.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher model logits.
    /// * `student_logits` - Student model logits.
    /// * `labels` - Optional ground-truth labels for hard target loss.
    /// * `weights` - Optional per-token loss weights.
    /// * `step` - Current training step (used by the progressive schedule).
    /// * `total_steps` - Total training steps (used by the progressive schedule).
    ///
    /// When the distillation method is `Progressive`, `temperature` and `alpha`
    /// are derived dynamically via `progressive_params(step, total_steps)` instead
    /// of reading static config values, so the schedule actually takes effect.
    pub fn compute_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        labels: Option<&Array>,
        weights: Option<&Array>,
        step: usize,
        total_steps: usize,
    ) -> Result<DistillLossOutput> {
        check_logit_pair(teacher_logits, student_logits)?;
        let (temperature, alpha) = self.schedule(step, total_steps);

        let soft_loss =
            self.loss
                .compute_weighted(teacher_logits, student_logits, temperature, weights)?;
        self.combine(
            soft_loss,
            self.loss.temperature_scale(temperature),
            alpha,
            student_logits,
            labels,
        )
    }

    /// [`compute_loss`](Self::compute_loss) against a top-k teacher read from
    /// an offline cache ([`LogitCache::load_sequence_sparse`](crate::LogitCache::load_sequence_sparse),
    /// collated with [`SparseLogits::stack`](crate::SparseLogits::stack)),
    /// without ever forming the dense `[batch, seq, vocab]` teacher.
    ///
    /// The soft term is the forward KL
    /// ([`SparseKlLoss`](crate::losses::SparseKlLoss)), so this needs the
    /// configured loss to be the forward KL (or the soft cross-entropy, which
    /// has the same gradient); any other loss needs the dense teacher and is
    /// refused rather than silently swapped for another.
    pub fn compute_loss_sparse(
        &self,
        teacher: &SparseLogits,
        student_logits: &Array,
        labels: Option<&Array>,
        weights: Option<&Array>,
        step: usize,
        total_steps: usize,
    ) -> Result<DistillLossOutput> {
        self.require_forward_kl()?;
        let (temperature, alpha) = self.schedule(step, total_steps);
        let soft_loss =
            SparseKlLoss::new().compute(teacher, student_logits, temperature, weights)?;
        self.combine(
            soft_loss,
            temperature * temperature,
            alpha,
            student_logits,
            labels,
        )
    }

    /// [`compute_loss_sparse`](Self::compute_loss_sparse) against sampled
    /// draws ([`SparseTargets`], Random-Sampling KD).
    ///
    /// The draws estimate the teacher at the temperature they were taken at,
    /// so the configured temperature must be that one (an error otherwise;
    /// with the progressive method, whose temperature moves, it never is).
    pub fn compute_loss_sampled(
        &self,
        teacher: &SparseTargets,
        student_logits: &Array,
        labels: Option<&Array>,
        weights: Option<&Array>,
        step: usize,
        total_steps: usize,
    ) -> Result<DistillLossOutput> {
        self.require_forward_kl()?;
        let (temperature, alpha) = self.schedule(step, total_steps);
        let soft_loss =
            SparseKlLoss::new().compute_sampled(teacher, student_logits, temperature, weights)?;
        self.combine(
            soft_loss,
            temperature * temperature,
            alpha,
            student_logits,
            labels,
        )
    }

    /// A sparse teacher only has a forward-KL form.
    fn require_forward_kl(&self) -> Result<()> {
        let cfg = &self.config.loss;
        let forward_kl = !cfg.rationale
            && match cfg.loss_type {
                LossType::KlDivergence => !cfg.reverse_kl,
                LossType::SoftCrossEntropy => true,
                _ => false,
            };
        if forward_kl {
            Ok(())
        } else {
            Err(DistillError::InvalidConfig(
                "a sparse teacher supports the forward KL (loss_type kl_divergence without \
                 reverse_kl, or soft_cross_entropy); this configuration needs the dense teacher \
                 logits"
                    .to_string(),
            ))
        }
    }

    /// `(temperature, alpha)` at `step`: the static config, or the progressive
    /// schedule when the method is `Progressive`, so the schedule takes effect.
    fn schedule(&self, step: usize, total_steps: usize) -> (f32, f32) {
        if self.config.method == DistillMethod::Progressive {
            self.progressive_params(step, total_steps)
        } else {
            (self.config.loss.temperature, self.config.loss.alpha)
        }
    }

    /// Scale the soft loss and blend it with the hard-label loss.
    fn combine(
        &self,
        soft_loss: Array,
        temperature_scale: f32,
        alpha: f32,
        student_logits: &Array,
        labels: Option<&Array>,
    ) -> Result<DistillLossOutput> {
        // Scale by T² (Hinton et al. 2015, §2.1): when teacher/student logits
        // are softened by `T`, the gradient with respect to student logits is
        // attenuated by 1/T². Multiplying the loss by T² restores the
        // gradient magnitude so a single optimizer step has the same effect
        // regardless of the temperature setting. A loss that never softens
        // anything (logit MSE) opts out through `temperature_scale`.
        let soft_scaled = soft_loss.multiply(&Array::from_f32(temperature_scale));

        // Combined with hard labels if provided
        let (total_loss, hard_loss_opt) = if let Some(labels) = labels {
            let hard_loss =
                compute_hard_loss(student_logits, labels, self.config.training.ignore_index)?;

            // total = alpha * soft + (1 - alpha) * hard
            let soft_weighted = soft_scaled.multiply(&Array::from_f32(alpha));
            let hard_weighted = hard_loss.multiply(&Array::from_f32(1.0 - alpha));

            let total = soft_weighted.add(&hard_weighted);
            (total, Some(hard_loss))
        } else {
            (soft_scaled.clone(), None)
        };

        Ok(DistillLossOutput {
            total: total_loss,
            soft: soft_scaled,
            hard: hard_loss_opt,
            hidden: None,
            metrics: std::collections::HashMap::new(),
        })
    }

    /// Compute loss with hidden state alignment.
    #[expect(
        clippy::too_many_arguments,
        reason = "public API: the inputs of one loss step"
    )]
    pub fn compute_loss_with_hidden(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        teacher_hiddens: &[Array],
        student_hiddens: &[Array],
        labels: Option<&Array>,
        weights: Option<&Array>,
        step: usize,
        total_steps: usize,
    ) -> Result<DistillLossOutput> {
        let mut output = self.compute_loss(
            teacher_logits,
            student_logits,
            labels,
            weights,
            step,
            total_steps,
        )?;

        // Add the hidden-state term when one is configured. The token weights
        // that mask the logit loss mask the hidden states too.
        if let Some(layer_distill) = &self.layer_distill {
            let hidden =
                layer_distill.compute_with_mask(teacher_hiddens, student_hiddens, weights)?;
            output.total = output.total.add(&hidden);
            output.hidden = Some(hidden);
        }

        Ok(output)
    }

    /// The attention-transfer term configured by `loss.attention`: the mean
    /// over the mapped layer pairs of the chosen attention loss, times the
    /// configured weight. `None` when no attention transfer is configured.
    ///
    /// `teacher_attn` and `student_attn` are the post-softmax attention maps
    /// of every layer, `[batch, heads, q, k]` each. Add the result to the
    /// total of [`compute_loss`](Self::compute_loss) like the hidden term.
    pub fn compute_attention_loss(
        &self,
        teacher_attn: &[Array],
        student_attn: &[Array],
    ) -> Result<Option<Array>> {
        let Some(term) = &self.attention else {
            return Ok(None);
        };
        let weights = vec![1.0; term.layer_mapping.len()];
        mean_over_layer_pairs(
            &term.layer_mapping,
            &weights,
            term.weight,
            teacher_attn,
            student_attn,
            |teacher, student| term.loss.compute(teacher, student),
        )
        .map(Some)
    }

    /// The name of the soft loss in use (`"kl_divergence"`, `"skew_kl"`, ...).
    pub fn loss_name(&self) -> &'static str {
        self.loss.name()
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistillConfig {
        &self.config
    }

    /// Get progressive schedule parameters for a given step.
    pub fn progressive_params(&self, step: usize, total_steps: usize) -> (f32, f32) {
        let progress = (step as f32) / (total_steps.max(1) as f32);

        // Linear decay from initial to final
        let initial_temp = self.config.loss.temperature;
        let final_temp = 1.0;
        let temperature = initial_temp + (final_temp - initial_temp) * progress;

        let initial_alpha = self.config.loss.alpha;
        let final_alpha = 0.0;
        let alpha = initial_alpha + (final_alpha - initial_alpha) * progress;

        (temperature, alpha)
    }
}

/// Output from distillation loss computation.
///
/// `total` / `soft` / `hard` / `hidden` are MLX arrays that stay lazy until a
/// caller materializes them with `.item()`. The `metrics` map carries opt-in,
/// already-evaluated f32 scalars suitable for JSONL logging or TUI streaming;
/// callers that don't want the eval cost simply leave it empty (the default).
#[derive(Debug)]
pub struct DistillLossOutput {
    /// Total combined loss.
    pub total: Array,
    /// Soft target loss component.
    pub soft: Array,
    /// Hard target loss component (if labels provided).
    pub hard: Option<Array>,
    /// Hidden state loss component.
    pub hidden: Option<Array>,
    /// Auxiliary scalar metrics, populated on demand via
    /// [`DistillLossOutput::with_metrics`]. Keys come from a small fixed
    /// vocabulary ("teacher_entropy", "student_entropy", "top1_agreement",
    /// "kl_per_token") so consumers can pattern-match without parsing.
    pub metrics: std::collections::HashMap<&'static str, f32>,
}

impl DistillLossOutput {
    /// Compute and attach observability scalars from the teacher/student
    /// logit pair used for this loss. This forces an eval on small reductions
    /// — only call it when you actually need to emit telemetry (e.g. once per
    /// log step, not on every micro-batch).
    ///
    /// Populated keys:
    /// - `teacher_entropy`: mean H(softmax(teacher / T)) in nats.
    /// - `student_entropy`: mean H(softmax(student / T)) in nats.
    /// - `top1_agreement`: fraction of tokens where argmax agrees.
    /// - `kl_per_token`: mean fwd-KL(teacher || student) in nats.
    pub fn with_metrics(
        self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Self {
        self.with_weighted_metrics(teacher_logits, student_logits, temperature, None)
    }

    /// [`with_metrics`](Self::with_metrics) averaged over the tokens the
    /// per-token `weights` (`[batch, seq]`, e.g. the loss mask) keep, so
    /// padding and prompt positions do not pollute the telemetry.
    ///
    /// The teacher and the student must share a vocabulary. All four values
    /// come back in one device-to-host transfer.
    pub fn with_weighted_metrics(
        mut self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Self {
        let teacher_logits = teacher_logits.stop_gradient();
        let student_logits = student_logits.stop_gradient();
        let t_log_p = tempered_log_probs(&teacher_logits, temperature);
        let s_log_p = tempered_log_probs(&student_logits, temperature);
        let t_p = t_log_p.exp();
        let s_p = s_log_p.exp();

        // Per-token entropies and forward KL(teacher || student).
        let h_teacher = t_p.multiply(&t_log_p).sum_axis(-1, false).negative();
        let h_student = s_p.multiply(&s_log_p).sum_axis(-1, false).negative();
        let kl = t_p
            .multiply(&t_log_p.subtract(&s_log_p))
            .sum_axis(-1, false);

        // top-1 agreement: whether argmax(teacher) == argmax(student).
        let agree = teacher_logits
            .argmax(-1)
            .equal(&student_logits.argmax(-1))
            .as_dtype(Dtype::Float32.as_i32());

        let mean = |per_token: &Array| -> Array {
            // A weight tensor of the wrong size is a caller bug; telemetry
            // reports it as NaN rather than panicking mid-training.
            reduce_per_token(per_token, weights)
                .unwrap_or_else(|_| Array::from_f32(f32::NAN))
                .reshape(&[1])
        };
        let packed = ops::concatenate_axis(
            &[
                &mean(&h_teacher),
                &mean(&h_student),
                &mean(&kl),
                &mean(&agree),
            ],
            0,
        );
        let [h_t, h_s, kl_mean, agree_mean] = packed
            .clone()
            .to_f32_vec(4)
            .and_then(|v| <[f32; 4]>::try_from(v).ok())
            .unwrap_or([f32::NAN; 4]);

        self.metrics.insert("teacher_entropy", h_t);
        self.metrics.insert("student_entropy", h_s);
        self.metrics.insert("kl_per_token", kl_mean);
        self.metrics.insert("top1_agreement", agree_mean);
        self
    }
}

/// Compute hard cross-entropy loss with labels, masking positions whose label
/// equals `ignore_index` (PyTorch convention: `-100`).
///
/// Teacher and student logits must agree on every axis but the vocabulary
/// (a differing vocabulary is a cross-vocabulary run, which the losses align).
///
/// An exception inside the runtime does not abort: it leaves a placeholder
/// array behind, so a mismatch has to be caught before the first op.
fn check_logit_pair(teacher: &Array, student: &Array) -> Result<()> {
    let (t, s) = (teacher.shape(), student.shape());
    if t.len() < 2 || t.len() != s.len() || t[..t.len() - 1] != s[..s.len() - 1] {
        return Err(DistillError::ShapeMismatch {
            teacher: t.to_vec(),
            student: s.to_vec(),
        });
    }
    Ok(())
}

/// The runtime's fused cross-entropy evaluates in f32 whatever the logits'
/// dtype (a bf16 normalizer over a large vocabulary is off by tenths of a
/// nat), keeps the labels out of the gradient trace, and ignores the masked
/// positions in both the sum and the divisor, as
/// `torch.nn.functional.cross_entropy(reduction='mean')` does. An all-ignored
/// batch gives zero.
fn compute_hard_loss(logits: &Array, labels: &Array, ignore_index: i32) -> Result<Array> {
    Ok(CrossEntropy::new()
        .with_ignore_index(ignore_index)
        .mean(logits, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn test_config() -> DistillConfig {
        DistillConfig {
            teacher: "teacher-model".to_string(),
            student: "student-model".to_string(),
            method: DistillMethod::Online,
            loss: crate::LossConfig::default(),
            offline: None,
            output_path: None,
            training: crate::TrainingConfig::default(),
        }
    }

    #[test]
    #[serial]
    fn test_distiller_builder() {
        let config = test_config();
        let distiller = DistillerBuilder::new().with_config(config).build().unwrap();

        assert_eq!(distiller.config().teacher, "teacher-model");
        assert_eq!(distiller.loss.name(), "kl_divergence");
    }

    #[test]
    fn test_distiller_with_custom_loss() {
        let config = test_config();
        let distiller = DistillerBuilder::new()
            .with_config(config)
            .with_loss(Box::new(JensenShannonLoss::new()))
            .build()
            .unwrap();

        assert_eq!(distiller.loss.name(), "jensen_shannon");
    }

    #[test]
    #[serial]
    fn test_compute_loss() {
        let config = test_config();
        let distiller = Distiller::new(config).unwrap();

        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let output = distiller
            .compute_loss(&teacher, &student, None, None, 0, 1000)
            .unwrap();

        // Loss should be positive
        let value: f32 = output.total.item();
        assert!(value > 0.0);
    }

    #[test]
    fn test_progressive_params() {
        let config = test_config();
        let distiller = Distiller::new(config).unwrap();

        // At start
        let (temp, alpha) = distiller.progressive_params(0, 1000);
        assert!((temp - 2.0).abs() < 1e-5); // initial temperature
        assert!((alpha - 0.5).abs() < 1e-5); // initial alpha

        // At end
        let (temp, alpha) = distiller.progressive_params(1000, 1000);
        assert!((temp - 1.0).abs() < 1e-5); // final temperature
        assert!(alpha.abs() < 1e-5); // final alpha = 0

        // At midpoint
        let (temp, alpha) = distiller.progressive_params(500, 1000);
        assert!((temp - 1.5).abs() < 1e-5);
        assert!((alpha - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_loss_types_from_config() {
        let mut config = test_config();

        // Test each loss type
        config.loss.loss_type = LossType::KlDivergence;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "kl_divergence");

        config.loss.loss_type = LossType::JensenShannon;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "jensen_shannon");

        config.loss.loss_type = LossType::SoftCrossEntropy;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "soft_cross_entropy");

        config.loss.loss_type = LossType::MseLoss;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "mse");
    }

    /// `compute_hard_loss` must obey `TrainingConfig.ignore_index`. The two
    /// canonical conventions in the wild are PyTorch's `-100` (default) and
    /// `255` from some HF segmentation pipelines. With one valid token at
    /// position 0 and the rest set to the chosen ignore value, the masked
    /// loss must equal the loss of the single remaining token — i.e. the
    /// ignored positions are excluded from both numerator and divisor.
    #[test]
    #[serial]
    fn hard_loss_respects_ignore_index_minus100() {
        let mut config = test_config();
        config.training.ignore_index = -100;
        let distiller = Distiller::new(config).unwrap();

        // [batch=1, seq=4, vocab=3]
        let logits = Array::from_f32_slice(
            &[
                0.0_f32, 1.0, 2.0, // token 0
                0.0, 1.0, 2.0, // token 1 (ignored)
                0.0, 1.0, 2.0, // token 2 (ignored)
                0.0, 1.0, 2.0, // token 3 (ignored)
            ],
            &[1, 4, 3],
        );
        let labels = Array::from_i32_slice_shaped(&[2, -100, -100, -100], &[1, 4]);
        let teacher = Array::from_f32_slice(
            &[
                0.0_f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
            ],
            &[1, 4, 3],
        );

        let output = distiller
            .compute_loss(&teacher, &logits, Some(&labels), None, 0, 1000)
            .unwrap();
        let hard = output.hard.expect("hard loss should be present");
        let value: f32 = hard.item();

        // Reference: −log_softmax(logits[2])[2] for a single token. logits=[0,1,2]
        // → softmax denom = e^0 + e^1 + e^2 ≈ 11.1073, log_softmax[2] ≈ 2 - log(11.1073) ≈ -0.4076
        // → hard loss ≈ 0.4076.
        let expected = 0.40760598_f32;
        assert!(
            (value - expected).abs() < 1e-3,
            "expected ~{}, got {}",
            expected,
            value
        );
    }

    #[test]
    #[serial]
    fn hard_loss_respects_ignore_index_255() {
        let mut config = test_config();
        config.training.ignore_index = 255;
        let distiller = Distiller::new(config).unwrap();

        let logits = Array::from_f32_slice(
            &[
                0.0_f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
            ],
            &[1, 4, 3],
        );
        // Same setup but with 255 as the ignore marker.
        let labels = Array::from_i32_slice_shaped(&[2, 255, 255, 255], &[1, 4]);
        let teacher = Array::from_f32_slice(
            &[
                0.0_f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
            ],
            &[1, 4, 3],
        );

        let output = distiller
            .compute_loss(&teacher, &logits, Some(&labels), None, 0, 1000)
            .unwrap();
        let value: f32 = output.hard.unwrap().item();
        let expected = 0.40760598_f32;
        assert!(
            (value - expected).abs() < 1e-3,
            "expected ~{}, got {}",
            expected,
            value
        );
    }

    /// `with_metrics` populates the four telemetry scalars. For identical
    /// teacher/student logits we expect: KL ≈ 0, top-1 agreement = 1.0, and
    /// teacher_entropy == student_entropy.
    #[test]
    #[serial]
    fn with_metrics_populates_all_keys() {
        let config = test_config();
        let distiller = Distiller::new(config).unwrap();

        let teacher =
            Array::from_f32_slice(&[1.0_f32, 2.0, 0.5, 4.0, 1.0, 2.0, 0.5, 4.0], &[1, 2, 4]);
        let student = teacher.clone();

        let out = distiller
            .compute_loss(&teacher, &student, None, None, 0, 1)
            .unwrap()
            .with_metrics(&teacher, &student, 1.0);

        for key in [
            "teacher_entropy",
            "student_entropy",
            "kl_per_token",
            "top1_agreement",
        ] {
            assert!(out.metrics.contains_key(key), "missing metric {}", key);
        }
        let kl = out.metrics["kl_per_token"];
        let agree = out.metrics["top1_agreement"];
        let h_t = out.metrics["teacher_entropy"];
        let h_s = out.metrics["student_entropy"];
        assert!(
            kl.abs() < 1e-4,
            "kl should be ~0 for identical inputs, got {}",
            kl
        );
        assert!(
            (agree - 1.0).abs() < 1e-6,
            "agree should be 1.0, got {}",
            agree
        );
        assert!(
            (h_t - h_s).abs() < 1e-4,
            "entropies should match, got {} vs {}",
            h_t,
            h_s
        );
        assert!(h_t.is_finite() && h_t > 0.0);
    }

    /// All-ignored labels must yield a finite, non-negative loss (zero) — no
    /// division-by-zero, no NaN. The implementation guards the divisor with
    /// `max(num_valid, 1.0)`.
    #[test]
    #[serial]
    fn hard_loss_all_ignored_is_zero_finite() {
        let mut config = test_config();
        config.training.ignore_index = -100;
        let distiller = Distiller::new(config).unwrap();

        let logits = Array::from_f32_slice(
            &[
                0.0_f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
            ],
            &[1, 4, 3],
        );
        let labels = Array::from_i32_slice_shaped(&[-100, -100, -100, -100], &[1, 4]);
        let teacher = Array::from_f32_slice(
            &[
                0.0_f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
            ],
            &[1, 4, 3],
        );

        let output = distiller
            .compute_loss(&teacher, &logits, Some(&labels), None, 0, 1000)
            .unwrap();
        let value: f32 = output.hard.unwrap().item();
        assert!(value.is_finite(), "loss must be finite");
        assert!(value.abs() < 1e-6, "all-ignored loss must be zero");
    }
}
