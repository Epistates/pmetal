//! Validation of the configuration types.
//!
//! A config the trainers cannot run, or would run as something other than
//! what it says, is refused here with every problem named, instead of being
//! clamped, defaulted or ignored somewhere downstream. Each message leads
//! with the dotted key (`training.learning_rate`) so it points into the file
//! the user wrote.

use crate::{
    ConfigValidator, DatasetConfig, DistributedCompression, DistributedTrainingConfig,
    FullTrainingConfig, LoraBias, LoraConfig, LrSchedulerType, ModelConfig, PMetalError, Result,
    TrainingConfig,
};

/// Collects the problems found in one config, so a user fixes them in one pass.
#[derive(Default)]
pub(crate) struct Problems(Vec<String>);

impl Problems {
    fn add(&mut self, key: &str, message: impl std::fmt::Display) {
        self.0.push(format!("{key}: {message}"));
    }

    /// `key` must be a finite number above `min` (or at it, when `inclusive`).
    fn finite_at_least(&mut self, key: &str, value: f64, min: f64, inclusive: bool) {
        let ok = value.is_finite() && (value > min || (inclusive && value == min));
        if !ok {
            let bound = if inclusive { ">=" } else { ">" };
            self.add(
                key,
                format!("must be a finite number {bound} {min}, got {value}"),
            );
        }
    }

    pub(crate) fn extend(&mut self, other: Self) {
        self.0.extend(other.0);
    }

    pub(crate) fn finish(self, what: &str) -> Result<()> {
        if self.0.is_empty() {
            return Ok(());
        }
        let list = self
            .0
            .iter()
            .map(|p| format!("  - {p}"))
            .collect::<Vec<_>>()
            .join("\n");
        Err(PMetalError::Config(format!("invalid {what}:\n{list}")))
    }
}

impl TrainingConfig {
    pub(crate) fn problems(&self, section: &str) -> Problems {
        let mut p = Problems::default();
        let key = |name: &str| format!("{section}.{name}");

        p.finite_at_least(&key("learning_rate"), self.learning_rate, 0.0, false);
        if let Some(lr) = self.embedding_learning_rate {
            p.finite_at_least(&key("embedding_learning_rate"), lr, 0.0, false);
        }
        for (name, value) in [
            ("batch_size", self.batch_size),
            (
                "gradient_accumulation_steps",
                self.gradient_accumulation_steps,
            ),
            ("num_epochs", self.num_epochs),
            ("logging_steps", self.logging_steps),
        ] {
            if value == 0 {
                p.add(&key(name), "must be at least 1");
            }
        }
        for (name, value) in [
            ("max_steps", self.max_steps),
            ("eval_steps", self.eval_steps),
            ("save_steps", self.save_steps),
        ] {
            if value == Some(0) {
                p.add(
                    &key(name),
                    "must be at least 1 when set (omit it to turn it off)",
                );
            }
        }
        p.finite_at_least(&key("weight_decay"), self.weight_decay, 0.0, true);
        p.finite_at_least(&key("max_grad_norm"), self.max_grad_norm, 0.0, true);

        match self.warmup_ratio {
            Some(ratio) if !(ratio.is_finite() && (0.0..1.0).contains(&ratio)) => p.add(
                &key("warmup_ratio"),
                format!("must be in [0, 1), got {ratio}"),
            ),
            Some(_) if self.warmup_steps > 0 => p.add(
                &key("warmup_steps"),
                "set either warmup_steps or warmup_ratio, not both (the ratio would win)",
            ),
            _ => {}
        }

        if let Some(min_lr) = self.min_lr {
            if !(min_lr.is_finite() && min_lr >= 0.0) {
                p.add(
                    &key("min_lr"),
                    format!("must be a finite number >= 0, got {min_lr}"),
                );
            } else if min_lr > self.learning_rate {
                p.add(
                    &key("min_lr"),
                    format!(
                        "the floor ({min_lr}) is above the learning rate ({}); the schedule \
                         would climb instead of decay",
                        self.learning_rate
                    ),
                );
            }
        }

        // A schedule's own knobs mean nothing under another schedule: a
        // config that sets one almost certainly forgot to pick the schedule.
        let scheduler = self.lr_scheduler;
        let only_for = |p: &mut Problems, name: &str, set: bool, want: LrSchedulerType| {
            if set && scheduler != want {
                p.add(
                    &key(name),
                    format!(
                        "only applies to lr_scheduler: {want}, but the schedule is {scheduler}"
                    ),
                );
            }
        };
        only_for(
            &mut p,
            "wsd_stable_ratio",
            self.wsd_stable_ratio.is_some(),
            LrSchedulerType::Wsd,
        );
        only_for(
            &mut p,
            "wsd_decay_style",
            self.wsd_decay_style.is_some(),
            LrSchedulerType::Wsd,
        );
        only_for(
            &mut p,
            "cosine_num_restarts",
            self.cosine_num_restarts.is_some(),
            LrSchedulerType::CosineWithRestarts,
        );
        only_for(
            &mut p,
            "polynomial_power",
            self.polynomial_power.is_some(),
            LrSchedulerType::Polynomial,
        );
        if let Some(ratio) = self.wsd_stable_ratio
            && !(ratio.is_finite() && (0.0..=1.0).contains(&ratio))
        {
            p.add(
                &key("wsd_stable_ratio"),
                format!("must be in [0, 1], got {ratio}"),
            );
        }
        if self.cosine_num_restarts == Some(0) {
            p.add(&key("cosine_num_restarts"), "must be at least 1 cycle");
        }
        if let Some(power) = self.polynomial_power {
            p.finite_at_least(&key("polynomial_power"), power, 0.0, false);
        }

        if !(self.label_smoothing.is_finite() && (0.0..1.0).contains(&self.label_smoothing)) {
            p.add(
                &key("label_smoothing"),
                format!("must be in [0, 1), got {}", self.label_smoothing),
            );
        }
        p.finite_at_least(&key("z_loss"), self.z_loss, 0.0, true);
        if let Some(alpha) = self.neftune_noise_alpha {
            p.finite_at_least(&key("neftune_noise_alpha"), alpha, 0.0, false);
        }
        if self.output_dir.trim().is_empty() {
            p.add(&key("output_dir"), "must not be empty");
        }
        p
    }
}

impl ConfigValidator for TrainingConfig {
    fn validate(&self) -> Result<()> {
        self.problems("training").finish("training config")
    }
}

impl LoraConfig {
    pub(crate) fn problems(&self, section: &str) -> Problems {
        let mut p = Problems::default();
        let key = |name: &str| format!("{section}.{name}");

        if self.r == 0 {
            p.add(&key("r"), "the rank must be at least 1");
        }
        p.finite_at_least(&key("alpha"), f64::from(self.alpha), 0.0, false);
        if !(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout)) {
            p.add(
                &key("dropout"),
                format!("must be in [0, 1), got {}", self.dropout),
            );
        }
        if let Some(ratio) = self.loraplus_lr_ratio {
            p.finite_at_least(&key("loraplus_lr_ratio"), f64::from(ratio), 0.0, false);
        }
        let mut seen = std::collections::HashSet::new();
        for module in &self.target_modules {
            if module.trim().is_empty() {
                p.add(&key("target_modules"), "contains an empty module name");
            } else if !seen.insert(module.as_str()) {
                p.add(
                    &key("target_modules"),
                    format!("names '{module}' more than once"),
                );
            }
        }
        if self.target_modules.len() > 1
            && self
                .target_modules
                .iter()
                .any(|m| m == LoraConfig::ALL_LINEAR)
        {
            p.add(
                &key("target_modules"),
                "`all-linear` stands for every linear layer and can't be combined with names",
            );
        }
        if self.bias != LoraBias::None {
            p.add(
                &key("bias"),
                "training LoRA bias terms is not implemented; use `none`",
            );
        }
        p
    }
}

impl ConfigValidator for LoraConfig {
    fn validate(&self) -> Result<()> {
        self.problems("lora").finish("LoRA config")
    }
}

impl ConfigValidator for ModelConfig {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConfigValidator for DatasetConfig {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConfigValidator for DistributedTrainingConfig {
    fn validate(&self) -> Result<()> {
        let mut p = Problems::default();
        if self.gradient_port == 0 {
            p.add("distributed.gradient_port", "must be a real port, not 0");
        }
        if self.compression == DistributedCompression::TopK
            && !(self.topk_ratio.is_finite() && self.topk_ratio > 0.0 && self.topk_ratio <= 1.0)
        {
            p.add(
                "distributed.topk_ratio",
                format!("must be in (0, 1], got {}", self.topk_ratio),
            );
        }
        if self.peers.iter().any(|peer| peer.trim().is_empty()) {
            p.add("distributed.peers", "contains an empty address");
        }
        p.finish("distributed config")
    }
}

impl ConfigValidator for FullTrainingConfig {
    fn validate(&self) -> Result<()> {
        let mut p = self.training.problems("training");
        p.extend(self.lora.problems("lora"));
        p.finish("training config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerType;

    fn message(result: Result<()>) -> String {
        match result {
            Err(PMetalError::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn the_defaults_are_valid() {
        TrainingConfig::default().validate().unwrap();
        LoraConfig::default().validate().unwrap();
        FullTrainingConfig::default().validate().unwrap();
        DistributedTrainingConfig::default().validate().unwrap();
    }

    #[test]
    fn every_problem_is_reported_with_its_key() {
        let config = TrainingConfig {
            learning_rate: f64::NAN,
            batch_size: 0,
            max_steps: Some(0),
            weight_decay: -0.1,
            ..Default::default()
        };
        let floor_above_peak = TrainingConfig {
            min_lr: Some(1.0),
            ..Default::default()
        };
        assert!(message(floor_above_peak.validate()).contains("training.min_lr"));
        let text = message(config.validate());
        for key in [
            "training.learning_rate",
            "training.batch_size",
            "training.max_steps",
            "training.weight_decay",
        ] {
            assert!(text.contains(key), "{key} missing from:\n{text}");
        }
    }

    #[test]
    fn schedule_knobs_must_match_the_schedule() {
        let wrong = TrainingConfig {
            lr_scheduler: LrSchedulerType::Cosine,
            wsd_stable_ratio: Some(0.8),
            ..Default::default()
        };
        assert!(message(wrong.validate()).contains("only applies to lr_scheduler: wsd"));

        let right = TrainingConfig {
            lr_scheduler: LrSchedulerType::Wsd,
            wsd_stable_ratio: Some(0.8),
            ..Default::default()
        };
        right.validate().unwrap();

        for bad in [-0.1, 1.5, f64::NAN] {
            let config = TrainingConfig {
                lr_scheduler: LrSchedulerType::Wsd,
                wsd_stable_ratio: Some(bad),
                ..Default::default()
            };
            assert!(message(config.validate()).contains("wsd_stable_ratio"));
        }
        let config = TrainingConfig {
            lr_scheduler: LrSchedulerType::Polynomial,
            polynomial_power: Some(0.0),
            ..Default::default()
        };
        assert!(message(config.validate()).contains("polynomial_power"));
        let config = TrainingConfig {
            lr_scheduler: LrSchedulerType::CosineWithRestarts,
            cosine_num_restarts: Some(0),
            ..Default::default()
        };
        assert!(message(config.validate()).contains("cosine_num_restarts"));
    }

    #[test]
    fn warmup_steps_and_ratio_cannot_both_be_set() {
        let config = TrainingConfig {
            warmup_steps: 100,
            warmup_ratio: Some(0.05),
            ..Default::default()
        };
        assert!(message(config.validate()).contains("warmup_steps"));
        let config = TrainingConfig {
            warmup_ratio: Some(1.0),
            ..Default::default()
        };
        assert!(message(config.validate()).contains("warmup_ratio"));
    }

    #[test]
    fn every_optimizer_kind_passes_validation() {
        for kind in OptimizerType::ALL {
            let config = TrainingConfig {
                optimizer: kind,
                ..Default::default()
            };
            assert!(config.validate().is_ok(), "{kind}");
        }
    }

    #[test]
    fn unimplemented_lora_choices_are_refused() {
        let config = LoraConfig {
            bias: LoraBias::All,
            r: 0,
            dropout: 1.0,
            target_modules: vec!["q_proj".into(), "q_proj".into()],
            ..Default::default()
        };
        let text = message(config.validate());
        for fragment in ["bias", "lora.r", "dropout", "more than once"] {
            assert!(text.contains(fragment), "{fragment} missing from:\n{text}");
        }
    }

    #[test]
    fn distributed_top_k_needs_a_ratio() {
        let config = DistributedTrainingConfig {
            compression: DistributedCompression::TopK,
            topk_ratio: 0.0,
            ..Default::default()
        };
        assert!(message(config.validate()).contains("topk_ratio"));
    }
}
