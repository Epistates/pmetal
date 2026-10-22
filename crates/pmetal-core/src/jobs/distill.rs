//! `pmetal distill` — knowledge distillation (online / offline / progressive).

use super::{default_lora_alpha, default_lora_r, default_seed};
use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Distill", subcommand = "distill", events)]
#[serde(rename_all = "snake_case")]
pub struct DistillSpec {
    #[job(
        label = "Teacher",
        group = "Models",
        argv = "--teacher",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub teacher: String,

    #[job(
        label = "Student",
        group = "Models",
        argv = "--student",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub student: String,

    #[job(
        label = "Dataset",
        group = "Data",
        argv = "--dataset",
        kind = "dataset_picker",
        required
    )]
    #[serde(default)]
    pub dataset: String,

    #[job(
        label = "Eval Dataset",
        group = "Data",
        argv = "--eval-dataset",
        kind = "dataset_picker",
        help = "Evaluated every Checkpoint Every steps"
    )]
    #[serde(default)]
    pub eval_dataset: Option<String>,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./output/distilled"
    )]
    #[serde(default = "default_output")]
    pub output_dir: String,

    #[job(label = "Method", group = "Distillation", argv = "--method", kind = "enum",
          enum_options = ["online", "offline", "progressive"], default = "online")]
    #[serde(default = "default_method")]
    pub method: String,

    #[job(
        label = "Offline Shortcut",
        group = "Distillation",
        argv = "--offline",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub offline: bool,

    #[job(
        label = "Offline Cache",
        group = "Distillation",
        argv = "--offline-cache",
        kind = "path"
    )]
    #[serde(default)]
    pub offline_cache: Option<String>,

    #[job(
        label = "Generate Offline Logits",
        group = "Distillation",
        argv = "--offline-generate",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub offline_generate: bool,

    #[job(label = "Offline Compression", group = "Distillation", argv = "--offline-compression", kind = "enum",
          enum_options = ["top_k", "none", "int8", "int4", "sampled"], default = "top_k")]
    #[serde(default = "default_compression")]
    pub offline_compression: String,

    #[job(
        label = "Offline Top-k",
        group = "Distillation",
        argv = "--offline-top-k",
        min = 1,
        max = 65536,
        default_int = 128
    )]
    #[serde(default = "default_offline_top_k")]
    pub offline_top_k: usize,

    #[job(label = "Loss Type", group = "Distillation", argv = "--loss-type", kind = "enum",
          enum_options = ["kl_divergence", "reverse_kl", "jensen_shannon", "soft_cross_entropy", "mse_loss", "jsd_skewed", "skew_kl", "skew_reverse_kl", "universal_logit"],
          default = "kl_divergence",
          help = "jsd_skewed and skew_kl take --skew-alpha, universal_logit (different tokenizers) --universal-logit-top-k; for GKD, pair --on-policy with jsd_skewed")]
    #[serde(default = "default_loss_type")]
    pub loss_type: String,

    #[job(
        label = "Skew Alpha",
        group = "Distillation",
        argv = "--skew-alpha",
        min = 0.0,
        max = 1.0,
        help = "Mixing weight of jsd_skewed (default 0.5, the symmetric JSD) and of skew_kl / skew_reverse_kl (default 0.1)"
    )]
    #[serde(default)]
    pub skew_alpha: Option<f32>,

    #[job(
        label = "Universal Logit Top-k",
        group = "Distillation",
        argv = "--universal-logit-top-k",
        min = 1,
        max = 1048576,
        help = "Truncate to this many entries before the universal_logit loss sorts them (default: the whole vocabulary)"
    )]
    #[serde(default)]
    pub universal_logit_top_k: Option<usize>,

    #[job(
        label = "Temperature",
        group = "Distillation",
        argv = "--temperature",
        min = 0.1,
        max = 20.0,
        default_float = 2.0
    )]
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[job(
        label = "Alpha (hard/soft)",
        group = "Distillation",
        argv = "--alpha",
        min = 0.0,
        max = 1.0,
        default_float = 0.5
    )]
    #[serde(default = "default_alpha")]
    pub alpha: f32,

    #[job(
        label = "Rationale Distillation",
        group = "Distillation",
        argv = "--rationale",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub rationale: bool,

    #[job(
        label = "Rationale Weight",
        group = "Distillation",
        argv = "--rationale-weight",
        min = 0.0,
        max = 100.0,
        default_float = 1.0
    )]
    #[serde(default = "default_rationale_weight")]
    pub rationale_weight: f32,

    #[job(
        label = "On-policy Share",
        group = "Distillation",
        argv = "--on-policy",
        min = 0.0,
        max = 1.0,
        default_float = 0.0,
        help = "Share of batches whose responses the student samples itself and the teacher scores (GKD); online and progressive runs only"
    )]
    #[serde(default)]
    pub on_policy: f32,

    #[job(
        label = "On-policy Max Tokens",
        group = "Distillation",
        argv = "--on-policy-max-tokens",
        min = 1,
        max = 32768,
        default_int = 256
    )]
    #[serde(default = "default_on_policy_max_tokens")]
    pub on_policy_max_tokens: usize,

    #[job(
        label = "LoRA r",
        group = "LoRA",
        argv = "--lora-r",
        min = 1,
        max = 1024,
        default_int = 16
    )]
    #[serde(default = "default_lora_r")]
    pub lora_r: usize,

    #[job(
        label = "LoRA α",
        group = "LoRA",
        argv = "--lora-alpha",
        min = 1.0,
        max = 1024.0,
        default_float = 32.0
    )]
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-8,
        max = 1.0,
        default_float = 0.00002
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f32,

    #[job(
        label = "Optimizer",
        group = "Optimization",
        argv = "--optimizer",
        kind = "enum",
        enum_options = ["adamw", "sgd", "lion", "adafactor", "muon", "adamw_schedule_free", "adamw_8bit"],
        help = "Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW",
        default = "adamw"
    )]
    #[serde(default = "super::default_optimizer")]
    pub optimizer: String,

    #[job(
        label = "Batch Size",
        group = "Training",
        argv = "--batch-size",
        min = 1,
        max = 1024,
        default_int = 1
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[job(
        label = "Epochs",
        group = "Training",
        argv = "--epochs",
        min = 1,
        max = 1000,
        default_int = 1
    )]
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    #[job(
        label = "Max Seq Len",
        group = "Training",
        argv = "--max-seq-len",
        default_int = 1024
    )]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[job(
        label = "Checkpoint Every",
        group = "Training",
        argv = "--checkpoint-every",
        min = 0,
        max = 1000000,
        default_int = 100,
        help = "Save a checkpoint every this many steps (0 disables interval checkpoints)"
    )]
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: usize,

    #[job(label = "Text Column", group = "Data", argv = "--text-column")]
    #[serde(default)]
    pub text_column: Option<String>,

    #[job(label = "Text Columns (csv)", group = "Data", argv = "--text-columns")]
    #[serde(default)]
    pub text_columns: Option<String>,

    #[job(
        label = "Column Separator",
        group = "Data",
        argv = "--column-separator"
    )]
    #[serde(default)]
    pub column_separator: Option<String>,

    #[job(label = "Prompt Column", group = "Data", argv = "--prompt-column")]
    #[serde(default)]
    pub prompt_column: Option<String>,

    #[job(label = "Response Column", group = "Data", argv = "--response-column")]
    #[serde(default)]
    pub response_column: Option<String>,

    #[job(
        label = "Log Metrics Path",
        group = "Output",
        argv = "--log-metrics",
        kind = "path"
    )]
    #[serde(default)]
    pub log_metrics: Option<String>,

    /// Resume from the latest checkpoint in the output directory.
    #[job(
        label = "Resume",
        group = "Training",
        argv = "--resume",
        flag,
        default_bool = false,
        help = "Continue from the latest checkpoint in the output directory"
    )]
    #[serde(default)]
    pub resume: bool,
}

impl Default for DistillSpec {
    fn default() -> Self {
        Self {
            teacher: String::new(),
            student: String::new(),
            dataset: String::new(),
            eval_dataset: None,
            output_dir: default_output(),
            method: default_method(),
            offline: false,
            offline_cache: None,
            offline_generate: false,
            offline_compression: default_compression(),
            offline_top_k: default_offline_top_k(),
            loss_type: default_loss_type(),
            skew_alpha: None,
            universal_logit_top_k: None,
            temperature: default_temperature(),
            alpha: default_alpha(),
            rationale: false,
            rationale_weight: default_rationale_weight(),
            on_policy: 0.0,
            on_policy_max_tokens: default_on_policy_max_tokens(),
            lora_r: 16,
            lora_alpha: 32.0,
            learning_rate: default_lr(),
            optimizer: super::default_optimizer(),
            batch_size: 1,
            epochs: 1,
            max_seq_len: default_max_seq_len(),
            seed: 42,
            checkpoint_every: default_checkpoint_every(),
            text_column: None,
            text_columns: None,
            column_separator: None,
            prompt_column: None,
            response_column: None,
            log_metrics: None,
            resume: false,
        }
    }
}

impl DistillSpec {
    /// The method the run uses: `--offline` is a shortcut for `--method
    /// offline`.
    #[must_use]
    pub fn effective_method(&self) -> &str {
        if self.offline {
            "offline"
        } else {
            &self.method
        }
    }

    /// Run descriptor validation, then the rules that span fields: the
    /// offline shortcut against the method, the flags that only an offline
    /// run reads, and on-policy sampling, which needs the live teacher.
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        if self.offline && !matches!(self.method.as_str(), "online" | "offline") {
            errs.push(FieldError::new(
                "offline",
                format!("conflicts with --method {}", self.method),
            ));
        }
        let offline = self.effective_method() == "offline";
        if !offline && self.offline_cache.is_some() {
            errs.push(FieldError::new(
                "offline_cache",
                "needs --offline or --method offline",
            ));
        }
        if !offline && self.offline_generate {
            errs.push(FieldError::new(
                "offline_generate",
                "needs --offline or --method offline",
            ));
        }
        if offline && self.on_policy > 0.0 {
            errs.push(FieldError::new(
                "on_policy",
                "needs the live teacher to score the student's samples, so it cannot \
                 run with an offline cache",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    crate::defaults::DISTILL_OUTPUT_DIR.to_string()
}
fn default_method() -> String {
    "online".to_string()
}
fn default_compression() -> String {
    "top_k".to_string()
}
fn default_offline_top_k() -> usize {
    128
}
fn default_loss_type() -> String {
    "kl_divergence".to_string()
}
fn default_temperature() -> f32 {
    2.0
}
fn default_alpha() -> f32 {
    0.5
}
fn default_on_policy_max_tokens() -> usize {
    256
}
fn default_rationale_weight() -> f32 {
    1.0
}
fn default_lr() -> f32 {
    2e-5
}
fn default_max_seq_len() -> usize {
    1024
}
fn default_batch_size() -> usize {
    1
}
fn default_epochs() -> usize {
    crate::defaults::EPOCHS
}
fn default_checkpoint_every() -> usize {
    100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = DistillSpec {
            teacher: "t".into(),
            student: "s".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--teacher".to_string()));
        assert!(argv.contains(&"--student".to_string()));
        assert!(argv.contains(&"--dataset".to_string()));
    }

    fn spec() -> DistillSpec {
        DistillSpec {
            teacher: "t".into(),
            student: "s".into(),
            dataset: "d".into(),
            ..Default::default()
        }
    }

    fn rejected(mut spec: DistillSpec) -> Vec<String> {
        spec.normalize()
            .expect_err("the spec is invalid")
            .into_iter()
            .map(|e| format!("{}: {}", e.field, e.message))
            .collect()
    }

    #[test]
    fn the_offline_shortcut_is_the_offline_method() {
        let mut offline = DistillSpec {
            offline: true,
            ..spec()
        };
        assert!(offline.normalize().is_ok());
        assert_eq!(offline.effective_method(), "offline");
        assert_eq!(spec().effective_method(), "online");
        let conflict = rejected(DistillSpec {
            offline: true,
            method: "progressive".into(),
            ..spec()
        });
        assert!(
            conflict[0].contains("conflicts with --method progressive"),
            "{conflict:?}"
        );
    }

    #[test]
    fn offline_only_flags_need_an_offline_run() {
        let cache = rejected(DistillSpec {
            offline_cache: Some("c".into()),
            ..spec()
        });
        assert!(cache[0].starts_with("offline_cache"), "{cache:?}");
        let generate = rejected(DistillSpec {
            offline_generate: true,
            ..spec()
        });
        assert!(generate[0].starts_with("offline_generate"), "{generate:?}");
        let mut ok = DistillSpec {
            method: "offline".into(),
            offline_cache: Some("c".into()),
            offline_generate: true,
            ..spec()
        };
        assert!(ok.normalize().is_ok());
    }

    #[test]
    fn on_policy_sampling_cannot_run_offline() {
        let errors = rejected(DistillSpec {
            offline: true,
            on_policy: 0.5,
            ..spec()
        });
        assert!(errors[0].starts_with("on_policy"), "{errors:?}");
    }

    #[test]
    fn every_loss_type_the_distiller_has_is_offered() {
        let crate::FieldKind::Enum { options } = DistillSpec::field_descriptors()
            .iter()
            .find(|d| d.name == "loss_type")
            .expect("a loss type field")
            .kind
        else {
            panic!("loss_type is an enum");
        };
        for name in [
            "kl_divergence",
            "reverse_kl",
            "jensen_shannon",
            "soft_cross_entropy",
            "mse_loss",
            "jsd_skewed",
            "skew_kl",
            "skew_reverse_kl",
            "universal_logit",
        ] {
            assert!(options.contains(&name), "{name} is not offered");
        }
    }
}
