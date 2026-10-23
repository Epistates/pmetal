//! `pmetal preference`: offline preference optimization with LoRA.

use std::path::PathBuf;

use pmetal_core::jobs::PreferenceSpec;
use pmetal_core::{LoraConfig, TrainingConfig};
use pmetal_data::Tokenizer;

use super::RunContext;
use pmetal_lora::DynamicLoraModel;
use pmetal_trainer::preference::{self, KtoConfig, PreferenceLoss, PreferenceStepMetrics};
use pmetal_trainer::{KtoTrainer, PreferenceTrainer, TrainOptimizer};

/// The objective a spec names.
enum Objective {
    Paired(PreferenceLoss),
    Kto(KtoConfig),
}

fn objective(spec: &PreferenceSpec) -> anyhow::Result<Objective> {
    let beta = spec.effective_beta();
    Ok(match spec.loss.as_str() {
        "dpo" => Objective::Paired(PreferenceLoss::Dpo {
            beta,
            label_smoothing: spec.label_smoothing,
        }),
        "ipo" => Objective::Paired(PreferenceLoss::Ipo { beta }),
        "hinge" => Objective::Paired(PreferenceLoss::Hinge { beta }),
        "simpo" => Objective::Paired(PreferenceLoss::Simpo {
            beta,
            gamma_beta_ratio: spec.simpo_gamma_ratio,
        }),
        "orpo" => Objective::Paired(PreferenceLoss::Orpo { beta }),
        "kto" => Objective::Kto(KtoConfig {
            beta,
            desirable_weight: spec.desirable_weight,
            undesirable_weight: spec.undesirable_weight,
        }),
        other => anyhow::bail!("unknown --loss '{other}'"),
    })
}

/// The trainer settings a spec describes.
fn training_config(spec: &PreferenceSpec) -> anyhow::Result<TrainingConfig> {
    Ok(TrainingConfig {
        learning_rate: spec.learning_rate,
        batch_size: spec.batch_size,
        gradient_accumulation_steps: spec.gradient_accumulation_steps,
        num_epochs: spec.epochs,
        max_steps: spec.max_steps,
        warmup_ratio: Some(spec.warmup_ratio),
        weight_decay: spec.weight_decay,
        optimizer: pmetal_core::jobs::parse_optimizer(&spec.optimizer)
            .map_err(|e| anyhow::anyhow!(e))?,
        max_grad_norm: spec.max_grad_norm,
        seed: spec.seed,
        logging_steps: 1,
        output_dir: spec.output_dir.clone(),
        max_seq_len: spec.max_length,
        ..Default::default()
    })
}

/// The steps a run took. A run a stop request ended has no history (the
/// trainer reports `Cancelled` instead of one) but has trained part of the
/// way: its adapter is saved like a finished run's.
fn stopped_run_history(
    result: preference::PreferenceResult<Vec<PreferenceStepMetrics>>,
) -> preference::PreferenceResult<Vec<PreferenceStepMetrics>> {
    super::stop::unless_stopped(result, |e| {
        matches!(e, pmetal_trainer::PreferenceError::Cancelled)
    })
    .map(Option::unwrap_or_default)
}

/// Run `pmetal preference` from a normalized spec.
pub(crate) async fn run_preference(spec: PreferenceSpec, ctx: RunContext) -> anyhow::Result<()> {
    let RunContext {
        emit_console_output,
        callbacks: extra_callbacks,
    } = ctx;
    let objective = objective(&spec)?;
    pmetal_runtime::compat::random::seed(spec.seed);

    let model_path = pmetal_hub::resolve_model_path(&spec.model, None, None).await?;
    let dataset_path = pmetal_trainer::resolve_dataset_path(&spec.dataset).await?;
    let output_dir = PathBuf::from(&spec.output_dir);
    std::fs::create_dir_all(&output_dir)?;
    super::record_training(
        &output_dir,
        &spec.model,
        &model_path,
        &spec.dataset,
        &spec.loss,
    );

    if emit_console_output {
        println!("========================================");
        println!("  PMetal Preference Optimization");
        println!("========================================");
        println!("Model:     {}", spec.model);
        println!("Dataset:   {}", dataset_path.display());
        println!("Output:    {}", spec.output_dir);
        println!("Loss:      {} (beta {})", spec.loss, spec.effective_beta());
        println!(
            "Batch:     {} x {} accumulation, lr {:.1e}",
            spec.batch_size, spec.gradient_accumulation_steps, spec.learning_rate
        );
        if spec.full_finetune {
            println!("Training:   every weight (full fine-tune)");
        } else {
            println!("LoRA:      r={} alpha={}", spec.lora_r, spec.lora_alpha);
        }
        println!("========================================\n");
    }

    let tokenizer = Tokenizer::from_model_dir(&model_path)?;
    let template = pmetal_data::chat_templates::detect_chat_template(&model_path, &spec.model);

    let lora_config = LoraConfig {
        r: spec.lora_r,
        alpha: spec.lora_alpha,
        dropout: 0.0,
        ..Default::default()
    };
    let training = training_config(&spec)?;

    // A full fine-tune trains every weight: the reference is the model as it
    // starts, which the trainer scores once before the first update, as it
    // does for LoRA's zero adapters. The run is refused before the model loads
    // if it will not fit, and the result is saved as a whole model.
    let full = spec
        .full_finetune
        .then_some(pmetal_trainer::full_finetune::FullFineTuneConfig {
            freeze_embeddings: spec.freeze_embeddings,
            skip_memory_check: spec.no_memory_check,
        });
    let mut model = match &full {
        Some(full) => {
            use pmetal_trainer::full_finetune::{RunShape, check_memory};
            check_memory(
                full,
                &model_path,
                &RunShape {
                    optimizer: training.optimizer,
                    // A pair is two sequences.
                    batch_size: spec.batch_size * 2,
                    seq_len: spec.max_length,
                    gradient_checkpointing: false,
                    cut_cross_entropy: false,
                },
            )?;
            tracing::info!("Loading {} for a full fine-tune", spec.model);
            DynamicLoraModel::from_pretrained_full(
                &model_path,
                pmetal_lora::FullFineTune {
                    freeze_embeddings: full.freeze_embeddings,
                },
            )?
        }
        None => {
            tracing::info!("Loading {} with LoRA r={}", spec.model, spec.lora_r);
            DynamicLoraModel::from_pretrained(&model_path, lora_config.clone())?
        }
    };
    let save_plan = match full {
        Some(_) => {
            use pmetal_runtime::compat::ModuleParametersExt;
            Some(pmetal_trainer::full_finetune::plan_for_model(
                &model_path,
                &model.flatten_params(),
            )?)
        }
        None => None,
    };

    let mut callbacks = extra_callbacks;
    if let Some(metrics_path) = &spec.log_metrics {
        let path = pmetal_core::paths::resolve_in(&output_dir, metrics_path);
        let model_name = spec.model.rsplit('/').next().unwrap_or(&spec.model);
        callbacks.push(Box::new(
            pmetal_trainer::MetricsJsonCallback::new(&path)?
                .with_run_name(format!("{}-{model_name}", spec.loss)),
        ));
    }

    let mut optimizer = TrainOptimizer::from_config(&training);
    let set_lr = |opt: &mut TrainOptimizer, lr: f32| opt.set_lr(lr);

    let history: Vec<PreferenceStepMetrics> = match objective {
        Objective::Paired(loss) => {
            let pairs = preference::load_preference_pairs(
                &dataset_path,
                &tokenizer,
                Some(&template),
                spec.max_prompt_length,
                spec.max_length,
            )?;
            tracing::info!("Loaded {} preference pairs", pairs.len());
            let mut trainer = PreferenceTrainer::new(loss, training)?;
            for cb in callbacks {
                trainer.add_callback(cb);
            }
            stopped_run_history(trainer.train(&mut model, &pairs, &mut optimizer, set_lr))?
        }
        Objective::Kto(kto) => {
            let samples = preference::load_kto_samples(
                &dataset_path,
                &tokenizer,
                Some(&template),
                spec.max_prompt_length,
                spec.max_length,
            )?;
            let desirable = samples.iter().filter(|s| s.desirable).count();
            tracing::info!(
                "Loaded {} KTO examples ({desirable} desirable, {} undesirable)",
                samples.len(),
                samples.len() - desirable
            );
            let mut trainer = KtoTrainer::new(kto, training)?;
            for cb in callbacks {
                trainer.add_callback(cb);
            }
            stopped_run_history(trainer.train(&mut model, &samples, &mut optimizer, set_lr))?
        }
    };

    let weights = match &save_plan {
        Some(plan) => {
            use pmetal_runtime::compat::ModuleParametersExt;
            pmetal_trainer::full_finetune::save_full_model(
                &model.flatten_params(),
                plan,
                &model_path,
                &output_dir,
            )?
        }
        None => pmetal_trainer::orchestrator::save_adapter(
            &model,
            &output_dir,
            &lora_config,
            Some(&spec.model),
        )?,
    };

    if emit_console_output {
        if let (Some(first), Some(last)) = (history.first(), history.last()) {
            println!(
                "\nLoss {:.4} -> {:.4} over {} steps; reward margin {:.4} -> {:.4}",
                first.loss,
                last.loss,
                history.len(),
                first.margin(),
                last.margin()
            );
        }
        if spec.full_finetune {
            println!("Fine-tuned model saved to {}", output_dir.display());
        } else {
            println!("LoRA adapter saved to {}", weights.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_optimizer_is_the_one_the_trainer_gets() {
        for kind in pmetal_core::OptimizerType::ALL {
            let spec = PreferenceSpec {
                optimizer: kind.to_string(),
                ..Default::default()
            };
            let training = training_config(&spec).unwrap();
            assert_eq!(training.optimizer, kind);
            assert_eq!(TrainOptimizer::from_config(&training).kind(), kind);
        }
    }
}
