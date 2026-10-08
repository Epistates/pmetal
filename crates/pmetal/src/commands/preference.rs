//! `pmetal preference`: offline preference optimization with LoRA.

use std::path::PathBuf;

use pmetal_bridge::compat::optimizers::{AdamW, AdamWBuilder};
use pmetal_core::jobs::PreferenceSpec;
use pmetal_core::{LoraConfig, TrainingCallback, TrainingConfig};
use pmetal_data::Tokenizer;
use pmetal_lora::{DynamicLoraModel, TrainableModel};
use pmetal_trainer::preference::{self, KtoConfig, PreferenceLoss, PreferenceStepMetrics};
use pmetal_trainer::{KtoTrainer, PreferenceTrainer};

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

/// Run `pmetal preference` from a normalized spec.
pub(crate) async fn run_preference(
    spec: PreferenceSpec,
    emit_console_output: bool,
    extra_callbacks: Vec<Box<dyn TrainingCallback>>,
) -> anyhow::Result<()> {
    let objective = objective(&spec)?;
    pmetal_bridge::compat::random::seed(spec.seed);

    let model_path = pmetal_hub::resolve_model_path(&spec.model, None, None).await?;
    let dataset_path = pmetal_trainer::resolve_dataset_path(&spec.dataset).await?;
    let output_dir = PathBuf::from(&spec.output_dir);
    std::fs::create_dir_all(&output_dir)?;

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
        println!("LoRA:      r={} alpha={}", spec.lora_r, spec.lora_alpha);
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
    tracing::info!("Loading {} with LoRA r={}", spec.model, spec.lora_r);
    let mut model = DynamicLoraModel::from_pretrained(&model_path, lora_config.clone())?;

    let training = TrainingConfig {
        learning_rate: spec.learning_rate,
        batch_size: spec.batch_size,
        gradient_accumulation_steps: spec.gradient_accumulation_steps,
        num_epochs: spec.epochs,
        max_steps: spec.max_steps,
        warmup_ratio: Some(spec.warmup_ratio),
        weight_decay: spec.weight_decay,
        max_grad_norm: spec.max_grad_norm,
        seed: spec.seed,
        logging_steps: 1,
        output_dir: spec.output_dir.clone(),
        max_seq_len: spec.max_length,
        ..Default::default()
    };

    let mut callbacks = extra_callbacks;
    if let Some(metrics_path) = &spec.log_metrics {
        let path = if metrics_path.contains('/') || metrics_path.contains('\\') {
            PathBuf::from(metrics_path)
        } else {
            output_dir.join(metrics_path)
        };
        let model_name = spec.model.rsplit('/').next().unwrap_or(&spec.model);
        callbacks.push(Box::new(
            pmetal_trainer::MetricsJsonCallback::new(&path)?
                .with_run_name(format!("{}-{model_name}", spec.loss)),
        ));
    }

    let mut optimizer: AdamW = AdamWBuilder::new(spec.learning_rate as f32)
        .weight_decay(spec.weight_decay as f32)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build optimizer: {e}"))?;
    let set_lr = |opt: &mut AdamW, lr: f32| opt.lr = pmetal_bridge::array!(lr);

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
            trainer.train(&mut model, &pairs, &mut optimizer, set_lr)?
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
            trainer.train(&mut model, &samples, &mut optimizer, set_lr)?
        }
    };

    let weights = output_dir.join("lora_weights.safetensors");
    model.save_lora_weights(&weights)?;
    pmetal_trainer::orchestrator::save_adapter_config_with_base(
        &weights,
        lora_config.r,
        lora_config.alpha,
        &lora_config.target_modules,
        lora_config.use_rslora,
        Some(&spec.model),
    )?;

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
        println!("LoRA adapter saved to {}", weights.display());
    }
    Ok(())
}
