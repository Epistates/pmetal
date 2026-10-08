//! A few GRPO steps on a real checkpoint, with several updates per batch.
//!
//! Ignored by default. Point `PMETAL_GRPO_MODEL` at a small causal LM
//! directory (config.json, tokenizer.json and safetensors; Qwen3-0.6B is
//! what this was written against) and run
//! `cargo test -p pmetal-trainer --test grpo_real_model -- --ignored --nocapture`.

use pmetal_core::{LoraConfig, LrSchedulerType, TrainingConfig};
use pmetal_data::Tokenizer;
use pmetal_lora::DynamicLoraModel;
use pmetal_trainer::{CompletionGroup, GrpoConfig, GrpoTrainer, TrainOptimizer};

/// Reward the share of a completion's characters that are digits: it varies
/// from completion to completion, so the group advantages are not all zero.
fn digit_share(text: &str) -> f64 {
    let chars = text.chars().count().max(1);
    text.chars().filter(char::is_ascii_digit).count() as f64 / chars as f64
}

fn run(loss_type: &str) {
    let Ok(dir) = std::env::var("PMETAL_GRPO_MODEL") else {
        eprintln!("PMETAL_GRPO_MODEL is not set; skipping");
        return;
    };
    pmetal_bridge::compat::random::seed(7);
    let tokenizer = Tokenizer::from_model_dir(&dir).unwrap();
    let lora = LoraConfig {
        r: 8,
        alpha: 16.0,
        ..Default::default()
    };
    let mut model = DynamicLoraModel::from_pretrained(&dir, lora).unwrap();

    let prompts = [
        "Q: What is 27 + 58?\nA:",
        "Q: What is 63 + 19?\nA:",
        "Q: What is 44 + 37?\nA:",
    ];
    let prompt_ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenizer.encode(p).unwrap())
        .collect();

    let config = GrpoConfig {
        num_generations: 8,
        max_completion_length: 32,
        max_prompt_length: 64,
        beta: 0.0,
        num_iterations: 2,
        seed: Some(7),
        ..GrpoConfig::default()
    }
    .with_loss_preset(loss_type)
    .unwrap();
    let training = TrainingConfig {
        learning_rate: 5e-5,
        num_epochs: 2,
        warmup_steps: 0,
        lr_scheduler: LrSchedulerType::Cosine,
        ..Default::default()
    };
    let mut trainer = GrpoTrainer::new(config, training).unwrap();
    let total = trainer.plan_schedule(prompts.len());
    let mut optimizer = TrainOptimizer::from_config(&trainer.training_config);
    let mut set_lr = |opt: &mut TrainOptimizer, lr: f32| opt.set_lr(lr);

    println!("{loss_type}: {total} optimizer steps planned");
    let mut clipped_any = false;
    for epoch in 0..2 {
        for ids in &prompt_ids {
            let out = trainer
                .generate_completions(&mut model, ids, &tokenizer)
                .unwrap();
            let mut group = CompletionGroup::new(ids.clone(), 8);
            for (seq, by_length) in out.token_ids.iter().zip(&out.stopped_by_length) {
                let completion = seq[ids.len()..].to_vec();
                let text = tokenizer.decode(&completion).unwrap();
                if std::env::var_os("PMETAL_GRPO_SHOW").is_some() {
                    println!(
                        "  {:?} -> {:?}",
                        &completion[..completion.len().min(8)],
                        text
                    );
                }
                group.add_completion(completion, digit_share(&text), *by_length);
            }
            let lr = optimizer.lr();
            let stats = trainer
                .train_step(
                    &mut model,
                    None::<&mut pmetal_models::DynamicModel>,
                    &[group],
                    &mut optimizer,
                    &mut set_lr,
                )
                .unwrap();
            pmetal_bridge::check_last_error().unwrap();
            println!(
                "epoch {epoch} step {:2}: loss {:+.5} policy {:+.5} reward {:.4} clip {:.3} \
                 ({} updates, lr before {:.2e} after {:.2e})",
                stats.step,
                stats.loss,
                stats.policy_loss,
                stats.reward,
                stats.clip_fraction,
                stats.iterations,
                lr,
                optimizer.lr(),
            );
            assert!(stats.loss.is_finite() && stats.policy_loss.is_finite());
            assert_eq!(stats.iterations, 2);
            clipped_any |= stats.clip_fraction > 0.0;
        }
    }
    assert_eq!(trainer.step, total);
    println!("{loss_type}: clipped on some update: {clipped_any}");
}

#[test]
#[ignore = "needs PMETAL_GRPO_MODEL"]
fn gspo_with_two_updates_per_batch() {
    run("gspo");
}

#[test]
#[ignore = "needs PMETAL_GRPO_MODEL"]
fn dapo_with_two_updates_per_batch() {
    run("dapo");
}
