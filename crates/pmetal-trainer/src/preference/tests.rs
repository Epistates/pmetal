use super::*;
use pmetal_bridge::compat::optimizers::{AdamW, AdamWBuilder};
use pmetal_core::LoraConfig;
use pmetal_lora::AdaptedModel;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::dispatcher::DynamicModel;
use serial_test::serial;

/// A tiny Llama with adapters attached: what `pmetal preference` trains.
fn small_model() -> AdaptedModel {
    let config = LlamaConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: None,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        ..Default::default()
    };
    pmetal_bridge::compat::random::seed(7);
    let base =
        DynamicModel::from_config(&serde_json::to_string(&config).unwrap()).expect("llama builds");
    AdaptedModel::attach(
        base,
        LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            ..Default::default()
        },
    )
    .expect("attach adapters")
}

/// Pairs whose chosen completion counts up from the prompt's last token and
/// whose rejected completion repeats a fixed token: a preference a small
/// model can learn in a few steps.
fn pairs(n: usize) -> Vec<PreferencePair> {
    (0..n)
        .map(|i| {
            let start = (i % 20) as u32 + 2;
            let prompt: Vec<u32> = (start..start + 4).collect();
            let chosen: Vec<u32> = (start + 4..start + 9).collect();
            let rejected = vec![60u32; 5];
            PreferencePair::new(&prompt, &chosen, &rejected)
        })
        .collect()
}

fn config(steps: usize, lr: f64) -> TrainingConfig {
    TrainingConfig {
        learning_rate: lr,
        batch_size: 4,
        gradient_accumulation_steps: 1,
        num_epochs: 100,
        max_steps: Some(steps),
        warmup_steps: 0,
        lr_scheduler: pmetal_core::LrSchedulerType::Constant,
        max_grad_norm: 1.0,
        logging_steps: 1000,
        seed: 3,
        ..Default::default()
    }
}

fn adamw(lr: f64) -> AdamW {
    AdamWBuilder::new(lr as f32).build().unwrap()
}

fn set_lr(opt: &mut AdamW, lr: f32) {
    opt.lr = pmetal_bridge::array!(lr);
}

#[test]
fn sequences_mask_the_prompt_and_pad_to_the_longest() {
    let s = Sequence::new(&[1, 2, 3], &[4, 5]);
    assert_eq!(s.ids, vec![1, 2, 3, 4, 5]);
    assert_eq!(s.labels, vec![-100, -100, -100, 4, 5]);
    assert_eq!(s.completion_len(), 2);

    let t = Sequence::new(&[9], &[8]);
    let (ids, labels) = pad(&[&s, &t]);
    ids.eval();
    labels.eval();
    assert_eq!(ids.shape(), &[2, 5]);
    assert_eq!(&ids.as_slice::<i32>()[5..], &[9, 8, 0, 0, 0]);
    let labels = labels.as_dtype(pmetal_bridge::compat::Dtype::Int32.as_i32());
    labels.eval();
    assert_eq!(&labels.as_slice::<i32>()[5..], &[-100, 8, -100, -100, -100]);
}

/// Before the first update the policy is the reference, so every DPO margin
/// is zero and the first step's loss is exactly ln 2. A reference taken from
/// anything but the starting model, or reduced differently from the policy,
/// would move it.
#[test]
#[serial]
fn dpo_starts_at_ln2_and_learns_the_preference() {
    let mut model = small_model();
    let data = pairs(16);
    let lr = 5e-3;
    let mut trainer = PreferenceTrainer::new(
        PreferenceLoss::Dpo {
            beta: 0.1,
            label_smoothing: 0.0,
        },
        config(30, lr),
    )
    .unwrap();
    let history = trainer
        .train(&mut model, &data, &mut adamw(lr), set_lr)
        .unwrap();
    pmetal_bridge::check_last_error().unwrap();

    assert_eq!(history.len(), 30);
    let first = &history[0];
    assert!(
        (first.loss - std::f32::consts::LN_2).abs() < 1e-4,
        "first DPO loss {} should be ln 2",
        first.loss
    );
    assert!(first.chosen_reward.abs() < 1e-4 && first.rejected_reward.abs() < 1e-4);
    let last = history.last().unwrap();
    assert!(
        last.loss < 0.5 * first.loss,
        "loss {} -> {}",
        first.loss,
        last.loss
    );
    assert!(last.margin() > 0.0);
    assert_eq!(last.accuracy, Some(1.0));
}

#[test]
#[serial]
fn reference_free_objectives_train() {
    for loss in [
        PreferenceLoss::Simpo {
            beta: 2.0,
            gamma_beta_ratio: 0.5,
        },
        PreferenceLoss::Orpo { beta: 0.1 },
        PreferenceLoss::Ipo { beta: 0.1 },
        PreferenceLoss::Hinge { beta: 0.1 },
    ] {
        let mut model = small_model();
        let lr = 5e-3;
        let mut trainer = PreferenceTrainer::new(loss, config(20, lr)).unwrap();
        let history = trainer
            .train(&mut model, &pairs(16), &mut adamw(lr), set_lr)
            .unwrap();
        pmetal_bridge::check_last_error().unwrap();
        let (first, last) = (&history[0], history.last().unwrap());
        assert!(
            last.loss < first.loss,
            "{}: loss {} -> {}",
            loss.name(),
            first.loss,
            last.loss
        );
        assert!(last.margin() > first.margin(), "{}", loss.name());
    }
}

#[test]
#[serial]
fn gradient_accumulation_takes_one_step_per_group() {
    let mut model = small_model();
    let mut cfg = config(3, 1e-3);
    cfg.batch_size = 2;
    cfg.gradient_accumulation_steps = 3;
    let mut trainer = PreferenceTrainer::new(PreferenceLoss::Orpo { beta: 0.1 }, cfg).unwrap();
    let history = trainer
        .train(&mut model, &pairs(18), &mut adamw(1e-3), set_lr)
        .unwrap();
    // 18 pairs / 2 per micro-batch / 3 per step = 3 steps an epoch.
    assert_eq!(history.len(), 3);
    assert!(history.iter().all(|m| m.loss.is_finite()));
}

#[test]
#[serial]
fn kto_learns_desirable_from_undesirable() {
    let data: Vec<KtoSample> = pairs(16)
        .into_iter()
        .enumerate()
        .flat_map(|(i, p)| {
            let next = pairs(16)[(i + 1) % 16].chosen.clone();
            [
                KtoSample {
                    sequence: p.chosen.clone(),
                    mismatched: next.clone(),
                    desirable: true,
                },
                KtoSample {
                    sequence: p.rejected,
                    mismatched: next,
                    desirable: false,
                },
            ]
        })
        .collect();
    let mut model = small_model();
    let lr = 5e-3;
    let mut trainer = KtoTrainer::new(KtoConfig::default(), config(30, lr)).unwrap();
    let history = trainer
        .train(&mut model, &data, &mut adamw(lr), set_lr)
        .unwrap();
    pmetal_bridge::check_last_error().unwrap();
    let (first, last) = (&history[0], history.last().unwrap());
    // At the start r = 0 and z = 0: every example's loss is 1 − σ(0) = 0.5.
    assert!(
        (first.loss - 0.5).abs() < 1e-4,
        "first KTO loss {}",
        first.loss
    );
    assert!(
        last.loss < first.loss,
        "loss {} -> {}",
        first.loss,
        last.loss
    );
    assert!(last.chosen_reward > last.rejected_reward);
    assert!(last.kl.is_some());
}
