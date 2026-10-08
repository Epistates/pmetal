# pmetal-trainer

Training loops and optimization strategies for LLM fine-tuning.

## Overview

This crate provides the training infrastructure for PMetal, including various training methods, learning rate scheduling, checkpointing, and callback systems.

## Training Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Supervised Fine-Tuning | General instruction tuning |
| **LoRA** | Low-Rank Adaptation | Parameter-efficient fine-tuning |
| **DPO / Robust DPO / IPO / hinge** | Preference optimization against a reference | Pairwise preference data |
| **GRPO** | Group Relative Policy Optimization | Efficient PPO alternative |
| **DAPO** | Decoupled Clip and Dynamic Sampling PO | GRPO mode (`GrpoConfig::for_dapo`) |
| **ORPO** | Odds Ratio Preference Optimization | Reference-free, SFT and alignment in one stage |
| **SimPO** | Simple Preference Optimization | Reference-free, length-normalized |
| **KTO** | Kahneman-Tversky Optimization | Unpaired good/bad feedback |
| **Distillation** | Knowledge distillation | Teacher→student transfer |
| **ANE** | Apple Neural Engine training | Power-efficient on-device training |
| **RLKD** | RL with Knowledge Distillation | GRPO + teacher distillation |
| **Embedding** | Sentence-transformer training | InfoNCE, Triplet, CoSENT contrastive |

## Usage

### Basic Training Loop

`TrainingLoop::new` takes a single `TrainingLoopConfig`. The model and dataset go to `run_packed`,
which returns the trained model.

```rust,no_run
use pmetal_core::TrainingConfig;
use pmetal_data::{DataLoaderConfig, TrainingDataset};
use pmetal_lora::DynamicLoraModel;
use pmetal_trainer::{ProgressCallback, TrainingLoop, TrainingLoopConfig};

fn train(
    model: DynamicLoraModel,
    train_dataset: TrainingDataset,
) -> anyhow::Result<DynamicLoraModel> {
    let loop_config = TrainingLoopConfig {
        training: TrainingConfig {
            batch_size: 4,
            gradient_accumulation_steps: 4,
            learning_rate: 2e-4,
            num_epochs: 1,
            max_grad_norm: 1.0,
            ..Default::default()
        },
        dataloader: DataLoaderConfig { batch_size: 4, ..Default::default() },
        use_sequence_packing: true,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(loop_config);
    training_loop.add_callback(Box::new(ProgressCallback::new(1_000)));

    Ok(training_loop.run_packed(model, train_dataset, None, None)?)
}
```

Drive the iteration yourself with `train_step(&mut model, &batch, &mut optimizer)` if you need
control over batching.

### With Checkpointing

```rust,no_run
use pmetal_data::TrainingDataset;
use pmetal_lora::{DynamicLoraModel, TrainableModel};
use pmetal_trainer::{CheckpointManager, TrainingLoop};

fn train_with_checkpoints(
    mut model: DynamicLoraModel,
    train_dataset: TrainingDataset,
    training_loop: &mut TrainingLoop,
) -> anyhow::Result<DynamicLoraModel> {
    let checkpoints = CheckpointManager::new("output/checkpoints")?
        .with_max_checkpoints(3)
        .with_save_best(true);

    // Resume from the newest checkpoint if there is one.
    if let Some((lora_params, metadata)) = checkpoints.load_latest()? {
        model.set_lora_parameters(&lora_params);
        training_loop.set_step(metadata.step);
        training_loop.set_epoch(metadata.epoch);
        println!("resumed at step {}", metadata.step);
    }

    // Hand the manager to run_packed and it saves on the configured cadence.
    Ok(training_loop.run_packed(model, train_dataset, None, Some(&checkpoints))?)
}
```

## Optimizers

| Optimizer | Description |
|-----------|-------------|
| **AdamW Groups** | AdamW with per-parameter-group learning rates |
| **Metal Fused** | GPU-accelerated AdamW parameter updates |

## Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| Constant | Fixed learning rate |
| Linear | Linear warmup and decay |
| Cosine | Cosine annealing |
| Cosine with Restarts | Cosine with periodic warm restarts |
| Polynomial | Polynomial decay |
| WSD | Warmup-Stable-Decay schedule |

## Modules

| Module | Description |
|--------|-------------|
| `training_loop` | Main training orchestration |
| `sft` | Supervised fine-tuning trainer |
| `preference` | DPO, Robust DPO, IPO, hinge, SimPO, ORPO (`PreferenceTrainer`) and KTO (`KtoTrainer`) |
| `grpo` | Group Relative Policy Optimization |
| `ane_training` | ANE training loop (feature-gated: `ane`) |
| `distillation` | Knowledge distillation orchestration |
| `rlkd` | Reinforcement Learning with Knowledge Distillation |
| `embedding_trainer` | Sentence-transformer fine-tuning |
| `contrastive_loss` | InfoNCE, Triplet, CoSENT loss functions |
| `orchestrator` | Unified training pipeline (shared across CLI/GUI/TUI/easy) |
| `adamw_groups` | AdamW with parameter groups |
| `mlx_metal_optimizer` | Metal-accelerated AdamW updates |
| `adaptive_lr` | EMA-based adaptive learning rate control |
| `checkpoint` | Checkpoint save/load |
| `scheduler` | Learning rate schedulers |
| `callbacks` | Training callbacks (`MetricsJsonCallback`, `StepMetrics`) |
| `param_groups` | Per-layer learning rates |
| `distributed_bridge` | Distributed training sync (feature-gated: `distributed`) |

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Micro-batch size | 4 |
| `gradient_accumulation_steps` | Accumulation steps | 1 |
| `learning_rate` | Initial learning rate | 2e-4 |
| `max_grad_norm` | Gradient clipping | 1.0 |
| `warmup_steps` | LR warmup steps | 0 |
| `weight_decay` | L2 regularization | 0.0 |

## License

MIT OR Apache-2.0
