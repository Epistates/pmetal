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
| **Dr. GRPO** | GRPO without length or difficulty bias | GRPO mode (`GrpoConfig::for_dr_grpo`) |
| **GSPO** | Group Sequence Policy Optimization | GRPO mode (`GrpoConfig::for_gspo`), sequence-level ratio |
| **ORPO** | Odds Ratio Preference Optimization | Reference-free, SFT and alignment in one stage |
| **SimPO** | Simple Preference Optimization | Reference-free, length-normalized |
| **KTO** | Kahneman-Tversky Optimization | Unpaired good/bad feedback |
| **Distillation** | Knowledge distillation | Teacher→student transfer |
| **ANE** | Apple Neural Engine training | Power-efficient on-device training |
| **RLKD** | RL with Knowledge Distillation | GRPO + teacher distillation |
| **Embedding** | Sentence-transformer training | InfoNCE, Triplet, CoSENT contrastive |

## Usage

### Basic Training Loop

`TrainingLoop::new` takes a single `TrainingLoopConfig` and checks it: a configuration that cannot
work (a zero batch size, a label smoothing outside `[0, 1)`) is an error here, not a failure
partway through a run. The model and dataset go to `run`, which trains the model in place.

```rust,no_run
use pmetal_core::TrainingConfig;
use pmetal_data::{DataLoaderConfig, TrainingDataset};
use pmetal_lora::DynamicLoraModel;
use pmetal_trainer::{ProgressCallback, TrainingLoop, TrainingLoopConfig};

fn train(
    mut model: DynamicLoraModel,
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

    let mut training_loop = TrainingLoop::new(loop_config)?;
    training_loop.add_callback(Box::new(ProgressCallback::new(1_000)));

    training_loop.run(&mut model, train_dataset, None, None)?;
    Ok(model)
}
```

`run` packs short sequences when `use_sequence_packing` is set and the model allows it, trains on
this rank's share of the data under data parallelism, takes an optimizer step every
`gradient_accumulation_steps` micro-batches (the mean gradient over all their tokens), and
evaluates once per micro-batch. Drive the iteration yourself with
`train_step(&mut model, &batch, &mut optimizer)` if you need control over batching.

### With Checkpointing

```rust,no_run
use pmetal_data::TrainingDataset;
use pmetal_lora::DynamicLoraModel;
use pmetal_trainer::{CheckpointManager, TrainingLoop};

fn train_with_checkpoints(
    mut model: DynamicLoraModel,
    train_dataset: TrainingDataset,
    training_loop: &mut TrainingLoop,
) -> anyhow::Result<DynamicLoraModel> {
    let checkpoints = CheckpointManager::new("output/checkpoints")?
        .with_max_checkpoints(3)
        .with_save_best(true);

    // Resume from the newest checkpoint if there is one: its weights, the optimizer's
    // state and the data loader's position, so the run takes the steps the interrupted
    // one would have.
    if training_loop.resume_from(&mut model, &checkpoints)? {
        println!("resumed at step {}", training_loop.current_step());
    }

    // Hand the manager to `run` and it saves on the configured cadence.
    training_loop.run(&mut model, train_dataset, None, Some(&checkpoints))?;
    Ok(model)
}
```

## Optimizers

| Optimizer | Description |
|-----------|-------------|
| **AdamW Groups** | AdamW with per-parameter-group learning rates |
| **SGD, Lion, Adafactor** | Momentum SGD, sign-momentum Lion, factored-moment Adafactor |
| **Muon** | Newton-Schulz-orthogonalized momentum on hidden matrices, AdamW on the rest |
| **Schedule-free AdamW** | Iterate averaging instead of a decaying schedule |
| **8-bit AdamW** | Blockwise-quantized moments, a quarter of the optimizer memory |

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
| `optimizer` | AdamW, SGD with momentum, Lion, Adafactor, Muon, schedule-free AdamW and 8-bit AdamW, in parameter groups |
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
