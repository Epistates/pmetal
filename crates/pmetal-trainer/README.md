# pmetal-trainer

Training loops and optimization strategies for LLM fine-tuning.

## Overview

This crate provides the training infrastructure for PMetal, including various training methods, learning rate scheduling, checkpointing, and callback systems.

## Training Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Supervised Fine-Tuning | General instruction tuning |
| **LoRA** | Low-Rank Adaptation | Parameter-efficient fine-tuning |
| **DPO** | Direct Preference Optimization | Preference-based alignment |
| **GRPO** | Group Relative Policy Optimization | Efficient PPO alternative |
| **Diffusion** | LLaDA-style diffusion training | Experimental |

## Usage

### Basic Training Loop

```rust
use pmetal_trainer::{TrainingLoop, TrainingConfig};

let config = TrainingConfig {
    batch_size: 4,
    gradient_accumulation_steps: 4,
    learning_rate: 2e-4,
    epochs: 1,
    max_grad_norm: 1.0,
    ..Default::default()
};

let mut trainer = TrainingLoop::new(model, optimizer, config)?;

// Train with optional callbacks
trainer.train(&dataloader, callbacks)?;
```

### With Checkpointing

```rust
use pmetal_trainer::CheckpointManager;

let checkpoint_mgr = CheckpointManager::new("output/checkpoints");

// Resume from checkpoint if available
if let Some(ckpt) = checkpoint_mgr.latest()? {
    trainer.load_checkpoint(&ckpt)?;
}

// Save checkpoints during training
trainer.train_with_checkpoints(&dataloader, &checkpoint_mgr, save_every: 500)?;
```

## Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| Constant | Fixed learning rate |
| Linear | Linear warmup and decay |
| Cosine | Cosine annealing |
| Polynomial | Polynomial decay |

## Modules

| Module | Description |
|--------|-------------|
| `training_loop` | Main training orchestration |
| `sft` | Supervised fine-tuning trainer |
| `lora_trainer` | LoRA-specific training |
| `dpo` | Direct Preference Optimization |
| `grpo` | Group Relative Policy Optimization |
| `diffusion` | Diffusion-based training |
| `checkpoint` | Checkpoint save/load |
| `scheduler` | Learning rate schedulers |
| `callbacks` | Training callbacks |
| `param_groups` | Per-layer learning rates |

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
